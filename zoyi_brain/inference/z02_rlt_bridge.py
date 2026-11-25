#!/usr/bin/env python3
"""Serve Z02 joint control and observations to RLinf without loading a policy."""

from array import array
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from enum import IntEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import secrets
import signal
from threading import Condition, Thread
import time

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from stereo_camera.frame_hub import FrameHub
from zoyi_robot_interfaces.msg import JointPositionCommands, JointState

from utils import pack_z02_state, process_bgr_rgb


ACTION_DIM = 18
MAX_REQUEST_BYTES = 8192
CAMERAS = ("head", "left_wrist", "right_wrist")
ANGLE_INDICES = [*range(7), *range(8, 15), 16, 17]
LEFT_FINGERS_OPEN = (4000, 1000, 1500, 4096, 4069, 4069, 4096, 4069, 0, 4069)
LEFT_FINGERS_CLOSED = (3000, 1000, 0, 4096, 4069, 0, 4096, 0, 0, 0)
RIGHT_FINGERS_OPEN = (4000, 3000, 1500, 0, 4069, 4069, 0, 4069, 4096, 4069)
RIGHT_FINGERS_CLOSED = (4000, 3000, 0, 0, 0, 0, 0, 0, 4096, 0)


class Outcome(IntEnum):
    ACTIVE = 0
    SUCCESS = 1
    FAILURE = 2
    ABORT = 3


class BridgeError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Observation:
    captured_at: float
    state: np.ndarray
    images: dict[str, np.ndarray]


@dataclass
class PendingAction:
    requested: np.ndarray
    executed: np.ndarray | None = None
    sent_at: float | None = None


class Z02RLTBridge(Node):
    def __init__(self):
        super().__init__("z02_rlt_bridge")
        self.host = self.declare_parameter("host", "127.0.0.1").value
        self.port = self.declare_parameter("port", 8765).value
        height = self.declare_parameter("image_height", 180).value
        width = self.declare_parameter("image_width", 320).value
        self.image_size = (width, height)
        self.ready_timeout = self._positive_parameter("ready_timeout_sec", 300.0)
        self.step_timeout = self._positive_parameter("step_timeout_sec", 2.0)
        self.observation_timeout = self._positive_parameter("observation_timeout_sec", 0.5)
        self.command_idle_timeout = self._positive_parameter("command_idle_timeout_sec", 60.0)
        self.step_period = 1.0 / self._positive_parameter("control_hz", 30.0)
        self.max_joint_step = self._positive_parameter("max_joint_step_rad", 0.04)
        if not 0 < self.port <= 65535 or height <= 0 or width <= 0:
            raise ValueError("Expected a valid HTTP port and positive camera dimensions")
        if self.step_timeout <= self.step_period:
            raise ValueError("step_timeout_sec must be greater than one control period")

        self._changed = Condition()
        self._closed = False
        self._episode_id = secrets.randbits(62)
        self._outcome = Outcome.ABORT
        self._actor_enabled = False
        self._pending_reset_id: str | None = None
        self._pending_reset_canceled = False
        self._unstarted_reset_id: str | None = None
        self._canceled_resets: deque[str] = deque(maxlen=64)
        self._pending: PendingAction | None = None
        self._has_control = False
        self._hold_requested = False
        self._last_command_at: float | None = None
        self._observation: Observation | None = None
        self._positions: np.ndarray | None = None
        self._joint_at = 0.0

        self._frame_hub = FrameHub(timeout_sec=self.observation_timeout)
        self._joint_pub = self.create_publisher(JointPositionCommands, "/joint/arm_head_waist_commands", 10)
        self._mode_pub = self.create_publisher(Bool, "/inference_mode", 10)
        self._joint_sub = self.create_subscription(
            JointState, "/joint/arm_head_waist_states", self._joint_callback, 10
        )
        self._event_sub = self.create_subscription(String, "/z02_rlt/event", self._operator_callback, 10)
        # Check pending requests more often than we can publish so network
        # round-trips do not add an entire control period to every command.
        self._timer = self.create_timer(
            self.step_period / 4,
            self._control_tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def _positive_parameter(self, name: str, default: float) -> float:
        value = self.declare_parameter(name, default).value
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _joint_callback(self, msg: JointState) -> None:
        captured_at = time.monotonic()
        positions = np.asarray(msg.position[:40], dtype=np.float32)
        if positions.shape != (40,) or not np.isfinite(positions).all():
            with self._changed:
                self._positions = None
                self._observation = None
                self._end_locked(Outcome.ABORT)
            return

        with self._changed:
            self._positions = positions
            self._joint_at = captured_at

        images = {}
        try:
            for name in CAMERAS:
                frame = self._frame_hub.latest_bgr(name)
                if frame is None:
                    return
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f"Invalid {name} frame format")
                images[name] = process_bgr_rgb(frame, rotate=name == "right_wrist", img_size=self.image_size)
        except (cv2.error, OSError, RuntimeError, ValueError) as exc:
            with self._changed:
                self._observation = None
                self._end_locked(Outcome.ABORT)
            self.get_logger().warning(f"Unable to capture Z02 cameras: {exc}")
            return
        with self._changed:
            self._observation = Observation(captured_at, pack_z02_state(positions, 25, 4000), images)
            self._changed.notify_all()

    def _operator_callback(self, msg: String) -> None:
        event = msg.data.strip().lower()
        with self._changed:
            if event == "ready":
                if (
                    self._pending_reset_id is None
                    or self._pending_reset_canceled
                    or self._outcome == Outcome.ACTIVE
                    or self._hold_requested
                    or self._fresh_locked() is None
                ):
                    self.get_logger().warning("Ready requires a pending reset and fresh joint and camera feedback")
                    return
                self._episode_id += 1
                self._outcome = Outcome.ACTIVE
                self._actor_enabled = False
                self._unstarted_reset_id = self._pending_reset_id
                self._last_command_at = None
                self._changed.notify_all()
            elif event == "actor" and self._outcome == Outcome.ACTIVE:
                self._actor_enabled = True
            elif event in ("success", "failure", "abort"):
                if event == "abort" and self._pending_reset_id is not None:
                    self._cancel_reset_locked(self._pending_reset_id)
                self._end_locked(Outcome[event.upper()])
            else:
                self.get_logger().warning(f"Ignoring Z02 event {event!r}")

    def _fresh_locked(self) -> Observation | None:
        obs = self._observation
        if obs is not None and time.monotonic() - obs.captured_at <= self.observation_timeout:
            return obs
        return None

    def _end_locked(self, outcome: Outcome) -> None:
        if self._outcome == Outcome.ACTIVE:
            self._outcome = outcome
            self._hold_requested = self._has_control
        self._changed.notify_all()

    def _cancel_reset_locked(self, reset_id: str) -> None:
        if reset_id not in self._canceled_resets:
            self._canceled_resets.append(reset_id)
        if reset_id == self._pending_reset_id:
            self._pending_reset_canceled = True
        # A late cancel can disarm an acknowledged reset until its first command.
        if reset_id == self._unstarted_reset_id:
            self._end_locked(Outcome.ABORT)
        else:
            self._changed.notify_all()

    def _require_open_locked(self) -> None:
        if self._closed:
            raise BridgeError(503, "Z02 bridge is shutting down")

    def _response_locked(self, executed: np.ndarray | None = None) -> dict[str, np.ndarray]:
        obs = self._fresh_locked()
        if obs is None:
            self._end_locked(Outcome.ABORT)
            raise BridgeError(503, "No fresh joint and camera observation")
        return {
            "episode_id": np.asarray(self._episode_id, dtype=np.int64),
            "actor_enabled": np.asarray(self._actor_enabled, dtype=np.bool_),
            "outcome": np.asarray(self._outcome, dtype=np.int8),
            "executed_actions": (
                np.empty((0, ACTION_DIM), dtype=np.float32) if executed is None else executed[None]
            ),
            "state": obs.state,
            **obs.images,
        }

    def reset(self, reset_id: str) -> dict[str, np.ndarray]:
        with self._changed:
            self._require_open_locked()
            if reset_id in self._canceled_resets:
                raise BridgeError(409, "Z02 reset was canceled")
            if self._pending_reset_id is not None or self._pending is not None:
                raise BridgeError(409, "A Z02 reset or step is already in progress")
            self._end_locked(Outcome.ABORT)
            self._pending_reset_id = reset_id
            self._pending_reset_canceled = False
            previous_episode_id = self._episode_id
            self.get_logger().info("Reset requested. Prepare the scene and send ready on /z02_rlt/event")
            try:
                ready = self._changed.wait_for(
                    lambda: (
                        self._closed
                        or self._pending_reset_canceled
                        or self._episode_id != previous_episode_id
                    ),
                    timeout=self.ready_timeout,
                )
                self._require_open_locked()
                if self._pending_reset_canceled:
                    raise BridgeError(409, "Z02 reset was canceled")
                if not ready:
                    raise BridgeError(503, "Timed out waiting for manual ready")
                return self._response_locked()
            finally:
                self._pending_reset_id = None

    def cancel_reset(self, reset_id: str) -> None:
        with self._changed:
            self._cancel_reset_locked(reset_id)

    def step(self, episode_id: int, action: np.ndarray) -> dict[str, np.ndarray]:
        with self._changed:
            self._require_open_locked()
            if episode_id != self._episode_id or self._pending_reset_id is not None or self._pending is not None:
                raise BridgeError(409, "Z02 step does not match the current idle episode")
            if self._outcome != Outcome.ACTIVE:
                return self._response_locked()
            pending = PendingAction(action)
            self._pending = pending
            try:
                finished = self._changed.wait_for(
                    lambda: (
                        self._closed
                        or self._outcome == Outcome.ABORT
                        or (self._outcome != Outcome.ACTIVE and pending.sent_at is None)
                        or (
                            pending.sent_at is not None
                            and self._observation is not None
                            and self._observation.captured_at >= pending.sent_at + self.step_period
                        )
                    ),
                    timeout=self.step_timeout,
                )
                self._require_open_locked()
                if not finished:
                    self._end_locked(Outcome.ABORT)
                    raise BridgeError(503, "Timed out waiting for Z02 joint and camera feedback after action")
                return self._response_locked(pending.executed)
            finally:
                self._pending = None

    def hold(self, episode_id: int) -> None:
        with self._changed:
            if episode_id == self._episode_id:
                self._end_locked(Outcome.ABORT)

    def _publish(self, positions: np.ndarray) -> None:
        command = JointPositionCommands()
        command.joint_positions = array("f", positions)
        command.control_mode = [False] * 4 + [True] * 36
        command.header.stamp = self.get_clock().now().to_msg()
        self._joint_pub.publish(command)

    def _send_action_locked(self, pending: PendingAction, current_positions: np.ndarray) -> None:
        action = pending.requested.copy()
        measured_angles = current_positions[4:20]
        lower = measured_angles - self.max_joint_step
        upper = measured_angles + self.max_joint_step
        action[ANGLE_INDICES] = np.clip(action[ANGLE_INDICES], lower, upper)
        action[[7, 15]] = action[[7, 15]] > 0.3

        positions = current_positions.copy()
        positions[4:11] = action[:7]
        positions[11:18] = action[8:15]
        positions[18:20] = action[16:18]
        positions[20:30] = LEFT_FINGERS_CLOSED if action[7] else LEFT_FINGERS_OPEN
        positions[30:40] = RIGHT_FINGERS_CLOSED if action[15] else RIGHT_FINGERS_OPEN
        self._mode_pub.publish(Bool(data=True))
        self._has_control = True
        self._publish(positions)
        pending.sent_at = time.monotonic()
        self._last_command_at = pending.sent_at
        self._unstarted_reset_id = None
        pending.executed = action

    def _control_tick(self) -> None:
        with self._changed:
            obs = self._fresh_locked()
            now = time.monotonic()
            last_command = self._last_command_at
            if self._outcome == Outcome.ACTIVE:
                if obs is None:
                    self._end_locked(Outcome.ABORT)
                elif last_command is not None and now - last_command >= self.command_idle_timeout:
                    self._end_locked(Outcome.ABORT)
                    self.get_logger().warning(
                        f"No Z02 commands for {self.command_idle_timeout:g}s. "
                        "Episode aborted. A new reset and ready event are required."
                    )
            positions = self._positions
            if self._hold_requested:
                self._hold_requested = False
                self._has_control = False
                try:
                    try:
                        if positions is not None and now - self._joint_at <= self.observation_timeout:
                            self._publish(positions)
                    finally:
                        self._mode_pub.publish(Bool(data=False))
                except Exception as exc:
                    self.get_logger().error(f"Z02 hold publishing failed: {exc}")
                return
            pending = self._pending
            if (
                self._outcome == Outcome.ACTIVE
                and positions is not None
                and pending is not None
                and pending.sent_at is None
                and (last_command is None or now - last_command >= self.step_period)
            ):
                try:
                    self._send_action_locked(pending, positions)
                except Exception as exc:
                    self._end_locked(Outcome.ABORT)
                    self.get_logger().error(f"Z02 command publishing failed: {exc}")

    def close(self) -> None:
        with self._changed:
            self._closed = True
            self._end_locked(Outcome.ABORT)
            release_pending = self._hold_requested
        try:
            if rclpy.ok() and release_pending:
                self._control_tick()
                if not self._mode_pub.wait_for_all_acked(Duration(seconds=0.25)):
                    self.get_logger().warning("Timed out sending the Z02 control release")
        finally:
            self._frame_hub.close()


class BridgeServer(ThreadingHTTPServer):
    def __init__(self, bridge: Z02RLTBridge):
        self.bridge = bridge
        super().__init__((bridge.host, bridge.port), BridgeHandler)


class BridgeHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5.0)

    def log_message(self, _format, *args):
        pass

    def _request_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= MAX_REQUEST_BYTES:
            raise ValueError("Missing or invalid request length")
        data = json.loads(self.rfile.read(size))
        if type(data) is not dict:
            raise ValueError("Expected a JSON object")
        return data

    def do_POST(self):
        bridge = self.server.bridge
        try:
            if self.path not in ("/reset", "/cancel-reset", "/step", "/hold"):
                raise BridgeError(404, "Unknown Z02 bridge endpoint")
            data = self._request_json()
            if self.path in ("/reset", "/cancel-reset"):
                reset_id = data.get("reset_id")
                if (
                    not isinstance(reset_id, str)
                    or len(reset_id) != 32
                    or any(char not in "0123456789abcdef" for char in reset_id)
                ):
                    raise ValueError("Expected reset_id to be a lower-case UUID hex string")
                if self.path == "/cancel-reset":
                    bridge.cancel_reset(reset_id)
                    self.send_response(204)
                    self.end_headers()
                    return
                result = bridge.reset(reset_id)
            else:
                episode_id = data.get("episode_id")
                if type(episode_id) is not int:
                    raise ValueError("Expected integer episode_id")
                if self.path == "/hold":
                    bridge.hold(episode_id)
                    self.send_response(204)
                    self.end_headers()
                    return
                actions = np.asarray(data.get("actions"), dtype=np.float32)
                if actions.shape != (1, ACTION_DIM) or not np.isfinite(actions).all():
                    raise ValueError(f"Expected one finite {ACTION_DIM}-D action")
                result = bridge.step(episode_id, actions[0])
        except BridgeError as exc:
            self.send_error(exc.status, str(exc))
            return
        except (OSError, TypeError, OverflowError, ValueError) as exc:
            self.send_error(400, str(exc))
            return

        buffer = BytesIO()
        np.savez(buffer, **result)
        payload = buffer.getbuffer()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            bridge.hold(int(result["episode_id"]))


def main() -> None:
    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    try:
        with ExitStack() as stack:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handler = signal.signal(signum, request_stop)
                stack.callback(signal.signal, signum, previous_handler)
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            stack.callback(rclpy.try_shutdown)
            bridge = Z02RLTBridge()
            stack.callback(bridge.destroy_node)
            stack.callback(bridge.close)
            server = stack.enter_context(BridgeServer(bridge))
            server_thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            server_thread.start()
            stack.callback(server_thread.join)
            stack.callback(server.shutdown)
            bridge.get_logger().info(f"Z02 RLT bridge listening on http://{bridge.host}:{bridge.port}")
            while not stopping and rclpy.ok():
                rclpy.spin_once(bridge, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
