#!/usr/bin/env python3

import math
import os
import sys
import time
from array import array
from dataclasses import dataclass, replace
from threading import Event, Lock, Thread

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Bool, String
from stereo_camera.frame_hub import FrameHub
from utils import pack_z02_state, process_bgr_array, stamp_to_sec
from zoyi_robot_interfaces.msg import JointPositionCommands, JointState

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, "scripts"))


class ArmControl(Node):
    def __init__(self):
        super().__init__("arm_control")

        # Modify the params
        config = "pi05_zoyi_z02_pytorch"
        # policy_dir = f"{parent_dir}/policies/open_the_gate/pi05_z02_pytorch/2026.10.28_only_arm_1/10000"
        policy_dir = f"/mnt/mydrive/zoyi_brain/policies/open_the_gate/pi05_z02_pytorch/power_ups_2_20261111/30000"
        # task_description = "Pick up the kettle and place it on the base. Then press the button on the kettle to boil the water."
        # task_description = "Move to the gate switch and press the button to open the gate."
        task_description = "Press the button switch to open the gate."
        # task_description = "Place the bottle to the pad."
        # task_description = "Pour water into the cup."
        # task_description = "Place the bottle into the basket."

        # Load parameters
        self.load_parameters()
        config = self.declare_parameter("config", config).value
        policy_dir = self.declare_parameter("policy_dir", policy_dir).value
        train_config = _config.get_config(config)
        self.joint_dim = train_config.data.joint_dim
        if self.joint_dim is None:
            self.joint_dim = 16 if train_config.data.dual_mode else 8
        if self.joint_dim not in (16, 25):
            raise ValueError(
                "This Z02 ROS client supports only the 16-D and 25-D collection layouts")
        # Use checkpoint assets explicitly; keep the original policy loader/compile mode.
        asset_id = self.declare_parameter(
            "asset_id", "z02_gate_25d" if self.joint_dim == 25 else train_config.data.assets.asset_id
        ).value
        train_config = replace(
            train_config, data=replace(
                train_config.data, assets=_config.AssetsConfig(asset_id=asset_id))
        )
        stats_path = os.path.join(
            policy_dir, "assets", asset_id, "norm_stats.json")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"Checkpoint normalization not found: {stats_path}")
        self.get_logger().info(f"Checkpoint: {policy_dir}; assets: {asset_id}")

        # Init the policy
        self.get_logger().info("Loading the PyTorch policy...")
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        weight_path = os.path.join(policy_dir, "model.safetensors")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"PyTorch checkpoint not found: {weight_path}")
        self.policy = _policy_config.create_trained_policy(
            train_config,
            policy_dir,
            pytorch_device=str(self.device),
        )

        # Init a subscriber for the task description
        self.task_description_sub = self.create_subscription(
            String,
            "/manipulation/task_description",
            self.task_description_callback,
            10,
        )
        self.task_description = task_description

        # Init the subscription for joint states
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint/arm_head_waist_states", self._joint_state_callback, 10
        )

        self.enable_inference_sub = self.create_subscription(
            Bool,
            "/enable_inference",
            self.enable_inference_callback,
            10,
        )
        self.enable_inference = False

        self.enable_reset_arm_pose_sub = self.create_subscription(
            Bool,
            "/enable_reset_arm_pose",
            self.enable_reset_arm_pose_callback,
            10,
        )
        self.enable_reset_arm_pose = False

        # Camera frames come from webrtc_cameras_receiver shared-memory hub.
        self.frame_hub = FrameHub(timeout_sec=60.0)
        self.get_logger().info("Reading camera frames from WebRTC frame hub")

        self._last_sync_time = None
        self._last_sync_log_time = None
        self._sync_count = 0
        self.create_timer(1.0, self._sync_watchdog)

        self.joint_position_cmd_pub = self.create_publisher(
            JointPositionCommands,
            "/joint/arm_head_waist_commands",
            10,
        )
        self.joint_position_cmd = JointPositionCommands()
        self.joint_position_cmd.joint_positions = array(
            "f", self.init_positions)
        self.joint_position_cmd.control_mode = [i >= 4 for i in range(len(self.init_positions))]

        self.inference_mode_pub = self.create_publisher(
            Bool, "/inference_mode", 10)
        self.inference_mode_msg = Bool()
        self.inference_mode_msg.data = True

        # Shared state between inference thread and publish timer
        self.obs_lock = Lock()
        self.actions_lock = Lock()
        self.arm_joints = None
        self.head_image = None
        self.right_wrist_image = None
        self.left_wrist_image = None
        self.latest_joint_positions = None
        self._fault_hold = False
        self.scheduler = PlanScheduler(
            SchedulerConfig(
                joint_dim=self.joint_dim,
                horizon=train_config.model.action_horizon,
                observation_timeout=self.observation_timeout,
            )
        )
        self._stop_event = Event()

        # Timer: publish commands only (never blocks on GPU)
        self.timer = self.create_timer(
            1.0 / self.timer_freq, self.control_callback)

        # Initialize the model
        self.get_logger().info("Initializing the model...")
        observation = {
            "state": torch.ones((self.joint_dim,), dtype=torch.float32),
            "observation.images.head": torch.ones((3, self.img_height, self.img_width), dtype=torch.float32),
            "observation.images.hand_right": torch.ones((3, self.img_height, self.img_width), dtype=torch.float32),
            "observation.images.hand_left": torch.ones((3, self.img_height, self.img_width), dtype=torch.float32),
            "prompt": self.task_description,
        }
        with torch.inference_mode():
            self.policy.infer(observation)["actions"]
        self.get_logger().info("Model initialized")

        self.inference_thread = Thread(
            target=self._inference_loop,
            name="policy_inference",
            daemon=True,
        )
        self.inference_thread.start()

    def _stop_execution(self, reason):
        with self.actions_lock:
            self.scheduler.invalidate(reason, time.monotonic())
        self._hold_current_joint_cmd()
        if self.latest_joint_positions is not None:
            self.joint_position_cmd.header.stamp = self.get_clock().now().to_msg()
            self.joint_position_cmd_pub.publish(self.joint_position_cmd)
        self._fault_hold = True
        self.get_logger().warn(
            f"Control stopped: {reason}; disable then enable to retry")

    def task_description_callback(self, msg):
        if msg.data != self.task_description:
            self.task_description = msg.data
            self._stop_execution("task_changed")

    def enable_inference_callback(self, msg):
        if bool(msg.data) != self.enable_inference:
            self._hold_current_joint_cmd()
            with self.actions_lock:
                self.scheduler.set_enabled(bool(msg.data), time.monotonic())
            self.enable_inference = bool(msg.data)
            self._fault_hold = False
            if not msg.data and self.latest_joint_positions is not None:
                self.joint_position_cmd.header.stamp = self.get_clock().now().to_msg()
                self.joint_position_cmd_pub.publish(self.joint_position_cmd)

    def enable_reset_arm_pose_callback(self, msg):
        if bool(msg.data) != self.enable_reset_arm_pose:
            self._stop_execution("reset_changed")
        self.enable_reset_arm_pose = bool(msg.data)

    def _sync_watchdog(self):
        if self._last_sync_time is None:
            self.get_logger().warn(
                "No joint and camera observation yet (waiting for joint states and frame hub)")
            return
        age = time.monotonic() - self._last_sync_time
        if age > 1.0:
            self.get_logger().warn(f"Joint and camera observation stalled for {age:.1f}s")

    def _joint_state_callback(self, joint_state_msg):
        positions = np.asarray(joint_state_msg.position[:40], dtype=np.float32)
        if positions.shape != (40,) or not np.all(np.isfinite(positions)):
            if self.enable_inference:
                self._stop_execution("invalid_joint_state")
            return
        head_bgr = self.frame_hub.latest_bgr('head')
        right_bgr = self.frame_hub.latest_bgr('right_wrist')
        left_bgr = self.frame_hub.latest_bgr('left_wrist')
        if head_bgr is None or right_bgr is None or left_bgr is None:
            return
        now = time.monotonic()
        if self._last_sync_log_time is None:
            self.get_logger().info(
                "joint and camera observation ready "
                f"(count={self._sync_count}, "
                f"joint={stamp_to_sec(joint_state_msg.header.stamp):.3f}, "
                f"frames=shm)"
            )
            self._last_sync_log_time = now
        self.inference_mode_pub.publish(self.inference_mode_msg)

        arm_joints = pack_z02_state(positions, self.joint_dim, self.gripper_state_threshold)
        head_image = process_bgr_array(
            head_bgr, img_size=self.img_size, as_numpy=True)
        right_wrist_image = process_bgr_array(
            right_bgr, rotate=True, img_size=self.img_size, as_numpy=True)
        left_wrist_image = process_bgr_array(
            left_bgr, rotate=False, img_size=self.img_size, as_numpy=True)
        latest_joint_positions = array("f", positions)

        with self.obs_lock:
            self.arm_joints = arm_joints
            self.head_image = head_image
            self.right_wrist_image = right_wrist_image
            self.left_wrist_image = left_wrist_image
            self.latest_joint_positions = latest_joint_positions
            self._last_sync_time = now
            self._sync_count += 1

    def _observation(self):
        with self.obs_lock:
            if self.arm_joints is None or self._last_sync_time is None:
                return None, None
            images = (self.head_image, self.right_wrist_image,
                      self.left_wrist_image)
            if any(image is None for image in images):
                return None, None
            state = self.arm_joints.copy()
            info = ObservationInfo(
                self._sync_count,
                self._last_sync_time,
                self._last_sync_time,
            )
        return {
            "state": state,
            "observation.images.head": images[0],
            "observation.images.hand_right": images[1],
            "observation.images.hand_left": images[2],
            "prompt": self.task_description,
        }, info

    def _inference_loop(self):
        """Prefetch full plans; never publish or replace an active plan here."""
        while not self._stop_event.is_set():
            if not self.enable_inference or self.enable_reset_arm_pose:
                self._stop_event.wait(0.01)
                continue
            observation, info = self._observation()
            with self.actions_lock:
                request = self.scheduler.request(time.monotonic(), info)
            if request is None:
                self._stop_event.wait(0.001)
                continue
            try:
                started = time.monotonic()
                with torch.inference_mode():
                    actions = self.policy.infer(observation)["actions"]
                finished = time.monotonic()
                with self.actions_lock:
                    accepted = self.scheduler.complete(
                        request, actions, finished)
                if accepted:
                    self.get_logger().info(
                        f"Inferred {len(actions)} actions in {finished - started:.3f}s")
            except Exception as exc:
                with self.actions_lock:
                    self.scheduler.fail(
                        request, "inference_error", time.monotonic())
                self.get_logger().error(f"Inference failed: {exc}")

    def control_callback(self):
        """Publish one time-aligned 25-D row; GPU work stays in the worker."""
        if self.enable_inference and self.enable_reset_arm_pose:
            self.publish_default_action()
            return
        if not self.enable_inference:
            return
        with self.obs_lock:
            have_joints = self.latest_joint_positions is not None
        if not have_joints:
            return
        _, info = self._observation()
        with self.actions_lock:
            decision = self.scheduler.tick(time.monotonic(), info)
        if decision.action is None:
            self._hold_current_joint_cmd()
            if decision.state == "FAULT_STOP" and not self._fault_hold:
                self._fault_hold = True
                self.get_logger().warn(
                    f"Control stopped: {decision.reason}; disable then enable to retry")
        else:
            self._fill_joint_position_cmd(decision.action)
        self.joint_position_cmd.header.stamp = self.get_clock().now().to_msg()
        self.joint_position_cmd_pub.publish(self.joint_position_cmd)

    def _hold_current_joint_cmd(self):
        """Set command to the latest measured joints to avoid stale jumps."""
        with self.obs_lock:
            positions = self.latest_joint_positions
        if positions is None:
            return
        self.joint_position_cmd.joint_positions = array("f", positions[:40])

    def _fill_joint_position_cmd(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.joint_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"Expected a finite {self.joint_dim}-D action; got {action.shape}")
        # Left arm
        self.joint_position_cmd.joint_positions[4:11] = array("f", action[:7])
        if action[7] > 0.3:
            self.joint_position_cmd.joint_positions[20:
                                                    30] = self.left_finger_target_pos
        else:
            self.joint_position_cmd.joint_positions[20:30] = self.left_finger_init_pos

        # Right arm
        self.joint_position_cmd.joint_positions[11:18] = array(
            "f", action[8:15])
        if action[15] > 0.3:
            self.joint_position_cmd.joint_positions[30:
                                                    40] = self.right_finger_target_pos
        else:
            self.joint_position_cmd.joint_positions[30:
                                                    40] = self.right_finger_init_pos
        if self.joint_dim == 25:
            self.joint_position_cmd.joint_positions[18:20] = array("f", action[20:22])

    def publish_default_action(self):
        self.joint_position_cmd.joint_positions = array(
            "f", self.init_positions)
        self.joint_position_cmd.header.stamp = self.get_clock().now().to_msg()
        self.joint_position_cmd_pub.publish(self.joint_position_cmd)

    def destroy_node(self):
        self._stop_event.set()
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        hub = getattr(self, 'frame_hub', None)
        if hub is not None:
            hub.close()
            self.frame_hub = None
        super().destroy_node()

    def load_parameters(self):
        self.img_height = 360
        self.img_width = 640
        self.img_size = (self.img_width, self.img_height)
        self.timer_freq = 30
        self.observation_timeout = 5.0
        self.history_size = 1
        self.gripper_state_threshold = 4000
        self.left_finger_init_pos = array(
            "f",
            [4000.0, 1000.0, 1500.0, 4096.0, 4069.0, 4069.0, 4096.0, 4069.0,    0.0, 4069.0],
        )
        self.left_finger_target_pos = array(
            "f",
            [3000.0, 1000.0,    0.0, 4096.0, 4069.0,    0.0, 4096.0,    0.0,    0.0,    0.0],
        )
        self.right_finger_init_pos = array(
            "f",
            [4000.0, 3000.0, 1500.0,    0.0, 4069.0, 4069.0,    0.0, 4069.0, 4096.0, 4069.0],
        )
        self.right_finger_target_pos = array(
            "f",
            [4000.0, 3000.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0, 4096.0,    0.0],
        )
        self.init_positions = [
            1.11, -2.38, -1.28, 0.0,
            0.78, 0.1, -0.2, -1.74, -0.17, -0.5, 0.0,
            -0.78, -0.1, 0.2, 1.74, 0.17, 0.5, 0.0,
            0.0, 0.1,
            *self.left_finger_init_pos,
            *self.right_finger_init_pos,
        ]


# Keep the bounded scheduler in this entry point: no extra runtime modules.
@dataclass(frozen=True)
class SchedulerConfig:
    fps: float = 30
    joint_dim: int = 25
    horizon: int = 50
    replan_steps: int = 20
    prefetch_lead_steps: int = 10
    max_extension_steps: int = 6
    inference_timeout: float = 2.0
    observation_timeout: float = 0.2
    mode: str = "prefetch"

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        for name in ("joint_dim", "horizon", "replan_steps", "prefetch_lead_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_extension_steps, int) or self.max_extension_steps < 0:
            raise ValueError(
                "max_extension_steps must be a nonnegative integer")
        if self.replan_steps > self.horizon or self.prefetch_lead_steps > self.replan_steps:
            raise ValueError(
                "require prefetch_lead_steps <= replan_steps <= horizon")
        for name in ("inference_timeout", "observation_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.mode not in ("prefetch", "legacy"):
            raise ValueError("mode must be 'prefetch' or 'legacy'")


@dataclass(frozen=True)
class ObservationInfo:
    observation_id: int
    time: float
    received: float
    healthy: bool = True


@dataclass(frozen=True)
class InferenceRequest:
    generation: int
    request_id: int
    observation_id: int
    observation_time: float
    started: float


@dataclass(frozen=True)
class Decision:
    action: np.ndarray | None
    state: str
    reason: str
    plan_id: int | None = None
    index: int | None = None
    switched: bool = False
    switch_delta: np.ndarray | None = None


@dataclass
class _Plan:
    request: InferenceRequest
    actions: np.ndarray
    activated: float | None = None
    legacy_index: int = 0


class PlanScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.state = "DISABLED"
        self.generation = 0
        self.enabled = False
        self._reason = "disabled"
        self._next_request_id = 1
        self._reset()

    def _reset(self):
        self._active: _Plan | None = None
        self._pending: _Plan | None = None
        self._inflight: InferenceRequest | None = None
        self._last_action: np.ndarray | None = None
        self._saw_observation = False

    def set_enabled(self, enabled: bool, now: float, reason: str = "operator"):  # noqa: FBT001
        self._check_now(now)
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.generation += 1
        self._reset()
        self.state = "WAITING" if enabled else "DISABLED"
        self._reason = reason

    def invalidate(self, reason: str, now: float):
        self._check_now(now)
        if self.state == "FAULT_STOP":
            return
        self.generation += 1
        self._reset()
        self.state = "FAULT_STOP"
        self._reason = reason

    @staticmethod
    def _check_now(now: float):
        if not math.isfinite(now):
            raise ValueError("now must be finite")

    def _observation_problem(self, obs: ObservationInfo, now: float) -> str | None:
        if not obs.healthy:
            return "observation_unhealthy"
        stamps = (obs.time, obs.received)
        if not all(math.isfinite(stamp) for stamp in stamps):
            return "observation_timestamp_invalid"
        if any(stamp > now + 1e-9 for stamp in stamps):
            return "observation_from_future"
        if any(now - stamp > self.config.observation_timeout + 1e-9 for stamp in stamps):
            return "observation_stale"
        return None

    def _ready(self, now: float, obs: ObservationInfo | None) -> bool:
        self._check_now(now)
        if not self.enabled or self.state == "FAULT_STOP":
            return False
        if self._inflight is not None and now - self._inflight.started > self.config.inference_timeout + 1e-9:
            self._inflight = None
            self._reason = "inference_timeout"
        if obs is None:
            if self._saw_observation:
                self.invalidate("observation_missing", now)
            return False
        problem = self._observation_problem(obs, now)
        if problem is not None:
            self.invalidate(problem, now)
            return False
        self._saw_observation = True
        return True

    def request(self, now: float, obs: ObservationInfo | None) -> InferenceRequest | None:
        if not self._ready(now, obs) or self._inflight is not None or self._pending is not None:
            return None
        if self._active is not None:
            if self.config.mode == "legacy":
                if self._active.legacy_index < self.config.replan_steps:
                    return None
            else:
                elapsed = now - self._active.activated
                prefetch_at = (self.config.replan_steps -
                               self.config.prefetch_lead_steps) / self.config.fps
                horizon_remaining = self._expires(self._active) - now
                if (
                    elapsed + 1e-9 < prefetch_at
                    and horizon_remaining > self.config.prefetch_lead_steps / self.config.fps
                ):
                    return None
        request = InferenceRequest(
            generation=self.generation,
            request_id=self._next_request_id,
            observation_id=obs.observation_id,
            observation_time=obs.time,
            started=now,
        )
        self._next_request_id += 1
        self._inflight = request
        return request

    def _current_request(self, request: InferenceRequest) -> bool:
        return (
            self.enabled
            and self.state != "FAULT_STOP"
            and request.generation == self.generation
            and self._inflight == request
        )

    def complete(self, request: InferenceRequest, actions, now: float) -> bool:
        self._check_now(now)
        if not self._current_request(request):
            return False
        if now < request.started or now - request.started > self.config.inference_timeout + 1e-9:
            self._inflight = None
            self._reason = "inference_timeout"
            return False
        if self._active is not None and self.config.mode != "legacy":
            deadline = min(self._expires(self._active), self._replan_deadline(self._active))
            if now >= deadline:
                self.invalidate("result_after_plan_deadline", now)
                return False
        try:
            actions = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError):
            self.invalidate("prediction_invalid", now)
            return False
        if actions.shape != (self.config.horizon, self.config.joint_dim) or not np.all(np.isfinite(actions)):
            self.invalidate("prediction_invalid", now)
            return False
        if self.config.mode != "legacy" and now >= request.observation_time + self.config.horizon / self.config.fps:
            self.invalidate("prediction_expired", now)
            return False
        actions = actions.copy()
        actions.flags.writeable = False
        self._pending = _Plan(request, actions)
        self._inflight = None
        return True

    def fail(self, request: InferenceRequest, reason: str, now: float):
        if self._current_request(request):
            self.invalidate(f"inference_failed:{reason}", now)

    def _expires(self, plan: _Plan) -> float:
        return plan.request.observation_time + self.config.horizon / self.config.fps

    def _replan_deadline(self, plan: _Plan) -> float:
        return plan.activated + (self.config.replan_steps + self.config.max_extension_steps) / self.config.fps

    def tick(self, now: float, obs: ObservationInfo | None) -> Decision:
        if not self._ready(now, obs):
            return Decision(None, self.state, self._reason)
        # A ready replacement cannot revive a plan after its hard deadline.
        # This check precedes installation, including after a delayed timer tick.
        if self._active is not None and self.config.mode != "legacy":
            expired_horizon = now + 1e-9 >= self._expires(self._active)
            late_replan = now + 1e-9 >= self._replan_deadline(self._active)
            if expired_horizon or late_replan:
                reason = "plan_horizon_exhausted" if expired_horizon else "replan_deadline"
                self.invalidate(reason, now)
                return Decision(None, self.state, self._reason)
        switched = False
        previous_action = self._last_action
        if self._pending is not None:
            install = self._active is None
            if self._active is not None:
                install = (
                    self._active.legacy_index >= self.config.replan_steps
                    if self.config.mode == "legacy"
                    else now + 1e-9 >= self._active.activated + self.config.replan_steps / self.config.fps
                )
            if install:
                if self.config.mode != "legacy" and now >= self._expires(self._pending):
                    self.invalidate("pending_plan_expired", now)
                    return Decision(None, self.state, self._reason)
                self._active = self._pending
                self._pending = None
                self._active.activated = now
                self.state = "NORMAL"
                self._reason = "plan_activated"
                switched = True
        if self._active is None:
            self.state = "WAITING"
            self._reason = "waiting_for_prediction"
            return Decision(None, self.state, self._reason)
        plan = self._active
        if self.config.mode == "legacy":
            index = plan.legacy_index
            if index >= self.config.replan_steps:
                self.state = "WAITING"
                self._reason = "legacy_queue_empty"
                return Decision(None, self.state, self._reason, plan.request.request_id)
            plan.legacy_index += 1
            reason = "legacy_action"
        else:
            index = max(0, math.floor((now - plan.request.observation_time) * self.config.fps + 1e-9))
            if now + 1e-9 >= self._expires(plan) or index >= self.config.horizon:
                self.invalidate("plan_horizon_exhausted", now)
                return Decision(None, self.state, self._reason)
            reason = (
                "bounded_extension"
                if now + 1e-9 >= plan.activated + self.config.replan_steps / self.config.fps
                else "planned_action"
            )
        action = plan.actions[index].copy()
        delta = action - previous_action if switched and previous_action is not None else None
        self._last_action = action.copy()
        self._reason = reason
        return Decision(action, self.state, reason, plan.request.request_id, index, switched, delta)


def main():
    rclpy.init()
    arm_control = ArmControl()
    try:
        rclpy.spin(arm_control)
    except KeyboardInterrupt:
        pass
    finally:
        arm_control.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
