#!/usr/bin/env python3

from array import array
from collections import deque
import os
import sys
from threading import Event
from threading import Lock
from threading import Thread
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
from stereo_camera.frame_hub import FrameHub
from utils import pack_z02_state
from utils import process_bgr_array
from utils import stamp_to_sec
from zoyi_robot_interfaces.msg import JointPositionCommands
from zoyi_robot_interfaces.msg import JointState

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, "scripts"))


class ArmControl(Node):
    def __init__(self):
        super().__init__("arm_control")

        # Modify the params
        config = "pi05_zoyi_z02"
        policy_dir = f"{parent_dir}/policies/boil_water/pi05_z02_jax/20261006"
        task_description = "Pick up the kettle and place it on the base. Then press the button on the kettle to boil the water."
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
            raise ValueError("This Z02 ROS client supports only the 16-D and 25-D collection layouts")

        # Init the policy (JAX checkpoint: <policy_dir>/params)
        self.get_logger().info("Loading the JAX policy...")
        params_path = os.path.join(policy_dir, "params")
        if not os.path.isdir(params_path):
            raise FileNotFoundError(
                f"JAX checkpoint not found: {params_path}"
            )
        self.policy = _policy_config.create_trained_policy(
            train_config,
            policy_dir,
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
            "f", self.init_positions
        )
        self.joint_position_cmd.control_mode = [i >= 4 for i in range(len(self.init_positions))]

        self.inference_mode_pub = self.create_publisher(
            Bool, "/inference_mode", 10
        )
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
        self.actions = deque()
        self._stop_event = Event()

        # Timer: publish commands only (never blocks on GPU)
        self.timer = self.create_timer(
            1.0 / self.timer_freq, self.control_callback
        )

        # Initialize the model (warmup / JIT compile)
        self.get_logger().info("Initializing the model...")
        observation = {
            "state": np.ones((self.joint_dim,), dtype=np.float32),
            "observation.images.head": np.ones(
                (3, self.img_height, self.img_width), dtype=np.float32
            ),
            "observation.images.hand_right": np.ones(
                (3, self.img_height, self.img_width), dtype=np.float32
            ),
            "observation.images.hand_left": np.ones(
                (3, self.img_height, self.img_width), dtype=np.float32
            ),
            "prompt": self.task_description,
        }
        self.policy.infer(observation)["actions"]
        self.get_logger().info("Model initialized")

        self.inference_thread = Thread(
            target=self._inference_loop,
            name="policy_inference",
            daemon=True,
        )
        self.inference_thread.start()

    def task_description_callback(self, msg):
        self.task_description = msg.data

    def enable_inference_callback(self, msg):
        if msg.data != self.enable_inference:
            with self.actions_lock:
                self.actions.clear()
            if msg.data:
                self._hold_current_joint_cmd()
            self.enable_inference = msg.data

    def enable_reset_arm_pose_callback(self, msg):
        if msg.data and not self.enable_reset_arm_pose:
            with self.actions_lock:
                self.actions.clear()
        self.enable_reset_arm_pose = msg.data

    def _sync_watchdog(self):
        if self._last_sync_time is None:
            self.get_logger().warn(
                "No joint and camera observation yet "
                "(waiting for joint states and frame hub)"
            )
            return
        age = time.monotonic() - self._last_sync_time
        if age > 1.0:
            self.get_logger().warn(
                f"Joint and camera observation stalled for {age:.1f}s"
            )

    def _joint_state_callback(self, joint_state_msg):
        positions = np.asarray(joint_state_msg.position[:40], dtype=np.float32)
        if positions.shape != (40,) or not np.all(np.isfinite(positions)):
            return
        head_bgr = self.frame_hub.latest_bgr('head')
        right_bgr = self.frame_hub.latest_bgr('right_wrist')
        left_bgr = self.frame_hub.latest_bgr('left_wrist')
        if head_bgr is None or right_bgr is None or left_bgr is None:
            return
        now = time.monotonic()
        self._last_sync_time = now
        self._sync_count += 1
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
            head_bgr, img_size=self.img_size, as_numpy=True
        )
        right_wrist_image = process_bgr_array(
            right_bgr, rotate=True, img_size=self.img_size, as_numpy=True
        )
        left_wrist_image = process_bgr_array(
            left_bgr, rotate=False, img_size=self.img_size, as_numpy=True
        )
        latest_joint_positions = array("f", positions)

        with self.obs_lock:
            self.arm_joints = arm_joints
            self.head_image = head_image
            self.right_wrist_image = right_wrist_image
            self.left_wrist_image = left_wrist_image
            self.latest_joint_positions = latest_joint_positions

    def _inference_loop(self):
        """Background thread: fill action queue via policy.infer."""
        while not self._stop_event.is_set():
            if not self.enable_inference or self.enable_reset_arm_pose:
                time.sleep(0.01)
                continue
            if self.joint_dim == 25 and (
                self._last_sync_time is None
                or time.monotonic() - self._last_sync_time > 1.0
            ):
                time.sleep(0.01)
                continue

            with self.actions_lock:
                need_infer = len(self.actions) == 0
            if not need_infer:
                time.sleep(0.001)
                continue

            with self.obs_lock:
                if (
                    self.arm_joints is None
                    or self.head_image is None
                    or self.right_wrist_image is None
                    or self.left_wrist_image is None
                ):
                    time.sleep(0.01)
                    continue
                observation = {
                    "state": self.arm_joints.copy(),
                    "observation.images.head": self.head_image.copy(),
                    "observation.images.hand_right": (
                        self.right_wrist_image.copy()
                    ),
                    "observation.images.hand_left": (
                        self.left_wrist_image.copy()
                    ),
                    "prompt": self.task_description,
                }

            actions = self.policy.infer(observation)["actions"][:20]

            with self.actions_lock:
                if (
                    self.enable_inference
                    and not self.enable_reset_arm_pose
                ):
                    self.actions = deque(actions)
                    self.get_logger().info(
                        f"Inferred {len(self.actions)} actions"
                    )

    def control_callback(self):
        """Timer callback: publish commands only, never block on inference."""
        if self.enable_inference and self.enable_reset_arm_pose:
            self.publish_default_action()
            return

        if not self.enable_inference:
            return
        if self.joint_dim == 25 and (
            self._last_sync_time is None
            or time.monotonic() - self._last_sync_time > 1.0
        ):
            with self.actions_lock:
                self.actions.clear()
            return

        with self.obs_lock:
            obs_ready = (
                self.arm_joints is not None
                and self.head_image is not None
                and self.right_wrist_image is not None
                and self.left_wrist_image is not None
            )
        if not obs_ready:
            return

        with self.actions_lock:
            action = self.actions.popleft() if self.actions else None

        if action is not None:
            self._fill_joint_position_cmd(action)
            with self.actions_lock:
                remaining = len(self.actions)
            self.get_logger().info(f"Actions remaining: {remaining}")

        self.joint_position_cmd.header.stamp = (
            self.get_clock().now().to_msg()
        )
        self.joint_position_cmd_pub.publish(self.joint_position_cmd)

    def _hold_current_joint_cmd(self):
        """Set command to the latest measured joints to avoid stale jumps."""
        with self.obs_lock:
            positions = self.latest_joint_positions
        if positions is None:
            return
        self.joint_position_cmd.joint_positions = array("f", positions)

    def _fill_joint_position_cmd(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.joint_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected a finite {self.joint_dim}-D action; got {action.shape}")
        # Left arm
        self.joint_position_cmd.joint_positions[4:11] = array(
            "f", action[:7]
        )
        if action[7] > 0.3:
            self.joint_position_cmd.joint_positions[20:30] = (
                self.left_finger_target_pos
            )
        else:
            self.joint_position_cmd.joint_positions[20:30] = (
                self.left_finger_init_pos
            )

        # Right arm
        self.joint_position_cmd.joint_positions[11:18] = array(
            "f", action[8:15]
        )
        if action[15] > 0.3:
            self.joint_position_cmd.joint_positions[30:40] = (
                self.right_finger_target_pos
            )
        else:
            self.joint_position_cmd.joint_positions[30:40] = (
                self.right_finger_init_pos
            )
        if self.joint_dim == 25:
            self.joint_position_cmd.joint_positions[18:20] = array("f", action[20:22])

    def publish_default_action(self):
        self.joint_position_cmd.joint_positions = array(
            "f", self.init_positions
        )
        self.joint_position_cmd.header.stamp = (
            self.get_clock().now().to_msg()
        )
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
        self.max_delay = 0.1
        self.history_size = 1
        self.gripper_state_threshold = 4000
        self.left_finger_init_pos = array(
            "f",
            [4000.0, 1800.0, 1500.0, 4096.0, 4069.0,
                4069.0, 4096.0, 4069.0, 0.0, 4069.0],
        )
        self.left_finger_target_pos = array(
            "f",
            [
                4000.0, 1000.0, 0.0, 4096.0, 0.0, 0.0, 4096.0, 0.0, 0.0, 0.0
            ],
        )
        self.right_finger_init_pos = array(
            "f",
            [
                4000.0, 2200.0, 1500.0, 0.0, 4069.0,
                4069.0, 0.0, 4069.0, 4096.0, 4069.0
            ],
        )
        self.right_finger_target_pos = array(
            "f",
            [
                4000.0, 2200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4096.0, 0.0
            ],
        )
        self.init_positions = [
            1.11, -2.38, -1.28, 0.0,
            0.78, 0.1, -0.2, -1.74, -0.17, -0.5, 0.0,
            -0.78, -0.1, 0.2, 1.74, 0.17, 0.5, 0.0,
            0.0, -0.34,
            *self.left_finger_init_pos,
            *self.right_finger_init_pos,
        ]


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
