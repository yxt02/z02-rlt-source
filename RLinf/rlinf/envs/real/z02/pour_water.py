# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Z02 water-pouring task with operator-scored episodes."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from http.client import HTTPException
from threading import Lock
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4
from zipfile import BadZipFile

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rlinf.utils.logging import get_logger

_CAMERAS = ("head", "left_wrist", "right_wrist")
_ACTION_DIM = 18
_STATE_DIM = 25
_CONTROLLED_STATE_INDICES = (*range(16), 20, 21)
_logger = get_logger()


class _Outcome(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2
    ABORT = 3


@dataclass(frozen=True, kw_only=True)
class Z02PourWaterConfig:
    """Configure the bridge connection and operator control."""

    task_description: str
    bridge_url: str = "http://127.0.0.1:8765"
    image_width: int = 320
    image_height: int = 180
    request_timeout: float = 3.0
    reset_timeout: float = 310.0
    episode_step_limit: int = 600
    manual_episode_control_only: bool = True

    def __post_init__(self) -> None:
        parts = urlsplit(self.bridge_url)
        if (
            parts.scheme != "http"
            or not parts.hostname
            or parts.query
            or parts.fragment
        ):
            raise ValueError(f"Expected a Z02 bridge HTTP URL, got {self.bridge_url!r}.")
        if min(self.image_height, self.image_width, self.episode_step_limit) <= 0:
            raise ValueError("Z02 image dimensions and episode step limit must be positive.")
        if any(
            not math.isfinite(timeout) or timeout <= 0
            for timeout in (self.request_timeout, self.reset_timeout)
        ):
            raise ValueError("Z02 bridge request and reset timeouts must be finite and positive.")
        if not self.task_description.strip():
            raise ValueError("Z02 task_description must contain the instruction used in Stage 1.")
        if not self.manual_episode_control_only:
            raise ValueError("Z02 episodes require manual_episode_control_only=true.")


@dataclass(frozen=True)
class _Snapshot:
    episode_id: int
    state: NDArray[np.float32]
    frames: dict[str, NDArray[np.uint8]]
    executed_actions: NDArray[np.float32]
    outcome: _Outcome
    actor_enabled: bool

    def observation(self) -> dict[str, Any]:
        return {"state": {"joint_positions": self.state}, "frames": self.frames}


class Z02BridgeError(RuntimeError):
    """The bridge could not provide a valid real-robot transition."""


class Z02PourWaterEnv(gym.Env):
    """Connect RLinf steps to a Z02 ROS 2 bridge."""

    def __init__(
        self,
        *,
        override_cfg: dict[str, Any],
        worker_info: Any,
        robot_info: Any,
        env_idx: int,
    ) -> None:
        del worker_info, robot_info, env_idx
        self.config = Z02PourWaterConfig(**override_cfg)
        self.task_description = self.config.task_description
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_ACTION_DIM,), dtype=np.float32
        )
        self._frame_shape = (self.config.image_height, self.config.image_width, 3)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_positions": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(_STATE_DIM,), dtype=np.float32
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(0, 255, self._frame_shape, np.uint8)
                        for name in _CAMERAS
                    }
                ),
            }
        )
        self._response_limit = len(_CAMERAS) * math.prod(self._frame_shape) + 16_384
        self._opener = build_opener(ProxyHandler({}))
        self._lock = Lock()
        self._closed = False
        self._reset_id: str | None = None
        self._snapshot: _Snapshot | None = None
        self._active_episode: int | None = None
        self._steps = 0

    def _post(
        self, endpoint: str, payload: dict[str, Any] | None, timeout: float
    ) -> bytes:
        request = Request(
            f"{self.config.bridge_url.rstrip('/')}/{endpoint}",
            data=b"" if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                if endpoint in ("hold", "cancel-reset"):
                    if response.status != 204:
                        raise Z02BridgeError(
                            f"Z02 /{endpoint} expected HTTP 204, got {response.status}."
                        )
                    return b""
                data = response.read(self._response_limit + 1)
        except HTTPError as exc:
            with exc:
                detail = exc.read(1024).decode(errors="replace")
            raise Z02BridgeError(f"Z02 /{endpoint} returned HTTP {exc.code}: {detail}") from exc
        except (HTTPException, OSError) as exc:
            raise Z02BridgeError(f"Z02 /{endpoint} request failed: {exc}") from exc
        if len(data) > self._response_limit:
            raise Z02BridgeError(f"Z02 /{endpoint} exceeded the configured observation size.")
        return data

    def _read_snapshot(self, data: bytes) -> _Snapshot:
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                snapshot = _Snapshot(
                    episode_id=int(archive["episode_id"].item()),
                    state=archive["state"],
                    frames={name: archive[name] for name in _CAMERAS},
                    executed_actions=archive["executed_actions"],
                    outcome=_Outcome(archive["outcome"].item()),
                    actor_enabled=bool(archive["actor_enabled"].item()),
                )
        except (BadZipFile, EOFError, KeyError, OSError, TypeError, ValueError) as exc:
            raise Z02BridgeError(f"Z02 bridge returned an invalid observation: {exc}") from exc
        if (
            snapshot.state.shape != (_STATE_DIM,)
            or snapshot.state.dtype != np.float32
            or not np.isfinite(snapshot.state).all()
            or np.any(snapshot.state[22:25])
        ):
            raise Z02BridgeError(
                "Z02 bridge must return 25 finite float32 state values with zero base velocities."
            )
        for name, frame in snapshot.frames.items():
            if frame.shape != self._frame_shape or frame.dtype != np.uint8:
                raise Z02BridgeError(
                    f"Z02 {name} expected RGB uint8 {self._frame_shape}, "
                    f"got {frame.dtype} {frame.shape}."
                )
        actions = snapshot.executed_actions
        if (
            actions.shape not in ((0, _ACTION_DIM), (1, _ACTION_DIM))
            or actions.dtype != np.float32
            or not np.isfinite(actions).all()
        ):
            raise Z02BridgeError("Z02 bridge must return zero or one finite 18-D applied action.")
        return snapshot

    def _hold(self) -> None:
        with self._lock:
            episode_id = self._active_episode
            self._active_episode = None
        if episode_id is None:
            return
        try:
            self._post("hold", {"episode_id": episode_id}, self.config.request_timeout)
        except Z02BridgeError:
            with self._lock:
                if self._active_episode is None and not self._closed:
                    self._active_episode = episode_id
            raise

    def close(self) -> None:
        """Disarm the active episode and close the client."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            reset_id = self._reset_id if self._active_episode is None else None
        if reset_id is not None:
            try:
                self._post(
                    "cancel-reset", {"reset_id": reset_id}, self.config.request_timeout
                )
            except Z02BridgeError as exc:
                _logger.warning("Could not cancel the pending Z02 reset: %s", exc)
        try:
            self._hold()
        except Z02BridgeError as exc:
            _logger.warning("Could not disarm the Z02 bridge: %s", exc)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Wait for the operator's ready event and read the reset observation."""
        super().reset(seed=seed)
        self._hold()
        reset_id = uuid4().hex
        with self._lock:
            if self._closed:
                raise Z02BridgeError("Z02 is closed.")
            self._reset_id = reset_id
        try:
            snapshot = self._read_snapshot(
                self._post("reset", {"reset_id": reset_id}, self.config.reset_timeout)
            )
            with self._lock:
                self._active_episode = snapshot.episode_id
                closed = self._closed
            if closed:
                self._hold()
                raise Z02BridgeError("Z02 closed during reset.")
            if snapshot.outcome is not _Outcome.RUNNING or len(snapshot.executed_actions):
                raise Z02BridgeError("Z02 reset expected a running episode and no applied actions.")
            self._snapshot = snapshot
            self._steps = 0
            return snapshot.observation(), {"rlt_switch_flags": snapshot.actor_enabled}
        except Z02BridgeError:
            self.close()
            raise
        finally:
            with self._lock:
                self._reset_id = None

    def get_hold_action(self, fallback: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
        """Return the most recently measured upper-body coordinates."""
        if self._snapshot is not None:
            return self._snapshot.state[list(_CONTROLLED_STATE_INDICES)]
        if fallback is not None:
            return fallback
        raise Z02BridgeError("Z02 has no observed hold action before reset.")

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one absolute upper-body command and read the operator outcome."""
        with self._lock:
            if self._closed:
                raise Z02BridgeError("Z02 is closed.")
            episode_id = self._active_episode
        if self._snapshot is None:
            raise Z02BridgeError("Reset Z02 before requesting an action.")
        if episode_id is None:
            return self._snapshot.observation(), 0.0, False, False, {
                "rlt_switch_flags": self._snapshot.actor_enabled,
                "executed_action": np.zeros(_ACTION_DIM, dtype=np.float32),
                "action_executed": False,
            }

        action = np.asarray(action, dtype=np.float32)
        try:
            if action.shape != (_ACTION_DIM,) or not np.isfinite(action).all():
                raise Z02BridgeError(f"Expected a finite 18-D Z02 command, got {action.shape}.")
            response = self._post(
                "step",
                {"episode_id": episode_id, "actions": [action.tolist()]},
                self.config.request_timeout,
            )
            snapshot = self._read_snapshot(response)
            if snapshot.episode_id != episode_id:
                raise Z02BridgeError("Z02 bridge replied with a different episode ID.")
            if snapshot.outcome is _Outcome.ABORT:
                raise Z02BridgeError(f"Z02 episode {snapshot.episode_id} was aborted.")
            action_executed = len(snapshot.executed_actions) == 1
            if not action_executed and snapshot.outcome is _Outcome.RUNNING:
                raise Z02BridgeError("A running Z02 step must report the applied command.")
            self._steps += int(action_executed)
            terminated = snapshot.outcome is not _Outcome.RUNNING
            if terminated or self._steps >= self.config.episode_step_limit:
                self._hold()
                terminated = True
        except Z02BridgeError:
            self.close()
            raise

        self._snapshot = snapshot
        info = {
            "rlt_switch_flags": snapshot.actor_enabled,
            "executed_action": (
                snapshot.executed_actions[0]
                if action_executed
                else np.zeros(_ACTION_DIM, dtype=np.float32)
            ),
            "action_executed": action_executed,
        }
        return (
            snapshot.observation(),
            float(snapshot.outcome is _Outcome.SUCCESS),
            terminated,
            False,
            info,
        )
