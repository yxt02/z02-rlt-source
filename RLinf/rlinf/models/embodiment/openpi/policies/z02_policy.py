# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Z02 policy input, normalization and upper-body action decoding."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

Z02_STATE_DIM = 25
Z02_MODEL_DIM = 32
Z02_ACTIVE_ACTION_INDICES = (*range(16), 20, 21)
_Z02_DELTA_INDICES = (*range(7), *range(8, 15))
_STATE_KEY = "observation.joint_positions"
Z02_ACTION_KEY = "action.joint_positions"


def prepare_z02_state(raw_state: np.ndarray) -> np.ndarray:
    """Pad dataset/robot states and zero the base velocity coordinates."""
    raw_state = np.asarray(raw_state, dtype=np.float32)
    if raw_state.shape[-1:] != (Z02_STATE_DIM,):
        raise ValueError(f"Expected a Z02 state with width 25, got {raw_state.shape}.")
    if not np.isfinite(raw_state[..., :22]).all():
        raise ValueError("Expected finite Z02 joint and gripper feedback.")
    state = np.zeros((*raw_state.shape[:-1], Z02_MODEL_DIM), dtype=np.float32)
    state[..., :22] = raw_state[..., :22]
    return state


def prepare_z02_training_actions(
    raw_state: np.ndarray, raw_actions: np.ndarray
) -> np.ndarray:
    """Pad absolute action chunks and express arm commands relative to the state."""
    raw_state = np.asarray(raw_state, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if (
        raw_state.shape[-1:] != (Z02_STATE_DIM,)
        or raw_actions.shape[-1:] != (Z02_STATE_DIM,)
        or raw_actions.ndim != raw_state.ndim + 1
        or raw_actions.shape[:-2] != raw_state.shape[:-1]
    ):
        raise ValueError(
            "Expected aligned Z02 state (..., 25) and actions (..., H, 25), got "
            f"{raw_state.shape} and {raw_actions.shape}."
        )
    active_actions = raw_actions[..., Z02_ACTIVE_ACTION_INDICES]
    if not np.isfinite(active_actions).all():
        raise ValueError("Expected finite Z02 upper-body action targets.")
    actions = np.zeros((*raw_actions.shape[:-1], Z02_MODEL_DIM), dtype=np.float32)
    actions[..., Z02_ACTIVE_ACTION_INDICES] = active_actions
    actions[..., _Z02_DELTA_INDICES] -= raw_state[..., _Z02_DELTA_INDICES][..., None, :]
    return actions


@dataclasses.dataclass(frozen=True)
class Z02Quantiles:
    """Normalization bounds and constant-coordinate metadata for one Z02 vector."""

    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    constant_mask: np.ndarray

    @classmethod
    def from_stats(cls, values: dict, *, key: str) -> Z02Quantiles:
        """Read the Zoyi statistics and resolve degenerate quantile intervals."""

        def vector(field: str) -> np.ndarray:
            result = np.asarray(values[field], dtype=np.float32)
            if result.shape != (Z02_MODEL_DIM,) or not np.isfinite(result).all():
                raise ValueError(f"Expected finite 32-D Z02 stats for {key}.{field}.")
            return result

        def mask(field: str) -> np.ndarray:
            result = np.asarray(values[field])
            if result.shape != (Z02_MODEL_DIM,) or result.dtype != np.bool_:
                raise ValueError(f"Expected a 32-D boolean Z02 mask for {key}.{field}.")
            return result

        low, high = vector("q01"), vector("q99")
        fallback = mask("quantile_fallback_mask")
        low = np.where(fallback, vector("min"), low)
        high = np.where(fallback, vector("max"), high)
        constant = mask("constant_mask").copy()
        if key == "actions":
            active = np.zeros(Z02_MODEL_DIM, dtype=bool)
            active[list(Z02_ACTIVE_ACTION_INDICES)] = True
            constant |= ~active
        else:
            constant[22:] = True
        if np.any(high[~constant] < low[~constant]):
            raise ValueError(f"Expected ordered Z02 quantile bounds for {key}.")
        return cls(vector("mean"), low, high, constant)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Apply the same quantile and constant-coordinate rules as Zoyi OpenPI."""
        scale = np.where(self.constant_mask, 1.0, self.high - self.low + 1e-6)
        return np.where(self.constant_mask, 0.0, (values - self.low) / scale * 2 - 1)

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        """Restore raw units and fill constant coordinates with their means."""
        decoded = (values + 1) / 2 * (self.high - self.low + 1e-6) + self.low
        return np.where(self.constant_mask, self.mean, decoded)


def load_z02_norm_stats(path: Path) -> tuple[Z02Quantiles, Z02Quantiles]:
    """Load the extended Zoyi statistics without dropping mask fields."""
    with path.open() as stream:
        stats = json.load(stream)["norm_stats"]
    return (
        Z02Quantiles.from_stats(stats["state"], key="state"),
        Z02Quantiles.from_stats(stats["actions"], key="actions"),
    )


@dataclasses.dataclass(frozen=True)
class Z02DemoRepack:
    """Map LeRobot Z02 fields to the observation keys used during rollout."""

    def __call__(self, data: dict) -> dict:
        repacked = {
            "observation/state": data[_STATE_KEY],
            "observation/image": data["observation.images.cam2"],
            "observation/extra_view_image-0": data["observation.images.cam4"],
            "observation/extra_view_image-1": data["observation.images.cam3"],
            "actions": data[Z02_ACTION_KEY],
        }
        if data.get("task"):
            repacked["prompt"] = data["task"]
        return repacked


def _rgb_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected a Z02 RGB image, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 Z02 RGB pixels, got {image.dtype}.")
    return image


@dataclasses.dataclass(frozen=True)
class Z02Inputs:
    """Apply the same Z02 feature transforms to demonstrations and rollout."""

    state_stats: Z02Quantiles
    action_stats: Z02Quantiles

    def __call__(self, data: dict) -> dict:
        raw_state = data["observation/state"]
        state = prepare_z02_state(raw_state)
        views = data.get("observation/extra_view_image")
        if views is None:
            left = data["observation/extra_view_image-0"]
            right = data["observation/extra_view_image-1"]
        else:
            if np.ndim(views) != 4 or len(views) != 2:
                raise ValueError("Expected two stacked Z02 wrist images ordered left, right.")
            left, right = views
        images = {
            "base_0_rgb": _rgb_image(data["observation/image"]),
            "left_wrist_0_rgb": _rgb_image(left),
            "right_wrist_0_rgb": _rgb_image(right),
        }
        result = {
            "state": self.state_stats.normalize(state),
            "raw_state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
            "action_loss_mask": ~self.action_stats.constant_mask,
        }
        if "actions" in data:
            actions = prepare_z02_training_actions(raw_state, data["actions"])
            result["actions"] = self.action_stats.normalize(actions)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Z02Outputs:
    """Recover absolute arm, gripper, and head targets for the 18-D environment."""

    action_stats: Z02Quantiles

    def __call__(self, data: dict) -> dict:
        raw_state = data["raw_state"]
        actions = self.action_stats.unnormalize(data["actions"])
        actions[..., _Z02_DELTA_INDICES] += raw_state[..., _Z02_DELTA_INDICES]
        return {"actions": actions[..., Z02_ACTIVE_ACTION_INDICES]}
