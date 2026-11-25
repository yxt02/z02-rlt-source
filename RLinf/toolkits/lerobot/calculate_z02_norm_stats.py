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

"""Compute Z02 RLT normalization statistics from local LeRobot episodes."""

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared.normalize import RunningStats

from rlinf.models.embodiment.openpi.policies.z02_policy import (
    Z02_ACTIVE_ACTION_INDICES,
    Z02_MODEL_DIM,
    prepare_z02_state,
    prepare_z02_training_actions,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

ACTION_HORIZON = 50
STATE_COLUMN = "observation.joint_positions"
ACTION_COLUMN = "action.joint_positions"


@dataclass
class FeatureStats:
    running: RunningStats = field(default_factory=RunningStats)
    minimum: np.ndarray = field(default_factory=lambda: np.full(Z02_MODEL_DIM, np.inf))
    maximum: np.ndarray = field(default_factory=lambda: np.full(Z02_MODEL_DIM, -np.inf))

    def update(self, values: np.ndarray) -> None:
        vectors = np.asarray(values, dtype=np.float64).reshape(-1, Z02_MODEL_DIM)
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Z02 normalization samples must be finite.")
        self.running.update(vectors)
        self.minimum = np.minimum(self.minimum, vectors.min(axis=0))
        self.maximum = np.maximum(self.maximum, vectors.max(axis=0))

    def serialize(self, constant_atol: float, fixed: np.ndarray) -> dict:
        stats = self.running.get_statistics()
        constant = (self.maximum - self.minimum <= constant_atol) | fixed
        fallback = (
            (stats.q99 - stats.q01 <= 1e-6)
            & (self.maximum > self.minimum)
            & ~constant
        )
        return {
            "mean": stats.mean.tolist(),
            "std": stats.std.tolist(),
            "q01": stats.q01.tolist(),
            "q99": stats.q99.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "constant_mask": constant.tolist(),
            "constant_atol": constant_atol,
            "quantile_fallback_mask": fallback.tolist(),
        }


def _dataset_files(root: Path) -> tuple[Path, list[Path]]:
    info_file = root / "meta" / "info.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"Expected a LeRobot dataset at {root}: missing {info_file}.")
    info = json.loads(info_file.read_text())
    for key in (STATE_COLUMN, ACTION_COLUMN):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [25]:
            raise ValueError(f"Expected {key} to have shape [25] at {root}; found {shape}.")
    episodes = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not episodes or len(episodes) != info.get("total_episodes"):
        raise ValueError(
            f"Expected {info.get('total_episodes')} episode files at {root}; "
            f"found {len(episodes)}."
        )
    return info_file, episodes


def _record(root: Path, path: Path) -> dict:
    metadata = path.stat()
    return {
        "file": str(path.relative_to(root)),
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def _update_episode(path: Path, state_stats: FeatureStats, action_stats: FeatureStats) -> None:
    table = pq.read_table(path, columns=[STATE_COLUMN, ACTION_COLUMN, "frame_index"])
    frame_index = table["frame_index"].to_numpy()
    if frame_index.size == 0 or not np.array_equal(frame_index, np.arange(frame_index.size)):
        raise ValueError(f"Expected contiguous frame_index values from zero in {path}.")
    states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
    offsets = np.arange(ACTION_HORIZON)
    for start in range(0, len(states), 512):
        indices = np.arange(start, min(start + 512, len(states)))
        # LeRobot repeats the last frame when the action window crosses an episode boundary.
        window = np.minimum(indices[:, None] + offsets, len(states) - 1)
        state_stats.update(prepare_z02_state(states[indices]))
        action_stats.update(
            prepare_z02_training_actions(states[indices], actions[window])
        )


def main() -> None:
    """Calculate statistics or check that they still match the selected datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constant-atol", type=float, default=0.05)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.constant_atol) or args.constant_atol < 0:
        parser.error("--constant-atol must be finite and nonnegative")
    roots = [path.expanduser().resolve() for path in args.dataset]
    if len(set(roots)) != len(roots):
        parser.error("each --dataset path must be unique")

    files = [_dataset_files(root) for root in roots]
    source = {
        "version": 1,
        "action_horizon": ACTION_HORIZON,
        "constant_atol": args.constant_atol,
        "active_actions": list(Z02_ACTIVE_ACTION_INDICES),
        "datasets": [
            {
                "root": str(root),
                "files": [_record(root, path) for path in (info, *episodes)],
            }
            for root, (info, episodes) in zip(roots, files, strict=True)
        ],
    }
    if args.check_only:
        if not args.output.is_file():
            raise FileNotFoundError(f"Missing Z02 statistics at {args.output}; run compute_norm_stats.sh.")
        if json.loads(args.output.read_text()).get("z02_rlt_source") != source:
            raise ValueError(f"Z02 statistics at {args.output} do not match the dataset settings; recompute them.")
        logger.info("Z02 statistics match the selected datasets at %s", args.output)
        return

    state_stats, action_stats = FeatureStats(), FeatureStats()
    for root, (_, episodes) in zip(roots, files, strict=True):
        logger.info("Reading %d Z02 episodes from %s", len(episodes), root)
        for episode in episodes:
            _update_episode(episode, state_stats, action_stats)

    state_fixed = np.arange(Z02_MODEL_DIM) >= 22
    action_fixed = np.ones(Z02_MODEL_DIM, dtype=bool)
    action_fixed[list(Z02_ACTIVE_ACTION_INDICES)] = False
    if np.all(action_stats.maximum - action_stats.minimum <= args.constant_atol):
        raise ValueError("All Z02 action dimensions are constant; check the data and constant tolerance.")
    payload = {
        "norm_stats": {
            "state": state_stats.serialize(args.constant_atol, state_fixed),
            "actions": action_stats.serialize(args.constant_atol, action_fixed),
        },
        "z02_rlt_source": source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.output.parent, delete=False
    ) as out:
        temporary_path = Path(out.name)
        try:
            json.dump(payload, out, indent=2, allow_nan=False)
            out.write("\n")
            out.close()
            temporary_path.replace(args.output)
        finally:
            temporary_path.unlink(missing_ok=True)
    logger.info("Wrote Z02 normalization statistics to %s", args.output)


if __name__ == "__main__":
    main()
