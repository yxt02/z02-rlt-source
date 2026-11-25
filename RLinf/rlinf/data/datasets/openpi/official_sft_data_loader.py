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

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from omegaconf import ListConfig, OmegaConf

from rlinf.data.storage.lerobot import (
    resolve_lerobot_dataset_root,
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    norm_stats_path_from_data_kwargs,
    select_openpi_norm_stats,
)


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig.z02_dataconfig import (
        LeRobotZ02RltDataConfig,
    )

    config = build_official_openpi_sft_config(cfg, world_size, data_paths, eval_dataset)
    if isinstance(config.data, LeRobotZ02RltDataConfig):
        return _build_z02_sft_dataloader(
            config, data_paths, world_size, rank, eval_dataset
        )
    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=not eval_dataset
    )
    return data_loader, data_loader.data_config()


def _collate_openpi_items(items: list[dict]) -> tuple[dict, Any]:
    from torch.utils.data import default_collate

    batch = default_collate(items)
    actions = batch.pop("actions")
    # The vendor Observation does not carry the Z02 action mask or raw state.
    return batch, actions


def _build_z02_sft_dataloader(
    config: Any,
    data_paths: Any,
    world_size: int,
    rank: int,
    eval_dataset: bool,
) -> tuple[Any, Any]:
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.training.data_loader import TransformedDataset
    from torch.utils.data import ConcatDataset, DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    data_config = config.data.create(config.assets_dirs, config.model)
    directories = (
        data_paths if isinstance(data_paths, (list, tuple, ListConfig)) else [data_paths]
    )
    datasets = []
    for directory in directories:
        path = resolve_lerobot_repo_id(directory)
        if path is None:
            raise ValueError("Expected a LeRobot directory in Z02 data.train_data_paths.")
        root = resolve_lerobot_dataset_root(path)
        with (root / "meta" / "info.json").open() as metadata:
            fps = json.load(metadata)["fps"]
        if fps <= 0:
            raise ValueError(f"Expected a positive Z02 dataset fps at {root}, got {fps}.")
        timestamps = {
            key: [step / fps for step in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        }
        datasets.append(
            LeRobotDataset(
                repo_id=f"local/{root.name}",
                root=root,
                delta_timestamps=timestamps,
                download_videos=False,
                video_backend="torchcodec",
            )
        )
    dataset = TransformedDataset(
        ConcatDataset(datasets),
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ],
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=config.seed,
        drop_last=True,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=config.batch_size // world_size,
        sampler=sampler,
        num_workers=config.num_workers,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        collate_fn=_collate_openpi_items,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    if len(loader) == 0:
        raise ValueError(
            f"Z02 dataset has {len(dataset)} frames, fewer than the global microbatch "
            f"size {config.batch_size}. Reduce actor.micro_batch_size or add frames."
        )
    return loader, data_config


def build_official_openpi_sft_config(
    cfg: Any,
    world_size: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> Any:
    """Resolve the shared model, normalization and dataset configuration."""
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    if not norm_stats_path_from_data_kwargs(data_kwargs):
        select_openpi_norm_stats(None, norm_stats_path=None)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=data_kwargs,
    )
    config = dataclasses.replace(
        config,
        num_workers=int(
            OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
        ),
        seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
    )
    _validate_openpi_model_shape(model_cfg, config)

    return config


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_openpi_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config.

    The official loader sizes the SFT action window from
    ``TrainConfig.model.action_horizon``, not ``num_action_chunks`` (that field
    is the env-executed chunk). ``get_model`` already builds the network from
    the same TrainConfig unless YAML overrides ``openpi.action_horizon``.
    """
    yaml_horizon = OmegaConf.select(model_cfg, "openpi.action_horizon", default=None)
    official_horizon = int(openpi_config.model.action_horizon)
    if yaml_horizon is not None and int(yaml_horizon) != official_horizon:
        raise ValueError(
            "openpi SFT data uses TrainConfig.model.action_horizon="
            f"{official_horizon} from {model_cfg.openpi.config_name}; "
            f"openpi.action_horizon={int(yaml_horizon)} would build a different "
            "network. Unset it, or change the TrainConfig."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
