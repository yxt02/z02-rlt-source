"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import json
import math
import pathlib

import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None and not data_config.repo_ids:
        raise ValueError("Data config must have a repo_id or repo_ids")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    shuffle = max_frames is not None and max_frames < len(dataset)
    if shuffle:
        indices = np.random.default_rng(0).choice(len(dataset), size=max_frames, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    num_batches = math.ceil(len(dataset) / batch_size)
    # Statistics must include the final partial batch (training loaders drop it).
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        collate_fn=_data_loader._collate_fn,  # noqa: SLF001
        worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def resolve_config(config_name, *, repo_ids=(), assets_dir=None, asset_id=None, train_args_json=None):
    config = _config.get_config(config_name)
    if train_args_json is not None:
        argv = json.loads(train_args_json)
        if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
            raise ValueError("train_args_json must be a JSON list of CLI argument strings")
        config = tyro.extras.overridable_config_cli({config_name: (config_name, config)}, args=[config_name, *argv])
    changes = {}
    if repo_ids:
        changes["repo_ids"] = tuple(repo_ids)
    if assets_dir is not None or asset_id is not None:
        assets = config.data.assets
        changes["assets"] = dataclasses.replace(
            assets,
            assets_dir=assets.assets_dir if assets_dir is None else assets_dir,
            asset_id=assets.asset_id if asset_id is None else asset_id,
        )
    return dataclasses.replace(config, data=dataclasses.replace(config.data, **changes)) if changes else config


def stats_source(config):
    data = config.data
    roots = list(data.repo_ids) if data.repo_ids else [data.repo_id]
    return {
        "repo_ids": [str(pathlib.Path(path).resolve()) if pathlib.Path(path).exists() else path for path in roots],
        "source_files": [
            {"path": str(file.resolve()), "size": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns}
            for path in roots
            if pathlib.Path(path).is_dir()
            for file in sorted([*pathlib.Path(path).glob("data/**/*.parquet"), pathlib.Path(path) / "meta/info.json"])
            if file.is_file()
        ],
        "joint_dim": getattr(data, "joint_dim", None),
        "dual_mode": getattr(data, "dual_mode", None),
        "use_delta_actions": getattr(data, "use_delta_actions", None),
        "delta_action_mask": list(data.delta_action_mask)
        if getattr(data, "delta_action_mask", None) is not None
        else None,
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
    }


def main(
    config_name: str,
    max_frames: int | None = None,
    constant_atol: float | None = 0.05,
    *,
    repo_ids: tuple[str, ...] = (),
    assets_dir: str | None = None,
    asset_id: str | None = None,
    train_args_json: str | None = None,
    check_only: bool = False,
):
    """Compute stats and mask dimensions whose observed peak-to-peak range <= constant_atol.

    The tolerance uses each feature's physical units, after data transforms (including
    delta actions). None disables masking. Prefer a full scan when identifying constants.
    """
    if constant_atol is not None and (not math.isfinite(constant_atol) or constant_atol < 0):
        raise ValueError("constant_atol must be finite and non-negative, or None")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    config = resolve_config(
        config_name, repo_ids=repo_ids, assets_dir=assets_dir, asset_id=asset_id, train_args_json=train_args_json
    )
    source = stats_source(config)
    stats_dir = pathlib.Path(config.data.assets.assets_dir or config.assets_dirs)
    stats_id = config.data.assets.asset_id or config.data.repo_id
    output_path = stats_dir / stats_id
    if check_only:
        normalize.load(output_path)
        manifest_path = output_path / "norm_stats_source.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["source"] != source:
            raise ValueError("Normalization data/geometry differs from training; recompute stats for this launcher")
        print(f"Normalization path checked: {output_path}")
        return
    if hasattr(config.data, "is_compute_norm"):
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, is_compute_norm=True))
    data_config = config.data.create(config.assets_dirs, config.model)
    # Stats only read state/actions; do not decode camera videos while iterating.
    data_config = dataclasses.replace(data_config, load_videos=False)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics(constant_atol=constant_atol) for key, stats in stats.items()}
    for key, value in norm_stats.items():
        if value.constant_mask is not None:
            print(f"{key}: constant dimensions {np.flatnonzero(value.constant_mask).tolist()} (atol={constant_atol})")
    if max_frames is not None and constant_atol is not None:
        print("Constant masks describe the sampled frames; omit --max-frames for a full dataset scan.")

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    (output_path / "norm_stats_source.json").write_text(
        json.dumps(
            {"config_name": config_name, "source": source, "max_frames": max_frames, "constant_atol": constant_atol},
            indent=2,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
