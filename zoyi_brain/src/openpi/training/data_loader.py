from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

try:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class AdvantagePreservingDataset(Dataset):
    """Restores the per-frame advantage label after transforms (RECAP / CFG training).

    Repack/data transforms drop unknown keys (e.g. `advantage_positive`), mirroring
    RLinf's `AdvantagePreservingDataset`: labels are collected from the *pre-transform*
    dataset at construction time and re-attached by row index after transforms.

    Only a `dict[int, bool]` is retained, so the wrapper stays picklable for
    multiprocessing DataLoader workers.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transformed_dataset: Dataset,
        *,
        column: str = "advantage_positive",
    ):
        self._dataset = transformed_dataset
        self._labels = self._collect_advantage(base_dataset, column=column)
        if len(self._labels) != len(transformed_dataset):
            raise ValueError(
                f"Advantage label count ({len(self._labels)}) does not match dataset length "
                f"({len(transformed_dataset)}). Labels would be misaligned with frames."
            )

    @classmethod
    def _collect_advantage(
        cls, dataset: Dataset, *, column: str, offset: int = 0, labels: dict[int, bool] | None = None
    ) -> dict[int, bool]:
        if labels is None:
            labels = {}

        if isinstance(dataset, torch.utils.data.ConcatDataset):
            for sub_dataset in dataset.datasets:
                cls._collect_advantage(sub_dataset, column=column, offset=offset, labels=labels)
                offset += len(sub_dataset)
            return labels
        if isinstance(dataset, TransformedDataset):
            # Unwrap prompt_from_task-style wrappers; they preserve row order.
            return cls._collect_advantage(dataset._dataset, column=column, offset=offset, labels=labels)  # noqa: SLF001

        hf_dataset = getattr(dataset, "hf_dataset", None)
        if hf_dataset is None:
            raise ValueError(
                f"Cannot read advantage column {column!r} from dataset of type {type(dataset).__name__}: "
                "no `hf_dataset` attribute. Only LeRobot datasets are supported for CFG training."
            )
        column_names = list(hf_dataset.column_names)
        if column not in column_names:
            hint_columns = [name for name in column_names if name.startswith("advantage_positive")]
            if hint_columns:
                raise ValueError(
                    f"Found multiple advantage label columns {hint_columns} but none named {column!r}. "
                    "Per the RECAP spec you must pick exactly one look-ahead horizon N; rename/copy the "
                    "chosen `advantage_positive_<N>` column to `advantage_positive` (or pass a custom "
                    "column name) before training."
                )
            raise ValueError(
                f"Column {column!r} not found in dataset columns {column_names}. "
                "Run `calculate_advantage.py` on the dataset first and make sure the column is "
                "registered in `meta/info.json` features."
            )
        values = hf_dataset[column]
        if len(values) != len(dataset):
            raise ValueError(f"Advantage column {column!r} has {len(values)} rows but dataset has {len(dataset)} rows.")
        for i, raw_value in enumerate(values):
            if hasattr(raw_value, "item"):  # numpy scalar
                value = raw_value.item()
            elif hasattr(raw_value, "as_py"):  # pyarrow scalar
                value = raw_value.as_py()
            else:
                value = raw_value
            if value is None:
                raise ValueError(f"Advantage column {column!r} contains a null value at row {offset + i}.")
            labels[offset + i] = bool(value)
        return labels

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._dataset[index]
        sample["advantage"] = self._labels[index.__index__()]
        return sample

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()
        # CFG/RECAP models need a per-sample advantage label; alternate labels give q=0.5.
        self._include_advantage = bool(getattr(model_config, "cfg_guidance", False))

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        sample = {
            **observation.to_dict(),
            "actions": action,
        }
        if self._include_advantage:
            sample["advantage"] = bool(index.__index__() % 2)
        return sample

    def __len__(self) -> int:
        return self._num_samples


def _lerobot_repo_ids(data_config: _config.DataConfig) -> list[str]:
    repo_ids = [str(repo_id) for repo_id in (data_config.repo_ids or []) if repo_id]
    if repo_ids:
        return repo_ids
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    return [data_config.repo_id]


class _NoVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobotDataset variant that skips video decoding in ``__getitem__``.

    ``LeRobotDataset`` decodes every camera before transforms can drop the images. Norm-stat
    computation only reads ``state``/``actions``, so decoding videos there is pure overhead
    (and can dominate runtime for long-GOP datasets). This mirrors the upstream ``__getitem__``
    but omits the ``_query_videos`` block; non-video image columns are still transformed.
    """

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Intentionally skip self._query_videos(...).
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        return item

    def get_episodes_file_paths(self) -> list[pathlib.Path]:
        """Only require parquet files at init; stats must work even without video files."""
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        return [self.meta.get_data_file_path(ep_idx) for ep_idx in episodes]

    def download_episodes(self, download_videos: bool = True) -> None:  # noqa: FBT001, FBT002
        # Ignore the caller's flag: this dataset class never consumes videos.
        super().download_episodes(download_videos=False)


def _make_lerobot_dataset(
    repo_id: str, action_horizon: int, action_sequence_keys: Sequence[str], *, load_videos: bool = True
) -> Dataset:
    dataset_kwargs = {}
    repo_path = pathlib.Path(repo_id)
    if repo_path.exists():
        dataset_kwargs["root"] = repo_path

    try:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, **dataset_kwargs)
    except TypeError:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    fps = dataset_meta.fps
    dataset_init_kwargs = {
        "delta_timestamps": {key: [t / fps for t in range(action_horizon)] for key in action_sequence_keys},
        **dataset_kwargs,
    }
    dataset_cls = lerobot_dataset.LeRobotDataset if load_videos else _NoVideoLeRobotDataset
    try:
        return dataset_cls(repo_id, **dataset_init_kwargs)
    except TypeError:
        dataset_init_kwargs.pop("root", None)
        return dataset_cls(repo_id, **dataset_init_kwargs)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    if data_config.repo_id == "fake" and not data_config.repo_ids:
        return FakeDataset(model_config, num_samples=1024)

    repo_ids = _lerobot_repo_ids(data_config)
    datasets = [
        _make_lerobot_dataset(
            repo_id, action_horizon, data_config.action_sequence_keys, load_videos=data_config.load_videos
        )
        for repo_id in repo_ids
    ]
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    logging.info("Loaded LeRobot datasets %s (length=%s)", repo_ids, len(dataset))

    if data_config.prompt_from_task:
        if len(repo_ids) != 1:
            raise ValueError("prompt_from_task is only supported for a single LeRobot dataset.")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0])
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, jax.Array]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, jax.Array]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    base_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)
    if data_config.read_advantage and not isinstance(base_dataset, FakeDataset):
        # FakeDataset emits its own advantage label; real datasets need the label
        # restored after transforms (which drop the parquet column).
        dataset = AdvantagePreservingDataset(base_dataset, dataset)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if self._data_config.read_advantage:
                # CFG/RECAP: 3-tuple batch (MD "三元组 batch"); advantage rides outside the Observation.
                yield observation, batch["actions"], batch["advantage"]
            else:
                yield observation, batch["actions"]
