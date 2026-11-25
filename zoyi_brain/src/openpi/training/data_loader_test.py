import dataclasses
import pathlib
import subprocess
import sys

import jax
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


@pytest.mark.parametrize(
    "config_name",
    [
        "pi0_zoyi_z02",
        "pi0_zoyi_z02_pytorch",
        "pi05_zoyi_z02",
        "pi05_zoyi_z02_pytorch",
        "pi05_zoyi_z02_lora",
        "pi05_zoyi_z02_recap",
        "pi05_zoyi_z02_recap_pytorch",
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_recap_cli_switch(config_name, enabled):
    # Exercise each CLI in its own process, as the training entry points are used.
    # Repeated in-process parsing also triggers tyro's pytest-only random cache checks.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from openpi.training import config as config_module
name, flag = sys.argv[1:]
original = config_module.get_config(name)
sys.argv = ['train.py', name, flag, '--exp-name', 'switch_test']
config = config_module.cli()
assert config.model.cfg_guidance is (flag == '--recap')
assert config.model.discrete_state_input == original.model.discrete_state_input
assert config.data == original.data
assert config.lr_schedule == original.lr_schedule
assert config_module.get_config(name) is original
""",
            config_name,
            "--recap" if enabled else "--no-recap",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.parametrize("enabled", [False, True])
def test_recap_switch_changes_batch_contract(framework, enabled):
    original = _config.get_config("debug_recap")
    config = dataclasses.replace(
        original, model=dataclasses.replace(original.model, cfg_guidance=enabled), num_workers=0
    )
    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1, framework=framework)
    batch = next(iter(loader))
    assert loader.data_config().read_advantage is enabled
    assert len(batch) == (3 if enabled else 2)
    observation = batch[0]
    assert (observation.tokenized_positive_guidance_prompt is not None) is enabled
    assert (observation.tokenized_negative_guidance_prompt is not None) is enabled


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("state_input", [False, True])
def test_zoyi_recap_switch_preserves_state_transform(monkeypatch, tmp_path, enabled, state_input):
    # Avoid loading a tokenizer model: this test checks the factory's routing and settings.
    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", lambda max_len: None)  # noqa: SLF001
    original = _config.get_config("pi05_zoyi_z02")
    model = dataclasses.replace(original.model, cfg_guidance=enabled, discrete_state_input=state_input)
    data = original.data.create(tmp_path, model)
    tokenize_cls = _transforms.TokenizePromptWithGuidance if enabled else _transforms.TokenizePrompt
    tokenize = next(transform for transform in data.model_transforms.inputs if isinstance(transform, tokenize_cls))
    assert data.read_advantage is enabled
    assert tokenize.discrete_state_input is state_input


class _StubHfDataset:
    """Minimal stand-in for a LeRobot `hf_dataset`: column store with `column_names`."""

    def __init__(self, columns: dict[str, list]):
        self.column_names = list(columns)
        self._columns = columns

    def __getitem__(self, key: str) -> list:
        return self._columns[key]


class _StubLerobotDataset:
    """Minimal stand-in for a LeRobot dataset: row views over a column store."""

    def __init__(self, num_rows: int, columns: dict[str, list]):
        self._num_rows = num_rows
        self.hf_dataset = _StubHfDataset(columns)

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int) -> dict:
        # Raw sample includes the parquet columns, exactly like pre-transform LeRobot data.
        return {name: self.hf_dataset[name][index] for name in self.hf_dataset.column_names}


def _drop_advantage_column():
    # Mimics RepackTransform stripping unknown keys after transforms.
    return _transforms.RepackTransform(structure={"state": "state"})


def test_create_torch_dataset_propagates_load_videos(monkeypatch):
    seen = []

    def _fake_make_lerobot_dataset(repo_id, action_horizon, action_sequence_keys, *, load_videos=True):
        seen.append((repo_id, load_videos))
        return [0]

    monkeypatch.setattr(_data_loader, "_make_lerobot_dataset", _fake_make_lerobot_dataset)
    data_config = _config.DataConfig(repo_id="toy", load_videos=False)
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=1, model_config=None)

    assert dataset == [0]
    assert seen == [("toy", False)]


def test_no_video_dataset_skips_video_decode():
    class _RowHfDataset:
        def __getitem__(self, index):
            return {
                "state": torch.tensor([1.0]),
                "episode_index": torch.tensor(0),
                "task_index": torch.tensor(0),
            }

    class _Meta:
        tasks = ("task",)
        camera_keys = ("observation.images.head",)
        total_episodes = 1

        def get_data_file_path(self, episode_index):
            return pathlib.Path(f"data/{episode_index}.parquet")

    dataset = object.__new__(_data_loader._NoVideoLeRobotDataset)  # noqa: SLF001
    dataset.hf_dataset = _RowHfDataset()
    dataset.delta_indices = None
    dataset.episodes = None
    dataset.meta = _Meta()
    dataset.image_transforms = None
    dataset._query_videos = lambda *args, **kwargs: pytest.fail("video should not be decoded")  # noqa: SLF001

    item = dataset[0]

    assert item["task"] == "task"
    torch.testing.assert_close(item["state"], torch.tensor([1.0]))
    assert dataset.get_episodes_file_paths() == [pathlib.Path("data/0.parquet")]


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_advantage_preserving_dataset():
    labels = [True, False, True, False, False, True]
    base = _StubLerobotDataset(6, {"state": [np.zeros(2, np.float32)] * 6, "advantage_positive": labels})
    # Transforms drop the label column; the wrapper restores it by row index.
    transformed = _data_loader.TransformedDataset(base, [_drop_advantage_column()])
    dataset = _data_loader.AdvantagePreservingDataset(base, transformed)

    assert len(dataset) == 6
    for i in range(6):
        sample = dataset[i]
        assert sample["advantage"] == labels[i]
        assert "advantage_positive" not in sample  # stripped by the transform
        assert "state" in sample


def test_advantage_preserving_dataset_missing_column():
    base = _StubLerobotDataset(4, {"state": [np.zeros(2, np.float32)] * 4})
    transformed = _data_loader.TransformedDataset(base, [_drop_advantage_column()])

    with pytest.raises(ValueError, match="calculate_advantage"):
        _data_loader.AdvantagePreservingDataset(base, transformed)


def test_advantage_preserving_dataset_multiple_horizons():
    columns = {
        "state": [np.zeros(2, np.float32)] * 4,
        "advantage_positive_10": [True, False, True, False],
        "advantage_positive_20": [False, True, False, True],
    }
    base = _StubLerobotDataset(4, columns)
    transformed = _data_loader.TransformedDataset(base, [_drop_advantage_column()])

    # Multiple look-ahead horizons require an explicit choice (MD §2).
    with pytest.raises(ValueError, match="advantage_positive_10") as excinfo:
        _data_loader.AdvantagePreservingDataset(base, transformed)
    assert "advantage_positive_20" in str(excinfo.value)


def test_advantage_preserving_dataset_concat_offsets():
    first = _StubLerobotDataset(3, {"state": [np.zeros(2, np.float32)] * 3, "advantage_positive": [True, False, True]})
    second = _StubLerobotDataset(
        3, {"state": [np.zeros(2, np.float32)] * 3, "advantage_positive": [False, False, True]}
    )
    base = torch.utils.data.ConcatDataset([first, second])
    transformed = _data_loader.TransformedDataset(base, [_drop_advantage_column()])
    dataset = _data_loader.AdvantagePreservingDataset(base, transformed)

    assert len(dataset) == 6
    # Second dataset's labels must land after the first dataset's rows.
    assert dataset[0]["advantage"] is True
    assert dataset[3]["advantage"] is False
    assert dataset[5]["advantage"] is True


def test_advantage_preserving_dataset_length_mismatch():
    base = _StubLerobotDataset(3, {"state": [np.zeros(2, np.float32)] * 3, "advantage_positive": [True, False, True]})
    # A transformed dataset of a different length would misalign labels.
    other = _StubLerobotDataset(5, {"state": [np.zeros(2, np.float32)] * 5, "advantage_positive": [False] * 5})
    transformed = _data_loader.TransformedDataset(other, [_drop_advantage_column()])

    with pytest.raises(ValueError, match="misaligned"):
        _data_loader.AdvantagePreservingDataset(base, transformed)


def test_advantage_preserving_dataset_null_value():
    columns = {"state": [np.zeros(2, np.float32)] * 3, "advantage_positive": [True, None, False]}
    base = _StubLerobotDataset(3, columns)
    transformed = _data_loader.TransformedDataset(base, [_drop_advantage_column()])

    with pytest.raises(ValueError, match="null value"):
        _data_loader.AdvantagePreservingDataset(base, transformed)


def test_with_fake_cfg_dataset():
    config = _config.get_config("debug_recap")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for observation, actions, advantage in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
        # 3-tuple batch with a per-sample bool label (MD "三元组 batch").
        assert advantage.shape == (config.batch_size,)
        assert advantage.dtype == bool
        # Guidance prompts ride along in the observation for CFG routing.
        assert observation.tokenized_positive_guidance_prompt is not None
        assert observation.tokenized_negative_guidance_prompt is not None
        assert observation.tokenized_positive_guidance_prompt.shape == (
            config.batch_size,
            config.model.max_token_len,
        )


def test_with_fake_dataset_stays_two_tuple():
    # Non-CFG configs must keep yielding plain 2-tuples.
    config = _config.get_config("debug")
    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1)

    batch = next(iter(loader))
    assert len(batch) == 2


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
