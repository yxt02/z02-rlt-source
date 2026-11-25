import dataclasses
import json
import os
import pathlib
import shlex
import shutil
import subprocess

import numpy as np
import pytest
import safetensors.torch
import torch

from openpi.training import config as configs
from openpi.training import data_loader

from . import compute_norm_stats
from . import policy_run_manifest
from . import train_pytorch


@pytest.fixture(scope="module")
def labeled_dataset(tmp_path_factory):
    python = os.environ.get("RECAP_VALUE_PYTHON")
    if not python or not pathlib.Path(python).exists():
        pytest.skip("Set RECAP_VALUE_PYTHON to run cross-environment integration tests")
    root = pathlib.Path(__file__).resolve().parents[1]
    work = tmp_path_factory.mktemp("recap_dataset")
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + '/tests')
from test_value_chain import run_chain
run_chain(Path(sys.argv[2]))
"""
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [python, "-c", code, str(root / "workflows/recap"), str(work)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return work / "labeled/tiny"


def pipeline_config(tmp_path, dataset, recap):
    base = configs.get_config("pi05_zoyi_z02_pytorch")
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, action_horizon=2, cfg_guidance=recap, pytorch_compile_mode=None),
        data=dataclasses.replace(
            base.data,
            repo_ids=(str(dataset),),
            assets=configs.AssetsConfig(assets_dir=str(tmp_path / "assets"), asset_id="fixture"),
        ),
        batch_size=2,
        num_workers=0,
        num_train_steps=2,
        save_interval=1,
        wandb_enabled=False,
        pytorch_training_precision="float32",
        exp_name="integration",
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        pytorch_weight_path=str(tmp_path / "base"),
    )


def pyav_backend(monkeypatch):
    original = data_loader._make_lerobot_dataset  # noqa: SLF001

    def create(*args, **kwargs):
        dataset = original(*args, **kwargs)
        dataset.video_backend = "pyav"  # This host's optional TorchCodec/FFmpeg runtime is unavailable.
        return dataset

    monkeypatch.setattr(data_loader, "_make_lerobot_dataset", create)


def compute_for_config(monkeypatch, config):
    monkeypatch.setattr(configs, "get_config", lambda _name: config)
    compute_norm_stats.main(config.name)
    compute_norm_stats.main(config.name, check_only=True)


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.parametrize("recap", [False, True])
def test_complete_dataset_stats_and_real_video_batch(monkeypatch, tmp_path, labeled_dataset, framework, recap):
    config = pipeline_config(tmp_path, labeled_dataset, recap)
    pyav_backend(monkeypatch)
    compute_for_config(monkeypatch, config)
    loader = data_loader.create_data_loader(config, num_batches=1, framework=framework)
    batch = next(iter(loader))
    assert len(batch) == (3 if recap else 2)
    observation, actions = batch[:2]
    assert actions.shape == (2, 2, 32)
    assert np.isfinite(np.asarray(actions)).all()
    assert observation.state.shape == (2, 32)
    mask = np.asarray(observation.action_loss_mask)
    assert not mask[:, 16:24].any()
    assert mask[:, 24].all()
    np.testing.assert_array_equal(np.asarray(actions)[..., 16:24], 0)
    if recap:
        assert np.asarray(batch[2]).dtype == np.bool_
    source = json.loads((tmp_path / "assets/fixture/norm_stats_source.json").read_text())["source"]
    assert source["repo_ids"] == [str(labeled_dataset)]
    assert source["joint_dim"] == 25
    mismatched = dataclasses.replace(config, model=dataclasses.replace(config.model, action_horizon=3))
    monkeypatch.setattr(configs, "get_config", lambda _name: mismatched)
    with pytest.raises(ValueError, match="differs from training"):
        compute_norm_stats.main(config.name, check_only=True)


@pytest.mark.parametrize("recap", [False, True])
def test_exported_data_pytorch_updates_and_checkpoint(monkeypatch, tmp_path, labeled_dataset, recap):
    from openpi.models.cfg_test import _TorchTrainingSampler
    from openpi.shared import normalize

    config = pipeline_config(tmp_path, labeled_dataset, recap)
    pyav_backend(monkeypatch)
    compute_for_config(monkeypatch, config)
    gradients = []

    class TrainingHead(_TorchTrainingSampler):
        def __init__(self, model_config):
            super().__init__(model_config)
            self.action_out_proj.velocity = torch.nn.Parameter(torch.linspace(0, 1, 32))
            self.action_out_proj.velocity.register_hook(lambda grad: gradients.append(grad.detach().clone()))

        def gradient_checkpointing_enable(self):
            pass

        def embed_suffix(self, state, actions, time):
            embeddings, pads, masks, condition = super().embed_suffix(state, actions, time)
            return embeddings, pads, masks.expand(actions.shape[0], -1), condition

    initial = TrainingHead(config.model)
    pathlib.Path(config.pytorch_weight_path).mkdir()
    safetensors.torch.save_model(initial, pathlib.Path(config.pytorch_weight_path) / "model.safetensors")
    monkeypatch.setattr(train_pytorch.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", TrainingHead)
    monkeypatch.setattr(train_pytorch, "init_wandb", lambda *_args, **_kwargs: None)
    # Use the actual LeRobot/transform/token/label pipeline in the real trainer.
    train_pytorch.train_loop(config)
    assert len(gradients) == 2
    for gradient in gradients:
        np.testing.assert_array_equal(gradient[16:24].numpy(), 0)
    checkpoint = config.checkpoint_dir / "2"
    restored = normalize.load(checkpoint / "assets/fixture")
    assert restored["actions"].constant_mask[16:24].all()
    weights = safetensors.torch.load_file(checkpoint / "model.safetensors")
    assert torch.isfinite(weights["action_out_proj.velocity"]).all()


def test_norm_overrides_match_training_config(tmp_path):
    argv = [
        "--exp-name=example",
        "--data.repo-ids",
        "/data/one",
        "/data/two",
        "--data.assets.assets-dir",
        str(tmp_path),
        "--data.assets.asset-id",
        "new_stats",
        "--recap",
    ]
    actual = compute_norm_stats.resolve_config("pi05_zoyi_z02_pytorch", train_args_json=json.dumps(argv))
    assert tuple(actual.data.repo_ids) == ("/data/one", "/data/two")
    assert actual.data.assets.assets_dir == str(tmp_path)
    assert actual.data.assets.asset_id == "new_stats"
    assert actual.model.cfg_guidance
    assert configs.get_config(actual.name).data.repo_ids != actual.data.repo_ids


@pytest.mark.parametrize("stage", ["norm", "policy"])
def test_policy_stage_ledger_reuse_and_changed_inputs(monkeypatch, tmp_path, stage):
    config = pipeline_config(tmp_path, tmp_path / "dataset", recap=False)
    monkeypatch.setattr(policy_run_manifest, "resolve_config", lambda *_args, **_kwargs: config)
    path = tmp_path / "run.json"
    assert not policy_run_manifest.record(config.name, "[]", path, stage, "reuse")
    assert not path.exists()
    policy_run_manifest.record(config.name, "[]", path, stage, "running")
    policy_run_manifest.record(config.name, "[]", path, stage, "failed")
    assert not policy_run_manifest.record(config.name, "[]", path, stage, "reuse")
    output = tmp_path / "assets/fixture" if stage == "norm" else config.checkpoint_dir
    output.mkdir(parents=True)
    artifact = output / "result"
    artifact.write_text("completed")
    policy_run_manifest.record(config.name, "[]", path, stage, "running")
    policy_run_manifest.record(config.name, "[]", path, stage, "complete")
    assert policy_run_manifest.record(config.name, "[]", path, stage, "reuse")
    changed = dataclasses.replace(config, model=dataclasses.replace(config.model, action_horizon=3))
    monkeypatch.setattr(policy_run_manifest, "resolve_config", lambda *_args, **_kwargs: changed)
    with pytest.raises(ValueError, match="changed"):
        policy_run_manifest.record(config.name, "[]", path, stage, "reuse")
    monkeypatch.setattr(policy_run_manifest, "resolve_config", lambda *_args, **_kwargs: config)
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        policy_run_manifest.record(config.name, "[]", path, stage, "reuse")


def test_unified_dry_run_is_read_only_and_uses_two_environments(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        RECAP_WORK_DIR=str(tmp_path / "work"),
        VALUE_PYTHON="/value/bin/python",
        POLICY_PYTHON="/policy/bin/python",
        POLICY_DATA_DIR="/labeled/dataset",
    )
    result = subprocess.run(
        ["bash", str(root / "train_recap_z02.sh")],
        cwd=root,
        env=dict(env, RECAP_DRY_RUN="true", POLICY_BACKEND="jax", RECAP_STAGE="all"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert not (tmp_path / "work").exists()
    assert "/value/bin/python" in result.stdout
    assert "/policy/bin/python" in result.stdout
    assert "/labeled/dataset" in result.stdout
    assert "--recap" in result.stdout
    assert "--split all" in result.stdout
    assert "--write-training-dataset" in result.stdout
    export = next(shlex.split(line) for line in result.stdout.splitlines() if "calculate_advantage.py" in line)
    assert export[export.index("--horizon") + 1] == "50"
    assert export[export.index("--label-rule") + 1] == "percentile"
    assert export[export.index("--percentile") + 1] == "40.0"
    assert export[export.index("--reference-split") + 1] == "train"
    assert "--force-intervention-positive" in export


@pytest.mark.parametrize("launcher", ["train_recap_z02.sh", "train_recap_value_z02.sh", "train_recap_policy_z02.sh"])
def test_shared_relative_paths_from_another_directory(tmp_path, launcher):
    source = pathlib.Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    for name in (
        "train_recap_z02.sh",
        "train_recap_value_z02.sh",
        "train_recap_policy_z02.sh",
        "scripts/recap_workflow.py",
        "scripts/train_z02_common.sh",
        "workflows/recap/runner.py",
    ):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    # Edit only the unified launcher's path block, as the user would.
    settings = repo / "train_recap_z02.sh"
    text = settings.read_text().replace(
        "POLICY_DATA_DIRS=()",
        'POLICY_DATA_DIRS=("outputs/recap_z02/labeled/demo dataset" "outputs/recap_z02/labeled/second")',
    )
    for original, replacement in (
        ("${RECAP_HORIZON:-50}", "${RECAP_HORIZON:-100}"),
        ("${RECAP_LABEL_RULE:-percentile}", "${RECAP_LABEL_RULE:-fixed}"),
        ("${RECAP_POSITIVE_PERCENT:-40}", "${RECAP_POSITIVE_PERCENT:-25}"),
        ("${RECAP_THRESHOLD:-0.0}", "${RECAP_THRESHOLD:--0.2}"),
        ("${RECAP_REFERENCE_SPLIT:-train}", "${RECAP_REFERENCE_SPLIT:-val}"),
        ("${RECAP_FORCE_INTERVENTION_POSITIVE:-true}", "${RECAP_FORCE_INTERVENTION_POSITIVE:-false}"),
    ):
        text = text.replace(original, replacement)
    settings.write_text(text)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("POLICY_", "RECAP_", "VALUE_", "SIGLIP_", "GEMMA3_", "TOKENIZER_")):
            env.pop(key)
    env.update(
        RECAP_DRY_RUN="true",
        POLICY_BACKEND="jax",
        VALUE_SOURCE_ROOT="../source data",
        VALUE_PYTHON="value env/bin/python",
        POLICY_PYTHON="policy env/bin/python",
        SIGLIP_PATH="weights/siglip demo",
        POLICY_ASSETS_DIR="assets/demo stats",
        POLICY_WEIGHT_PATH="gs://openpi-assets/checkpoints/pi05_base/params",
        RECAP_TRAINING_DIR="outputs/custom labels",
    )
    result = subprocess.run(
        ["bash", str(repo / launcher)], cwd=tmp_path, env=env, check=True, capture_output=True, text=True
    )
    # Value stages change cwd internally; all paths must still refer to the repository.
    words = [word for line in result.stdout.splitlines() for word in shlex.split(line)]
    if launcher != "train_recap_policy_z02.sh":
        for path in (
            "../source data/2026.09.08",
            "value env/bin/python",
            "weights/siglip demo",
            "outputs/custom labels",
            "workflows/recap/config/z02_data.yaml",
        ):
            assert str((repo / path).resolve()) in words
        export = next(shlex.split(line) for line in result.stdout.splitlines() if "calculate_advantage.py" in line)
        assert export[export.index("--horizon") + 1] == "100"
        assert export[export.index("--label-rule") + 1] == "fixed"
        assert export[export.index("--percentile") + 1] == "25.0"
        assert export[export.index("--threshold") + 1] == "-0.2"
        assert export[export.index("--reference-split") + 1] == "val"
        assert "--no-force-intervention-positive" in export
    if launcher != "train_recap_value_z02.sh":
        for path in (
            "outputs/recap_z02/labeled/demo dataset",
            "outputs/recap_z02/labeled/second",
            "assets/demo stats",
        ):
            assert str(repo / path) in words
        assert str(repo / "policy env/bin/python") in words
        assert "gs://openpi-assets/checkpoints/pi05_base/params" in words
    assert not (repo / "outputs").exists()
