import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import jax
import numpy as np
import pytest
import safetensors.torch
import torch
import tyro

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config
from openpi.training import weight_loaders

from . import serve_policy
from . import train


@pytest.mark.parametrize(("batch_size", "world_size"), [(0, 1), (-1, 1), (1, 2), (3, 2)])
def test_pytorch_invalid_global_batch_fails_before_checkpoint_changes(monkeypatch, tmp_path, batch_size, world_size):
    from . import train_pytorch

    config = dataclasses.replace(
        _config.get_config("pi05_zoyi_z02_pytorch"),
        batch_size=batch_size,
        exp_name="protected",
        checkpoint_base_dir=str(tmp_path),
        overwrite=True,
    )
    config.checkpoint_dir.mkdir(parents=True)
    marker = config.checkpoint_dir / "preserve.txt"
    marker.write_text("existing checkpoint")
    monkeypatch.setattr(train_pytorch, "setup_ddp", lambda: (True, 0, torch.device("cpu")))
    monkeypatch.setattr(train_pytorch.dist, "get_world_size", lambda: world_size)
    with pytest.raises(ValueError, match="positive multiple"):
        train_pytorch.train_loop(config)
    assert marker.read_text() == "existing checkpoint"


@pytest.mark.parametrize("config_name", ["pi05_zoyi_z02_pytorch", "pi05_zoyi_z02_recap_pytorch"])
@pytest.mark.parametrize("override", [False, True])
def test_pytorch_launcher_preserves_config_and_optional_overrides(tmp_path, config_name, override):
    repo = pathlib.Path(__file__).resolve().parents[1]
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    recorder = stub_dir / "torchrun"
    recorder.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "open(os.environ['ZOYI_TEST_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    recorder.chmod(0o755)
    args_file = tmp_path / "argv.json"
    env = dict(
        os.environ,
        PATH=f"{stub_dir}:{os.environ['PATH']}",
        NPROC="2",
        ZOYI_TEST_ARGV=str(args_file),
        CHECK_NORM_SOURCE="false",
        RECORD_POLICY_RUN="false",
    )
    # The ordinary preset is also the default when only flags are provided.
    # Isolate the launcher from this checkout's .venv activation before using the recorder.
    launcher = tmp_path / "train_pi05_z02_pytorch.sh"
    launcher.write_text((repo / "train_pi05_z02_pytorch.sh").read_text())
    selection = [config_name] if config_name.endswith("recap_pytorch") else []
    flags = ["--exp-name=batch_test"]
    if override:
        flags += ["--batch-size=64", "--pytorch-weight-path=/converted/base", "--no-pytorch-gradient-checkpointing"]
    subprocess.run(
        ["bash", str(launcher), *selection, *flags],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    recorded = json.loads(args_file.read_text())
    assert "--nproc_per_node=2" in recorded
    config_args = recorded[recorded.index("scripts/train_pytorch.py") + 1 :]
    assert config_args[0] == config_name
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from openpi.training import config as configs
original = configs.get_config(sys.argv[1])
sys.argv = ['train_pytorch.py', *sys.argv[1:]]
cfg = configs.cli()
assert cfg.exp_name == 'batch_test'
assert cfg.model == original.model
assert cfg.lr_schedule == original.lr_schedule
assert cfg.data.joint_dim == 25
assert cfg.data.assets.asset_id == 'z02_25d'
for field in ('num_train_steps', 'num_workers', 'pytorch_training_precision',
              'checkpoint_base_dir', 'save_interval', 'wandb_enabled', 'overwrite'):
    assert getattr(cfg, field) == getattr(original, field), field
if '--batch-size=64' in sys.argv:
    assert cfg.batch_size == 64
    assert cfg.pytorch_weight_path == '/converted/base'
    assert not cfg.pytorch_gradient_checkpointing
else:
    assert cfg.batch_size == original.batch_size
    assert cfg.pytorch_weight_path == original.pytorch_weight_path
    assert cfg.pytorch_gradient_checkpointing == original.pytorch_gradient_checkpointing
""",
            *config_args,
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("backend", ["jax", "pytorch"])
@pytest.mark.parametrize("recap", [False, True])
def test_dedicated_policy_launcher_without_arguments_runs_norm_then_training(tmp_path, backend, recap):
    repo = pathlib.Path(__file__).resolve().parents[1]
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    recorder = """#!/usr/bin/env python3
import json, os, pathlib, sys
if sys.argv[1:2] == ['-c']:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
path = pathlib.Path(os.environ['ZOYI_TEST_COMMANDS'])
commands = json.loads(path.read_text()) if path.exists() else []
commands.append(sys.argv[1:])
path.write_text(json.dumps(commands))
if os.environ.get('ZOYI_TEST_FAIL_NORM') == 'true' and sys.argv[1:2] == ['scripts/compute_norm_stats.py']:
    sys.exit(17)
"""
    for name in ("python", "torchrun"):
        path = stub_dir / name
        path.write_text(recorder)
        path.chmod(0o755)
    log = tmp_path / "commands.json"
    env = dict(
        os.environ,
        POLICY_PYTHON=str(stub_dir / "python"),
        TORCHRUN=str(stub_dir / "torchrun"),
        POLICY_BACKEND=backend,
        POLICY_MODEL="pi05",
        POLICY_RECAP=str(recap).lower(),
        POLICY_EXP_NAME="plain_bash",
        POLICY_STAGE="all",
        RECAP_DRY_RUN="false",
        RECAP_REUSE="false",
        RECORD_POLICY_RUN="false",
        CHECK_NORM_SOURCE="true",
        ZOYI_TEST_COMMANDS=str(log),
        NPROC="2",
    )
    command = ["bash", str(repo / "train_recap_policy_z02.sh")]
    subprocess.run(command, cwd=repo, env=env, check=True, capture_output=True, text=True)
    commands = json.loads(log.read_text())
    assert len(commands) == 3
    norm, check, training = commands
    assert norm[0] == check[0] == "scripts/compute_norm_stats.py"
    assert "--check-only" not in norm
    assert "--check-only" in check
    config_name = "pi05_zoyi_z02" + ("_recap" if recap else "") + ("_pytorch" if backend == "pytorch" else "")
    assert norm[norm.index("--config-name") + 1] == config_name
    norm_args = json.loads(norm[norm.index("--train-args-json") + 1])
    assert "--exp-name=plain_bash" in norm_args
    assert ("--recap" if recap else "--no-recap") in norm_args
    train_script = "scripts/train_pytorch.py" if backend == "pytorch" else "scripts/train.py"
    offset = training.index(train_script)
    assert training[offset + 1 :] == [config_name, *norm_args]
    if backend == "pytorch":
        assert "--nproc_per_node=2" in training
    log.unlink()
    failure = subprocess.run(
        command, cwd=repo, env=dict(env, ZOYI_TEST_FAIL_NORM="true"), capture_output=True, text=True, check=False
    )
    assert failure.returncode == 17
    assert len(json.loads(log.read_text())) == 1  # Training must not start after a failed norm scan.


@pytest.mark.parametrize("recap", [False, True])
@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_pytorch_25d_training_and_checkpoint(monkeypatch, tmp_path, recap, gradient_checkpointing):
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models.cfg_test import _TorchTrainingSampler
    from openpi.shared import normalize

    from . import train_pytorch

    original = _config.get_config("pi05_zoyi_z02_pytorch")
    config = dataclasses.replace(
        original,
        model=dataclasses.replace(original.model, cfg_guidance=recap, action_horizon=2, pytorch_compile_mode=None),
        batch_size=64,
        num_train_steps=2,
        save_interval=1,
        num_workers=0,
        wandb_enabled=False,
        pytorch_training_precision="float32",
        pytorch_gradient_checkpointing=gradient_checkpointing,
        exp_name="cpu_flow",
        checkpoint_base_dir=str(tmp_path / "ckpts"),
        pytorch_weight_path=str(tmp_path / "base"),
    )
    samples = np.zeros((1000, 32))
    samples[:, :16] = np.linspace(-1, 1, 1000)[:, None]
    samples[:, 16:22] = 0.3
    samples[-1, 24] = 0.2  # Rare rotation remains trainable and uses full-range scaling.
    running = normalize.RunningStats()
    running.update(samples)
    stats = running.get_statistics(constant_atol=1e-6)
    norm_stats = {"state": stats, "actions": stats}
    normalized = transforms.Normalize(norm_stats, use_quantiles=True)({"state": samples[-1], "actions": samples[-2:]})
    active = ~stats.constant_mask
    batch = config.batch_size
    observation = model_api.Observation(
        images={},
        image_masks={},
        state=torch.zeros(batch, 32),
        tokenized_prompt=torch.ones(batch, 2, dtype=torch.int32),
        tokenized_prompt_mask=torch.ones(batch, 2, dtype=torch.bool),
        tokenized_positive_guidance_prompt=torch.full((batch, 2), 2, dtype=torch.int32) if recap else None,
        tokenized_positive_guidance_prompt_mask=torch.ones(batch, 2, dtype=torch.bool) if recap else None,
        tokenized_negative_guidance_prompt=torch.full((batch, 2), 3, dtype=torch.int32) if recap else None,
        tokenized_negative_guidance_prompt_mask=torch.ones(batch, 2, dtype=torch.bool) if recap else None,
        action_loss_mask=torch.tensor(np.broadcast_to(active, (batch, 32)).copy()),
    )
    actions = torch.tensor(normalized["actions"], dtype=torch.float32)[None].expand(batch, -1, -1)
    batch_item = (observation, actions)
    if recap:
        batch_item += (torch.arange(batch) % 2 == 0,)
    gradients = []
    models = []

    class TrainingHead(_TorchTrainingSampler):
        def __init__(self, model_config):
            super().__init__(model_config)
            self.gradient_checkpointing_enabled = False
            models.append(self)
            self.action_out_proj.velocity = torch.nn.Parameter(torch.linspace(0, 1, 32))
            self.action_out_proj.velocity.register_hook(lambda grad: gradients.append(grad.detach().clone()))

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True

        def embed_suffix(self, state, actions, time):
            embs, pads, masks, condition = super().embed_suffix(state, actions, time)
            return embs, pads, masks.expand(actions.shape[0], -1), condition

    initial = TrainingHead(config.model)
    pathlib.Path(config.pytorch_weight_path).mkdir()
    safetensors.torch.save_model(initial, pathlib.Path(config.pytorch_weight_path) / "model.safetensors")
    data_config = _config.DataConfig(asset_id="z02_25d", norm_stats=norm_stats, read_advantage=recap)
    monkeypatch.setattr(train_pytorch, "build_datasets", lambda _config: ([batch_item], data_config))
    monkeypatch.setattr(train_pytorch, "init_wandb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_pytorch.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", TrainingHead)
    train_pytorch.train_loop(config)
    assert config.model.dtype == original.model.dtype  # Precision overrides do not mutate the config preset.
    assert len(gradients) == 2
    assert models[-1].gradient_checkpointing_enabled == gradient_checkpointing
    for gradient in gradients:
        np.testing.assert_array_equal(gradient.numpy()[~active], 0)
        assert gradient[24] != 0
    checkpoint = config.checkpoint_dir / "2"
    restored = normalize.load(checkpoint / "assets/z02_25d")
    np.testing.assert_array_equal(restored["actions"].constant_mask, stats.constant_mask)
    np.testing.assert_array_equal(restored["actions"].quantile_fallback_mask, stats.quantile_fallback_mask)
    weights = safetensors.torch.load_file(checkpoint / "model.safetensors")
    assert torch.isfinite(weights["action_out_proj.velocity"]).all()
    assert not torch.equal(
        weights["action_out_proj.velocity"][active], initial.action_out_proj.velocity.detach()[active]
    )


@pytest.mark.parametrize("target_dim", [25, 40, 64])
def test_action_dimension_checkpoint_resize(monkeypatch, target_dim):
    source_dim = 32
    loaded = {
        "action_in_proj": {"kernel": np.arange(source_dim * 4, dtype=np.float32).reshape(source_dim, 4)},
        "action_out_proj": {
            "kernel": np.arange(4 * source_dim, dtype=np.float32).reshape(4, source_dim),
            "bias": np.arange(source_dim, dtype=np.float32),
        },
        "state_proj": {"kernel": np.ones((source_dim, 4), np.float32)},
        "backbone": {"kernel": np.ones((4, 4), np.float32)},
    }
    reference = {
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((target_dim, 4), np.float32)},
        "action_out_proj": {
            "kernel": jax.ShapeDtypeStruct((4, target_dim), np.float32),
            "bias": jax.ShapeDtypeStruct((target_dim,), np.float32),
        },
        "state_proj": {"kernel": jax.ShapeDtypeStruct((target_dim, 4), np.float32)},
        "backbone": {"kernel": jax.ShapeDtypeStruct((4, 4), np.float32)},
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: loaded)  # noqa: SLF001
    loader = weight_loaders.CheckpointWeightLoader("fake", allow_action_dim_resize=True)
    result = train._load_weights_and_validate(loader, reference)  # noqa: SLF001
    overlap = min(source_dim, target_dim)
    np.testing.assert_array_equal(
        result["action_in_proj"]["kernel"][:overlap], loaded["action_in_proj"]["kernel"][:overlap]
    )
    np.testing.assert_array_equal(
        result["action_out_proj"]["kernel"][:, :overlap], loaded["action_out_proj"]["kernel"][:, :overlap]
    )
    np.testing.assert_array_equal(
        result["action_out_proj"]["bias"][:overlap], loaded["action_out_proj"]["bias"][:overlap]
    )
    np.testing.assert_array_equal(result["action_in_proj"]["kernel"][source_dim:], 0)
    np.testing.assert_array_equal(result["action_out_proj"]["kernel"][:, source_dim:], 0)
    np.testing.assert_array_equal(result["action_out_proj"]["bias"][source_dim:], 0)
    np.testing.assert_array_equal(result["state_proj"]["kernel"][source_dim:], 0)
    np.testing.assert_array_equal(result["backbone"]["kernel"], loaded["backbone"]["kernel"])
    with pytest.raises(ValueError, match="[Ss]hape"):
        train._load_weights_and_validate(weight_loaders.CheckpointWeightLoader("fake"), reference)  # noqa: SLF001


def test_action_resize_rejects_backbone_width_change():
    loaded = {"action_in_proj": {"kernel": np.ones((32, 4), np.float32)}}
    reference = {"action_in_proj": {"kernel": jax.ShapeDtypeStruct((40, 8), np.float32)}}
    with pytest.raises(ValueError, match="non-action axes"):
        weight_loaders._resize_action_projections(loaded, reference)  # noqa: SLF001


@pytest.mark.parametrize("config_name", ["pi05_zoyi_z02", "pi05_zoyi_z02_recap_pytorch"])
@pytest.mark.parametrize("recap", [None, False, True])
@pytest.mark.parametrize("state_input", [None, False, True])
def test_serve_policy_input_overrides(monkeypatch, config_name, recap, state_input):
    argv = ["policy:checkpoint", "--policy.config", config_name, "--policy.dir", "/fake/checkpoint"]
    if recap is not None:
        argv += ["--policy.recap", str(recap)]
    if state_input is not None:
        argv += ["--policy.discrete-state-input", str(state_input)]
    args = tyro.cli(serve_policy.Args, args=argv)
    original = _config.get_config(config_name)
    captured = {}

    def create_policy(config, checkpoint_dir, **kwargs):
        captured.update(config=config, checkpoint_dir=checkpoint_dir)
        return "policy"

    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", create_policy)  # noqa: SLF001
    assert serve_policy.create_policy(args) == "policy"
    model = captured["config"].model
    assert model.cfg_guidance is (original.model.cfg_guidance if recap is None else recap)
    assert model.discrete_state_input is (original.model.discrete_state_input if state_input is None else state_input)
    assert captured["checkpoint_dir"] == "/fake/checkpoint"
    assert _config.get_config(config_name) is original


def test_serve_policy_rejects_overrides_for_fast(monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Unsupported overrides must be rejected before checkpoint loading")

    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", unexpected_load)  # noqa: SLF001
    args = serve_policy.Args(
        policy=serve_policy.Checkpoint(config="pi0_fast_droid", dir="/fake/checkpoint", recap=True)
    )
    with pytest.raises(ValueError, match="pi0/pi0.5"):
        serve_policy.create_policy(args)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
