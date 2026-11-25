# ZOYI Brain

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone git@github.com:Zoyi-Tech/zoyi_brain.git
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. This project targets **Python 3.10.12** (see `.python-version`). Once uv is installed, run the following to set up the environment:

```bash
uv venv --python 3.10.12
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.10/site-packages/transformers/
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                          | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning                       | `gs://openpi-assets/checkpoints/pi05_base`     |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case                | Description                                                                                                                                                                                                                           | Checkpoint Path                                       |
| ------------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference               | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform                              | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning             | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                                             | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference               | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                                                       | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference               | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                   | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference               | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                                                       | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO       | Inference               | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md))                                                | `gs://openpi-assets/checkpoints/pi05_libero`          |
| $\pi_{0.5}$-DROID        | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`           |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.

## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.

## Training Preparation for Zoyi

Fine-tune $\pi_{0.5}$ on local Zoyi LeRobot datasets with the `pi05_zoyi_z02` config. Before launching training, align the dataset paths, `joint_dim`, and delta-action mask with your robot.

The relevant files are:

- [`ZoyiInputs` / `ZoyiOutputs`](src/openpi/policies/zoyi_policy.py): map LeRobot samples to the model observation / action format.
- [`LeRobotZoyiDataConfig`](src/openpi/training/config.py): camera remapping, `joint_dim`, and the delta-action mask.
- [`TrainConfig` `pi05_zoyi_z02`](src/openpi/training/config.py): dataset paths, optimizer, and checkpoint settings.

### 1. Point the config at your LeRobot datasets

`LeRobotZoyiDataConfig` expects these keys in each dataset:

| Dataset key | Used as |
| ----------- | ------- |
| `observation.images.cam2` | head camera |
| `observation.images.cam4` | left wrist camera |
| `observation.images.cam3` | right wrist camera |
| `observation.joint_positions` | proprioceptive state |
| `action.joint_positions` | action targets |
| `task` | language prompt (LeRobot task string) |

Set `repo_ids` to one or more local LeRobot dataset directories, and keep `repo_id` / `asset_id` as a short name for the norm-stats folder:

```python
TrainConfig(
    name="pi05_zoyi_z02",
    data=LeRobotZoyiDataConfig(
        repo_id="z02",
        repo_ids=(
            "/path/to/lerobot/Z02/task_a/date",
            "/path/to/lerobot/Z02/task_b/date",
        ),
        assets=AssetsConfig(asset_id="z02_25d"),
        use_delta_actions=True,
        dual_mode=True,
        joint_dim=25,
        delta_action_mask=make_bool_mask(7, -1, 7, -10),
    ),
    ...
)
```

Update the camera mapping in `LeRobotZoyiDataConfig` if your LeRobot image keys differ.

### 2. Choose `joint_dim`

`observation.joint_positions` and `action.joint_positions` may be longer than the control space (for example 20-D). `ZoyiInputs` keeps only the first `joint_dim` values and pads the rest to the model `action_dim`; later dimensions are ignored.

Set `LeRobotZoyiDataConfig.joint_dim` explicitly for whole-body control. The JAX `pi05_zoyi_z02` / `pi05_zoyi_z02_recap` and PyTorch `pi05_zoyi_z02_pytorch` / `pi05_zoyi_z02_recap_pytorch` presets keep 25 dimensions, pad to the model's 32 dimensions, and return 25 dimensions during inference. Their `z02_25d` assets are separate from legacy 16-D statistics.

When `joint_dim=None`, the legacy defaults still come from `dual_mode`:

| `dual_mode` | `joint_dim` | Layout used for training |
| ----------- | ----------- | ------------------------ |
| `True` (default) | 16 | left 7 joints + left gripper + right 7 joints + right gripper |
| `False` | 8 | 7 joints + gripper |

The factory passes the configured dimension to both `ZoyiInputs` and `ZoyiOutputs`; editing their implementation is unnecessary. Data shorter than `joint_dim`, or `joint_dim > model.action_dim`, now raises a clear error. Other presets, including LoRA and π0 PyTorch, retain their legacy layouts; set their dimension, mask, data paths, and asset id explicitly before using a 25-D dataset. Recompute norm stats whenever the layout changes.

For more than 32 robot dimensions, also increase `Pi0Config.action_dim` (for example, `joint_dim=40`, `action_dim=64`). JAX can initialize from a 32-D base checkpoint by explicitly setting `CheckpointWeightLoader(..., allow_action_dim_resize=True)`: overlapping action/state projection coordinates are copied, added coordinates start at zero, and all other shapes remain strictly validated. This is an initialization for further training, not a way to make an old robot checkpoint control new joints. Use a new assets id, recompute statistics, and load the trained checkpoint with the same dimensions. With state tokens, increase `max_token_len` as needed (for example 512 for a 64-D model) and check for truncation warnings. PyTorch pretrained weight resizing is not added here.

### 3. Specify the delta-action mask

When `use_delta_actions=True`, joint actions are converted to deltas relative to the current state. Gripper dimensions stay absolute. The mask must cover exactly `joint_dim` and match the joint / gripper layout:

```python
# dual-arm (16-D): delta, abs, delta, abs
make_bool_mask(7, -1, 7, -1)

# single-arm (8-D)
make_bool_mask(7, -1)

# 25-D: left arm delta, left hand absolute, right arm delta,
# right hand and the final 9 body coordinates absolute.
make_bool_mask(7, -1, 7, -10)
```

`make_bool_mask` treats a **positive** integer as that many `True` (delta) dims and a **negative** integer as that many `False` (absolute) dims. Training uses `DeltaActions`; inference uses the same mask with `AbsoluteActions`. Set `use_delta_actions=False` to train in absolute action space.

Set `delta_action_mask` to a boolean tuple of exactly `joint_dim` entries for a custom delta layout. The legacy mask is used only for the legacy 8/16-D layout; custom layouts must provide a mask or disable delta actions.

### 4. Compute norm stats, then train

The original `train_pi*_z02*.sh` launchers keep their pre-workflow behavior: compute statistics separately, then launch training with bash. New workflow settings live in one commented configuration block at the top of `train_recap_z02.sh`; both standalone RECAP launchers read it. Paths may be relative to the repository root, regardless of the launch directory. Edit that block and launch with bash without arguments. See the [offline RECAP workflow guide](workflows/recap/README.md) for complete labeled exports, isolated environments and stage reuse. Policy datasets remain the directories you explicitly select.

Norm stats are computed after joint selection, model padding and delta conversion, so their arrays have `model.action_dim` entries. Automatically masked padding does not contribute to training loss. Re-run this whenever the dataset, `joint_dim`, model dimension, or delta mask changes:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_zoyi_z02_pytorch #--constant-atol 1e-6
```

Only coordinates whose full observed range is at most `constant_atol` are frozen. A coordinate with rare motion remains active even if its 1st and 99th percentiles coincide. Newly computed statistics record `quantile_fallback_mask`: active coordinates with `q99-q01 <= 1e-6` use the full observed min/max for both normalization and inverse normalization. Normal quantile intervals keep the original formula. Old statistics without this field keep their original scaling, including files that already contain min/max; regenerate statistics before training to enable this protection.

Then start training:

```bash
bash train_pi05_z02_pytorch.sh
```

Or equivalently:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_zoyi_z02 --exp-name=my_experiment --overwrite
```

Checkpoints are written under `checkpoints/pi05_zoyi_z02/<exp-name>/`. Serve a checkpoint with:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_zoyi_z02 \
    --policy.dir=checkpoints/pi05_zoyi_z02/my_experiment/20000
```

The served policy returns `joint_dim` actions (25-D in the current JAX/PyTorch Zoyi π0.5 presets). It preserves input order. The current mapping uses indices 0–6 and 8–14 as deltas, with 7, 15, and 16–24 absolute. Changing this mapping requires new statistics and training; do not reuse statistics that treated those body coordinates as padding.

### 5. Use the 25-D ROS clients

[`pi05_infer_z02_jax.py`](inference/pi05_infer_z02_jax.py) and [`pi05_infer_z02_pytorch.py`](inference/pi05_infer_z02_pytorch.py) read the matching config and use its `joint_dim` for observations and warmup. Both use this 25-D Z02 collection layout:

| Model indices | Observation source | Command destination |
| --- | --- | --- |
| 0–6 | `JointState.position[4:11]` | `joint_positions[4:11]` (left arm) |
| 7 | Binary state from left fingers `[22,24,25,27]` | Existing left-hand open/close targets |
| 8–14 | `JointState.position[11:18]` | `joint_positions[11:18]` (right arm) |
| 15 | Binary state from right fingers `[32,34,35,37]` | Existing right-hand open/close targets |
| 16–19 | `JointState.position[:4]` (waist and lift feedback) | Disabled |
| 20–21 | `JointState.position[18:20]` | `joint_positions[18:20]` (head) |
| 22–24 | Zeros in the base-velocity slots | Disabled |

Each joint-state message triggers an observation using the latest head and wrist frames from the camera frame hub. Both layouts disable the first four joint control flags. The 25-D layout keeps the measured waist and lift positions and fills the base-velocity slots with zeros to match the checkpoint input shape.

Load a newly trained 25-D checkpoint with its `z02_25d` assets, for example:

```bash
python inference/pi05_infer_z02_jax.py --ros-args \
    -p config:=pi05_zoyi_z02 \
    -p policy_dir:=/home/zoyi/my_work/zoyi_brain/checkpoints/pi05_zoyi_z02/sft_25d/20000
```

Set `config` to the matching RECAP preset when using a RECAP checkpoint. For PyTorch, use `pi05_infer_z02_pytorch.py`, select the corresponding `_pytorch` config, and point `policy_dir` to a PyTorch training checkpoint containing `model.safetensors` and its assets.

### PyTorch continuous inference

The original launch command and `/enable_inference` switch are unchanged; there are no required YAML profiles, calibration commands, benchmark reports, or helper scripts. The PyTorch client defaults to `/home/zoyi/Downloads/20000` and `z02_gate_25d` assets; override `policy_dir` and `asset_id` for another checkpoint. It retains the existing model loader and `torch.compile` configuration.

Normal execution prefetches at 10 cycles and updates at 20 cycles using time-aligned rows from the full prediction. The scheduler allows up to 6 extra cycles for a late replacement. Stale joint and camera observations, stop/reset, and expired plans stop execution. Toggle `/enable_inference` off and back on to retry a latched fault.

### Z02 RL Token bridge

Source ROS Humble and the robot's workspace overlay, start the joint driver and camera producer, and run the bridge with the robot's Python 3.10 environment. Stop the other policy client before starting the bridge so only one process publishes joint commands.

```bash
.venv/bin/python inference/z02_rlt_bridge.py
```

The bridge listens on `127.0.0.1:8765` by default. After RLinf requests a reset, prepare the robot and scene manually, then send `ready` on `/z02_rlt/event` as a `std_msgs/msg/String`. The same topic accepts `actor`, `success`, `failure`, and `abort`. The bridge holds the current joint positions when fresh feedback is available and releases control after an abort or shutdown.

After the first command in an episode, the bridge allows up to 60 seconds between commands. An idle episode is aborted without a task reward. Start a new reset and send `ready` to resume. If learner updates routinely take longer, set `--ros-args -p command_idle_timeout_sec:=90.0` on the bridge command.

## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.10/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint \
    --precision float32
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Z02 π0.5 PyTorch configurations

Edit training parameters in the matching `TrainConfig` in [`config.py`](src/openpi/training/config.py): `pi05_zoyi_z02_pytorch` for ordinary training, or `pi05_zoyi_z02_recap_pytorch` for RECAP. Their default geometry is 25-D, with `boil_water/2026.09.30` and `assets/pi05_zoyi_z02/z02_25d/norm_stats.json`. Ordinary launchers read the preset's data paths. The shared `POLICY_DATA_DIRS` block in `train_recap_z02.sh` can select complete labeled copies; RECAP requires boolean `advantage_positive`. Compute statistics after choosing directories; sharing assets requires the same geometry and statistics source.

Training settings are read from the selected config. For example, change these fields in that preset to request a larger batch:

```python
batch_size=64,                         # Global batch; 2 GPUs receive 32 each.
pytorch_training_precision="bfloat16",
pytorch_gradient_checkpointing=True,   # Save activations memory by recomputing during backward.
pytorch_weight_path="./checkpoints/pi05_base_pytorch",
num_train_steps=30_000,
num_workers=16,
```

The shipped presets keep `batch_size=16`; choose a value that fits your GPUs. The trainer requires a positive global batch divisible by the number of processes. The launcher uses the preset's learning rate, batch, precision, workers, training steps, checkpoint path, and logging settings. Its optional first argument selects a config; later flags can override settings for one run.

```bash
# Convert the π0.5 base weights to the configured pytorch_weight_path first.
# Original ordinary launcher, after computing statistics separately.
bash train_pi05_z02_pytorch.sh
# Fill labeled paths and choose the backend in train_recap_z02.sh (shared configuration).
bash train_recap_policy_z02.sh
# Optional temporary override.
NPROC=2 bash train_pi05_z02_pytorch.sh --exp-name=sft_batch64 --batch-size=64
```

Both paths use the same constant-dimension loss mask and inverse normalization, including the fallback scaling for rare motion. `discrete_state_input=False` remains the current preset default; if changed, keep training and inference configs consistent. CPU tests cover ordinary/RECAP updates and checkpoint statistics with a small model at batch 64. Full-model GPU memory capacity and robot performance have not been measured. PyTorch DDP replicates the model on each GPU, so a larger batch than JAX is not guaranteed.

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`).                                                                                                                                                                                                                                                                                                                         |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA. |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                                                                                                                                                                                                                                                                                                                                                    |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                                                                                                                                                                                                                                                                                                                                         |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts.                                                                                                                                     |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                                                                                                                                                                                                                                                                                                                                                                                  |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                                                                                                                                                                                                                                                                                                                                          |
| Diverging training loss                   | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround.                                                                                                                                                                                                                    |
