#!/usr/bin/env bash
# Fine-tune pi0 in PyTorch on the Zoyi Z02 pick_red_cup LeRobot datasets.
# Reuses norm stats from pi0_zoyi_z02. Convert the JAX base checkpoint first.
# Usage:
#   1. python scripts/compute_norm_stats.py --config-name=pi0_zoyi_z02
#   2. python examples/convert_jax_model_to_pytorch.py \
#        --config_name=pi0_zoyi_z02 \
#        --checkpoint_dir="${HOME}/.cache/openpi/openpi-assets/checkpoints/pi0_base" \
#        --output_path=./checkpoints/pi0_base_pytorch
#   3. bash train_pi0_z02_pytorch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CURRENT_DATE="$(date +%Y%m%d)"
NPROC="${NPROC:-2}"
# WEIGHT_PATH="${SCRIPT_DIR}/checkpoints/pi0_base_pytorch"
WEIGHT_PATH="${HOME}/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  scripts/train_pytorch.py \
  pi0_zoyi_z02_pytorch \
  --overwrite \
  --no-wandb_enabled \
  --pytorch_weight_path="${WEIGHT_PATH}" \
  --exp-name="${CURRENT_DATE}" \
  --checkpoint_base_dir="${SCRIPT_DIR}/checkpoints" \
  --pytorch_training_precision="bfloat16" \
  --save_interval=2000 \
  --batch_size=16 \
  --num_workers=16 \
  --num_train_steps=30000 \
  --lr-schedule.warmup-steps=1000 \
  --lr-schedule.peak-lr=1.2e-5 \
  --lr-schedule.decay-steps=30000 \
  --lr-schedule.decay-lr=1.2e-6 \
  "$@"
