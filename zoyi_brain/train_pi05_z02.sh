#!/usr/bin/env bash
# Fine-tune pi0.5 on the Zoyi Z02 pick_red_cup LeRobot datasets.
# Usage:
#   1. python scripts/compute_norm_stats.py --config-name=pi05_zoyi_z02
#   2. bash train_pi05_z02.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
CURRENT_DATE="$(date +%Y%m%d)"
# WEIGHT_PATH="${SCRIPT_DIR}/outputs/checkpoints/pi05_jax_z02/20260908/17000/params"
WEIGHT_PATH="gs://openpi-assets/checkpoints/pi05_base/params"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  python scripts/train.py \
  pi05_zoyi_z02 \
  --overwrite \
  --no-wandb_enabled \
  --fsdp_devices=2 \
  --weight_loader.params-path="${WEIGHT_PATH}" \
  --exp-name="${CURRENT_DATE}" \
  --checkpoint_base_dir="${SCRIPT_DIR}/checkpoints" \
  --save_interval=2000 \
  --batch_size=2 \
  --ema_decay=None \
  --num_workers=16 \
  --num_train_steps=30000 \
  --lr-schedule.warmup-steps=1000 \
  --lr-schedule.peak-lr=1.2e-5 \
  --lr-schedule.decay-steps=30000 \
  --lr-schedule.decay-lr=1.2e-6 \
  "$@"
