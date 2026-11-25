#!/usr/bin/env bash
# 使用 PyTorch 微调 Zoyi Z02 的 25 维 pi0.5 模型。
# 复用 pi05_zoyi_z02 的统计量；首次运行前先转换 JAX 基础权重。
# Usage:
#   1. python scripts/compute_norm_stats.py --config-name=pi05_zoyi_z02
#   2. python examples/convert_jax_model_to_pytorch.py \
#        --config_name=pi05_zoyi_z02 \
#        --checkpoint_dir="${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_base" \
#        --output_path="./checkpoints/pi05_base_pytorch"
#   3. bash train_pi05_z02_pytorch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CURRENT_DATE="$(date +%Y%m%d)"

# 启动配置；相对路径从仓库根目录算起。
NPROC="${NPROC:-2}" # GPU 数量；config.py 的 batch_size 为所有 GPU 的总 batch。
TRAIN_CONFIG="pi05_zoyi_z02_pytorch"
WEIGHT_PATH="./checkpoints/pi05_base_pytorch"

# 可选：第一个位置参数指定其他预设，其余参数临时覆盖配置。
# 例如：NPROC=2 bash train_pi05_z02_pytorch.sh --exp-name=sft_25d --batch-size=64
if [[ $# -gt 0 && "$1" != -* ]]; then
  TRAIN_CONFIG="$1"
  shift
fi

# RECAP 预设沿用当前对应的学习率设置。
WARMUP_STEPS=1000
PEAK_LR=1.2e-5
DECAY_LR=1.2e-6
if [[ "${TRAIN_CONFIG}" == "pi05_zoyi_z02_recap_pytorch" ]]; then
  WARMUP_STEPS=5000
  PEAK_LR=1e-5
  DECAY_LR=0.0
fi

# 启动训练；在这里逐项修改参数，会覆盖 config.py 的同名设置。
# 当前训练步数保持 30000；如需覆盖已有同名实验，可添加 --overwrite。
# 末尾 "$@" 保留临时覆盖参数的方式。
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  scripts/train_pytorch.py \
  "${TRAIN_CONFIG}" \
  --no-wandb_enabled \
  --pytorch_weight_path="${WEIGHT_PATH}" \
  --exp-name="${CURRENT_DATE}" \
  --checkpoint_base_dir="./checkpoints" \
  --pytorch_training_precision="bfloat16" \
  --save_interval=2000 \
  --batch_size=16 \
  --num_workers=16 \
  --num_train_steps=45000 \
  --lr-schedule.warmup-steps="${WARMUP_STEPS}" \
  --lr-schedule.peak-lr="${PEAK_LR}" \
  --lr-schedule.decay-steps=45000 \
  --lr-schedule.decay-lr="${DECAY_LR}" \
  "$@"
