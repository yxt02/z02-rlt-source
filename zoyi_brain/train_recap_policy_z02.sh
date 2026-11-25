#!/usr/bin/env bash
# RECAP 策略流程：计算归一化统计、检查统计来源并启动 JAX/PyTorch 训练。
# Usage:
#   1. 在 train_recap_z02.sh 中填写标签数据目录、策略环境和预训练权重。
#   2. bash train_recap_policy_z02.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 共用配置；训练环境由 POLICY_PYTHON 指定，超参数由 config.py 管理。
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/train_recap_z02.sh"

# 按框架、模型和 RECAP 开关选择预设；POLICY_CONFIG 可指定自定义预设。
case "${USE_RECAP}" in
  true|false)
    ;;
  *)
    echo "USE_RECAP must be true or false" >&2
    exit 2
    ;;
esac

case "${TRAIN_BACKEND}:${MODEL}" in
  jax:pi05)
    preset="pi05_zoyi_z02"
    if [[ "${USE_RECAP}" == true ]]; then
      preset="pi05_zoyi_z02_recap"
    fi
    ;;
  pytorch:pi05)
    preset="pi05_zoyi_z02_pytorch"
    if [[ "${USE_RECAP}" == true ]]; then
      preset="pi05_zoyi_z02_recap_pytorch"
    fi
    ;;
  jax:pi0)
    preset="pi0_zoyi_z02"
    ;;
  pytorch:pi0)
    preset="pi0_zoyi_z02_pytorch"
    ;;
  jax:pi05_lora)
    preset="pi05_zoyi_z02_lora"
    ;;
  *)
    echo "Unsupported backend/model: ${TRAIN_BACKEND}/${MODEL}" >&2
    exit 2
    ;;
esac

DEFAULT_TRAIN_CONFIG="${POLICY_CONFIG:-${preset}}"
DEFAULT_RECAP="${USE_RECAP}"
DEFAULT_STAGE="${POLICY_STAGE}"
DEFAULT_DRY_RUN="${DRY_RUN}"
DEFAULT_REUSE="${REUSE}"
DATA_DIRS=("${POLICY_DATA_DIRS[@]}")
ASSETS_DIR="${POLICY_ASSETS_DIR}"
ASSET_ID="${POLICY_ASSET_ID}"
WEIGHT_PATH="${POLICY_WEIGHT_PATH}"
CHECKPOINT_DIR="${POLICY_CHECKPOINT_DIR}"

# 启动策略流程；共用入口保证归一化和训练使用相同的最终参数。
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/train_z02_common.sh" "$@"
