#!/usr/bin/env bash
# RECAP 价值流程：准备数据、价值训练、评估和完整标签副本导出。
# Usage:
#   1. 在 train_recap_z02.sh 中填写数据目录、价值环境和预训练权重。
#   2. bash train_recap_value_z02.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 共用配置；价值模型始终由 VALUE_PYTHON 启动，此处不激活策略 .venv。
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/train_recap_z02.sh"

# 可选参数；留空的路径沿用工作流默认值。
EXTRA_ARGS=()
case "${RECAP_FORCE_INTERVENTION_POSITIVE}" in
  true)
    EXTRA_ARGS+=(--force-intervention-positive)
    ;;
  false)
    EXTRA_ARGS+=(--no-force-intervention-positive)
    ;;
  *)
    echo "RECAP_FORCE_INTERVENTION_POSITIVE must be true or false" >&2
    exit 2
    ;;
esac
if [[ -n "${VALUE_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--value-checkpoint "${VALUE_CHECKPOINT}")
fi
if [[ -n "${RECAP_TRAINING_DIR}" ]]; then
  EXTRA_ARGS+=(--training-dir "${RECAP_TRAINING_DIR}")
fi
if [[ "${DRY_RUN}" == true ]]; then
  EXTRA_ARGS+=(--dry-run)
fi
if [[ "${REUSE}" == true ]]; then
  EXTRA_ARGS+=(--reuse)
fi

# 启动价值工作流；VALUE_STAGE 默认为 all，可通过末尾参数临时覆盖。
python3 "${SCRIPT_DIR}/workflows/recap/runner.py" \
  --stage "${VALUE_STAGE}" \
  --value-python "${VALUE_PYTHON}" \
  --source-dirs "${VALUE_DATA_DIRS[@]}" \
  --work-root "${RECAP_WORK_DIR}" \
  --data-config "${RECAP_DATA_CONFIG}" \
  --value-config "${RECAP_VALUE_CONFIG}" \
  --siglip-path "${SIGLIP_PATH}" \
  --gemma3-path "${GEMMA3_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --nproc "${VALUE_NPROC}" \
  --horizon "${RECAP_HORIZON}" \
  --label-rule "${RECAP_LABEL_RULE}" \
  --percentile "${RECAP_POSITIVE_PERCENT}" \
  --threshold "${RECAP_THRESHOLD}" \
  --reference-split "${RECAP_REFERENCE_SPLIT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
