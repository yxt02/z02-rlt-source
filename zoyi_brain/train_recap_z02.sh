#!/usr/bin/env bash
# RECAP 完整流程：准备数据、价值训练、评估、标签导出、归一化和策略训练。
# Usage:
#   1. 在下方填写数据目录、环境和预训练权重。
#   2. bash train_recap_z02.sh
# 价值和策略独立入口也读取这里的配置；相对路径从仓库根目录算起。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CURRENT_DATE="$(date +%Y%m%d)"

# 模型和本次实验。
TRAIN_BACKEND="${POLICY_BACKEND:-pytorch}" # jax 或 pytorch；完整流程以这里为准
MODEL="${POLICY_MODEL:-pi05}"             # pi0 / pi05 / pi05_lora（LoRA 仅支持 JAX）
USE_RECAP="${POLICY_RECAP:-true}"         # true：使用优势标签；false：普通策略训练
STAGE="${RECAP_STAGE:-all}"              # all：完整流程；也可填 prepare/value/evaluate/export/norm/policy
EXP_NAME="${POLICY_EXP_NAME:-recap_z02_${CURRENT_DATE}}" # checkpoint 的实验名

# 优势计算和正负标签（完整流程、单独价值流程共用，修改这里即可）。
RECAP_HORIZON="${RECAP_HORIZON:-50}"                   # 计算优势时向未来看多少帧；训练导出只选这一个长度
RECAP_LABEL_RULE="${RECAP_LABEL_RULE:-percentile}"     # percentile：按优势排名选正例；fixed：按固定阈值判断
RECAP_POSITIVE_PERCENT="${RECAP_POSITIVE_PERCENT:-40}" # percentile 模式：优势最高的约 40% 判正，填 0~100 之间的数
RECAP_THRESHOLD="${RECAP_THRESHOLD:-0.0}"              # 仅 fixed 模式生效：优势 > 此值判正
RECAP_REFERENCE_SPLIT="${RECAP_REFERENCE_SPLIT:-train}" # 分位阈值的参考划分；默认 train，避免用测试数据定阈值
RECAP_FORCE_INTERVENTION_POSITIVE="${RECAP_FORCE_INTERVENTION_POSITIVE:-true}" # true：接管帧强制置正
# 正例比例按参考划分中的全部优势分数计算；接管置正和同分会让实际比例偏高。
# 当前对所选数据统一计算阈值，多任务不会自动分别定阈值。

# 数据和产物目录（主要修改这里）。
# 支持 ../、绝对路径和带空格的目录。
VALUE_SOURCE_ROOT="${VALUE_SOURCE_ROOT:-data/raw}" # 原始数据根目录；数据可放这里或链接到这里
VALUE_DATA_DIRS=(                                # 价值模型读取的完整 LeRobot 数据集
  "${VALUE_SOURCE_ROOT}/2026.09.08"
  "${VALUE_SOURCE_ROOT}/2026.09.15"
  "${VALUE_SOURCE_ROOT}/2026.09.15_2"
  "${VALUE_SOURCE_ROOT}/2026.09.16_error"
)
RECAP_WORK_DIR="${RECAP_WORK_DIR:-outputs/recap_z02}" # 工作副本、划分、价值 checkpoint、报告
RECAP_TRAINING_DIR="${RECAP_TRAINING_DIR:-}"        # 空：导出到上面工作目录的 labeled/ 下
POLICY_DATA_DIRS=()                              # 策略读取的数据集；请按本次训练填写
# RECAP 示例：POLICY_DATA_DIRS=("outputs/recap_z02/labeled/2026.09.08" "outputs/recap_z02/labeled/2026.09.15")
# 普通模式示例：POLICY_DATA_DIRS=("data/raw/2026.09.08")
# 填 labeled/ 下的每个数据集子目录；不要只填 labeled/ 聚合目录。
# 空数组沿用 config.py 的 repo_ids；流程不会自动替换成刚导出的标签目录。
POLICY_ASSETS_DIR="${POLICY_ASSETS_DIR:-}"         # 空：沿用 config.py 的统计目录；可填 assets/recap_z02
POLICY_ASSET_ID="${POLICY_ASSET_ID:-}"             # 空：沿用配置；改 ID 后推理配置也须使用同一 ID
POLICY_CHECKPOINT_DIR="${POLICY_CHECKPOINT_DIR:-}" # 空：沿用配置；可填 checkpoints/recap_z02

# 两个独立环境和预训练权重。
# 调度器只使用标准库；模型由各自的解释器启动，此处不激活策略 .venv。
VALUE_PYTHON="${VALUE_PYTHON:-${HOME}/anaconda3/envs/value_function/bin/python}"
# 已有 conda 环境可用上面的路径；仓库内价值环境也可填 workflows/recap/.venv/bin/python。
POLICY_PYTHON="${POLICY_PYTHON:-.venv/bin/python}" # 策略环境；不要与价值环境混用
RECAP_MODEL_DIR="${RECAP_MODEL_DIR:-workflows/recap/models}"
SIGLIP_PATH="${SIGLIP_PATH:-${RECAP_MODEL_DIR}/siglip2-so400m-patch14-224}"
GEMMA3_PATH="${GEMMA3_PATH:-${RECAP_MODEL_DIR}/gemma-3-270m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${GEMMA3_PATH}}" # 默认与 Gemma3 权重在同一目录
POLICY_WEIGHT_PATH="${POLICY_WEIGHT_PATH:-}"     # 空：沿用 config.py 的策略初始权重
# PyTorch 示例：checkpoints/pi05_base_pytorch；JAX 示例：gs://openpi-assets/checkpoints/pi05_base/params

# 可选设置（通常保留默认）。
NPROC="${NPROC:-2}"                    # PyTorch 策略使用的 GPU 数量；JAX 设备数由训练配置控制
VALUE_NPROC="${VALUE_NPROC:-1}"        # 价值模型使用的 GPU 数量
DRY_RUN="${RECAP_DRY_RUN:-false}"      # true：仅打印命令，不训练也不创建产物
REUSE="${RECAP_REUSE:-false}"          # true：复用输入与产物均未变的已完成阶段
POLICY_CONFIG="${RECAP_POLICY_CONFIG:-}" # 空：按模型、框架、RECAP 开关自动选择预设
VALUE_STAGE="${RECAP_VALUE_STAGE:-all}" # 单独 bash train_recap_value_z02.sh 时的阶段
POLICY_STAGE="${POLICY_STAGE:-all}"     # 单独 bash train_recap_policy_z02.sh：norm/train/all
RECAP_DATA_CONFIG="${RECAP_DATA_CONFIG:-workflows/recap/config/z02_data.yaml}" # 回报规则和数据划分
RECAP_VALUE_CONFIG="${RECAP_VALUE_CONFIG:-workflows/recap/config/train_value.yaml}" # 价值训练超参数
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-}" # 空：使用工作目录下 value/best_model.pt
# 策略 batch、学习率、训练步数等仍在 config.py 的对应 TrainConfig 中设置。
# 环境变量可临时覆盖上述设置；POLICY_DATA_DIR 可临时指定单个策略数据集。

# 路径处理：启动另一个环境前，将相对路径锚定到仓库根目录。
recap_repo_path() {
  case "$1" in
    ""|/*|*://*) printf '%s' "$1" ;;
    *) printf '%s/%s' "${SCRIPT_DIR}" "$1" ;;
  esac
}

if [[ -n "${POLICY_DATA_DIR:-}" ]]; then
  POLICY_DATA_DIRS=("${POLICY_DATA_DIR}")
fi

for path_name in VALUE_SOURCE_ROOT RECAP_WORK_DIR RECAP_TRAINING_DIR POLICY_ASSETS_DIR \
                 POLICY_CHECKPOINT_DIR VALUE_PYTHON POLICY_PYTHON RECAP_MODEL_DIR SIGLIP_PATH \
                 GEMMA3_PATH TOKENIZER_PATH POLICY_WEIGHT_PATH RECAP_DATA_CONFIG \
                 RECAP_VALUE_CONFIG VALUE_CHECKPOINT; do
  printf -v "${path_name}" '%s' "$(recap_repo_path "${!path_name}")"
done
for path_index in "${!VALUE_DATA_DIRS[@]}"; do
  VALUE_DATA_DIRS[path_index]="$(recap_repo_path "${VALUE_DATA_DIRS[path_index]}")"
done
for path_index in "${!POLICY_DATA_DIRS[@]}"; do
  POLICY_DATA_DIRS[path_index]="$(recap_repo_path "${POLICY_DATA_DIRS[path_index]}")"
done

export RECAP_WORK_DIR
export POLICY_EXP_NAME="${EXP_NAME}"

# 启动完整流程；被价值/策略入口 source 时跳过此段。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  EXTRA_ARGS=()
  case "${USE_RECAP}" in
    true)
      EXTRA_ARGS+=(--recap)
      ;;
    false)
      EXTRA_ARGS+=(--no-recap)
      ;;
    *)
      echo "USE_RECAP must be true or false" >&2
      exit 2
      ;;
  esac

  if [[ -n "${POLICY_CONFIG}" ]]; then
    EXTRA_ARGS+=(--policy-config "${POLICY_CONFIG}")
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    EXTRA_ARGS+=(--dry-run)
  fi
  if [[ "${REUSE}" == true ]]; then
    EXTRA_ARGS+=(--reuse)
  fi

  python3 "${SCRIPT_DIR}/scripts/recap_workflow.py" \
    --stage "${STAGE}" \
    --backend "${TRAIN_BACKEND}" \
    --model "${MODEL}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
fi
