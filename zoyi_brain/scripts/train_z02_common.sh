#!/usr/bin/env bash
# Sourced by train_*_z02.sh after its editable path block.
set -euo pipefail
cd "${SCRIPT_DIR}"
export PATH="$(dirname "${POLICY_PYTHON}"):${PATH}"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRAIN_CONFIG="${DEFAULT_TRAIN_CONFIG}"
STAGE="${DEFAULT_STAGE:-train}"
DRY_RUN="${DEFAULT_DRY_RUN:-false}"
REUSE="${DEFAULT_REUSE:-false}"
if [[ $# -gt 0 && "$1" != -* ]]; then TRAIN_CONFIG="$1"; shift; fi
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --stage=*) STAGE="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --reuse) REUSE=true; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
case "${STAGE}" in norm|train|all) ;; *) echo "Unknown stage: ${STAGE}" >&2; exit 2;; esac
TRAIN_ARGS=(--exp-name="${EXP_NAME}")
if [[ "${DEFAULT_RECAP:-}" == true ]]; then TRAIN_ARGS+=(--recap)
elif [[ "${DEFAULT_RECAP:-}" == false ]]; then TRAIN_ARGS+=(--no-recap); fi
if [[ ${#DATA_DIRS[@]} -gt 0 ]]; then TRAIN_ARGS+=(--data.repo-ids "${DATA_DIRS[@]}"); fi
[[ -z "${ASSETS_DIR}" ]] || TRAIN_ARGS+=(--data.assets.assets-dir "${ASSETS_DIR}")
[[ -z "${ASSET_ID}" ]] || TRAIN_ARGS+=(--data.assets.asset-id "${ASSET_ID}")
[[ -z "${CHECKPOINT_DIR}" ]] || TRAIN_ARGS+=(--checkpoint-base-dir "${CHECKPOINT_DIR}")
if [[ -n "${WEIGHT_PATH}" ]]; then
  if [[ "${TRAIN_BACKEND}" == pytorch ]]; then TRAIN_ARGS+=(--pytorch-weight-path "${WEIGHT_PATH}")
  else TRAIN_ARGS+=(--weight-loader.params-path "${WEIGHT_PATH}"); fi
fi
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
run_command() {
  if [[ "${DRY_RUN}" == true ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if [[ "${DRY_RUN}" != true && ( "${STAGE}" != train || "${REUSE}" == true || "${CHECK_NORM_SOURCE:-true}" == true || "${RECORD_POLICY_RUN:-true}" == true ) ]]; then
  TRAIN_JSON="$("${POLICY_PYTHON}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${TRAIN_ARGS[@]}")"
fi
record_stage() {
  if [[ "${DRY_RUN}" != true && "${RECORD_POLICY_RUN:-true}" == true ]]; then
    "${POLICY_PYTHON}" scripts/policy_run_manifest.py --config-name "${TRAIN_CONFIG}" --train-args-json "${TRAIN_JSON}" \
      --manifest "${RECAP_WORK_DIR:-${SCRIPT_DIR}/outputs/recap_z02}/policy_runs.json" --stage "$1" --status "$2"
  fi
}
should_run() {
  if [[ "${DRY_RUN}" != true && "${REUSE}" == true ]]; then
    if "${POLICY_PYTHON}" scripts/policy_run_manifest.py --config-name "${TRAIN_CONFIG}" --train-args-json "${TRAIN_JSON}" \
        --manifest "${RECAP_WORK_DIR:-${SCRIPT_DIR}/outputs/recap_z02}/policy_runs.json" --stage "$1" --status reuse; then
      echo "Reusing completed $1"
      return 1
    else
      result=$?
      [[ "${result}" == 3 ]] || exit "${result}"
    fi
  fi
  return 0
}
if [[ ( "${STAGE}" == norm || "${STAGE}" == all ) ]] && should_run norm; then
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'Normalization uses config %s and the same training arguments: ' "${TRAIN_CONFIG}"
    printf '%q ' "${TRAIN_ARGS[@]}"; printf '\n'
  else
    record_stage norm running
    if run_command "${POLICY_PYTHON}" scripts/compute_norm_stats.py --config-name "${TRAIN_CONFIG}" --train-args-json "${TRAIN_JSON}"; then
      record_stage norm complete
    else
      result=$?; record_stage norm failed; exit "${result}"
    fi
  fi
fi
if [[ ( "${STAGE}" == train || "${STAGE}" == all ) ]] && should_run policy; then
  record_stage policy running
  if [[ "${DRY_RUN}" != true && "${CHECK_NORM_SOURCE:-true}" == true ]]; then
    if ! "${POLICY_PYTHON}" scripts/compute_norm_stats.py --config-name "${TRAIN_CONFIG}" --train-args-json "${TRAIN_JSON}" --check-only; then
      record_stage policy failed; exit 1
    fi
  fi
  if [[ "${TRAIN_BACKEND}" == pytorch ]]; then
    command=("${TORCHRUN:-torchrun}" --standalone --nnodes=1 --nproc_per_node="${NPROC}" scripts/train_pytorch.py "${TRAIN_CONFIG}" "${TRAIN_ARGS[@]}")
  else
    command=("${POLICY_PYTHON}" scripts/train.py "${TRAIN_CONFIG}" "${TRAIN_ARGS[@]}")
  fi
  if run_command "${command[@]}"; then
    record_stage policy complete
  else
    result=$?; record_stage policy failed; exit "${result}"
  fi
fi
