#!/usr/bin/env bash
set -euo pipefail

config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(cd "${config_dir}/../.." && pwd)"
source "${config_dir}/config.sh"

case "${1:-stage1}" in
    stage1|stage2) stage="${1:-stage1}" ;;
    *) printf '%s\n' "Usage: bash $0 [stage1|stage2]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    printf '%s\n' "Edit ${config_dir}/config.sh to change run settings." >&2
    exit 2
fi

cd "$RLINF_ROOT"
python_bin="${RLINF_ROOT}/.venv/bin/python"
export PYTHONPATH="$RLINF_ROOT"
export Z02_RLT_PROMPT="$TASK_PROMPT"
export Z02_RLT_BRIDGE_URL="$BRIDGE_URL"
export HYDRA_FULL_ERROR=1

mkdir -p "$OUTPUT_ROOT"
run_dir="$(mktemp -d "${OUTPUT_ROOT%/}/${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
export Z02_RLT_RUN_DIR="$(cd "$run_dir" && pwd)"
printf 'Run directory: %s\n' "$Z02_RLT_RUN_DIR"

if [[ "$stage" == stage1 ]]; then
    if [[ ! "$STAGE1_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' "STAGE1_GPU_COUNT must be a positive integer." >&2
        exit 2
    fi
    if [[ ! -f "${BASE_MODEL_PATH}/model.safetensors" && ! -f "${BASE_MODEL_PATH}/model.safetensors.index.json" ]]; then
        printf 'Set BASE_MODEL_PATH to a π₀.₅ safetensors directory. Current value: %s\n' "$BASE_MODEL_PATH" >&2
        exit 2
    fi
    if [[ ! -f "$NORM_STATS_PATH" ]]; then
        printf 'Missing %s. Run bash %s/compute_norm_stats.sh first.\n' "$NORM_STATS_PATH" "$config_dir" >&2
        exit 2
    fi
    export Z02_RLT_MODEL_PATH="$(cd "$BASE_MODEL_PATH" && pwd)"
    export Z02_RLT_NORM_STATS="$Z02_RLT_RUN_DIR/stage1/norm_stats.json"
    mkdir -p "$Z02_RLT_RUN_DIR/stage1"
    cp -- "$NORM_STATS_PATH" "$Z02_RLT_NORM_STATS"
    bash "$config_dir/compute_norm_stats.sh" --check-only --output "$Z02_RLT_NORM_STATS"
    datasets_json="$("$python_bin" -c 'import json, pathlib, sys; print(json.dumps([str(pathlib.Path(p).resolve()) for p in sys.argv[1:]], ensure_ascii=False))' "${DATA_DIRS[@]}")"
    export EMBODIED_PATH="$RLINF_ROOT/examples/sft"
    command=("$python_bin" -u "$EMBODIED_PATH/train_vla_sft.py" --config-name realworld_z02_rlt_stage1
        "data.train_data_paths=$datasets_json"
        "cluster.component_placement.actor=0-$((STAGE1_GPU_COUNT - 1))"
        "runner.max_steps=$STAGE1_STEPS" "runner.save_interval=$STAGE1_SAVE_INTERVAL")
else
    if [[ -z "$STAGE1_ACTOR_DIR" || ! -f "$STAGE1_ACTOR_DIR/model_state_dict/full_weights.pt" ]]; then
        printf '%s\n' "Set STAGE1_ACTOR_DIR to a completed Stage 1 global_step_<N>/actor directory." >&2
        exit 2
    fi
    export Z02_RLT_MODEL_PATH="$(cd "$STAGE1_ACTOR_DIR" && pwd)"
    stage1_dir="$(cd "$Z02_RLT_MODEL_PATH/../../.." && pwd)"
    export Z02_RLT_NORM_STATS="$stage1_dir/norm_stats.json"
    if [[ ! -f "$Z02_RLT_NORM_STATS" ]]; then
        printf 'Missing the Stage 1 normalization snapshot at %s.\n' "$Z02_RLT_NORM_STATS" >&2
        exit 2
    fi
    export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
    command=("$python_bin" -u "$EMBODIED_PATH/train_async.py" --config-name realworld_z02_rlt_stage2
        "runner.max_steps=$STAGE2_STEPS" "runner.save_interval=$STAGE2_SAVE_INTERVAL")
    if [[ -n "$STAGE2_RESUME_DIR" ]]; then
        command+=("runner.resume_dir=$STAGE2_RESUME_DIR")
    fi
fi

"${command[@]}" 2>&1 | tee "$Z02_RLT_RUN_DIR/train.log"
