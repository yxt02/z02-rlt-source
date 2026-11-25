#!/usr/bin/env bash
set -euo pipefail

config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(cd "${config_dir}/../.." && pwd)"
source "${config_dir}/config.sh"

cd "$RLINF_ROOT"
export PYTHONPATH="$RLINF_ROOT"
export CUDA_VISIBLE_DEVICES=""
dataset_args=()
for dataset in "${DATA_DIRS[@]}"; do
    dataset_args+=(--dataset "$dataset")
done

exec "${RLINF_ROOT}/.venv/bin/python" toolkits/lerobot/calculate_z02_norm_stats.py \
    "${dataset_args[@]}" --output "$NORM_STATS_PATH" --constant-atol "$CONSTANT_ATOL" "$@"
