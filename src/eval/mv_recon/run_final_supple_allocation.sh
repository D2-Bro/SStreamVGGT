#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage: bash eval/mv_recon/run_final_supple_allocation.sh [TOTAL_BUDGET] [STREAM_CHUNK_SIZE]

Run the current mv_recon final experiment once for each allocation value. Each
value is applied to both layer_budget_alpha and layer_budget_value_gamma.

Environment variables:
  ALLOCATION_VALUES     Space-separated allocation values (default: "1.0 0.7 0.5 0.3")
  TOTAL_BUDGET          KV cache token budget used for every run (default: 60000)
  STREAM_CHUNK_SIZE     Number of stream frames processed per chunk (default: 1)
  RANDOM_SEED           Seed shared by every run (default: 42)
  OUTPUT_DIR            Optional base output directory
  RUN_SUFFIX            Optional suffix appended after the allocation suffix
  NUM_PROCESSES         accelerate process count (default: 4)
  MAIN_PROCESS_PORT     accelerate main process port (default: 29502)
USAGE
}

total_budget="${1:-${TOTAL_BUDGET:-60000}}"
stream_chunk_size="${2:-${STREAM_CHUNK_SIZE:-1}}"
allocation_values_raw="${ALLOCATION_VALUES:-1.0 0.7 0.5 0.3}"
random_seed="${RANDOM_SEED:-42}"

if [ "$#" -gt 2 ]; then
    usage >&2
    exit 2
fi
if ! [[ "$total_budget" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: TOTAL_BUDGET must be a positive integer: $total_budget" >&2
    exit 2
fi
if ! [[ "$stream_chunk_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: STREAM_CHUNK_SIZE must be a positive integer: $stream_chunk_size" >&2
    exit 2
fi
if ! [[ "$random_seed" =~ ^[0-9]+$ ]]; then
    echo "Error: RANDOM_SEED must be a non-negative integer: $random_seed" >&2
    exit 2
fi

read -r -a allocation_values <<< "$allocation_values_raw"
if [ "${#allocation_values[@]}" -eq 0 ]; then
    echo "Error: ALLOCATION_VALUES must contain at least one value." >&2
    exit 2
fi
for allocation_value in "${allocation_values[@]}"; do
    if ! [[ "$allocation_value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error: every ALLOCATION_VALUES entry must be a non-negative number: $allocation_value" >&2
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src_dir="$(cd -- "$script_dir/../.." && pwd)"
run_script="$script_dir/run_final_early_attention.sh"

for index in "${!allocation_values[@]}"; do
    allocation_value="${allocation_values[$index]}"
    run_suffix="alloc${allocation_value}${RUN_SUFFIX:+_${RUN_SUFFIX}}"

    echo "[$((index + 1))/${#allocation_values[@]}] Starting mv_recon (alpha=$allocation_value, gamma=$allocation_value)"
    (
        cd "$src_dir"
        RUN_SUFFIX="$run_suffix" \
        TOTAL_BUDGET="$total_budget" \
        STREAM_CHUNK_SIZE="$stream_chunk_size" \
        RANDOM_SEED="$random_seed" \
        LAYER_BUDGET_ALPHA="$allocation_value" \
        LAYER_BUDGET_VALUE_GAMMA="$allocation_value" \
        bash "$run_script"
    )
    echo "[$((index + 1))/${#allocation_values[@]}] Finished mv_recon (alpha=$allocation_value, gamma=$allocation_value)"
done

echo "Completed ${#allocation_values[@]} mv_recon allocation run(s): ${allocation_values[*]}"
