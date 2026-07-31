#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash eval/mv_recon/run_final_repeat.sh [REPEAT_COUNT] [TOTAL_BUDGET] [STREAM_CHUNK_SIZE]

Run the current mv_recon final experiment repeatedly and save every run in a
separate output directory.

Environment variables:
  REPEAT_COUNT          Number of runs when no positional argument is given (default: 5)
  TOTAL_BUDGET          KV cache token budget used for every run (default: 30000)
  STREAM_CHUNK_SIZE     Number of stream frames processed per chunk (default: 1)
  START_INDEX           First repeat index used in output suffixes (default: 1)
  LEVERAGE_RANDOM_SEED  Seed used for every run (default: run_final.sh default)
  VARY_SEED             Set to true to increment the seed for each run (default: false)
  OUTPUT_DIR            Optional base output directory
  NUM_PROCESSES         accelerate process count (default: 4)
  MAIN_PROCESS_PORT     accelerate main process port (default: 29502)
EOF
}

repeat_count="${1:-${REPEAT_COUNT:-2}}"
total_budget="${2:-${TOTAL_BUDGET:-60000}}"
stream_chunk_size="${3:-${STREAM_CHUNK_SIZE:-1}}"
start_index="${START_INDEX:-1}"
vary_seed="${VARY_SEED:-true}"

if [ "$#" -gt 3 ]; then
    usage >&2
    exit 2
fi
if ! [[ "$repeat_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: REPEAT_COUNT must be a positive integer: $repeat_count" >&2
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
if ! [[ "$start_index" =~ ^[0-9]+$ ]]; then
    echo "Error: START_INDEX must be a non-negative integer: $start_index" >&2
    exit 2
fi
if [[ "$vary_seed" != "true" && "$vary_seed" != "false" ]]; then
    echo "Error: VARY_SEED must be true or false: $vary_seed" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src_dir="$(cd -- "$script_dir/../.." && pwd)"
run_script="$script_dir/run_final_early_attention.sh"
base_seed="${RANDOM_SEED:-45}"

if [[ "$vary_seed" == "true" ]] && ! [[ "$base_seed" =~ ^[0-9]+$ ]]; then
    echo "Error: LEVERAGE_RANDOM_SEED must be a non-negative integer when VARY_SEED=true: $base_seed" >&2
    exit 2
fi

for ((offset = 0; offset < repeat_count; offset++)); do
    repeat_index=$((start_index + offset))
    repeat_tag="$(printf 'repeat%02d' "$repeat_index")"
    seed="$base_seed"
    if [[ "$vary_seed" == "true" ]]; then
        seed=$((base_seed + offset))
    fi

    echo "[$((offset + 1))/$repeat_count] Starting mv_recon ($repeat_tag, seed=$seed)"
    (
        cd "$src_dir"
        RUN_SUFFIX="${repeat_tag}_seed${seed}" \
        TOTAL_BUDGET="$total_budget" \
        STREAM_CHUNK_SIZE="$stream_chunk_size" \
        RANDOM_SEED="$seed" \
        bash "$run_script"
    )
    echo "[$((offset + 1))/$repeat_count] Finished mv_recon ($repeat_tag)"
done

echo "Completed $repeat_count mv_recon run(s)."
