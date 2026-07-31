#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage: bash eval/mv_recon/run_final_supple_dimension.sh [TOTAL_BUDGET] [STREAM_CHUNK_SIZE]

Run the current mv_recon final experiment once for each projection dimension.

Environment variables:
  DIMENSIONS            Space-separated projection dimensions (default: "64 256 1024")
  TOTAL_BUDGET          KV cache token budget used for every run (default: 60000)
  STREAM_CHUNK_SIZE     Number of stream frames processed per chunk (default: 1)
  RANDOM_SEED           Seed shared by every run (default: 42)
  OUTPUT_DIR            Optional base output directory; "_dimN" is appended
  RUN_SUFFIX            Optional suffix appended after the dimension suffix
  NUM_PROCESSES         accelerate process count (default: 4)
  MAIN_PROCESS_PORT     accelerate main process port (default: 29502)
USAGE
}

total_budget="${1:-${TOTAL_BUDGET:-60000}}"
stream_chunk_size="${2:-${STREAM_CHUNK_SIZE:-1}}"
dimensions_raw="${DIMENSIONS:-64 256 512 1024}"
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

read -r -a dimensions <<< "$dimensions_raw"
if [ "${#dimensions[@]}" -eq 0 ]; then
    echo "Error: DIMENSIONS must contain at least one dimension." >&2
    exit 2
fi
for dimension in "${dimensions[@]}"; do
    if ! [[ "$dimension" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: every DIMENSIONS value must be a positive integer: $dimension" >&2
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src_dir="$(cd -- "$script_dir/../.." && pwd)"
run_script="$script_dir/run_final_early_attention.sh"

for index in "${!dimensions[@]}"; do
    dimension="${dimensions[$index]}"
    run_suffix="${RUN_SUFFIX:-}"
    if [ -n "${OUTPUT_DIR:-}" ]; then
        run_suffix="dim${dimension}${run_suffix:+_${run_suffix}}"
    fi

    echo "[$((index + 1))/${#dimensions[@]}] Starting mv_recon (dimension=$dimension)"
    (
        cd "$src_dir"
        RUN_SUFFIX="$run_suffix" \
        TOTAL_BUDGET="$total_budget" \
        STREAM_CHUNK_SIZE="$stream_chunk_size" \
        RANDOM_SEED="$random_seed" \
        DIMENSION="$dimension" \
        bash "$run_script"
    )
    echo "[$((index + 1))/${#dimensions[@]}] Finished mv_recon (dimension=$dimension)"
done

echo "Completed ${#dimensions[@]} mv_recon dimension run(s): ${dimensions[*]}"
