#!/bin/bash
# Experiment 4 (7-Scenes trajectory): large keyframe strides. DiaStream at 60K and 30K, and StreamVGGT with a full (unbounded)
# cache, i.e. a budget far above any sequence's total token count so that no eviction happens.
# Usage: bash stride.sh [stride ...]   (default: 20 50 100; BUDGETS="60000 30000 full" by default).
# Set CUDA_VISIBLE_DEVICES to pick the GPU.
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
strides=("$@"); [ ${#strides[@]} -eq 0 ] && strides=(20 50 100)
out_root="${OUT_ROOT:-../eval_results/pose_evaluation/rebuttal/stride}"
full_cache_budget=1000000000
for stride in "${strides[@]}"; do
    for which in ${BUDGETS:-60000 30000 full}; do
        budget="$which"; name="B${budget}"
        [ "$which" = full ] && { budget="$full_cache_budget"; name="fullcache"; }
        echo "=== stride ${stride}: ${name}"
        KF_EVERY="$stride" TOTAL_BUDGET="$budget" \
        OUTPUT_DIR="${out_root}/stride${stride}_${name}" EXTRA_ARGS="${EXTRA_ARGS:-}" \
            bash "${script_dir}/../../pose_evaluation/run_pose_early_attention.sh"
    done
done
