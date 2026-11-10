#!/bin/bash
# Experiment 5 on trajectory datasets (TUM, Replica): IDC with the cache redundancy / local density log,
# through the pose evaluation (POSE_EVAL_DATASET). MAX_FRAMES (default 100) and REDUNDANCY_LOG_EVERY (default 10).
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_root="${OUT_ROOT:-../eval_results/pose_evaluation/rebuttal/pr_analysis}"
MAX_FRAMES="${MAX_FRAMES:-100}" OUTPUT_DIR="${out_root}/IDC_frames${MAX_FRAMES:-100}" \
EXTRA_ARGS="--redundancy_log --redundancy_log_every ${REDUNDANCY_LOG_EVERY:-10} ${EXTRA_ARGS:-}" \
    bash "${script_dir}/../../pose_evaluation/run_pose_early_attention.sh"
