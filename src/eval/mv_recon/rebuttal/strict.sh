#!/bin/bash
# Experiment 2: strict budget. soft = current behaviour (the first-frame anchor floor may push the
# total cache over the budget); strict = --strict_budget (anchor reserved first, total <= budget).
# Usage: bash strict.sh [budget ...]   (default: 30000 60000; MODES="soft strict" by default).
# Set CUDA_VISIBLE_DEVICES to pick the GPU.
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
budgets=("$@"); [ ${#budgets[@]} -eq 0 ] && budgets=(30000 60000)
out_root="${OUT_ROOT:-../eval_results/mv_recon/rebuttal/strict}"
for budget in "${budgets[@]}"; do
    for mode in ${MODES:-soft strict}; do
        extra="--log-anchor-budget-overage ${EXTRA_ARGS:-}"
        [ "$mode" = strict ] && extra="$extra --strict_budget"
        echo "=== strict budget: B=${budget} ${mode}"
        TOTAL_BUDGET="$budget" EVAL_DEVICE="${EVAL_DEVICE:-cpu}" \
        OUTPUT_DIR="${out_root}/B${budget}_${mode}" EXTRA_ARGS="$extra" \
            bash "${script_dir}/../run_final_early_attention.sh"
    done
done
