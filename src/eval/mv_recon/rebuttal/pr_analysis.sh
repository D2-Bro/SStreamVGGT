#!/bin/bash
# Experiment 5: leverage participation ratio (psi) counterexample. Runs the baseline (IDC) and two control
# eviction policies with the cache redundancy log on, then compares them.
#   IDC    : baseline setting (svd_leverage + attention utility + KABA)
#   random : random kept subset (uniform layer budgets: random/mean produce no KABA statistics)
#   mean   : centroid-based (InfiniteVGGT-style) cosine-to-mean policy, uniform layer budgets
# Usage: bash pr_analysis.sh [condition ...]   (default: IDC random mean). Set CUDA_VISIBLE_DEVICES.
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conditions=("$@"); [ ${#conditions[@]} -eq 0 ] && conditions=(IDC random mean)
out_root="${OUT_ROOT:-../eval_results/mv_recon/rebuttal/pr_analysis}"
every="${REDUNDANCY_LOG_EVERY:-10}"
for cond in "${conditions[@]}"; do
    extra="--redundancy_log --redundancy_log_every ${every} ${EXTRA_ARGS:-}"
    case "$cond" in
        IDC) ;;
        random) extra="$extra --eviction_policy random --layer_budget_strategy uniform" ;;
        mean) extra="$extra --eviction_policy mean --layer_budget_strategy uniform" ;;
        *) echo "unknown condition: $cond" >&2; exit 1 ;;
    esac
    echo "=== redundancy log: ${cond}"
    EVAL_DEVICE="${EVAL_DEVICE:-cuda}" OUTPUT_DIR="${out_root}/${cond}" EXTRA_ARGS="$extra" \
        bash "${script_dir}/../run_final_early_attention.sh"
done
echo "Analyze with (from the repo root; paths assume the default OUT_ROOT):"
echo "  python tools/analyze_cache_redundancy.py --run IDC=eval_results/mv_recon/rebuttal/pr_analysis/IDC \\"
echo "      --run random=eval_results/mv_recon/rebuttal/pr_analysis/random --run mean=eval_results/mv_recon/rebuttal/pr_analysis/mean \\"
echo "      --out eval_results/mv_recon/rebuttal/pr_analysis/summary"
echo "  python tools/psi_counterexample.py --redundancy_csv 'eval_results/mv_recon/rebuttal/pr_analysis/IDC/*/redundancy/*.csv'"
