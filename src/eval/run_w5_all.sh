#!/bin/bash
# W5 full experiment matrix (Ours only, run_final_early_attention.sh setting).
#   R1 x3 (rep0 with metrics, rep1/2 timing only) -> R2 (perf_trace) -> R3 (profile_eviction)
#   -> optional budget sweep {30k,120k} R1 timing only -> tables/figures.
# Env:
#   GPU              CUDA_VISIBLE_DEVICES                                [0]
#   BENCHES          mv_recon and/or pose                                [mv_recon pose]
#   MV_DATASETS      mv_recon datasets                                   [7scenes,NRGBD,ETH3D]
#   POSE_DATASETS    pose datasets                                       [tum scannet]
#   REPS             number of R1 repetitions                            [3]
#   SWEEP            true -> also run budget sweep                       [false]
#   SWEEP_BUDGETS    budgets for the sweep                               [30000 120000]
#   SWEEP_MAX_SCENES mv_recon scenes per dataset for the sweep (0 = all) [0]
#   SWEEP_POSE_SEQS  "dataset:seq1,seq2 ..." pose sequences for sweep    [] (= all)
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/w5/config.sh"
w5_activate_env || { echo "conda env '$W5_CONDA_ENV' not found; run: bash w5/setup_env.sh" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" python "${repo_root}/w5/preflight.py" || exit 1
export GPU="${GPU:-0}"
benches="${BENCHES:-mv_recon pose}"
mv_datasets="${MV_DATASETS:-7scenes,NRGBD,ETH3D}"
pose_datasets="${POSE_DATASETS:-tum scannet}"
reps="${REPS:-3}"
sweep="${SWEEP:-false}"
sweep_budgets="${SWEEP_BUDGETS:-30000 120000}"

run_mv() {  # mode budget rep [extra env...]
    RUN_MODE="$1" TOTAL_BUDGET="$2" REP="$3" DATASETS="$mv_datasets" \
        bash "${script_dir}/mv_recon/run_w5_profile.sh"
}
run_pose() {  # dataset mode budget rep
    POSE_EVAL_DATASET="$1" RUN_MODE="$2" TOTAL_BUDGET="$3" REP="$4" \
        bash "${script_dir}/pose_evaluation/run_w5_profile.sh"
}

echo "[W5] GPU=${GPU} benches=${benches}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

for bench in $benches; do
    for ((rep = 0; rep < reps; rep++)); do
        echo "[W5] ${bench} R1 rep${rep}"
        if [ "$bench" = mv_recon ]; then run_mv R1 60000 "$rep"; else
            for ds in $pose_datasets; do run_pose "$ds" R1 60000 "$rep"; done; fi
    done
    for mode in R2 R3; do
        echo "[W5] ${bench} ${mode}"
        if [ "$bench" = mv_recon ]; then run_mv "$mode" 60000 0; else
            for ds in $pose_datasets; do run_pose "$ds" "$mode" 60000 0; done; fi
    done
    if [ "$sweep" = true ]; then
        for budget in $sweep_budgets; do
            echo "[W5] ${bench} sweep budget=${budget}"
            if [ "$bench" = mv_recon ]; then
                MAX_SCENES="${SWEEP_MAX_SCENES:-0}" SKIP_EVAL=true run_mv R1 "$budget" 0
            else
                for ds in $pose_datasets; do
                    seqs=""
                    for entry in ${SWEEP_POSE_SEQS:-}; do
                        [ "${entry%%:*}" = "$ds" ] && seqs="${entry#*:}"
                    done
                    SEQS="$seqs" SKIP_EVAL=true run_pose "$ds" R1 "$budget" 0
                done
            fi
        done
    fi
done

cd "$repo_root"
python tools/summarize_w5_profile.py --root eval_results/w5
