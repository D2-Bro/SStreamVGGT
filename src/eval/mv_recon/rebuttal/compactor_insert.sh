#!/bin/bash
# Compactor with the frame-block attention score (block = frame, computed once when the frame enters: its queries
# over its own keys, no causal mask; fixed afterwards). Same configurations as compactor.sh otherwise.
# Usage: [TASK=pose] [MAX_FRAMES=full|100] bash compactor_insert.sh <config ...>
#   configs: compactor_uniform_ins compactor_kaba_ins (= C0) C1_headnorm_ins C2_concat_ins
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compactor_env="LEVERAGE_ATTENTION_UTILITY=false CONF_GATE=false LEVERAGE_PROJECTED_KEY_CACHE=false"
idc_env="LEVERAGE_ATTENTION_BETA=0.5 LEVERAGE_ATTENTION_EMA_DECAY=0.9 LEVERAGE_ATTENTION_COLSUM_SUBSAMPLE_RATIO=1.0"
declare -A CFG
CFG[compactor_uniform_ins]="$compactor_env | --eviction_policy compactor --compactor_attention insert --layer_budget_strategy uniform"
CFG[compactor_kaba_ins]="$compactor_env | --eviction_policy compactor --compactor_attention insert"
CFG[C1_headnorm_ins]="$compactor_env | --eviction_policy compactor --compactor_attention insert --compactor_headwise_norm"
CFG[C2_concat_ins]="$idc_env LEVERAGE_ATTENTION_UTILITY=false CONF_GATE=false | --leverage_attention_source noncausal_insert --leverage_score_norm zscore --leverage_attention_beta 0.5"
if [ "${TASK:-recon}" = pose ]; then
    out_root="${OUT_ROOT:-../eval_results/pose_evaluation/rebuttal/compactor_insert}"; frames=full
    runner=eval/pose_evaluation/run_pose_ablation.sh
else
    out_root="${OUT_ROOT:-../eval_results/mv_recon/rebuttal/compactor_insert}"; frames="${MAX_FRAMES:-full}"
    runner=eval/mv_recon/run_final_ablation.sh
fi
cd "${script_dir}/../../.."   # src/
for name in "$@"; do
    [ -n "${CFG[$name]+x}" ] || { echo "unknown config: $name" >&2; exit 1; }
    spec="${CFG[$name]}"; env_part="${spec%%|*}"; flag_part="${spec#*|}"
    echo "=== compactor insert (${TASK:-recon}): ${name} (max_frames=${frames})"
    # shellcheck disable=SC2086
    env $env_part MAX_FRAMES="$frames" EVAL_DEVICE="${EVAL_DEVICE:-cpu}" \
        OUTPUT_DIR="${out_root}/${name}_frames${frames}" EXTRA_ARGS="${flag_part} ${EXTRA_ARGS:-}" bash "$runner"
done
