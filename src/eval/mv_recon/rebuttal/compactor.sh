#!/bin/bash
# Experiment 3: Compactor vs DiaStream (IDC), and the six differences between them.
#
# Usage: bash compactor.sh <table|config ...>
#   tables: A (full sequence)      = compactor_uniform compactor_kaba ours
#           B (first 100 frames)   = cumulative chain C0..C7 and leave-one-out ours_wo_*
#   configs: any name below. Set CUDA_VISIBLE_DEVICES to pick the GPU.
#   MAX_FRAMES defaults to full for table A configs and 100 for table B configs (override with MAX_FRAMES).
#
# Differences (DiaStream on / Compactor off):
#   (1) causal attention history   history from later frames / current-frame queries only, recomputed each step
#   (2) head-wise normalization    unit norm per head / raw keys
#   (3) concatenation              heads concatenated, one token set per layer / per-head leverage and selection
#   (4) freezing importance        frozen after T=5 updates / never frozen
#   (5) mean-normalization         lev/mean + util/mean, equal weights / z(util) + 0.5 z(lev)
#   (6) 2x leverage for new tokens 2x / 1x
# The Compactor port uses the official compactor-vllm choices (lambda 0.5, ridge 5e-3, mean-centred keys, pooled
# z-score) with three changes for a fair comparison: exact per-head leverage (head dim 64 < sketch 256), the
# attention block is the current frame, softmax scaled by 1/sqrt(D). The anchor (first frame) replaces its
# protected tokens. Step C2 also switches the leverage implementation to IDC's (layer sketch 256, ridge 0, no
# centring), and the confidence gate (not one of the six) is added last (C7 = ours).
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_root="${OUT_ROOT:-../eval_results/mv_recon/rebuttal/compactor}"

# name -> "ENV=... ENV=... | launch.py flags"
declare -A CFG
compactor_env="LEVERAGE_ATTENTION_UTILITY=false CONF_GATE=false LEVERAGE_PROJECTED_KEY_CACHE=false"
idc_env="LEVERAGE_ATTENTION_BETA=0.5 LEVERAGE_ATTENTION_EMA_DECAY=0.9 LEVERAGE_ATTENTION_COLSUM_SUBSAMPLE_RATIO=1.0"
# Table A
CFG[compactor_uniform]="$compactor_env | --eviction_policy compactor --layer_budget_strategy uniform"
CFG[compactor_kaba]="$compactor_env | --eviction_policy compactor"
CFG[ours]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 | "
# Table B, cumulative (KABA throughout)
CFG[C0_compactor]="${CFG[compactor_kaba]}"
CFG[C1_headnorm]="$compactor_env | --eviction_policy compactor --compactor_headwise_norm"
# Same Compactor configurations with each head's keys right-sketched 64 -> 16 before leverage
CFG[compactor_uniform_sk16]="$compactor_env | --eviction_policy compactor --layer_budget_strategy uniform --compactor_sketch_dim 16"
CFG[compactor_kaba_sk16]="$compactor_env | --eviction_policy compactor --compactor_sketch_dim 16"
CFG[C0_compactor_sk16]="$compactor_env | --eviction_policy compactor --compactor_sketch_dim 16"
CFG[C1_headnorm_sk16]="$compactor_env | --eviction_policy compactor --compactor_headwise_norm --compactor_sketch_dim 16"
CFG[C2_concat]="$idc_env LEVERAGE_ATTENTION_UTILITY=false CONF_GATE=false | --leverage_attention_source noncausal_current --leverage_score_norm zscore --leverage_attention_beta 0.5"
CFG[C3_history]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=0 CONF_GATE=false | --leverage_score_norm zscore --leverage_unobserved_lev_multiplier 1"
CFG[C4_freeze]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 CONF_GATE=false | --leverage_score_norm zscore --leverage_unobserved_lev_multiplier 1"
CFG[C5_meannorm]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 CONF_GATE=false | --leverage_unobserved_lev_multiplier 1"
CFG[C6_double]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 CONF_GATE=false | --leverage_unobserved_lev_multiplier 2"
CFG[C7_ours]="${CFG[ours]}"
# Table B, leave-one-out from ours. (3) cannot be removed alone: per-head token sets would need per-head
# confidence/attention state; C1 -> C2 measures it instead.
CFG[ours_wo_history]="$idc_env LEVERAGE_ATTENTION_UTILITY=false | --leverage_attention_source noncausal_current --leverage_attention_beta 0.5"
CFG[ours_wo_headnorm]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 HEADWISE_NORM=false | "
CFG[ours_wo_freeze]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=0 | "
CFG[ours_wo_meannorm]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 | --leverage_score_norm zscore --leverage_compactor_lambda 1.0 --leverage_unobserved_lev_multiplier 2"
CFG[ours_wo_double]="$idc_env LEVERAGE_ATTENTION_UTILITY=true LEVERAGE_ATTENTION_FREEZE_UPDATES=5 | --leverage_unobserved_lev_multiplier 1"

TABLE_A=(compactor_uniform compactor_kaba ours compactor_uniform_sk16 compactor_kaba_sk16)
TABLE_B=(C0_compactor C1_headnorm C2_concat C3_history C4_freeze C5_meannorm C6_double C7_ours
         ours_wo_history ours_wo_headnorm ours_wo_freeze ours_wo_meannorm ours_wo_double
         C0_compactor_sk16 C1_headnorm_sk16)

names=()
for arg in "$@"; do
    case "$arg" in
        A) names+=("${TABLE_A[@]}") ;;
        B) names+=("${TABLE_B[@]}") ;;
        *) [ -n "${CFG[$arg]+x}" ] || { echo "unknown config: $arg" >&2; exit 1; }; names+=("$arg") ;;
    esac
done
[ ${#names[@]} -gt 0 ] || { echo "usage: bash compactor.sh <A|B|config ...>" >&2; exit 1; }

cd "${script_dir}/../../.."   # src/
for name in "${names[@]}"; do
    spec="${CFG[$name]}"
    env_part="${spec%%|*}"
    flag_part="${spec#*|}"
    frames="${MAX_FRAMES:-full}"
    table="A"
    if [[ " ${TABLE_B[*]} " == *" $name "* ]] && [[ " ${TABLE_A[*]} " != *" $name "* ]]; then
        frames="${MAX_FRAMES:-100}"; table="B"
    fi
    echo "=== compactor ${table}: ${name} (max_frames=${frames})"
    run=(env); [ "${DRY_RUN:-0}" = 1 ] && run=(echo env)
    # shellcheck disable=SC2086
    "${run[@]}" $env_part MAX_FRAMES="$frames" EVAL_DEVICE="${EVAL_DEVICE:-cpu}" \
        OUTPUT_DIR="${out_root}/${table}_${name}_frames${frames}" \
        EXTRA_ARGS="${flag_part} ${EXTRA_ARGS:-}" \
        bash eval/mv_recon/run_final_ablation.sh
done
