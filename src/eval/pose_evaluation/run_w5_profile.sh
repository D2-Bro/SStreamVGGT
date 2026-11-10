#!/bin/bash
# W5 analysis: pose_evaluation/run_final_early_attention.sh setting with profiling. Env:
#   POSE_EVAL_DATASET tum | scannet                                                   [tum]
#   RUN_MODE          R1 (w5_profile, clean) | R2 (perf_trace) | R3 (profile_eviction) [R1]
#   TOTAL_BUDGET      total KV budget over 24 global layers                           [60000]
#   REP               repetition index (output suffix)                                [0]
#   MAX_SEQS          first N sequences (0 = all)                                     [0]
#   SEQS              comma list of sequence names (default all)                      []
#   MAX_FRAMES        frame cap after stride (empty = full)                           []
#   SKIP_EVAL         true -> skip ATE/RPE            [false for R1 rep0, true otherwise]
#   GPU               CUDA_VISIBLE_DEVICES                                            [0]
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Machine-specific settings (dataset roots, conda env, GPU) live in w5/config.sh.
source "${script_dir}/../../../w5/config.sh"
w5_activate_env || { echo "conda env '$W5_CONDA_ENV' not found; run: bash w5/setup_env.sh" >&2; exit 1; }
cd "${script_dir}/../.."

eval_dataset="${POSE_EVAL_DATASET:-tum}"
run_mode="${RUN_MODE:-R1}"
total_budget="${TOTAL_BUDGET:-60000}"
rep="${REP:-0}"
max_seqs="${MAX_SEQS:-0}"
seqs="${SEQS:-}"
max_frames="${MAX_FRAMES:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
# Accuracy is identical across repetitions, so metrics are computed once (R1, rep 0).
if [ "$run_mode" = R1 ] && [ "$rep" = 0 ]; then
    skip_eval="${SKIP_EVAL:-false}"
else
    skip_eval="${SKIP_EVAL:-true}"
fi

workdir=".."
model_weights="${workdir}/ckpt/checkpoints.pth"

case "$run_mode" in
    R1) mode_args=(--w5_profile) ;;
    R2) mode_args=(--perf_trace) ;;
    R3) mode_args=(--profile_eviction) ;;
    *) echo "Unknown RUN_MODE=$run_mode"; exit 1 ;;
esac
extra_args=()
if [ -n "$seqs" ]; then
    extra_args+=(--w5_seqs "$seqs")
fi
if [ "$skip_eval" = true ]; then
    extra_args+=(--w5_skip_eval)
fi
if [ -n "$max_frames" ]; then
    extra_args+=(--max_frames "$max_frames")
fi

output_dir="${workdir}/eval_results/w5/pose_evaluation/${eval_dataset}/ours_b${total_budget}_${run_mode}_rep${rep}"
echo "$output_dir"
mkdir -p "$output_dir"
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,memory.total --format=csv > "${output_dir}/gpu_info.csv" || true

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no --main_process_port "${MAIN_PORT:-29312}" ./eval/pose_evaluation/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name StreamVGGT \
    --eval_dataset "$eval_dataset" \
    --size 518 \
    --pose_eval_stride 2 \
    --eviction_policy svd_leverage \
    --leverage_granularity layer \
    --leverage_feature key \
    --leverage_projection random \
    --leverage_normalize_before_projection \
    --leverage_normalize_before_projection_headwise \
    --leverage_eviction_selector topk \
    --layer_budget_strategy value_weighted_leverage_pr \
    --layer_budget_alpha 0.7 \
    --layer_budget_min_tokens 0 \
    --layer_budget_eps 0 \
    --leverage_approx_method right_sketch_ridge \
    --leverage_ridge_lambda 0 \
    --leverage_ridge_lambda_mode absolute \
    --leverage_ridge_score_chunk_size 16384 \
    --leverage_ridge_jitter 0 \
    --leverage_ridge_dim 1024 \
    --random_seed 42 \
    --layer_budget_value_gamma 0.7 \
    --layer_budget_value_norm_type mean \
    --layer_budget_norm_source key \
    --budget "$total_budget" \
    --leverage_conf_gate \
    --leverage_conf_gate_floor 0.0 \
    --leverage_conf_gate_depth_alpha 1.0 \
    --leverage_conf_gate_point_beta 0.0 \
    --leverage_conf_gate_k 1.0 \
    --leverage_conf_gate_transform sigmoid \
    --leverage_conf_gate_special_mode mean \
    --leverage_conf_gate_init mean \
    --stream_chunk_size 1 \
    --empty_cache_interval 1 \
    --rls_refresh_interval 8 \
    --leverage_projected_key_cache \
    --leverage_attention_utility \
    --leverage_attention_beta 0.5 \
    --leverage_attention_ema_decay 0.9 \
    --leverage_attention_freeze_updates 5 \
    --leverage_attention_colsum_subsample_ratio 1.0 \
    --w5_max_seqs "$max_seqs" \
    --log-anchor-budget-overage \
    "${mode_args[@]}" \
    "${extra_args[@]}" 2>&1 | tee "${output_dir}/run.log"
