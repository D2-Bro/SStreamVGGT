#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../.."

workdir=".."
model_name="StreamVGGT"
ckpt_name="checkpoints"
model_weights="${workdir}/ckpt/${ckpt_name}.pth"

eval_dataset="${VIDEO_DEPTH_DATASET:-bonn}"
input_frame_stride="${VIDEO_DEPTH_INPUT_STRIDE:-1}"
eval_stride="${VIDEO_DEPTH_EVAL_STRIDE:-1}"
depth_align="${VIDEO_DEPTH_ALIGN:-scale}"

size="518"
empty_cache_interval="1"
stream_chunk_size="${STREAM_CHUNK_SIZE:-1}"
eviction_policy="svd_leverage"
layer_budget_strategy="value_weighted_leverage_pr"
layer_budget_alpha="0.7"
layer_budget_min_tokens="0"
layer_budget_eps="0"
layer_budget_value_gamma="0.7"
layer_budget_value_norm_type="mean"
layer_budget_norm_source="key"
total_budget="60000"
budget_frame_multiplier=""
leverage_feature="key"
leverage_eviction_selector="topk"

leverage_approx_method="right_sketch_ridge"
leverage_ridge_lambda="0"
leverage_ridge_lambda_mode="absolute"
leverage_ridge_score_chunk_size="16384"
leverage_ridge_jitter="0"
leverage_ridge_dim="64"
random_seed="${RANDOM_SEED:-42}"
rls_refresh_interval="8"

leverage_conf_gate_floor="0.0"
leverage_conf_gate_depth_alpha="1.0"
leverage_conf_gate_point_beta="0.0"
leverage_conf_gate_k="1.0"
leverage_conf_gate_transform="${LEVERAGE_CONF_GATE_TRANSFORM:-sigmoid}"
leverage_conf_gate_special_mode="mean"
leverage_conf_gate_init="mean"

attention_beta="${LEVERAGE_ATTENTION_BETA:-0.5}"
attention_ema_decay="${LEVERAGE_ATTENTION_EMA_DECAY:-0.9}"
attention_freeze_updates="${LEVERAGE_ATTENTION_FREEZE_UPDATES:-5}"
attention_colsum_subsample_ratio="${LEVERAGE_ATTENTION_COLSUM_SUBSAMPLE_RATIO:-1.0}"
attention_utility_suffix="_EarlyAttnK${attention_freeze_updates}_g${attention_ema_decay}_b${attention_beta}_sub${attention_colsum_subsample_ratio%.*}_PreEvict_MeanNorm"

budget_args=(--budget "$total_budget")
budget_suffix="budget${total_budget}"
if [ -n "$budget_frame_multiplier" ]; then
    budget_args=(--budget_frame_multiplier "$budget_frame_multiplier")
    budget_suffix="budgetFrameMult${budget_frame_multiplier}"
fi


error_visual_args=()
if [ "${VIDEO_DEPTH_SAVE_ERROR_VISUALS:-false}" = true ]; then
    error_visual_args=(--save_error_visuals --save_error_overlays --error_overlay_alpha 0.7)
fi

output_dir="${workdir}/eval_results/video_depth/Final_result_${eval_dataset}_supple_dim1024"
echo "$output_dir"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

accelerate launch --num_processes 1 --main_process_port 29402 ./eval/video_depth/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --eval_dataset "$eval_dataset" \
    --size "$size" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_feature "$leverage_feature" \
    --leverage_projection random \
    --leverage_normalize_before_projection \
    --leverage_normalize_before_projection_headwise \
    --leverage_projected_key_cache \
    --leverage_eviction_selector "$leverage_eviction_selector" \
    --layer_budget_strategy "$layer_budget_strategy" \
    --layer_budget_alpha "$layer_budget_alpha" \
    --layer_budget_min_tokens "$layer_budget_min_tokens" \
    --layer_budget_eps "$layer_budget_eps" \
    --layer_budget_value_gamma "$layer_budget_value_gamma" \
    --layer_budget_value_norm_type "$layer_budget_value_norm_type" \
    --layer_budget_norm_source "$layer_budget_norm_source" \
    --leverage_approx_method "$leverage_approx_method" \
    --leverage_ridge_lambda "$leverage_ridge_lambda" \
    --leverage_ridge_lambda_mode "$leverage_ridge_lambda_mode" \
    --leverage_ridge_score_chunk_size "$leverage_ridge_score_chunk_size" \
    --leverage_ridge_jitter "$leverage_ridge_jitter" \
    --leverage_ridge_dim "$leverage_ridge_dim" \
    --random_seed "$random_seed" \
    --rls_refresh_interval "$rls_refresh_interval" \
    "${budget_args[@]}" \
    --leverage_conf_gate \
    --leverage_conf_gate_floor "$leverage_conf_gate_floor" \
    --leverage_conf_gate_depth_alpha "$leverage_conf_gate_depth_alpha" \
    --leverage_conf_gate_point_beta "$leverage_conf_gate_point_beta" \
    --leverage_conf_gate_k "$leverage_conf_gate_k" \
    --leverage_conf_gate_transform "$leverage_conf_gate_transform" \
    --leverage_conf_gate_special_mode "$leverage_conf_gate_special_mode" \
    --leverage_conf_gate_init "$leverage_conf_gate_init" \
    --leverage_attention_utility \
    --leverage_attention_beta "$attention_beta" \
    --leverage_attention_ema_decay "$attention_ema_decay" \
    --leverage_attention_freeze_updates "$attention_freeze_updates" \
    --leverage_attention_colsum_subsample_ratio "$attention_colsum_subsample_ratio" \
    --stream_chunk_size "$stream_chunk_size" \
    --empty_cache_interval "$empty_cache_interval" \
    --stream_depth_save

python ./eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$eval_dataset" \
    --align "$depth_align" \
    "${error_visual_args[@]}"
