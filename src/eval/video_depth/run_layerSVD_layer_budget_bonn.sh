#!/bin/bash

set -e

workdir='..'
model_name='SStreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='svd_leverage'
# Switch to leverage_entropy to test the entropy effective-count allocator.
layer_budget_strategy='value_weighted_leverage_pr'
layer_budget_alpha=1.0
layer_budget_min_tokens=0
layer_budget_eps=1e-12
layer_budget_value_gamma=0.5
layer_budget_value_norm_type='rms'
layer_budget_norm_source='value' #[value, key]
layer_budget_debug=false
layer_budget_log_scores=true
total_budget=200000
kf_interval=1
evict_interval=1
layer_budget_debug_args=()
if [ "$layer_budget_debug" = true ]; then
    layer_budget_debug_args=(--layer_budget_debug)
fi
layer_budget_log_args=()
if [ "$layer_budget_log_scores" = true ]; then
    layer_budget_log_args=(--layer_budget_log_scores)
fi
# datasets=('sintel' 'bonn' 'kitti')
# datasets=('kitti_s1_500')
datasets=('bonn_500')
leverage_eviction_selector=fast_dpp
leverage_dpp_candidate_multiplier=3
leverage_dpp_greedy_block_size=64
leverage_dpp_diversity_beta=3.0
leverage_approx_method=right_sketch_ridge
leverage_ridge_lambda=1e-5
leverage_ridge_lambda_mode=relative
leverage_ridge_score_chunk_size=4096
leverage_ridge_jitter=1e-6
leverage_ridge_dim=256
leverage_random_seed=0

leverage_approx_tag=""
if [ "$leverage_approx_method" = "right_sketch_ridge" ]; then
    leverage_approx_tag="_rightSketchRidge_r${leverage_ridge_dim}"
fi

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_S${model_name}_layerBudget_${layer_budget_strategy}_a${layer_budget_alpha}_min${layer_budget_min_tokens}_vg${layer_budget_value_gamma}_${layer_budget_value_norm_type}_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}${leverage_approx_tag}_budget${total_budget}_kf${kf_interval}_evict${evict_interval}_dppRatio${leverage_dpp_diversity_beta}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1 ../src/eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518 \
        --max_frames "$max_frames" \
        --eviction_policy "$eviction_policy" \
        --leverage_granularity layer \
        --leverage_projection random \
        --leverage_eviction_selector "$leverage_eviction_selector" \
        --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
        --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
        --leverage_approx_method "$leverage_approx_method" \
        --leverage_ridge_lambda "$leverage_ridge_lambda" \
        --leverage_ridge_lambda_mode "$leverage_ridge_lambda_mode" \
        --leverage_ridge_score_chunk_size "$leverage_ridge_score_chunk_size" \
        --leverage_ridge_jitter "$leverage_ridge_jitter" \
        --leverage_ridge_dim "$leverage_ridge_dim" \
        --leverage_random_seed "$leverage_random_seed" \
        --layer_budget_strategy "$layer_budget_strategy" \
        --layer_budget_alpha "$layer_budget_alpha" \
        --layer_budget_min_tokens "$layer_budget_min_tokens" \
        --layer_budget_eps "$layer_budget_eps" \
        --layer_budget_value_gamma "$layer_budget_value_gamma" \
        --layer_budget_value_norm_type "$layer_budget_value_norm_type" \
        --layer_budget_norm_source "$layer_budget_norm_source" \
        --budget "$total_budget" \
        --kf_interval "$kf_interval" \
        --evict_interval "$evict_interval" \
        --leverage_dpp_diversity_beta "$leverage_dpp_diversity_beta" \
        "${layer_budget_debug_args[@]}" \
        "${layer_budget_log_args[@]}"
    python ../src/eval/video_depth/eval_depth.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --align "scale"
done
