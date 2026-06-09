#!/bin/bash

set -e

workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"

datasets=('vbr')
size='518'
max_frames='500'
kf_every='12'
pose_eval_stride='1'

eviction_policy='svd_leverage'
layer_budget_strategy='value_weighted_leverage_pr' # [leverage_pr, uniform, leverage_entropy, value_weighted_leverage_pr]
layer_budget_alpha=0.5
layer_budget_min_tokens=0
layer_budget_eps=1e-12
layer_budget_value_gamma=0.7
layer_budget_value_norm_type='mean' # [mean, rms]
layer_budget_norm_source='key' # [value, key]
total_budget=200000
special_token_interval=5
layer_budget_debug=false
layer_budget_debug_args=()
if [ "$layer_budget_debug" = true ]; then
    layer_budget_debug_args=(--layer_budget_debug)
fi

leverage_sketch_dim=0
leverage_granularity='layer'
leverage_feature='key'
leverage_projection='random'
leverage_head_mean_dim=1
leverage_eviction_selector='fast_dpp' # [topk, fast_dpp, layer_head_fast_dpp]
leverage_dpp_candidate_multiplier=3
leverage_dpp_greedy_block_size=64
leverage_dpp_quality_beta=0.0
leverage_dpp_diversity_beta=1.0

leverage_approx_method='right_sketch_ridge'
leverage_ridge_lambda=1e-5
leverage_ridge_lambda_mode='relative'
leverage_ridge_score_chunk_size=4096
leverage_ridge_jitter=1e-6
leverage_ridge_dim=256
leverage_random_seed=42

eviction_protect_recent_frames=0
history_anchor_strategy='none'
anchor_interval=250
min_anchor_interval=100
window_protect_frames=0
max_anchors=0
coverage_threshold=0.2
anchor_keep_ratio=1.0

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose_evaluation/${data}_S${model_name}_${max_frames}_layerBudget_${layer_budget_strategy}_a${layer_budget_alpha}_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}_qb${leverage_dpp_quality_beta}_db${leverage_dpp_diversity_beta}_rightSketchRidge_r${leverage_ridge_dim}_min${layer_budget_min_tokens}_budget${total_budget}_kf${kf_every}"
    echo "$output_dir"

    accelerate launch --num_processes 1 --main_process_port 29702 ./eval/pose_evaluation/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --model_name "$model_name" \
        --size "$size" \
        --kf_every "$kf_every" \
        --pose_eval_stride "$pose_eval_stride" \
        --budget "$total_budget" \
        --eviction_policy "$eviction_policy" \
        --leverage_sketch_dim "$leverage_sketch_dim" \
        --leverage_granularity "$leverage_granularity" \
        --leverage_feature "$leverage_feature" \
        --leverage_projection "$leverage_projection" \
        --leverage_head_mean_dim "$leverage_head_mean_dim" \
        --leverage_eviction_selector "$leverage_eviction_selector" \
        --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
        --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
        --leverage_dpp_quality_beta "$leverage_dpp_quality_beta" \
        --leverage_dpp_diversity_beta "$leverage_dpp_diversity_beta" \
        --layer_budget_strategy "$layer_budget_strategy" \
        --layer_budget_alpha "$layer_budget_alpha" \
        --layer_budget_min_tokens "$layer_budget_min_tokens" \
        --layer_budget_eps "$layer_budget_eps" \
        --leverage_approx_method "$leverage_approx_method" \
        --leverage_ridge_lambda "$leverage_ridge_lambda" \
        --leverage_ridge_lambda_mode "$leverage_ridge_lambda_mode" \
        --leverage_ridge_score_chunk_size "$leverage_ridge_score_chunk_size" \
        --leverage_ridge_jitter "$leverage_ridge_jitter" \
        --leverage_ridge_dim "$leverage_ridge_dim" \
        --leverage_random_seed "$leverage_random_seed" \
        --layer_budget_value_gamma "$layer_budget_value_gamma" \
        --layer_budget_value_norm_type "$layer_budget_value_norm_type" \
        --layer_budget_norm_source "$layer_budget_norm_source" \
        --leverage_evictable_only \
        --eviction_protect_recent_frames "$eviction_protect_recent_frames" \
        --history_anchor_strategy "$history_anchor_strategy" \
        --anchor_interval "$anchor_interval" \
        --min_anchor_interval "$min_anchor_interval" \
        --window_protect_frames "$window_protect_frames" \
        --max_anchors "$max_anchors" \
        --coverage_threshold "$coverage_threshold" \
        --anchor_keep_ratio "$anchor_keep_ratio" \
        "${layer_budget_debug_args[@]}"
done
# Add --profile_eviction to the launch command above when measuring eviction latency.
# To run a VBR stride variant, change pose_eval_stride above, e.g. pose_eval_stride='3'.
# To protect special tokens, add:
# --eviction_protect_special_tokens \
# --eviction_protect_special_token_interval "$special_token_interval"
