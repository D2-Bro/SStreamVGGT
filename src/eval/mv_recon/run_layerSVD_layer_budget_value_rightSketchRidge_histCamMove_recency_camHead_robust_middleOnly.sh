#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='svd_leverage'
eviction_policy_layers='9:20'
eviction_policy_layers_tag='EvictLayers9-20'
# Switch to leverage_entropy to test the entropy effective-count allocator.
layer_budget_strategy='value_weighted_leverage_pr' #[leverage_pr, uniform, leverage_entropy, value_weighted_leverage_pr]
layer_budget_alpha=0.5
layer_budget_min_tokens=0
layer_budget_eps=1e-12
layer_budget_value_gamma=0.7
layer_budget_value_norm_type='mean' #[mean, rms]
layer_budget_norm_source='key' #[value, key]
total_budget=200000
leverage_eviction_selector=fast_dpp # [topk, fast_dpp, layer_head_fast_dpp]
leverage_eviction_risk_mode=outlier_then_low # [low_leverage, outlier_then_low]
leverage_high_outlier_z=3.0
leverage_dpp_candidate_multiplier=3
leverage_dpp_greedy_block_size=128
leverage_dpp_quality_beta=0.0
leverage_dpp_diversity_beta=1.0
leverage_dpp_feature_projection=random

leverage_approx_method=right_sketch_ridge
leverage_ridge_lambda=1e-5
leverage_ridge_lambda_mode=relative
leverage_ridge_score_chunk_size=4096
leverage_ridge_jitter=1e-6
leverage_ridge_dim=64
leverage_random_seed=42

history_anchor_strategy=camera_motion
camera_motion_threshold=0.2
max_anchors=10
min_anchor_interval=5

leverage_dpp_recency_lambda=1.0
leverage_dpp_recency_window=10
leverage_dpp_recency_gate_power=0.0

output_dir="${workdir}/eval_results/mv_recon/S${model_name}_${max_frames}_layerBudget_${layer_budget_strategy}_a${layer_budget_alpha}_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}_qb${leverage_dpp_quality_beta}_db${leverage_dpp_diversity_beta}_dppProjRidge_r${leverage_ridge_dim}_rightSketchRidge_r${leverage_ridge_dim}_min${layer_budget_min_tokens}_budget${total_budget}_histCamMove_recency_robust_mid${eviction_policy_layers_tag}"
echo "$output_dir"

accelerate launch --num_processes 2 --main_process_port 29502 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --eviction_policy "$eviction_policy" \
    --eviction_policy_layers "$eviction_policy_layers" \
    --leverage_granularity layer \
    --leverage_projection random \
    --leverage_eviction_selector "$leverage_eviction_selector" \
    --leverage_eviction_risk_mode "$leverage_eviction_risk_mode" \
    --leverage_high_outlier_z "$leverage_high_outlier_z" \
    --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
    --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
    --leverage_dpp_quality_beta "$leverage_dpp_quality_beta" \
    --leverage_dpp_diversity_beta "$leverage_dpp_diversity_beta" \
    --leverage_dpp_feature_projection "$leverage_dpp_feature_projection" \
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
    --budget "$total_budget" \
    --history_anchor_strategy "$history_anchor_strategy" \
    --camera_motion_threshold "$camera_motion_threshold" \
    --max_anchors "$max_anchors" \
    --min_anchor_interval "$min_anchor_interval" \
    --leverage_dpp_recency_bonus \
    --leverage_dpp_recency_lambda "$leverage_dpp_recency_lambda" \
    --leverage_dpp_recency_window "$leverage_dpp_recency_window" \
    --leverage_dpp_recency_gate_power "$leverage_dpp_recency_gate_power" \
    --camera_cache_history_anchors_only \
    --camera_cache_keep_dropped_anchors \
# Add --profile_eviction to the launch command above when measuring eviction latency.
