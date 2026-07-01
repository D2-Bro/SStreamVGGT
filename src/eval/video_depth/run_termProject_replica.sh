#!/bin/bash

set -e

workdir='..'
model_name='SStreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='2000'
eviction_policy='svd_leverage'
# Switch to leverage_entropy to test the entropy effective-count allocator.
layer_budget_strategy='leverage_pr'  # [leverage_pr, uniform, value_weighted_leverage_pr, cosine_precomputed]
layer_budget_proportions_path='../cosine_budget.json' # Required only for cosine_precomputed.
layer_budget_alpha=0.5
layer_budget_min_tokens=0
layer_budget_eps=1e-12
layer_budget_value_gamma=0.7
layer_budget_value_norm_type='mean' #[mean, rms]
layer_budget_norm_source='key' #[value, key]
layer_budget_log_scores=true
total_budget=200000
kf_interval=1
evict_interval=1
layer_budget_proportions_args=()
if [ "$layer_budget_strategy" = cosine_precomputed ]; then
    if [ -z "$layer_budget_proportions_path" ]; then
        echo "layer_budget_proportions_path is required for cosine_precomputed" >&2
        exit 1
    fi
    layer_budget_proportions_args=(--layer_budget_proportions_path "$layer_budget_proportions_path")
fi
layer_budget_log_args=()
if [ "$layer_budget_log_scores" = true ]; then
    layer_budget_log_args=(--layer_budget_log_scores)
fi
# datasets=('sintel' 'bonn' 'kitti')
datasets=('replica')
# datasets=('bonn_500')
leverage_eviction_selector=topk # [topk, fast_dpp, layer_head_fast_dpp]
leverage_eviction_risk_mode=low_leverage # [low_leverage, outlier_then_low]
leverage_high_outlier_z=4.0
leverage_dpp_candidate_multiplier=3
leverage_dpp_greedy_block_size=64
leverage_dpp_quality_beta=0.0
leverage_dpp_diversity_beta=1.0

leverage_approx_method=right_sketch_ridge
leverage_ridge_lambda=0
leverage_ridge_lambda_mode=relative
leverage_ridge_score_chunk_size=4096
leverage_ridge_jitter=1e-6
leverage_ridge_dim=64
leverage_random_seed=42
eviction_protect_recent_frames=0
special_token_interval=1

leverage_approx_tag=""
if [ "$leverage_approx_method" = "right_sketch_ridge" ]; then
    leverage_approx_tag="_rightSketchRidge_r${leverage_ridge_dim}"
fi

history_anchor_strategy=none
camera_motion_threshold=0.3
max_anchors=5
min_anchor_interval=5

leverage_dpp_recency_lambda=1.0
leverage_dpp_recency_window=10
leverage_dpp_recency_gate_power=0.0

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/replica_SStreamVGGT_layerBudget_value_weighted_leverage_pr_a0.5_min0_vg0.7_mean_key_fast_dpp_block64_rightSketchRidge_r256_budget200000_kf1_evict1_dppQuality0.0_dppRatio1.0_histCamMove_recencyOnly_lessSpecial_riskOutlierThenLow_z4.0_norm"
    echo "$output_dir"
    # CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1 ../src/eval/video_depth/launch.py \
    #     --weights "$model_weights" \
    #     --output_dir "$output_dir" \
    #     --eval_dataset "$data" \
    #     --size 518 \
    #     --max_frames "$max_frames" \
    #     --eviction_policy "$eviction_policy" \
    #     --leverage_granularity layer \
    #     --leverage_projection random \
    #     --leverage_eviction_selector "$leverage_eviction_selector" \
    #     --leverage_eviction_risk_mode "$leverage_eviction_risk_mode" \
    #     --leverage_high_outlier_z "$leverage_high_outlier_z" \
    #     --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
    #     --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
    #     --leverage_dpp_quality_beta "$leverage_dpp_quality_beta" \
    #     --leverage_approx_method "$leverage_approx_method" \
    #     --leverage_ridge_lambda "$leverage_ridge_lambda" \
    #     --leverage_ridge_lambda_mode "$leverage_ridge_lambda_mode" \
    #     --leverage_ridge_score_chunk_size "$leverage_ridge_score_chunk_size" \
    #     --leverage_ridge_jitter "$leverage_ridge_jitter" \
    #     --leverage_ridge_dim "$leverage_ridge_dim" \
    #     --leverage_random_seed "$leverage_random_seed" \
    #     --layer_budget_strategy "$layer_budget_strategy" \
    #     "${layer_budget_proportions_args[@]}" \
    #     --layer_budget_alpha "$layer_budget_alpha" \
    #     --layer_budget_min_tokens "$layer_budget_min_tokens" \
    #     --layer_budget_eps "$layer_budget_eps" \
    #     --layer_budget_value_gamma "$layer_budget_value_gamma" \
    #     --layer_budget_value_norm_type "$layer_budget_value_norm_type" \
    #     --layer_budget_norm_source "$layer_budget_norm_source" \
    #     --budget "$total_budget" \
    #     --kf_interval "$kf_interval" \
    #     --evict_interval "$evict_interval" \
    #     --leverage_dpp_diversity_beta "$leverage_dpp_diversity_beta" \
    #     --eviction_protect_recent_frames "$eviction_protect_recent_frames" \
    #     --history_anchor_strategy "$history_anchor_strategy" \
    #     --stream_depth_save \
    #     "${layer_budget_log_args[@]}"
    python ../src/eval/video_depth/eval_depth.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --align "scale" \
        --save_error_visuals \
        --save_error_overlays \
        --error_overlay_alpha 0.7 \
        --eval_stride 5

done

        # --eviction_protect_special_tokens
        # --eviction_protect_special_token_interval "$special_token_interval" \