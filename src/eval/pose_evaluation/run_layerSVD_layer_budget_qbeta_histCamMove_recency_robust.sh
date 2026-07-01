#!/bin/bash

set -e

workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"

datasets=('replica')
size='518'
max_frames='500'
kf_every='2'
pose_eval_stride='1'

eviction_policy='svd_leverage'
# Switch to leverage_entropy to test the entropy effective-count allocator.
layer_budget_strategy='value_weighted_leverage_pr'  # [leverage_pr, uniform, value_weighted_leverage_pr, cosine_precomputed]
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
datasets=('NRGBD')
# datasets=('bonn_500')
leverage_eviction_selector=fast_dpp # [topk, fast_dpp, layer_head_fast_dpp]
leverage_eviction_risk_mode=outlier_then_low # [low_leverage, outlier_then_low]
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
leverage_random_seed=0
eviction_protect_recent_frames=0
special_token_interval=1

leverage_approx_tag=""
if [ "$leverage_approx_method" = "right_sketch_ridge" ]; then
    leverage_approx_tag="_rightSketchRidge_r${leverage_ridge_dim}"
fi

history_anchor_strategy=camera_motion
camera_motion_threshold=0.2
max_anchors=10
min_anchor_interval=5

leverage_dpp_recency_lambda=1.0
leverage_dpp_recency_window=10
leverage_dpp_recency_gate_power=0.0

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose_evaluation/${data}_S${model_name}_${max_frames}_layerBudget_${layer_budget_strategy}_a${layer_budget_alpha}_min${layer_budget_min_tokens}_vg${layer_budget_value_gamma}_${layer_budget_value_norm_type}_${layer_budget_norm_source}_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}${leverage_approx_tag}_budget${total_budget}_kf${kf_interval}_evict${evict_interval}_dppQuality${leverage_dpp_quality_beta}_dppRatio${leverage_dpp_diversity_beta}_histCamMove_recencyOnly_lessSpecial_riskOutlierThenLow_z${leverage_high_outlier_z}" # _protectSpecial${special_token_interval}_evictableSVD"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29402 ./eval/pose_evaluation/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --model_name "$model_name" \
        --size "$size" \
        --max_frames "$max_frames" \
        --kf_every "$kf_every" \
        --pose_eval_stride "$pose_eval_stride" \
        --budget "$total_budget" \
        --eviction_policy "$eviction_policy" \
        --leverage_granularity layer \
        --leverage_projection random \
        --leverage_eviction_selector "$leverage_eviction_selector" \
        --leverage_eviction_risk_mode "$leverage_eviction_risk_mode" \
        --leverage_high_outlier_z "$leverage_high_outlier_z" \
        --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
        --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
        --leverage_dpp_quality_beta "$leverage_dpp_quality_beta" \
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
        --kf_interval "$kf_interval" \
        --evict_interval "$evict_interval" \
        --leverage_dpp_diversity_beta "$leverage_dpp_diversity_beta" \
        --eviction_protect_recent_frames "$eviction_protect_recent_frames" \
        --history_anchor_strategy "$history_anchor_strategy" \
        --camera_motion_threshold "$camera_motion_threshold" \
        --max_anchors "$max_anchors" \
        --min_anchor_interval "$min_anchor_interval" \
        --leverage_dpp_recency_bonus \
        --leverage_dpp_recency_lambda "$leverage_dpp_recency_lambda" \
        --leverage_dpp_recency_window "$leverage_dpp_recency_window" \
        --leverage_dpp_recency_gate_power "$leverage_dpp_recency_gate_power"

done

        # --eviction_protect_special_tokens
        # --eviction_protect_special_token_interval "$special_token_interval" \