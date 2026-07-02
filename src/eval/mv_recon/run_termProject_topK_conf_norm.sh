#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='svd_leverage'
# Switch to leverage_entropy to test the entropy effective-count allocator.
layer_budget_strategy='value_weighted_leverage_pr' #[leverage_pr, uniform, leverage_entropy, value_weighted_leverage_pr]
layer_budget_alpha=0.7
layer_budget_min_tokens=0
layer_budget_eps=1e-12
layer_budget_value_gamma=0.7
layer_budget_value_norm_type='mean' #[mean, rms]
layer_budget_norm_source='key' #[value, key]
total_budget=200000
leverage_feature=key # [key, key_value, key_value_lowdim_concat]
leverage_eviction_selector=topk # [topk, fast_dpp, layer_head_fast_dpp, similarity_topk]
leverage_similarity_granularity=layer # [layer, head]
leverage_similarity_feature_projection=raw # [raw, random]
leverage_similarity_leverage_gamma=0.5
leverage_eviction_risk_mode=low_leverage # [low_leverage, outlier_then_low]
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
leverage_ridge_dim=128
leverage_random_seed=42

history_anchor_strategy=none
camera_motion_threshold=0.2
max_anchors=10
min_anchor_interval=5
first_frame_special_tokens_only=false
first_frame_special_args=()
if [ "$first_frame_special_tokens_only" = true ]; then
    first_frame_special_args=(--first_frame_special_tokens_only)
fi

leverage_dpp_recency_lambda=1.0
leverage_dpp_recency_window=10
leverage_dpp_recency_gate_power=0.0
icp_voxel_size=0

leverage_conf_gate_floor=0.0
leverage_conf_gate_depth_alpha=1.0
leverage_conf_gate_point_beta=0.0
leverage_conf_gate_k=1.0
leverage_conf_gate_special_mode=mean    #[one, mean]
leverage_conf_gate_init=mean

eval_frame_stride=1

output_dir="${workdir}/eval_results/mv_recon/S${model_name}_${max_frames}_termProject${leverage_ridge_dim}_a${layer_budget_alpha}_TopK_ridge${leverage_ridge_lambda}_noCrop_icpvox${icp_voxel_size}_confDepth_init${leverage_conf_gate_init}_norm_${layer_budget_strategy}"
echo "$output_dir"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

accelerate launch --num_processes 2 --main_process_port 29202 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_feature "$leverage_feature" \
    --leverage_projection random \
    --leverage_eviction_selector "$leverage_eviction_selector" \
    --leverage_similarity_granularity "$leverage_similarity_granularity" \
    --leverage_similarity_feature_projection "$leverage_similarity_feature_projection" \
    --leverage_similarity_leverage_gamma "$leverage_similarity_leverage_gamma" \
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
    --icp_voxel_size "$icp_voxel_size" \
    --history_anchor_strategy "$history_anchor_strategy" \
    --camera_motion_threshold "$camera_motion_threshold" \
    --max_anchors "$max_anchors" \
    --min_anchor_interval "$min_anchor_interval" \
    --leverage_conf_gate \
    --leverage_conf_gate_floor "$leverage_conf_gate_floor" \
    --leverage_conf_gate_depth_alpha "$leverage_conf_gate_depth_alpha" \
    --leverage_conf_gate_point_beta "$leverage_conf_gate_point_beta" \
    --leverage_conf_gate_k "$leverage_conf_gate_k" \
    --leverage_conf_gate_special_mode "$leverage_conf_gate_special_mode" \
    --leverage_normalize_rows \
    --leverage_conf_gate_init "$leverage_conf_gate_init" \
    --eval_frame_stride "$eval_frame_stride" \
    "${first_frame_special_args[@]}" \
# Add --profile_eviction to the launch command above when measuring eviction latency.
