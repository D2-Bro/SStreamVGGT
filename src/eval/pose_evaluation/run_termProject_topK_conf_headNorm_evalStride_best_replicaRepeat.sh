#!/bin/bash

set -e

workdir=".."
model_name="StreamVGGT"
ckpt_name="checkpoints"
model_weights="${workdir}/ckpt/${ckpt_name}.pth"

# Replica repeat is read virtually as frames + reversed(frames) in
# pose_evaluation metadata without creating duplicated files on disk.
export SSTREAMVGGT_REPLICA_ROOT="${SSTREAMVGGT_REPLICA_ROOT:-/home/dongjae/data/replica/Replica}"

eval_dataset="replica"
# To run only one scene, uncomment for example:
# seq_args=(--seq_list Apart-1)
seq_args=()

size="518"
max_frames="full_seq"
kf_every="20"
pose_eval_stride="1"
empty_cache_interval="1"
eviction_policy="svd_leverage"
layer_budget_strategy="value_weighted_leverage_pr"
layer_budget_alpha="0.7"
layer_budget_min_tokens="0"
layer_budget_eps="1e-12"
layer_budget_depth_mu="0.6"
layer_budget_depth_sigma="0.2"
layer_budget_depth_floor="0.1"
layer_budget_value_gamma="0.7"
layer_budget_value_norm_type="mean"
layer_budget_norm_source="key"
total_budget="200000"
budget_frame_multiplier="8"
leverage_feature="key"
leverage_eviction_selector="topk"
leverage_similarity_granularity="layer"
leverage_similarity_feature_projection="raw"
leverage_similarity_leverage_gamma="0.5"
leverage_eviction_risk_mode="low_leverage"
leverage_high_outlier_z="3.0"
leverage_dpp_candidate_multiplier="3"
leverage_dpp_greedy_block_size="128"
leverage_dpp_quality_beta="0.0"
leverage_dpp_diversity_beta="1.0"
leverage_dpp_feature_projection="random"

leverage_approx_method="right_sketch_ridge"
leverage_ridge_lambda="0"
leverage_ridge_lambda_mode="absolute"
leverage_ridge_score_chunk_size="8192"
leverage_ridge_jitter="1e-6"
leverage_ridge_dim="256"
leverage_random_seed="42"

history_anchor_strategy="none"
camera_motion_threshold="0.2"
max_anchors="10"
min_anchor_interval="5"
first_frame_special_tokens_only="false"
first_frame_special_args=()
if [ "$first_frame_special_tokens_only" = true ]; then
    first_frame_special_args=(--first_frame_special_tokens_only)
fi

leverage_dpp_recency_lambda="1.0"
leverage_dpp_recency_window="10"
leverage_dpp_recency_gate_power="0.0"

leverage_conf_gate_floor="0.0"
leverage_conf_gate_depth_alpha="1.0"
leverage_conf_gate_point_beta="0.0"
leverage_conf_gate_k="1.0"
leverage_conf_gate_special_mode="mean"
leverage_conf_gate_init="mean"
leverage_projected_key_cache=false

budget_args=(--budget "$total_budget")
budget_suffix="budget${total_budget}"
if [ -n "$budget_frame_multiplier" ]; then
    budget_args=(--budget_frame_multiplier "$budget_frame_multiplier")
    budget_suffix="budgetFrameMult${budget_frame_multiplier}"
fi
projected_key_cache_args=()
if [ "$leverage_projected_key_cache" = true ]; then
    projected_key_cache_args=(--leverage_projected_key_cache)
fi
output_dir="${workdir}/eval_results/pose_evaluation/${eval_dataset}_S${model_name}_${max_frames}_termProject${leverage_ridge_dim}_a${layer_budget_alpha}_TopK_ridge${leverage_ridge_lambda}_confDepth_spe${leverage_conf_gate_special_mode}_init${leverage_conf_gate_init}_headNorm_poseStride${pose_eval_stride}_${layer_budget_strategy}${layer_budget_depth_mu}${layer_budget_depth_sigma}f${layer_budget_depth_floor}_${budget_suffix}_seed${leverage_random_seed}_stride${kf_every}_chunk1_cacheRLS8"
echo "$output_dir"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

accelerate launch --num_processes 1 --main_process_port 29302 ./eval/pose_evaluation/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --eval_dataset "$eval_dataset" \
    "${seq_args[@]}" \
    --size "$size" \
    --kf_every "$kf_every" \
    --pose_eval_stride "$pose_eval_stride" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_feature "$leverage_feature" \
    --leverage_projection random \
    --leverage_normalize_before_projection \
    --leverage_normalize_before_projection_headwise \
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
    --layer_budget_depth_mu "$layer_budget_depth_mu" \
    --layer_budget_depth_sigma "$layer_budget_depth_sigma" \
    --layer_budget_depth_floor "$layer_budget_depth_floor" \
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
    "${budget_args[@]}" \
    --history_anchor_strategy "$history_anchor_strategy" \
    --camera_motion_threshold "$camera_motion_threshold" \
    --max_anchors "$max_anchors" \
    --min_anchor_interval "$min_anchor_interval" \
    --leverage_dpp_recency_lambda "$leverage_dpp_recency_lambda" \
    --leverage_dpp_recency_window "$leverage_dpp_recency_window" \
    --leverage_dpp_recency_gate_power "$leverage_dpp_recency_gate_power" \
    --leverage_conf_gate \
    --leverage_conf_gate_floor "$leverage_conf_gate_floor" \
    --leverage_conf_gate_depth_alpha "$leverage_conf_gate_depth_alpha" \
    --leverage_conf_gate_point_beta "$leverage_conf_gate_point_beta" \
    --leverage_conf_gate_k "$leverage_conf_gate_k" \
    --leverage_conf_gate_special_mode "$leverage_conf_gate_special_mode" \
    --leverage_conf_gate_init "$leverage_conf_gate_init" \
    --stream_chunk_size 1 \
    --empty_cache_interval "$empty_cache_interval" \
    --rls_refresh_interval 8 \
    --profile_eviction \
    "${projected_key_cache_args[@]}" \
    "${first_frame_special_args[@]}"

# Add --profile_eviction to the launch command above when measuring eviction latency.
