#!/bin/bash

set -e

workdir=".."
model_name="StreamVGGT"

ckpt_name="checkpoints"
model_weights="${workdir}/ckpt/${ckpt_name}.pth"

datasets=("NRGBD")
# datasets=("kitti_trajectory")       # 00-10, GT metric available
# datasets=("kitti_trajectory_test")  # 11-21, prediction-only official test split
# For a single NRGBD scene, add: --seq_list breakfast_room

size="518"
max_frames="500"
kf_every="1"
pose_eval_stride="1"
budget="200000"

eviction_policy="svd_leverage"
leverage_sketch_dim="0"
leverage_granularity="layer"
leverage_feature="key"
leverage_projection="random"
leverage_head_mean_dim="1"
# leverage_approx_method="right_sketch"
# leverage_random_seed="0"
leverage_eviction_selector="fast_dpp"
leverage_dpp_candidate_multiplier="3"
leverage_dpp_greedy_block_size="64"
layer_budget_strategy="uniform"
layer_budget_alpha="0.5"
layer_budget_min_tokens="0"
layer_budget_eps="1e-12"
eviction_protect_recent_frames="0"

history_anchor_strategy="none"
anchor_interval="250"
min_anchor_interval="100"
window_protect_frames="0"
max_anchors="0"
coverage_threshold="0.2"
anchor_keep_ratio="1.0"

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose_evaluation/${data}_${eviction_policy}_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}"
    echo "$output_dir"
    accelerate launch --num_processes 2 --main_process_port 29302 ./eval/pose_evaluation/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size "$size" \
        --kf_every "$kf_every" \
        --pose_eval_stride "$pose_eval_stride" \
        --budget "$budget" \
        --eviction_policy "$eviction_policy" \
        --leverage_sketch_dim "$leverage_sketch_dim" \
        --leverage_granularity "$leverage_granularity" \
        --leverage_feature "$leverage_feature" \
        --leverage_projection "$leverage_projection" \
        --leverage_head_mean_dim "$leverage_head_mean_dim" \
        --leverage_eviction_selector "$leverage_eviction_selector" \
        --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
        --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
        --layer_budget_strategy "$layer_budget_strategy" \
        --layer_budget_alpha "$layer_budget_alpha" \
        --layer_budget_min_tokens "$layer_budget_min_tokens" \
        --layer_budget_eps "$layer_budget_eps" \
        --eviction_protect_recent_frames "$eviction_protect_recent_frames" \
        --history_anchor_strategy "$history_anchor_strategy" \
        --anchor_interval "$anchor_interval" \
        --min_anchor_interval "$min_anchor_interval" \
        --window_protect_frames "$window_protect_frames" \
        --max_anchors "$max_anchors" \
        --coverage_threshold "$coverage_threshold" \
        --anchor_keep_ratio "$anchor_keep_ratio"
done
