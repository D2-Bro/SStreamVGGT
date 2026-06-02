#!/bin/bash

set -e

workdir='..'
model_name='SStreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='dpp'
# datasets=('sintel' 'bonn' 'kitti')
datasets=('kitti_s1_500')
leverage_eviction_selector=fast_dpp
leverage_dpp_candidate_multiplier=4
leverage_dpp_greedy_block_size=32

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_S${model_name}_${max_frames}_layerSVD_headmean4_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}_recentProtect"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518 \
        --max_frames "$max_frames" \
        --eviction_policy "$eviction_policy" \
        --leverage_granularity layer \
        --leverage_projection head_mean \
        --leverage_head_mean_dim 4 \
        --leverage_eviction_selector "$leverage_eviction_selector" \
        --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
        --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
        --eviction_protect_recent_frames 1
    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
