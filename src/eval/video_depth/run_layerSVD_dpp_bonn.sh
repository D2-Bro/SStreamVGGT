#!/bin/bash

set -e

workdir='..'
model_name='SStreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='svd_leverage'
# datasets=('sintel' 'bonn' 'kitti')
datasets=("bonn_${max_frames}")
leverage_eviction_selector=fast_dpp
leverage_dpp_candidate_multiplier=3
leverage_dpp_greedy_block_size=64

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_S${model_name}_layerSVD_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
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
        --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size"
    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
