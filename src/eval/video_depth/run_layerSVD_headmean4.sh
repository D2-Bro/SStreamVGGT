#!/bin/bash

set -e

workdir='..'
model_name='SStreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='svd_leverage'
# datasets=('sintel' 'bonn' 'kitti')
datasets=('kitti_s1_500')
leverage_eviction_selector=topk

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_S${model_name}_${max_frames}_layerSVD_${leverage_eviction_selector}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --max_frames "$max_frames" \
        --eviction_policy "$eviction_policy" \
        --leverage_granularity layer \
        --leverage_projection head_mean \
        --leverage_head_mean_dim 4 \
        --leverage_eviction_selector "$leverage_eviction_selector"

    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
