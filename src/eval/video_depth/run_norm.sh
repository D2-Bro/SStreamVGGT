#!/bin/bash

set -e

workdir='..'
model_name='SStreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
leverage_approx_method="right_sketch"
leverage_left_sketch_dim="2048"
leverage_right_jl_dim="64"
leverage_random_seed="0"
max_frames='500'
eviction_policy='svd_leverage'
# datasets=('sintel' 'bonn' 'kitti')
datasets=('kitti_s1_500')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_S${model_name}_${max_frames}_layerSVD_${leverage_approx_method}_norm"
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
        --leverage_approx_method "$leverage_approx_method" \
        --leverage_left_sketch_dim "$leverage_left_sketch_dim" \
        --leverage_right_jl_dim "$leverage_right_jl_dim" \
        --leverage_random_seed "$leverage_random_seed" \
        --leverage_normalize_rows

    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
