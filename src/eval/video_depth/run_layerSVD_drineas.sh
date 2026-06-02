#!/bin/bash

set -e

workdir=".."
model_name="StreamVGGT"
ckpt_name="checkpoints"
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames="500"
eviction_policy="svd_leverage"
leverage_approx_method="drineas_srht"
leverage_left_sketch_dim="512"
leverage_right_jl_dim="128"
leverage_random_seed="0"
# datasets=("sintel" "bonn" "kitti")
datasets=("kitti_s1_500")

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_S${model_name}_${max_frames}_layerSVD_drineas_r1${leverage_left_sketch_dim}_r2${leverage_right_jl_dim}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1 ../src/eval/video_depth/launch.py \
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
        --leverage_random_seed "$leverage_random_seed"

    python ../src/eval/video_depth/eval_depth.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --align "scale"
done
