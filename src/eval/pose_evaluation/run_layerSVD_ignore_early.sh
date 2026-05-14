#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='300'
kf_every='2'
eviction_policy='svd_leverage'
merge_window='1'
merge_similarity_threshold='0.9'
merge_voxel_size='0.05'


output_dir="${workdir}/eval_results/pose_evaluation/S${model_name}_${ckpt_name}_layerSVD_ignore_early"
echo "$output_dir"

# --merge_candidate_mode = [spatial, voxel, voxel_spatial]
accelerate launch --num_processes 2 --main_process_port 29602 ./eval/pose_evaluation/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --eval_dataset 7scenes \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --kf_every "$kf_every" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_sketch_dim 16 \
    --leverage_projection head_mean \
    --global-attn-idx-ranges 9:
