#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='300'
eviction_policy='svd_leverage'
merge_window='1'
merge_similarity_threshold='0.9'
merge_voxel_size='0.05'


output_dir="${workdir}/eval_results/mv_recon/S${model_name}_${ckpt_name}_layerSVD_headmean2_ignore_early"
echo "$output_dir"

# --merge_candidate_mode = [spatial, voxel, voxel_spatial]
accelerate launch --num_processes 4 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_projection head_mean \
    --leverage_head_mean_dim 2 \
    --global-attn-idx-ranges 9: \

    
