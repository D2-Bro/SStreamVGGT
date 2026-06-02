#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='dpp'
merge_window='1'
merge_similarity_threshold='0.9'
merge_voxel_size='0.05'
leverage_eviction_selector=fast_dpp
leverage_dpp_candidate_multiplier=2
leverage_dpp_greedy_block_size=64


output_dir="${workdir}/eval_results/mv_recon/S${model_name}_${max_frames}_layerSVD_headmean4_dpp"
echo "$output_dir"

# --merge_candidate_mode = [spatial, voxel, voxel_spatial]
accelerate launch --num_processes 2 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_projection head_mean \
    --leverage_head_mean_dim 4 \
    --leverage_eviction_selector "$leverage_eviction_selector" \
    --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
    --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size"
# Add --profile_eviction to the launch command above when measuring eviction latency.
