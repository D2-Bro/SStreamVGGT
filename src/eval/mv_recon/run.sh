#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='300'
eval_frame_stride=2
eval_backend="${eval_backend:-sstream}"

output_dir="${workdir}/eval_results/mv_recon/${model_name}_${max_frames}_baseline_evalstride${eval_frame_stride}_${eval_backend}"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --eval_frame_stride "$eval_frame_stride" \
    --eval_backend "$eval_backend" \
    --eviction_policy "mean"
     