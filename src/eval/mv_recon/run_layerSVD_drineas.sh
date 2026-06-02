#!/bin/bash

set -e

workdir=..
model_name=StreamVGGT
ckpt_name=checkpoints
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames=500
budget=200000
eviction_debug=0
eviction_policy=svd_leverage
leverage_approx_method=drineas_srht
leverage_left_sketch_dim=1024
leverage_right_jl_dim=128
leverage_random_seed=0
leverage_eviction_selector=topk
leverage_dpp_candidate_multiplier=2

output_dir="${workdir}/eval_results/mv_recon/S${model_name}_${max_frames}_layerSVD_drineas_r1${leverage_left_sketch_dim}_r2${leverage_right_jl_dim}_fastdpp${leverage_dpp_candidate_multiplier}"
echo "$output_dir"
echo "Budget: $budget"
echo "Leverage: $leverage_approx_method r1=$leverage_left_sketch_dim r2=$leverage_right_jl_dim seed=$leverage_random_seed"
echo "Selector: $leverage_eviction_selector candidate_multiplier=$leverage_dpp_candidate_multiplier"
echo "Eviction debug: $eviction_debug"

eviction_debug_args=()
if [ "${eviction_debug}" = "1" ]; then
    eviction_debug_args=(--eviction_debug)
fi

accelerate launch --num_processes 1 --main_process_port 29650 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --budget "$budget" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_projection random \
    --leverage_approx_method "$leverage_approx_method" \
    --leverage_left_sketch_dim "$leverage_left_sketch_dim" \
    --leverage_right_jl_dim "$leverage_right_jl_dim" \
    --leverage_random_seed "$leverage_random_seed" \
    --leverage_eviction_selector "$leverage_eviction_selector" \
    # --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
    "${eviction_debug_args[@]}"
