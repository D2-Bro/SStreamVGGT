#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='500'
eviction_policy='svd_leverage'
# Switch to leverage_entropy to test the entropy effective-count allocator.
layer_budget_strategy='leverage_pr'
layer_budget_alpha=1.0
layer_budget_min_tokens=0
layer_budget_eps=1e-12
layer_budget_debug=false
layer_budget_debug_args=()
if [ "$layer_budget_debug" = true ]; then
    layer_budget_debug_args=(--layer_budget_debug)
fi
leverage_eviction_selector=fast_dpp
leverage_dpp_candidate_multiplier=3
leverage_dpp_greedy_block_size=64

leverage_approx_method=right_sketch
leverage_sketch_dim=256
leverage_random_seed=0

output_dir="${workdir}/eval_results/mv_recon/S${model_name}_${max_frames}_layerBudget_${layer_budget_strategy}_a${layer_budget_alpha}_${leverage_eviction_selector}_block${leverage_dpp_greedy_block_size}_rightSketch"
echo "$output_dir"

accelerate launch --num_processes 1 --main_process_port 29202 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --leverage_projection random \
    --leverage_eviction_selector "$leverage_eviction_selector" \
    --leverage_dpp_candidate_multiplier "$leverage_dpp_candidate_multiplier" \
    --leverage_dpp_greedy_block_size "$leverage_dpp_greedy_block_size" \
    --layer_budget_strategy "$layer_budget_strategy" \
    --layer_budget_alpha "$layer_budget_alpha" \
    --layer_budget_min_tokens "$layer_budget_min_tokens" \
    --layer_budget_eps "$layer_budget_eps" \
    --leverage_approx_method "$leverage_approx_method" \
    --leverage_sketch_dim "$leverage_sketch_dim" \
    --leverage_random_seed "$leverage_random_seed" \
    "${layer_budget_debug_args[@]}"
# Add --profile_eviction to the launch command above when measuring eviction latency.
