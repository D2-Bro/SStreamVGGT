#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/../OVGGT/ckpt/${ckpt_name}.pth"
max_frames='full_seq'
eviction_policy='svd_leverage'
layer_budget_strategy='leverage_pr' #[leverage_pr, uniform, leverage_entropy, value_weighted_leverage_pr]


leverage_approx_method=right_sketch_ridge
leverage_ridge_lambda=0
leverage_ridge_dim=64
leverage_random_seed=42


input_root="/home/dongjae/data/Long3D"
gt_root="/home/dongjae/data/Long3D"

output_dir="${workdir}/eval_results/mv_recon_long/S${model_name}_${max_frames}_termProject"

# ==========================================

echo "========================================"
echo "Running Evaluation"
echo "Weights: ${model_weights}"
echo "Input root: ${input_root}"
echo "GT root:    ${gt_root}"
echo "Output:  ${output_dir}"
echo "========================================"

accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon_long/launch.py \
    --weights "$model_weights" \
    --input_root "$input_root" \
    --gt_root "$gt_root" \
    --output_dir "$output_dir" \
    --eviction_policy "$eviction_policy" \
    --leverage_granularity layer \
    --layer_budget_strategy "$layer_budget_strategy" \
    --leverage_approx_method "$leverage_approx_method" \
    --leverage_ridge_lambda "$leverage_ridge_lambda" \
    --leverage_ridge_dim "$leverage_ridge_dim" \
    --leverage_random_seed "$leverage_random_seed" \
    --skip_inference

# Add --profile_eviction to the launch command above when measuring eviction latency.
