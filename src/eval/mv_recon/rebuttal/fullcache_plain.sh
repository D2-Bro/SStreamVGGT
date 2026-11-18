#!/bin/bash
# StreamVGGT with a plain full KV cache in SStreamVGGT: no eviction (budget far above any sequence), and none of
# DiaStream's per-token machinery -- no confidence gate (so no IDC confidence state and no per-frame gate update loop),
# no attention column sums (attention-based importance; plain fused SDPA instead of the attn-cuda colsum kernel), and
# no projected-key cache; keys are cached in the value dtype (bf16) as in the official StreamVGGT cache. Usage: bash fullcache_plain.sh <stride ...>; OUT_ROOT, EXTRA_ARGS, CUDA_VISIBLE_DEVICES as usual.
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_root="${OUT_ROOT:-../eval_results/mv_recon/rebuttal/stride_resource}"
for stride in "$@"; do
    echo "=== stride ${stride}: fullcache_plain"
    (cd "${script_dir}/../../.." && KF_EVERY="$stride" TOTAL_BUDGET=1000000000 CONF_GATE=false LEVERAGE_ATTENTION_UTILITY=false \
        LEVERAGE_PROJECTED_KEY_CACHE=false EVAL_DEVICE="${EVAL_DEVICE:-cpu}" OUTPUT_DIR="${out_root}/stride${stride}_fullcache_plain" \
        EXTRA_ARGS="--kv_cache_key_dtype value ${EXTRA_ARGS:-}" bash "${script_dir}/../run_final_ablation.sh")
done
