#!/bin/bash
# Single entry point: setup (only if needed) -> preflight -> full W5 matrix -> tables/figures.
# Usage: bash w5/run_all.sh        (settings: w5/config.sh, or override via env)
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${here}/config.sh"
if ! w5_activate_env || ! CUDA_VISIBLE_DEVICES="$GPU" python "${here}/check_attn_cuda.py" >/dev/null 2>&1 \
    || [ ! -s "$W5_CKPT" ]; then
    bash "${here}/setup_env.sh"
fi
exec bash "${W5_REPO_ROOT}/src/eval/run_w5_all.sh"
