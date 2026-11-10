#!/bin/bash
# One-time setup on a new machine (idempotent; safe to re-run).
#   1. conda env  ($W5_CONDA_ENV, Python 3.11, torch 2.4.1+cu121, pinned w5/requirements.txt)
#   2. attn-cuda  (rebuilt for this machine's GPU; needs nvcc from a CUDA 12.x toolkit)
#   3. checkpoint (ckpt/checkpoints.pth from huggingface lch01/StreamVGGT)
# Usage: bash w5/setup_env.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$W5_REPO_ROOT"

# ---- 1. conda env ----------------------------------------------------------
if w5_activate_env; then
    echo "[setup] conda env '$W5_CONDA_ENV' found: $(command -v python)"
else
    base="$(w5_find_conda)"
    if [ -z "$base" ]; then
        echo "[setup] conda not found. Install miniconda or set W5_CONDA_ROOT in w5/config.sh" >&2
        exit 1
    fi
    echo "[setup] creating conda env '$W5_CONDA_ENV' under $base"
    "$base/bin/conda" create -y -n "$W5_CONDA_ENV" python=3.11
    w5_activate_env
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r w5/requirements.txt
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- 2. attn-cuda ----------------------------------------------------------
# The compiled extension is GPU/CUDA specific, so it is rebuilt when it does not
# import on this machine (e.g. after copying the folder).
if CUDA_VISIBLE_DEVICES="$GPU" python w5/check_attn_cuda.py; then
    echo "[setup] attn-cuda OK"
else
    if ! command -v nvcc >/dev/null 2>&1; then
        for c in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.1 /usr/local/cuda-12; do
            [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { export CUDA_HOME="$c"; export PATH="$c/bin:$PATH"; break; }
        done
    fi
    command -v nvcc >/dev/null 2>&1 || { echo "[setup] nvcc not found; install CUDA 12.x toolkit or set CUDA_HOME" >&2; exit 1; }
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        TORCH_CUDA_ARCH_LIST="$(CUDA_VISIBLE_DEVICES="$GPU" python -c "import torch; m,n=torch.cuda.get_device_capability(0); print(f'{m}.{n}')")"
        export TORCH_CUDA_ARCH_LIST
    fi
    echo "[setup] building attn-cuda (nvcc $(nvcc --version | tail -1), arch $TORCH_CUDA_ARCH_LIST)"
    rm -rf attn-cuda/build attn-cuda/attn_cuda/_ext*.so
    CUDA_VISIBLE_DEVICES="$GPU" pip install -e ./attn-cuda --no-build-isolation
    CUDA_VISIBLE_DEVICES="$GPU" python w5/check_attn_cuda.py
fi

# ---- 3. checkpoint ---------------------------------------------------------
if [ -s "$W5_CKPT" ] && [ "$(stat -c %s "$W5_CKPT")" -gt 4000000000 ]; then
    echo "[setup] checkpoint found: $W5_CKPT"
else
    echo "[setup] downloading checkpoint to ckpt/"
    mkdir -p ckpt
    python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('lch01/StreamVGGT', 'checkpoints.pth', local_dir='ckpt'))"
fi

CUDA_VISIBLE_DEVICES="$GPU" python w5/preflight.py
