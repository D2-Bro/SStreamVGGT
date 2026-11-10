#!/bin/bash
# W5 experiment configuration. This is the ONLY file to edit after copying the
# repository to another machine. Every value can also be overridden from the
# environment, e.g. `W5_DATA_ROOT=/data GPU=1 bash w5/run_all.sh`.

# ---- Datasets -------------------------------------------------------------
# Root that contains the dataset folders below.
export W5_DATA_ROOT="${W5_DATA_ROOT:-/mnt/hdd1/dongjae/dataset}"
# Per-dataset folders (relative to W5_DATA_ROOT unless given as absolute paths).
export W5_7SCENES_DIR="${W5_7SCENES_DIR:-7scenes}"
export W5_NRGBD_DIR="${W5_NRGBD_DIR:-neural_rgbd_data}"
export W5_ETH3D_DIR="${W5_ETH3D_DIR:-eth3d}"
export W5_TUM_DIR="${W5_TUM_DIR:-tum}"
export W5_SCANNET_DIR="${W5_SCANNET_DIR:-scannetv2}"

# ---- Python environment ---------------------------------------------------
# Conda env name used for every run. Created by w5/setup_env.sh if missing.
export W5_CONDA_ENV="${W5_CONDA_ENV:-sstream-custom}"
# Optional: explicit conda install prefix (auto-detected when empty).
export W5_CONDA_ROOT="${W5_CONDA_ROOT:-}"

# ---- Hardware -------------------------------------------------------------
export GPU="${GPU:-0}"

# ---- Experiment matrix (see src/eval/run_w5_all.sh) ------------------------
export BENCHES="${BENCHES:-mv_recon pose}"
export MV_DATASETS="${MV_DATASETS:-7scenes,NRGBD,ETH3D}"
export POSE_DATASETS="${POSE_DATASETS:-tum scannet}"
export REPS="${REPS:-3}"
export SWEEP="${SWEEP:-false}"
export SWEEP_BUDGETS="${SWEEP_BUDGETS:-30000 120000}"

# ---- Derived values (do not edit) -------------------------------------------
_w5_abs() { case "$1" in /*) echo "$1" ;; *) echo "${W5_DATA_ROOT}/$1" ;; esac; }
export SSTREAMVGGT_7SCENES_ROOT="$(_w5_abs "$W5_7SCENES_DIR")"
export SSTREAMVGGT_NRGBD_ROOT="$(_w5_abs "$W5_NRGBD_DIR")"
export SSTREAMVGGT_ETH3D_ROOT="$(_w5_abs "$W5_ETH3D_DIR")"
export SSTREAMVGGT_TUM_ROOT="$(_w5_abs "$W5_TUM_DIR")"
export SSTREAMVGGT_SCANNET_ROOT="$(_w5_abs "$W5_SCANNET_DIR")"
W5_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export W5_REPO_ROOT
export W5_CKPT="${W5_REPO_ROOT}/ckpt/checkpoints.pth"

w5_find_conda() {
    if [ -n "$W5_CONDA_ROOT" ]; then echo "$W5_CONDA_ROOT"; return; fi
    if command -v conda >/dev/null 2>&1; then conda info --base 2>/dev/null && return; fi
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        [ -x "$c/bin/conda" ] && { echo "$c"; return; }
    done
}

# Puts the env's bin first on PATH (python, pip, accelerate). Returns 1 if missing.
w5_activate_env() {
    local base
    base="$(w5_find_conda)"
    if [ -z "$base" ] || [ ! -x "$base/envs/$W5_CONDA_ENV/bin/python" ]; then
        return 1
    fi
    export PATH="$base/envs/$W5_CONDA_ENV/bin:$PATH"
    hash -r
}
