#!/bin/bash

set -euo pipefail

# Run run_inference.py for eviction nearest-retained analysis with the
# left-sketch Drineas-style approximate full leverage mode.
#
# Usage examples:
#   bash run_eviction_nn_inference_drineas.sh
#   DATASET=kitti bash run_eviction_nn_inference_drineas.sh
#   DATASET=7scenes bash run_eviction_nn_inference_drineas.sh
#   DATASET=nrgbd bash run_eviction_nn_inference_drineas.sh
#   DATASET=all bash run_eviction_nn_inference_drineas.sh
#
# Optional overrides:
#   MAX_FRAMES=300 FRAME_STRIDE=2 ANALYSIS_SPACE=both bash run_eviction_nn_inference_drineas.sh
#   LEVERAGE_LEFT_SKETCH_DIM=4096 LEVERAGE_RIGHT_JL_DIM=128 bash run_eviction_nn_inference_drineas.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${DATASET:-kitti}"
MAX_FRAMES="${MAX_FRAMES:-500}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
TOTAL_BUDGET="${TOTAL_BUDGET:-200000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/ckpt/checkpoints.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_results/eviction_nn_run_inference_drineas}"
LINK_ROOT="${LINK_ROOT:-${OUTPUT_ROOT}/_input_links}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"

EVICTION_POLICY="${EVICTION_POLICY:-svd_leverage}"
LEVERAGE_GRANULARITY="${LEVERAGE_GRANULARITY:-layer}"
LEVERAGE_SKETCH_DIM="${LEVERAGE_SKETCH_DIM:-0}"
LEVERAGE_PROJECTION="${LEVERAGE_PROJECTION:-random}"
LEVERAGE_APPROX_METHOD="${LEVERAGE_APPROX_METHOD:-drineas_srht}"
LEVERAGE_LEFT_SKETCH_DIM="${LEVERAGE_LEFT_SKETCH_DIM:-2048}"
LEVERAGE_RIGHT_JL_DIM="${LEVERAGE_RIGHT_JL_DIM:-64}"
LEVERAGE_RANDOM_SEED="${LEVERAGE_RANDOM_SEED:-0}"
ANALYSIS_SPACE="${ANALYSIS_SPACE:-full_key}"
ANALYSIS_MAX_EVICTED="${ANALYSIS_MAX_EVICTED:-8192}"
ANALYSIS_CHUNK_SIZE="${ANALYSIS_CHUNK_SIZE:-2048}"
ANALYSIS_TOPK="${ANALYSIS_TOPK:-256}"
ANALYSIS_LAYERS="${ANALYSIS_LAYERS:-all}"
ANALYSIS_HEADS="${ANALYSIS_HEADS:-all}"
ANALYSIS_STEPS="${ANALYSIS_STEPS:-all}"
ANALYSIS_SEED="${ANALYSIS_SEED:-0}"
ANALYSIS_KNN_K="${ANALYSIS_KNN_K:-5}"

# Defaults chosen from local dataset layout inspected in /home/dongjae/data.
KITTI_INPUT_DIR="${KITTI_INPUT_DIR:-/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/image_gathered_500/2011_09_26_drive_0036_sync_02}"
SCENES7_INPUT_DIR="${SCENES7_INPUT_DIR:-/home/dongjae/data/7scenes_sfm/office/seq-01}"
NRGBD_INPUT_DIR="${NRGBD_INPUT_DIR:-/home/dongjae/data/neural_rgbd_data/breakfast_room/images}"

mkdir -p "${OUTPUT_ROOT}" "${LINK_ROOT}"

prepare_input_dir() {
    local src_dir="$1"
    local name="$2"
    local dst_dir="${LINK_ROOT}/${name}"

    if compgen -G "${src_dir}/*color.*" > /dev/null; then
        echo "${src_dir}"
        return
    fi

    rm -rf "${dst_dir}"
    mkdir -p "${dst_dir}"

    local idx=0
    local file
    while IFS= read -r file; do
        ext="${file##*.}"
        printf -v link_name "%06d.color.%s" "${idx}" "${ext}"
        ln -s "${file}" "${dst_dir}/${link_name}"
        idx=$((idx + 1))
    done < <(find "${src_dir}" -maxdepth 1 -type f \( -iname *.png -o -iname *.jpg -o -iname *.jpeg \) | sort)

    if [ "${idx}" -eq 0 ]; then
        echo "No image files found in ${src_dir}" >&2
        exit 1
    fi

    echo "${dst_dir}"
}

aggregate_one() {
    local name="$1"
    local display_name="$2"
    local analysis_dir="${OUTPUT_ROOT}/${name}_eviction_nn"
    local compare_dir="${OUTPUT_ROOT}/${name}_compare"

    if [ "${RUN_AGGREGATE}" != "1" ]; then
        return
    fi
    if [ ! -f "${analysis_dir}/summary.jsonl" ]; then
        echo "No summary.jsonl found at ${analysis_dir}; skipping aggregate." >&2
        return
    fi

    echo "========================================"
    echo "Aggregate analysis: ${display_name}"
    echo "Input : ${analysis_dir}"
    echo "Output: ${compare_dir}"
    echo "========================================"

    python tools/analyze_eviction_nn_summary.py \
        --input_dirs "${analysis_dir}" \
        --names "${display_name}" \
        --output_dir "${compare_dir}"
}

aggregate_all() {
    if [ "${RUN_AGGREGATE}" != "1" ]; then
        return
    fi

    echo "========================================"
    echo "Aggregate analysis: KITTI vs 7Scenes vs NRGBD"
    echo "Output: ${OUTPUT_ROOT}/compare"
    echo "========================================"

    python tools/analyze_eviction_nn_summary.py \
        --input_dirs \
            "${OUTPUT_ROOT}/kitti_eviction_nn" \
            "${OUTPUT_ROOT}/7scenes_eviction_nn" \
            "${OUTPUT_ROOT}/nrgbd_eviction_nn" \
        --names KITTI 7Scenes NRGBD \
        --output_dir "${OUTPUT_ROOT}/compare"
}

run_one() {
    local name="$1"
    local source_dir="$2"
    local input_dir
    input_dir="$(prepare_input_dir "${source_dir}" "${name}")"

    local analysis_dir="${OUTPUT_ROOT}/${name}_eviction_nn"
    mkdir -p "${analysis_dir}"

    echo "========================================"
    echo "Dataset: ${name}"
    echo "Source : ${source_dir}"
    echo "Input  : ${input_dir}"
    echo "Output : ${analysis_dir}"
    echo "Budget : ${TOTAL_BUDGET}"
    echo "Leverage: ${LEVERAGE_APPROX_METHOD} r1=${LEVERAGE_LEFT_SKETCH_DIM} r2=${LEVERAGE_RIGHT_JL_DIM} seed=${LEVERAGE_RANDOM_SEED}"
    echo "========================================"

    python run_inference.py \
        --input_dir "${input_dir}" \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --frame_stride "${FRAME_STRIDE}" \
        --max_frames "${MAX_FRAMES}" \
        --total_budget "${TOTAL_BUDGET}" \
        --no_cache_results \
        --eviction_policy "${EVICTION_POLICY}" \
        --leverage_granularity "${LEVERAGE_GRANULARITY}" \
        --leverage_sketch_dim "${LEVERAGE_SKETCH_DIM}" \
        --leverage_projection "${LEVERAGE_PROJECTION}" \
        --leverage_approx_method "${LEVERAGE_APPROX_METHOD}" \
        --leverage_left_sketch_dim "${LEVERAGE_LEFT_SKETCH_DIM}" \
        --leverage_right_jl_dim "${LEVERAGE_RIGHT_JL_DIM}" \
        --leverage_random_seed "${LEVERAGE_RANDOM_SEED}" \
        --eviction_nn_analysis_dir "${analysis_dir}" \
        --eviction_nn_analysis_layers "${ANALYSIS_LAYERS}" \
        --eviction_nn_analysis_heads "${ANALYSIS_HEADS}" \
        --eviction_nn_analysis_steps "${ANALYSIS_STEPS}" \
        --eviction_nn_analysis_space "${ANALYSIS_SPACE}" \
        --eviction_nn_analysis_max_evicted "${ANALYSIS_MAX_EVICTED}" \
        --eviction_nn_analysis_chunk_size "${ANALYSIS_CHUNK_SIZE}" \
        --eviction_nn_analysis_save_topk_pairs "${ANALYSIS_TOPK}" \
        --eviction_nn_analysis_seed "${ANALYSIS_SEED}" \
        --eviction_nn_analysis_knn_k "${ANALYSIS_KNN_K}" \
        --leverage_normalize_rows
}

case "${DATASET}" in
    kitti)
        run_one "kitti" "${KITTI_INPUT_DIR}"
        aggregate_one "kitti" "KITTI"
        ;;
    7scenes|scenes7)
        run_one "7scenes" "${SCENES7_INPUT_DIR}"
        aggregate_one "7scenes" "7Scenes"
        ;;
    nrgbd|NRGBD)
        run_one "nrgbd" "${NRGBD_INPUT_DIR}"
        aggregate_one "nrgbd" "NRGBD"
        ;;
    all)
        run_one "kitti" "${KITTI_INPUT_DIR}"
        run_one "7scenes" "${SCENES7_INPUT_DIR}"
        run_one "nrgbd" "${NRGBD_INPUT_DIR}"
        aggregate_all
        ;;
    *)
        echo "Unknown DATASET=${DATASET}. Use kitti, 7scenes, nrgbd, or all." >&2
        exit 1
        ;;
esac
