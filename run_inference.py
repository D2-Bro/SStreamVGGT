import os
import torch
import numpy as np
import sys
import glob
import time
import argparse
from typing import List, Dict, Optional

# Add project source to the Python path
sys.path.append("src/")

# Import necessary components from the StreamVGGT project
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import FrameDiskCache
from streamvggt.utils.cache_analysis import (
    CacheAnalysisConfig,
    PreEvictionSnapshotConfig,
    add_eviction_nn_analysis_args,
    eviction_nn_config_from_args,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)



def run_inference(args: argparse.Namespace):
    """
    Main function to load the model, run inference on input images, and save the results.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("Error: CUDA device not available.")
        return

    print("Initializing and loading StreamVGGT model ...")

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    frame_writer = None
    cache_results = not args.no_cache_results

    if args.frame_cache_dir:
        frame_writer = FrameDiskCache(args.frame_cache_dir)

    cache_analysis_config = CacheAnalysisConfig.from_cli(
        args.cache_analysis_dir,
        layers=args.cache_analysis_layers,
        heads=args.cache_analysis_heads,
        steps=args.cache_analysis_steps,
        max_snapshots=args.cache_analysis_max_snapshots,
    )
    try:
        pre_eviction_snapshot_config = PreEvictionSnapshotConfig.from_cli(
            args.snapshot_before_eviction,
            args.snapshot_output_dir,
            frame_count=args.snapshot_frame_count,
            layers=args.snapshot_layers,
            heads=args.snapshot_heads,
            max_snapshots=args.snapshot_max_snapshots,
        )
        eviction_nn_analysis_config = eviction_nn_config_from_args(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    if cache_analysis_config is not None:
        print(f"Cache analysis snapshots enabled: {cache_analysis_config.output_dir}")
    if pre_eviction_snapshot_config is not None:
        print(
            "Pre-eviction common cache snapshot enabled: "
            f"{pre_eviction_snapshot_config.output_dir} at frame_count={pre_eviction_snapshot_config.frame_count}"
        )
    if eviction_nn_analysis_config is not None:
        print(f"Eviction NN analysis enabled: {eviction_nn_analysis_config.output_dir}")
    if args.stream_chunk_size < 1:
        print(
            "Error: --stream_chunk_size must be >= 1, "
            f"got {args.stream_chunk_size}."
        )
        return
    if args.total_budget < 1:
        print(f"Error: --total_budget must be >= 1, got {args.total_budget}.")
        return
    if args.leverage_ridge_lambda < 0:
        print(
            "Error: --leverage_ridge_lambda must be >= 0, "
            f"got {args.leverage_ridge_lambda}."
        )
        return
    if args.leverage_diag_interval < 0:
        print(
            "Error: --leverage_diag_interval must be >= 0, "
            f"got {args.leverage_diag_interval}."
        )
        return
    if args.leverage_ridge_jitter <= 0:
        print(
            "Error: --leverage_ridge_jitter must be > 0, "
            f"got {args.leverage_ridge_jitter}."
        )
        return
    if args.leverage_ridge_score_chunk_size < 1:
        print(
            "Error: --leverage_ridge_score_chunk_size must be >= 1, "
            f"got {args.leverage_ridge_score_chunk_size}."
        )
        return
    if args.leverage_ridge_dim is not None and args.leverage_ridge_dim < 1:
        print(
            "Error: --leverage_ridge_dim must be >= 1 when provided, "
            f"got {args.leverage_ridge_dim}."
        )
        return
    if args.rls_refresh_interval <= 0:
        print(
            "Error: --rls_refresh_interval must be >= 1, "
            f"got {args.rls_refresh_interval}."
        )
        return
    if args.leverage_approx_method == "right_sketch_ridge":
        resolved_ridge_dim = args.leverage_ridge_dim if args.leverage_ridge_dim is not None else args.leverage_right_jl_dim
        if resolved_ridge_dim is None or int(resolved_ridge_dim) < 1:
            print(
                "Error: --leverage_approx_method right_sketch_ridge requires "
                "--leverage_ridge_dim >= 1 or --leverage_right_jl_dim >= 1."
            )
            return
        args.leverage_ridge_dim = int(resolved_ridge_dim)

    print(f"Using eviction policy: {args.eviction_policy}")
    if args.eviction_policy == "svd_leverage":
        sketch_label = "exact" if args.leverage_sketch_dim == 0 else str(args.leverage_sketch_dim)
        print(f"Using SVD leverage sketch dim: {sketch_label}")
        print(
            "Using SVD leverage approximation: "
            f"{args.leverage_approx_method} (r1={args.leverage_left_sketch_dim}, "
            f"r2={args.leverage_right_jl_dim}, ridge_dim={args.leverage_ridge_dim}, "
            f"ridge_lambda={args.leverage_ridge_lambda}, ridge_lambda_mode={args.leverage_ridge_lambda_mode}, "
            f"ridge_chunk={args.leverage_ridge_score_chunk_size}, ridge_jitter={args.leverage_ridge_jitter}, "
            f"rls_refresh_interval={args.rls_refresh_interval}, "
            f"seed={args.leverage_random_seed})"
        )
        print(
            "Using SVD leverage granularity: "
            f"{args.leverage_granularity} (feature={args.leverage_feature}, "
            f"projection={args.leverage_projection}, head_mean_dim={args.leverage_head_mean_dim}, "
            f"normalize_before_projection={args.leverage_normalize_before_projection})"
        )
        print(
            "Using SVD leverage selector: "
            f"{args.leverage_eviction_selector} "
            f"(dpp_candidate_multiplier={args.leverage_dpp_candidate_multiplier}, "
            f"dpp_greedy_block_size={args.leverage_dpp_greedy_block_size}, "
            f"dpp_quality_beta={args.leverage_dpp_quality_beta})"
        )

    print(f"Using total KV cache budget: {args.total_budget}")
    model = StreamVGGT(total_budget=args.total_budget)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")

    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    del ckpt
    print("Model loaded successfully onto the GPU.")

    print(f"Loading images from input directory: {args.input_dir}")
    # image_names = sorted(glob.glob(os.path.join(args.input_dir, "*color.*")))
    image_names = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    if not image_names:
        print(f"Error: No images found in {args.input_dir}. Please check the path and file extensions.")
        return

    if args.frame_stride < 1:
        print(f"Error: --frame_stride must be >= 1, got {args.frame_stride}.")
        return
    original_num_images = len(image_names)
    image_names = image_names[::args.frame_stride]

    if args.max_frames is not None:
        if args.max_frames < 1:
            print(f"Error: --max_frames must be >= 1, got {args.max_frames}.")
            return
        image_names = image_names[:args.max_frames]

    if pre_eviction_snapshot_config is not None and len(image_names) > pre_eviction_snapshot_config.frame_count:
        image_names = image_names[: pre_eviction_snapshot_config.frame_count]
        print(
            "Snapshot mode truncates inference input to "
            f"{pre_eviction_snapshot_config.frame_count} frames before eviction comparison."
        )

    if not image_names:
        print("Error: No images remain after applying --frame_stride/--max_frames.")
        return

    if args.frame_stride > 1 or args.max_frames is not None:
        print(
            f"Frame selection: {original_num_images} input images -> {len(image_names)} frames "
            f"(stride={args.frame_stride}, max_frames={args.max_frames})"
        )

    print(f"Found {len(image_names)} images to process.")
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images tensor shape: {images.shape}")

    frames: List[Dict[str, torch.Tensor]] = []
    for i in range(images.shape[0]):
        image_frame = images[i].unsqueeze(0)
        frame = {"img": image_frame}
        frames.append(frame)

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time_model = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(
                frames,
                frame_writer=frame_writer,
                cache_results=cache_results,
                stream_chunk_size=args.stream_chunk_size,
                cache_analysis_config=cache_analysis_config,
                pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                eviction_nn_analysis_config=eviction_nn_analysis_config,
                eviction_policy=args.eviction_policy,
                eviction_debug=args.eviction_debug or args.profile_eviction,
                leverage_sketch_dim=args.leverage_sketch_dim,
                leverage_granularity=args.leverage_granularity,
                leverage_feature=args.leverage_feature,
                leverage_projection=args.leverage_projection,
                leverage_head_mean_dim=args.leverage_head_mean_dim,
                leverage_normalize_before_projection=args.leverage_normalize_before_projection,
                leverage_approx_method=args.leverage_approx_method,
                leverage_ridge_lambda=args.leverage_ridge_lambda,
                leverage_ridge_lambda_mode=args.leverage_ridge_lambda_mode,
                leverage_ridge_score_chunk_size=args.leverage_ridge_score_chunk_size,
                leverage_ridge_jitter=args.leverage_ridge_jitter,
                leverage_ridge_dim=args.leverage_ridge_dim,
                rls_refresh_interval=args.rls_refresh_interval,
                leverage_diag=args.leverage_diag,
                leverage_diag_interval=args.leverage_diag_interval,
                leverage_random_seed=args.leverage_random_seed,
                leverage_eviction_selector=args.leverage_eviction_selector,
                leverage_normalize_rows=args.leverage_normalize_rows,
            )

    torch.cuda.synchronize()
    end_time_model = time.time()

    model_execution_time = end_time_model - start_time_model
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024**3)

    print("\n" + "="*50)
    print("INFERENCE PERFORMANCE")
    print(f"  Model Execution Time: {model_execution_time:.4f} seconds")
    print(f"  Peak GPU Memory Usage: {peak_memory_gb:.2f} GB")
    print("="*50 + "\n")

    if (not cache_results) or output.ress is None or len(output.ress) == 0:
        summary = {"per_frame_only": True}
        if args.frame_cache_dir:
            summary["frame_cache_dir"] = args.frame_cache_dir
        torch.cuda.empty_cache()
        return summary

    # Extract results from the output structure
    all_pts3d = [res['pts3d_in_other_view'].squeeze(0) for res in output.ress]
    all_conf = [res['conf'].squeeze(0) for res in output.ress]
    all_depth = [res['depth'].squeeze(0) for res in output.ress]
    all_depth_conf = [res['depth_conf'].squeeze(0) for res in output.ress]
    all_camera_pose = [res['camera_pose'].squeeze(0) for res in output.ress]

    # Create a dictionary to hold all prediction tensors
    predictions = {
        "world_points": torch.stack(all_pts3d, dim=0),
        "world_points_conf": torch.stack(all_conf, dim=0),
        "depth": torch.stack(all_depth, dim=0),
        "depth_conf": torch.stack(all_depth_conf, dim=0),
        "pose_enc": torch.stack(all_camera_pose, dim=0),
        "images": images
    }

    # Convert pose encoding to extrinsic and intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].unsqueeze(0),
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic.squeeze(0)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None

    # Clean up GPU cache
    torch.cuda.empty_cache()

    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions[key] = value.detach().cpu()

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run InfiniteVGGT inference from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/examples",
        help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../OVGGT/ckpt/checkpoints.pth",
        help="Path to the model checkpoint file (.pth)."
    )
    parser.add_argument(
        "--frame_cache_dir",
        type=str,
        default=None,
        help="Write the prediction for each frame to cache dir",
    )
    parser.add_argument(
        "--no_cache_results",
        action="store_true",
        help="Prediction results will not be accumulated in GPU memory",
    )
    parser.add_argument(
        "--stream_chunk_size",
        "--stream-chunk-size",
        type=int,
        default=1,
        help="Number of consecutive stream frames to process in one chunk-causal forward",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=2,
        help="Use every Nth frame from the sorted input image list",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=150,
        help="Maximum number of frames to process after applying frame_stride",
    )
    parser.add_argument(
        "--total_budget",
        "--total-budget",
        type=int,
        default=1200000,
        help="Total KV cache token budget distributed across streaming global attention layers",
    )
    parser.add_argument(
        "--cache_analysis_dir",
        type=str,
        default=None,
        help="Optional directory for per-head cache eviction analysis snapshots",
    )
    parser.add_argument(
        "--cache_analysis_layers",
        type=str,
        default="all",
        help="Layers to dump, e.g. '0,3,8-10' or 'all'",
    )
    parser.add_argument(
        "--cache_analysis_heads",
        type=str,
        default="all",
        help="Heads to dump, e.g. '0,4,12-15' or 'all'",
    )
    parser.add_argument(
        "--cache_analysis_steps",
        type=str,
        default="all",
        help="Streaming steps to dump, e.g. '10,20-25' or 'all'",
    )
    parser.add_argument(
        "--cache_analysis_max_snapshots",
        type=int,
        default=None,
        help="Optional global cap on the number of per-head snapshots written",
    )
    add_eviction_nn_analysis_args(parser)
    parser.add_argument(
        "--snapshot_before_eviction",
        "--snapshot-before-eviction",
        action="store_true",
        help="Dump a shared per-head KV cache snapshot before eviction at snapshot_frame_count",
    )
    parser.add_argument(
        "--snapshot_frame_count",
        "--snapshot-frame-count",
        type=int,
        default=40,
        help="Number of sequential frames to accumulate before dumping the common cache snapshot",
    )
    parser.add_argument(
        "--snapshot_output_dir",
        "--snapshot-output-dir",
        type=str,
        default=None,
        help="Directory for pre-eviction common cache snapshot .pt/.json files",
    )
    parser.add_argument(
        "--snapshot_layers",
        "--snapshot-layers",
        type=str,
        default="all",
        help="Layers to dump for pre-eviction snapshots, e.g. '0,3,8-10' or 'all'",
    )
    parser.add_argument(
        "--snapshot_heads",
        "--snapshot-heads",
        type=str,
        default="all",
        help="Heads to dump for pre-eviction snapshots, e.g. '0,4,12-15' or 'all'",
    )
    parser.add_argument(
        "--snapshot_max_snapshots",
        "--snapshot-max-snapshots",
        type=int,
        default=None,
        help="Optional global cap on pre-eviction per-head snapshots written",
    )
    parser.add_argument(
        "--eviction_policy",
        "--eviction-policy",
        type=str,
        default="mean",
        choices=("mean", "baseline_mean", "svd_leverage"),
        help="KV cache eviction policy for streaming global attention",
    )
    parser.add_argument(
        "--eviction_debug",
        "--eviction-debug",
        action="store_true",
        help="Print lightweight eviction policy shape/count diagnostics",
    )
    parser.add_argument(
        "--profile_eviction",
        "--profile-eviction",
        action="store_true",
        help="Print per-eviction svd_leverage timing/profile fields without changing eviction behavior",
    )
    parser.add_argument(
        "--leverage_sketch_dim",
        "--leverage-sketch-dim",
        type=int,
        default=16,
        help="Right sketch dimension for svd_leverage eviction; set 0 for exact full-space QR",
    )
    parser.add_argument(
        "--leverage_granularity",
        "--leverage-granularity",
        type=str,
        default="head",
        choices=("head", "layer"),
        help="Granularity for svd_leverage eviction: per-head or one shared layer-wise score vector",
    )
    parser.add_argument(
        "--leverage_feature",
        "--leverage-feature",
        type=str,
        default="key",
        choices=("key", "key_value"),
        help="Feature tensor for svd_leverage eviction: keys only or concatenated keys and values",
    )
    parser.add_argument(
        "--leverage_projection",
        "--leverage-projection",
        type=str,
        default="random",
        choices=("random", "head_mean"),
        help="Projection mode for svd_leverage eviction: random/full feature path or deterministic per-head means",
    )
    parser.add_argument(
        "--leverage_head_mean_dim",
        "--leverage-head-mean-dim",
        type=int,
        default=1,
        help="Number of mean-pooled channel groups per head for leverage_projection=head_mean",
    )
    parser.add_argument(
        "--leverage_normalize_before_projection",
        "--leverage-normalize-before-projection",
        action="store_true",
        help="L2 normalize layer-wise key rows before random leverage projection",
    )
    parser.add_argument(
        "--leverage_approx_method",
        "--leverage-approx-method",
        type=str,
        default="right_sketch",
        choices=("exact_qr", "right_sketch", "drineas_srht", "full_d_ridge", "right_sketch_ridge"),
        help="Leverage approximation: exact QR, right-sketched/Compactor-style, or Drineas left SRHT",
    )
    parser.add_argument(
        "--leverage_left_sketch_dim",
        "--leverage-left-sketch-dim",
        type=int,
        default=2048,
        help="Left SRHT row count r1 for leverage_approx_method=drineas_srht",
    )
    parser.add_argument(
        "--leverage_right_jl_dim",
        "--leverage-right-jl-dim",
        type=int,
        default=64,
        help="Right JL dimension r2 for leverage_approx_method=drineas_srht; <=0 uses no right JL",
    )
    parser.add_argument(
        "--leverage_ridge_lambda",
        "--leverage-ridge-lambda",
        type=float,
        default=1e-3,
        help="Ridge lambda for full_d_ridge and right_sketch_ridge leverage scoring",
    )
    parser.add_argument(
        "--leverage_ridge_lambda_mode",
        "--leverage-ridge-lambda-mode",
        type=str,
        default="relative",
        choices=("relative", "absolute"),
        help="Use absolute lambda or scale it by trace(X^T X) / D",
    )
    parser.add_argument(
        "--leverage_ridge_score_chunk_size",
        "--leverage-ridge-score-chunk-size",
        type=int,
        default=4096,
        help="Token chunk size for ridge leverage Cholesky solves",
    )
    parser.add_argument(
        "--leverage_ridge_jitter",
        "--leverage-ridge-jitter",
        type=float,
        default=1e-6,
        help="Relative diagonal jitter added to ridge systems before Cholesky",
    )
    parser.add_argument(
        "--leverage_ridge_dim",
        "--leverage-ridge-dim",
        type=int,
        default=None,
        help="Projection dimension for right_sketch_ridge; defaults to --leverage_right_jl_dim when omitted",
    )
    parser.add_argument(
        "--rls_refresh_interval",
        "--rls-refresh-interval",
        type=int,
        default=1,
        help="Refresh interval for ridge leverage K^T K and Cholesky factorization; 1 refreshes every frame",
    )
    parser.add_argument(
        "--leverage_diag",
        "--leverage-diag",
        action="store_true",
        help="Print ridge leverage diagnostic statistics at selected eviction steps",
    )
    parser.add_argument(
        "--leverage_diag_interval",
        "--leverage-diag-interval",
        type=int,
        default=0,
        help="Diagnostic interval; 0 prints only the first eviction step, positive values print every N steps",
    )
    parser.add_argument(
        "--leverage_random_seed",
        "--leverage-random-seed",
        type=int,
        default=0,
        help="Random seed for leverage sketches",
    )
    parser.add_argument(
        "--leverage_eviction_selector",
        "--leverage-eviction-selector",
        type=str,
        default="topk",
        choices=("topk", "fast_dpp", "layer_head_fast_dpp"),
        help="Eviction selector for svd_leverage scores: topk, shared Fast DPP, or GPU head-wise Fast DPP with layer scores",
    )
    parser.add_argument(
        "--leverage_dpp_candidate_multiplier",
        "--leverage-dpp-candidate-multiplier",
        type=int,
        default=2,
        help="Fast DPP low-score candidate pool multiplier relative to eviction count",
    )
    parser.add_argument(
        "--leverage_dpp_greedy_block_size",
        "--leverage-dpp-greedy-block-size",
        type=int,
        default=32,
        help="Fast DPP greedy selection block size; 1 is quality-oriented, larger values favor speed",
    )
    parser.add_argument(
        "--leverage_dpp_quality_beta",
        "--leverage-dpp-quality-beta",
        type=float,
        default=1.0,
        help="Fast DPP quality log-term weight; 0 removes quality from DPP selection",
    )
    parser.add_argument("--leverage_dpp_recency_bonus", "--leverage-dpp-recency-bonus", action="store_true", help="Add a weak recency quality prior inside Fast DPP retain selection")
    parser.add_argument("--leverage_dpp_recency_lambda", "--leverage-dpp-recency-lambda", type=float, default=0.2, help="Strength of the Fast DPP recency quality prior")
    parser.add_argument("--leverage_dpp_recency_window", "--leverage-dpp-recency-window", type=int, default=5, help="Frame window for linear Fast DPP freshness")
    parser.add_argument("--leverage_dpp_recency_gate_power", "--leverage-dpp-recency-gate-power", type=float, default=1.0, help="Power applied to the low-score gate for Fast DPP recency")
    parser.add_argument("--leverage_dpp_recency_debug", "--leverage-dpp-recency-debug", action="store_true", help="Print Fast DPP recency prior summary statistics")
    parser.add_argument(
        "--global_attn_idx_ranges",
        "--global-attn-idx-ranges",
        type=str,
        default=None,
        help="Half-open global attention index ranges to keep global, e.g. '9:', '9:20', '6:10,14:20'",
    )
    parser.add_argument(
        "--middle_global_only",
        "--middle-global-only",
        action="store_true",
        help="Shortcut for --global-attn-idx-ranges 9:",
    )
    parser.add_argument(
        "--global_attn_debug",
        "--global-attn-debug",
        action="store_true",
        help="Print per-block global-to-frame and KV cache decisions",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./inference_results",
        help="Path to the directory containing the complete results"
    )
    parser.add_argument(
        "--leverage_normalize_rows",
        "--leverage-normalize-rows",
        action="store_true",
        help="L2 normalize leverage score rows before SVD eviction candidate generation",
    )

    args = parser.parse_args()
    result = run_inference(args)

    if result is None:
        print("Inference aborted due to previous errors.")
    elif result.get("per_frame_only", False):
        cache_dir = result.get("frame_cache_dir", args.frame_cache_dir)
        if cache_dir:
            print(f"Inference finished. Per-frame outputs saved under {cache_dir}.")
        else:
            print("Inference finished. Per-frame outputs were written via custom frame_writer.")
    else:
        torch.save(result, args.output_path)
        print(f"Inference finished. Results saved to {args.output_path}")
