import os
import math
import torch
import numpy as np
import sys
import time
import argparse
from typing import List, Dict

# Add project source to the Python path
sys.path.append("src/")

# Import necessary components from the StreamVGGT project
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import FrameDiskCache
from streamvggt.utils.inference_export import (
    MV_RECON_DEFAULTS,
    discover_images,
    export_inference_artifacts,
    resolve_output_directory,
    select_images,
)
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
    if args.no_cache_results and not args.frame_cache_dir:
        print("Error: --no_cache_results requires --frame_cache_dir for per-frame output.")
        return
    if not math.isfinite(args.conf_thresh):
        print(f"Error: --conf_thresh must be finite, got {args.conf_thresh}.")
        return
    if args.point_stride < 1:
        print(f"Error: --point_stride must be >= 1, got {args.point_stride}.")
        return
    if not math.isfinite(args.voxel_size) or args.voxel_size < 0:
        print(f"Error: --voxel_size must be finite and >= 0, got {args.voxel_size}.")
        return
    if args.camera_size is not None and (
        not math.isfinite(args.camera_size) or args.camera_size <= 0
    ):
        print(f"Error: --camera_size must be finite and > 0 when provided, got {args.camera_size}.")
        return
    if args.camera_thickness is not None and (
        not math.isfinite(args.camera_thickness) or args.camera_thickness <= 0
    ):
        print(
            "Error: --camera_thickness must be finite and > 0 when provided, "
            f"got {args.camera_thickness}."
        )
        return
    if args.camera_stride < 0:
        print(f"Error: --camera_stride must be >= 0, got {args.camera_stride}.")
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
    if args.layer_budget_score_only:
        if args.eviction_policy != "svd_leverage" or args.leverage_granularity != "layer":
            print(
                "Error: --layer-budget-score-only requires "
                "--eviction-policy svd_leverage and --leverage-granularity layer."
            )
            return
        if args.layer_budget_strategy != "value_weighted_leverage_pr":
            print(
                "Error: --layer-budget-score-only requires "
                "--layer-budget-strategy value_weighted_leverage_pr."
            )
            return
        incompatible_outputs = []
        if cache_analysis_config is not None:
            incompatible_outputs.append("--cache_analysis_dir")
        if eviction_nn_analysis_config is not None:
            incompatible_outputs.append("--eviction_nn_analysis_dir")
        if incompatible_outputs:
            print(
                "Error: --layer-budget-score-only does not produce kept/evicted indices and "
                f"cannot be combined with: {', '.join(incompatible_outputs)}."
            )
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
    if args.leverage_ridge_jitter < 0:
        print(
            "Error: --leverage_ridge_jitter must be >= 0, "
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

    print(f"Using total KV cache budget: {args.total_budget}")
    print(
        "Using mv_recon layer-budget settings: "
        f"conf_gate={args.leverage_conf_gate}, "
        f"projected_key_cache={args.leverage_projected_key_cache}, "
        f"layer_budget={args.layer_budget_strategy}, "
        f"score_only={args.layer_budget_score_only}, "
        f"alpha={args.layer_budget_alpha}, value_gamma={args.layer_budget_value_gamma}"
    )
    model = StreamVGGT(total_budget=args.total_budget)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    del ckpt
    print("Model loaded successfully onto the GPU.")

    print(f"Loading images from input directory: {args.input_dir}")
    try:
        discovered_images = discover_images(args.input_dir)
        image_names = select_images(
            discovered_images,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if not discovered_images:
        print(
            f"Error: No supported images found in {args.input_dir}. "
            "Supported extensions: png, jpg, jpeg, webp, bmp, tif, tiff."
        )
        return
    original_num_images = len(discovered_images)

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
    image_names = [str(path) for path in image_names]
    # StreamVGGT moves each stream chunk to the model device internally. Keeping
    # the full preprocessed sequence on CPU avoids a sequence-length-sized GPU
    # allocation and also keeps RGB colors ready for PLY export.
    images = load_and_preprocess_images(image_names)
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
        with torch.amp.autocast("cuda", dtype=dtype):
            output = model.inference(
                frames,
                frame_writer=frame_writer,
                cache_results=cache_results,
                stream_chunk_size=args.stream_chunk_size,
                cache_analysis_config=cache_analysis_config,
                pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                eviction_nn_analysis_config=eviction_nn_analysis_config,
                eviction_policy=args.eviction_policy,
                profile_eviction=args.profile_eviction,
                eviction_debug=args.eviction_debug,
                leverage_sketch_dim=args.leverage_sketch_dim,
                leverage_granularity=args.leverage_granularity,
                leverage_feature=args.leverage_feature,
                leverage_projection=args.leverage_projection,
                leverage_head_mean_dim=args.leverage_head_mean_dim,
                leverage_normalize_rows=args.leverage_normalize_rows,
                leverage_normalize_before_projection=args.leverage_normalize_before_projection,
                leverage_normalize_before_projection_headwise=args.leverage_normalize_before_projection_headwise,
                leverage_projected_key_cache=args.leverage_projected_key_cache,
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
                leverage_conf_gate=args.leverage_conf_gate,
                leverage_conf_gate_floor=args.leverage_conf_gate_floor,
                leverage_conf_gate_depth_alpha=args.leverage_conf_gate_depth_alpha,
                leverage_conf_gate_point_beta=args.leverage_conf_gate_point_beta,
                leverage_conf_gate_k=args.leverage_conf_gate_k,
                leverage_conf_gate_transform=args.leverage_conf_gate_transform,
                leverage_conf_gate_init=args.leverage_conf_gate_init,
                leverage_conf_gate_special_mode=args.leverage_conf_gate_special_mode,
                layer_budget_strategy=args.layer_budget_strategy,
                layer_budget_value_gamma=args.layer_budget_value_gamma,
                layer_budget_value_norm_type=args.layer_budget_value_norm_type,
                layer_budget_norm_source=args.layer_budget_norm_source,
                layer_budget_alpha=args.layer_budget_alpha,
                layer_budget_min_tokens=args.layer_budget_min_tokens,
                layer_budget_eps=args.layer_budget_eps,
                layer_budget_score_only=args.layer_budget_score_only,
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
        "images": images,
        "image_paths": image_names,
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
        default=os.path.join(PROJECT_ROOT, "examples"),
        help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "ckpt", "checkpoints.pth"),
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
        help="Write only per-frame cache results; combined PLY/depth/pose exports are skipped",
    )
    parser.add_argument(
        "--stream_chunk_size",
        "--stream-chunk-size",
        type=int,
        default=1,
        help="Frames per streaming chunk; chunks attend causally to past chunks while frames inside a chunk attend bidirectionally",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=MV_RECON_DEFAULTS["frame_stride"],
        help="Use every Nth frame from the naturally sorted input image list",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=MV_RECON_DEFAULTS["max_frames"],
        help="Maximum number of frames after frame_stride; omitted processes the full sequence",
    )
    parser.add_argument(
        "--total_budget",
        "--total-budget",
        type=int,
        default=MV_RECON_DEFAULTS["total_budget"],
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
        default=MV_RECON_DEFAULTS["eviction_policy"],
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
        default=MV_RECON_DEFAULTS["leverage_sketch_dim"],
        help="Right sketch dimension for svd_leverage eviction; set 0 for exact full-space QR",
    )
    parser.add_argument(
        "--leverage_granularity",
        "--leverage-granularity",
        type=str,
        default=MV_RECON_DEFAULTS["leverage_granularity"],
        choices=("head", "layer"),
        help="Granularity for svd_leverage eviction: per-head or one shared layer-wise score vector",
    )
    parser.add_argument(
        "--leverage_feature",
        "--leverage-feature",
        type=str,
        default=MV_RECON_DEFAULTS["leverage_feature"],
        choices=("key", "key_value"),
        help="Feature tensor for svd_leverage eviction: keys only or concatenated keys and values",
    )
    parser.add_argument(
        "--leverage_projection",
        "--leverage-projection",
        type=str,
        default=MV_RECON_DEFAULTS["leverage_projection"],
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
        action=argparse.BooleanOptionalAction,
        default=MV_RECON_DEFAULTS["leverage_normalize_before_projection"],
        help="L2 normalize layer-wise key rows before random leverage projection",
    )
    parser.add_argument(
        "--leverage_normalize_before_projection_headwise",
        "--leverage-normalize-before-projection-headwise",
        action=argparse.BooleanOptionalAction,
        default=MV_RECON_DEFAULTS["leverage_normalize_before_projection_headwise"],
        help="Normalize each head independently before layer-wise random projection",
    )
    parser.add_argument(
        "--leverage_projected_key_cache",
        "--leverage-projected-key-cache",
        action=argparse.BooleanOptionalAction,
        default=MV_RECON_DEFAULTS["leverage_projected_key_cache"],
        help="Reuse projected key features for retained tokens",
    )
    parser.add_argument(
        "--leverage_conf_gate",
        "--leverage-conf-gate",
        action=argparse.BooleanOptionalAction,
        default=MV_RECON_DEFAULTS["leverage_conf_gate"],
        help="Apply transformed depth/world-point confidence to leverage keep scores",
    )
    parser.add_argument("--leverage_conf_gate_floor", "--leverage-conf-gate-floor", type=float, default=MV_RECON_DEFAULTS["leverage_conf_gate_floor"])
    parser.add_argument("--leverage_conf_gate_depth_alpha", "--leverage-conf-gate-depth-alpha", type=float, default=MV_RECON_DEFAULTS["leverage_conf_gate_depth_alpha"])
    parser.add_argument("--leverage_conf_gate_point_beta", "--leverage-conf-gate-point-beta", type=float, default=MV_RECON_DEFAULTS["leverage_conf_gate_point_beta"])
    parser.add_argument("--leverage_conf_gate_k", "--leverage-conf-gate-k", type=float, default=MV_RECON_DEFAULTS["leverage_conf_gate_k"])
    parser.add_argument("--leverage_conf_gate_transform", "--leverage-conf-gate-transform", choices=("ratio", "sigmoid"), default=MV_RECON_DEFAULTS["leverage_conf_gate_transform"])
    parser.add_argument("--leverage_conf_gate_init", "--leverage-conf-gate-init", default=MV_RECON_DEFAULTS["leverage_conf_gate_init"])
    parser.add_argument("--leverage_conf_gate_special_mode", "--leverage-conf-gate-special-mode", choices=("mean", "one"), default=MV_RECON_DEFAULTS["leverage_conf_gate_special_mode"])
    parser.add_argument(
        "--layer_budget_strategy",
        "--layer-budget-strategy",
        choices=("uniform", "leverage_pr", "key_norm", "value_weighted_leverage_pr"),
        default=MV_RECON_DEFAULTS["layer_budget_strategy"],
    )
    parser.add_argument(
        "--layer_budget_score_only",
        "--layer-budget-score-only",
        action=argparse.BooleanOptionalAction,
        default=MV_RECON_DEFAULTS["layer_budget_score_only"],
        help="Compute value_weighted_leverage_pr every frame without removing KV cache tokens",
    )
    parser.add_argument("--layer_budget_alpha", "--layer-budget-alpha", type=float, default=MV_RECON_DEFAULTS["layer_budget_alpha"])
    parser.add_argument("--layer_budget_min_tokens", "--layer-budget-min-tokens", type=int, default=MV_RECON_DEFAULTS["layer_budget_min_tokens"])
    parser.add_argument("--layer_budget_eps", "--layer-budget-eps", type=float, default=MV_RECON_DEFAULTS["layer_budget_eps"])
    parser.add_argument("--layer_budget_value_gamma", "--layer-budget-value-gamma", type=float, default=MV_RECON_DEFAULTS["layer_budget_value_gamma"])
    parser.add_argument("--layer_budget_value_norm_type", "--layer-budget-value-norm-type", choices=("mean", "rms"), default=MV_RECON_DEFAULTS["layer_budget_value_norm_type"])
    parser.add_argument("--layer_budget_norm_source", "--layer-budget-norm-source", choices=("key", "value"), default=MV_RECON_DEFAULTS["layer_budget_norm_source"])
    parser.add_argument(
        "--leverage_approx_method",
        "--leverage-approx-method",
        type=str,
        default=MV_RECON_DEFAULTS["leverage_approx_method"],
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
        default=MV_RECON_DEFAULTS["leverage_ridge_lambda"],
        help="Ridge lambda for full_d_ridge and right_sketch_ridge leverage scoring",
    )
    parser.add_argument(
        "--leverage_ridge_lambda_mode",
        "--leverage-ridge-lambda-mode",
        type=str,
        default=MV_RECON_DEFAULTS["leverage_ridge_lambda_mode"],
        choices=("relative", "absolute"),
        help="Use absolute lambda or scale it by trace(X^T X) / D",
    )
    parser.add_argument(
        "--leverage_ridge_score_chunk_size",
        "--leverage-ridge-score-chunk-size",
        type=int,
        default=MV_RECON_DEFAULTS["leverage_ridge_score_chunk_size"],
        help="Token chunk size for ridge leverage Cholesky solves",
    )
    parser.add_argument(
        "--leverage_ridge_jitter",
        "--leverage-ridge-jitter",
        type=float,
        default=MV_RECON_DEFAULTS["leverage_ridge_jitter"],
        help="Diagonal jitter added to ridge systems before Cholesky; zero matches mv_recon run_final",
    )
    parser.add_argument(
        "--leverage_ridge_dim",
        "--leverage-ridge-dim",
        type=int,
        default=MV_RECON_DEFAULTS["leverage_ridge_dim"],
        help="Projection dimension for right_sketch_ridge",
    )
    parser.add_argument(
        "--rls_refresh_interval",
        "--rls-refresh-interval",
        type=int,
        default=MV_RECON_DEFAULTS["rls_refresh_interval"],
        help="Refresh interval for ridge leverage K^T K and Cholesky factorization",
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
        default=MV_RECON_DEFAULTS["leverage_random_seed"],
        help="Random seed for leverage sketches",
    )
    parser.add_argument(
        "--leverage_eviction_selector",
        "--leverage-eviction-selector",
        type=str,
        default=MV_RECON_DEFAULTS["leverage_eviction_selector"],
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
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory for reconstruction PLY, cameras, poses, depth images, and manifest",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional legacy path for saving the complete raw torch prediction dictionary",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.0, help="Minimum direct point-head confidence written to reconstruction.ply")
    parser.add_argument("--point_stride", type=int, default=1, help="Spatial pixel stride used while writing reconstruction.ply")
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Optional global voxel downsample size in model scene units; zero disables it")
    parser.add_argument("--camera_size", type=float, default=None, help="Camera frustum depth in model scene units; default is 2%% of reconstruction diagonal")
    parser.add_argument("--camera_thickness", type=float, default=None, help="Wireframe tube radius; default is 1.5%% of camera_size")
    parser.add_argument("--camera_stride", type=int, default=0, help="Draw every Nth camera frustum; zero auto-limits display to about 50 cameras")
    parser.add_argument(
        "--camera_axes",
        "--camera-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw RGB local coordinate axes on every displayed camera",
    )
    parser.add_argument(
        "--camera_trajectory",
        "--camera-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Connect all camera centers with a light trajectory tube",
    )
    parser.add_argument(
        "--leverage_normalize_rows",
        "--leverage-normalize-rows",
        action="store_true",
        help="L2 normalize leverage score rows before SVD eviction candidate generation",
    )

    args = parser.parse_args()
    resolved_output_dir = resolve_output_directory(args.output_dir)
    if str(resolved_output_dir) != str(args.output_dir):
        print(
            f"Output path {args.output_dir!r} is an existing file; preserving it and "
            f"writing artifacts to {str(resolved_output_dir)!r} instead."
        )
        args.output_dir = str(resolved_output_dir)
    result = run_inference(args)

    if result is None:
        print("Inference aborted due to previous errors.")
    elif result.get("per_frame_only", False):
        cache_dir = result.get("frame_cache_dir", args.frame_cache_dir)
        if cache_dir:
            print(f"Inference finished. Per-frame outputs saved under {cache_dir}.")
            print("Combined PLY/depth/pose artifacts were skipped because --no_cache_results was used.")
        else:
            print("Inference finished. Per-frame outputs were written via custom frame_writer.")
    else:
        numpy_predictions = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
            if key != "image_paths"
        }
        inference_config = {
            key: getattr(args, key)
            for key in MV_RECON_DEFAULTS
            if hasattr(args, key)
        }
        inference_config["stream_chunk_size"] = args.stream_chunk_size
        try:
            manifest = export_inference_artifacts(
                numpy_predictions,
                result["image_paths"],
                args.output_dir,
                conf_thresh=args.conf_thresh,
                point_stride=args.point_stride,
                voxel_size=args.voxel_size,
                camera_size=args.camera_size,
                camera_thickness=args.camera_thickness,
                camera_stride=args.camera_stride,
                camera_axes=args.camera_axes,
                camera_trajectory=args.camera_trajectory,
                inference_config=inference_config,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Failed to export inference artifacts: {exc}") from exc
        if args.output_path:
            torch.save(result, args.output_path)
            print(f"Raw predictions saved to {args.output_path}")
        print(
            f"Inference finished. Artifacts saved to {args.output_dir} "
            f"({manifest['reconstruction']['vertex_count']} reconstruction vertices)."
        )
