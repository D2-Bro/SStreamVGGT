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
from streamvggt.utils.cache_analysis import CacheAnalysisConfig, PreEvictionSnapshotConfig
from streamvggt.layers.recent_merge import RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)


def resolve_global_attn_idx_ranges(args: argparse.Namespace) -> Optional[str]:
    if args.middle_global_only and args.global_attn_idx_ranges is not None:
        raise ValueError("--middle-global-only cannot be combined with --global-attn-idx-ranges")
    if args.middle_global_only:
        return "9:"
    return args.global_attn_idx_ranges


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
    if args.merge_window < 1:
        print(f"Error: --merge_window must be >= 1, got {args.merge_window}.")
        return
    if not (0.0 <= args.merge_similarity_threshold <= 1.0):
        print(
            "Error: --merge_similarity_threshold must be in [0, 1], "
            f"got {args.merge_similarity_threshold}."
        )
        return
    if args.merge_voxel_size <= 0:
        print(f"Error: --merge_voxel_size must be > 0, got {args.merge_voxel_size}.")
        return
    if args.merge_chunk_size < 1:
        print(f"Error: --merge_chunk_size must be >= 1, got {args.merge_chunk_size}.")
        return
    if args.merge_patch_radius < 0:
        print(f"Error: --merge_patch_radius must be >= 0, got {args.merge_patch_radius}.")
        return
    if args.merge_voxel_neighbor_radius < 0:
        print(
            "Error: --merge_voxel_neighbor_radius must be >= 0, "
            f"got {args.merge_voxel_neighbor_radius}."
        )
        return
    if args.merge_max_candidates_per_token < 1:
        print(
            "Error: --merge_max_candidates_per_token must be >= 1, "
            f"got {args.merge_max_candidates_per_token}."
        )
        return
    if args.merge_recall_debug_max_tokens < 1:
        print(
            "Error: --merge_recall_debug_max_tokens must be >= 1, "
            f"got {args.merge_recall_debug_max_tokens}."
        )
        return
    if args.eviction_protect_recent_frames < 0:
        print(
            "Error: --eviction_protect_recent_frames must be >= 0, "
            f"got {args.eviction_protect_recent_frames}."
        )
        return
    if args.svd_eviction_merge_candidate_axes < 1:
        print("Error: --svd_eviction_merge_candidate_axes must be >= 1.")
        return
    if args.svd_eviction_merge_reps_per_axis < 1:
        print("Error: --svd_eviction_merge_reps_per_axis must be >= 1.")
        return
    if not (0.0 <= args.svd_eviction_merge_similarity_threshold <= 1.0):
        print("Error: --svd_eviction_merge_similarity_threshold must be in [0, 1].")
        return
    if args.svd_eviction_merge_voxel_neighbor_radius < 0:
        print("Error: --svd_eviction_merge_voxel_neighbor_radius must be >= 0.")
        return
    if not (0.0 <= args.svd_eviction_merge_ema_decay <= 1.0):
        print("Error: --svd_eviction_merge_ema_decay must be in [0, 1].")
        return
    if args.svd_eviction_merge_max_candidates_per_token < 1:
        print("Error: --svd_eviction_merge_max_candidates_per_token must be >= 1.")
        return
    if args.svd_eviction_merge_chunk_size < 1:
        print("Error: --svd_eviction_merge_chunk_size must be >= 1.")
        return
    if args.voxel_size <= 0:
        print(f"Error: --voxel_size must be > 0, got {args.voxel_size}.")
        return
    if args.covis_min_shared_voxels < 0:
        print(
            "Error: --covis_min_shared_voxels must be >= 0, "
            f"got {args.covis_min_shared_voxels}."
        )
        return
    if not (0.0 <= args.covis_min_overlap <= 1.0):
        print(
            "Error: --covis_min_overlap must be in [0, 1], "
            f"got {args.covis_min_overlap}."
        )
        return
    if args.covis_fallback_recent < 0:
        print(
            "Error: --covis_fallback_recent must be >= 0, "
            f"got {args.covis_fallback_recent}."
        )
        return
    try:
        global_attn_idx_ranges = resolve_global_attn_idx_ranges(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    print(f"Using eviction policy: {args.eviction_policy}")
    if args.eviction_policy == "svd_leverage":
        sketch_label = "exact" if args.leverage_sketch_dim == 0 else str(args.leverage_sketch_dim)
        print(f"Using SVD leverage sketch dim: {sketch_label}")
        print(
            "Using SVD leverage granularity: "
            f"{args.leverage_granularity} (feature={args.leverage_feature})"
        )
        print(f"Using SVD leverage recent-frame protection: {args.eviction_protect_recent_frames}")
    recent_merge_config = RecentMergeConfig(
        enabled=args.enable_recent_merge,
        window=args.merge_window,
        similarity_threshold=args.merge_similarity_threshold,
        voxel_size=args.merge_voxel_size,
        use_depth_confidence=args.merge_use_depth_confidence,
        debug=args.merge_debug,
        chunk_size=args.merge_chunk_size,
        disable_geometry_check=args.merge_disable_geometry_check,
        candidate_mode=args.merge_candidate_mode,
        patch_radius=args.merge_patch_radius,
        voxel_neighbor_radius=args.merge_voxel_neighbor_radius,
        max_candidates_per_token=args.merge_max_candidates_per_token,
        local_fallback=args.merge_local_fallback,
        profile=args.merge_profile,
        recall_debug=args.merge_recall_debug,
        recall_debug_max_tokens=args.merge_recall_debug_max_tokens,
    )
    if recent_merge_config.enabled:
        print(
            "Recent similarity merge enabled: "
            f"mode={recent_merge_config.candidate_mode}, "
            f"window={recent_merge_config.window}, "
            f"threshold={recent_merge_config.similarity_threshold}, "
            f"voxel_size={recent_merge_config.voxel_size}"
        )
    svd_eviction_merge_config = SvdEvictionMergeConfig(
        enabled=args.enable_svd_eviction_merge,
        mode=args.svd_eviction_merge_mode,
        candidate_axes=args.svd_eviction_merge_candidate_axes,
        reps_per_axis=args.svd_eviction_merge_reps_per_axis,
        similarity_threshold=args.svd_eviction_merge_similarity_threshold,
        use_u_sigma=args.svd_eviction_merge_use_u_sigma,
        geometry_gate=args.svd_eviction_merge_geometry_gate,
        voxel_neighbor_radius=args.svd_eviction_merge_voxel_neighbor_radius,
        allow_missing_geometry=args.svd_eviction_merge_allow_missing_geometry,
        ema_decay=args.svd_eviction_merge_ema_decay,
        use_depth_confidence=args.svd_eviction_merge_use_depth_confidence,
        max_candidates_per_token=args.svd_eviction_merge_max_candidates_per_token,
        chunk_size=args.svd_eviction_merge_chunk_size,
        debug=args.svd_eviction_merge_debug,
        profile=args.svd_eviction_merge_profile,
    )
    if svd_eviction_merge_config.enabled:
        print(
            "SVD eviction merge enabled: "
            f"mode={svd_eviction_merge_config.mode}, "
            f"axes={svd_eviction_merge_config.candidate_axes}, "
            f"reps_per_axis={svd_eviction_merge_config.reps_per_axis}, "
            f"threshold={svd_eviction_merge_config.similarity_threshold}, "
            f"geometry_gate={svd_eviction_merge_config.geometry_gate}"
        )
    voxel_covis_config = VoxelCovisConfig(
        enabled=args.use_voxel_covis,
        voxel_size=args.voxel_size,
        min_shared_voxels=args.covis_min_shared_voxels,
        min_overlap=args.covis_min_overlap,
        max_covis_frames=args.max_covis_frames,
        fallback_recent=args.covis_fallback_recent,
        debug=args.covis_debug_log,
    )
    if voxel_covis_config.enabled:
        print(
            "Voxel covisibility KV filtering enabled: "
            f"voxel_size={voxel_covis_config.voxel_size}, "
            f"min_shared={voxel_covis_config.min_shared_voxels}, "
            f"min_overlap={voxel_covis_config.min_overlap}, "
            f"max_frames={voxel_covis_config.max_covis_frames}, "
            f"fallback_recent={voxel_covis_config.fallback_recent}"
        )
    if global_attn_idx_ranges is not None:
        print(f"Global attention index ranges enabled: {global_attn_idx_ranges}")

    model = StreamVGGT(total_budget=1200000)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")

    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    del ckpt
    print("Model loaded successfully onto the GPU.")

    print(f"Loading images from input directory: {args.input_dir}")
    image_names = sorted(glob.glob(os.path.join(args.input_dir, "*color.*")))
    
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
                cache_analysis_config=cache_analysis_config,
                pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                eviction_policy=args.eviction_policy,
                eviction_debug=args.eviction_debug,
                leverage_sketch_dim=args.leverage_sketch_dim,
                leverage_granularity=args.leverage_granularity,
                leverage_feature=args.leverage_feature,
                eviction_protect_recent_frames=args.eviction_protect_recent_frames,
                recent_merge_config=recent_merge_config,
                svd_eviction_merge_config=svd_eviction_merge_config,
                voxel_covis_config=voxel_covis_config,
                global_attn_idx_ranges=global_attn_idx_ranges,
                global_attn_debug=args.global_attn_debug,
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
        "--eviction_protect_recent_frames",
        "--eviction-protect-recent-frames",
        type=int,
        default=0,
        help=(
            "Protect tokens from the most recent N processed frames from eviction while still "
            "including them in SVD leverage computation."
        ),
    )
    parser.add_argument(
        "--enable_svd_eviction_merge",
        "--enable-svd-eviction-merge",
        action="store_true",
        help="Enable feature-first SVD-guided merge for tokens selected by svd_leverage eviction",
    )
    parser.add_argument(
        "--svd_eviction_merge_mode",
        "--svd-eviction-merge-mode",
        choices=("head", "layer_candidates", "layer"),
        default="head",
        help="SVD eviction merge mode: per-head, shared layer candidates, or shared layer merge pairs",
    )
    parser.add_argument(
        "--svd_eviction_merge_candidate_axes",
        "--svd-eviction-merge-candidate-axes",
        type=int,
        default=2,
        help="Number of low-rank leverage axes used for SVD eviction merge candidate generation",
    )
    parser.add_argument(
        "--svd_eviction_merge_reps_per_axis",
        "--svd-eviction-merge-reps-per-axis",
        type=int,
        default=8,
        help="Nearest retained representatives sampled per leverage axis",
    )
    parser.add_argument(
        "--svd_eviction_merge_similarity_threshold",
        "--svd-eviction-merge-similarity-threshold",
        type=float,
        default=0.9,
        help="Minimum full-key cosine similarity for SVD eviction merge",
    )
    parser.add_argument(
        "--svd_eviction_merge_use_u_sigma",
        "--svd-eviction-merge-use-u-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale leverage coordinates by the QR diagonal proxy before candidate generation",
    )
    parser.add_argument(
        "--svd_eviction_merge_geometry_gate",
        "--svd-eviction-merge-geometry-gate",
        choices=("none", "voxel_neighbor"),
        default="voxel_neighbor",
        help="Optional soft geometry gate for SVD eviction merge",
    )
    parser.add_argument(
        "--svd_eviction_merge_voxel_neighbor_radius",
        "--svd-eviction-merge-voxel-neighbor-radius",
        type=int,
        default=1,
        help="Chebyshev voxel radius accepted by SVD eviction merge geometry gate",
    )
    parser.add_argument(
        "--svd_eviction_merge_allow_missing_geometry",
        "--svd-eviction-merge-allow-missing-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow feature-only SVD eviction merge when voxel metadata is unavailable",
    )
    parser.add_argument(
        "--svd_eviction_merge_ema_decay",
        "--svd-eviction-merge-ema-decay",
        type=float,
        default=0.5,
        help="EMA retained-token weight when confidence weighting is disabled",
    )
    parser.add_argument(
        "--svd_eviction_merge_use_depth_confidence",
        "--svd-eviction-merge-use-depth-confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use accumulated depth confidence to weight SVD eviction merges",
    )
    parser.add_argument(
        "--svd_eviction_merge_max_candidates_per_token",
        "--svd-eviction-merge-max-candidates-per-token",
        type=int,
        default=32,
        help="Maximum retained candidates checked per evicted token",
    )
    parser.add_argument(
        "--svd_eviction_merge_chunk_size",
        "--svd-eviction-merge-chunk-size",
        type=int,
        default=512,
        help="Evicted-token chunk size for SVD eviction merge",
    )
    parser.add_argument(
        "--svd_eviction_merge_debug",
        "--svd-eviction-merge-debug",
        action="store_true",
        help="Print per-layer SVD eviction merge diagnostics",
    )
    parser.add_argument(
        "--svd_eviction_merge_profile",
        "--svd-eviction-merge-profile",
        action="store_true",
        help="Print SVD eviction merge profiling timings",
    )
    parser.add_argument(
        "--enable_recent_merge",
        "--enable-recent-merge",
        action="store_true",
        help="Enable sliding-window geometry-validated KV similarity merging",
    )
    parser.add_argument(
        "--merge_window",
        "--merge-window",
        type=int,
        default=3,
        help="Number of previous frames considered by recent KV merging",
    )
    parser.add_argument(
        "--merge_similarity_threshold",
        "--merge-similarity-threshold",
        type=float,
        default=0.9,
        help="Minimum cosine similarity for recent KV merge candidates",
    )
    parser.add_argument(
        "--merge_voxel_size",
        "--merge-voxel-size",
        type=float,
        default=0.05,
        help="Voxel size in world units for recent merge geometry validation",
    )
    parser.add_argument(
        "--merge_use_depth_confidence",
        "--merge-use-depth-confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use depth confidence to weight recent KV EMA merges",
    )
    parser.add_argument(
        "--merge_debug",
        "--merge-debug",
        action="store_true",
        help="Print per-layer recent merge diagnostics",
    )
    parser.add_argument(
        "--merge_chunk_size",
        "--merge-chunk-size",
        type=int,
        default=512,
        help="Current-token chunk size for batched recent merge cosine search",
    )
    parser.add_argument(
        "--merge_disable_geometry_check",
        "--merge-disable-geometry-check",
        action="store_true",
        help="Disable voxel validation for ablations; geometry check is enabled by default",
    )
    parser.add_argument(
        "--merge_candidate_mode",
        "--merge-candidate-mode",
        choices=("full", "spatial", "voxel", "voxel_spatial"),
        default="full",
        help="Candidate search mode for recent merge",
    )
    parser.add_argument(
        "--merge_patch_radius",
        "--merge-patch-radius",
        type=int,
        default=1,
        help="Patch-grid radius for local spatial recent merge candidate search",
    )
    parser.add_argument(
        "--merge_voxel_neighbor_radius",
        "--merge-voxel-neighbor-radius",
        type=int,
        default=0,
        help="Chebyshev voxel neighbor radius for local voxel recent merge candidates",
    )
    parser.add_argument(
        "--merge_max_candidates_per_token",
        "--merge-max-candidates-per-token",
        type=int,
        default=64,
        help="Maximum local recent merge candidates retained per current token",
    )
    parser.add_argument(
        "--merge_local_fallback",
        "--merge-local-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow local candidate modes to fall back to weaker local candidates",
    )
    parser.add_argument(
        "--merge_profile",
        "--merge-profile",
        action="store_true",
        help="Print recent merge profiling timings",
    )
    parser.add_argument(
        "--merge_recall_debug",
        "--merge-recall-debug",
        action="store_true",
        help="Compare local recent merge candidates against full-window candidates for diagnostics",
    )
    parser.add_argument(
        "--merge_recall_debug_max_tokens",
        "--merge-recall-debug-max-tokens",
        type=int,
        default=1024,
        help="Maximum source tokens sampled per layer/head for recent merge recall diagnostics",
    )
    parser.add_argument(
        "--use_voxel_covis",
        "--use-voxel-covis",
        action="store_true",
        help="Enable read-only voxel covisibility filtering for streaming KV cache reads",
    )
    parser.add_argument(
        "--voxel_size",
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size in world units for covisibility frame selection",
    )
    parser.add_argument(
        "--covis_min_shared_voxels",
        "--covis-min-shared-voxels",
        type=int,
        default=20,
        help="Minimum shared voxels required for a covisible frame",
    )
    parser.add_argument(
        "--covis_min_overlap",
        "--covis-min-overlap",
        type=float,
        default=0.05,
        help="Minimum shared/min voxel overlap required for a covisible frame",
    )
    parser.add_argument(
        "--max_covis_frames",
        "--max-covis-frames",
        type=int,
        default=8,
        help="Maximum number of covisible previous frames to read from KV cache; <=0 disables the cap",
    )
    parser.add_argument(
        "--covis_fallback_recent",
        "--covis-fallback-recent",
        type=int,
        default=1,
        help="Fallback recent frames when covisibility selection is empty",
    )
    parser.add_argument(
        "--covis_debug_log",
        "--covis-debug-log",
        action="store_true",
        help="Print per-frame voxel covisibility KV filtering diagnostics",
    )
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
