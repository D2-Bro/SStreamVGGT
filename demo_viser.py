"""
demo_viser.py
StreamVGGT Inference & Visualization Pipeline
"""

import os
import sys
import argparse
import glob
import tempfile
import shutil
import numpy as np
import torch
import cv2
import imageio.v2 as iio
from natsort import natsorted

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.layers.recent_merge import RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig
from viser_utils import PointCloudViewer


def resolve_global_attn_idx_ranges(args):
    if args.middle_global_only and args.global_attn_idx_ranges is not None:
        raise ValueError("--middle-global-only cannot be combined with --global-attn-idx-ranges")
    if args.middle_global_only:
        return "9:"
    return args.global_attn_idx_ranges


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'}


def _is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _image_paths_in_dir(seq_path):
    paths = [p for p in natsorted(glob.glob(os.path.join(seq_path, "*"))) if _is_image_file(p)]
    color_paths = [p for p in paths if ".color." in os.path.basename(p).lower()]
    if color_paths:
        return color_paths
    return [p for p in paths if ".depth." not in os.path.basename(p).lower()]


def _sequence_dirs(root):
    candidates = []
    for pattern in ("seq-*", "*/seq-*"):
        candidates.extend(glob.glob(os.path.join(root, pattern)))
    return [p for p in natsorted(set(candidates)) if os.path.isdir(p)]


def _nrgbd_image_dirs(root):
    candidates = []
    direct = os.path.join(root, "images")
    if os.path.isdir(direct):
        candidates.append(direct)
    candidates.extend(glob.glob(os.path.join(root, "*", "images")))
    return [p for p in natsorted(set(candidates)) if os.path.isdir(p)]


def get_image_paths(seq_path, interval=1, max_frames=None):
    """Returns list of image paths. Extracts frames if video."""
    tmp_dir = None
    if os.path.isdir(seq_path):
        paths = _image_paths_in_dir(seq_path)
        seq_dirs = _sequence_dirs(seq_path)
        has_color_frames = any(".color." in os.path.basename(p).lower() for p in paths)
        if seq_dirs and not has_color_frames:
            paths = []
            for seq_dir in seq_dirs:
                paths.extend(_image_paths_in_dir(seq_dir))
        if not paths:
            for image_dir in _nrgbd_image_dirs(seq_path):
                paths.extend(_image_paths_in_dir(image_dir))
        paths = paths[::interval]
        if max_frames is not None:
            paths = paths[:max_frames]
        return paths, None
    
    # Video handling
    cap = cv2.VideoCapture(seq_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {seq_path}")
    
    tmp_dir = tempfile.mkdtemp()
    paths = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, total, interval):
        if max_frames is not None and len(paths) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        p = os.path.join(tmp_dir, f"{i:06d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
    cap.release()
    return paths, tmp_dir


def _pose_path_for_image(image_path):
    base = os.path.basename(image_path)
    if ".color." not in base:
        return None
    return os.path.join(os.path.dirname(image_path), base.replace(".color.", ".pose.", 1).rsplit(".", 1)[0] + ".txt")


def _load_pose_stack(pose_paths):
    return np.stack([np.loadtxt(p).astype(np.float32).reshape(4, 4) for p in pose_paths])


def _load_nrgbd_pose_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    poses = []
    for i in range(0, len(lines), 4):
        block = lines[i : i + 4]
        if len(block) < 4:
            break
        if any("nan" in line.lower() for line in block):
            pose = np.eye(4, dtype=np.float32)
        else:
            pose = np.array([[float(x) for x in line.split()] for line in block], dtype=np.float32)
            pose[:, 1:3] *= -1.0
        poses.append(pose)
    return poses


def _nrgbd_pose_for_image(image_path):
    image_dir = os.path.dirname(image_path)
    if os.path.basename(image_dir) != "images":
        return None
    pose_path = os.path.join(os.path.dirname(image_dir), "poses.txt")
    if not os.path.exists(pose_path):
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if not stem.startswith("img") or not stem[3:].isdigit():
        return None
    poses = _load_nrgbd_pose_file(pose_path)
    idx = int(stem[3:])
    if idx >= len(poses):
        return None
    return poses[idx]


def load_gt_poses(path, interval=1, max_frames=None, image_paths=None):
    if image_paths is not None and (path is None or path == "auto"):
        pose_paths = [_pose_path_for_image(p) for p in image_paths]
        if pose_paths and all(p is not None and os.path.exists(p) for p in pose_paths):
            return _load_pose_stack(pose_paths)
        nrgbd_poses = [_nrgbd_pose_for_image(p) for p in image_paths]
        if nrgbd_poses and all(p is not None for p in nrgbd_poses):
            return np.stack(nrgbd_poses)

    if not path or path == "auto" or not os.path.exists(path):
        return None
    if os.path.isdir(path):
        pose_paths = natsorted(glob.glob(os.path.join(path, "frame-*.pose.txt")))[::interval]
        if max_frames:
            pose_paths = pose_paths[:max_frames]
        if pose_paths:
            return _load_pose_stack(pose_paths)
        return None

    poses = np.loadtxt(path).reshape(-1, 4, 4)[::interval]
    if max_frames: poses = poses[:max_frames]
    return poses

def align_sim3(gt_poses, pred_centers):
    
    n = min(len(gt_poses), len(pred_centers))
    gt_p, pred_p = gt_poses[:n, :3, 3], pred_centers[:n]
    
    # Center alignment
    mu_gt, mu_pred = gt_p.mean(0), pred_p.mean(0)
    gt_c, pred_c = gt_p - mu_gt, pred_p - mu_pred
    
    # Rotation & Scale
    H = gt_c.T @ pred_c
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2] *= -1; R = Vh.T @ U.T
    
    scale = S.sum() / np.trace(gt_c.T @ gt_c) if S.sum() > 1e-6 else 1.0
    
    # Anchor to t0
    gt_t0_aligned = scale * (R @ gt_poses[0, :3, 3])
    t_anchor = pred_centers[0] - gt_t0_aligned
    
    aligned = []
    for pose in gt_poses[:n]:
        new_pose = np.eye(4, dtype=np.float32)
        new_pose[:3, :3] = R @ pose[:3, :3]
        new_pose[:3, 3] = scale * (R @ pose[:3, 3]) + t_anchor
        aligned.append(new_pose)
    return np.array(aligned)


def validate_args(args):
    if args.frame_interval < 1:
        raise ValueError(f"--frame_interval must be >= 1, got {args.frame_interval}")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError(f"--max_frames must be >= 1, got {args.max_frames}")
    if args.budget < 1:
        raise ValueError(f"--budget must be >= 1, got {args.budget}")
    if args.leverage_head_mean_dim < 1:
        raise ValueError(
            f"--leverage_head_mean_dim must be >= 1, got {args.leverage_head_mean_dim}"
        )
    if args.eviction_protect_recent_frames < 0:
        raise ValueError(
            "--eviction_protect_recent_frames must be >= 0, "
            f"got {args.eviction_protect_recent_frames}"
        )
    if args.merge_window < 1:
        raise ValueError(f"--merge_window must be >= 1, got {args.merge_window}")
    if not (0.0 <= args.merge_similarity_threshold <= 1.0):
        raise ValueError(
            "--merge_similarity_threshold must be in [0, 1], "
            f"got {args.merge_similarity_threshold}"
        )
    if args.merge_voxel_size <= 0:
        raise ValueError(f"--merge_voxel_size must be > 0, got {args.merge_voxel_size}")
    if args.merge_chunk_size < 1:
        raise ValueError(f"--merge_chunk_size must be >= 1, got {args.merge_chunk_size}")
    if args.merge_patch_radius < 0:
        raise ValueError(f"--merge_patch_radius must be >= 0, got {args.merge_patch_radius}")
    if args.merge_voxel_neighbor_radius < 0:
        raise ValueError(
            "--merge_voxel_neighbor_radius must be >= 0, "
            f"got {args.merge_voxel_neighbor_radius}"
        )
    if args.merge_max_candidates_per_token < 1:
        raise ValueError(
            "--merge_max_candidates_per_token must be >= 1, "
            f"got {args.merge_max_candidates_per_token}"
        )
    if args.merge_recall_debug_max_tokens < 1:
        raise ValueError(
            "--merge_recall_debug_max_tokens must be >= 1, "
            f"got {args.merge_recall_debug_max_tokens}"
        )
    if args.svd_eviction_merge_candidate_axes < 1:
        raise ValueError("--svd_eviction_merge_candidate_axes must be >= 1")
    if args.svd_eviction_merge_reps_per_axis < 1:
        raise ValueError("--svd_eviction_merge_reps_per_axis must be >= 1")
    if not (0.0 <= args.svd_eviction_merge_similarity_threshold <= 1.0):
        raise ValueError("--svd_eviction_merge_similarity_threshold must be in [0, 1]")
    if args.svd_eviction_merge_voxel_neighbor_radius < 0:
        raise ValueError("--svd_eviction_merge_voxel_neighbor_radius must be >= 0")
    if not (0.0 <= args.svd_eviction_merge_ema_decay <= 1.0):
        raise ValueError("--svd_eviction_merge_ema_decay must be in [0, 1]")
    if args.svd_eviction_merge_max_candidates_per_token < 1:
        raise ValueError("--svd_eviction_merge_max_candidates_per_token must be >= 1")
    if args.svd_eviction_merge_chunk_size < 1:
        raise ValueError("--svd_eviction_merge_chunk_size must be >= 1")
    if args.voxel_size <= 0:
        raise ValueError(f"--voxel_size must be > 0, got {args.voxel_size}")
    if args.covis_min_shared_voxels < 0:
        raise ValueError(
            "--covis_min_shared_voxels must be >= 0, "
            f"got {args.covis_min_shared_voxels}"
        )
    if not (0.0 <= args.covis_min_overlap <= 1.0):
        raise ValueError(f"--covis_min_overlap must be in [0, 1], got {args.covis_min_overlap}")
    if args.covis_fallback_recent < 0:
        raise ValueError(f"--covis_fallback_recent must be >= 0, got {args.covis_fallback_recent}")


@torch.no_grad()
def run_inference(model, img_paths, args, global_attn_idx_ranges=None):
    device = args.device
    images = load_and_preprocess_images(img_paths).to(device)
    inputs = [{"img": img.unsqueeze(0)} for img in images]

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
    voxel_covis_config = VoxelCovisConfig(
        enabled=args.use_voxel_covis,
        voxel_size=args.voxel_size,
        min_shared_voxels=args.covis_min_shared_voxels,
        min_overlap=args.covis_min_overlap,
        max_covis_frames=args.max_covis_frames,
        fallback_recent=args.covis_fallback_recent,
        debug=args.covis_debug_log,
    )
    covis_log_fn = print if args.covis_debug_log else None

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    with torch.amp.autocast('cuda', dtype=dtype):
        output = model.inference(
            inputs,
            eviction_policy=args.eviction_policy,
            eviction_debug=args.eviction_debug,
            leverage_sketch_dim=args.leverage_sketch_dim,
            leverage_granularity=args.leverage_granularity,
            leverage_feature=args.leverage_feature,
            leverage_projection=args.leverage_projection,
            leverage_head_mean_dim=args.leverage_head_mean_dim,
            eviction_protect_recent_frames=args.eviction_protect_recent_frames,
            recent_merge_config=recent_merge_config,
            svd_eviction_merge_config=svd_eviction_merge_config,
            voxel_covis_config=voxel_covis_config,
            covis_log_fn=covis_log_fn,
            global_attn_idx_ranges=global_attn_idx_ranges,
            global_attn_debug=args.global_attn_debug,
        )

    # Unpack structured results directly to tensors
    res_keys = ['pts3d_in_other_view', 'conf', 'depth', 'depth_conf', 'camera_pose']
    raw = {k: torch.stack([r[k].squeeze(0) for r in output.ress]) for k in res_keys}
    
    # Process camera poses
    pose_enc = raw['camera_pose']
    pose_enc_in = pose_enc.unsqueeze(0) if pose_enc.ndim == 2 else pose_enc
    ext, intr = pose_encoding_to_extri_intri(pose_enc_in, images.shape[-2:])
    
    return {
        "world_points": raw['pts3d_in_other_view'].cpu().numpy(),
        "conf": raw['conf'].cpu().numpy(),
        "depth": raw['depth'].squeeze(-1).cpu().numpy(),
        "images": images.permute(0, 2, 3, 1).cpu().numpy(),
        "extrinsic": ext.squeeze(0).cpu().numpy(),
        "intrinsic": intr.squeeze(0).cpu().numpy() if intr is not None else None
    }

def save_and_format(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subdirs = {k: os.path.join(out_dir, k) for k in ['depth', 'conf', 'color', 'camera']}
    for d in subdirs.values(): os.makedirs(d, exist_ok=True)

    N, H, W, _ = data["images"].shape
    
    # Vectorized Camera Math (w2c -> c2w)
    w2c = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
    w2c[:, :3, :] = data["extrinsic"]
    c2w = np.linalg.inv(w2c)
    
    R_list, t_list = c2w[:, :3, :3], c2w[:, :3, 3]
    
    # Intrinsics
    if data["intrinsic"] is not None:
        focal = data["intrinsic"][:, 0, 0]
        pp = data["intrinsic"][:, :2, 2]
    else:
        focal = np.full(N, W / 2.0)
        pp = np.tile([W/2.0, H/2.0], (N, 1))

    # Saving loop
    for i in range(N):
        np.save(f"{subdirs['depth']}/{i:06d}.npy", data["depth"][i])
        np.save(f"{subdirs['conf']}/{i:06d}.npy", data["conf"][i])
        iio.imwrite(f"{subdirs['color']}/{i:06d}.png", (data["images"][i] * 255).astype(np.uint8))
        
        # Save camera dict
        intr = np.eye(3)
        intr[0,0] = intr[1,1] = focal[i]
        intr[:2,2] = pp[i]
        np.savez(f"{subdirs['camera']}/{i:06d}.npz", pose=c2w[i], intrinsics=intr)

    return {
        "R": R_list, "t": t_list, "focal": focal, "pp": pp,
        "pts": list(data["world_points"]), 
        "colors": list(data["images"]), 
        "confs": list(data["conf"])
    }

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_path", default="./examples", help="Input folder or video")
    parser.add_argument("--gt_path", default=None, help="GT poses (optional)")
    parser.add_argument("--output_dir", default="./demo_tmp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--weights", "--checkpoint_path", default=os.path.join(PROJECT_ROOT, "ckpt", "checkpoints.pth"))
    parser.add_argument("--budget", "--total_budget", dest="budget", type=int, default=200000)
    parser.add_argument(
        "--eviction_policy",
        "--eviction-policy",
        type=str,
        default="svd_leverage",
        choices=("mean", "baseline_mean", "svd_leverage"),
    )
    parser.add_argument("--eviction_debug", "--eviction-debug", action="store_true")
    parser.add_argument("--leverage_sketch_dim", "--leverage-sketch-dim", type=int, default=16)
    parser.add_argument(
        "--leverage_granularity",
        "--leverage-granularity",
        type=str,
        default="layer",
        choices=("head", "layer"),
    )
    parser.add_argument(
        "--leverage_feature",
        "--leverage-feature",
        type=str,
        default="key",
        choices=("key", "key_value"),
    )
    parser.add_argument(
        "--leverage_projection",
        "--leverage-projection",
        type=str,
        default="head_mean",
        choices=("random", "head_mean"),
    )
    parser.add_argument("--leverage_head_mean_dim", "--leverage-head-mean-dim", type=int, default=4)
    parser.add_argument("--eviction_protect_recent_frames", "--eviction-protect-recent-frames", type=int, default=0)
    parser.add_argument("--enable_svd_eviction_merge", "--enable-svd-eviction-merge", action="store_true")
    parser.add_argument(
        "--svd_eviction_merge_mode",
        "--svd-eviction-merge-mode",
        choices=("head", "layer_candidates", "layer"),
        default="head",
    )
    parser.add_argument("--svd_eviction_merge_candidate_axes", "--svd-eviction-merge-candidate-axes", type=int, default=2)
    parser.add_argument("--svd_eviction_merge_reps_per_axis", "--svd-eviction-merge-reps-per-axis", type=int, default=8)
    parser.add_argument("--svd_eviction_merge_similarity_threshold", "--svd-eviction-merge-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--svd_eviction_merge_use_u_sigma", "--svd-eviction-merge-use-u-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--svd_eviction_merge_geometry_gate",
        "--svd-eviction-merge-geometry-gate",
        choices=("none", "voxel_neighbor"),
        default="voxel_neighbor",
    )
    parser.add_argument("--svd_eviction_merge_voxel_neighbor_radius", "--svd-eviction-merge-voxel-neighbor-radius", type=int, default=1)
    parser.add_argument("--svd_eviction_merge_allow_missing_geometry", "--svd-eviction-merge-allow-missing-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd_eviction_merge_ema_decay", "--svd-eviction-merge-ema-decay", type=float, default=0.5)
    parser.add_argument("--svd_eviction_merge_use_depth_confidence", "--svd-eviction-merge-use-depth-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd_eviction_merge_max_candidates_per_token", "--svd-eviction-merge-max-candidates-per-token", type=int, default=32)
    parser.add_argument("--svd_eviction_merge_chunk_size", "--svd-eviction-merge-chunk-size", type=int, default=512)
    parser.add_argument("--svd_eviction_merge_debug", "--svd-eviction-merge-debug", action="store_true")
    parser.add_argument("--svd_eviction_merge_profile", "--svd-eviction-merge-profile", action="store_true")
    parser.add_argument("--enable_recent_merge", "--enable-recent-merge", action="store_true")
    parser.add_argument("--merge_window", "--merge-window", type=int, default=1)
    parser.add_argument("--merge_similarity_threshold", "--merge-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--merge_voxel_size", "--merge-voxel-size", type=float, default=0.05)
    parser.add_argument("--merge_use_depth_confidence", "--merge-use-depth-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge_debug", "--merge-debug", action="store_true")
    parser.add_argument("--merge_chunk_size", "--merge-chunk-size", type=int, default=512)
    parser.add_argument("--merge_disable_geometry_check", "--merge-disable-geometry-check", action="store_true")
    parser.add_argument(
        "--merge_candidate_mode",
        "--merge-candidate-mode",
        choices=("full", "spatial", "voxel", "voxel_spatial"),
        default="full",
    )
    parser.add_argument("--merge_patch_radius", "--merge-patch-radius", type=int, default=1)
    parser.add_argument("--merge_voxel_neighbor_radius", "--merge-voxel-neighbor-radius", type=int, default=0)
    parser.add_argument("--merge_max_candidates_per_token", "--merge-max-candidates-per-token", type=int, default=64)
    parser.add_argument("--merge_local_fallback", "--merge-local-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge_profile", "--merge-profile", action="store_true")
    parser.add_argument("--merge_recall_debug", "--merge-recall-debug", action="store_true")
    parser.add_argument("--merge_recall_debug_max_tokens", "--merge-recall-debug-max-tokens", type=int, default=1024)
    parser.add_argument("--use_voxel_covis", "--use-voxel-covis", action="store_true")
    parser.add_argument("--voxel_size", "--voxel-size", type=float, default=0.05)
    parser.add_argument("--covis_min_shared_voxels", "--covis-min-shared-voxels", type=int, default=20)
    parser.add_argument("--covis_min_overlap", "--covis-min-overlap", type=float, default=0.05)
    parser.add_argument("--max_covis_frames", "--max-covis-frames", type=int, default=8)
    parser.add_argument("--covis_fallback_recent", "--covis-fallback-recent", type=int, default=1)
    parser.add_argument("--covis_debug_log", "--covis-debug-log", action="store_true")
    parser.add_argument("--global_attn_idx_ranges", "--global-attn-idx-ranges", type=str, default=None)
    parser.add_argument("--middle_global_only", "--middle-global-only", action="store_true")
    parser.add_argument("--global_attn_debug", "--global-attn-debug", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    global_attn_idx_ranges = resolve_global_attn_idx_ranges(args)

    # Load Data
    print(f"Loading from {args.seq_path}...")
    img_paths, tmp_dir = get_image_paths(args.seq_path, args.frame_interval, args.max_frames)
    if not img_paths:
        sys.exit(f"No images found at {args.seq_path}")
    
    # Load Model
    if not os.path.exists(args.weights):
        sys.exit(f"Checkpoint not found at {args.weights}")
        
    print(
        "Using mv_recon headmean4 settings: "
        f"frames={len(img_paths)}/{args.max_frames}, "
        f"budget={args.budget}, "
        f"eviction_policy={args.eviction_policy}, "
        f"granularity={args.leverage_granularity}, "
        f"projection={args.leverage_projection}, "
        f"head_mean_dim={args.leverage_head_mean_dim}"
    )
    model = StreamVGGT(total_budget=args.budget).to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=True)
    model.eval()

    # Inference & Save
    preds = run_inference(model, img_paths, args, global_attn_idx_ranges)
    vis_data = save_and_format(preds, args.output_dir)

    # Alignment (if GT exists)
    gt_poses = load_gt_poses(args.gt_path, args.frame_interval, len(vis_data['t']), image_paths=img_paths)
    if gt_poses is not None:
        gt_poses = align_sim3(gt_poses, vis_data['t'])

    # Visualization
    print(f"Launching Viser on port {args.port}...")
    viewer = PointCloudViewer(
        model, None, 
        vis_data['pts'], vis_data['colors'], vis_data['confs'],
        {"R": vis_data['R'], "t": vis_data['t'], "focal": vis_data['focal'], "pp": vis_data['pp']},
        gt_poses=gt_poses,
        device=args.device,
        vis_threshold=args.vis_threshold,
        size=args.size,
        port=args.port,
        edge_color_list=[None]*len(img_paths),
        show_camera=True
    )
    viewer.run()

    if tmp_dir: shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()