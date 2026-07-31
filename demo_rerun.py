#!/usr/bin/env python3
"""Rerun visualization demo for one mv_recon sequence."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from add_ckpt_path import add_path_to_dust3r
from eval.mv_recon.criterion import L21, Regr3D_t_ScaleShiftInv
from eval.mv_recon.data import NRGBD, SevenScenes
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.geometry import unproject_depth_map_to_point_map
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


DATASET_ROOTS = {
    "7scenes": Path("/home/dongjae/data/7scenes_sfm"),
    "NRGBD": Path("/home/dongjae/data/neural_rgbd_data"),
}

IGNORE_DEVICE_KEYS = {
    "depthmap",
    "dataset",
    "label",
    "instance",
    "idx",
    "true_shape",
    "rng",
}


def resolution_from_size(size: int) -> tuple[int, int] | int:
    if size == 512:
        return (512, 384)
    if size == 224:
        return 224
    if size == 518:
        return (518, 392)
    raise ValueError(f"Unsupported --size {size}; expected 224, 512, or 518.")


def parse_7scenes_sequence(sequence: str) -> tuple[str, str]:
    parts = sequence.strip("/").split("/")
    if len(parts) != 2 or not parts[0] or not parts[1].startswith("seq-"):
        raise ValueError("--sequence for 7scenes must look like 'scene/seq-XX', e.g. 'stairs/seq-06'.")
    return parts[0], parts[1]


def build_dataset(args: argparse.Namespace) -> Any:
    root = Path(args.root) if args.root is not None else DATASET_ROOTS[args.dataset]
    resolution = resolution_from_size(args.size)
    common = dict(
        split="test",
        ROOT=str(root),
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=args.stride,
        max_frames=args.max_frames,
    )
    if args.dataset == "7scenes":
        test_id, seq_id = parse_7scenes_sequence(args.sequence)
        return SevenScenes(
            test_id=test_id,
            seq_id=seq_id,
            depth_variant=args.seven_scenes_depth,
            **common,
        )
    if args.dataset == "NRGBD":
        return NRGBD(test_id=args.sequence, **common)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def move_batch_to_device(batch: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    for view in batch:
        for name, value in list(view.items()):
            if name in IGNORE_DEVICE_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                view[name] = value.to(device, non_blocking=True)
            elif isinstance(value, list):
                view[name] = [x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in value]
    return batch


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float()[0].permute(1, 2, 0).numpy()
    if image.min() < -0.1:
        image = image * 0.5 + 0.5
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def squeeze_map(value: torch.Tensor) -> torch.Tensor:
    value = value.squeeze(0)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def safe_torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(args: argparse.Namespace, device: torch.device) -> StreamVGGT:
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights}")

    add_path_to_dust3r(str(weights))
    model = StreamVGGT(total_budget=args.budget)
    checkpoint = safe_torch_load(weights, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model.to(device)


@torch.no_grad()
def run_inference(model: StreamVGGT, batch: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        autocast = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast = contextlib.nullcontext()

    inference_started_at = time.perf_counter()
    frame_elapsed_s = np.full(len(batch), np.nan, dtype=np.float64)
    frame_timestamp_s = np.full(len(batch), np.nan, dtype=np.float64)
    frame_duration_s = np.full(len(batch), np.nan, dtype=np.float64)
    previous_elapsed_s = 0.0

    def record_frame_completion(frame_idx: int, _frame: dict[str, Any], _result: dict[str, Any]) -> None:
        nonlocal previous_elapsed_s
        completed_at = time.perf_counter()
        elapsed_s = completed_at - inference_started_at
        frame_elapsed_s[frame_idx] = elapsed_s
        frame_timestamp_s[frame_idx] = time.time()
        frame_duration_s[frame_idx] = elapsed_s - previous_elapsed_s
        previous_elapsed_s = elapsed_s
        print(
            f"[Inference] frame {frame_idx + 1}/{len(batch)} ready at "
            f"+{elapsed_s:.3f}s (frame {frame_duration_s[frame_idx]:.3f}s)."
        )

    with autocast:
        results = model.inference(
            batch,
            frame_writer=record_frame_completion,
            eviction_policy=args.eviction_policy,
            stream_chunk_size=args.stream_chunk_size,
            budget_frame_multiplier=args.budget_frame_multiplier,
            leverage_granularity="layer",
            leverage_feature="key",
            leverage_projection="random",
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
            leverage_random_seed=args.random_seed,
            leverage_eviction_selector=args.leverage_eviction_selector,
            leverage_conf_gate=args.leverage_conf_gate,
            leverage_conf_gate_floor=args.leverage_conf_gate_floor,
            leverage_conf_gate_depth_alpha=args.leverage_conf_gate_depth_alpha,
            leverage_conf_gate_point_beta=args.leverage_conf_gate_point_beta,
            leverage_conf_gate_k=args.leverage_conf_gate_k,
            leverage_conf_gate_transform=args.leverage_conf_gate_transform,
            leverage_conf_gate_init=args.leverage_conf_gate_init,
            leverage_attention_utility=args.leverage_attention_utility,
            leverage_attention_beta=args.leverage_attention_beta,
            leverage_attention_ema_decay=args.leverage_attention_ema_decay,
            leverage_attention_freeze_updates=args.leverage_attention_freeze_updates,
            leverage_attention_colsum_subsample_ratio=args.leverage_attention_colsum_subsample_ratio,
            leverage_conf_gate_special_mode=args.leverage_conf_gate_special_mode,
            layer_budget_strategy=args.layer_budget_strategy,
            layer_budget_alpha=args.layer_budget_alpha,
            layer_budget_min_tokens=args.layer_budget_min_tokens,
            layer_budget_eps=args.layer_budget_eps,
            layer_budget_value_gamma=args.layer_budget_value_gamma,
            layer_budget_value_norm_type=args.layer_budget_value_norm_type,
            layer_budget_norm_source=args.layer_budget_norm_source,
        )

    preds = results.ress
    pose_enc = torch.stack([pred["camera_pose"] for pred in preds], dim=1)
    depth = torch.stack([pred["depth"] for pred in preds], dim=1)
    depth_conf = torch.stack([pred["depth_conf"] for pred in preds], dim=1)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch[0]["img"].shape[-2:])

    predicted_points = torch.stack([squeeze_map(pred["pts3d_in_other_view"]) for pred in preds], dim=0)
    predicted_conf = torch.stack([squeeze_map(pred["conf"]) for pred in preds], dim=0)

    predictions = {
        "predicted_points": predicted_points.detach().cpu().numpy(),
        "predicted_conf": predicted_conf.detach().cpu().numpy(),
        "depth": depth.detach().cpu(),
        "depth_conf": depth_conf.detach().cpu(),
        "extrinsic": extrinsic.detach().cpu().numpy()[0],
        "intrinsic": intrinsic.detach().cpu().numpy()[0] if intrinsic is not None else None,
        "inference_frame_elapsed_s": frame_elapsed_s,
        "inference_frame_timestamp_s": frame_timestamp_s,
        "inference_frame_duration_s": frame_duration_s,
    }

    if args.point_color_mode == "error":
        predictions.update(compute_accuracy_error(batch, preds, args))

    return predictions


@torch.no_grad()
def compute_accuracy_error(batch: list[dict[str, Any]], preds: list[dict[str, torch.Tensor]], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import open3d as o3d
        from scipy.spatial import cKDTree as KDTree
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--point_color_mode error needs open3d and scipy to match mv_recon Acc evaluation."
        ) from exc

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    target_device = batch[0]["pts3d"].device
    criterion_preds = [
        {"pts3d_in_other_view": pred["pts3d_in_other_view"].to(target_device, non_blocking=True)}
        for pred in preds
    ]
    gt_pts, pred_pts, _, _, masks, monitoring = criterion.get_all_pts3d_t(batch, criterion_preds)

    gt_shift_z = float(monitoring["gt_shift_z"].detach().cpu())
    pred_shift_z = float(monitoring["pred_shift_z"].detach().cpu())
    gt_scale = float(monitoring["gt_scale"].detach().cpu())
    pred_scale = float(monitoring["pred_scale"].detach().cpu())
    prediction_scale = gt_scale / pred_scale

    prediction_eval_from_model = np.eye(4, dtype=np.float32)
    prediction_eval_from_model[:3, :3] *= prediction_scale
    prediction_eval_from_model[2, 3] = -prediction_scale * pred_shift_z

    gt_reference_from_eval = np.eye(4, dtype=np.float32)
    gt_reference_from_eval[2, 3] = gt_shift_z

    pred_maps = []
    gt_maps = []
    valid_masks = []
    flat_pred_parts = []
    flat_gt_parts = []

    error_frame_stride = int(args.error_frame_stride)
    for frame_idx, (gt, pred, mask) in enumerate(zip(gt_pts, pred_pts, masks)):
        pred_np = pred.detach().float().cpu().numpy()[0]
        gt_np = gt.detach().float().cpu().numpy()[0]
        valid_np = mask.detach().cpu().numpy()[0].astype(bool)
        valid_np &= np.isfinite(pred_np).all(axis=-1) & np.isfinite(gt_np).all(axis=-1)
        if frame_idx % error_frame_stride != 0:
            valid_np[:] = False
        if args.error_eval_crop_size > 0:
            height, width = valid_np.shape
            crop = min(args.error_eval_crop_size, height, width)
            cx = width // 2
            cy = height // 2
            left = cx - crop // 2
            top = cy - crop // 2
            crop_mask = np.zeros_like(valid_np, dtype=bool)
            crop_mask[top : top + crop, left : left + crop] = True
            valid_np &= crop_mask

        pred_maps.append(pred_np)
        gt_maps.append(gt_np)
        valid_masks.append(valid_np)
        flat_pred_parts.append(pred_np[valid_np].reshape(-1, 3))
        flat_gt_parts.append(gt_np[valid_np].reshape(-1, 3))

    flat_pred = np.concatenate(flat_pred_parts, axis=0) if flat_pred_parts else np.empty((0, 3), dtype=np.float32)
    flat_gt = np.concatenate(flat_gt_parts, axis=0) if flat_gt_parts else np.empty((0, 3), dtype=np.float32)

    if len(flat_pred) == 0 or len(flat_gt) == 0:
        error_maps_np = np.stack(
            [np.full(mask.shape, np.nan, dtype=np.float32) for mask in valid_masks], axis=0
        )
        eval_points_np = np.stack(pred_maps, axis=0).astype(np.float32)
        valid_masks_np = np.stack(valid_masks, axis=0).astype(bool)
        color_max = 1.0
        error_colors = colorize_errors(error_maps_np, color_max)
        print_error_summary(np.empty((0,), dtype=np.float32), color_max)
        return {
            "eval_points": eval_points_np,
            "error_map": error_maps_np,
            "error_valid_mask": valid_masks_np,
            "error_colors": error_colors,
            "error_color_max_resolved": color_max,
            "icp_transform": None,
            "prediction_to_gt_reference_transform": None,
        }

    icp_pred, icp_gt = sample_icp_points(flat_pred, flat_gt, args)
    transformation, icp_backend = estimate_icp_transform(o3d, icp_pred, icp_gt, args)
    print(
        f"[Error] ICP backend: {icp_backend}, "
        f"icp_voxel_points={len(icp_pred)}/{len(flat_pred)}"
    )

    query_idx, query_voxel_size = sample_error_query_indices(flat_pred, args.error_query_voxel_size)
    query_pred = flat_pred[query_idx]
    query_gt = flat_gt[query_idx]
    if args.error_query_voxel_size > 0.0:
        tree_gt = query_gt
        tree_msg = "voxel"
    else:
        tree_gt = flat_gt
        tree_msg = "all"
    print(
        f"[Error] distance query points={len(query_pred)}/{len(flat_pred)}, "
        f"query_voxel_size={query_voxel_size:.6f}, gt_tree={tree_msg}"
    )

    prediction_to_gt_reference = (
        gt_reference_from_eval @ transformation @ prediction_eval_from_model
    ).astype(np.float32)
    eval_points = [
        transform_points(transform_points(pred_map, transformation), gt_reference_from_eval)
        for pred_map in pred_maps
    ]
    aligned_query_pred = transform_points(query_pred, transformation)
    distances, _ = KDTree(tree_gt).query(aligned_query_pred, workers=-1)
    distances = distances.astype(np.float32)

    selected_flat = np.zeros(len(flat_pred), dtype=bool)
    selected_flat[query_idx] = True
    error_maps = []
    selected_masks = []
    offset = 0
    distance_offset = 0
    for valid_mask in valid_masks:
        error_map = np.full(valid_mask.shape, np.nan, dtype=np.float32)
        selected_mask = np.zeros(valid_mask.shape, dtype=bool)
        count = int(valid_mask.sum())
        frame_selected = selected_flat[offset : offset + count]
        selected_count = int(frame_selected.sum())
        if selected_count > 0:
            valid_coords = np.flatnonzero(valid_mask)
            selected_coords = valid_coords[frame_selected]
            error_map.reshape(-1)[selected_coords] = distances[distance_offset : distance_offset + selected_count]
            selected_mask.reshape(-1)[selected_coords] = True
            distance_offset += selected_count
        offset += count
        error_maps.append(error_map)
        selected_masks.append(selected_mask)

    eval_points_np = np.stack(eval_points, axis=0).astype(np.float32)
    error_maps_np = np.stack(error_maps, axis=0).astype(np.float32)
    valid_masks_np = np.stack(selected_masks, axis=0).astype(bool)

    valid_errors = error_maps_np[valid_masks_np]
    color_max = resolve_error_color_max(valid_errors, args)
    error_colors = colorize_errors(error_maps_np, color_max)

    print_error_summary(valid_errors, color_max)

    return {
        "eval_points": eval_points_np,
        "error_map": error_maps_np,
        "error_valid_mask": valid_masks_np,
        "error_colors": error_colors,
        "error_color_max_resolved": color_max,
        "icp_transform": transformation.astype(np.float32),
        "prediction_to_gt_reference_transform": prediction_to_gt_reference,
    }


def resolve_error_color_max(valid_errors: np.ndarray, args: argparse.Namespace) -> float:
    if args.error_color_max is not None:
        return float(args.error_color_max)
    if valid_errors.size == 0:
        return 1.0
    color_max = float(np.percentile(valid_errors, args.error_color_percentile))
    return max(color_max, 1e-8)


def colorize_errors(errors: np.ndarray, color_max: float) -> np.ndarray:
    # A compact blue -> cyan -> green -> yellow -> red ramp for residual magnitude.
    anchors = np.array(
        [
            [49, 54, 149],
            [69, 117, 180],
            [116, 173, 209],
            [171, 221, 164],
            [254, 224, 144],
            [253, 174, 97],
            [215, 48, 39],
        ],
        dtype=np.float32,
    )
    safe_errors = np.nan_to_num(errors, nan=color_max, posinf=color_max, neginf=0.0)
    scaled = np.clip(safe_errors / max(color_max, 1e-8), 0.0, 1.0)
    positions = scaled * (len(anchors) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(anchors) - 1)
    weight = (positions - lower)[..., None]
    colors = anchors[lower] * (1.0 - weight) + anchors[upper] * weight
    return np.clip(colors, 0, 255).round().astype(np.uint8)


def print_error_summary(valid_errors: np.ndarray, color_max: float) -> None:
    if valid_errors.size == 0:
        print("[Error] no valid GT points available for accuracy residuals.")
        print(f"[Error] colormap max: {color_max:.6f}")
        return

    mean = float(np.mean(valid_errors))
    median, p90, p95, p99 = np.percentile(valid_errors, [50, 90, 95, 99])
    max_error = float(np.max(valid_errors))
    print(
        "[Error] mv_recon Acc distances: "
        f"valid={valid_errors.size}, "
        f"mean={mean:.6f}, median={median:.6f}, "
        f"p90={p90:.6f}, p95={p95:.6f}, p99={p99:.6f}, "
        f"max={max_error:.6f}, color_max={color_max:.6f}"
    )


def sample_icp_points(src: np.ndarray, dst: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.error_icp_voxel_size is None or args.error_icp_voxel_size <= 0.0:
        print("[Error] ICP voxel sampling disabled (matching mv_recon final evaluation).")
        return src, dst
    idx, voxel_size = voxel_sample_indices(src, args.error_icp_voxel_size)
    print(f"[Error] ICP voxel size: {voxel_size:.6f}")
    return src[idx], dst[idx]


def sample_error_query_indices(points: np.ndarray, voxel_size: float | None) -> tuple[np.ndarray, float]:
    if voxel_size is None or voxel_size <= 0.0:
        return np.arange(len(points), dtype=np.int64), 0.0
    return voxel_sample_indices(points, voxel_size)


def voxel_sample_indices(points: np.ndarray, voxel_size: float | None) -> tuple[np.ndarray, float]:
    finite = np.isfinite(points).all(axis=-1)
    valid_idx = np.flatnonzero(finite)
    if len(valid_idx) == 0:
        return valid_idx, 1.0

    finite_points = points[valid_idx]
    if voxel_size is None or voxel_size <= 0.0:
        bounds = np.ptp(finite_points, axis=0)
        scene_extent = float(np.max(bounds))
        voxel_size = max(scene_extent / 128.0, 1e-6)

    keys = np.floor(finite_points / voxel_size).astype(np.int64)
    _, unique_pos = np.unique(keys, axis=0, return_index=True)
    sampled = valid_idx[np.sort(unique_pos)]
    return sampled, float(voxel_size)


def estimate_icp_transform(o3d: Any, src: np.ndarray, dst: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.error_icp_device in ("auto", "cuda") and hasattr(o3d, "core"):
        try:
            cuda_available = bool(o3d.core.cuda.is_available())
        except Exception:
            cuda_available = False
        if cuda_available:
            try:
                device = o3d.core.Device("CUDA:0")
                src_pcd = o3d.t.geometry.PointCloud(device)
                dst_pcd = o3d.t.geometry.PointCloud(device)
                src_pcd.point["positions"] = o3d.core.Tensor(src.astype(np.float32), device=device)
                dst_pcd.point["positions"] = o3d.core.Tensor(dst.astype(np.float32), device=device)
                init = o3d.core.Tensor(np.eye(4, dtype=np.float32), device=device)
                result = o3d.t.pipelines.registration.icp(
                    src_pcd,
                    dst_pcd,
                    0.1,
                    init,
                    o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                return result.transformation.cpu().numpy().astype(np.float32), "open3d-tensor-cuda"
            except Exception as exc:
                if args.error_icp_device == "cuda":
                    raise RuntimeError("Requested --error_icp_device cuda, but Open3D CUDA ICP failed.") from exc
                print(f"[Error] Open3D CUDA ICP unavailable, falling back to CPU: {exc}")
        elif args.error_icp_device == "cuda":
            raise RuntimeError("Requested --error_icp_device cuda, but Open3D CUDA is not available.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src.reshape(-1, 3))
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(dst.reshape(-1, 3))
    result = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        0.1,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return np.asarray(result.transformation, dtype=np.float32), "open3d-legacy-cpu"


def select_point_source(predictions: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.point_source == "predicted":
        return predictions["predicted_points"], predictions["predicted_conf"]

    intrinsic = predictions["intrinsic"]
    if intrinsic is None:
        raise RuntimeError("Cannot use --point_source unprojected because predicted intrinsics are unavailable.")

    depth = predictions["depth"].squeeze(0).numpy()
    points = unproject_depth_map_to_point_map(depth, predictions["extrinsic"], intrinsic)
    conf = predictions["depth_conf"].squeeze(0).numpy()
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    return points, conf


def frame_points(
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    *,
    conf_thresh: float,
    point_stride: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = points[::point_stride, ::point_stride]
    colors = colors[::point_stride, ::point_stride]
    if valid_mask is not None:
        valid = np.isfinite(points).all(axis=-1) & valid_mask[::point_stride, ::point_stride].astype(bool)
    else:
        conf = conf[::point_stride, ::point_stride]
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf >= conf_thresh)
    return points[valid].reshape(-1, 3).astype(np.float32), colors[valid].reshape(-1, 3).astype(np.uint8)


def frame_error_stats(error_map: np.ndarray, valid_mask: np.ndarray, point_stride: int) -> tuple[int, float | None, float | None]:
    frame_errors = error_map[::point_stride, ::point_stride]
    frame_valid = valid_mask[::point_stride, ::point_stride].astype(bool) & np.isfinite(frame_errors)
    valid_errors = frame_errors[frame_valid]
    if valid_errors.size == 0:
        return 0, None, None
    return int(valid_errors.size), float(np.mean(valid_errors)), float(np.percentile(valid_errors, 95))


def limit_points(points: np.ndarray, colors: np.ndarray, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx], colors[idx]


def c2w_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :4] = extrinsic
    return np.linalg.inv(w2c)


def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points with any leading shape."""
    points = np.asarray(points)
    transformation = np.asarray(transformation)
    flat = points.reshape(-1, 3)
    transformed = flat @ transformation[:3, :3].T + transformation[:3, 3]
    return transformed.reshape(points.shape).astype(np.float32)


def transform_camera_pose(c2w: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Move a camera-to-world pose from source coordinates into target coordinates."""
    transformed = (np.asarray(target_from_source) @ np.asarray(c2w)).astype(np.float32)
    # A scale-shift-invariant evaluation can introduce uniform scale. Camera
    # centers need that scale, while camera orientation axes must remain unit.
    axis_norms = np.linalg.norm(transformed[:3, :3], axis=0, keepdims=True)
    if np.any(axis_norms <= 1e-12):
        raise ValueError("Camera transform produced a degenerate orientation matrix.")
    transformed[:3, :3] /= axis_norms
    return transformed


def setup_rerun(args: argparse.Namespace):
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The rerun package is not installed. Install dependencies with "
            "`pip install -r requirements.txt` or install `rerun-sdk==0.22.1`."
        ) from exc

    spawn = bool(args.spawn)
    connect = args.connect.strip() if args.connect else None

    if spawn and connect:
        raise SystemExit(
            "Use either --spawn or --connect, not both. "
            "For local viewer, use --spawn. For an already running Rerun server, use --no-spawn --connect <addr>."
        )

    if spawn and not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET")):
        if args.rrd_path:
            print("No GUI display found; disabling Rerun viewer spawn and saving the .rrd file only.")
            spawn = False
        else:
            raise SystemExit(
                "No GUI display found for Rerun viewer spawn. Re-run with "
                "`--no-spawn --rrd_path /tmp/output.rrd`, then open the .rrd on a machine with a display."
            )

    rr.init("mv_recon_rerun_demo", spawn=spawn)

    if connect:
        rr.connect_grpc(connect)

    if args.rrd_path:
        rrd_path = Path(args.rrd_path)
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(rrd_path))

    setup_rerun_blueprint(rr)

    return rr


def setup_rerun_blueprint(rr: Any) -> None:
    import rerun.blueprint as rrb

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    name="3D Reconstruction",
                ),
                rrb.Spatial2DView(
                    origin="/current_frame",
                    name="Current Frame",
                ),
                column_shares=[2.0, 1.0],
            ),
            rrb.TimePanel(expanded=True),
            collapse_panels=False,
        )
    )


def set_rerun_frame_time(
    rr: Any,
    frame_idx: int,
    *,
    inference_elapsed_s: float | None = None,
    inference_timestamp_s: float | None = None,
) -> None:
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=frame_idx)
        if inference_elapsed_s is not None:
            rr.set_time("inference_time", duration=float(inference_elapsed_s))
        if inference_timestamp_s is not None:
            rr.set_time("inference_timestamp", timestamp=float(inference_timestamp_s))
    else:
        rr.set_time_sequence("frame", frame_idx)
        if inference_elapsed_s is not None:
            rr.set_time_seconds("inference_time", float(inference_elapsed_s))
        if inference_timestamp_s is not None:
            rr.set_time_seconds("inference_timestamp", float(inference_timestamp_s))


def log_gt_rerun(rr: Any, batch: list[dict[str, Any]], args: argparse.Namespace) -> None:
    images = [denormalize_image(view["img"]) for view in batch]
    all_points = []
    all_colors = []

    for frame_idx, (view, image) in enumerate(zip(batch, images)):
        set_rerun_frame_time(rr, frame_idx)
        rr.log("current_frame/image", rr.Image(image))
        cam_path = f"world/gt_cameras/{frame_idx:06d}"
        c2w = to_numpy(view["camera_pose"])[0].astype(np.float32)
        rr.log(
            cam_path,
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
            ),
        )

        h, w = image.shape[:2]
        intrinsic = to_numpy(view["camera_intrinsics"])[0]
        pinhole_kwargs = {
            "focal_length": [float(intrinsic[0, 0]), float(intrinsic[1, 1])],
            "principal_point": [float(intrinsic[0, 2]), float(intrinsic[1, 2])],
            "width": int(w),
            "height": int(h),
        }
        if hasattr(rr.ViewCoordinates, "RDF"):
            pinhole_kwargs["camera_xyz"] = rr.ViewCoordinates.RDF
        rr.log(cam_path, rr.Pinhole(**pinhole_kwargs))
        rr.log(f"{cam_path}/image", rr.Image(image))

        points = to_numpy(view["pts3d"])[0].astype(np.float32)
        valid_mask = to_numpy(view["valid_mask"])[0].astype(bool)
        conf = np.ones(valid_mask.shape, dtype=np.float32)
        frame_pts, frame_colors = frame_points(
            points,
            image,
            conf,
            conf_thresh=0.0,
            point_stride=args.point_stride,
            valid_mask=valid_mask,
        )
        all_points.append(frame_pts)
        all_colors.append(frame_colors)

        if args.log_current_frame_points and len(frame_pts) > 0:
            rr.log(
                "world/gt_points_current",
                rr.Points3D(frame_pts, colors=frame_colors, radii=args.point_radius),
            )

        if args.log_per_frame_points and len(frame_pts) > 0:
            rr.log(
                f"world/gt_points_by_frame/{frame_idx:06d}",
                rr.Points3D(frame_pts, colors=frame_colors, radii=args.point_radius),
            )

        if args.log_accumulated_points_each_frame and all_points:
            accumulated_points = np.concatenate(all_points, axis=0)
            accumulated_colors = np.concatenate(all_colors, axis=0)
            accumulated_points, accumulated_colors = limit_points(
                accumulated_points,
                accumulated_colors,
                args.max_points,
            )
            rr.log(
                "world/gt_points_accumulated",
                rr.Points3D(accumulated_points, colors=accumulated_colors, radii=args.point_radius),
            )

        print(
            f"[Rerun][GT] logged frame {frame_idx + 1}/{len(images)} "
            f"with {len(frame_pts)} GT points."
        )

    if all_points:
        merged_points = np.concatenate(all_points, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
    else:
        merged_points = np.empty((0, 3), dtype=np.float32)
        merged_colors = np.empty((0, 3), dtype=np.uint8)
    merged_points, merged_colors = limit_points(merged_points, merged_colors, args.max_points)

    set_rerun_frame_time(rr, max(len(images) - 1, 0))
    rr.log("world/gt_points", rr.Points3D(merged_points, colors=merged_colors, radii=args.point_radius))
    print(f"Logged {len(merged_points)} GT points and {len(images)} GT poses.")


def log_rerun(
    rr: Any,
    batch: list[dict[str, Any]],
    predictions: dict[str, Any],
    points: np.ndarray,
    conf: np.ndarray,
    args: argparse.Namespace,
) -> None:
    images = [denormalize_image(view["img"]) for view in batch]
    all_points = []
    all_colors = []
    error_mode = args.point_color_mode == "error"
    prediction_to_gt_reference = (
        predictions.get("prediction_to_gt_reference_transform") if error_mode else None
    )
    alignment_available = not error_mode or prediction_to_gt_reference is not None

    all_gt_points = []
    all_gt_colors = []
    if error_mode:
        first_gt_c2w = to_numpy(batch[0]["camera_pose"])[0].astype(np.float32)
        reference_from_gt_world = np.linalg.inv(first_gt_c2w).astype(np.float32)
        if not alignment_available:
            print(
                "[Rerun][Warning] No valid ICP alignment is available; "
                "logging GT only and skipping predicted points/cameras."
            )

    for frame_idx, (view, image) in enumerate(zip(batch, images)):
        inference_elapsed_s = float(predictions["inference_frame_elapsed_s"][frame_idx])
        inference_timestamp_s = float(predictions["inference_frame_timestamp_s"][frame_idx])
        set_rerun_frame_time(
            rr,
            frame_idx,
            inference_elapsed_s=inference_elapsed_s,
            inference_timestamp_s=inference_timestamp_s,
        )
        rr.log("current_frame/image", rr.Image(image))

        h, w = image.shape[:2]
        if error_mode:
            gt_c2w = to_numpy(view["camera_pose"])[0].astype(np.float32)
            gt_c2w_ref = transform_camera_pose(gt_c2w, reference_from_gt_world)
            gt_cam_path = f"world/gt_cameras/{frame_idx:06d}"
            rr.log(
                gt_cam_path,
                rr.Transform3D(
                    translation=gt_c2w_ref[:3, 3],
                    mat3x3=gt_c2w_ref[:3, :3],
                ),
            )

            gt_intrinsic = to_numpy(view["camera_intrinsics"])[0]
            gt_pinhole_kwargs = {
                "focal_length": [float(gt_intrinsic[0, 0]), float(gt_intrinsic[1, 1])],
                "principal_point": [float(gt_intrinsic[0, 2]), float(gt_intrinsic[1, 2])],
                "width": int(w),
                "height": int(h),
            }
            if hasattr(rr.ViewCoordinates, "RDF"):
                gt_pinhole_kwargs["camera_xyz"] = rr.ViewCoordinates.RDF
            rr.log(gt_cam_path, rr.Pinhole(**gt_pinhole_kwargs))
            rr.log(f"{gt_cam_path}/image", rr.Image(image))

            gt_points_world = to_numpy(view["pts3d"])[0].astype(np.float32)
            gt_points_ref = transform_points(gt_points_world, reference_from_gt_world)
            gt_valid_mask = to_numpy(view["valid_mask"])[0].astype(bool)
            gt_conf = np.ones(gt_valid_mask.shape, dtype=np.float32)
            frame_gt_pts, frame_gt_colors = frame_points(
                gt_points_ref,
                image,
                gt_conf,
                conf_thresh=0.0,
                point_stride=args.point_stride,
                valid_mask=gt_valid_mask,
            )
            all_gt_points.append(frame_gt_pts)
            all_gt_colors.append(frame_gt_colors)

            if args.log_current_frame_points and len(frame_gt_pts) > 0:
                rr.log(
                    "world/gt_points_current",
                    rr.Points3D(frame_gt_pts, colors=frame_gt_colors, radii=args.point_radius),
                )
            if args.log_per_frame_points and len(frame_gt_pts) > 0:
                rr.log(
                    f"world/gt_points_by_frame/{frame_idx:06d}",
                    rr.Points3D(frame_gt_pts, colors=frame_gt_colors, radii=args.point_radius),
                )
            if args.log_accumulated_points_each_frame and all_gt_points:
                accumulated_gt_points = np.concatenate(all_gt_points, axis=0)
                accumulated_gt_colors = np.concatenate(all_gt_colors, axis=0)
                accumulated_gt_points, accumulated_gt_colors = limit_points(
                    accumulated_gt_points,
                    accumulated_gt_colors,
                    args.max_points,
                )
                rr.log(
                    "world/gt_points_accumulated",
                    rr.Points3D(
                        accumulated_gt_points,
                        colors=accumulated_gt_colors,
                        radii=args.point_radius,
                    ),
                )

        if alignment_available:
            cam_path = f"world/cameras/{frame_idx:06d}"
            c2w = c2w_from_extrinsic(predictions["extrinsic"][frame_idx])
            if error_mode:
                c2w = transform_camera_pose(c2w, prediction_to_gt_reference)
            rr.log(
                cam_path,
                rr.Transform3D(
                    translation=c2w[:3, 3].astype(np.float32),
                    mat3x3=c2w[:3, :3].astype(np.float32),
                ),
            )

            if predictions["intrinsic"] is None:
                focal = [float(w) / 2.0, float(w) / 2.0]
                principal_point = [float(w) / 2.0, float(h) / 2.0]
            else:
                intrinsic = predictions["intrinsic"][frame_idx]
                focal = [float(intrinsic[0, 0]), float(intrinsic[1, 1])]
                principal_point = [float(intrinsic[0, 2]), float(intrinsic[1, 2])]
            pinhole_kwargs = {
                "focal_length": focal,
                "principal_point": principal_point,
                "width": int(w),
                "height": int(h),
            }
            if hasattr(rr.ViewCoordinates, "RDF"):
                pinhole_kwargs["camera_xyz"] = rr.ViewCoordinates.RDF
            rr.log(cam_path, rr.Pinhole(**pinhole_kwargs))
            rr.log(f"{cam_path}/image", rr.Image(image))

        if error_mode:
            frame_color_source = predictions["error_colors"][frame_idx]
            frame_valid_mask = predictions["error_valid_mask"][frame_idx]
        else:
            frame_color_source = image
            frame_valid_mask = None

        if alignment_available:
            frame_pts, frame_colors = frame_points(
                points[frame_idx],
                frame_color_source,
                conf[frame_idx],
                conf_thresh=args.conf_thresh,
                point_stride=args.point_stride,
                valid_mask=frame_valid_mask,
            )
            all_points.append(frame_pts)
            all_colors.append(frame_colors)
        else:
            frame_pts = np.empty((0, 3), dtype=np.float32)
            frame_colors = np.empty((0, 3), dtype=np.uint8)

        # 1) 현재 프레임 포인트만 같은 entity에 계속 갱신
        # Rerun timeline을 재생하면 frame마다 해당 frame의 point cloud가 보임
        if args.log_current_frame_points and len(frame_pts) > 0:
            rr.log(
                "world/points_current",
                rr.Points3D(frame_pts, colors=frame_colors, radii=args.point_radius),
            )

        # 2) frame별 포인트를 별도 entity로 저장
        # 시간이 지날수록 frame별 point cloud가 누적되어 보일 수 있음
        if args.log_per_frame_points and len(frame_pts) > 0:
            rr.log(
                f"world/points_by_frame/{frame_idx:06d}",
                rr.Points3D(frame_pts, colors=frame_colors, radii=args.point_radius),
            )

        # 3) 지금까지 처리된 frame들의 누적 point cloud를 매 frame마다 갱신
        # 실제 streaming visualization 느낌에 가장 가까움
        if args.log_accumulated_points_each_frame and all_points:
            accumulated_points = np.concatenate(all_points, axis=0)
            accumulated_colors = np.concatenate(all_colors, axis=0)

            accumulated_points, accumulated_colors = limit_points(
                accumulated_points,
                accumulated_colors,
                args.max_points,
            )

            rr.log(
                "world/points_accumulated",
                rr.Points3D(
                    accumulated_points,
                    colors=accumulated_colors,
                    radii=args.point_radius,
                ),
            )

        if error_mode:
            count, mean_error, p95_error = frame_error_stats(
                predictions["error_map"][frame_idx],
                predictions["error_valid_mask"][frame_idx],
                args.point_stride,
            )
            if mean_error is None or p95_error is None:
                error_msg = "no valid error points"
            else:
                error_msg = f"error_valid={count}, mean={mean_error:.6f}, p95={p95_error:.6f}"
            print(
                f"[Rerun] logged frame {frame_idx + 1}/{len(images)} "
                f"with {len(frame_pts)} points ({error_msg})."
            )
        else:
            print(
                f"[Rerun] logged frame {frame_idx + 1}/{len(images)} "
                f"with {len(frame_pts)} points."
            )

    if all_points:
        merged_points = np.concatenate(all_points, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
    else:
        merged_points = np.empty((0, 3), dtype=np.float32)
        merged_colors = np.empty((0, 3), dtype=np.uint8)
    merged_points, merged_colors = limit_points(merged_points, merged_colors, args.max_points)

    final_frame_idx = max(len(images) - 1, 0)
    set_rerun_frame_time(
        rr,
        final_frame_idx,
        inference_elapsed_s=float(predictions["inference_frame_elapsed_s"][final_frame_idx]),
        inference_timestamp_s=float(predictions["inference_frame_timestamp_s"][final_frame_idx]),
    )
    if alignment_available:
        rr.log("world/points", rr.Points3D(merged_points, colors=merged_colors, radii=args.point_radius))

    if error_mode:
        if all_gt_points:
            merged_gt_points = np.concatenate(all_gt_points, axis=0)
            merged_gt_colors = np.concatenate(all_gt_colors, axis=0)
        else:
            merged_gt_points = np.empty((0, 3), dtype=np.float32)
            merged_gt_colors = np.empty((0, 3), dtype=np.uint8)
        merged_gt_points, merged_gt_colors = limit_points(
            merged_gt_points,
            merged_gt_colors,
            args.max_points,
        )
        rr.log(
            "world/gt_points",
            rr.Points3D(merged_gt_points, colors=merged_gt_colors, radii=args.point_radius),
        )
        print(
            f"Logged {len(merged_points)} points from {len(images)} frames "
            f"using mv_recon Acc-distance colormap and {len(merged_gt_points)} GT points "
            "in the first-GT-camera coordinate system."
        )
    else:
        print(f"Logged {len(merged_points)} points from {len(images)} frames using {args.point_source!r} geometry.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="7scenes", choices=("7scenes", "NRGBD"))
    parser.add_argument("--sequence", required=True, help="7scenes: scene/seq-XX; NRGBD: sequence directory name")
    parser.add_argument("--root", default=None, help="Dataset root override")
    parser.add_argument(
        "--seven_scenes_depth",
        choices=("auto", "projected", "raw"),
        default="auto",
        help=(
            "7Scenes depth source. auto prefers final's depth.proj.png and falls back "
            "to depth.png for 7scenes_sfm"
        ),
    )
    parser.add_argument(
        "--stride",
        "--kf_every",
        dest="stride",
        type=int,
        default=2,
        help="Frame stride passed to mv_recon kf_every",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames; omit for the full sequence")
    parser.add_argument("--weights", "--checkpoint_path", default=str(PROJECT_ROOT / "ckpt" / "checkpoints.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--budget", "--total_budget", dest="budget", type=int, default=60000)
    parser.add_argument(
        "--budget_frame_multiplier",
        type=float,
        default=None,
        help="Per-layer budget in frames; overrides --budget, matching run_final.sh",
    )

    # Keep these defaults in sync with run_final_early_attention.sh -> run_final.sh.
    eviction = parser.add_argument_group("mv_recon final early-attention eviction")
    eviction.add_argument("--stream_chunk_size", type=int, default=1)
    eviction.add_argument("--eviction_policy", default="svd_leverage")
    eviction.add_argument(
        "--leverage_eviction_selector",
        choices=("topk", "fast_dpp", "layer_head_fast_dpp", "similarity_topk"),
        default="topk",
    )
    eviction.add_argument("--leverage_normalize_rows", action=argparse.BooleanOptionalAction, default=False)
    eviction.add_argument(
        "--leverage_normalize_before_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eviction.add_argument(
        "--leverage_normalize_before_projection_headwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eviction.add_argument(
        "--leverage_projected_key_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eviction.add_argument(
        "--leverage_approx_method",
        choices=("exact_qr", "right_sketch", "full_d_ridge", "right_sketch_ridge"),
        default="right_sketch_ridge",
    )
    eviction.add_argument("--leverage_ridge_lambda", type=float, default=0)
    eviction.add_argument("--leverage_ridge_lambda_mode", choices=("relative", "absolute"), default="absolute")
    eviction.add_argument("--leverage_ridge_score_chunk_size", type=int, default=16384)
    eviction.add_argument("--leverage_ridge_jitter", type=float, default=0.0)
    eviction.add_argument("--leverage_ridge_dim", type=int, default=256)
    eviction.add_argument("--rls_refresh_interval", type=int, default=8)
    eviction.add_argument("--random_seed", "--leverage_random_seed", dest="random_seed", type=int, default=42)
    eviction.add_argument("--leverage_conf_gate", action=argparse.BooleanOptionalAction, default=True)
    eviction.add_argument("--leverage_conf_gate_floor", type=float, default=0.0)
    eviction.add_argument("--leverage_conf_gate_depth_alpha", type=float, default=1.0)
    eviction.add_argument("--leverage_conf_gate_point_beta", type=float, default=0.0)
    eviction.add_argument("--leverage_conf_gate_k", type=float, default=1.0)
    eviction.add_argument("--leverage_conf_gate_transform", choices=("ratio", "sigmoid"), default="sigmoid")
    eviction.add_argument("--leverage_conf_gate_init", default="mean")
    eviction.add_argument("--leverage_conf_gate_special_mode", choices=("mean", "one"), default="mean")
    eviction.add_argument("--leverage_attention_utility", action=argparse.BooleanOptionalAction, default=True)
    eviction.add_argument("--leverage_attention_beta", type=float, default=0.5)
    eviction.add_argument("--leverage_attention_ema_decay", type=float, default=0.9)
    eviction.add_argument("--leverage_attention_freeze_updates", type=int, default=5)
    eviction.add_argument("--leverage_attention_colsum_subsample_ratio", type=float, default=1.0)
    eviction.add_argument(
        "--layer_budget_strategy",
        choices=("uniform", "leverage_pr", "key_norm", "value_weighted_leverage_pr"),
        default="value_weighted_leverage_pr",
    )
    eviction.add_argument("--layer_budget_alpha", type=float, default=0.7)
    eviction.add_argument("--layer_budget_min_tokens", type=int, default=0)
    eviction.add_argument("--layer_budget_eps", type=float, default=0.0)
    eviction.add_argument("--layer_budget_value_gamma", type=float, default=0.7)
    eviction.add_argument("--layer_budget_value_norm_type", choices=("mean", "rms"), default="mean")
    eviction.add_argument("--layer_budget_norm_source", choices=("value", "key"), default="key")
    parser.add_argument("--point_source", choices=("predicted", "unprojected"), default="predicted")
    parser.add_argument("--gt_only", action="store_true", help="Visualize only GT poses and GT point cloud; skip model loading/inference.")
    parser.add_argument("--point_color_mode", choices=("rgb", "error"), default="rgb")
    parser.add_argument(
        "--error_frame_stride",
        type=int,
        default=1,
        help="Evaluate error on every Nth inferred frame; inference itself still uses --stride.",
    )
    parser.add_argument("--error_color_percentile", type=float, default=95.0)
    parser.add_argument("--error_color_max", type=float, default=None)
    parser.add_argument(
        "--error_eval_crop_size",
        type=int,
        default=224,
        help="Center crop size for error stats/colors; 224 matches mv_recon eval, <=0 disables cropping.",
    )
    parser.add_argument(
        "--error_icp_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ICP backend for error mode. CUDA requires an Open3D build with CUDA support.",
    )
    parser.add_argument(
        "--error_icp_voxel_size",
        type=float,
        default=0.0,
        help="Voxel size for ICP sampling; <=0 uses all points, matching mv_recon final evaluation.",
    )
    parser.add_argument(
        "--error_query_voxel_size",
        type=float,
        default=0,
        help="Voxel size for error distance/color queries; <=0 uses all valid eval points and is slowest.",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--point_stride", type=int, default=1)
    parser.add_argument("--point_radius", type=float, default=0.002)
    parser.add_argument("--max_points", type=int, default=50100000, help="<=0 disables aggregate point sampling")
    parser.add_argument("--log_per_frame_points", action="store_true")
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rrd_path", default=None)
    parser.add_argument("--connect", type=str, default=None, help="Rerun gRPC server address to connect to (e.g. localhost:9090)")
    parser.add_argument(
    "--log_current_frame_points",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Log current frame points at each Rerun frame step.",
    )
    parser.add_argument(
        "--log_accumulated_points_each_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log accumulated point cloud after each frame. Can be slower for long sequences.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")
    if args.stream_chunk_size < 1:
        raise ValueError(f"--stream_chunk_size must be >= 1, got {args.stream_chunk_size}")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError(f"--max_frames must be >= 1, got {args.max_frames}")
    if args.point_stride < 1:
        raise ValueError(f"--point_stride must be >= 1, got {args.point_stride}")
    if args.error_frame_stride < 1:
        raise ValueError(f"--error_frame_stride must be >= 1, got {args.error_frame_stride}")
    if args.budget < 1:
        raise ValueError(f"--budget must be >= 1, got {args.budget}")
    if args.rls_refresh_interval < 1:
        raise ValueError(f"--rls_refresh_interval must be >= 1, got {args.rls_refresh_interval}")
    if not 0.0 <= args.leverage_attention_beta <= 1.0:
        raise ValueError(f"--leverage_attention_beta must be in [0, 1], got {args.leverage_attention_beta}")
    if not 0.0 <= args.leverage_attention_ema_decay <= 1.0:
        raise ValueError(
            f"--leverage_attention_ema_decay must be in [0, 1], got {args.leverage_attention_ema_decay}"
        )
    if not 1 <= args.leverage_attention_freeze_updates <= 255:
        raise ValueError(
            "--leverage_attention_freeze_updates must be in [1, 255], "
            f"got {args.leverage_attention_freeze_updates}"
        )
    if not 0.0 < args.leverage_attention_colsum_subsample_ratio <= 1.0:
        raise ValueError(
            "--leverage_attention_colsum_subsample_ratio must be in (0, 1], "
            f"got {args.leverage_attention_colsum_subsample_ratio}"
        )
    if args.gt_only and args.point_color_mode != "rgb":
        raise ValueError("--gt_only visualizes GT data only; use the default --point_color_mode rgb")
    if args.point_color_mode == "error" and args.point_source != "predicted":
        raise ValueError("--point_color_mode error requires --point_source predicted to match mv_recon eval geometry")
    if not 0.0 < args.error_color_percentile <= 100.0:
        raise ValueError(
            f"--error_color_percentile must be in (0, 100], got {args.error_color_percentile}"
        )
    if args.error_color_max is not None and args.error_color_max <= 0.0:
        raise ValueError(f"--error_color_max must be > 0, got {args.error_color_max}")

def convert_batch_images_to_unit_range(batch: list[dict[str, Any]]) -> None:
    for view in batch:
        image = view.get("img")
        if isinstance(image, torch.Tensor) and image.min() < -0.1:
            view["img"] = (image + 1.0) / 2.0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if not args.gt_only and not Path(args.weights).exists():
        raise SystemExit(f"Checkpoint not found at {args.weights}")

    rr = setup_rerun(args)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    dataset = build_dataset(args)
    if len(dataset) < 1:
        raise SystemExit(f"No sequence found for dataset={args.dataset!r}, sequence={args.sequence!r}.")

    batch = default_collate([dataset[0]])
    convert_batch_images_to_unit_range(batch)
    if args.gt_only:
        log_gt_rerun(rr, batch, args)
        return

    batch = move_batch_to_device(batch, device)
    model = load_model(args, device)

    predictions = run_inference(model, batch, args, device)
    if args.point_color_mode == "error":
        points = predictions["eval_points"]
        conf = np.ones_like(predictions["error_map"], dtype=np.float32)
    else:
        points, conf = select_point_source(predictions, args)
    log_rerun(rr, batch, predictions, points, conf, args)


if __name__ == "__main__":
    main()
