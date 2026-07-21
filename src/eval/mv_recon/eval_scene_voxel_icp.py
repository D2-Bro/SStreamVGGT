"""Full-point voxelized reconstruction evaluation.

This module intentionally lives beside the legacy evaluator instead of replacing
it. It uses a correspondence-based Umeyama Sim(3) initialization, estimates an
ICP refinement on fixed-origin voxel clouds, then evaluates the aligned
full-resolution clouds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch
from pykdtree.kdtree import KDTree as PyKDTree

from eval.mv_recon.criterion import L21, Regr3D_t
from eval.mv_recon.utils import umeyama

@dataclass(frozen=True)
class VoxelIcpEvalResult:
    acc: float
    comp: float
    nc1: float
    nc2: float
    acc_med: float
    comp_med: float
    nc1_med: float
    nc2_med: float
    raw_valid_points: int
    sampled_valid_points: int
    removed_nonfinite_points: int
    pred_voxels: int
    gt_voxels: int
    metric_pred_points: int
    metric_gt_points: int
    eval_frame_count: int
    icp_fitness: float
    icp_inlier_rmse: float


class StreamingVoxelCentroids:
    """Exact fixed-origin voxel centroids with bounded pending input memory.

    Points are reduced within each input chunk, then periodically merged with
    the accumulated voxel state.  Peak reduction memory is proportional to the
    number of occupied voxels plus ``merge_rows``, not the raw point count.
    """

    def __init__(self, voxel_size: float, *, merge_rows: int = 1_000_000):
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError(f"voxel_size must be finite and > 0, got {voxel_size}")
        if merge_rows < 1:
            raise ValueError(f"merge_rows must be >= 1, got {merge_rows}")
        self.voxel_size = float(voxel_size)
        self.merge_rows = int(merge_rows)
        self._keys = np.empty((0, 3), dtype=np.int64)
        self._sums = np.empty((0, 3), dtype=np.float64)
        self._counts = np.empty((0,), dtype=np.int64)
        self._pending_keys: list[np.ndarray] = []
        self._pending_sums: list[np.ndarray] = []
        self._pending_counts: list[np.ndarray] = []
        self._pending_rows = 0

    @staticmethod
    def _reduce(
        keys: np.ndarray,
        sums: np.ndarray,
        counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        reduced_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(reduced_sums, inverse, sums)
        reduced_counts = np.zeros((len(unique_keys),), dtype=np.int64)
        np.add.at(reduced_counts, inverse, counts)
        return unique_keys, reduced_sums, reduced_counts

    def add(self, points: np.ndarray) -> None:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        if len(points) == 0:
            return
        if not np.isfinite(points).all():
            raise ValueError("StreamingVoxelCentroids.add received non-finite points")

        points64 = points.astype(np.float64, copy=False)
        keys = np.floor(points64 / self.voxel_size).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        local_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(local_sums, inverse, points64)
        local_counts = np.bincount(inverse, minlength=len(unique_keys)).astype(
            np.int64, copy=False
        )

        self._pending_keys.append(unique_keys)
        self._pending_sums.append(local_sums)
        self._pending_counts.append(local_counts)
        self._pending_rows += len(unique_keys)
        if self._pending_rows >= self.merge_rows:
            self._merge_pending()

    def _merge_pending(self) -> None:
        if self._pending_rows == 0:
            return
        keys = np.concatenate([self._keys, *self._pending_keys], axis=0)
        sums = np.concatenate([self._sums, *self._pending_sums], axis=0)
        counts = np.concatenate([self._counts, *self._pending_counts], axis=0)
        self._keys, self._sums, self._counts = self._reduce(keys, sums, counts)
        self._pending_keys.clear()
        self._pending_sums.clear()
        self._pending_counts.clear()
        self._pending_rows = 0

    def centroids(self) -> np.ndarray:
        self._merge_pending()
        if len(self._keys) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return self._sums / self._counts[:, None]


def _as_numpy_points(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        return points.detach().float().cpu().numpy()
    return np.asarray(points, dtype=np.float32)


def _build_voxel_clouds(
    gt_points_by_frame,
    pred_points_by_frame,
    masks_by_frame,
    *,
    voxel_size: float,
    eval_frame_stride: int,
    metrics_on_voxels: bool,
) -> tuple[
    o3d.geometry.PointCloud | None,
    o3d.geometry.PointCloud | None,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    dict[str, int],
]:
    if eval_frame_stride < 1:
        raise ValueError(
            f"eval_frame_stride must be >= 1, got {eval_frame_stride}"
        )
    frame_counts = (
        len(gt_points_by_frame),
        len(pred_points_by_frame),
        len(masks_by_frame),
    )
    if len(set(frame_counts)) != 1:
        raise ValueError(
            "GT points, predicted points, and masks must have the same frame count, "
            f"got {frame_counts}"
        )

    raw_valid_points = 0
    sampled_valid_points = 0
    removed_nonfinite_points = 0
    eval_frame_count = 0
    finite_pair_parts: list[np.ndarray] = []
    sampling_rng = np.random.default_rng(seed=42)

    for frame_idx, (gt_points, pred_points, valid_mask) in enumerate(
        zip(gt_points_by_frame, pred_points_by_frame, masks_by_frame)
    ):
        if frame_idx % eval_frame_stride != 0:
            continue
        eval_frame_count += 1

        point_shape = gt_points.shape
        pred_shape = pred_points.shape
        mask_shape = valid_mask.shape
        if point_shape != pred_shape or point_shape[:-1] != mask_shape:
            raise ValueError(
                "GT points, predicted points, and valid mask shapes do not match: "
                f"gt={point_shape}, pred={pred_shape}, mask={mask_shape}"
            )

        if all(
            isinstance(value, torch.Tensor)
            for value in (gt_points, pred_points, valid_mask)
        ):
            gt_frame = gt_points.detach()[0].reshape(-1, 3)
            pred_frame = pred_points.detach()[0].reshape(-1, 3)
            valid_flat = valid_mask.detach()[0].reshape(-1).bool()
            raw_count = int(valid_flat.sum().item())
            if tuple(valid_mask.shape[-2:]) == (392, 518):
                valid_indices = torch.nonzero(valid_flat, as_tuple=False).flatten()
                max_samples = 224 * 224
                if len(valid_indices) > max_samples:
                    selected_offsets = sampling_rng.choice(
                        len(valid_indices), size=max_samples, replace=False
                    )
                    selected_offsets = torch.from_numpy(selected_offsets).to(
                        valid_indices.device
                    )
                    valid_indices = valid_indices[selected_offsets]
                sampled_flat = torch.zeros_like(valid_flat)
                sampled_flat[valid_indices] = True
            else:
                sampled_flat = valid_flat
            finite_mask = (
                sampled_flat
                & torch.isfinite(gt_frame).all(dim=1)
                & torch.isfinite(pred_frame).all(dim=1)
            )
            sampled_count = int(sampled_flat.sum().item())
            finite_pairs = torch.cat(
                (pred_frame[finite_mask], gt_frame[finite_mask]), dim=1
            ).float().cpu().numpy()
        else:
            gt_frame = _as_numpy_points(gt_points)[0].reshape(-1, 3)
            pred_frame = _as_numpy_points(pred_points)[0].reshape(-1, 3)
            valid_flat = _as_numpy_points(valid_mask)[0].reshape(-1).astype(
                bool, copy=False
            )
            raw_count = int(valid_flat.sum())
            if tuple(valid_mask.shape[-2:]) == (392, 518):
                valid_indices = np.flatnonzero(valid_flat)
                max_samples = 224 * 224
                if len(valid_indices) > max_samples:
                    valid_indices = sampling_rng.choice(
                        valid_indices, size=max_samples, replace=False
                    )
                sampled_flat = np.zeros(valid_flat.size, dtype=bool)
                sampled_flat[valid_indices] = True
            else:
                sampled_flat = valid_flat
            finite_mask = (
                sampled_flat
                & np.isfinite(gt_frame).all(axis=1)
                & np.isfinite(pred_frame).all(axis=1)
            )
            sampled_count = int(sampled_flat.sum())
            finite_pairs = np.concatenate(
                (pred_frame[finite_mask], gt_frame[finite_mask]), axis=1
            ).astype(np.float32, copy=False)

        raw_valid_points += raw_count
        sampled_valid_points += sampled_count
        removed_nonfinite_points += sampled_count - len(finite_pairs)
        finite_pair_parts.append(finite_pairs)

    full_pred = (
        np.concatenate([pairs[:, :3] for pairs in finite_pair_parts], axis=0)
        if finite_pair_parts
        else np.empty((0, 3), dtype=np.float32)
    )
    full_gt = (
        np.concatenate([pairs[:, 3:] for pairs in finite_pair_parts], axis=0)
        if finite_pair_parts
        else np.empty((0, 3), dtype=np.float32)
    )
    del finite_pair_parts
    if len(full_pred) < 3:
        raise ValueError(
            "Umeyama alignment requires at least three finite point pairs, "
            f"got {len(full_pred)}"
        )

    scale, rotation, translation = umeyama(full_pred.T, full_gt.T)
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError("Umeyama alignment produced a non-finite Sim(3) transform")

    full_pred = (
        scale * np.einsum("nj,ij->ni", full_pred, rotation)
        + translation.reshape(1, 3)
    )

    # Build each Open3D cloud once and keep the downsampled objects for ICP.
    full_pred_pcd = o3d.geometry.PointCloud()
    full_pred_pcd.points = o3d.utility.Vector3dVector(full_pred)
    full_gt_pcd = o3d.geometry.PointCloud()
    full_gt_pcd.points = o3d.utility.Vector3dVector(full_gt)

    voxel_pred_pcd = full_pred_pcd.voxel_down_sample(voxel_size)
    voxel_gt_pcd = full_gt_pcd.voxel_down_sample(voxel_size)
    del full_pred, full_gt

    # Voxel metrics do not need the full-resolution Open3D clouds after
    # downsampling, so do not retain them beyond this function.
    metric_full_pred_pcd = None if metrics_on_voxels else full_pred_pcd
    metric_full_gt_pcd = None if metrics_on_voxels else full_gt_pcd

    return (
        metric_full_pred_pcd,
        metric_full_gt_pcd,
        voxel_pred_pcd,
        voxel_gt_pcd,
        {
            "raw_valid_points": raw_valid_points,
            "sampled_valid_points": sampled_valid_points,
            "removed_nonfinite_points": removed_nonfinite_points,
            "eval_frame_count": eval_frame_count,
        },
    )


def _evaluate_clouds_with_voxel_icp(
    full_pred_pcd: o3d.geometry.PointCloud | None,
    full_gt_pcd: o3d.geometry.PointCloud | None,
    voxel_pred_pcd: o3d.geometry.PointCloud,
    voxel_gt_pcd: o3d.geometry.PointCloud,
    *,
    icp_threshold: float,
    metrics_on_voxels: bool,
) -> tuple[
    dict[str, float],
    o3d.pipelines.registration.RegistrationResult,
    int,
    int,
]:
    registration = o3d.pipelines.registration.registration_icp(
        full_pred_pcd,
        full_gt_pcd,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    if metrics_on_voxels:
        voxel_pred_pcd.transform(registration.transformation)
        metric_pred_points = np.asarray(voxel_pred_pcd.points)
        metric_gt_points = np.asarray(voxel_gt_pcd.points)
    else:
        if full_pred_pcd is None or full_gt_pcd is None:
            raise RuntimeError("Full clouds are required for full-point metrics")
        full_pred_pcd.transform(registration.transformation)
        metric_pred_points = np.asarray(full_pred_pcd.points)
        metric_gt_points = np.asarray(full_gt_pcd.points)

    gt_tree = PyKDTree(metric_gt_points)
    pred_to_gt_distances, _ = gt_tree.query(metric_pred_points)
    acc = float(np.mean(pred_to_gt_distances))
    acc_med = float(np.median(pred_to_gt_distances))
    del gt_tree, pred_to_gt_distances

    pred_tree = PyKDTree(metric_pred_points)
    gt_to_pred_distances, _ = pred_tree.query(metric_gt_points)
    comp = float(np.mean(gt_to_pred_distances))
    comp_med = float(np.median(gt_to_pred_distances))
    del pred_tree, gt_to_pred_distances

    missing_normal_metric = float("nan")
    return {
        "acc": acc,
        "comp": comp,
        "nc1": missing_normal_metric,
        "nc2": missing_normal_metric,
        "acc_med": acc_med,
        "comp_med": comp_med,
        "nc1_med": missing_normal_metric,
        "nc2_med": missing_normal_metric,
    }, registration, len(metric_pred_points), len(metric_gt_points)


def eval_scene_voxel_icp(
    batch,
    preds,
    *,
    scene_id: str,
    voxel_size: float = 0.02,
    eval_frame_stride: int = 1,
    icp_threshold: float = 0.1,
    metrics_on_voxels: bool = False,
    criterion=None,
) -> VoxelIcpEvalResult:
    """Umeyama-align, refine with voxel ICP, and evaluate selected clouds."""
    
    # Extract points in the first-camera coordinate system without the legacy
    # median-Z/scale normalization. Umeyama below performs the complete Sim(3).
    if criterion is None:
        criterion = Regr3D_t(L21, norm_mode=False)
    gt_points, pred_points, _, _, masks, _ = criterion.get_all_pts3d_t(batch, preds)
    full_pred_pcd, full_gt_pcd, pred_voxel_pcd, gt_voxel_pcd, stats = (
        _build_voxel_clouds(
            gt_points,
            pred_points,
            masks,
            voxel_size=voxel_size,
            eval_frame_stride=eval_frame_stride,
            metrics_on_voxels=metrics_on_voxels,
        )
    )

    metrics, registration, metric_pred_points, metric_gt_points = (
        _evaluate_clouds_with_voxel_icp(
            full_pred_pcd,
            full_gt_pcd,
            pred_voxel_pcd,
            gt_voxel_pcd,
            icp_threshold=icp_threshold,
            metrics_on_voxels=metrics_on_voxels,
        )
    )
    return VoxelIcpEvalResult(
        **metrics,
        raw_valid_points=stats["raw_valid_points"],
        sampled_valid_points=stats["sampled_valid_points"],
        removed_nonfinite_points=stats["removed_nonfinite_points"],
        pred_voxels=len(pred_voxel_pcd.points),
        gt_voxels=len(gt_voxel_pcd.points),
        metric_pred_points=metric_pred_points,
        metric_gt_points=metric_gt_points,
        eval_frame_count=stats["eval_frame_count"],
        icp_fitness=float(registration.fitness),
        icp_inlier_rmse=float(registration.inlier_rmse),
    )
