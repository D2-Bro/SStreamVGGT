"""Utilities for exporting StreamVGGT inference results.

This module intentionally has no torch dependency so that output formatting can
be tested on CPU-only machines.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import matplotlib
import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)

# Keep this in one place so the command-line runner and CPU tests agree on the
# currently selected mv_recon preset.
MV_RECON_DEFAULTS = {
    "frame_stride": 1,
    "max_frames": None,
    "total_budget": 60000,
    "eviction_policy": "svd_leverage",
    "leverage_sketch_dim": 256,
    "leverage_granularity": "layer",
    "leverage_feature": "key",
    "leverage_projection": "random",
    "leverage_normalize_rows": False,
    "leverage_normalize_before_projection": True,
    "leverage_normalize_before_projection_headwise": True,
    "leverage_projected_key_cache": True,
    "leverage_approx_method": "right_sketch_ridge",
    "leverage_ridge_lambda": 0.0,
    "leverage_ridge_lambda_mode": "absolute",
    "leverage_ridge_score_chunk_size": 16384,
    "leverage_ridge_jitter": 0.0,
    "leverage_ridge_dim": 256,
    "rls_refresh_interval": 8,
    "leverage_random_seed": 42,
    "leverage_eviction_selector": "topk",
    "leverage_conf_gate": True,
    "leverage_conf_gate_floor": 0.0,
    "leverage_conf_gate_depth_alpha": 1.0,
    "leverage_conf_gate_point_beta": 0.0,
    "leverage_conf_gate_k": 1.0,
    "leverage_conf_gate_transform": "sigmoid",
    "leverage_conf_gate_init": "mean",
    "leverage_conf_gate_special_mode": "mean",
    "layer_budget_strategy": "value_weighted_leverage_pr",
    "layer_budget_score_only": False,
    "layer_budget_alpha": 0.7,
    "layer_budget_min_tokens": 0,
    "layer_budget_eps": 0.0,
    "layer_budget_value_gamma": 0.7,
    "layer_budget_value_norm_type": "mean",
    "layer_budget_norm_source": "key",
}


def natural_sort_key(path: str | Path) -> tuple[object, ...]:
    """Return a case-insensitive key that sorts embedded integers numerically."""

    name = Path(path).name.casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def discover_images(input_dir: str | Path) -> list[Path]:
    """Discover supported image files directly under *input_dir*."""

    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {root}")
    paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_sort_key)


def select_images(
    image_paths: Sequence[str | Path],
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[Path]:
    """Apply validated stride and frame limit selection."""

    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be >= 1 when provided, got {max_frames}")
    selected = [Path(path) for path in image_paths][::frame_stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def resolve_output_directory(output_dir: str | Path) -> Path:
    """Select a directory path without overwriting a legacy result file.

    Older versions saved a torch archive directly at ``./inference_results``.
    When that file is still present, select a deterministic sibling directory
    and leave the archive untouched.
    """

    requested = Path(output_dir).expanduser()
    if not requested.exists() or requested.is_dir():
        return requested

    base = requested.with_name(f"{requested.name}_artifacts")
    candidate = base
    suffix = 2
    while candidate.exists() and not candidate.is_dir():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def _as_depth_stack(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape [N,H,W] or [N,H,W,1], got {depth.shape}")
    return depth


def _as_image_stack(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must be rank 4, got {images.shape}")
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] != 3:
        raise ValueError(f"images must have three RGB channels, got {images.shape}")
    return images.astype(np.float32, copy=False)


def _as_point_stack(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"world points must have shape [N,H,W,3], got {points.shape}")
    return points


def _as_conf_stack(confidence: np.ndarray) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim != 3:
        raise ValueError(
            f"world-point confidence must have shape [N,H,W] or [N,H,W,1], got {confidence.shape}"
        )
    return confidence


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if np.issubdtype(rgb.dtype, np.integer):
        return np.clip(rgb, 0, 255).astype(np.uint8)
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _iter_filtered_point_batches(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    conf_thresh: float,
    point_stride: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for frame_idx in range(points.shape[0]):
        frame_points = points[frame_idx, ::point_stride, ::point_stride]
        frame_colors = colors[frame_idx, ::point_stride, ::point_stride]
        frame_conf = confidence[frame_idx, ::point_stride, ::point_stride]
        mask = (
            np.isfinite(frame_points).all(axis=-1)
            & np.isfinite(frame_conf)
            & (frame_conf >= conf_thresh)
        )
        yield frame_points[mask], _rgb_to_uint8(frame_colors[mask])


def _voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    point_sum = np.stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)], axis=1
    )
    color_sum = np.stack(
        [np.bincount(inverse, weights=colors[:, axis]) for axis in range(3)], axis=1
    )
    down_points = (point_sum / count[:, None]).astype(np.float32)
    down_colors = np.rint(color_sum / count[:, None]).clip(0, 255).astype(np.uint8)
    return down_points, down_colors


@dataclass(frozen=True)
class PointCloudSummary:
    vertex_count: int
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None


def _write_ply_header(handle: Any, vertex_count: int, face_count: int = 0) -> None:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if face_count:
        lines.extend(
            [
                f"element face {face_count}",
                "property list uchar int vertex_indices",
            ]
        )
    lines.append("end_header")
    handle.write(("\n".join(lines) + "\n").encode("ascii"))


_PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _write_vertex_records(handle: Any, points: np.ndarray, colors: np.ndarray) -> None:
    records = np.empty(len(points), dtype=_PLY_VERTEX_DTYPE)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["red"] = colors[:, 0]
    records["green"] = colors[:, 1]
    records["blue"] = colors[:, 2]
    handle.write(records.tobytes())


def write_reconstruction_ply(
    path: str | Path,
    world_points: np.ndarray,
    images: np.ndarray,
    confidence: np.ndarray,
    *,
    conf_thresh: float = 0.0,
    point_stride: int = 1,
    voxel_size: float = 0.0,
) -> PointCloudSummary:
    """Write a colored binary PLY from direct world-point predictions."""

    if not math.isfinite(conf_thresh):
        raise ValueError(f"conf_thresh must be finite, got {conf_thresh}")
    if point_stride < 1:
        raise ValueError(f"point_stride must be >= 1, got {point_stride}")
    if not math.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError(f"voxel_size must be finite and >= 0, got {voxel_size}")

    points = _as_point_stack(world_points)
    colors = _as_image_stack(images)
    confidence = _as_conf_stack(confidence)
    expected_shape = points.shape[:3]
    if colors.shape[:3] != expected_shape or confidence.shape != expected_shape:
        raise ValueError(
            "world points, RGB images, and confidence must share [N,H,W]: "
            f"{points.shape}, {colors.shape}, {confidence.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if voxel_size > 0:
        batches = list(
            _iter_filtered_point_batches(
                points, colors, confidence, conf_thresh, point_stride
            )
        )
        nonempty = [(pts, rgb) for pts, rgb in batches if len(pts)]
        if nonempty:
            all_points = np.concatenate([item[0] for item in nonempty], axis=0)
            all_colors = np.concatenate([item[1] for item in nonempty], axis=0)
            all_points, all_colors = _voxel_downsample(
                all_points, all_colors, voxel_size
            )
        else:
            all_points = np.empty((0, 3), dtype=np.float32)
            all_colors = np.empty((0, 3), dtype=np.uint8)
        with path.open("wb") as handle:
            _write_ply_header(handle, len(all_points))
            _write_vertex_records(handle, all_points, all_colors)
        if len(all_points):
            return PointCloudSummary(
                len(all_points), all_points.min(axis=0), all_points.max(axis=0)
            )
        return PointCloudSummary(0, None, None)

    vertex_count = 0
    bbox_min = None
    bbox_max = None
    for batch_points, _ in _iter_filtered_point_batches(
        points, colors, confidence, conf_thresh, point_stride
    ):
        vertex_count += len(batch_points)
        if len(batch_points):
            batch_min = batch_points.min(axis=0)
            batch_max = batch_points.max(axis=0)
            bbox_min = batch_min if bbox_min is None else np.minimum(bbox_min, batch_min)
            bbox_max = batch_max if bbox_max is None else np.maximum(bbox_max, batch_max)

    with path.open("wb") as handle:
        _write_ply_header(handle, vertex_count)
        for batch_points, batch_colors in _iter_filtered_point_batches(
            points, colors, confidence, conf_thresh, point_stride
        ):
            _write_vertex_records(handle, batch_points, batch_colors)
    return PointCloudSummary(vertex_count, bbox_min, bbox_max)


def homogeneous_world_to_camera(extrinsic: np.ndarray) -> np.ndarray:
    """Convert [N,3,4] or [N,4,4] world-to-camera matrices to [N,4,4]."""

    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsic must have shape [N,3,4] or [N,4,4], got {extrinsic.shape}")
    if extrinsic.shape[1:] == (4, 4):
        return extrinsic.copy()
    result = np.broadcast_to(np.eye(4, dtype=np.float64), (len(extrinsic), 4, 4)).copy()
    result[:, :3, :4] = extrinsic
    return result


def camera_to_world_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    return np.linalg.inv(homogeneous_world_to_camera(extrinsic))


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape [3,3], got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[3] < 0:
        quat *= -1
    return quat


def write_trajectory(path: str | Path, camera_to_world: np.ndarray) -> None:
    """Write frame_index tx ty tz qx qy qz qw rows."""

    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise ValueError(
            f"camera_to_world must have shape [N,4,4], got {camera_to_world.shape}"
        )
    rows = []
    for frame_idx, pose in enumerate(camera_to_world):
        quat = rotation_matrix_to_quaternion_xyzw(pose[:3, :3])
        rows.append(
            np.concatenate(
                ([float(frame_idx)], pose[:3, 3], quat),
                dtype=np.float64,
            )
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows), fmt="%.9g")


def _camera_color(frame_idx: int, frame_count: int) -> np.ndarray:
    position = 0.5 if frame_count <= 1 else frame_idx / (frame_count - 1)
    rgb = matplotlib.colormaps["turbo"](position)[:3]
    return _rgb_to_uint8(np.asarray(rgb))


@dataclass(frozen=True)
class CameraMeshSummary:
    camera_count: int
    displayed_camera_count: int
    camera_stride: int
    vertex_count: int
    face_count: int
    thickness: float


def _append_tube(
    vertices: list[np.ndarray],
    colors: list[np.ndarray],
    faces: list[np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: np.ndarray,
    *,
    segments: int = 8,
) -> bool:
    """Append an open cylinder between two points to triangle-mesh buffers."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 1e-12:
        return False
    axis = direction / length
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(axis[0])) < 0.8
        else np.array([0.0, 1.0, 0.0])
    )
    basis_u = np.cross(axis, helper)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = radius * (
        np.cos(angles)[:, None] * basis_u[None]
        + np.sin(angles)[:, None] * basis_v[None]
    )
    tube_vertices = np.concatenate((start + ring, end + ring), axis=0)
    offset = len(vertices)
    vertices.extend(tube_vertices)
    colors.extend(np.broadcast_to(np.asarray(color, dtype=np.uint8), (2 * segments, 3)))
    for segment_idx in range(segments):
        next_idx = (segment_idx + 1) % segments
        a = offset + segment_idx
        b = offset + next_idx
        c = offset + segments + segment_idx
        d = offset + segments + next_idx
        faces.append(np.array([a, c, b], dtype=np.int32))
        faces.append(np.array([b, c, d], dtype=np.int32))
    return True


def write_cameras_ply(
    path: str | Path,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    *,
    camera_size: float,
    camera_thickness: float | None = None,
    camera_stride: int = 0,
    show_axes: bool = True,
    show_trajectory: bool = True,
) -> CameraMeshSummary:
    """Write readable tube-wireframe frustums, RGB axes, and trajectory."""

    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise ValueError(
            f"camera_to_world must have shape [N,4,4], got {camera_to_world.shape}"
        )
    if intrinsic.shape != (len(camera_to_world), 3, 3):
        raise ValueError(
            f"intrinsic must have shape [N,3,3], got {intrinsic.shape}"
        )
    if not math.isfinite(camera_size) or camera_size <= 0:
        raise ValueError(f"camera_size must be finite and > 0, got {camera_size}")
    if camera_stride < 0:
        raise ValueError(f"camera_stride must be >= 0, got {camera_stride}")
    resolved_camera_stride = (
        max(1, math.ceil(len(camera_to_world) / 50))
        if camera_stride == 0
        else camera_stride
    )
    if camera_thickness is None:
        camera_thickness = camera_size * 0.015
    if not math.isfinite(camera_thickness) or camera_thickness <= 0:
        raise ValueError(
            f"camera_thickness must be finite and > 0, got {camera_thickness}"
        )

    height, width = image_hw
    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    selected_indices = list(range(0, len(camera_to_world), resolved_camera_stride))
    if camera_to_world.shape[0] and selected_indices[-1] != len(camera_to_world) - 1:
        selected_indices.append(len(camera_to_world) - 1)

    frustum_edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    axis_specs = (
        (np.array([camera_size * 0.6, 0.0, 0.0]), np.array([255, 64, 64], dtype=np.uint8)),
        (np.array([0.0, camera_size * 0.6, 0.0]), np.array([64, 255, 64], dtype=np.uint8)),
        (np.array([0.0, 0.0, camera_size * 0.6]), np.array([64, 128, 255], dtype=np.uint8)),
    )
    for frame_idx in selected_indices:
        pose = camera_to_world[frame_idx]
        intr = intrinsic[frame_idx]
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        if fx <= 0 or fy <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
            raise ValueError(f"Invalid camera intrinsics at frame {frame_idx}: {intr}")
        corners_px = np.asarray(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            dtype=np.float64,
        )
        corners_cam = np.column_stack(
            (
                (corners_px[:, 0] - cx) / fx * camera_size,
                (corners_px[:, 1] - cy) / fy * camera_size,
                np.full(4, camera_size),
            )
        )
        keypoints_cam = np.vstack((np.zeros((1, 3)), corners_cam))
        keypoints_world = keypoints_cam @ pose[:3, :3].T + pose[:3, 3]
        frustum_color = _camera_color(frame_idx, len(camera_to_world))
        for start_idx, end_idx in frustum_edges:
            _append_tube(
                vertices,
                colors,
                faces,
                keypoints_world[start_idx],
                keypoints_world[end_idx],
                camera_thickness,
                frustum_color,
            )
        if show_axes:
            center = pose[:3, 3]
            for endpoint_cam, axis_color in axis_specs:
                endpoint_world = endpoint_cam @ pose[:3, :3].T + center
                _append_tube(
                    vertices,
                    colors,
                    faces,
                    center,
                    endpoint_world,
                    camera_thickness * 1.15,
                    axis_color,
                )

    if show_trajectory and len(camera_to_world) > 1:
        centers = camera_to_world[:, :3, 3]
        trajectory_color = np.array([235, 235, 235], dtype=np.uint8)
        for frame_idx in range(len(centers) - 1):
            _append_tube(
                vertices,
                colors,
                faces,
                centers[frame_idx],
                centers[frame_idx + 1],
                camera_thickness * 0.75,
                trajectory_color,
            )

    vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    colors_np = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    faces_np = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        _write_ply_header(handle, len(vertices_np), len(faces_np))
        _write_vertex_records(handle, vertices_np, colors_np)
        for face in faces_np:
            handle.write(struct.pack("<Biii", 3, *face.tolist()))
    return CameraMeshSummary(
        camera_count=len(camera_to_world),
        displayed_camera_count=len(selected_indices),
        camera_stride=resolved_camera_stride,
        vertex_count=len(vertices_np),
        face_count=len(faces_np),
        thickness=float(camera_thickness),
    )


def save_depth_outputs(
    output_dir: str | Path, depth: np.ndarray
) -> tuple[float | None, float | None]:
    """Save float32 NPY depth and sequence-normalized Turbo PNG images."""

    depth = _as_depth_stack(depth)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        depth_min, depth_max = np.percentile(depth[valid], [2.0, 98.0]).astype(float)
        if depth_max <= depth_min:
            depth_max = depth_min + max(abs(depth_min) * 1e-6, 1e-6)
    else:
        depth_min = depth_max = None

    cmap = matplotlib.colormaps["turbo"]
    for frame_idx, frame_depth in enumerate(depth):
        np.save(output_dir / f"{frame_idx:06d}.npy", frame_depth.astype(np.float32))
        frame_valid = np.isfinite(frame_depth) & (frame_depth > 0)
        rgb = np.zeros((*frame_depth.shape, 3), dtype=np.uint8)
        if depth_min is not None and frame_valid.any():
            normalized = np.clip(
                (frame_depth[frame_valid] - depth_min) / (depth_max - depth_min),
                0.0,
                1.0,
            )
            rgb[frame_valid] = _rgb_to_uint8(cmap(normalized)[..., :3])
        Image.fromarray(rgb, mode="RGB").save(output_dir / f"{frame_idx:06d}.png")
    return depth_min, depth_max


def export_inference_artifacts(
    predictions: dict[str, np.ndarray],
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    conf_thresh: float = 0.0,
    point_stride: int = 1,
    voxel_size: float = 0.0,
    camera_size: float | None = None,
    camera_thickness: float | None = None,
    camera_stride: int = 0,
    camera_axes: bool = True,
    camera_trajectory: bool = True,
    inference_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export reconstruction, cameras, poses, depth, and a JSON manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _as_image_stack(predictions["images"])
    world_points = _as_point_stack(predictions["world_points"])
    confidence = _as_conf_stack(predictions["world_points_conf"])
    depth = _as_depth_stack(predictions["depth"])
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(predictions["intrinsic"], dtype=np.float64)
    pose_enc = np.asarray(predictions["pose_enc"], dtype=np.float32)
    frame_count, height, width = world_points.shape[:3]
    if len(image_paths) != frame_count:
        raise ValueError(
            f"image path count {len(image_paths)} does not match predictions {frame_count}"
        )

    point_summary = write_reconstruction_ply(
        output_dir / "reconstruction.ply",
        world_points,
        images,
        confidence,
        conf_thresh=conf_thresh,
        point_stride=point_stride,
        voxel_size=voxel_size,
    )
    world_to_camera = homogeneous_world_to_camera(extrinsic)
    camera_to_world = np.linalg.inv(world_to_camera)
    if camera_size is None:
        if point_summary.bbox_min is not None:
            scene_diagonal = float(
                np.linalg.norm(point_summary.bbox_max - point_summary.bbox_min)
            )
        else:
            centers = camera_to_world[:, :3, 3]
            scene_diagonal = float(np.linalg.norm(np.ptp(centers, axis=0)))
        camera_size = 0.02 * scene_diagonal if scene_diagonal > 0 else 0.1
    camera_summary = write_cameras_ply(
        output_dir / "cameras.ply",
        camera_to_world,
        intrinsic,
        (height, width),
        camera_size=camera_size,
        camera_thickness=camera_thickness,
        camera_stride=camera_stride,
        show_axes=camera_axes,
        show_trajectory=camera_trajectory,
    )
    np.savez_compressed(
        output_dir / "poses.npz",
        pose_enc=pose_enc,
        extrinsic_world_to_camera=world_to_camera.astype(np.float32),
        camera_to_world=camera_to_world.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        image_paths=np.asarray([str(Path(path)) for path in image_paths]),
    )
    write_trajectory(output_dir / "trajectory.txt", camera_to_world)
    depth_min, depth_max = save_depth_outputs(output_dir / "depth", depth)

    manifest = {
        "frame_count": frame_count,
        "image_paths": [str(Path(path)) for path in image_paths],
        "preprocessed_image_hw": [height, width],
        "coordinate_system": {
            "extrinsic": "OpenCV world-to-camera",
            "camera_to_world": "inverse of predicted extrinsic",
            "units": "model-predicted common scene units; no GT scale/shift alignment",
        },
        "reconstruction": {
            "source": "pts3d_in_other_view",
            "vertex_count": point_summary.vertex_count,
            "confidence_threshold": conf_thresh,
            "point_stride": point_stride,
            "voxel_size": voxel_size,
        },
        "camera_visualization": {
            "camera_size": camera_size,
            "camera_thickness": camera_summary.thickness,
            "requested_camera_stride": camera_stride,
            "camera_stride": camera_summary.camera_stride,
            "camera_count": camera_summary.camera_count,
            "displayed_camera_count": camera_summary.displayed_camera_count,
            "vertex_count": camera_summary.vertex_count,
            "face_count": camera_summary.face_count,
            "show_axes": camera_axes,
            "show_trajectory": camera_trajectory,
            "type": "tube wireframe frustums with RGB axes and trajectory",
        },
        "depth_visualization": {
            "normalization": "shared sequence 2nd-98th percentile over finite positive depth",
            "min": depth_min,
            "max": depth_max,
            "colormap": "turbo",
            "invalid_color": [0, 0, 0],
        },
        "inference_config": inference_config or {},
        "outputs": {
            "reconstruction": "reconstruction.ply",
            "cameras": "cameras.ply",
            "poses": "poses.npz",
            "trajectory": "trajectory.txt",
            "depth_dir": "depth",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest
