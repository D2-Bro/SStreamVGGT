#!/usr/bin/env python3
"""CPU checks for run_inference image discovery and artifact exports."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from streamvggt.utils.inference_export import (  # noqa: E402
    MV_RECON_DEFAULTS,
    discover_images,
    export_inference_artifacts,
    resolve_output_directory,
    select_images,
    write_reconstruction_ply,
)


def _read_ply_header(path: Path) -> dict[str, int]:
    elements: dict[str, int] = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("element "):
                _, name, count = line.split()
                elements[name] = int(count)
            if line == "end_header":
                break
            if not line:
                raise AssertionError(f"unterminated PLY header: {path}")
    return elements


def _read_point_vertices(path: Path) -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
        assert vertex_count is not None
        return np.fromfile(handle, dtype=dtype, count=vertex_count)


def _check_discovery(root: Path) -> None:
    for name in ("frame10.JPG", "frame2.png", "frame1.WeBp", "ignore.txt"):
        (root / name).touch()
    discovered = discover_images(root)
    names = [path.name for path in discovered]
    assert names == ["frame1.WeBp", "frame2.png", "frame10.JPG"], names
    assert [path.name for path in select_images(discovered, 2, None)] == [
        "frame1.WeBp",
        "frame10.JPG",
    ]
    assert [path.name for path in select_images(discovered, 1, 2)] == [
        "frame1.WeBp",
        "frame2.png",
    ]


def _check_output_directory_resolution(root: Path) -> None:
    requested = root / "inference_results"
    assert resolve_output_directory(requested) == requested
    requested.write_bytes(b"legacy torch archive")
    fallback = root / "inference_results_artifacts"
    assert resolve_output_directory(requested) == fallback
    fallback.write_bytes(b"another file")
    assert resolve_output_directory(requested) == root / "inference_results_artifacts_2"
    fallback.unlink()
    fallback.mkdir()
    assert resolve_output_directory(requested) == fallback


def _synthetic_predictions() -> dict[str, np.ndarray]:
    frame_count, height, width = 2, 2, 3
    yy, xx = np.mgrid[:height, :width]
    base = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float32)
    world_points = np.stack((base, base + np.array([1.0, 0.0, 0.0])), axis=0)
    world_points[0, 0, 0] = np.nan

    images = np.zeros((frame_count, 3, height, width), dtype=np.float32)
    images[0, 0] = 1.0
    images[1, 1] = 1.0
    confidence = np.full((frame_count, height, width), 2.0, dtype=np.float32)
    confidence[1, 1, 2] = 1.0
    depth = np.asarray(
        [
            [[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]],
            [[[6.0], [7.0], [8.0]], [[9.0], [10.0], [np.nan]]],
        ],
        dtype=np.float32,
    )

    extrinsic = np.zeros((frame_count, 3, 4), dtype=np.float32)
    extrinsic[:, :3, :3] = np.eye(3, dtype=np.float32)
    extrinsic[1, 0, 3] = -1.0
    intrinsic = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
    intrinsic[:, 0, 0] = 100.0
    intrinsic[:, 1, 1] = 100.0
    intrinsic[:, 0, 2] = width / 2.0
    intrinsic[:, 1, 2] = height / 2.0
    return {
        "world_points": world_points,
        "world_points_conf": confidence,
        "depth": depth,
        "pose_enc": np.zeros((frame_count, 9), dtype=np.float32),
        "images": images,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
    }


def _check_exports(root: Path) -> None:
    predictions = _synthetic_predictions()
    image_paths = [root / "frame1.png", root / "frame2.png"]
    manifest = export_inference_artifacts(
        predictions,
        image_paths,
        root / "outputs",
        conf_thresh=1.5,
        point_stride=1,
        voxel_size=0.0,
        camera_size=0.25,
        inference_config={"eviction_policy": "svd_leverage"},
    )
    output_dir = root / "outputs"
    assert _read_ply_header(output_dir / "reconstruction.ply")["vertex"] == 10
    reconstruction_vertices = _read_point_vertices(output_dir / "reconstruction.ply")
    colors = np.column_stack(
        (
            reconstruction_vertices["red"],
            reconstruction_vertices["green"],
            reconstruction_vertices["blue"],
        )
    )
    assert np.count_nonzero(np.all(colors == [255, 0, 0], axis=1)) == 5
    assert np.count_nonzero(np.all(colors == [0, 255, 0], axis=1)) == 5

    stride_summary = write_reconstruction_ply(
        output_dir / "stride2.ply",
        predictions["world_points"],
        predictions["images"],
        predictions["world_points_conf"],
        conf_thresh=0.0,
        point_stride=2,
    )
    assert stride_summary.vertex_count == 3

    camera_header = _read_ply_header(output_dir / "cameras.ply")
    assert camera_header == {"vertex": 368, "face": 368}, camera_header
    camera_manifest = manifest["camera_visualization"]
    assert camera_manifest["type"] == "tube wireframe frustums with RGB axes and trajectory"
    assert camera_manifest["camera_count"] == 2
    assert camera_manifest["displayed_camera_count"] == 2
    assert camera_manifest["show_axes"] is True
    assert camera_manifest["show_trajectory"] is True
    np.testing.assert_allclose(camera_manifest["camera_thickness"], 0.25 * 0.015)

    poses = np.load(output_dir / "poses.npz")
    assert poses["camera_to_world"].shape == (2, 4, 4)
    np.testing.assert_allclose(poses["camera_to_world"][1, :3, 3], [1.0, 0.0, 0.0])
    trajectory = np.loadtxt(output_dir / "trajectory.txt")
    np.testing.assert_allclose(trajectory[1, :4], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(trajectory[:, 4:], [[0.0, 0.0, 0.0, 1.0]] * 2)

    expected_depth = predictions["depth"][..., 0]
    for frame_idx in range(2):
        saved = np.load(output_dir / "depth" / f"{frame_idx:06d}.npy")
        np.testing.assert_allclose(saved, expected_depth[frame_idx], equal_nan=True)
        with Image.open(output_dir / "depth" / f"{frame_idx:06d}.png") as image:
            assert image.mode == "RGB"
            assert image.size == (3, 2)

    with (output_dir / "manifest.json").open(encoding="utf-8") as handle:
        saved_manifest = json.load(handle)
    assert saved_manifest == manifest
    valid_depth = expected_depth[np.isfinite(expected_depth) & (expected_depth > 0)]
    np.testing.assert_allclose(
        manifest["depth_visualization"]["min"], np.percentile(valid_depth, 2)
    )
    np.testing.assert_allclose(
        manifest["depth_visualization"]["max"], np.percentile(valid_depth, 98)
    )


def _check_parser_uses_mv_recon_defaults() -> None:
    source = (PROJECT_ROOT / "run_inference.py").read_text()
    parser_bound_defaults = (
        "frame_stride",
        "max_frames",
        "total_budget",
        "eviction_policy",
        "leverage_granularity",
        "leverage_approx_method",
        "leverage_ridge_jitter",
        "leverage_ridge_dim",
        "rls_refresh_interval",
        "leverage_conf_gate",
        "leverage_projected_key_cache",
        "layer_budget_strategy",
        "layer_budget_score_only",
    )
    for key in parser_bound_defaults:
        assert f'MV_RECON_DEFAULTS["{key}"]' in source, key
    assert "if args.leverage_ridge_jitter < 0:" in source


def _check_mv_recon_defaults() -> None:
    expected = {
        "frame_stride": 1,
        "max_frames": None,
        "total_budget": 60000,
        "eviction_policy": "svd_leverage",
        "leverage_granularity": "layer",
        "leverage_approx_method": "right_sketch_ridge",
        "leverage_ridge_lambda": 0.0,
        "leverage_ridge_lambda_mode": "absolute",
        "leverage_ridge_jitter": 0.0,
        "leverage_ridge_dim": 256,
        "rls_refresh_interval": 8,
        "leverage_conf_gate": True,
        "leverage_projected_key_cache": True,
        "layer_budget_strategy": "value_weighted_leverage_pr",
        "layer_budget_score_only": False,
    }
    for key, value in expected.items():
        assert MV_RECON_DEFAULTS[key] == value, (key, MV_RECON_DEFAULTS[key], value)


def main() -> None:
    _check_mv_recon_defaults()
    _check_parser_uses_mv_recon_defaults()
    with tempfile.TemporaryDirectory(prefix="sstreamvggt_inference_export_") as tmp:
        root = Path(tmp)
        discovery_dir = root / "images"
        discovery_dir.mkdir()
        _check_discovery(discovery_dir)
        _check_output_directory_resolution(root)
        _check_exports(root)
    print("run_inference output checks passed")


if __name__ == "__main__":
    main()
