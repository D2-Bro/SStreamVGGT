#!/usr/bin/env python3
"""CPU smoke tests for the full-point voxel + ICP reconstruction evaluator."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from eval.mv_recon.eval_scene_voxel_icp import (  # noqa: E402
    StreamingVoxelCentroids,
    eval_scene_voxel_icp,
)
from eval.mv_recon.launch import get_args_parser, main  # noqa: E402


def test_fixed_origin_streaming_centroids() -> None:
    accumulator = StreamingVoxelCentroids(0.02, merge_rows=1)
    accumulator.add(
        np.array(
            [
                [-0.001, 0.0, 0.0],
                [-0.019, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.019, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    centroids = accumulator.centroids()
    if centroids.shape != (2, 3):
        raise AssertionError(f"expected two fixed-origin voxels, got {centroids}")
    np.testing.assert_allclose(
        centroids[:, 0], np.array([-0.01, 0.01]), atol=1e-7, rtol=0
    )


def _synthetic_scene(*, include_nonfinite: bool = False, rotate: bool = False):
    yy, xx = torch.meshgrid(
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
        indexing="ij",
    )
    base = torch.stack(
        (0.1 * xx, 0.1 * yy, 1.0 + 0.03 * xx + 0.02 * yy), dim=-1
    )
    gt_frames = [base, base + torch.tensor([0.03, 0.01, 0.02])]
    batch = []
    preds = []
    for frame_idx, gt in enumerate(gt_frames):
        pred = gt * 2.0
        pred[..., 2] += 3.0
        if rotate:
            rotation = torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
            pred = pred @ rotation.T
            pred += torch.tensor([1.2, -0.7, 0.4])
        if include_nonfinite and frame_idx == 1:
            pred = pred.clone()
            pred[0, 0, 0] = torch.nan
        batch.append(
            {
                "camera_pose": torch.eye(4).unsqueeze(0),
                "pts3d": gt.unsqueeze(0),
                "valid_mask": torch.ones((1, 4, 4), dtype=torch.bool),
            }
        )
        preds.append({"pts3d_in_other_view": pred.unsqueeze(0)})
    return batch, preds


def test_umeyama_voxel_icp() -> None:
    batch, preds = _synthetic_scene(rotate=True)
    result = eval_scene_voxel_icp(
        batch,
        preds,
        scene_id="synthetic",
        voxel_size=0.02,
        eval_frame_stride=1,
    )
    if result.eval_frame_count != 2 or result.raw_valid_points != 32:
        raise AssertionError(f"unexpected input statistics: {result}")
    if result.pred_voxels < 3 or result.gt_voxels < 3:
        raise AssertionError(f"unexpected voxel counts: {result}")
    if result.acc > 1e-5 or result.comp > 1e-5:
        raise AssertionError(f"Umeyama Sim(3) alignment should be exact: {result}")
    if not all(
        np.isnan(value)
        for value in (result.nc1, result.nc2, result.nc1_med, result.nc2_med)
    ):
        raise AssertionError(f"normal metrics should be unavailable: {result}")


def test_nonfinite_filtering() -> None:
    batch, preds = _synthetic_scene(include_nonfinite=True)
    result = eval_scene_voxel_icp(
        batch,
        preds,
        scene_id="synthetic_nonfinite",
        voxel_size=0.02,
    )
    if result.raw_valid_points != 32 or result.removed_nonfinite_points != 1:
        raise AssertionError(f"non-finite filtering statistics are wrong: {result}")


def test_legacy_matching_random_sampling() -> None:
    height, width = 392, 518
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    gt = torch.stack((0.01 * xx, 0.01 * yy, 1.0 + 0.001 * xx * yy), dim=-1)
    pred = 2.0 * gt + torch.tensor([0.4, -0.2, 1.3])
    batch = [
        {
            "camera_pose": torch.eye(4).unsqueeze(0),
            "pts3d": gt.unsqueeze(0),
            "valid_mask": torch.ones((1, height, width), dtype=torch.bool),
        }
    ]
    preds = [{"pts3d_in_other_view": pred.unsqueeze(0)}]
    result = eval_scene_voxel_icp(
        batch,
        preds,
        scene_id="synthetic_sampled",
        voxel_size=0.05,
    )
    if result.raw_valid_points != height * width:
        raise AssertionError(f"unexpected raw valid count: {result}")
    if result.sampled_valid_points != 224 * 224:
        raise AssertionError(f"unexpected sampled valid count: {result}")
    if result.metric_pred_points != 224 * 224:
        raise AssertionError(f"full-point metrics must use sampled points: {result}")


def test_too_few_voxels() -> None:
    batch, preds = _synthetic_scene()
    try:
        eval_scene_voxel_icp(
            batch,
            preds,
            scene_id="synthetic_degenerate",
            voxel_size=100.0,
        )
    except ValueError as exc:
        if "Too few occupied voxels" not in str(exc):
            raise
    else:
        raise AssertionError("expected a clear error for a degenerate voxel cloud")


def test_cli_defaults_and_conflict() -> None:
    parser = get_args_parser()
    defaults = parser.parse_args([])
    if defaults.recon_eval_mode != "legacy" or defaults.eval_voxel_size != 0.02:
        raise AssertionError(f"unexpected reconstruction eval defaults: {defaults}")

    conflicting = parser.parse_args(
        ["--recon_eval_mode", "voxel_icp", "--use_proj"]
    )
    try:
        main(conflicting)
    except SystemExit as exc:
        if "cannot be combined with --use_proj" not in str(exc):
            raise
    else:
        raise AssertionError("voxel_icp and --use_proj should be rejected")


if __name__ == "__main__":
    test_fixed_origin_streaming_centroids()
    test_umeyama_voxel_icp()
    test_nonfinite_filtering()
    test_legacy_matching_random_sampling()
    test_too_few_voxels()
    test_cli_defaults_and_conflict()
    print("mv_recon full-point voxel + ICP checks passed")
