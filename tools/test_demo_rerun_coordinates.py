import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_rerun import transform_camera_pose, transform_points


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def test_first_gt_camera_becomes_identity() -> None:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    first_gt_c2w = make_transform(rotation, np.array([2.0, -3.0, 1.5], dtype=np.float32))
    reference_from_world = np.linalg.inv(first_gt_c2w)

    first_gt_in_reference = transform_camera_pose(first_gt_c2w, reference_from_world)

    np.testing.assert_allclose(first_gt_in_reference, np.eye(4), atol=1e-6)


def test_gt_points_and_cameras_use_the_same_reference_transform() -> None:
    reference_from_world = make_transform(
        np.eye(3, dtype=np.float32),
        np.array([-10.0, 2.0, 0.5], dtype=np.float32),
    )
    gt_c2w = make_transform(
        np.eye(3, dtype=np.float32),
        np.array([12.0, 1.0, 3.5], dtype=np.float32),
    )
    camera_origin_world = gt_c2w[:3, 3][None]

    camera_in_reference = transform_camera_pose(gt_c2w, reference_from_world)
    origin_in_reference = transform_points(camera_origin_world, reference_from_world)[0]

    np.testing.assert_allclose(camera_in_reference[:3, 3], origin_in_reference, atol=1e-6)


def test_icp_transform_aligns_predicted_points_and_camera_together() -> None:
    rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    icp_transform = make_transform(rotation, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    predicted_c2w = make_transform(
        np.eye(3, dtype=np.float32),
        np.array([4.0, -1.0, 2.0], dtype=np.float32),
    )
    predicted_camera_origin = predicted_c2w[:3, 3][None]

    aligned_camera = transform_camera_pose(predicted_c2w, icp_transform)
    aligned_origin = transform_points(predicted_camera_origin, icp_transform)[0]

    np.testing.assert_allclose(aligned_camera[:3, 3], aligned_origin, atol=1e-6)
    np.testing.assert_allclose(aligned_camera[:3, :3], rotation, atol=1e-6)


def test_prediction_scale_moves_camera_center_without_scaling_orientation() -> None:
    prediction_to_reference = np.eye(4, dtype=np.float32)
    prediction_to_reference[:3, :3] *= 2.5
    prediction_to_reference[:3, 3] = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    predicted_c2w = make_transform(
        np.eye(3, dtype=np.float32),
        np.array([2.0, 4.0, -1.0], dtype=np.float32),
    )

    aligned_camera = transform_camera_pose(predicted_c2w, prediction_to_reference)
    aligned_origin = transform_points(predicted_c2w[:3, 3][None], prediction_to_reference)[0]

    np.testing.assert_allclose(aligned_camera[:3, 3], aligned_origin, atol=1e-6)
    np.testing.assert_allclose(aligned_camera[:3, :3], np.eye(3), atol=1e-6)
