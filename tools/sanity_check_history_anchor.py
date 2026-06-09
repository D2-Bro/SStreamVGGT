#!/usr/bin/env python3

import torch

from streamvggt.models.aggregator import Aggregator
from streamvggt.utils.history_anchor import (
    HistoryAnchorConfig,
    HistoryAnchorManager,
    compute_camera_motion_score,
)


def _pose(translation=None, quat=None):
    pose = torch.zeros(9, dtype=torch.float32)
    if translation is None:
        translation = (0.0, 0.0, 0.0)
    if quat is None:
        quat = (0.0, 0.0, 0.0, 1.0)
    pose[:3] = torch.tensor(translation, dtype=torch.float32)
    pose[3:7] = torch.tensor(quat, dtype=torch.float32)
    return pose


def check_camera_motion_scores():
    base = _pose()
    assert compute_camera_motion_score(base, base) == 0.0
    assert compute_camera_motion_score(base, _pose(quat=(0.0, 0.0, 0.0, -1.0))) == 0.0

    translation_score = compute_camera_motion_score(base, _pose(translation=(0.3, 0.0, 0.0)))
    if abs(translation_score - 0.3) > 1e-6:
        raise AssertionError(f"unexpected translation score: {translation_score}")

    rotation_score = compute_camera_motion_score(base, _pose(quat=(0.0, 0.0, 0.70710678, 0.70710678)))
    if rotation_score <= 0.2:
        raise AssertionError(f"rotation score should cross threshold: {rotation_score}")


def check_camera_motion_registration_and_fifo():
    manager = HistoryAnchorManager(
        HistoryAnchorConfig(
            strategy="camera_motion",
            camera_motion_threshold=0.2,
            min_anchor_interval=0,
            max_anchors=2,
            history_protect_token_count=5,
        ),
        tokens_per_frame=10,
    )
    should, _, _, score = manager.should_become_anchor_camera_motion(0, _pose())
    assert not should and score == 0.0

    should, _, _, score = manager.should_become_anchor_camera_motion(1, _pose(translation=(0.1, 0.0, 0.0)))
    assert not should and 0.0 < score < 0.2

    should, is_fifo, _, score = manager.should_become_anchor_camera_motion(2, _pose(translation=(0.21, 0.0, 0.0)))
    assert should and not is_fifo and score >= 0.2
    manager.register_anchor_camera_motion(2, _pose(translation=(0.21, 0.0, 0.0)))

    should, _, _, score = manager.should_become_anchor_camera_motion(3, _pose(translation=(0.3, 0.0, 0.0)))
    assert not should and score < 0.2

    should, is_fifo, _, _ = manager.should_become_anchor_camera_motion(4, _pose(translation=(0.5, 0.0, 0.0)))
    assert should and not is_fifo
    manager.register_anchor_camera_motion(4, _pose(translation=(0.5, 0.0, 0.0)))

    should, is_fifo, _, _ = manager.should_become_anchor_camera_motion(5, _pose(translation=(0.8, 0.0, 0.0)))
    assert should and is_fifo
    manager.register_anchor_camera_motion(5, _pose(translation=(0.8, 0.0, 0.0)))

    if manager.num_history_anchors != 2 or manager.history_anchor_frames != [4, 5]:
        raise AssertionError(f"unexpected FIFO state: {manager}")
    if manager.get_protected_token_count() != 20:
        raise AssertionError(f"unexpected protected token count: {manager.get_protected_token_count()}")


def check_sync_promotes_only_first_five_history_tokens():
    dummy = type("DummyAggregator", (), {"depth": 1})()
    k = torch.arange(30, dtype=torch.float32).view(1, 1, 30, 1)
    v = k.clone()
    anchor_indices = torch.arange(5).view(1, 5)
    synced = Aggregator.sync_anchor_change(
        dummy,
        [(k, v)],
        anchor_token_count=15,
        tokens_per_frame=10,
        anchor_keep_ratio=1.0,
        anchor_token_indices=anchor_indices,
    )
    k_synced = synced[0][0].view(-1)
    expected_protected = torch.tensor(list(range(10)) + list(range(20, 25)), dtype=torch.float32)
    if not torch.equal(k_synced[:15], expected_protected):
        raise AssertionError(f"unexpected protected prefix: {k_synced[:15].tolist()}")
    if any(int(x) in set(range(25, 30)) for x in k_synced[:15].tolist()):
        raise AssertionError("patch tokens from the history frame entered the protected prefix")


def main():
    check_camera_motion_scores()
    check_camera_motion_registration_and_fifo()
    check_sync_promotes_only_first_five_history_tokens()
    print("history anchor sanity checks passed")


if __name__ == "__main__":
    main()
