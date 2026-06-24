#!/usr/bin/env python3

import torch

from streamvggt.models.aggregator import Aggregator
from streamvggt.models.streamvggt import _filter_grouped_camera_kv_cache
from streamvggt.layers.recent_merge import KVCacheMetadata
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


def check_first_frame_special_only_protected_count():
    default_manager = HistoryAnchorManager(
        HistoryAnchorConfig(max_anchors=2, history_protect_token_count=5),
        tokens_per_frame=10,
    )
    default_manager.num_history_anchors = 2
    if default_manager.get_protected_token_count() != 20:
        raise AssertionError(
            f"default mode should protect full first frame plus history specials: "
            f"{default_manager.get_protected_token_count()}"
        )

    special_only_manager = HistoryAnchorManager(
        HistoryAnchorConfig(
            max_anchors=2,
            history_protect_token_count=5,
            global_protect_token_count=5,
        ),
        tokens_per_frame=10,
    )
    special_only_manager.num_history_anchors = 2
    if special_only_manager.get_protected_token_count() != 15:
        raise AssertionError(
            f"special-only mode should protect first-frame specials plus history specials: "
            f"{special_only_manager.get_protected_token_count()}"
        )


def check_first_frame_patch_tokens_are_evictable():
    dummy = type("DummyAggregator", (), {"depth": 1})()
    k = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1)
    v = k.clone()
    anchor_indices = torch.arange(5).view(1, 5)
    synced = Aggregator.sync_anchor_change(
        dummy,
        [(k, v)],
        anchor_token_count=10,
        tokens_per_frame=10,
        anchor_keep_ratio=1.0,
        anchor_token_indices=anchor_indices,
        global_anchor_token_count=5,
    )
    k_synced = synced[0][0].view(-1)
    expected_protected = torch.tensor(list(range(5)) + list(range(10, 15)), dtype=torch.float32)
    if not torch.equal(k_synced[:10], expected_protected):
        raise AssertionError(f"unexpected special-only protected prefix: {k_synced[:10].tolist()}")
    if any(int(x) in set(range(5, 10)) for x in k_synced[:10].tolist()):
        raise AssertionError("first-frame patch tokens remained in the protected prefix")


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


def check_camera_kv_anchor_only_filter():
    k = torch.arange(1 * 1 * 16 * 1, dtype=torch.float32).view(1, 1, 16, 1)
    v = k + 100

    filtered_kv, filtered_frame_ids = _filter_grouped_camera_kv_cache(
        [(k, v)],
        cached_frame_ids=[0, 1, 2, 3],
        keep_frame_ids=[0, 2],
        tokens_per_frame_group=4,
    )

    expected_k = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11], dtype=torch.float32)
    expected_v = expected_k + 100
    if filtered_frame_ids != [0, 2]:
        raise AssertionError(f"unexpected filtered frame ids: {filtered_frame_ids}")
    if not torch.equal(filtered_kv[0][0].view(-1), expected_k):
        raise AssertionError(f"unexpected filtered k: {filtered_kv[0][0].view(-1).tolist()}")
    if not torch.equal(filtered_kv[0][1].view(-1), expected_v):
        raise AssertionError(f"unexpected filtered v: {filtered_kv[0][1].view(-1).tolist()}")

    persistent_kv, persistent_frame_ids = _filter_grouped_camera_kv_cache(
        [(k, v)],
        cached_frame_ids=[0, 1, 2, 3],
        keep_frame_ids=[0, 2, 3],
        tokens_per_frame_group=4,
    )
    expected_persistent_k = torch.tensor(
        [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
        dtype=torch.float32,
    )
    if persistent_frame_ids != [0, 2, 3]:
        raise AssertionError(f"unexpected persistent frame ids: {persistent_frame_ids}")
    if not torch.equal(persistent_kv[0][0].view(-1), expected_persistent_k):
        raise AssertionError(f"unexpected persistent k: {persistent_kv[0][0].view(-1).tolist()}")


def check_global_special_sidecar_insert_and_fifo():
    dummy = type("DummyAggregator", (), {"depth": 1})()
    side_k = torch.arange(20, 25, dtype=torch.float32).view(1, 1, 5, 1)
    side_v = side_k + 100
    side_meta = KVCacheMetadata.for_current_frame(1, 1, 5, frame_id=2)

    k = torch.arange(18, dtype=torch.float32).view(1, 1, 18, 1)
    v = k + 100
    meta = KVCacheMetadata.for_current_frame(1, 1, 18, frame_id=-1)
    synced = Aggregator.sync_anchor_special_tokens_from_sidecars(
        dummy,
        [(k, v, meta)],
        [(side_k, side_v, side_meta)],
        anchor_token_count=15,
        tokens_per_frame=10,
        is_fifo=False,
    )
    k_synced, _v_synced, meta_synced = synced[0]
    expected_prefix = torch.tensor(list(range(10)) + list(range(20, 25)), dtype=torch.float32)
    if not torch.equal(k_synced.view(-1)[:15], expected_prefix):
        raise AssertionError(f"unexpected inserted prefix: {k_synced.view(-1)[:15].tolist()}")
    if not torch.equal(meta_synced.frame_ids.view(-1)[10:15], torch.full((5,), 2, dtype=torch.long)):
        raise AssertionError("sidecar metadata frame ids were not inserted into protected prefix")
    if not torch.equal(meta_synced.token_indices.view(-1)[10:15], torch.arange(5, dtype=torch.int32)):
        raise AssertionError("sidecar metadata token indices were not preserved")

    fifo_k = torch.arange(25, dtype=torch.float32).view(1, 1, 25, 1)
    fifo_v = fifo_k + 100
    fifo_synced = Aggregator.sync_anchor_special_tokens_from_sidecars(
        dummy,
        [(fifo_k, fifo_v)],
        [(side_k, side_v, None)],
        anchor_token_count=15,
        tokens_per_frame=10,
        is_fifo=True,
    )
    expected_fifo_prefix = torch.tensor(list(range(10)) + list(range(20, 25)), dtype=torch.float32)
    if not torch.equal(fifo_synced[0][0].view(-1)[:15], expected_fifo_prefix):
        raise AssertionError(f"unexpected FIFO prefix: {fifo_synced[0][0].view(-1)[:15].tolist()}")


def main():
    check_camera_motion_scores()
    check_camera_motion_registration_and_fifo()
    check_first_frame_special_only_protected_count()
    check_first_frame_patch_tokens_are_evictable()
    check_sync_promotes_only_first_five_history_tokens()
    check_camera_kv_anchor_only_filter()
    check_global_special_sidecar_insert_and_fifo()
    print("history anchor sanity checks passed")


if __name__ == "__main__":
    main()
