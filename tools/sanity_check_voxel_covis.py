"""CPU sanity checks for voxel covisibility KV filtering."""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from streamvggt.layers.attention import _filter_kv_for_voxel_covis
from streamvggt.layers.recent_merge import KVCacheMetadata
from streamvggt.layers.voxel_covis import VoxelCovisConfig, VoxelCovisibilityGraph


def test_selection_thresholds_and_topk():
    graph = VoxelCovisibilityGraph(
        VoxelCovisConfig(
            enabled=True,
            min_shared_voxels=2,
            min_overlap=0.5,
            max_covis_frames=2,
        )
    )
    graph.set_frame_voxels(0, torch.tensor([1, 2, 3, 4]))
    graph.set_frame_voxels(1, torch.tensor([2, 3, 5, 6]))
    graph.set_frame_voxels(2, torch.tensor([2, 3, 7, 8]))

    selection = graph.select_for_frame(3)

    assert selection.reference_frame_id == 2
    assert selection.selected_frame_ids == [0, 2]
    assert not selection.fallback_used


def test_fallback_previous_frame_when_threshold_rejects_all():
    graph = VoxelCovisibilityGraph(
        VoxelCovisConfig(
            enabled=True,
            min_shared_voxels=99,
            min_overlap=1.0,
            fallback_recent=1,
        )
    )
    graph.set_frame_voxels(0, torch.tensor([1, 2, 3]))
    graph.set_frame_voxels(1, torch.tensor([10, 11, 12]))

    selection = graph.select_for_frame(2)

    assert selection.selected_frame_ids == [0, 1]
    assert selection.fallback_used


def test_first_frame_survives_topk():
    graph = VoxelCovisibilityGraph(
        VoxelCovisConfig(
            enabled=True,
            min_shared_voxels=1,
            min_overlap=0.01,
            max_covis_frames=2,
        )
    )
    graph.set_frame_voxels(0, torch.tensor([100]))
    graph.set_frame_voxels(1, torch.tensor([1, 2, 3]))
    graph.set_frame_voxels(2, torch.tensor([1, 2, 4]))
    graph.set_frame_voxels(3, torch.tensor([1, 2, 5]))

    selection = graph.select_for_frame(4)

    assert 0 in selection.selected_frame_ids
    assert 3 in selection.selected_frame_ids
    assert len(selection.selected_frame_ids) == 2


def test_first_frame_has_priority_when_max_one():
    graph = VoxelCovisibilityGraph(
        VoxelCovisConfig(
            enabled=True,
            min_shared_voxels=1,
            min_overlap=0.01,
            max_covis_frames=1,
        )
    )
    graph.set_frame_voxels(0, torch.tensor([100]))
    graph.set_frame_voxels(1, torch.tensor([1, 2, 3]))
    graph.set_frame_voxels(2, torch.tensor([1, 2, 4]))

    selection = graph.select_for_frame(3)

    assert selection.selected_frame_ids == [0]


def _metadata_for_frames(batch: int, heads: int, tokens_per_frame: int, frames: int):
    metadata = KVCacheMetadata.for_current_frame(batch, heads, tokens_per_frame, 0)
    for frame_id in range(1, frames):
        metadata = metadata.concat(
            KVCacheMetadata.for_current_frame(batch, heads, tokens_per_frame, frame_id)
        )
    return metadata


def test_kv_filter_preserves_master_cache_and_metadata():
    metadata = _metadata_for_frames(batch=1, heads=1, tokens_per_frame=2, frames=3)
    k = torch.arange(1 * 1 * 6 * 3, dtype=torch.float32).view(1, 1, 6, 3)
    v = k + 100
    k_before = k.clone()
    v_before = v.clone()
    meta_before = metadata.frame_ids.clone()

    k_read, v_read, mask = _filter_kv_for_voxel_covis(
        k,
        v,
        metadata,
        selected_frame_ids=[0],
        current_frame_id=2,
        query_len=2,
    )

    assert torch.equal(k, k_before)
    assert torch.equal(v, v_before)
    assert torch.equal(metadata.frame_ids, meta_before)
    assert k_read.shape == v_read.shape == (1, 1, 4, 3)
    assert mask.shape == (1, 1, 2, 4)
    assert torch.equal(k_read[0, 0], torch.cat([k[0, 0, :2], k[0, 0, 4:]], dim=0))


def test_ragged_head_filter_is_padded_and_masked():
    metadata = _metadata_for_frames(batch=1, heads=2, tokens_per_frame=2, frames=3)
    metadata.frame_ids[0, 0, 1] = 1
    k = torch.arange(1 * 2 * 6 * 2, dtype=torch.float32).view(1, 2, 6, 2)
    v = k + 100

    k_read, _, mask = _filter_kv_for_voxel_covis(
        k,
        v,
        metadata,
        selected_frame_ids=[0],
        current_frame_id=2,
        query_len=2,
    )

    assert k_read.shape == (1, 2, 4, 2)
    assert mask.shape == (1, 2, 2, 4)
    assert torch.isfinite(mask[0, 1]).all()
    assert torch.isfinite(mask[0, 0, :, :3]).all()
    assert torch.isneginf(mask[0, 0, :, 3]).all() or (mask[0, 0, :, 3] < -1e20).all()


def main():
    test_selection_thresholds_and_topk()
    test_fallback_previous_frame_when_threshold_rejects_all()
    test_first_frame_survives_topk()
    test_first_frame_has_priority_when_max_one()
    test_kv_filter_preserves_master_cache_and_metadata()
    test_ragged_head_filter_is_padded_and_masked()
    print("voxel covisibility sanity checks passed")


if __name__ == "__main__":
    main()
