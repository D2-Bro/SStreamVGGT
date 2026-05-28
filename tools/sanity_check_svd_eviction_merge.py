#!/usr/bin/env python3
"""Lightweight checks for SVD-guided eviction merge."""

from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.eviction import EvictionManager, EvictionResult, SvdLeverageBasis
from streamvggt.layers.recent_merge import KVCacheMetadata
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig, SvdEvictionMerger


def check_selection_is_unchanged_by_basis() -> None:
    torch.manual_seed(23)
    k = torch.randn(1, 2, 9, 4)
    v = torch.randn_like(k)
    manager = EvictionManager(policy="svd_leverage", leverage_sketch_dim=4, leverage_granularity="head")
    base = manager.select(k, cache_budget=5, num_anchor_tokens=1, v=v)
    with_basis = manager.select(k, cache_budget=5, num_anchor_tokens=1, v=v, need_leverage_basis=True)
    assert torch.equal(base.kept_candidate_indices, with_basis.kept_candidate_indices)
    assert with_basis.leverage_basis is not None
    assert with_basis.leverage_basis.q.shape[:3] == (1, 2, 8)


def _merge_inputs():
    k = torch.zeros(1, 1, 5, 3)
    v = torch.zeros_like(k)
    k[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    k[0, 0, 2] = torch.tensor([0.98, 0.1, 0.0])
    k[0, 0, 3] = torch.tensor([0.0, 1.0, 0.0])
    k[0, 0, 4] = torch.tensor([0.0, 0.0, 1.0])
    v[0, 0, 1] = torch.tensor([10.0, 0.0, 0.0])
    v[0, 0, 2] = torch.tensor([2.0, 0.0, 0.0])
    v[0, 0, 3] = torch.tensor([0.0, 3.0, 0.0])
    v[0, 0, 4] = torch.tensor([0.0, 0.0, 4.0])
    metadata = KVCacheMetadata.for_current_frame(1, 1, 5, frame_id=0)
    q = torch.tensor([[[[0.0, 0.0], [0.01, 0.0], [10.0, 0.0], [20.0, 0.0]]]])
    basis = SvdLeverageBasis(q=q, r_diag=torch.ones(1, 1, 2), granularity="head")
    result = EvictionResult(
        kept_candidate_indices=torch.tensor([[[1, 2]]]),
        policy_scores=torch.empty(1, 1, 4),
        mean_scores=torch.empty(1, 1, 0),
        summary_score=0.0,
        leverage_basis=basis,
    )
    return k, v, metadata, result


def run_merge(config: SvdEvictionMergeConfig, metadata=None):
    k, v, default_metadata, result = _merge_inputs()
    if metadata is None:
        metadata = default_metadata
    stats = SvdEvictionMerger(config, num_anchor_tokens=1).merge(k, v, metadata, result, layer_id=0, step_idx=1)
    return k, v, metadata, stats


def check_merge_updates_retained_kv() -> None:
    config = SvdEvictionMergeConfig(
        enabled=True,
        similarity_threshold=0.9,
        geometry_gate="none",
        ema_decay=0.25,
        use_depth_confidence=False,
        max_candidates_per_token=1,
    )
    k, v, metadata, stats = run_merge(config)
    assert stats.accepted_pairs == 1
    assert stats.candidate_pairs <= 2
    assert not torch.allclose(k[0, 0, 2], torch.tensor([0.98, 0.1, 0.0]))
    assert torch.allclose(v[0, 0, 2], torch.tensor([8.0, 0.0, 0.0]))
    assert int(metadata.merge_counts[0, 0, 2].item()) == 1


def check_missing_geometry_modes() -> None:
    allow = SvdEvictionMergeConfig(enabled=True, geometry_gate="voxel_neighbor", allow_missing_geometry=True)
    _, _, _, allow_stats = run_merge(allow)
    assert allow_stats.accepted_pairs == 1
    assert allow_stats.missing_geometry >= 1

    reject = SvdEvictionMergeConfig(enabled=True, geometry_gate="voxel_neighbor", allow_missing_geometry=False)
    _, _, _, reject_stats = run_merge(reject)
    assert reject_stats.accepted_pairs == 0
    assert reject_stats.missing_geometry >= 1


def check_voxel_neighbor_radius() -> None:
    config = SvdEvictionMergeConfig(
        enabled=True,
        geometry_gate="voxel_neighbor",
        allow_missing_geometry=False,
        voxel_neighbor_radius=1,
    )
    k, v, metadata, result = _merge_inputs()
    metadata.voxel_valid[0, 0, 1] = True
    metadata.voxel_valid[0, 0, 2] = True
    metadata.voxel_ids[0, 0, 1] = torch.tensor([0, 0, 0], dtype=torch.int32)
    metadata.voxel_ids[0, 0, 2] = torch.tensor([1, 1, 1], dtype=torch.int32)
    stats = SvdEvictionMerger(config, num_anchor_tokens=1).merge(k, v, metadata, result)
    assert stats.accepted_pairs == 1

    k, v, metadata, result = _merge_inputs()
    metadata.voxel_valid[0, 0, 1] = True
    metadata.voxel_valid[0, 0, 2] = True
    metadata.voxel_ids[0, 0, 1] = torch.tensor([0, 0, 0], dtype=torch.int32)
    metadata.voxel_ids[0, 0, 2] = torch.tensor([3, 0, 0], dtype=torch.int32)
    stats = SvdEvictionMerger(config, num_anchor_tokens=1).merge(k, v, metadata, result)
    assert stats.accepted_pairs == 0
    assert stats.rejected_geometry == 1


def check_metadata_geometry_pruning() -> None:
    metadata = KVCacheMetadata.for_current_frame(1, 2, 5, frame_id=7)
    metadata.voxel_valid[:, :, 3] = True
    metadata.voxel_ids[:, :, 3] = torch.tensor([4, 5, 6], dtype=torch.int32)
    gathered = metadata.gather(torch.tensor([[[0, 3]], [[0, 3]]]).transpose(0, 1))
    assert gathered.voxel_ids.shape == (1, 2, 2, 3)
    assert bool(gathered.voxel_valid[:, :, 1].all())
    pruned = metadata.prune_after_eviction(torch.tensor([[[2, 3], [2, 3]]]), num_anchor_tokens=1)
    assert pruned.voxel_ids.shape == (1, 2, 3, 3)
    assert bool(pruned.voxel_valid[:, :, 1].all())



def _layer_merge_inputs(head_specific: bool = False):
    k = torch.zeros(1, 2, 5, 2)
    v = torch.zeros_like(k)
    # Absolute token 0 is the anchor. Candidate-local kept indices [1, 2]
    # map to absolute retained tokens [2, 3]; evicted [0, 3] map to [1, 4].
    if head_specific:
        k[0, 0, 1] = torch.tensor([1.0, 0.0])
        k[0, 0, 2] = torch.tensor([1.0, 0.0])
        k[0, 0, 3] = torch.tensor([0.0, 1.0])
        k[0, 1, 1] = torch.tensor([0.0, 1.0])
        k[0, 1, 2] = torch.tensor([1.0, 0.0])
        k[0, 1, 3] = torch.tensor([0.0, 1.0])
    else:
        k[0, :, 1] = torch.tensor([1.0, 0.0])
        k[0, :, 2] = torch.tensor([1.0, 0.0])
        k[0, :, 3] = torch.tensor([0.0, 1.0])
    k[0, :, 4] = torch.tensor([0.0, -1.0])
    v[0, 0, 1] = torch.tensor([11.0, 0.0])
    v[0, 1, 1] = torch.tensor([0.0, 17.0])
    metadata = KVCacheMetadata.for_current_frame(1, 2, 5, frame_id=0)
    q = torch.tensor([[[0.5, 0.0], [0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]])
    basis = SvdLeverageBasis(q=q, r_diag=torch.ones(1, 2), granularity="layer")
    result = EvictionResult(
        kept_candidate_indices=torch.tensor([[[1, 2], [1, 2]]]),
        policy_scores=torch.empty(1, 4),
        mean_scores=torch.empty(1, 2, 0),
        summary_score=0.0,
        leverage_basis=basis,
    )
    return k, v, metadata, result


def check_layer_candidates_shares_candidates_keeps_headwise_match() -> None:
    k, v, metadata, result = _layer_merge_inputs(head_specific=True)
    config = SvdEvictionMergeConfig(
        enabled=True,
        mode="layer_candidates",
        similarity_threshold=0.99,
        candidate_axes=1,
        geometry_gate="none",
        ema_decay=0.0,
        use_depth_confidence=False,
        reps_per_axis=2,
        max_candidates_per_token=2,
    )
    stats = SvdEvictionMerger(config, num_anchor_tokens=1).merge(k, v, metadata, result)
    assert stats.accepted_pairs == 2
    assert torch.allclose(v[0, 0, 2], torch.tensor([11.0, 0.0]))
    assert torch.allclose(v[0, 1, 3], torch.tensor([0.0, 17.0]))
    assert int(metadata.merge_counts[0, 0, 2].item()) == 1
    assert int(metadata.merge_counts[0, 1, 3].item()) == 1


def check_layer_mode_applies_shared_pair_to_all_heads() -> None:
    k, v, metadata, result = _layer_merge_inputs(head_specific=False)
    config = SvdEvictionMergeConfig(
        enabled=True,
        mode="layer",
        similarity_threshold=0.99,
        candidate_axes=1,
        geometry_gate="none",
        ema_decay=0.0,
        use_depth_confidence=False,
        reps_per_axis=2,
        max_candidates_per_token=2,
    )
    stats = SvdEvictionMerger(config, num_anchor_tokens=1).merge(k, v, metadata, result)
    assert stats.accepted_pairs == 1
    assert torch.allclose(v[0, 0, 2], torch.tensor([11.0, 0.0]))
    assert torch.allclose(v[0, 1, 2], torch.tensor([0.0, 17.0]))
    assert int(metadata.merge_counts[0, 0, 2].item()) == 1
    assert int(metadata.merge_counts[0, 1, 2].item()) == 1
    assert int(metadata.merge_counts[0, 1, 3].item()) == 0


def check_layer_mode_requires_shared_kept_indices() -> None:
    k, v, metadata, result = _layer_merge_inputs(head_specific=False)
    result.kept_candidate_indices = torch.tensor([[[1, 2], [0, 2]]])
    config = SvdEvictionMergeConfig(enabled=True, mode="layer", geometry_gate="none")
    try:
        SvdEvictionMerger(config, num_anchor_tokens=1).merge(k, v, metadata, result)
    except ValueError as exc:
        assert "identical kept indices" in str(exc)
    else:
        raise AssertionError("layer mode accepted per-head kept indices")

def main() -> None:
    check_selection_is_unchanged_by_basis()
    check_merge_updates_retained_kv()
    check_missing_geometry_modes()
    check_voxel_neighbor_radius()
    check_metadata_geometry_pruning()
    check_layer_candidates_shares_candidates_keeps_headwise_match()
    check_layer_mode_applies_shared_pair_to_all_heads()
    check_layer_mode_requires_shared_kept_indices()
    print("svd eviction merge sanity checks passed")


if __name__ == "__main__":
    main()
