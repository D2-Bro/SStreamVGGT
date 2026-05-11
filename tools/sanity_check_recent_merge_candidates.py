"""Lightweight CPU sanity checks for local recent KV merge candidate modes."""

from __future__ import annotations

import torch

from streamvggt.layers.recent_merge import (
    FrameGeometry,
    KVCacheMetadata,
    RecentMergeConfig,
    RecentSimilarityMerge,
)


def _metadata(num_tokens: int, current_frame: int = 1) -> KVCacheMetadata:
    prev = KVCacheMetadata.for_current_frame(1, 1, num_tokens, current_frame - 1)
    cur = KVCacheMetadata.for_current_frame(1, 1, num_tokens, current_frame)
    return prev.concat(cur)


def _geometry(voxels, patch_height: int, patch_width: int, confidence=None) -> FrameGeometry:
    voxel_tensor = torch.tensor(voxels, dtype=torch.int32).view(1, -1, 3)
    tokens = voxel_tensor.shape[1]
    if confidence is None:
        confidence_tensor = torch.ones((1, tokens), dtype=torch.float32)
    else:
        confidence_tensor = torch.tensor(confidence, dtype=torch.float32).view(1, tokens)
    valid = torch.ones((1, tokens), dtype=torch.bool)
    return FrameGeometry(
        voxel_ids=voxel_tensor,
        confidence=confidence_tensor,
        valid=valid,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_tokens=tokens,
        tokens_per_frame=tokens,
    )


def _merger(config: RecentMergeConfig, prev_geom: FrameGeometry, cur_geom: FrameGeometry) -> RecentSimilarityMerge:
    merger = RecentSimilarityMerge(config, patch_start_idx=0, patch_size=1)
    merger._geometry[0] = prev_geom
    merger._geometry[1] = cur_geom
    return merger


def _local_masks(merger: RecentSimilarityMerge, metadata: KVCacheMetadata, cur_token_ids, cand_token_ids):
    cur_pos = torch.tensor([metadata.frame_ids.shape[2] // 2 + t for t in cur_token_ids], dtype=torch.long)
    cand_pos = torch.tensor(cand_token_ids, dtype=torch.long)
    cur_vox, cur_valid, _ = merger._lookup_geometry(metadata, 0, 0, cur_pos)
    cand_vox, cand_valid, cand_conf = merger._lookup_geometry(metadata, 0, 0, cand_pos)
    return merger._build_local_candidate_masks(
        metadata,
        0,
        0,
        cur_pos,
        cand_pos,
        cur_vox,
        cand_vox,
        cur_valid,
        cand_valid,
        cand_conf,
    )


def test_spatial_candidate_generation():
    voxels = [[0, 0, 0]] * 9
    metadata = _metadata(9)
    geom = _geometry(voxels, 3, 3)
    merger = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="spatial", patch_radius=1),
        geom,
        geom,
    )
    mask, _, _ = _local_masks(merger, metadata, cur_token_ids=[0], cand_token_ids=list(range(9)))
    assert set(torch.nonzero(mask[0], as_tuple=False).flatten().tolist()) == {0, 1, 3, 4}


def test_voxel_candidate_generation():
    prev = _geometry([[1, 1, 1], [2, 1, 1], [5, 5, 5]], 1, 3)
    cur = _geometry([[1, 1, 1], [9, 9, 9], [5, 5, 6]], 1, 3)
    metadata = _metadata(3)

    exact = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="voxel", voxel_neighbor_radius=0, local_fallback=False),
        prev,
        cur,
    )
    mask, _, _ = _local_masks(exact, metadata, cur_token_ids=[0], cand_token_ids=[0, 1, 2])
    assert torch.nonzero(mask[0], as_tuple=False).flatten().tolist() == [0]

    none, _, _ = _local_masks(exact, metadata, cur_token_ids=[1], cand_token_ids=[0, 1, 2])
    assert int(none.sum().item()) == 0

    neighbor = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="voxel", voxel_neighbor_radius=1, local_fallback=False),
        prev,
        cur,
    )
    mask, _, _ = _local_masks(neighbor, metadata, cur_token_ids=[0], cand_token_ids=[0, 1, 2])
    assert set(torch.nonzero(mask[0], as_tuple=False).flatten().tolist()) == {0, 1}


def _kv_for_merge(num_tokens: int = 4):
    metadata = _metadata(num_tokens)
    base = torch.eye(num_tokens, dtype=torch.float32)
    k = torch.cat([base, base], dim=0).view(1, 1, num_tokens * 2, num_tokens)
    v = (torch.cat([base, base], dim=0) + 1.0).view(1, 1, num_tokens * 2, num_tokens)
    voxels = [[idx, 0, 0] for idx in range(num_tokens)]
    geom = _geometry(voxels, 2, 2)
    return k, v, metadata, geom


def test_full_and_local_merge_behavior():
    k, v, metadata, geom = _kv_for_merge()
    full = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="full", similarity_threshold=0.99),
        geom,
        geom,
    )
    k_full, v_full, meta_full, stats_full = full.merge_layer(k.clone(), v.clone(), metadata, 0, 1)

    spatial = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="spatial", patch_radius=8, similarity_threshold=0.99),
        geom,
        geom,
    )
    k_spatial, v_spatial, meta_spatial, stats_spatial = spatial.merge_layer(
        k.clone(), v.clone(), metadata, 0, 1
    )

    assert stats_full.merged_tokens == stats_spatial.merged_tokens == 4
    assert k_full.shape == k_spatial.shape == (1, 1, 4, 4)
    assert v_full.shape == v_spatial.shape == (1, 1, 4, 4)
    assert meta_full.frame_ids.shape == meta_spatial.frame_ids.shape == (1, 1, 4)


def test_high_threshold_no_candidates_and_empty_recent_cache():
    k, v, metadata, geom = _kv_for_merge()
    high = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="full", similarity_threshold=0.99999),
        geom,
        geom,
    )
    noisy_k = k.clone()
    noisy_k[:, :, 4:] = torch.roll(noisy_k[:, :, 4:], shifts=1, dims=-1)
    k_out, v_out, meta_out, stats = high.merge_layer(noisy_k, v.clone(), metadata, 0, 1)
    assert stats.merged_tokens == 0
    assert k_out.shape == v_out.shape == (1, 1, 8, 4)
    assert meta_out.frame_ids.shape == (1, 1, 8)

    empty_metadata = KVCacheMetadata.for_current_frame(1, 1, 4, 1)
    empty_merger = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="spatial"),
        geom,
        geom,
    )
    k_cur = k[:, :, :4].clone()
    v_cur = v[:, :, :4].clone()
    k_empty, v_empty, meta_empty, empty_stats = empty_merger.merge_layer(k_cur, v_cur, empty_metadata, 0, 1)
    assert empty_stats.merged_tokens == 0
    assert k_empty.shape == v_empty.shape == (1, 1, 4, 4)
    assert meta_empty.frame_ids.shape == (1, 1, 4)


def test_dtype_shape_metadata_and_no_nans():
    k, v, metadata, geom = _kv_for_merge()
    metadata.accumulated_confidence[0, 0, 0] = float("nan")
    nan_conf_geom = _geometry([[idx, 0, 0] for idx in range(4)], 2, 2, confidence=[float("nan"), 1, 1, 1])
    merger = _merger(
        RecentMergeConfig(enabled=True, candidate_mode="full", similarity_threshold=0.99),
        geom,
        nan_conf_geom,
    )
    k_out, v_out, meta_out, stats = merger.merge_layer(k.clone(), v.clone(), metadata, 0, 1)
    assert stats.merged_tokens == 4
    assert k_out.dtype == k.dtype
    assert v_out.dtype == v.dtype
    assert k_out.device == k.device
    assert v_out.device == v.device
    assert k_out.shape[2] == v_out.shape[2] == meta_out.frame_ids.shape[2]
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()
    assert not torch.isnan(meta_out.accumulated_confidence).any()


def main():
    test_spatial_candidate_generation()
    test_voxel_candidate_generation()
    test_full_and_local_merge_behavior()
    test_high_threshold_no_candidates_and_empty_recent_cache()
    test_dtype_shape_metadata_and_no_nans()
    print("recent merge candidate sanity checks passed")


if __name__ == "__main__":
    main()
