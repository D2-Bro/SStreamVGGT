#!/usr/bin/env python3
"""Lightweight checks for SVD leverage eviction granularity."""

from __future__ import annotations

import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.attention import Attention
from streamvggt.layers.eviction import EvictionManager
from streamvggt.layers.recent_merge import KVCacheMetadata


def _make_cache(dtype: torch.dtype = torch.float32):
    torch.manual_seed(7)
    batch_size, num_heads, num_tokens, head_dim = 2, 3, 11, 5
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, num_tokens, head_dim, dtype=dtype)
    return k, v


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor.float()).all()):
        raise AssertionError(f"{name} contains non-finite values")


def check_head_mode_shape() -> None:
    k, v = _make_cache()
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="head",
    )
    result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    assert result.kept_candidate_indices.shape == (2, 3, 5)
    assert result.policy_scores.shape == (2, 3, 9)
    _assert_finite(result.policy_scores, "head policy scores")


def check_layer_score_shape() -> None:
    k, _ = _make_cache()
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="layer",
    )
    scores = manager._layer_svd_leverage_scores(k[:, :, 2:, :])
    assert scores.shape == (2, 9)
    _assert_finite(scores, "layer policy scores")


def check_shared_layer_indices() -> None:
    k, v = _make_cache()
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="layer",
    )
    result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    kept = result.kept_candidate_indices
    assert kept.shape == (2, 3, 5)
    for head_idx in range(1, kept.shape[1]):
        if not torch.equal(kept[:, 0], kept[:, head_idx]):
            raise AssertionError("layer-wise mode did not share kept indices across heads")


def check_eviction_alignment() -> None:
    k, v = _make_cache()
    batch_size, num_heads, num_tokens, head_dim = k.shape
    attention = Attention(dim=num_heads * head_dim, num_heads=num_heads)
    metadata = KVCacheMetadata.for_current_frame(
        batch_size=batch_size,
        num_heads=num_heads,
        num_tokens=num_tokens,
        frame_id=3,
    )
    final_k, final_v, final_metadata, _ = attention.eviction(
        k,
        v,
        metadata,
        cache_budget=7,
        num_anchor_tokens=2,
        eviction_policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="layer",
    )
    assert final_k.shape == (batch_size, num_heads, 7, head_dim)
    assert final_v.shape == final_k.shape
    assert final_metadata is not None
    assert final_metadata.frame_ids.shape == (batch_size, num_heads, 7)


def check_sketch_and_exact_modes() -> None:
    k, v = _make_cache()
    for sketch_dim in (0, 16):
        manager = EvictionManager(
            policy="svd_leverage",
            leverage_sketch_dim=sketch_dim,
            leverage_granularity="layer",
        )
        result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
        assert result.kept_candidate_indices.shape == (2, 3, 5)
        _assert_finite(result.policy_scores, f"layer scores sketch_dim={sketch_dim}")


def check_approx_methods() -> None:
    k, v = _make_cache()
    configs = (
        {"leverage_approx_method": "exact_qr", "leverage_sketch_dim": 16},
        {"leverage_approx_method": "right_sketch", "leverage_sketch_dim": 4},
        {
            "leverage_approx_method": "drineas_srht",
            "leverage_sketch_dim": 4,
            "leverage_left_sketch_dim": 16,
            "leverage_right_jl_dim": 4,
        },
    )
    for granularity in ("head", "layer"):
        for cfg in configs:
            manager = EvictionManager(
                policy="svd_leverage",
                leverage_granularity=granularity,
                **cfg,
            )
            result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v, need_leverage_basis=True)
            assert result.kept_candidate_indices.shape == (2, 3, 5)
            expected_scores = (2, 3, 9) if granularity == "head" else (2, 9)
            assert result.policy_scores.shape == expected_scores
            assert result.leverage_basis is not None
            assert result.leverage_basis.q.shape[-2] == 9
            method_name = cfg["leverage_approx_method"]
            _assert_finite(result.policy_scores, f"{granularity} {method_name} scores")


def check_drineas_deterministic() -> None:
    k, _ = _make_cache()
    kwargs = dict(
        policy="svd_leverage",
        leverage_granularity="head",
        leverage_approx_method="drineas_srht",
        leverage_left_sketch_dim=16,
        leverage_right_jl_dim=4,
        leverage_random_seed=123,
    )
    first = EvictionManager(**kwargs)._svd_leverage_scores(k[:, :, 2:, :])
    second = EvictionManager(**kwargs)._svd_leverage_scores(k[:, :, 2:, :])
    if not torch.allclose(first, second):
        raise AssertionError("drineas_srht leverage scores are not deterministic for a fixed seed")


def check_key_value_feature() -> None:
    k, v = _make_cache()
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="layer",
        leverage_feature="key_value",
    )
    scores = manager._layer_svd_leverage_scores(k[:, :, 2:, :], v[:, :, 2:, :])
    assert scores.shape == (2, 9)
    expected_feature_dim = k.shape[1] * k.shape[3] * 2
    assert manager._last_layer_feature_shape == (9, expected_feature_dim)
    _assert_finite(scores, "key_value layer scores")


def check_low_precision_inputs() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        k, v = _make_cache(dtype=dtype)
        manager = EvictionManager(
            policy="svd_leverage",
            leverage_sketch_dim=4,
            leverage_granularity="layer",
        )
        result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
        _assert_finite(result.policy_scores, f"{dtype} layer scores")


def _frame_metadata(batch_size: int, num_heads: int, frame_ids: torch.Tensor) -> KVCacheMetadata:
    frame_ids = frame_ids.to(dtype=torch.long).view(1, 1, -1).expand(batch_size, num_heads, -1).clone()
    shape = frame_ids.shape
    token_indices = torch.arange(shape[2], dtype=torch.int32).view(1, 1, -1).expand(shape).clone()
    return KVCacheMetadata(
        frame_ids=frame_ids,
        token_indices=token_indices,
        accumulated_confidence=torch.ones(shape, dtype=torch.float32),
        merge_counts=torch.zeros(shape, dtype=torch.int16),
        last_updated_frame=frame_ids.clone(),
    )


def _assert_recent_frames_not_evicted(kept: torch.Tensor, frame_ids: torch.Tensor, protected_frames) -> None:
    candidate_frames = frame_ids[2:]
    protected_local = {
        idx for idx, frame_id in enumerate(candidate_frames.tolist()) if int(frame_id) in protected_frames
    }
    for row in kept.reshape(-1, kept.shape[-1]):
        evicted = set(range(candidate_frames.numel())) - set(row.tolist())
        bad = evicted & protected_local
        if bad:
            raise AssertionError(f"protected candidate indices were evicted: {sorted(bad)}")


def check_recent_frame_protection_head_mode() -> None:
    torch.manual_seed(11)
    batch_size, num_heads, num_tokens, head_dim = 1, 2, 10, 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=0,
        leverage_granularity="head",
    )
    result = manager.select(
        k,
        cache_budget=4,
        num_anchor_tokens=2,
        v=v,
        current_frame_idx=3,
        protect_recent_frames=2,
        candidate_frame_ids=metadata.frame_ids[:, :, 2:],
    )
    assert result.policy_scores.shape[-1] == num_tokens - 2
    _assert_recent_frames_not_evicted(result.kept_candidate_indices, frame_ids, {2, 3})


def check_recent_frame_protection_layer_modes() -> None:
    torch.manual_seed(13)
    batch_size, num_heads, num_tokens, head_dim = 1, 3, 10, 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    for feature in ("key", "key_value"):
        manager = EvictionManager(
            policy="svd_leverage",
            leverage_sketch_dim=4,
            leverage_granularity="layer",
            leverage_feature=feature,
        )
        result = manager.select(
            k,
            cache_budget=4,
            num_anchor_tokens=2,
            v=v,
            current_frame_idx=3,
            protect_recent_frames=2,
            candidate_frame_ids=metadata.frame_ids[:, :, 2:],
        )
        assert result.policy_scores.shape[-1] == num_tokens - 2
        _assert_recent_frames_not_evicted(result.kept_candidate_indices, frame_ids, {2, 3})
        for head_idx in range(1, result.kept_candidate_indices.shape[1]):
            if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
                raise AssertionError("layer-wise protected eviction did not share kept indices across heads")



def _assert_evicted_subset_of_low_score_pool(result, candidate_count: int, multiplier: int) -> None:
    kept = result.kept_candidate_indices
    all_candidates = set(range(candidate_count))
    if result.policy_scores.ndim == 2:
        for batch_idx in range(kept.shape[0]):
            evicted = all_candidates - set(kept[batch_idx, 0].tolist())
            pool_size = min(candidate_count, max(len(evicted), len(evicted) * multiplier))
            low_pool = set(torch.argsort(result.policy_scores[batch_idx], stable=True)[:pool_size].tolist())
            if not evicted <= low_pool:
                raise AssertionError(f"layer fast_dpp evicted outside low-score pool: {sorted(evicted - low_pool)}")
    else:
        for batch_idx in range(kept.shape[0]):
            for head_idx in range(kept.shape[1]):
                evicted = all_candidates - set(kept[batch_idx, head_idx].tolist())
                pool_size = min(candidate_count, max(len(evicted), len(evicted) * multiplier))
                low_pool = set(torch.argsort(result.policy_scores[batch_idx, head_idx], stable=True)[:pool_size].tolist())
                if not evicted <= low_pool:
                    raise AssertionError(f"head fast_dpp evicted outside low-score pool: {sorted(evicted - low_pool)}")


def check_fast_dpp_shapes_and_low_score_pool() -> None:
    k, v = _make_cache()
    multiplier = 2
    for granularity in ("head", "layer"):
        for block_size in (1, 32):
            manager = EvictionManager(
                policy="svd_leverage",
                leverage_sketch_dim=4,
                leverage_granularity=granularity,
                leverage_eviction_selector="fast_dpp",
                leverage_dpp_candidate_multiplier=multiplier,
                leverage_dpp_greedy_block_size=block_size,
            )
            result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
            assert result.kept_candidate_indices.shape == (2, 3, 5)
            expected_scores = (2, 3, 9) if granularity == "head" else (2, 9)
            assert result.policy_scores.shape == expected_scores
            _assert_finite(result.policy_scores, f"{granularity} fast_dpp block={block_size} scores")
            _assert_evicted_subset_of_low_score_pool(result, candidate_count=9, multiplier=multiplier)
            if granularity == "layer":
                for head_idx in range(1, result.kept_candidate_indices.shape[1]):
                    if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
                        raise AssertionError("layer-wise fast_dpp did not share kept indices across heads")


def check_fast_dpp_retain_diversity() -> None:
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
    )
    scores = torch.tensor([0.0, 0.0, 0.05, 0.06], dtype=torch.float32)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    evicted = set(
        manager._fast_dpp_evicted_indices(
            scores,
            torch.ones_like(scores, dtype=torch.bool),
            2,
            lambda idx: features.index_select(0, idx),
        ).tolist()
    )
    retained = set(range(scores.numel())) - evicted
    if retained != {2, 3}:
        raise AssertionError(
            "fast_dpp should retain diverse higher-score pool representatives, "
            f"got retained={sorted(retained)} evicted={sorted(evicted)}"
        )


def check_fast_dpp_recent_frame_protection() -> None:
    torch.manual_seed(19)
    batch_size, num_heads, num_tokens, head_dim = 1, 3, 10, 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="layer",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=32,
    )
    result = manager.select(
        k,
        cache_budget=4,
        num_anchor_tokens=2,
        v=v,
        current_frame_idx=3,
        protect_recent_frames=2,
        candidate_frame_ids=metadata.frame_ids[:, :, 2:],
    )
    _assert_recent_frames_not_evicted(result.kept_candidate_indices, frame_ids, {2, 3})
    for head_idx in range(1, result.kept_candidate_indices.shape[1]):
        if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
            raise AssertionError("layer-wise protected fast_dpp eviction did not share kept indices across heads")


def check_dpp_shapes_and_full_pool() -> None:
    k, v = _make_cache()
    for granularity in ("head", "layer"):
        manager = EvictionManager(
            policy="dpp",
            leverage_granularity=granularity,
            leverage_eviction_selector="topk",
            leverage_dpp_candidate_multiplier=1,
            leverage_dpp_greedy_block_size=1,
        )
        result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
        assert result.kept_candidate_indices.shape == (2, 3, 5)
        expected_scores = (2, 3, 9) if granularity == "head" else (2, 9)
        assert result.policy_scores.shape == expected_scores
        if not torch.equal(result.policy_scores, torch.zeros_like(result.policy_scores)):
            raise AssertionError("dpp policy should use neutral policy scores")
        if granularity == "layer":
            for head_idx in range(1, result.kept_candidate_indices.shape[1]):
                if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
                    raise AssertionError("layer-wise dpp did not share kept indices across heads")

    k = torch.zeros(1, 1, 8, 2, dtype=torch.float32)
    v = torch.zeros_like(k)
    k[0, 0, 2:] = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    manager = EvictionManager(
        policy="dpp",
        leverage_granularity="head",
        leverage_dpp_candidate_multiplier=1,
        leverage_dpp_greedy_block_size=1,
    )
    result = manager.select(k, cache_budget=6, num_anchor_tokens=2, v=v)
    evicted = set(range(6)) - set(result.kept_candidate_indices[0, 0].tolist())
    low_score_pool = {0, 1}
    if evicted <= low_score_pool:
        raise AssertionError(f"dpp-only eviction was restricted to low-score pool: {sorted(evicted)}")


def check_dpp_recent_frame_protection() -> None:
    torch.manual_seed(23)
    batch_size, num_heads, num_tokens, head_dim = 1, 3, 10, 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    manager = EvictionManager(
        policy="dpp",
        leverage_granularity="layer",
        leverage_eviction_selector="topk",
        leverage_dpp_greedy_block_size=1,
    )
    result = manager.select(
        k,
        cache_budget=4,
        num_anchor_tokens=2,
        v=v,
        current_frame_idx=3,
        protect_recent_frames=2,
        candidate_frame_ids=metadata.frame_ids[:, :, 2:],
    )
    _assert_recent_frames_not_evicted(result.kept_candidate_indices, frame_ids, {2, 3})
    for head_idx in range(1, result.kept_candidate_indices.shape[1]):
        if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
            raise AssertionError("layer-wise protected dpp eviction did not share kept indices across heads")


def check_recent_frame_eviction_alignment() -> None:
    torch.manual_seed(17)
    batch_size, num_heads, num_tokens, head_dim = 1, 2, 10, 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    attention = Attention(dim=num_heads * head_dim, num_heads=num_heads)
    final_k, final_v, final_metadata, _ = attention.eviction(
        k,
        v,
        metadata,
        cache_budget=4,
        num_anchor_tokens=2,
        layer_id=0,
        step_idx=3,
        eviction_policy="svd_leverage",
        leverage_sketch_dim=0,
        leverage_granularity="head",
        eviction_protect_recent_frames=2,
    )
    assert final_k.shape[2] == final_v.shape[2] == final_metadata.frame_ids.shape[2]
    assert final_k.shape[2] > 4
    remaining_frames = final_metadata.frame_ids[0, 0].tolist()
    for protected_frame in (2, 3):
        if protected_frame not in remaining_frames:
            raise AssertionError(f"protected frame {protected_frame} disappeared after eviction")


def check_protection_disabled_matches_previous_behavior() -> None:
    k, v = _make_cache()
    metadata = KVCacheMetadata.for_current_frame(k.shape[0], k.shape[1], k.shape[2], frame_id=3)
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="head",
    )
    baseline = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    disabled = manager.select(
        k,
        cache_budget=7,
        num_anchor_tokens=2,
        v=v,
        current_frame_idx=3,
        protect_recent_frames=0,
        candidate_frame_ids=metadata.frame_ids[:, :, 2:],
    )
    if not torch.equal(baseline.kept_candidate_indices, disabled.kept_candidate_indices):
        raise AssertionError("protect_recent_frames=0 changed eviction indices")


def check_keep_after_eviction_zero_evict() -> None:
    scores = torch.randn(2, 3, 5)
    mask = torch.ones_like(scores, dtype=torch.bool)
    kept = EvictionManager._keep_after_eviction(scores, mask, num_to_evict=0, evict_highest=False)
    expected = torch.arange(5).view(1, 1, 5).expand(2, 3, 5)
    if not torch.equal(kept, expected):
        raise AssertionError("zero-eviction fast path did not return all candidate indices")


def check_small_cache_no_eviction() -> None:
    k, v = _make_cache()
    attention = Attention(dim=k.shape[1] * k.shape[3], num_heads=k.shape[1])
    metadata = KVCacheMetadata.for_current_frame(
        batch_size=k.shape[0],
        num_heads=k.shape[1],
        num_tokens=k.shape[2],
        frame_id=0,
    )
    final_k, final_v, final_metadata, score = attention.eviction(
        k,
        v,
        metadata,
        cache_budget=k.shape[2],
        num_anchor_tokens=2,
        eviction_policy="svd_leverage",
        leverage_sketch_dim=4,
        leverage_granularity="layer",
    )
    assert final_k is k
    assert final_v is v
    assert final_metadata is metadata
    assert score == 0.0


def main() -> None:
    check_head_mode_shape()
    check_layer_score_shape()
    check_shared_layer_indices()
    check_eviction_alignment()
    check_sketch_and_exact_modes()
    check_approx_methods()
    check_drineas_deterministic()
    check_fast_dpp_shapes_and_low_score_pool()
    check_fast_dpp_retain_diversity()
    check_fast_dpp_recent_frame_protection()
    check_dpp_shapes_and_full_pool()
    check_dpp_recent_frame_protection()
    check_key_value_feature()
    check_low_precision_inputs()
    check_recent_frame_protection_head_mode()
    check_recent_frame_protection_layer_modes()
    check_recent_frame_eviction_alignment()
    check_protection_disabled_matches_previous_behavior()
    check_keep_after_eviction_zero_evict()
    check_small_cache_no_eviction()
    print("leverage granularity sanity checks passed")


if __name__ == "__main__":
    main()
