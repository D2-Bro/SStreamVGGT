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
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig


def _make_cache(dtype: torch.dtype = torch.float32):
    torch.manual_seed(7)
    batch_size, num_heads, num_tokens, head_dim = 2, 3, 11, 5
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, num_tokens, head_dim, dtype=dtype)
    return k, v


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor.float()).all()):
        raise AssertionError(f"{name} contains non-finite values")


def _assert_nonnegative(tensor: torch.Tensor, name: str) -> None:
    if bool((tensor.float() < -1e-6).any().item()):
        raise AssertionError(f"{name} contains negative values")


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
        {"leverage_approx_method": "full_d_ridge", "leverage_sketch_dim": 4},
        {
            "leverage_approx_method": "right_sketch_ridge",
            "leverage_sketch_dim": 4,
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
            _assert_nonnegative(result.policy_scores, f"{granularity} {method_name} scores")


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
    methods = (
        {"leverage_approx_method": "right_sketch", "leverage_sketch_dim": 4},
        {"leverage_approx_method": "full_d_ridge", "leverage_sketch_dim": 4},
        {
            "leverage_approx_method": "right_sketch_ridge",
            "leverage_sketch_dim": 4,
            "leverage_right_jl_dim": 4,
        },
    )
    for granularity in ("head", "layer"):
        for block_size in (1, 32):
            for method_cfg in methods:
                manager = EvictionManager(
                    policy="svd_leverage",
                    leverage_granularity=granularity,
                    leverage_eviction_selector="fast_dpp",
                    leverage_dpp_candidate_multiplier=multiplier,
                    leverage_dpp_greedy_block_size=block_size,
                    **method_cfg,
                )
                result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
                assert result.kept_candidate_indices.shape == (2, 3, 5)
                expected_scores = (2, 3, 9) if granularity == "head" else (2, 9)
                assert result.policy_scores.shape == expected_scores
                method = method_cfg["leverage_approx_method"]
                _assert_finite(result.policy_scores, f"{granularity} {method} fast_dpp block={block_size} scores")
                _assert_nonnegative(result.policy_scores, f"{granularity} {method} fast_dpp block={block_size} scores")
                _assert_evicted_subset_of_low_score_pool(result, candidate_count=9, multiplier=multiplier)
                if granularity == "layer":
                    for head_idx in range(1, result.kept_candidate_indices.shape[1]):
                        if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
                            raise AssertionError("layer-wise fast_dpp did not share kept indices across heads")


def _reference_layer_score_head_dpp(
    manager: EvictionManager,
    scores: torch.Tensor,
    evictable_mask: torch.Tensor,
    num_to_evict: int,
    candidate_k: torch.Tensor,
    candidate_v: torch.Tensor,
) -> torch.Tensor:
    B, H, N, _ = candidate_k.shape
    all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
    kept = torch.empty(B, H, N - num_to_evict, device=scores.device, dtype=torch.long)
    for batch_idx in range(B):
        for head_idx in range(H):
            def feature_fn(indices):
                features = candidate_k[batch_idx, head_idx].index_select(0, indices)
                if manager.leverage_feature == "key_value":
                    features = torch.cat(
                        [features, candidate_v[batch_idx, head_idx].index_select(0, indices)],
                        dim=-1,
                    )
                return features

            evicted = manager._fast_dpp_evicted_indices(
                scores[batch_idx],
                evictable_mask[batch_idx, head_idx],
                num_to_evict,
                feature_fn,
            )
            keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
            keep_mask[evicted] = False
            kept[batch_idx, head_idx] = all_indices[keep_mask]
    return kept


def check_layer_head_fast_dpp_matches_reference() -> None:
    k, v = _make_cache()
    candidate_k = k[:, :, 2:]
    candidate_v = v[:, :, 2:]
    B, H, N, _ = candidate_k.shape
    scores = torch.linspace(0.0, 1.0, N).view(1, N).expand(B, N)
    mask = torch.ones(B, H, N, dtype=torch.bool)
    for feature in ("key", "key_value"):
        manager = EvictionManager(
            policy="svd_leverage",
            leverage_granularity="layer",
            leverage_feature=feature,
            leverage_eviction_selector="layer_head_fast_dpp",
            leverage_dpp_candidate_multiplier=2,
            leverage_dpp_greedy_block_size=1,
        )
        actual = manager._keep_after_layer_head_fast_dpp(scores, mask, 4, candidate_k, candidate_v)
        expected = _reference_layer_score_head_dpp(manager, scores, mask, 4, candidate_k, candidate_v)
        if not torch.equal(actual, expected):
            raise AssertionError(f"vectorized layer_head_fast_dpp differs from sequential reference for {feature}")


def check_layer_head_fast_dpp_head_specific_and_protection() -> None:
    scores = torch.tensor([[0.0, 0.1, 0.2, 0.3, 1.0, 2.0]])
    candidate_k = torch.zeros(1, 2, 6, 2)
    candidate_k[0, 0] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    candidate_k[0, 1] = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="layer_head_fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_diversity_beta=3.0,
    )
    kept = manager._keep_after_layer_head_fast_dpp(
        scores,
        torch.ones(1, 2, 6, dtype=torch.bool),
        2,
        candidate_k,
        None,
    )
    if torch.equal(kept[:, 0], kept[:, 1]):
        raise AssertionError("layer_head_fast_dpp did not produce head-specific keep sets")

    k, v = _make_cache()
    candidate_count = k.shape[2] - 2
    mask = torch.ones(k.shape[0], k.shape[1], candidate_count, dtype=torch.bool)
    mask[0, 0, :3] = False
    mask[0, 1, 3:6] = False
    mask[1, 2, 6:] = False
    result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v, candidate_evictable_mask=mask)
    for batch_idx in range(k.shape[0]):
        for head_idx in range(k.shape[1]):
            protected = set(torch.nonzero(~mask[batch_idx, head_idx], as_tuple=False).flatten().tolist())
            missing = protected - set(result.kept_candidate_indices[batch_idx, head_idx].tolist())
            if missing:
                raise AssertionError(f"layer_head_fast_dpp evicted protected head tokens: {sorted(missing)}")

    limited_mask = torch.zeros_like(mask)
    limited_mask[..., -1] = True
    limited = manager.select(k, cache_budget=5, num_anchor_tokens=2, v=v, candidate_evictable_mask=limited_mask)
    if limited.kept_candidate_indices.shape[-1] != candidate_count - 1:
        raise AssertionError("layer_head_fast_dpp did not limit eviction to the minimum head-wise capacity")


def check_layer_head_fast_dpp_recent_protection_and_edge_cases() -> None:
    torch.manual_seed(53)
    batch_size, num_heads, num_tokens, head_dim = 1, 3, 10, 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_sketch_dim=4,
        leverage_eviction_selector="layer_head_fast_dpp",
        leverage_dpp_candidate_multiplier=2,
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

    scores = torch.zeros(1, 6)
    candidate_k = torch.randn(1, 2, 6, 3)
    mask = torch.ones(1, 2, 6, dtype=torch.bool)
    kept = manager._keep_after_layer_head_fast_dpp(scores, mask, 2, candidate_k, None)
    assert kept.shape == (1, 2, 4)
    keep_all = manager._keep_after_layer_head_fast_dpp(scores, mask, 0, candidate_k, None)
    expected = torch.arange(6).view(1, 1, 6).expand(1, 2, 6)
    if not torch.equal(keep_all, expected):
        raise AssertionError("layer_head_fast_dpp zero-eviction path did not retain every token")


def check_layer_head_fast_dpp_invalid_configs_and_merge() -> None:
    invalid = (
        {"policy": "svd_leverage", "leverage_granularity": "head"},
        {"policy": "dpp", "leverage_granularity": "layer"},
    )
    for config in invalid:
        try:
            EvictionManager(leverage_eviction_selector="layer_head_fast_dpp", **config)
        except ValueError:
            continue
        raise AssertionError(f"invalid layer_head_fast_dpp config was accepted: {config}")

    k, v = _make_cache()
    attention = Attention(dim=k.shape[1] * k.shape[3], num_heads=k.shape[1])
    metadata = KVCacheMetadata.for_current_frame(k.shape[0], k.shape[1], k.shape[2], frame_id=0)
    try:
        attention.eviction(
            k,
            v,
            metadata,
            cache_budget=7,
            num_anchor_tokens=2,
            eviction_policy="svd_leverage",
            leverage_granularity="layer",
            leverage_eviction_selector="layer_head_fast_dpp",
            svd_eviction_merge_config=SvdEvictionMergeConfig(enabled=True, mode="layer"),
        )
    except ValueError:
        return
    raise AssertionError("layer_head_fast_dpp accepted a shared-set SVD merge mode")


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


def check_fast_dpp_diversity_beta_zero_prefers_quality() -> None:
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_diversity_beta=0.0,
    )
    scores = torch.tensor([0.0, 0.4, 0.9, 1.0], dtype=torch.float32)
    features = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
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
            "fast_dpp with diversity beta 0 should retain the highest-quality pool tokens, "
            f"got retained={sorted(retained)} evicted={sorted(evicted)}"
        )


def check_fast_dpp_quality_beta_zero_uses_only_diversity() -> None:
    scores = torch.tensor([0.0, 0.4, 0.9, 1.0], dtype=torch.float32)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_quality_beta=0.0,
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
    if retained != {0, 2}:
        raise AssertionError(
            "fast_dpp with quality beta 0 should retain pure-diversity representatives, "
            f"got retained={sorted(retained)} evicted={sorted(evicted)}"
        )

    layer_head_manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="layer_head_fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_quality_beta=0.0,
    )
    kept = layer_head_manager._keep_after_layer_head_fast_dpp(
        scores.unsqueeze(0),
        torch.ones(1, 1, scores.numel(), dtype=torch.bool),
        2,
        features.view(1, 1, scores.numel(), -1),
        None,
    )
    retained = set(kept[0, 0].tolist())
    if retained != {0, 2}:
        raise AssertionError(
            "layer_head_fast_dpp with quality beta 0 should retain pure-diversity representatives, "
            f"got retained={sorted(retained)}"
        )


def check_fast_dpp_recency_prior() -> None:
    scores = torch.zeros(4, dtype=torch.float32)
    features = torch.eye(4, dtype=torch.float32)
    mask = torch.ones_like(scores, dtype=torch.bool)
    frame_ids = torch.tensor([0, 5, 0, 0], dtype=torch.long)

    baseline = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_diversity_beta=0.0,
    )
    baseline_evicted = set(
        baseline._fast_dpp_evicted_indices(
            scores,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
        ).tolist()
    )

    recency = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_diversity_beta=0.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=0.2,
        leverage_dpp_recency_window=5,
    )
    if set(
        recency._fast_dpp_evicted_indices(
            scores,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
            current_frame_id=5,
        ).tolist()
    ) != baseline_evicted:
        raise AssertionError("missing frame ids should fall back to baseline fast_dpp")
    if set(
        recency._fast_dpp_evicted_indices(
            scores,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
            row_frame_ids=frame_ids,
        ).tolist()
    ) != baseline_evicted:
        raise AssertionError("missing current frame id should fall back to baseline fast_dpp")

    zero_lambda = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_diversity_beta=0.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=0.0,
    )
    if set(
        zero_lambda._fast_dpp_evicted_indices(
            scores,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
            row_frame_ids=frame_ids,
            current_frame_id=5,
        ).tolist()
    ) != baseline_evicted:
        raise AssertionError("recency lambda 0 should match disabled recency")

    evicted = set(
        recency._fast_dpp_evicted_indices(
            scores,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
            row_frame_ids=frame_ids,
            current_frame_id=5,
        ).tolist()
    )
    retained = set(range(scores.numel())) - evicted
    if retained != {1}:
        raise AssertionError(f"recency prior should retain the recent tied low-score token, got {sorted(retained)}")

    layer_head = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="layer_head_fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_diversity_beta=0.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=0.2,
        leverage_dpp_recency_window=5,
    )
    candidate_k = features.view(1, 1, 4, 4).expand(1, 2, 4, 4).contiguous()
    kept_2d = layer_head._keep_after_layer_head_fast_dpp(
        scores.view(1, 4),
        torch.ones(1, 2, 4, dtype=torch.bool),
        3,
        candidate_k,
        None,
        candidate_frame_ids=frame_ids.view(1, 4),
        current_frame_id=5,
    )
    if not torch.equal(kept_2d, torch.ones_like(kept_2d)):
        raise AssertionError("[B, N] frame ids should expand across heads and retain index 1")

    head_frame_ids = frame_ids.view(1, 1, 4).expand(1, 2, 4).clone()
    kept_3d = layer_head._keep_after_layer_head_fast_dpp(
        scores.view(1, 4),
        torch.ones(1, 2, 4, dtype=torch.bool),
        3,
        candidate_k,
        None,
        candidate_frame_ids=head_frame_ids,
        current_frame_id=5,
    )
    if not torch.equal(kept_3d, torch.ones_like(kept_3d)):
        raise AssertionError("[B, H, N] frame ids should retain index 1")

    high_score = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    high_evicted = set(
        recency._fast_dpp_evicted_indices(
            high_score,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
            row_frame_ids=frame_ids,
            current_frame_id=5,
        ).tolist()
    )
    high_retained = set(range(high_score.numel())) - high_evicted
    if high_retained != {2}:
        raise AssertionError(
            "high-score token should remain preferred because recency is gated by low score, "
            f"got {sorted(high_retained)}"
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


def _evictable_only_configs():
    return (
        {"leverage_approx_method": "exact_qr", "leverage_sketch_dim": 0},
        {"leverage_approx_method": "right_sketch", "leverage_sketch_dim": 3},
        {
            "leverage_approx_method": "right_sketch_ridge",
            "leverage_sketch_dim": 3,
            "leverage_ridge_dim": 3,
        },
    )


def check_evictable_only_matches_subset_reference() -> None:
    k, v = _make_cache()
    num_anchor_tokens = 2
    candidate_k = k[:, :, num_anchor_tokens:]
    candidate_v = v[:, :, num_anchor_tokens:]
    mask = torch.ones(candidate_k.shape[:3], dtype=torch.bool)
    mask[0, 0, :2] = False
    mask[0, 1, :6] = False
    mask[1, 0, 3:5] = False
    mask[1, 1, 5:] = False

    for granularity in ("head", "layer"):
        for config in _evictable_only_configs():
            manager = EvictionManager(
                policy="svd_leverage",
                leverage_granularity=granularity,
                leverage_evictable_only=True,
                **config,
            )
            result = manager.select(
                k,
                cache_budget=7,
                num_anchor_tokens=num_anchor_tokens,
                v=v,
                candidate_evictable_mask=mask,
                need_leverage_basis=True,
            )
            basis = result.leverage_basis
            assert basis is not None
            assert basis.q.shape[-2] == candidate_k.shape[2]

            if granularity == "head":
                if not torch.equal(result.policy_scores[~mask], torch.zeros_like(result.policy_scores[~mask])):
                    raise AssertionError("protected head-wise rows received non-zero leverage scores")
                if not torch.equal(basis.q[~mask], torch.zeros_like(basis.q[~mask])):
                    raise AssertionError("protected head-wise rows received non-zero leverage coordinates")
                for batch_idx in range(k.shape[0]):
                    for head_idx in range(k.shape[1]):
                        indices = torch.nonzero(mask[batch_idx, head_idx], as_tuple=False).flatten()
                        reference = EvictionManager(
                            policy="svd_leverage",
                            leverage_granularity="head",
                            **config,
                        )._svd_leverage_scores(
                            candidate_k[batch_idx : batch_idx + 1, head_idx : head_idx + 1].index_select(2, indices)
                        )[0, 0]
                        if not torch.allclose(result.policy_scores[batch_idx, head_idx, indices], reference, atol=1e-5, rtol=1e-4):
                            raise AssertionError(f"head evictable-only scores differ from subset reference: {config}")
            else:
                shared_mask = mask.all(dim=1)
                if not torch.equal(result.policy_scores[~shared_mask], torch.zeros_like(result.policy_scores[~shared_mask])):
                    raise AssertionError("protected layer-wise rows received non-zero leverage scores")
                if not torch.equal(basis.q[~shared_mask], torch.zeros_like(basis.q[~shared_mask])):
                    raise AssertionError("protected layer-wise rows received non-zero leverage coordinates")
                for batch_idx in range(k.shape[0]):
                    indices = torch.nonzero(shared_mask[batch_idx], as_tuple=False).flatten()
                    reference = EvictionManager(
                        policy="svd_leverage",
                        leverage_granularity="layer",
                        **config,
                    )._layer_svd_leverage_scores(
                        candidate_k[batch_idx : batch_idx + 1].index_select(2, indices),
                        candidate_v[batch_idx : batch_idx + 1].index_select(2, indices),
                    )[0]
                    if not torch.allclose(result.policy_scores[batch_idx, indices], reference, atol=1e-5, rtol=1e-4):
                        raise AssertionError(f"layer evictable-only scores differ from subset reference: {config}")


def check_evictable_only_disabled_and_recent_only_are_unchanged() -> None:
    k, v = _make_cache()
    metadata = KVCacheMetadata.for_current_frame(k.shape[0], k.shape[1], k.shape[2], frame_id=3)
    config = dict(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=3,
        leverage_sketch_dim=3,
    )
    baseline = EvictionManager(**config).select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    enabled_without_mask = EvictionManager(leverage_evictable_only=True, **config).select(
        k,
        cache_budget=7,
        num_anchor_tokens=2,
        v=v,
    )
    enabled_recent_only = EvictionManager(leverage_evictable_only=True, **config).select(
        k,
        cache_budget=7,
        num_anchor_tokens=2,
        v=v,
        current_frame_idx=3,
        protect_recent_frames=1,
        candidate_frame_ids=metadata.frame_ids[:, :, 2:],
    )
    if not torch.equal(baseline.policy_scores, enabled_without_mask.policy_scores):
        raise AssertionError("evictable-only without a special-token mask changed leverage scores")
    if not torch.equal(baseline.policy_scores, enabled_recent_only.policy_scores):
        raise AssertionError("recent-frame protection unexpectedly changed evictable-only leverage scores")


def check_evictable_only_empty_mask() -> None:
    k, v = _make_cache()
    candidate_count = k.shape[2] - 2
    mask = torch.zeros(k.shape[0], k.shape[1], candidate_count, dtype=torch.bool)
    for granularity in ("head", "layer"):
        result = EvictionManager(
            policy="svd_leverage",
            leverage_granularity=granularity,
            leverage_evictable_only=True,
            leverage_sketch_dim=0,
        ).select(
            k,
            cache_budget=7,
            num_anchor_tokens=2,
            v=v,
            candidate_evictable_mask=mask,
            need_leverage_basis=True,
        )
        assert result.leverage_basis is not None
        assert result.leverage_basis.q.shape[-2:] == (candidate_count, 0)
        if not torch.equal(result.policy_scores, torch.zeros_like(result.policy_scores)):
            raise AssertionError("empty evictable mask did not produce zero leverage scores")
        if result.kept_candidate_indices.shape[-1] != candidate_count:
            raise AssertionError("empty evictable mask should retain every candidate")


def _make_multiframe_metadata(batch_size: int, num_heads: int, tokens_per_frame: int, num_frames: int) -> KVCacheMetadata:
    metadata = KVCacheMetadata.for_current_frame(batch_size, num_heads, tokens_per_frame, 0)
    for frame_id in range(1, num_frames):
        metadata = metadata.concat(
            KVCacheMetadata.for_current_frame(batch_size, num_heads, tokens_per_frame, frame_id)
        )
    return metadata


def _assert_candidate_indices_retained(kept: torch.Tensor, required) -> None:
    required = set(required)
    for row in kept.reshape(-1, kept.shape[-1]):
        missing = required - set(row.tolist())
        if missing:
            raise AssertionError(f"protected candidate indices were evicted: {sorted(missing)}")


def check_special_token_protection_policies() -> None:
    torch.manual_seed(29)
    batch_size, num_heads, tokens_per_frame, num_frames, head_dim = 1, 2, 4, 3, 4
    num_tokens = tokens_per_frame * num_frames
    num_anchor_tokens = tokens_per_frame
    special_token_count = 2
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    metadata = _make_multiframe_metadata(batch_size, num_heads, tokens_per_frame, num_frames)
    candidate_mask = metadata.token_indices[:, :, num_anchor_tokens:] >= special_token_count
    protected_local = {0, 1, 4, 5}

    configs = (
        {"policy": "mean", "leverage_granularity": "head"},
        {"policy": "svd_leverage", "leverage_granularity": "head", "leverage_sketch_dim": 0},
        {"policy": "svd_leverage", "leverage_granularity": "layer", "leverage_sketch_dim": 0},
        {
            "policy": "svd_leverage",
            "leverage_granularity": "layer",
            "leverage_sketch_dim": 0,
            "leverage_eviction_selector": "fast_dpp",
        },
        {"policy": "dpp", "leverage_granularity": "layer"},
    )
    for config in configs:
        result = EvictionManager(**config).select(
            k,
            cache_budget=5,
            num_anchor_tokens=num_anchor_tokens,
            v=v,
            candidate_evictable_mask=candidate_mask,
        )
        _assert_candidate_indices_retained(result.kept_candidate_indices, protected_local)
        if result.kept_candidate_indices.shape[-1] <= 1:
            raise AssertionError("special-token protection should allow the cache to exceed its target budget")


def check_special_and_recent_protection_combine() -> None:
    torch.manual_seed(37)
    batch_size, num_heads, tokens_per_frame, num_frames, head_dim = 1, 2, 4, 3, 4
    num_anchor_tokens = tokens_per_frame
    num_tokens = tokens_per_frame * num_frames
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v = torch.randn_like(k)
    metadata = _make_multiframe_metadata(batch_size, num_heads, tokens_per_frame, num_frames)
    result = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_sketch_dim=0,
    ).select(
        k,
        cache_budget=5,
        num_anchor_tokens=num_anchor_tokens,
        v=v,
        current_frame_idx=2,
        protect_recent_frames=1,
        candidate_frame_ids=metadata.frame_ids[:, :, num_anchor_tokens:],
        candidate_evictable_mask=metadata.token_indices[:, :, num_anchor_tokens:] >= 2,
    )
    _assert_candidate_indices_retained(result.kept_candidate_indices, {0, 1, 4, 5, 6, 7})


def check_attention_special_token_protection() -> None:
    torch.manual_seed(41)
    attention = Attention(dim=4, num_heads=1)
    attention.eval()
    frames = [torch.randn(1, 4, 4) for _ in range(3)]
    past = None
    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            _, past, _ = attention(
                frame,
                past_key_values=past,
                use_cache=True,
                cache_budget=3,
                step_idx=frame_idx,
                eviction_policy="svd_leverage",
                leverage_granularity="layer",
                leverage_sketch_dim=0,
                leverage_evictable_only=True,
                eviction_protect_special_tokens=True,
                special_token_count=2,
            )

    k, v, metadata = past
    if k.shape != v.shape or k.shape[2] != metadata.token_indices.shape[2]:
        raise AssertionError("protected cache tensors and metadata lost alignment")
    for frame_id in range(3):
        frame_tokens = metadata.token_indices[0, 0][metadata.frame_ids[0, 0] == frame_id]
        if not {0, 1}.issubset(set(frame_tokens.tolist())):
            raise AssertionError(f"frame {frame_id} special tokens were evicted")
    if k.shape[2] <= 3:
        raise AssertionError("special-token protection should take priority over the hard cache budget")


def check_attention_special_token_interval() -> None:
    torch.manual_seed(43)
    attention = Attention(dim=4, num_heads=1)
    attention.eval()
    frames = [torch.randn(1, 4, 4) for _ in range(4)]
    past = None
    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            _, past, _ = attention(
                frame,
                past_key_values=past,
                use_cache=True,
                cache_budget=6,
                step_idx=frame_idx,
                eviction_policy="svd_leverage",
                leverage_granularity="layer",
                leverage_sketch_dim=0,
                eviction_protect_special_tokens=True,
                eviction_protect_special_token_interval=2,
                special_token_count=2,
            )

    k, v, metadata = past
    if k.shape != v.shape or k.shape[2] != metadata.token_indices.shape[2]:
        raise AssertionError("interval-protected cache tensors and metadata lost alignment")
    if k.shape[2] != 6:
        raise AssertionError(f"expected interval-protected cache size 6, got {k.shape[2]}")
    frame_two_tokens = metadata.token_indices[0, 0][metadata.frame_ids[0, 0] == 2]
    if not {0, 1}.issubset(set(frame_two_tokens.tolist())):
        raise AssertionError("interval-protected frame 2 special tokens were evicted")
    for frame_id in (1, 3):
        frame_tokens = metadata.token_indices[0, 0][metadata.frame_ids[0, 0] == frame_id]
        if {0, 1} & set(frame_tokens.tolist()):
            raise AssertionError(f"non-interval frame {frame_id} special tokens were unexpectedly protected")


def check_invalid_special_token_interval() -> None:
    k, v = _make_cache()
    attention = Attention(dim=k.shape[1] * k.shape[3], num_heads=k.shape[1])
    metadata = KVCacheMetadata.for_current_frame(k.shape[0], k.shape[1], k.shape[2], frame_id=0)
    try:
        attention.eviction(
            k,
            v,
            metadata,
            cache_budget=7,
            num_anchor_tokens=2,
            eviction_protect_special_tokens=True,
            eviction_protect_special_token_interval=0,
            special_token_count=2,
        )
    except ValueError:
        return
    raise AssertionError("invalid special-token protection interval was accepted")


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


def check_cache_evict_gate_forward() -> None:
    torch.manual_seed(31)
    attention = Attention(dim=4, num_heads=1)
    attention.eval()
    first = torch.randn(1, 3, 4)
    second = torch.randn(1, 3, 4)

    with torch.no_grad():
        _, past, _ = attention(
            first,
            use_cache=True,
            cache_budget=4,
            eviction_policy="svd_leverage",
            leverage_granularity="layer",
            leverage_sketch_dim=0,
        )
        _, skipped, skipped_score = attention(
            second,
            past_key_values=past,
            use_cache=True,
            cache_budget=4,
            eviction_policy="svd_leverage",
            leverage_granularity="layer",
            leverage_sketch_dim=0,
            cache_write_current_frame=True,
            cache_evict_current_frame=False,
        )
        _, pruned, pruned_score = attention(
            second,
            past_key_values=past,
            use_cache=True,
            cache_budget=4,
            eviction_policy="svd_leverage",
            leverage_granularity="layer",
            leverage_sketch_dim=0,
            cache_write_current_frame=True,
            cache_evict_current_frame=True,
        )

    if past[0].shape[2] != 3:
        raise AssertionError("initial cache write should keep the first frame tokens")
    if skipped[0].shape[2] != 6:
        raise AssertionError("disabled eviction gate should allow cache to exceed budget")
    if skipped_score is not None:
        raise AssertionError("disabled eviction gate should not report pruning scores")
    if pruned[0].shape[2] != 4:
        raise AssertionError("enabled eviction gate should prune cache to budget")
    if pruned_score is None:
        raise AssertionError("enabled eviction gate should report pruning scores")


def check_ridge_invalid_args() -> None:
    invalid_configs = (
        {"leverage_approx_method": "full_d_ridge", "leverage_ridge_lambda": -1e-3},
        {"leverage_approx_method": "full_d_ridge", "leverage_ridge_jitter": 0.0},
        {"leverage_approx_method": "full_d_ridge", "leverage_ridge_score_chunk_size": 0},
        {"leverage_approx_method": "right_sketch_ridge", "leverage_ridge_dim": 0},
        {"leverage_approx_method": "right_sketch_ridge", "leverage_ridge_dim": None, "leverage_right_jl_dim": 0},
        {"leverage_dpp_quality_beta": -1e-3},
        {"leverage_dpp_diversity_beta": -1e-3},
        {"leverage_dpp_recency_lambda": -1e-3},
        {"leverage_dpp_recency_window": 0},
        {"leverage_dpp_recency_gate_power": -1e-3},
    )
    for cfg in invalid_configs:
        try:
            EvictionManager(policy="svd_leverage", **cfg)
        except ValueError:
            continue
        raise AssertionError(f"invalid ridge config was accepted: {cfg}")


def main() -> None:
    check_head_mode_shape()
    check_layer_score_shape()
    check_shared_layer_indices()
    check_eviction_alignment()
    check_sketch_and_exact_modes()
    check_approx_methods()
    check_drineas_deterministic()
    check_ridge_invalid_args()
    check_fast_dpp_shapes_and_low_score_pool()
    check_layer_head_fast_dpp_matches_reference()
    check_layer_head_fast_dpp_head_specific_and_protection()
    check_layer_head_fast_dpp_recent_protection_and_edge_cases()
    check_layer_head_fast_dpp_invalid_configs_and_merge()
    check_fast_dpp_retain_diversity()
    check_fast_dpp_diversity_beta_zero_prefers_quality()
    check_fast_dpp_quality_beta_zero_uses_only_diversity()
    check_fast_dpp_recency_prior()
    check_fast_dpp_recent_frame_protection()
    check_dpp_shapes_and_full_pool()
    check_dpp_recent_frame_protection()
    check_key_value_feature()
    check_low_precision_inputs()
    check_recent_frame_protection_head_mode()
    check_recent_frame_protection_layer_modes()
    check_recent_frame_eviction_alignment()
    check_protection_disabled_matches_previous_behavior()
    check_evictable_only_matches_subset_reference()
    check_evictable_only_disabled_and_recent_only_are_unchanged()
    check_evictable_only_empty_mask()
    check_special_token_protection_policies()
    check_special_and_recent_protection_combine()
    check_attention_special_token_protection()
    check_attention_special_token_interval()
    check_invalid_special_token_interval()
    check_keep_after_eviction_zero_evict()
    check_small_cache_no_eviction()
    check_cache_evict_gate_forward()
    print("leverage granularity sanity checks passed")


if __name__ == "__main__":
    main()
