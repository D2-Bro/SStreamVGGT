#!/usr/bin/env python3
"""Lightweight checks for SVD leverage eviction granularity."""

from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
import tempfile

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.attention import Attention
from streamvggt.layers.confidence_state import KVConfidenceState, make_token_confidence_gate, parse_confidence_gate_init, sample_token_confidence
from streamvggt.layers.eviction import EvictionManager
from streamvggt.layers.recent_merge import KVCacheMetadata
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.models.aggregator import Aggregator
from streamvggt.utils.cache_analysis import LeverageScoreHistogramConfig


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
        {"leverage_approx_method": "full_d_ridge", "leverage_sketch_dim": 4},
        {
            "leverage_approx_method": "right_sketch_ridge",
            "leverage_sketch_dim": 4,
            "leverage_ridge_dim": 4,
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


def check_key_value_lowdim_concat_feature() -> None:
    k, v = _make_cache()
    ridge_dim = 4
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_feature="key_value_lowdim_concat",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=ridge_dim,
    )
    scores = manager._layer_svd_leverage_scores(k[:, :, 2:, :], v[:, :, 2:, :])
    assert scores.shape == (2, 9)
    assert manager._last_layer_feature_shape == (9, 2 * ridge_dim)
    _assert_finite(scores, "key_value_lowdim_concat layer scores")
    _assert_nonnegative(scores, "key_value_lowdim_concat layer scores")

    try:
        manager._layer_svd_leverage_scores(k[:, :, 2:, :], None)
    except ValueError as exc:
        if "requires value cache tensor" not in str(exc):
            raise AssertionError(f"unexpected missing-value error: {exc}") from exc
    else:
        raise AssertionError("key_value_lowdim_concat should require candidate_v")


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


def check_confidence_state_sidecar() -> None:
    default_state = KVConfidenceState.for_current_frame(1, 2, 3, frame_id=-1, device=torch.device("cpu"))
    if not torch.allclose(default_state.confidence_gate, torch.ones(1, 3)):
        raise AssertionError("default confidence gate should initialize to 1.0")
    init_state = KVConfidenceState.for_current_frame(2, 2, 3, frame_id=-1, device=torch.device("cpu"), initial_gate=torch.tensor([0.4, 0.8]))
    if not torch.allclose(init_state.confidence_gate, torch.tensor([[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])):
        raise AssertionError(f"initial gate [B] did not broadcast: {init_state.confidence_gate}")

    state0 = KVConfidenceState.for_current_frame(1, 2, 4, frame_id=0, device=torch.device("cpu"), initial_gate=0.5)
    state1 = KVConfidenceState.for_current_frame(1, 2, 4, frame_id=1, device=torch.device("cpu"), initial_gate=0.5)
    combined = state0.concat(state1)
    depth_tokens = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
    point_tokens = torch.tensor([[1.0, 0.5, 0.25, 0.125]])
    gate_tokens = make_token_confidence_gate(
        depth_tokens,
        point_tokens,
        floor=0.2,
        depth_alpha=1.0,
        point_beta=1.0,
        preserve_prefix_tokens=1,
    )
    expected_depth = depth_tokens / (depth_tokens + 1.0)
    expected_point = point_tokens / (point_tokens + 1.0)
    expected_gate = 0.2 + 0.8 * expected_depth * expected_point
    expected_gate[:, 0] = expected_gate[:, 1:].mean(dim=1)
    if not torch.allclose(gate_tokens, expected_gate):
        raise AssertionError(f"normalized confidence gate mismatch: {gate_tokens} vs {expected_gate}")
    gate_tokens_k2 = make_token_confidence_gate(
        depth_tokens,
        point_tokens,
        floor=0.2,
        depth_alpha=1.0,
        point_beta=1.0,
        normalizer_k=2.0,
        preserve_prefix_tokens=1,
    )
    expected_depth_k2 = depth_tokens / (depth_tokens + 2.0)
    expected_point_k2 = point_tokens / (point_tokens + 2.0)
    expected_gate_k2 = 0.2 + 0.8 * expected_depth_k2 * expected_point_k2
    expected_gate_k2[:, 0] = expected_gate_k2[:, 1:].mean(dim=1)
    if not torch.allclose(gate_tokens_k2, expected_gate_k2):
        raise AssertionError(f"k=2 confidence gate mismatch: {gate_tokens_k2} vs {expected_gate_k2}")
    try:
        make_token_confidence_gate(depth_tokens, point_tokens, floor=0.2, depth_alpha=1.0, point_beta=1.0, normalizer_k=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("normalizer_k <= 0 should be rejected")
    if not torch.isclose(gate_tokens[0, 0], gate_tokens[0, 1:].mean()):
        raise AssertionError("prefix token gate should use patch gate mean")
    gate_tokens_prefix_one = make_token_confidence_gate(
        depth_tokens,
        point_tokens,
        floor=0.2,
        depth_alpha=1.0,
        point_beta=1.0,
        preserve_prefix_tokens=1,
        prefix_token_mode="one",
    )
    if not torch.isclose(gate_tokens_prefix_one[0, 0], torch.tensor(1.0)):
        raise AssertionError("prefix_token_mode='one' should set prefix gate to 1.0")
    combined.update_frame_gate(1, gate_tokens)
    if not torch.isclose(combined.confidence_gate[0, 6], gate_tokens[0, 2]):
        raise AssertionError("confidence gate update did not use frame/token provenance")
    if not torch.isclose(combined.confidence_gate[0, 7], gate_tokens[0, 3]):
        raise AssertionError("confidence gate update did not broadcast across heads")

    indices = torch.tensor([[[0, 5, 7], [1, 4, 6]]], dtype=torch.long)
    gathered = combined.gather(indices)
    assert gathered.frame_ids.shape == (1, 3)
    if gathered.frame_ids[0].tolist() != [0, 1, 1]:
        raise AssertionError(f"unexpected gathered frame ids: {gathered.frame_ids}")

    dense_conf = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    sampled = sample_token_confidence(
        dense_conf,
        image_hw=(4, 4),
        tokens_per_frame=5,
        patch_start_idx=1,
        patch_size=2,
    )
    expected = torch.tensor([[1.0, 2.5, 4.5, 10.5, 12.5]])
    if not torch.allclose(sampled, expected):
        raise AssertionError(f"unexpected sampled confidence: {sampled} vs {expected}")

    k = torch.randn(1, 1, 6, 2)
    v = torch.randn_like(k)
    attention = Attention(dim=2, num_heads=1)
    conf_state = KVConfidenceState.for_current_frame(1, 1, 6, frame_id=3, device=k.device)
    final_k, final_v, final_metadata, final_conf_state, _ = attention.eviction(
        k,
        v,
        None,
        cache_budget=4,
        num_anchor_tokens=1,
        confidence_state=conf_state,
        eviction_policy="svd_leverage",
        layer_id=0,
        leverage_conf_gate=True,
    )
    assert final_k.shape == final_v.shape == (1, 1, 4, 2)
    if final_metadata is not None:
        raise AssertionError("confidence sidecar path should not create KV metadata")
    if final_conf_state.frame_ids.shape != (1, 4):
        raise AssertionError("confidence sidecar was not gathered with KV cache")

    attention2 = Attention(dim=2, num_heads=1)
    x0 = torch.randn(1, 4, 2)
    x1 = torch.randn(1, 4, 2)
    _, kv0, _ = attention2(
        x0,
        use_cache=True,
        cache_budget=8,
        anchor_token_count=1,
        eviction_policy="svd_leverage",
        layer_id=0,
        step_idx=0,
        leverage_conf_gate=True,
    )
    kv0[3].confidence_gate[:] = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
    kv0[3].gate_sum, kv0[3].gate_count = KVConfidenceState._stats_from_gate(kv0[3].confidence_gate)
    _, kv1, _ = attention2(
        x1,
        past_key_values=kv0,
        use_cache=True,
        cache_budget=8,
        anchor_token_count=1,
        eviction_policy="svd_leverage",
        layer_id=0,
        step_idx=1,
        leverage_conf_gate=True,
    )
    if not torch.allclose(kv1[3].confidence_gate[:, -4:], torch.full((1, 4), 0.5)):
        raise AssertionError(f"default new frame temporary gate should use cached mean: {kv1[3].confidence_gate}")
    for init_value, expected_value in (("1", 1.0), ("1.0", 1.0), ("0.5", 0.5)):
        _, kv_const, _ = attention2(
            x1,
            past_key_values=kv0,
            use_cache=True,
            cache_budget=8,
            anchor_token_count=1,
            eviction_policy="svd_leverage",
            layer_id=0,
            step_idx=1,
            leverage_conf_gate=True,
            leverage_conf_gate_init=init_value,
        )
        expected = torch.full((1, 4), expected_value)
        if not torch.allclose(kv_const[3].confidence_gate[:, -4:], expected):
            raise AssertionError(
                f"new frame temporary gate init {init_value!r} mismatch: {kv_const[3].confidence_gate}"
            )
    for invalid_init in ("bad", "nan", "inf", "-0.1"):
        try:
            parse_confidence_gate_init(invalid_init)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid confidence gate init was accepted: {invalid_init}")
    kv1[3].update_frame_gate(1, torch.tensor([[0.9, 0.8, 0.7, 0.6]]))
    if not torch.allclose(kv1[3].confidence_gate[:, -4:], torch.tensor([[0.9, 0.8, 0.7, 0.6]])):
        raise AssertionError("post-head update did not replace temporary current-frame gate")

    agg = object.__new__(Aggregator)
    agg.depth = 1
    base_k = torch.randn(1, 1, 6, 2)
    base_v = torch.randn_like(base_k)
    base_conf = KVConfidenceState.for_current_frame(1, 1, 6, frame_id=4, device=base_k.device)
    side_k = torch.randn(1, 1, 2, 2)
    side_v = torch.randn_like(side_k)
    side_conf = KVConfidenceState.for_current_frame(1, 1, 2, frame_id=9, device=base_k.device)
    past = [(base_k, base_v, None, base_conf)]
    agg.sync_anchor_special_tokens_from_sidecars(
        past,
        [(side_k, side_v, None, side_conf)],
        anchor_token_count=3,
        tokens_per_frame=6,
        global_anchor_token_count=1,
    )
    synced = past[0]
    if len(synced) != 4 or synced[3].frame_ids.shape[1] != synced[0].shape[2]:
        raise AssertionError("special-token sidecar sync did not preserve confidence state alignment")


def check_confidence_gate_head_mode() -> None:
    B, H, N, D = 1, 1, 4, 2
    k = torch.zeros(B, H, N, D)
    v = torch.zeros_like(k)
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="head",
        leverage_conf_gate=True,
        leverage_conf_gate_floor=0.2,
        leverage_conf_gate_depth_alpha=1.0,
        leverage_conf_gate_point_beta=1.0,
    )
    base_scores = torch.tensor([[[0.20, 0.21, 0.22, 0.23]]], dtype=torch.float32)
    manager._svd_leverage_scores = lambda candidate_k, return_basis=False: base_scores.clone()
    result = manager.select(
        k,
        cache_budget=2,
        num_anchor_tokens=0,
        v=v,
        current_frame_idx=1,
        candidate_frame_ids=torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long),
        candidate_conf_gate=torch.tensor([[[1.0, 0.208, 0.208, 0.208]]], dtype=torch.float32),
    )
    kept = result.kept_candidate_indices[0, 0].tolist()
    if kept != [2, 3]:
        raise AssertionError(f"confidence gate/current-frame exclusion selected {kept}, expected [2, 3]")
    expected = torch.tensor([[[0.20, 0.21 * 0.208, 0.22, 0.23]]], dtype=torch.float32)
    if not torch.allclose(result.policy_scores, expected, atol=1e-6):
        raise AssertionError(f"unexpected gated scores: {result.policy_scores} vs {expected}")


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
            "leverage_ridge_dim": 4,
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



def check_similarity_topk_shapes_and_low_score_pool() -> None:
    k, v = _make_cache()
    multiplier = 2
    methods = (
        {"leverage_approx_method": "right_sketch", "leverage_sketch_dim": 4},
        {"leverage_approx_method": "full_d_ridge", "leverage_sketch_dim": 4},
        {
            "leverage_approx_method": "right_sketch_ridge",
            "leverage_sketch_dim": 4,
            "leverage_ridge_dim": 4,
        },
    )
    for granularity in ("head", "layer"):
        for method_cfg in methods:
            manager = EvictionManager(
                policy="svd_leverage",
                leverage_granularity=granularity,
                leverage_eviction_selector="similarity_topk",
                leverage_dpp_candidate_multiplier=multiplier,
                **method_cfg,
            )
            result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
            assert result.kept_candidate_indices.shape == (2, 3, 5)
            expected_scores = (2, 3, 9) if granularity == "head" else (2, 9)
            assert result.policy_scores.shape == expected_scores
            method = method_cfg["leverage_approx_method"]
            _assert_finite(result.policy_scores, f"{granularity} {method} similarity_topk scores")
            _assert_nonnegative(result.policy_scores, f"{granularity} {method} similarity_topk scores")
            _assert_evicted_subset_of_low_score_pool(result, candidate_count=9, multiplier=multiplier)
            if granularity == "layer":
                for head_idx in range(1, result.kept_candidate_indices.shape[1]):
                    if not torch.equal(result.kept_candidate_indices[:, 0], result.kept_candidate_indices[:, head_idx]):
                        raise AssertionError("layer-wise similarity_topk did not share kept indices across heads")


def check_similarity_topk_evicts_redundant_pool_key() -> None:
    scores = torch.tensor([[[0.0, 0.1, 1.0, 1.1, 1.2, 1.3]]])
    mask = torch.ones_like(scores, dtype=torch.bool)
    candidate_k = torch.tensor(
        [[[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, -1.0],
        ]]],
        dtype=torch.float32,
    )
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="head",
        leverage_eviction_selector="similarity_topk",
        leverage_dpp_candidate_multiplier=2,
    )
    kept = manager._keep_after_head_similarity_topk(scores, mask, 1, candidate_k)
    evicted = set(range(candidate_k.shape[2])) - set(kept.reshape(-1).tolist())
    if evicted != {0}:
        raise AssertionError(f"similarity_topk did not evict the duplicated low-score key first: {evicted}")




def check_similarity_topk_uses_similarity_over_leverage_ratio() -> None:
    scores = torch.tensor([[[0.2, 0.9, 1.0, 1.1]]], dtype=torch.float32)
    mask = torch.ones_like(scores, dtype=torch.bool)
    candidate_k = torch.tensor(
        [[[
            [0.8, 0.6],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]]],
        dtype=torch.float32,
    )
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="head",
        leverage_eviction_selector="similarity_topk",
        leverage_dpp_candidate_multiplier=2,
    )
    kept = manager._keep_after_head_similarity_topk(scores, mask, 1, candidate_k)
    evicted = set(range(candidate_k.shape[2])) - set(kept.reshape(-1).tolist())
    if evicted != {0}:
        raise AssertionError(f"similarity_topk should evict by max_cosine/leverage ratio, got {evicted}")



def check_similarity_topk_leverage_gamma_controls_denominator() -> None:
    scores = torch.tensor([[[0.2, 0.9, 1.0, 1.1]]], dtype=torch.float32)
    mask = torch.ones_like(scores, dtype=torch.bool)
    candidate_k = torch.tensor(
        [[[
            [0.8, 0.6],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]]],
        dtype=torch.float32,
    )
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="head",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_leverage_gamma=0.0,
        leverage_dpp_candidate_multiplier=2,
    )
    kept = manager._keep_after_head_similarity_topk(scores, mask, 1, candidate_k)
    evicted = set(range(candidate_k.shape[2])) - set(kept.reshape(-1).tolist())
    if evicted != {1}:
        raise AssertionError(f"gamma=0 should reduce similarity_topk to max-cosine ranking, got {evicted}")


def check_similarity_topk_head_granularity_head_specific_eviction() -> None:
    scores = torch.tensor([[0.0, 0.1, 1.0, 1.1]], dtype=torch.float32)
    mask = torch.ones_like(scores, dtype=torch.bool)
    candidate_k = torch.tensor(
        [[
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="head",
        leverage_dpp_candidate_multiplier=2,
    )
    kept = manager._keep_after_layer_head_similarity_topk(scores, mask, 1, candidate_k)
    evicted_h0 = set(range(candidate_k.shape[2])) - set(kept[0, 0].tolist())
    evicted_h1 = set(range(candidate_k.shape[2])) - set(kept[0, 1].tolist())
    if evicted_h0 != {0} or evicted_h1 != {1}:
        raise AssertionError(f"head-wise similarity_topk evicted {evicted_h0=} {evicted_h1=}")
    if torch.equal(kept[:, 0], kept[:, 1]):
        raise AssertionError("head-wise similarity_topk should allow head-specific keep sets")


def check_similarity_topk_head_granularity_select_and_merge_guard() -> None:
    k, v = _make_cache()
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="head",
        leverage_dpp_candidate_multiplier=2,
        leverage_sketch_dim=4,
    )
    result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    assert result.kept_candidate_indices.shape == (2, 3, 5)
    _assert_evicted_subset_of_low_score_pool(result, candidate_count=9, multiplier=2)

    attention = Attention(dim=5, num_heads=1)
    try:
        attention.eviction(
            k[:, :1],
            v[:, :1],
            None,
            cache_budget=7,
            num_anchor_tokens=2,
            eviction_policy="svd_leverage",
            leverage_granularity="layer",
            leverage_eviction_selector="similarity_topk",
            leverage_similarity_granularity="head",
            svd_eviction_merge_config=SvdEvictionMergeConfig(enabled=True, mode="layer"),
        )
    except ValueError as exc:
        if "Head-specific eviction keep sets" not in str(exc):
            raise AssertionError(f"unexpected merge guard error: {exc}") from exc
    else:
        raise AssertionError("head-wise similarity_topk should reject non-head SVD merge mode")


def check_similarity_topk_protection_and_recent_frames() -> None:
    k, v = _make_cache()
    candidate_count = k.shape[2] - 2
    mask = torch.ones(k.shape[0], candidate_count, dtype=torch.bool)
    mask[0, :3] = False
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_sketch_dim=4,
        leverage_eviction_selector="similarity_topk",
        leverage_dpp_candidate_multiplier=2,
    )
    result = manager.select(k, cache_budget=7, num_anchor_tokens=2, v=v, candidate_evictable_mask=mask)
    protected = set(torch.nonzero(~mask[0], as_tuple=False).flatten().tolist())
    missing = protected - set(result.kept_candidate_indices[0, 0].tolist())
    if missing:
        raise AssertionError(f"similarity_topk evicted protected layer tokens: {sorted(missing)}")

    limited_mask = torch.zeros_like(mask)
    limited_mask[..., -1] = True
    limited = manager.select(k, cache_budget=5, num_anchor_tokens=2, v=v, candidate_evictable_mask=limited_mask)
    if limited.kept_candidate_indices.shape[-1] != candidate_count - 1:
        raise AssertionError("similarity_topk did not limit eviction to evictable capacity")

    torch.manual_seed(59)
    batch_size, num_heads, num_tokens, head_dim = 1, 3, 10, 4
    k_recent = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    v_recent = torch.randn_like(k_recent)
    frame_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    metadata = _frame_metadata(batch_size, num_heads, frame_ids)
    recent = manager.select(
        k_recent,
        cache_budget=4,
        num_anchor_tokens=2,
        v=v_recent,
        current_frame_idx=3,
        protect_recent_frames=2,
        candidate_frame_ids=metadata.frame_ids[:, :, 2:],
    )
    _assert_recent_frames_not_evicted(recent.kept_candidate_indices, frame_ids, {2, 3})


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
                if manager.leverage_feature in ("key_value", "key_value_lowdim_concat"):
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


def check_fast_dpp_random_projection_features() -> None:
    scores = torch.linspace(0.0, 1.0, 8, dtype=torch.float32)
    mask = torch.ones_like(scores, dtype=torch.bool)
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.1],
            [0.0, 1.0, 0.0, 0.0, 0.4, 0.2],
            [0.0, 0.0, 1.0, 0.0, 0.3, 0.3],
            [0.0, 0.0, 0.0, 1.0, 0.2, 0.4],
            [1.0, 1.0, 0.0, 0.0, 0.1, 0.5],
            [0.0, 1.0, 1.0, 0.0, 0.6, 0.1],
            [0.0, 0.0, 1.0, 1.0, 0.2, 0.6],
            [1.0, 0.0, 0.0, 1.0, 0.7, 0.2],
        ],
        dtype=torch.float32,
    )

    default_manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
    )
    raw_manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_feature_projection="raw",
        leverage_ridge_dim=2,
    )
    default_evicted = default_manager._fast_dpp_evicted_indices(
        scores, mask, 4, lambda idx: features.index_select(0, idx)
    )
    raw_evicted = raw_manager._fast_dpp_evicted_indices(
        scores, mask, 4, lambda idx: features.index_select(0, idx)
    )
    if not torch.equal(default_evicted, raw_evicted):
        raise AssertionError("explicit raw Fast DPP projection changed default behavior")

    random_manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_feature_projection="random",
        leverage_ridge_dim=3,
        leverage_random_seed=7,
    )
    random_evicted = random_manager._fast_dpp_evicted_indices(
        scores, mask, 4, lambda idx: features.index_select(0, idx)
    )
    repeat_manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=1,
        leverage_dpp_feature_projection="random",
        leverage_ridge_dim=3,
        leverage_random_seed=7,
    )
    repeat_evicted = repeat_manager._fast_dpp_evicted_indices(
        scores, mask, 4, lambda idx: features.index_select(0, idx)
    )
    if not torch.equal(random_evicted, repeat_evicted):
        raise AssertionError("same-seed random projected Fast DPP was not deterministic")
    if random_manager._last_layer_feature_shape != (8, 3):
        raise AssertionError(
            "random projected Fast DPP did not use min(leverage_ridge_dim, feature_dim), "
            f"got {random_manager._last_layer_feature_shape}"
        )

    found_seed_sensitive_case = False
    search_scores = torch.linspace(0.0, 1.0, 10, dtype=torch.float32)
    search_mask = torch.ones_like(search_scores, dtype=torch.bool)
    for trial_seed in range(128):
        generator = torch.Generator().manual_seed(trial_seed)
        search_features = torch.randn(10, 8, generator=generator)
        seed_outputs = []
        for projection_seed in (0, 1):
            manager = EvictionManager(
                policy="svd_leverage",
                leverage_eviction_selector="fast_dpp",
                leverage_dpp_candidate_multiplier=2,
                leverage_dpp_greedy_block_size=1,
                leverage_dpp_quality_beta=0.0,
                leverage_dpp_feature_projection="random",
                leverage_ridge_dim=2,
                leverage_random_seed=projection_seed,
            )
            seed_outputs.append(
                manager._fast_dpp_evicted_indices(
                    search_scores,
                    search_mask,
                    5,
                    lambda idx, search_features=search_features: search_features.index_select(0, idx),
                )
            )
        if not torch.equal(seed_outputs[0], seed_outputs[1]):
            found_seed_sensitive_case = True
            break
    if not found_seed_sensitive_case:
        raise AssertionError("different projection seeds never changed Fast DPP choices in the fixture search")


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

    quality_zero_recency = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_diversity_beta=0.0,
        leverage_dpp_quality_beta=0.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=0.2,
        leverage_dpp_recency_window=5,
    )
    evicted = set(
        quality_zero_recency._fast_dpp_evicted_indices(
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
        raise AssertionError(
            "quality beta 0 should keep the recency prior active, "
            f"got {sorted(retained)}"
        )

    gate_free_scores = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    gate_free_frame_ids = torch.tensor([0, 0, 5, 0], dtype=torch.long)
    gate_free_recency = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_diversity_beta=0.0,
        leverage_dpp_quality_beta=0.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=0.2,
        leverage_dpp_recency_window=5,
        leverage_dpp_recency_gate_power=0.0,
    )
    evicted = set(
        gate_free_recency._fast_dpp_evicted_indices(
            gate_free_scores,
            mask,
            3,
            lambda idx: features.index_select(0, idx),
            row_frame_ids=gate_free_frame_ids,
            current_frame_id=5,
        ).tolist()
    )
    retained = set(range(gate_free_scores.numel())) - evicted
    if retained != {2}:
        raise AssertionError(
            "recency gate power 0 should ignore leverage score gating, "
            f"got {sorted(retained)}"
        )

    pure_similarity = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_quality_beta=0.0,
    )
    lambda_zero_recency = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_quality_beta=0.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=0.0,
    )
    pure_similarity_evicted = set(
        pure_similarity._fast_dpp_evicted_indices(
            scores,
            mask,
            2,
            lambda idx: features.index_select(0, idx),
        ).tolist()
    )
    lambda_zero_evicted = set(
        lambda_zero_recency._fast_dpp_evicted_indices(
            scores,
            mask,
            2,
            lambda idx: features.index_select(0, idx),
            row_frame_ids=frame_ids,
            current_frame_id=5,
        ).tolist()
    )
    if lambda_zero_evicted != pure_similarity_evicted:
        raise AssertionError("recency lambda 0 should match pure-similarity DPP when quality beta is 0")

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


def check_layer_range_fifo_override() -> None:
    k = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
    v = k + 100.0
    metadata = KVCacheMetadata.for_current_frame(
        batch_size=1,
        num_heads=1,
        num_tokens=8,
        frame_id=0,
    )
    attention = Attention(dim=1, num_heads=1)

    fifo_k, fifo_v, fifo_metadata, fifo_score = attention.eviction(
        k,
        v,
        metadata,
        cache_budget=5,
        num_anchor_tokens=2,
        layer_id=3,
        step_idx=1,
        eviction_policy="svd_leverage",
        eviction_policy_layers={4},
        leverage_granularity="layer",
        leverage_sketch_dim=0,
    )
    expected = torch.tensor([0.0, 1.0, 5.0, 6.0, 7.0]).view(1, 1, 5, 1)
    if not torch.equal(fifo_k, expected):
        raise AssertionError(f"FIFO override kept wrong k indices: {fifo_k.flatten().tolist()}")
    if not torch.equal(fifo_v, expected + 100.0):
        raise AssertionError("FIFO override kept wrong v indices")
    if fifo_metadata is None or not torch.equal(fifo_metadata.token_indices, expected.long().squeeze(-1)):
        raise AssertionError("FIFO override kept wrong metadata indices")
    if fifo_score != 0.0:
        raise AssertionError("FIFO override should not emit policy scores")

    _, _, _, selected_score = attention.eviction(
        k,
        v,
        metadata,
        cache_budget=5,
        num_anchor_tokens=2,
        layer_id=3,
        step_idx=1,
        eviction_policy="svd_leverage",
        eviction_policy_layers={3},
        leverage_granularity="layer",
        leverage_sketch_dim=0,
        layer_budget_strategy="leverage_pr",
    )
    if not isinstance(selected_score, tuple):
        raise AssertionError("selected layer should use svd_leverage and emit layer-budget payload")

    _, _, _, default_score = attention.eviction(
        k,
        v,
        metadata,
        cache_budget=5,
        num_anchor_tokens=2,
        layer_id=3,
        step_idx=1,
        eviction_policy="svd_leverage",
        eviction_policy_layers=None,
        leverage_granularity="layer",
        leverage_sketch_dim=0,
        layer_budget_strategy="leverage_pr",
    )
    if not isinstance(default_score, tuple):
        raise AssertionError("empty layer override should preserve existing svd_leverage behavior")


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


def check_attention_anchor_only_special_cache_write() -> None:
    torch.manual_seed(41)
    baseline = Attention(dim=4, num_heads=1)
    special_only = Attention(dim=4, num_heads=1)
    special_only.load_state_dict(baseline.state_dict())
    baseline.eval()
    special_only.eval()
    x = torch.randn(1, 5, 4)

    with torch.no_grad():
        baseline_out, baseline_kv, _ = baseline(
            x,
            use_cache=True,
            cache_budget=16,
            step_idx=1,
            special_token_count=2,
        )
        special_result = special_only(
            x,
            use_cache=True,
            cache_budget=16,
            step_idx=1,
            special_token_count=2,
            global_cache_history_anchor_special_tokens_only=True,
        )

    if len(special_result) != 4:
        raise AssertionError("anchor-only special cache mode should return a sidecar")
    special_out, special_kv, _scores, sidecar = special_result
    if not torch.allclose(baseline_out, special_out, atol=1e-5, rtol=1e-5):
        raise AssertionError("read-only current special tokens should preserve current-frame attention output")
    if baseline_kv[0].shape[2] != 5:
        raise AssertionError("baseline cache should keep all current tokens")
    if special_kv[0].shape[2] != 3:
        raise AssertionError(f"special-only cache should write only patch tokens, got {special_kv[0].shape[2]}")
    if sidecar[0].shape[2] != 2 or sidecar[1].shape[2] != 2:
        raise AssertionError("sidecar should contain the skipped special tokens")
    metadata = special_kv[2]
    if not torch.equal(metadata.token_indices.view(-1), torch.tensor([2, 3, 4], dtype=torch.int32)):
        raise AssertionError(f"unexpected persistent token indices: {metadata.token_indices.view(-1).tolist()}")



def check_attention_first_frame_special_only_anchor_prefix() -> None:
    torch.manual_seed(43)
    attention = Attention(dim=4, num_heads=1)
    k = torch.randn(1, 1, 20, 4)
    v = torch.randn(1, 1, 20, 4)
    first_meta = KVCacheMetadata.for_current_frame(1, 1, 10, frame_id=0)
    second_meta = KVCacheMetadata.for_current_frame(1, 1, 10, frame_id=1)
    metadata = first_meta.concat(second_meta)

    with torch.no_grad():
        _k, _v, pruned_metadata, _scores = attention.eviction(
            k,
            v,
            metadata,
            cache_budget=10,
            num_anchor_tokens=5,
            eviction_policy="mean",
        )

    token_indices = pruned_metadata.token_indices.view(-1)
    frame_ids = pruned_metadata.frame_ids.view(-1)
    if not torch.equal(token_indices[:5], torch.arange(5, dtype=torch.int32)):
        raise AssertionError(f"unexpected protected token prefix: {token_indices[:5].tolist()}")
    if not torch.equal(frame_ids[:5], torch.zeros(5, dtype=torch.long)):
        raise AssertionError(f"unexpected protected frame prefix: {frame_ids[:5].tolist()}")
    first_frame_patch_positions = torch.nonzero((frame_ids == 0) & (token_indices >= 5), as_tuple=False).flatten()
    if first_frame_patch_positions.numel() > 0 and int(first_frame_patch_positions.min().item()) < 5:
        raise AssertionError("first-frame patch tokens entered the protected prefix")


def check_outlier_then_low_risk_mode() -> None:
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_eviction_risk_mode="outlier_then_low",
        leverage_high_outlier_z=1.0,
    )
    scores = torch.tensor([[0.2, 0.25, 0.3, 10.0]], dtype=torch.float32)
    evictable = torch.ones_like(scores, dtype=torch.bool)
    kept = manager._keep_after_outlier_then_low_topk(scores, evictable, num_to_evict=1)
    evicted = set(range(scores.shape[-1])) - set(kept[0].tolist())
    if evicted != {3}:
        raise AssertionError(f"high outlier was not evicted first: {sorted(evicted)}")

    scores = torch.tensor([[0.0, 0.1, 0.2, 8.0, 9.0]], dtype=torch.float32)
    kept = manager._keep_after_outlier_then_low_topk(scores, torch.ones_like(scores, dtype=torch.bool), num_to_evict=3)
    evicted = set(range(scores.shape[-1])) - set(kept[0].tolist())
    if evicted != {0, 3, 4}:
        raise AssertionError(f"expected high outliers then lowest score, got {sorted(evicted)}")

    scores = torch.tensor([[0.0, 0.1, 10.0]], dtype=torch.float32)
    evictable = torch.tensor([[True, True, False]])
    kept = manager._keep_after_outlier_then_low_topk(scores, evictable, num_to_evict=1)
    evicted = set(range(scores.shape[-1])) - set(kept[0].tolist())
    if evicted != {0}:
        raise AssertionError(f"protected high outlier should not be evicted: {sorted(evicted)}")

    k = torch.randn(1, 2, 4, 3)
    v = torch.randn_like(k)
    kept = manager._keep_after_layer_fast_dpp(
        torch.tensor([[0.2, 0.25, 0.3, 10.0]], dtype=torch.float32),
        torch.ones(1, 4, dtype=torch.bool),
        num_to_evict=1,
        candidate_k=k,
        candidate_v=v,
        outlier_then_low=True,
    )
    evicted = set(range(4)) - set(kept[0].tolist())
    if evicted != {3}:
        raise AssertionError(f"Fast DPP path did not direct-evict high outlier: {sorted(evicted)}")



def check_leverage_score_histogram_config() -> None:
    with tempfile.TemporaryDirectory(prefix="lev_hist_sanity_") as tmpdir:
        config = LeverageScoreHistogramConfig(output_dir=tmpdir, bins=4, min_value=0.0, max_value=1.0)
        config.record(
            torch.tensor([[-0.1, 0.1, 0.2, 0.8, 1.0, 1.2, float("nan"), float("inf")]]),
            layer_id=2,
            step_idx=0,
        )
        config.record(
            torch.tensor([[[0.0, 0.49, 0.51], [0.99, -1.0, 2.0]]]),
            layer_id=3,
            step_idx=1,
        )
        filtered = LeverageScoreHistogramConfig(
            output_dir=os.path.join(tmpdir, "filtered"),
            bins=2,
            min_value=0.0,
            max_value=1.0,
            layers={4},
        )
        filtered.record(torch.tensor([[0.2, 0.4]]), layer_id=5, step_idx=0)
        filtered.record(torch.tensor([[0.2, 0.4]]), layer_id=4, step_idx=0)
        config.flush()
        filtered.flush()

        hist_path = os.path.join(tmpdir, "leverage_histograms.csv")
        summary_path = os.path.join(tmpdir, "leverage_histogram_summary.csv")
        with open(hist_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        counts = {}
        for row in rows:
            counts.setdefault(int(row["layer_id"]), []).append(int(row["count"]))
        if counts[2] != [2, 0, 0, 2]:
            raise AssertionError(f"unexpected layer 2 histogram counts: {counts[2]}")
        if counts[3] != [1, 1, 1, 1]:
            raise AssertionError(f"unexpected layer 3 histogram counts: {counts[3]}")

        with open(summary_path, newline="", encoding="utf-8") as f:
            summary = {int(row["layer_id"]): row for row in csv.DictReader(f)}
        if int(summary[2]["total_tokens"]) != 6 or int(summary[2]["underflow"]) != 1 or int(summary[2]["overflow"]) != 1:
            raise AssertionError(f"unexpected layer 2 summary: {summary[2]}")
        if int(summary[3]["total_tokens"]) != 6 or int(summary[3]["underflow"]) != 1 or int(summary[3]["overflow"]) != 1:
            raise AssertionError(f"unexpected layer 3 summary: {summary[3]}")

        filtered_summary_path = os.path.join(tmpdir, "filtered", "leverage_histogram_summary.csv")
        with open(filtered_summary_path, newline="", encoding="utf-8") as f:
            filtered_summary = {int(row["layer_id"]): row for row in csv.DictReader(f)}
        if set(filtered_summary) != {4}:
            raise AssertionError(f"layer filter did not skip non-selected layer: {filtered_summary}")



def check_similarity_topk_reused_projection_features() -> None:
    k, v = _make_cache()
    candidate_count = k.shape[2] - 2
    ridge_dim = 4

    default_raw = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=ridge_dim,
        leverage_random_seed=123,
    )
    explicit_raw = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="layer",
        leverage_similarity_feature_projection="raw",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=ridge_dim,
        leverage_random_seed=123,
    )
    default_result = default_raw.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    explicit_result = explicit_raw.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    if not torch.equal(default_result.kept_candidate_indices, explicit_result.kept_candidate_indices):
        raise AssertionError("explicit raw similarity projection changed default similarity_topk behavior")

    layer_projected = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="layer",
        leverage_similarity_feature_projection="random",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=ridge_dim,
        leverage_random_seed=123,
    )
    layer_result = layer_projected.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    assert layer_result.kept_candidate_indices.shape == (2, 3, 5)
    if layer_projected._last_similarity_layer_features is None:
        raise AssertionError("layer projected similarity features were not stored")
    if layer_projected._last_similarity_layer_features.shape != (2, candidate_count, ridge_dim):
        raise AssertionError(
            f"unexpected layer projected feature shape: {tuple(layer_projected._last_similarity_layer_features.shape)}"
        )
    _assert_evicted_subset_of_low_score_pool(layer_result, candidate_count=candidate_count, multiplier=2)

    head_projected = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="head",
        leverage_similarity_feature_projection="random",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=ridge_dim,
        leverage_random_seed=123,
    )
    head_result = head_projected.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    assert head_result.kept_candidate_indices.shape == (2, 3, 5)
    if head_projected._last_similarity_head_features is None:
        raise AssertionError("head projected similarity features were not stored")
    if head_projected._last_similarity_head_features.shape != (2, 3, candidate_count, ridge_dim):
        raise AssertionError(
            f"unexpected head projected feature shape: {tuple(head_projected._last_similarity_head_features.shape)}"
        )
    _assert_evicted_subset_of_low_score_pool(head_result, candidate_count=candidate_count, multiplier=2)

    missing_head_projection = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_projection="head_mean",
        leverage_head_mean_dim=1,
        leverage_eviction_selector="similarity_topk",
        leverage_similarity_granularity="head",
        leverage_similarity_feature_projection="random",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=ridge_dim,
    )
    try:
        missing_head_projection.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
    except ValueError as exc:
        if "head-wise leverage features" not in str(exc):
            raise AssertionError(f"unexpected missing projection error: {exc}") from exc
    else:
        raise AssertionError("head-wise random similarity should reject missing projected head features")


def check_leverage_diag_output_and_invariance() -> None:
    k, v = _make_cache()
    base = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=4,
        leverage_random_seed=99,
    )
    diag = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=4,
        leverage_random_seed=99,
        leverage_diag=True,
        leverage_diag_interval=0,
    )
    base_result = base.select(k, cache_budget=7, num_anchor_tokens=2, v=v, layer_id=3, step_idx=0)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        diag_result = diag.select(k, cache_budget=7, num_anchor_tokens=2, v=v, layer_id=3, step_idx=0)
    output = buffer.getvalue()
    for expected in ("[LeverageDiag]", "effective_dim", "lambda/mean_eig", "deff sweep"):
        if expected not in output:
            raise AssertionError(f"missing leverage diagnostic output: {expected}")
    if not torch.equal(base_result.kept_candidate_indices, diag_result.kept_candidate_indices):
        raise AssertionError("leverage diagnostics changed kept candidate indices")
    if not torch.allclose(base_result.policy_scores, diag_result.policy_scores):
        raise AssertionError("leverage diagnostics changed policy scores")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        diag.select(k, cache_budget=7, num_anchor_tokens=2, v=v, layer_id=3, step_idx=1)
    if "[LeverageDiag]" in buffer.getvalue():
        raise AssertionError("leverage_diag_interval=0 should only print once per manager")

def _layer_token_normalize_keys(k: torch.Tensor) -> torch.Tensor:
    norm = k.float().square().sum(dim=(1, 3), keepdim=True).sqrt().clamp_min(1e-12)
    return k.float() / norm


def _head_token_normalize_keys(k: torch.Tensor) -> torch.Tensor:
    norm = k.float().square().sum(dim=3, keepdim=True).sqrt().clamp_min(1e-12)
    return k.float() / norm


def check_layer_key_normalize_before_projection() -> None:
    k, v = _make_cache()
    candidate_k = k[:, :, 2:, :]
    manual_k = _layer_token_normalize_keys(candidate_k)
    manual_head_k = _head_token_normalize_keys(candidate_k)
    base_kwargs = {
        "policy": "svd_leverage",
        "leverage_granularity": "layer",
        "leverage_feature": "key",
        "leverage_projection": "random",
        "leverage_random_seed": 123,
    }
    for approx_kwargs in (
        {"leverage_approx_method": "exact_qr", "leverage_sketch_dim": 0},
        {"leverage_approx_method": "right_sketch", "leverage_sketch_dim": 4},
        {"leverage_approx_method": "right_sketch_ridge", "leverage_ridge_dim": 4},
    ):
        normalized = EvictionManager(
            **base_kwargs,
            **approx_kwargs,
            leverage_normalize_before_projection=True,
        )
        manual = EvictionManager(**base_kwargs, **approx_kwargs)
        normalized_scores = normalized._layer_svd_leverage_scores(candidate_k)
        manual_scores = manual._layer_svd_leverage_scores(manual_k)
        if not torch.allclose(normalized_scores, manual_scores, atol=1e-5, rtol=1e-4):
            raise AssertionError(
                "normalize_before_projection did not match manual layer-token key normalization "
                f"for {approx_kwargs['leverage_approx_method']}"
            )

        headwise = EvictionManager(
            **base_kwargs,
            **approx_kwargs,
            leverage_normalize_before_projection=True,
            leverage_normalize_before_projection_headwise=True,
        )
        headwise_scores = headwise._layer_svd_leverage_scores(candidate_k)
        manual_head_scores = manual._layer_svd_leverage_scores(manual_head_k)
        if not torch.allclose(headwise_scores, manual_head_scores, atol=1e-5, rtol=1e-4):
            raise AssertionError(
                "headwise normalize_before_projection did not match manual per-head key normalization "
                f"for {approx_kwargs['leverage_approx_method']}"
            )

    for cache_headwise in (False, True):
        no_cache = EvictionManager(
            **base_kwargs,
            leverage_approx_method="right_sketch_ridge",
            leverage_ridge_dim=4,
            leverage_normalize_before_projection=True,
            leverage_normalize_before_projection_headwise=cache_headwise,
        )
        cached = EvictionManager(
            **base_kwargs,
            leverage_approx_method="right_sketch_ridge",
            leverage_ridge_dim=4,
            leverage_normalize_before_projection=True,
            leverage_normalize_before_projection_headwise=cache_headwise,
            leverage_projected_key_cache=True,
        )
        no_cache_result = no_cache.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
        cached_result = cached.select(k, cache_budget=7, num_anchor_tokens=2, v=v)
        if not torch.allclose(no_cache_result.policy_scores, cached_result.policy_scores, atol=1e-5, rtol=1e-4):
            raise AssertionError(
                "projected key cache changed normalize_before_projection policy scores "
                f"for headwise={cache_headwise}"
            )
        if not torch.equal(no_cache_result.kept_candidate_indices, cached_result.kept_candidate_indices):
            raise AssertionError(
                "projected key cache changed normalize_before_projection kept indices "
                f"for headwise={cache_headwise}"
            )

    invalid_headwise_configs = (
        {},
    )
    for cfg in invalid_headwise_configs:
        kwargs = dict(base_kwargs)
        kwargs.update(cfg)
        try:
            EvictionManager(**kwargs, leverage_normalize_before_projection_headwise=True)
        except ValueError as exc:
            if "leverage_normalize_before_projection_headwise" not in str(exc):
                raise AssertionError(f"unexpected headwise normalize-before-projection error: {exc}") from exc
        else:
            raise AssertionError(f"invalid headwise normalize-before-projection config was accepted: {cfg}")

    invalid_configs = (
        {"leverage_granularity": "head"},
        {"leverage_feature": "key_value"},
        {"leverage_projection": "head_mean"},
        {"leverage_approx_method": "full_d_ridge"},
    )
    for cfg in invalid_configs:
        kwargs = dict(base_kwargs)
        kwargs.update(cfg)
        try:
            EvictionManager(
                **kwargs,
                leverage_normalize_before_projection=True,
                leverage_normalize_before_projection_headwise=True,
            )
        except ValueError as exc:
            if "leverage_normalize_before_projection" not in str(exc):
                raise AssertionError(f"unexpected normalize-before-projection error: {exc}") from exc
        else:
            raise AssertionError(f"invalid normalize-before-projection config was accepted: {cfg}")


def check_ridge_invalid_args() -> None:
    invalid_configs = (
        {"leverage_approx_method": "full_d_ridge", "leverage_ridge_lambda": -1e-3},
        {"leverage_approx_method": "full_d_ridge", "leverage_ridge_jitter": 0.0},
        {"leverage_approx_method": "full_d_ridge", "leverage_ridge_score_chunk_size": 0},
        {"leverage_approx_method": "right_sketch_ridge", "leverage_ridge_dim": 0},
        {"leverage_approx_method": "right_sketch_ridge", "leverage_ridge_dim": None},
        {"leverage_dpp_quality_beta": -1e-3},
        {"leverage_dpp_diversity_beta": -1e-3},
        {"leverage_dpp_feature_projection": "bad"},
        {"leverage_dpp_feature_projection": "random"},
        {"leverage_dpp_feature_projection": "random", "leverage_ridge_dim": 0},
        {"leverage_similarity_granularity": "bad"},
        {"leverage_similarity_feature_projection": "bad"},
        {"leverage_similarity_leverage_gamma": -1e-3},
        {"leverage_similarity_feature_projection": "random", "leverage_approx_method": "full_d_ridge"},
        {"leverage_dpp_recency_lambda": -1e-3},
        {"leverage_dpp_recency_window": 0},
        {"leverage_dpp_recency_gate_power": -1e-3},
        {"leverage_eviction_risk_mode": "unknown"},
        {"leverage_high_outlier_z": -1e-3},
        {"leverage_diag_interval": -1},
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
    check_leverage_diag_output_and_invariance()
    check_layer_key_normalize_before_projection()
    check_ridge_invalid_args()
    check_leverage_score_histogram_config()
    check_outlier_then_low_risk_mode()
    check_fast_dpp_shapes_and_low_score_pool()
    check_similarity_topk_shapes_and_low_score_pool()
    check_similarity_topk_evicts_redundant_pool_key()
    check_similarity_topk_uses_similarity_over_leverage_ratio()
    check_similarity_topk_leverage_gamma_controls_denominator()
    check_similarity_topk_head_granularity_head_specific_eviction()
    check_similarity_topk_head_granularity_select_and_merge_guard()
    check_similarity_topk_protection_and_recent_frames()
    check_similarity_topk_reused_projection_features()
    check_layer_head_fast_dpp_matches_reference()
    check_layer_head_fast_dpp_head_specific_and_protection()
    check_layer_head_fast_dpp_recent_protection_and_edge_cases()
    check_layer_head_fast_dpp_invalid_configs_and_merge()
    check_fast_dpp_retain_diversity()
    check_fast_dpp_diversity_beta_zero_prefers_quality()
    check_fast_dpp_quality_beta_zero_uses_only_diversity()
    check_fast_dpp_random_projection_features()
    check_fast_dpp_recency_prior()
    check_fast_dpp_recent_frame_protection()
    check_dpp_shapes_and_full_pool()
    check_dpp_recent_frame_protection()
    check_key_value_feature()
    check_key_value_lowdim_concat_feature()
    check_low_precision_inputs()
    check_confidence_state_sidecar()
    check_confidence_gate_head_mode()
    check_recent_frame_protection_head_mode()
    check_recent_frame_protection_layer_modes()
    check_recent_frame_eviction_alignment()
    check_protection_disabled_matches_previous_behavior()
    check_special_token_protection_policies()
    check_special_and_recent_protection_combine()
    check_attention_special_token_protection()
    check_attention_special_token_interval()
    check_invalid_special_token_interval()
    check_keep_after_eviction_zero_evict()
    check_small_cache_no_eviction()
    check_layer_range_fifo_override()
    check_cache_evict_gate_forward()
    check_attention_anchor_only_special_cache_write()
    check_attention_first_frame_special_only_anchor_prefix()
    print("leverage granularity sanity checks passed")


if __name__ == "__main__":
    main()
