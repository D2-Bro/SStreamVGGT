#!/usr/bin/env python3
"""Unit and optional CUDA checks for frozen early-attention utility."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamvggt.layers.attention import Attention, scale_stac_colsum
from streamvggt.layers.confidence_state import unpack_kv_cache
from streamvggt.layers.confidence_state import KVConfidenceState
from streamvggt.layers.eviction import EvictionManager


def _state(tokens: int, frame_id: int = 0) -> KVConfidenceState:
    return KVConfidenceState.for_current_frame(
        batch_size=1,
        num_heads=2,
        num_tokens=tokens,
        frame_id=frame_id,
        device=torch.device("cpu"),
        initialize_attention=True,
    )


def test_five_updates_then_freeze() -> None:
    state = _state(2)
    values = [torch.tensor([[float(i), float(2 * i)]]) for i in range(1, 8)]
    for value in values:
        state.update_attention_utility(value, ema_decay=0.9, freeze_updates=5)

    weights = torch.tensor([0.9**4, 0.9**3, 0.9**2, 0.9, 1.0])
    expected = (weights * torch.arange(1.0, 6.0)).sum() / weights.sum()
    assert torch.all(state.attention_count == 5)
    assert torch.allclose(state.attention_utility(), torch.tensor([[expected, 2.0 * expected]]))


def test_attention_normalizer_matches_bias_corrected_ema() -> None:
    state = _state(1)
    numerator = torch.tensor(0.0)
    normalizer = torch.tensor(0.0)
    for value in (1.0, 3.0, 2.0, 6.0, 4.0):
        state.update_attention_utility(
            torch.tensor([[value]]), ema_decay=0.9, freeze_updates=5
        )
        numerator = 0.9 * numerator + 0.1 * value
        normalizer = 0.9 * normalizer + 0.1
    expected = numerator / normalizer
    assert torch.allclose(state.attention_utility(), expected.view(1, 1))


def test_current_frame_keeps_initialized_mean_confidence_gate() -> None:
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_conf_gate=True,
    )
    scores = torch.ones(1, 2)
    initialized_gate = torch.tensor([[0.4, 0.6]])
    frame_ids = torch.tensor([[[7, 7]]], dtype=torch.int32)
    gated = manager._apply_confidence_gate(
        scores,
        candidate_depth_confidence=None,
        candidate_point_confidence=None,
        candidate_conf_gate=initialized_gate,
        candidate_frame_ids=frame_ids,
        current_frame_id=7,
        shared_across_heads=True,
    )
    assert torch.allclose(gated, initialized_gate)


def test_concat_gather_slice_alignment() -> None:
    left = _state(3, frame_id=0)
    right = _state(2, frame_id=1)
    left.update_attention_utility(torch.tensor([[1.0, 2.0, 3.0]]), ema_decay=0.9, freeze_updates=5)
    right.update_attention_utility(torch.tensor([[4.0, 5.0]]), ema_decay=0.9, freeze_updates=5)
    joined = left.concat(right)
    selected = joined.gather(torch.tensor([[4, 1, 3]]))
    assert torch.equal(selected.frame_ids, torch.tensor([[1, 0, 1]], dtype=torch.int32))
    assert torch.allclose(selected.attention_utility(), torch.tensor([[5.0, 2.0, 4.0]]))
    assert torch.allclose(joined.slice(1, 4).attention_utility(), torch.tensor([[2.0, 3.0, 4.0]]))
    assert selected.attention_accum.dtype == torch.float32
    assert selected.attention_normalizer.dtype == torch.float32
    assert selected.attention_count.dtype == torch.uint8


def test_stac_scaling_is_cache_length_stable() -> None:
    for query_tokens, live_tokens in ((4, 8), (4, 32), (13, 7)):
        uniform_colsum = torch.full((1, 3, live_tokens), query_tokens / live_tokens)
        scaled = scale_stac_colsum(uniform_colsum, live_tokens, query_tokens)
        assert torch.allclose(scaled, torch.ones_like(scaled))


def test_beta_zero_and_attention_tiebreak() -> None:
    raw_scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    def fixed_scores(self, candidate_k, candidate_v=None, return_basis=False):
        out = raw_scores.to(candidate_k.device)
        return (out, None) if return_basis else out

    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="topk",
    )
    manager._layer_svd_leverage_scores = MethodType(fixed_scores, manager)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn_like(k)
    utility = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    baseline = manager.select(k, 2, 0, v=v)
    beta_zero = manager.select(
        k, 2, 0, v=v, candidate_attention_utility=utility, attention_utility_beta=0.0
    )
    mixed = manager.select(
        k, 2, 0, v=v, candidate_attention_utility=utility, attention_utility_beta=0.8
    )
    assert torch.equal(baseline.kept_candidate_indices, beta_zero.kept_candidate_indices)
    assert 0 in mixed.kept_candidate_indices[0, 0].tolist()


def test_mean_normalization_preserves_score_ratios() -> None:
    scores = torch.tensor([[0.10, 0.11, 0.12]])
    uniform_utility = torch.ones_like(scores)
    mixed = EvictionManager._blend_attention_utility(
        scores,
        uniform_utility,
        beta=0.2,
        shared_across_heads=True,
    )
    expected = 0.8 * (scores / scores.mean(dim=-1, keepdim=True)) + 0.2
    assert torch.allclose(mixed, expected)
    assert torch.allclose(
        mixed[:, 1:] - mixed[:, :-1],
        torch.full((1, 2), 0.8 * 0.01 / 0.11),
        atol=1e-6,
    )


def test_unobserved_tokens_keep_full_leverage_score() -> None:
    scores = torch.tensor([[0.10, 0.20, 0.30, 0.40]])
    utility = torch.tensor([[2.0, 100.0, 4.0, 100.0]])
    observed = torch.tensor([[True, False, True, False]])
    mixed = EvictionManager._blend_attention_utility(
        scores,
        utility,
        beta=0.5,
        shared_across_heads=True,
        attention_observed=observed,
    )
    normalized_scores = scores / scores.mean(dim=-1, keepdim=True)
    expected = normalized_scores.clone()
    expected[:, 0] = 0.5 * normalized_scores[:, 0] + 0.5 * (2.0 / 3.0)
    expected[:, 2] = 0.5 * normalized_scores[:, 2] + 0.5 * (4.0 / 3.0)
    assert torch.allclose(mixed, expected)
    assert torch.allclose(mixed[:, [1, 3]], normalized_scores[:, [1, 3]])

    all_unobserved = EvictionManager._blend_attention_utility(
        scores,
        utility,
        beta=0.5,
        shared_across_heads=True,
        attention_observed=torch.zeros_like(observed),
    )
    assert torch.equal(all_unobserved, scores)


def test_cuda_attention_reference_if_available() -> None:
    if not torch.cuda.is_available():
        return
    try:
        import attn_cuda
    except (ImportError, OSError):
        return
    if not attn_cuda.is_available():
        return

    torch.manual_seed(0)
    q = torch.randn(1, 17, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 23, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 23, 2, 64, device="cuda", dtype=torch.float16)
    out, lse, colsum = attn_cuda.flash_attn_bias_colsum(q, k, v, return_colsum=True)
    q_bhmd = q.transpose(1, 2).contiguous()
    k_bhnd = k.transpose(1, 2).contiguous()
    v_bhnd = v.transpose(1, 2).contiguous()
    out_bhmd, lse_bhmd, colsum_bhnd = attn_cuda.flash_attn_bias_colsum_bhnd(
        q_bhmd, k_bhnd, v_bhnd, return_colsum=True
    )
    scale = 64**-0.5
    logits = torch.einsum("bmhd,bnhd->bhmn", q.float(), k.float()) * scale
    probs = logits.softmax(dim=-1)
    ref_out = torch.einsum("bhmn,bnhd->bmhd", probs, v.float())
    ref_lse = torch.logsumexp(logits, dim=-1)
    ref_colsum = probs.sum(dim=-2)
    assert out.dtype == torch.float16
    assert lse.dtype == torch.float32 and colsum.dtype == torch.float32
    assert torch.allclose(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-2, rtol=2e-2)
    assert torch.allclose(colsum, ref_colsum, atol=2e-2, rtol=2e-2)
    assert torch.allclose(out_bhmd.transpose(1, 2), out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse_bhmd, lse, atol=2e-2, rtol=2e-2)
    assert torch.allclose(colsum_bhnd, colsum, atol=2e-2, rtol=2e-2)


def test_cuda_beta_zero_pre_eviction_cache_parity_if_available() -> None:
    if not torch.cuda.is_available():
        return
    try:
        import attn_cuda
    except (ImportError, OSError):
        return
    if not attn_cuda.is_available():
        return

    torch.manual_seed(2)
    baseline_attention = Attention(dim=128, num_heads=2, rope=None).cuda().half().eval()
    utility_attention = copy.deepcopy(baseline_attention)
    common_kwargs = dict(
        use_cache=True,
        cache_budget=10,
        tokens_per_frame=8,
        eviction_policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="topk",
        leverage_conf_gate=True,
        leverage_attention_beta=0.0,
    )
    baseline_cache = None
    utility_cache = None
    for frame_idx in range(6):
        frame_x = torch.randn(1, 8, 128, device="cuda", dtype=torch.float16)
        _, baseline_cache, _ = baseline_attention(
            frame_x,
            past_key_values=baseline_cache,
            step_idx=frame_idx,
            current_frame_ids=[frame_idx],
            current_frame_idx=frame_idx,
            leverage_attention_utility=False,
            **common_kwargs,
        )
        _, utility_cache, _ = utility_attention(
            frame_x,
            past_key_values=utility_cache,
            step_idx=frame_idx,
            current_frame_ids=[frame_idx],
            current_frame_idx=frame_idx,
            leverage_attention_utility=True,
            leverage_attention_ema_decay=0.9,
            leverage_attention_freeze_updates=5,
            leverage_attention_colsum_subsample_ratio=1.0,
            **common_kwargs,
        )
        baseline_k, baseline_v, baseline_state = unpack_kv_cache(baseline_cache)
        utility_k, utility_v, utility_state = unpack_kv_cache(utility_cache)
        assert baseline_k.shape[2] <= 10 and utility_k.shape[2] <= 10
        assert torch.equal(baseline_k, utility_k)
        assert torch.equal(baseline_v, utility_v)
        assert baseline_state is not None and utility_state is not None
        assert torch.equal(baseline_state.frame_ids, utility_state.frame_ids)
        assert torch.equal(baseline_state.token_indices, utility_state.token_indices)
        assert torch.equal(baseline_state.confidence_gate, utility_state.confidence_gate)


def test_cuda_pre_attention_eviction_if_available() -> None:
    if not torch.cuda.is_available():
        return
    try:
        import attn_cuda
    except (ImportError, OSError):
        return
    if not attn_cuda.is_available():
        return

    torch.manual_seed(1)
    attention = Attention(dim=128, num_heads=2, rope=None).cuda().eval()
    kwargs = dict(
        use_cache=True,
        cache_budget=10,
        tokens_per_frame=8,
        eviction_policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="topk",
        leverage_attention_utility=True,
        leverage_attention_beta=0.2,
        leverage_attention_ema_decay=0.9,
        leverage_attention_freeze_updates=5,
        leverage_attention_colsum_subsample_ratio=1.0,
    )
    first_x = torch.randn(1, 8, 128, device="cuda", dtype=torch.float32)
    _, first_cache, _ = attention(first_x, step_idx=0, current_frame_ids=[0], current_frame_idx=0, **kwargs)
    cache = first_cache
    out = first_x
    for frame_idx in range(1, 20):
        frame_x = torch.randn(1, 8, 128, device="cuda", dtype=torch.float32)
        out, cache, _ = attention(
            frame_x,
            past_key_values=cache,
            step_idx=frame_idx,
            current_frame_ids=[frame_idx],
            current_frame_idx=frame_idx,
            **kwargs,
        )
        k, _, state = unpack_kv_cache(cache)
        assert out.shape == frame_x.shape
        assert out.dtype == torch.float32
        assert k.dtype == torch.float16
        assert k.shape[2] <= 10
        assert state is not None and state.attention_count is not None
        assert state.attention_count.shape == state.frame_ids.shape
        assert state.attention_count.shape[1] == k.shape[2]
        assert torch.all((state.attention_count >= 1) & (state.attention_count <= 5))
        current_mask = state.frame_ids.eq(frame_idx)
        if torch.any(current_mask):
            assert torch.all(state.attention_count[current_mask] == 1)
    k, _, state = unpack_kv_cache(cache)
    assert state is not None and state.attention_count is not None
    assert torch.all((state.attention_count >= 1) & (state.attention_count <= 5))


def main() -> None:
    test_five_updates_then_freeze()
    test_attention_normalizer_matches_bias_corrected_ema()
    test_current_frame_keeps_initialized_mean_confidence_gate()
    test_concat_gather_slice_alignment()
    test_stac_scaling_is_cache_length_stable()
    test_beta_zero_and_attention_tiebreak()
    test_mean_normalization_preserves_score_ratios()
    test_unobserved_tokens_keep_full_leverage_score()
    test_cuda_attention_reference_if_available()
    test_cuda_beta_zero_pre_eviction_cache_parity_if_available()
    test_cuda_pre_attention_eviction_if_available()
    print("frozen early-attention utility tests passed")


if __name__ == "__main__":
    main()
