#!/usr/bin/env python3
"""Unit checks for the compactor policy and the IDC ablation switches (rebuttal experiment 3).

Run: python tools/test_compactor_ablation.py   (CPU; the attention module test also runs without CUDA)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamvggt.layers.attention import _materialize_kv_by_keep_indices  # noqa: E402
from streamvggt.layers.confidence_state import KVConfidenceState  # noqa: E402
from streamvggt.layers.eviction import EvictionManager  # noqa: E402
from streamvggt.utils import ablation_config  # noqa: E402


def _reset_config():
    ablation_config.configure(**{f: getattr(ablation_config.AblationConfig(), f)
                                 for f in ablation_config.AblationConfig.__dataclass_fields__})


def test_noncausal_colsum_matches_dense() -> None:
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 3, 7, 8, generator=g)
    k = torch.randn(1, 3, 11, 8, generator=g)
    got = EvictionManager.noncausal_attention_colsum(q, k, query_chunk=3)
    ref = torch.softmax(q @ k.transpose(-1, -2) / 8 ** 0.5, dim=-1).sum(dim=2)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def _numpy_compactor(k, q, anchor, lam, ridge, headwise_norm):
    k = k.double().numpy()[0]
    q = q.double().numpy()[0]
    H, N, D = k.shape
    attn = np.zeros((H, N))
    for h in range(H):
        logits = q[h] @ k[h].T / np.sqrt(D)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        attn[h] = p.sum(axis=0)
    attn = attn[:, anchor:]
    lev = np.zeros_like(attn)
    for h in range(H):
        x = k[h, anchor:]
        if headwise_norm:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        x = x - x.mean(axis=0)
        inv = np.linalg.inv(x.T @ x + ridge * np.eye(D))
        lev[h] = np.einsum("nd,de,ne->n", x, inv, x)
    z = lambda a: (a - a.mean()) / a.std()  # noqa: E731  (pooled over heads and tokens, population std)
    return z(attn) + lam * z(lev), lev


def test_compactor_scores_match_numpy() -> None:
    g = torch.Generator().manual_seed(1)
    k = torch.randn(1, 4, 40, 8, generator=g) * 3 + 1
    q = torch.randn(1, 4, 9, 8, generator=g)
    for headwise in (False, True):
        ablation_config.configure(compactor_lambda=0.5, compactor_ridge=5e-3, compactor_headwise_norm=headwise)
        manager = EvictionManager(policy="compactor", layer_budget_strategy="uniform")
        scores, lev = manager._compactor_scores(k, q, num_anchor_tokens=5)
        ref_scores, ref_lev = _numpy_compactor(k, q, 5, 0.5, 5e-3, headwise)
        assert np.allclose(lev.numpy()[0], ref_lev, rtol=1e-4, atol=1e-6)
        assert np.allclose(scores.numpy()[0], ref_scores, rtol=1e-4, atol=1e-5)
    _reset_config()


def test_compactor_select_is_per_head_and_materializes_per_head() -> None:
    g = torch.Generator().manual_seed(2)
    B, H, N, D, anchor, budget = 1, 4, 30, 8, 6, 16
    k = torch.randn(B, H, N, D, generator=g)
    v = torch.randn(B, H, N, D, generator=g)
    q = torch.randn(B, H, 5, D, generator=g)
    manager = EvictionManager(policy="compactor", layer_budget_strategy="uniform")
    result = manager.select(k, budget, anchor, v=v, query=q)
    kept = result.kept_candidate_indices
    assert kept.shape == (B, H, budget - anchor)
    scores, _ = manager._compactor_scores(k, q, anchor)
    for h in range(H):
        ref = torch.topk(scores[0, h], budget - anchor).indices.sort().values
        assert torch.equal(kept[0, h], ref)
    assert not all(torch.equal(kept[0, 0], kept[0, h]) for h in range(1, H)), "heads should differ"
    keep = torch.cat([torch.arange(anchor).view(1, 1, -1).expand(B, H, anchor), kept + anchor], dim=2)
    fk, fv = _materialize_kv_by_keep_indices(k, v, keep, head_shared=False)
    for h in range(H):
        assert torch.equal(fk[0, h], k[0, h, keep[0, h]])
        assert torch.equal(fv[0, h], v[0, h, keep[0, h]])


def test_compactor_kaba_score_uses_per_head_pr() -> None:
    g = torch.Generator().manual_seed(3)
    k = torch.randn(1, 4, 30, 8, generator=g)
    v = torch.randn(1, 4, 30, 8, generator=g)
    q = torch.randn(1, 4, 5, 8, generator=g)
    manager = EvictionManager(policy="compactor", layer_budget_strategy="value_weighted_leverage_pr")
    result = manager.select(k, 16, 6, v=v, query=q)
    assert result.layer_budget_score is not None


def test_blend_defaults_unchanged_and_multiplier() -> None:
    _reset_config()
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    util = torch.tensor([[0.5, 0.1, 0.0, 0.0]])
    observed = torch.tensor([[True, True, False, False]])
    legacy = EvictionManager._blend_attention_utility(
        scores, util, 0.5, shared_across_heads=True, attention_observed=observed)
    ablation_config.configure(unobserved_lev_multiplier=2.0)
    doubled = EvictionManager._blend_attention_utility(
        scores, util, 0.5, shared_across_heads=True, attention_observed=observed)
    assert torch.allclose(legacy, doubled), "m=2 with beta=0.5 must equal the legacy rule"
    ablation_config.configure(unobserved_lev_multiplier=1.0)
    single = EvictionManager._blend_attention_utility(
        scores, util, 0.5, shared_across_heads=True, attention_observed=observed)
    assert torch.allclose(single[:, 2:], 0.5 * legacy[:, 2:]) and torch.allclose(single[:, :2], legacy[:, :2])
    ablation_config.configure(unobserved_lev_multiplier=None, score_norm="zscore", compactor_lambda=0.5)
    z = EvictionManager._blend_attention_utility(
        scores, util, 0.5, shared_across_heads=True, attention_observed=observed)
    zs = (scores - scores.mean()) / scores.std(unbiased=False)
    obs_u = util[observed]
    zu = (util - obs_u.mean()) / obs_u.std(unbiased=False)
    expected = torch.where(observed, zu + 0.5 * zs, zs)
    assert torch.allclose(z, expected, atol=1e-6), (z, expected)
    _reset_config()


def test_freeze_zero_never_freezes() -> None:
    state = KVConfidenceState.for_current_frame(1, 2, 2, 0, torch.device("cpu"), initialize_attention=True)
    for i in range(300):
        state.update_attention_utility(torch.tensor([[1.0, float(i)]]), ema_decay=0.9, freeze_updates=0)
    assert int(state.attention_count.max()) == 255
    weights = 0.9 ** torch.arange(299, -1, -1, dtype=torch.float64)
    expected = float((weights * torch.arange(300, dtype=torch.float64)).sum() / weights.sum())
    assert abs(float(state.attention_utility()[0, 1]) - expected) < 1e-2 * expected


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
