from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamvggt.layers.eviction import EvictionManager


def _fixed_layer_scores(scores: torch.Tensor):
    def _impl(self, candidate_k, candidate_v=None, return_basis=False):
        out = scores.to(device=candidate_k.device)
        if return_basis:
            return out, None
        return out

    return _impl


def test_recency_score_bias_changes_topk_eviction():
    k = torch.randn(1, 1, 5, 4)
    v = torch.randn(1, 1, 5, 4)
    raw_scores = torch.tensor([[0.0, 0.0, 0.2, 0.8, 1.0]])
    frame_ids = torch.tensor([[[0, 10, 0, 0, 0]]])
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="topk",
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=1.0,
        leverage_dpp_recency_window=10,
        leverage_dpp_recency_gate_power=0.0,
        layer_budget_strategy="leverage_pr",
    )
    manager._layer_svd_leverage_scores = MethodType(_fixed_layer_scores(raw_scores), manager)

    result = manager.select(
        k,
        cache_budget=3,
        num_anchor_tokens=0,
        v=v,
        candidate_frame_ids=frame_ids,
        current_frame_idx=10,
    )

    kept = result.kept_candidate_indices[0, 0].tolist()
    assert 1 in kept, f"recent low-score token should receive a soft score boost: kept={kept}"
    assert 0 not in kept, f"older equally low-score token should be evicted first: kept={kept}"
    assert result.layer_budget_score is not None
    assert torch.allclose(result.layer_budget_score, torch.tensor(2.3810), atol=1e-4)


def test_recency_score_bias_fast_dpp_smoke():
    torch.manual_seed(0)
    k = torch.randn(1, 1, 6, 4)
    v = torch.randn(1, 1, 6, 4)
    raw_scores = torch.tensor([[0.0, 0.0, 0.1, 0.3, 0.7, 1.0]])
    frame_ids = torch.tensor([[[0, 10, 0, 0, 0, 0]]])
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_eviction_selector="fast_dpp",
        leverage_dpp_candidate_multiplier=2,
        leverage_dpp_greedy_block_size=2,
        leverage_dpp_quality_beta=0.0,
        leverage_dpp_diversity_beta=1.0,
        leverage_dpp_recency_bonus=True,
        leverage_dpp_recency_lambda=1.0,
        leverage_dpp_recency_window=10,
        leverage_dpp_recency_gate_power=0.0,
    )
    manager._layer_svd_leverage_scores = MethodType(_fixed_layer_scores(raw_scores), manager)

    result = manager.select(
        k,
        cache_budget=3,
        num_anchor_tokens=0,
        v=v,
        candidate_frame_ids=frame_ids,
        current_frame_idx=10,
    )

    assert result.kept_candidate_indices.shape == (1, 1, 3)
    adjusted = result.policy_scores[0]
    assert adjusted[1] > adjusted[0], adjusted


def main():
    test_recency_score_bias_changes_topk_eviction()
    test_recency_score_bias_fast_dpp_smoke()
    print("recency-weighted eviction tests passed")


if __name__ == "__main__":
    main()
