import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from compute_cosine_layer_budget import compute_cosine_similarity, cosine_to_proportions

from streamvggt.layers.eviction import (
    EvictionManager,
    _allocate_layer_budgets_from_scores,
    _combine_layer_budget_pr_base,
    _combine_value_weighted_leverage_pr_scores,
    _covariance_participation_ratio,
    _layer_score_leverage_entropy,
    _layer_score_leverage_pr,
    _layer_value_norm_score,
)


def _assert_budget_invariants(budgets, capacities, total_budget):
    assert sum(budgets.values()) == min(total_budget, sum(capacities.values()))
    for layer, budget in budgets.items():
        assert 0 <= budget <= capacities[layer]



from streamvggt.models.aggregator import Aggregator
from streamvggt.models.streamvggt import StreamVGGT


def _make_proportion_holder(depth):
    return SimpleNamespace(
        aggregator=SimpleNamespace(
            depth=depth,
            last_scores=torch.zeros(depth),
            layer_budget_proportions=None,
        )
    )


def test_precomputed_proportions_load_and_normalize():
    holder = _make_proportion_holder(3)
    normalized = StreamVGGT.set_layer_budget_proportions(holder, [1.0, 2.0, 1.0])
    assert torch.allclose(normalized, torch.tensor([0.25, 0.5, 0.25]))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "budget.json"
        path.write_text(json.dumps({"strategy": "cosine", "num_layers": 3, "proportions": [2.0, 1.0, 1.0]}), encoding="utf-8")
        loaded = StreamVGGT.set_layer_budget_proportions(holder, path)
    assert torch.allclose(loaded, torch.tensor([0.5, 0.25, 0.25]))


def test_precomputed_proportions_reject_invalid_values():
    holder = _make_proportion_holder(3)
    bad_values = (
        [1.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 1.0],
        [1.0, float("nan"), 1.0],
        {"num_layers": 4, "proportions": [1.0, 1.0, 1.0]},
    )
    for values in bad_values:
        try:
            StreamVGGT.set_layer_budget_proportions(holder, values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid proportions accepted: {values}")


def test_cosine_precomputed_is_independent_of_leverage_mode():
    EvictionManager(policy="mean", leverage_granularity="head", layer_budget_strategy="cosine_precomputed")


def test_cosine_precomputed_capacity_aware_allocation():
    agg = Aggregator(
        img_size=16,
        patch_size=8,
        embed_dim=32,
        depth=3,
        num_heads=4,
        mlp_ratio=1.0,
        num_register_tokens=2,
        patch_embed="conv",
    )
    agg.layer_budget_proportions = torch.tensor([0.8, 0.1, 0.1])

    def make_kv(tokens):
        value = torch.zeros(1, 1, tokens, 1)
        return value, value

    budgets = agg._calculate_dynamic_budgets(
        10,
        layer_budget_strategy="cosine_precomputed",
        past_key_values=[make_kv(2), make_kv(100), make_kv(100)],
    )
    assert budgets.tolist() == [2, 4, 4]
    assert int(budgets.sum().item()) == 10

    ranged_budgets = agg._calculate_dynamic_budgets(
        10,
        enabled_global_idx_ranges=[(1, None)],
        layer_budget_strategy="cosine_precomputed",
        past_key_values=[make_kv(100), make_kv(100), make_kv(100)],
    )
    assert ranged_budgets.tolist() == [0, 5, 5]
    assert int(ranged_budgets.sum().item()) == 10


def test_cosine_helpers():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    identical = compute_cosine_similarity(x, x)
    opposite = compute_cosine_similarity(x, -x)
    assert torch.allclose(identical, torch.tensor(1.0))
    assert torch.allclose(opposite, torch.tensor(-1.0))
    importance, proportions = cosine_to_proportions([identical, opposite], temperature=0.5)
    assert proportions[1] > proportions[0]
    assert torch.allclose(proportions.sum(), torch.tensor(1.0))
    assert importance[1] > importance[0]




def test_covariance_pr_rank_and_scale_properties():
    rank1 = torch.diag(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(_covariance_participation_ratio(rank1), torch.tensor(1.0))

    four_equal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(_covariance_participation_ratio(four_equal), torch.tensor(4.0))
    assert torch.allclose(
        _covariance_participation_ratio(four_equal),
        _covariance_participation_ratio(10.0 * four_equal),
    )

    zero = torch.zeros(4, 4)
    zero_pr = _covariance_participation_ratio(zero)
    assert torch.isfinite(zero_pr)
    assert zero_pr.item() == 0.0

    nonfinite = torch.diag(torch.tensor([float("nan"), float("inf"), 3.0]))
    nonfinite_pr = _covariance_participation_ratio(nonfinite)
    assert torch.isfinite(nonfinite_pr)
    assert torch.allclose(nonfinite_pr, torch.tensor(1.0))


def test_spectral_pr_uses_raw_participation_ratio():
    leverage_pr = torch.tensor(100.0)
    covariance_pr = torch.tensor(3.5)
    kwargs = {
        "slots_per_direction": 4.0,
        "hybrid_beta": 0.5,
        "eps": 1e-12,
    }
    covariance_score = _combine_layer_budget_pr_base(
        leverage_pr,
        covariance_pr,
        "covariance_pr",
        **kwargs,
    )
    spectral_score = _combine_layer_budget_pr_base(
        leverage_pr,
        covariance_pr,
        "spectral_pr",
        **kwargs,
    )
    value_weighted_spectral_score = _combine_layer_budget_pr_base(
        leverage_pr,
        covariance_pr,
        "value_weighted_spectral_pr",
        **kwargs,
    )
    assert torch.allclose(spectral_score, covariance_pr)
    assert torch.allclose(value_weighted_spectral_score, covariance_pr)
    assert torch.allclose(covariance_score, covariance_pr * kwargs["slots_per_direction"])

    other_pr = torch.tensor(1.0)
    covariance_other_score = _combine_layer_budget_pr_base(
        leverage_pr,
        other_pr,
        "covariance_pr",
        **kwargs,
    )
    spectral_other_score = _combine_layer_budget_pr_base(
        leverage_pr,
        other_pr,
        "spectral_pr",
        **kwargs,
    )
    capacities = {0: 100, 1: 100}
    covariance_budgets = _allocate_layer_budgets_from_scores(
        {0: covariance_score, 1: covariance_other_score},
        capacities,
        total_budget=81,
        alpha=0.7,
    )
    spectral_budgets = _allocate_layer_budgets_from_scores(
        {0: spectral_score, 1: spectral_other_score},
        capacities,
        total_budget=81,
        alpha=0.7,
    )
    assert spectral_budgets == covariance_budgets


def test_hybrid_budget_pr_base_helpers():
    leverage_pr = torch.tensor(100.0)
    covariance_pr = torch.tensor(10.0)
    cap = _combine_layer_budget_pr_base(
        leverage_pr,
        covariance_pr,
        "hybrid_cap",
        slots_per_direction=4.0,
        hybrid_beta=0.5,
        eps=1e-12,
    )
    assert torch.allclose(cap, torch.tensor(40.0))

    geom = _combine_layer_budget_pr_base(
        leverage_pr,
        covariance_pr,
        "hybrid_geom",
        slots_per_direction=4.0,
        hybrid_beta=0.5,
        eps=1e-12,
    )
    assert torch.allclose(geom, torch.sqrt(torch.tensor(100.0 * 40.0)))


def test_ridge_layer_budget_reuses_raw_gram_for_covariance_pr():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="full_d_ridge",
        layer_budget_strategy="covariance_pr",
        slots_per_direction=4.0,
    )
    candidate_k = torch.zeros(1, 1, 4, 4)
    candidate_k[0, 0, :, :] = torch.eye(4)
    scores = manager._layer_svd_leverage_scores(candidate_k)
    assert scores.shape == (1, 4)
    assert manager._last_layer_covariance_pr is not None
    assert torch.allclose(manager._last_layer_covariance_pr.reshape(()), torch.tensor(4.0), atol=1e-5)
    payload = manager._compute_layer_budget_score(scores, candidate_k=candidate_k)
    assert torch.allclose(payload, torch.tensor(16.0), atol=1e-5)


def test_spectral_pr_reuses_ridge_gram_cache():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=4,
        rls_refresh_interval=8,
        layer_budget_strategy="spectral_pr",
        slots_per_direction=4.0,
    )
    candidate_k = torch.zeros(1, 1, 4, 4)
    candidate_k[0, 0, :, :] = torch.eye(4)

    manager._set_leverage_diag_context(
        layer_id=0,
        step_idx=0,
        current_frame_idx=0,
        granularity="layer",
        batch_size=1,
        num_heads=1,
    )
    first_scores = manager._layer_svd_leverage_scores(candidate_k)
    first_gram = manager.cached_ktk.clone()
    first_pr = manager._last_layer_covariance_pr.clone()
    first_payload = manager._compute_layer_budget_score(first_scores, candidate_k=candidate_k)

    manager._set_leverage_diag_context(
        layer_id=0,
        step_idx=1,
        current_frame_idx=1,
        granularity="layer",
        batch_size=1,
        num_heads=1,
    )
    second_scores = manager._layer_svd_leverage_scores(torch.ones_like(candidate_k))
    second_payload = manager._compute_layer_budget_score(second_scores, candidate_k=candidate_k)

    assert manager.rls_refresh_count == 1
    assert manager.rls_cache_hit_count == 1
    assert torch.allclose(manager.cached_ktk, first_gram)
    assert torch.allclose(manager._last_layer_covariance_pr, first_pr)
    assert torch.allclose(second_payload, first_payload)


def test_new_layer_budget_strategies_are_validated():
    strategies = (
        "covariance_pr",
        "spectral_pr",
        "hybrid_cap",
        "hybrid_geom",
        "value_weighted_covariance_pr",
        "value_weighted_spectral_pr",
        "value_weighted_hybrid_cap",
        "value_weighted_hybrid_geom",
    )
    for strategy in strategies:
        EvictionManager(policy="svd_leverage", leverage_granularity="layer", layer_budget_strategy=strategy)

def test_value_norm_helper_mean_and_rms():
    values = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    assert torch.allclose(_layer_value_norm_score(values, "mean"), torch.tensor(8.5))
    assert torch.allclose(_layer_value_norm_score(values, "rms"), torch.sqrt(torch.tensor(84.5)))


def test_value_norm_helper_empty_and_nonfinite_are_safe():
    empty = torch.empty(0, 2)
    assert _layer_value_norm_score(empty, "rms").item() == 0.0
    nonfinite = torch.tensor([[float("nan"), float("inf")], [3.0, 4.0]])
    assert torch.isfinite(_layer_value_norm_score(nonfinite, "rms"))


def test_value_weighted_gamma_zero_matches_pr():
    base = torch.tensor([10.0, 20.0])
    value_norms = torch.tensor([1.0, 100.0])
    scores = _combine_value_weighted_leverage_pr_scores(base, value_norms, gamma=0.0)
    assert torch.allclose(scores, base)


def test_value_weighted_equal_values_preserve_pr():
    base = torch.tensor([10.0, 20.0])
    value_norms = torch.tensor([7.0, 7.0])
    scores = _combine_value_weighted_leverage_pr_scores(base, value_norms, gamma=0.5)
    assert torch.allclose(scores, base)


def test_value_weighted_higher_value_norm_increases_equal_pr_score():
    base = torch.tensor([10.0, 10.0])
    value_norms = torch.tensor([1.0, 9.0])
    scores = _combine_value_weighted_leverage_pr_scores(base, value_norms, gamma=1.0)
    assert scores[1] > scores[0]


def test_value_weighted_zero_values_fall_back_to_pr():
    base = torch.tensor([10.0, 20.0])
    value_norms = torch.zeros(2)
    scores = _combine_value_weighted_leverage_pr_scores(base, value_norms, gamma=1.0)
    assert torch.allclose(scores, base)


def test_value_weighted_manager_returns_base_pr_and_value_norm_payload():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        layer_budget_strategy="value_weighted_leverage_pr",
        layer_budget_value_norm_type="rms",
    )
    policy_scores = torch.ones(1, 4)
    candidate_v = torch.ones(1, 2, 4, 3)
    payload = manager._compute_layer_budget_score(policy_scores, candidate_v=candidate_v)
    assert payload.shape == (2,)
    assert torch.allclose(payload[0], _layer_score_leverage_pr(policy_scores[0]))
    assert payload[1] > 0


def test_value_weighted_spectral_pr_manager_payload_uses_raw_pr_and_selected_norm():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        layer_budget_strategy="value_weighted_spectral_pr",
        layer_budget_value_norm_type="mean",
        layer_budget_norm_source="key",
        slots_per_direction=17.0,
    )
    manager._last_layer_covariance_pr = torch.tensor([3.5])
    policy_scores = torch.ones(1, 4)
    candidate_k = torch.full((1, 2, 4, 3), 3.0)
    expected_rows = candidate_k[0].transpose(0, 1).reshape(4, 6)
    expected_norm = _layer_value_norm_score(expected_rows, "mean")

    payload = manager._compute_layer_budget_score(policy_scores, candidate_k=candidate_k)

    assert payload.shape == (2,)
    assert torch.allclose(payload[0], torch.tensor(3.5))
    assert torch.allclose(payload[1], expected_norm)


def test_value_weighted_manager_key_norm_source_uses_candidate_k():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        layer_budget_strategy="value_weighted_leverage_pr",
        layer_budget_value_norm_type="mean",
        layer_budget_norm_source="key",
    )
    policy_scores = torch.ones(1, 4)
    candidate_k = torch.full((1, 2, 4, 3), 3.0)
    candidate_v = torch.ones(1, 2, 4, 3)
    payload = manager._compute_layer_budget_score(
        policy_scores,
        candidate_k=candidate_k,
        candidate_v=candidate_v,
    )
    value_manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        layer_budget_strategy="value_weighted_leverage_pr",
        layer_budget_value_norm_type="mean",
        layer_budget_norm_source="value",
    )
    value_payload = value_manager._compute_layer_budget_score(
        policy_scores,
        candidate_k=candidate_k,
        candidate_v=candidate_v,
    )
    assert payload[1] > 0
    assert payload[1] > value_payload[1]


def test_key_norm_layer_budget_uses_candidate_key_norm_only():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        layer_budget_strategy="key_norm",
        layer_budget_value_norm_type="mean",
    )
    low_pr_scores = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    high_pr_scores = torch.ones(1, 4)
    candidate_k = torch.full((1, 2, 4, 3), 3.0)
    expected_rows = candidate_k[0].transpose(0, 1).reshape(4, 6)
    expected = _layer_value_norm_score(expected_rows, "mean")

    low_payload = manager._compute_layer_budget_score(low_pr_scores, candidate_k=candidate_k)
    high_payload = manager._compute_layer_budget_score(high_pr_scores, candidate_k=candidate_k)

    assert torch.allclose(low_payload, expected)
    assert torch.allclose(high_payload, expected)


def test_invalid_layer_budget_norm_source_rejected():
    try:
        EvictionManager(
            policy="svd_leverage",
            leverage_granularity="layer",
            layer_budget_strategy="value_weighted_leverage_pr",
            layer_budget_norm_source="query",
        )
    except ValueError as exc:
        assert "layer_budget_norm_source" in str(exc)
    else:
        raise AssertionError("invalid layer_budget_norm_source was accepted")


def test_value_weighted_allocation_invariants():
    base = torch.tensor([100.0, 25.0, 25.0])
    value_norms = torch.tensor([1.0, 5.0, 10.0])
    weighted = _combine_value_weighted_leverage_pr_scores(base, value_norms, gamma=0.5)
    capacities = {0: 100, 1: 100, 2: 100}
    budgets = _allocate_layer_budgets_from_scores(
        {idx: weighted[idx] for idx in range(3)},
        capacities,
        121,
        alpha=1.0,
    )
    _assert_budget_invariants(budgets, capacities, 121)

def test_equal_distributions_are_uniform():
    lev0 = torch.ones(100)
    lev1 = torch.ones(100)
    for score_fn in (_layer_score_leverage_pr, _layer_score_leverage_entropy):
        scores = {0: score_fn(lev0), 1: score_fn(lev1)}
        budgets = _allocate_layer_budgets_from_scores(scores, {0: 100, 1: 100}, 101, alpha=1.0)
        assert abs(budgets[0] - budgets[1]) <= 1
        _assert_budget_invariants(budgets, {0: 100, 1: 100}, 101)


def test_spread_layer_gets_larger_budget():
    spread = torch.ones(1000)
    concentrated = torch.cat([torch.ones(10), torch.zeros(990)])
    for score_fn in (_layer_score_leverage_pr, _layer_score_leverage_entropy):
        scores = {0: score_fn(spread), 1: score_fn(concentrated)}
        budgets = _allocate_layer_budgets_from_scores(scores, {0: 1000, 1: 1000}, 200, alpha=1.0)
        assert budgets[0] > budgets[1]
        _assert_budget_invariants(budgets, {0: 1000, 1: 1000}, 200)


def test_alpha_zero_is_uniform():
    scores = {0: torch.tensor(1000.0), 1: torch.tensor(1.0)}
    budgets = _allocate_layer_budgets_from_scores(scores, {0: 100, 1: 100}, 51, alpha=0.0)
    assert abs(budgets[0] - budgets[1]) <= 1
    _assert_budget_invariants(budgets, {0: 100, 1: 100}, 51)


def test_zero_leverage_is_finite_and_uniform_fallback():
    zero = torch.zeros(32)
    assert _layer_score_leverage_pr(zero).item() == 0.0
    assert _layer_score_leverage_entropy(zero).item() == 0.0
    budgets, details = _allocate_layer_budgets_from_scores(
        {0: torch.tensor(0.0), 1: torch.tensor(float("nan"))},
        {0: 10, 1: 10},
        7,
        return_debug=True,
    )
    assert details["fallback_uniform"]
    assert abs(budgets[0] - budgets[1]) <= 1
    _assert_budget_invariants(budgets, {0: 10, 1: 10}, 7)


def test_total_budget_retains_all():
    budgets = _allocate_layer_budgets_from_scores(
        {0: torch.tensor(1.0), 1: torch.tensor(100.0)},
        {0: 3, 1: 5},
        99,
    )
    assert budgets == {0: 3, 1: 5}
    _assert_budget_invariants(budgets, {0: 3, 1: 5}, 99)


def test_min_tokens_and_capacity_caps():
    budgets = _allocate_layer_budgets_from_scores(
        {0: torch.tensor(100.0), 1: torch.tensor(1.0), 2: torch.tensor(1.0)},
        {0: 10, 1: 4, 2: 4},
        12,
        alpha=1.0,
        min_tokens=3,
    )
    assert budgets[1] >= 3
    assert budgets[2] >= 3
    _assert_budget_invariants(budgets, {0: 10, 1: 4, 2: 4}, 12)


if __name__ == "__main__":
    test_precomputed_proportions_load_and_normalize()
    test_precomputed_proportions_reject_invalid_values()
    test_cosine_precomputed_is_independent_of_leverage_mode()
    test_cosine_precomputed_capacity_aware_allocation()
    test_cosine_helpers()
    test_covariance_pr_rank_and_scale_properties()
    test_spectral_pr_uses_raw_participation_ratio()
    test_hybrid_budget_pr_base_helpers()
    test_ridge_layer_budget_reuses_raw_gram_for_covariance_pr()
    test_spectral_pr_reuses_ridge_gram_cache()
    test_new_layer_budget_strategies_are_validated()
    test_value_norm_helper_mean_and_rms()
    test_value_norm_helper_empty_and_nonfinite_are_safe()
    test_value_weighted_gamma_zero_matches_pr()
    test_value_weighted_equal_values_preserve_pr()
    test_value_weighted_higher_value_norm_increases_equal_pr_score()
    test_value_weighted_zero_values_fall_back_to_pr()
    test_value_weighted_manager_returns_base_pr_and_value_norm_payload()
    test_value_weighted_spectral_pr_manager_payload_uses_raw_pr_and_selected_norm()
    test_value_weighted_manager_key_norm_source_uses_candidate_k()
    test_key_norm_layer_budget_uses_candidate_key_norm_only()
    test_invalid_layer_budget_norm_source_rejected()
    test_value_weighted_allocation_invariants()
    test_equal_distributions_are_uniform()
    test_spread_layer_gets_larger_budget()
    test_alpha_zero_is_uniform()
    test_zero_leverage_is_finite_and_uniform_fallback()
    test_total_budget_retains_all()
    test_min_tokens_and_capacity_caps()
    print("layer budget strategy sanity checks passed")
