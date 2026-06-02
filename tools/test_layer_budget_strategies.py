import torch

from streamvggt.layers.eviction import (
    _allocate_layer_budgets_from_scores,
    _layer_score_leverage_entropy,
    _layer_score_leverage_pr,
)


def _assert_budget_invariants(budgets, capacities, total_budget):
    assert sum(budgets.values()) == min(total_budget, sum(capacities.values()))
    for layer, budget in budgets.items():
        assert 0 <= budget <= capacities[layer]


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
    test_equal_distributions_are_uniform()
    test_spread_layer_gets_larger_budget()
    test_alpha_zero_is_uniform()
    test_zero_leverage_is_finite_and_uniform_fallback()
    test_total_budget_retains_all()
    test_min_tokens_and_capacity_caps()
    print("layer budget strategy sanity checks passed")
