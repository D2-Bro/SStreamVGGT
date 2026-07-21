import torch

from streamvggt.layers.eviction import (
    EvictionManager,
    _allocate_layer_budgets_from_scores,
    _combine_value_weighted_leverage_pr_scores,
    _integer_allocate_by_weights,
    _layer_score_leverage_pr,
    _layer_value_norm_score,
)


def test_leverage_participation_ratio():
    concentrated = torch.tensor([1.0, 0.0, 0.0])
    uniform = torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(_layer_score_leverage_pr(concentrated), torch.tensor(1.0))
    assert torch.allclose(_layer_score_leverage_pr(uniform), torch.tensor(3.0))


def test_value_norm_score():
    rows = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    mean_norm = _layer_value_norm_score(rows, norm_type="mean", eps=0.0)
    rms_norm = _layer_value_norm_score(rows, norm_type="rms", eps=0.0)
    assert torch.allclose(mean_norm, torch.tensor(2.5))
    assert rms_norm > mean_norm


def test_value_weighted_scores():
    base = torch.tensor([1.0, 1.0, 1.0])
    value_norms = torch.tensor([1.0, 2.0, 4.0])
    combined = _combine_value_weighted_leverage_pr_scores(
        base,
        value_norms,
        gamma=1.0,
        eps=0.0,
    )
    assert combined[2] > combined[1] > combined[0]


def test_integer_capacity_redistribution():
    capacities = {0: 2, 1: 100, 2: 100}
    weights = {0: 10.0, 1: 1.0, 2: 1.0}
    budgets = _integer_allocate_by_weights(capacities, weights, total_budget=20)
    assert sum(budgets.values()) == 20
    assert budgets[0] == 2
    assert budgets[1] + budgets[2] == 18


def test_score_allocation_invariants():
    capacities = {0: 3, 1: 20, 2: 30}
    scores = {0: torch.tensor(10.0), 1: torch.tensor(2.0), 2: torch.tensor(1.0)}
    budgets = _allocate_layer_budgets_from_scores(
        scores,
        capacities,
        total_budget=25,
        alpha=0.7,
    )
    assert sum(budgets.values()) == 25
    assert all(0 <= budgets[layer] <= capacities[layer] for layer in capacities)


def test_current_manager_configuration():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=256,
        rls_refresh_interval=8,
        leverage_eviction_selector="topk",
        layer_budget_strategy="value_weighted_leverage_pr",
        layer_budget_value_gamma=0.7,
        layer_budget_value_norm_type="mean",
        layer_budget_norm_source="key",
    )
    assert manager.leverage_ridge_dim == 256
    assert manager.rls_refresh_interval == 8


if __name__ == "__main__":
    test_leverage_participation_ratio()
    test_value_norm_score()
    test_value_weighted_scores()
    test_integer_capacity_redistribution()
    test_score_allocation_invariants()
    test_current_manager_configuration()
    print("active layer budget strategy sanity checks passed")
