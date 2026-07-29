import tempfile
from pathlib import Path

import torch

from streamvggt.layers.attention import Attention
from streamvggt.layers.eviction import (
    EvictionManager,
    _allocate_layer_budgets_from_scores,
    _combine_value_weighted_leverage_pr_scores,
    _integer_allocate_by_weights,
    _layer_score_leverage_pr,
)
from streamvggt.models.aggregator import Aggregator


def test_leverage_participation_ratio():
    concentrated = torch.tensor([1.0, 0.0, 0.0])
    uniform = torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(_layer_score_leverage_pr(concentrated), torch.tensor(1.0))
    assert torch.allclose(_layer_score_leverage_pr(uniform), torch.tensor(3.0))


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


def _make_budget_aggregator(depth):
    aggregator = Aggregator.__new__(Aggregator)
    torch.nn.Module.__init__(aggregator)
    aggregator.depth = depth
    aggregator.last_scores = torch.zeros(depth)
    aggregator.last_layer_budget_scores = torch.zeros(depth)
    aggregator.last_layer_budget_base_scores = torch.zeros(depth)
    aggregator.last_layer_budget_value_norms = torch.zeros(depth)
    return aggregator


def _cache_with_tokens(num_tokens):
    if num_tokens <= 0:
        return None
    cache = torch.zeros(1, 1, num_tokens, 1)
    return cache, cache.clone()


def test_uniform_budget_preserves_total():
    for depth, total_budget in ((3, 10), (24, 100)):
        aggregator = _make_budget_aggregator(depth)
        budgets = aggregator._calculate_dynamic_budgets(
            total_budget,
            layer_budget_strategy="uniform",
            past_key_values=[None] * depth,
            current_token_count=100,
        )
        assert int(budgets.sum()) == total_budget
        assert int(budgets.max() - budgets.min()) <= 1
        if depth == 3:
            assert budgets.tolist() == [4, 3, 3]
        else:
            assert budgets[:4].tolist() == [5, 5, 5, 5]
            assert budgets[4:].tolist() == [4] * 20


def test_uniform_budget_respects_and_redistributes_capacity():
    aggregator = _make_budget_aggregator(3)
    budgets = aggregator._calculate_dynamic_budgets(
        20,
        layer_budget_strategy="uniform",
        past_key_values=[
            _cache_with_tokens(2),
            _cache_with_tokens(100),
            _cache_with_tokens(100),
        ],
    )
    assert budgets.tolist() == [2, 9, 9]


def test_uniform_budget_edge_cases():
    aggregator = _make_budget_aggregator(3)
    past_key_values = [
        None,
        _cache_with_tokens(4),
        _cache_with_tokens(6),
    ]
    zero_budgets = aggregator._calculate_dynamic_budgets(
        -1,
        layer_budget_strategy="uniform",
        past_key_values=past_key_values,
    )
    full_budgets = aggregator._calculate_dynamic_budgets(
        100,
        layer_budget_strategy="uniform",
        past_key_values=past_key_values,
    )
    assert zero_budgets.tolist() == [0, 0, 0]
    assert full_budgets.tolist() == [0, 4, 6]


def test_uniform_svd_leverage_skips_layer_score():
    manager = EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_dim=4,
        layer_budget_strategy="uniform",
    )
    result = manager.select(
        torch.randn(1, 2, 8, 4),
        cache_budget=6,
        num_anchor_tokens=2,
        v=torch.randn(1, 2, 8, 4),
    )
    assert result.kept_candidate_indices.shape == (1, 2, 4)
    assert result.layer_budget_score is None


def test_batched_budget_conversion_matches_item_reference():
    aggregator = _make_budget_aggregator(4)
    aggregator.last_layer_budget_scores = torch.tensor([1.25, 3.5, 2.0, 7.75])
    base_scores = torch.tensor([2.0, 5.0, 1.0, 4.0])
    value_norms = torch.tensor([0.5, 3.0, 2.0, 8.0])
    aggregator.last_layer_budget_base_scores = base_scores
    aggregator.last_layer_budget_value_norms = value_norms
    past_key_values = [
        _cache_with_tokens(9),
        _cache_with_tokens(11),
        _cache_with_tokens(13),
        _cache_with_tokens(15),
    ]
    capacities = {0: 9, 1: 11, 2: 13, 3: 15}
    total_budget = 31
    alpha = 0.6
    gamma = 0.7

    for strategy in ("uniform", "leverage_pr", "key_norm", "value_weighted_leverage_pr"):
        if strategy == "uniform":
            active_scores = torch.zeros_like(aggregator.last_layer_budget_scores)
            allocator_alpha = 0.0
        elif strategy == "value_weighted_leverage_pr":
            active_scores = _combine_value_weighted_leverage_pr_scores(
                base_scores,
                value_norms,
                gamma=gamma,
                eps=0.0,
                alpha=alpha,
            )
            allocator_alpha = 1.0
        else:
            active_scores = aggregator.last_layer_budget_scores
            allocator_alpha = alpha
        item_score_dict = {
            idx: float(torch.nan_to_num(active_scores[idx], nan=0.0, posinf=0.0, neginf=0.0).item())
            for idx in range(active_scores.numel())
        }
        expected = _allocate_layer_budgets_from_scores(
            item_score_dict,
            capacities,
            total_budget,
            alpha=allocator_alpha,
            eps=0.0,
        )
        actual = aggregator._calculate_dynamic_budgets(
            total_budget,
            layer_budget_strategy=strategy,
            layer_budget_value_gamma=gamma,
            layer_budget_alpha=alpha,
            layer_budget_eps=0.0,
            past_key_values=past_key_values,
        )
        assert actual.tolist() == [expected[idx] for idx in range(4)], strategy
def test_layer_budget_payload_stays_tensor():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        payload = torch.tensor([2.0, 4.0], device=device)
        score, base, norm = Aggregator._coerce_layer_budget_payload(
            payload,
            fallback_score=torch.tensor(0.0),
            fallback_base=torch.tensor(0.0),
            fallback_value_norm=torch.tensor(0.0),
            strategy="value_weighted_leverage_pr",
            device=device,
        )
        assert all(
            isinstance(value, torch.Tensor) and value.device.type == device.type for value in (score, base, norm)
        )
        assert torch.equal(torch.stack((score, base, norm)), torch.tensor([2.0, 2.0, 4.0], device=device))
        pscore, pbase, pnorm = Aggregator._coerce_layer_budget_payload(payload[:1], 0.0, 0.0, 0.0, "leverage_pr", device)
        assert torch.equal(torch.stack((pscore, pbase, pnorm)), torch.tensor([2.0, 2.0, 0.0], device=device))


def _score_only_manager(*, projected_key_cache=False):
    return EvictionManager(
        policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_lambda=1e-3,
        leverage_ridge_lambda_mode="absolute",
        leverage_ridge_jitter=1e-6,
        leverage_ridge_dim=4,
        rls_refresh_interval=1,
        leverage_random_seed=17,
        leverage_projected_key_cache=projected_key_cache,
        layer_budget_strategy="value_weighted_leverage_pr",
        layer_budget_value_gamma=0.7,
        layer_budget_value_norm_type="mean",
        layer_budget_norm_source="key",
    )


def test_score_only_matches_eviction_raw_payload():
    torch.manual_seed(5)
    k = torch.randn(1, 2, 10, 4)
    v = torch.randn_like(k)
    eviction_manager = _score_only_manager()
    score_manager = _score_only_manager()

    eviction_result = eviction_manager.select(
        k,
        cache_budget=8,
        num_anchor_tokens=2,
        v=v,
        step_idx=1,
        current_frame_idx=1,
    )
    score_result = score_manager.score_layer_budget(
        k,
        2,
        v=v,
        step_idx=1,
        current_frame_idx=1,
    )
    assert torch.allclose(score_result.policy_scores, eviction_result.policy_scores)
    assert torch.allclose(score_result.layer_budget_score, eviction_result.layer_budget_score)


def test_score_only_attention_preserves_full_cache():
    torch.manual_seed(7)
    attention = Attention(dim=8, num_heads=2)
    k = torch.randn(1, 2, 9, 4)
    v = torch.randn_like(k)
    final_k, final_v, payload = attention.eviction(
        k,
        v,
        cache_budget=3,
        num_anchor_tokens=2,
        eviction_policy="svd_leverage",
        leverage_granularity="layer",
        leverage_approx_method="right_sketch_ridge",
        leverage_ridge_lambda=1e-3,
        leverage_ridge_lambda_mode="absolute",
        leverage_ridge_jitter=1e-6,
        leverage_ridge_dim=4,
        rls_refresh_interval=1,
        leverage_random_seed=17,
        layer_budget_strategy="value_weighted_leverage_pr",
        layer_budget_value_gamma=0.7,
        layer_budget_value_norm_type="mean",
        layer_budget_norm_source="key",
        layer_budget_score_only=True,
    )
    assert torch.equal(final_k, k)
    assert torch.equal(final_v, v)
    assert isinstance(payload, tuple) and len(payload) == 2
    assert payload[1].shape == (2,)


def test_score_only_projected_cache_grows_without_eviction():
    torch.manual_seed(11)
    manager = _score_only_manager(projected_key_cache=True)
    first_k = torch.randn(1, 2, 8, 4)
    first_v = torch.randn_like(first_k)
    manager.score_layer_budget(first_k, 2, v=first_v, step_idx=1, current_frame_idx=1)
    cached_prefix = manager._projected_key_cache.clone()
    assert cached_prefix.shape[1] == 6

    tail_k = torch.randn(1, 2, 3, 4)
    tail_v = torch.randn_like(tail_k)
    manager.score_layer_budget(
        torch.cat((first_k, tail_k), dim=2),
        2,
        v=torch.cat((first_v, tail_v), dim=2),
        step_idx=2,
        current_frame_idx=2,
    )
    assert manager._projected_key_cache.shape[1] == 9
    assert torch.equal(manager._projected_key_cache[:, :6], cached_prefix)


def test_score_only_multiframe_forward_grows_cache():
    torch.manual_seed(13)
    attention = Attention(dim=8, num_heads=2)
    common_kwargs = {
        "use_cache": True,
        "cache_budget": 1,
        "eviction_policy": "svd_leverage",
        "leverage_granularity": "layer",
        "leverage_approx_method": "right_sketch_ridge",
        "leverage_ridge_lambda": 1e-3,
        "leverage_ridge_lambda_mode": "absolute",
        "leverage_ridge_jitter": 1e-6,
        "leverage_ridge_dim": 4,
        "rls_refresh_interval": 1,
        "leverage_random_seed": 17,
        "leverage_projected_key_cache": True,
        "layer_budget_strategy": "value_weighted_leverage_pr",
        "layer_budget_value_gamma": 0.7,
        "layer_budget_value_norm_type": "mean",
        "layer_budget_norm_source": "key",
        "layer_budget_score_only": True,
    }
    _, first_cache, first_scores = attention(
        torch.randn(1, 3, 8),
        step_idx=0,
        current_frame_idx=0,
        **common_kwargs,
    )
    assert first_cache[0].shape[2] == 3
    assert torch.equal(first_scores[1], torch.zeros_like(first_scores[1]))

    _, second_cache, second_scores = attention(
        torch.randn(1, 3, 8),
        past_key_values=first_cache,
        step_idx=1,
        current_frame_idx=1,
        **common_kwargs,
    )
    assert second_cache[0].shape[2] == 6
    assert second_scores[1].shape == (2,)
    assert torch.isfinite(second_scores[1]).all()


def test_score_only_rejects_invalid_combinations():
    attention = Attention(dim=8, num_heads=2)
    x = torch.randn(1, 3, 8)

    def expect_error(fragment, **overrides):
        kwargs = {
            "use_cache": True,
            "cache_budget": 1,
            "eviction_policy": "svd_leverage",
            "leverage_granularity": "layer",
            "layer_budget_strategy": "value_weighted_leverage_pr",
            "layer_budget_score_only": True,
        }
        kwargs.update(overrides)
        try:
            attention(x, **kwargs)
        except ValueError as exc:
            assert fragment in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected ValueError containing {fragment!r}")

    expect_error("eviction_policy='svd_leverage'", eviction_policy="mean")
    expect_error("leverage_granularity='layer'", leverage_granularity="head")
    expect_error(
        "layer_budget_strategy='value_weighted_leverage_pr'",
        layer_budget_strategy="leverage_pr",
    )
    expect_error(
        "incompatible with leverage_attention_utility",
        leverage_attention_utility=True,
    )
    expect_error(
        "cache_analysis_config",
        cache_analysis_config=object(),
    )
    expect_error(
        "eviction_nn_analysis_config",
        eviction_nn_analysis_config=object(),
    )
    expect_error(
        "token_overlay_dump_config",
        token_overlay_dump_config=object(),
    )


def test_score_only_shadow_budget_log_event():
    aggregator = _make_budget_aggregator(2)
    aggregator.last_layer_budget_base_scores = torch.tensor([1.0, 2.0])
    aggregator.last_layer_budget_value_norms = torch.tensor([1.0, 3.0])
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "layer_budget_scores.csv"
        aggregator._calculate_dynamic_budgets(
            5,
            layer_budget_strategy="value_weighted_leverage_pr",
            layer_budget_score_only=True,
            layer_budget_log_path=str(log_path),
            current_token_count=4,
        )
        assert "score_only_shadow" in log_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_leverage_participation_ratio()
    test_value_weighted_scores()
    test_integer_capacity_redistribution()
    test_score_allocation_invariants()
    test_current_manager_configuration()
    test_uniform_budget_preserves_total()
    test_uniform_budget_respects_and_redistributes_capacity()
    test_uniform_budget_edge_cases()
    test_uniform_svd_leverage_skips_layer_score()
    test_batched_budget_conversion_matches_item_reference()
    test_layer_budget_payload_stays_tensor()
    test_score_only_matches_eviction_raw_payload()
    test_score_only_attention_preserves_full_cache()
    test_score_only_projected_cache_grows_without_eviction()
    test_score_only_multiframe_forward_grows_cache()
    test_score_only_rejects_invalid_combinations()
    test_score_only_shadow_budget_log_event()
    print("active layer budget strategy sanity checks passed")
