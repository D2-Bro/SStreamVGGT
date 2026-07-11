"""Head-wise KV cache eviction policies for streaming attention."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


VALID_LAYER_BUDGET_STRATEGIES = (
    "uniform",
    "cosine_precomputed",
    "leverage_pr",
    "covariance_pr",
    "spectral_pr",
    "hybrid_cap",
    "hybrid_geom",
    "leverage_entropy",
    "key_norm",
    "depth_weighted_leverage_pr",
    "value_weighted_leverage_pr",
    "value_weighted_covariance_pr",
    "value_weighted_spectral_pr",
    "value_weighted_hybrid_cap",
    "value_weighted_hybrid_geom",
)


LAYER_BUDGET_SCORE_STRATEGIES = (
    "leverage_pr",
    "covariance_pr",
    "spectral_pr",
    "hybrid_cap",
    "hybrid_geom",
    "leverage_entropy",
    "key_norm",
    "depth_weighted_leverage_pr",
    "value_weighted_leverage_pr",
    "value_weighted_covariance_pr",
    "value_weighted_spectral_pr",
    "value_weighted_hybrid_cap",
    "value_weighted_hybrid_geom",
)


VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES = (
    "value_weighted_leverage_pr",
    "value_weighted_covariance_pr",
    "value_weighted_spectral_pr",
    "value_weighted_hybrid_cap",
    "value_weighted_hybrid_geom",
)


DEPTH_WEIGHTED_LAYER_BUDGET_STRATEGIES = (
    "depth_weighted_leverage_pr",
)


COVARIANCE_LAYER_BUDGET_STRATEGIES = (
    "covariance_pr",
    "spectral_pr",
    "hybrid_cap",
    "hybrid_geom",
    "value_weighted_covariance_pr",
    "value_weighted_spectral_pr",
    "value_weighted_hybrid_cap",
    "value_weighted_hybrid_geom",
)


KV_LEVERAGE_FEATURES = ("key_value", "key_value_lowdim_concat")


def _participation_ratio_from_values(values: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    if values is None or values.numel() == 0:
        device = values.device if values is not None else "cpu"
        return torch.tensor(0.0, device=device)
    values = values.float()
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.clamp(values, min=0.0)
    sum_values = values.sum(dim=-1)
    sum_squares = values.square().sum(dim=-1)
    pr = sum_values.square() / sum_squares.clamp_min(eps)
    return torch.where(sum_values > eps, pr, torch.zeros_like(pr))


def _covariance_participation_ratio(covariance: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    if covariance is None or covariance.numel() == 0:
        device = covariance.device if covariance is not None else "cpu"
        return torch.tensor(0.0, device=device)
    covariance = torch.nan_to_num(covariance.float(), nan=0.0, posinf=0.0, neginf=0.0)
    trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    frob2 = covariance.square().sum(dim=(-2, -1))
    pr = trace.square() / frob2.clamp_min(eps)
    return torch.where(trace.abs() > eps, pr, torch.zeros_like(pr))


def _layer_score_leverage_pr(lev: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Effective count from the participation ratio of QR leverage mass."""
    return _participation_ratio_from_values(lev, eps)


def _combine_layer_budget_pr_base(
    leverage_pr: torch.Tensor,
    covariance_pr: Optional[torch.Tensor],
    strategy: str,
    slots_per_direction: float,
    hybrid_beta: float,
    eps: float,
) -> torch.Tensor:
    if strategy in ("leverage_pr", "depth_weighted_leverage_pr", "value_weighted_leverage_pr"):
        return leverage_pr
    if covariance_pr is None:
        raise ValueError(f"{strategy} requires covariance participation ratio from layer features")
    covariance_pr = torch.as_tensor(covariance_pr, device=leverage_pr.device, dtype=leverage_pr.dtype)
    if strategy in ("spectral_pr", "value_weighted_spectral_pr"):
        return covariance_pr.clamp_min(0.0)
    direction_base = covariance_pr.clamp_min(0.0) * float(slots_per_direction)
    if strategy in ("covariance_pr", "value_weighted_covariance_pr"):
        return direction_base
    if strategy in ("hybrid_cap", "value_weighted_hybrid_cap"):
        return torch.minimum(leverage_pr, direction_base)
    if strategy in ("hybrid_geom", "value_weighted_hybrid_geom"):
        lev_base = leverage_pr.clamp_min(eps)
        dir_base = direction_base.clamp_min(eps)
        return lev_base.pow(float(hybrid_beta)) * dir_base.pow(1.0 - float(hybrid_beta))
    raise AssertionError(f"Unhandled layer budget strategy: {strategy}")


def _layer_score_leverage_entropy(lev: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Effective count from entropy of normalized QR leverage mass."""
    if lev is None or lev.numel() == 0:
        device = lev.device if lev is not None else "cpu"
        return torch.tensor(0.0, device=device)
    lev = lev.float()
    lev = torch.nan_to_num(lev, nan=0.0, posinf=0.0, neginf=0.0)
    lev = torch.clamp(lev, min=0.0)
    total = lev.sum()
    if total <= eps:
        return torch.zeros((), device=lev.device, dtype=lev.dtype)
    p = lev / total.clamp_min(eps)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    return torch.exp(entropy)


def _layer_value_norm_score(
    values: Optional[torch.Tensor],
    norm_type: str = "rms",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Layer-level value magnitude score from token value rows."""
    if values is None or values.numel() == 0:
        device = values.device if values is not None else "cpu"
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    if norm_type not in ("mean", "rms"):
        raise ValueError(f"layer_budget_value_norm_type must be 'mean' or 'rms', got {norm_type!r}")
    with torch.no_grad():
        v = values.float()
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.numel() == 0:
            return torch.zeros((), device=values.device, dtype=torch.float32)
        v_norm = torch.linalg.vector_norm(v, ord=2, dim=-1)
        if v_norm.numel() == 0:
            return torch.zeros((), device=values.device, dtype=torch.float32)
        if norm_type == "mean":
            return v_norm.mean()
        return torch.sqrt(v_norm.square().mean().clamp_min(eps))


def _combine_value_weighted_leverage_pr_scores(
    base_scores,
    value_norms,
    gamma: float,
    eps: float = 1e-12,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base = torch.nan_to_num(torch.as_tensor(base_scores).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    base = torch.clamp(base, min=0.0)
    values = torch.nan_to_num(torch.as_tensor(value_norms).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.clamp(values, min=0.0).to(device=base.device)
    prior = torch.ones_like(base)
    active = torch.ones_like(base, dtype=torch.bool)
    if active_mask is not None:
        active = torch.as_tensor(active_mask, device=base.device, dtype=torch.bool)
        if active.shape != base.shape:
            raise ValueError(f"active_mask shape {tuple(active.shape)} must match base_scores {tuple(base.shape)}")
    if float(gamma) != 0.0 and active.any():
        active_values = values[active]
        value_mean = active_values.mean() if active_values.numel() else values.new_tensor(0.0)
        if torch.isfinite(value_mean) and float(value_mean.item()) > eps:
            prior_active = (active_values / value_mean.clamp_min(eps)).clamp_min(0.0).pow(float(gamma))
            prior[active] = torch.nan_to_num(prior_active, nan=1.0, posinf=1.0, neginf=1.0)
    return base * prior


def _as_nonnegative_float(value, eps: float) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        value = value.detach().float()
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        value = value.mean().item()
    value = float(value)
    if not math.isfinite(value) or value <= eps:
        return 0.0
    return value


def _integer_allocate_by_weights(
    capacities: Dict[int, int],
    weights: Dict[int, float],
    total_budget: int,
) -> Dict[int, int]:
    budgets = {int(layer): 0 for layer in capacities}
    total_capacity = sum(max(int(cap), 0) for cap in capacities.values())
    target = min(max(int(total_budget), 0), total_capacity)
    if target <= 0:
        return budgets
    if target >= total_capacity:
        return {int(layer): max(int(cap), 0) for layer, cap in capacities.items()}

    remaining = target
    while remaining > 0:
        active = [
            layer
            for layer, cap in capacities.items()
            if max(int(cap), 0) - budgets.get(layer, 0) > 0
        ]
        if not active:
            break
        weight_sum = sum(max(float(weights.get(layer, 0.0)), 0.0) for layer in active)
        if weight_sum <= 0.0 or not math.isfinite(weight_sum):
            active_weight = {layer: 1.0 for layer in active}
            weight_sum = float(len(active))
        else:
            active_weight = {
                layer: max(float(weights.get(layer, 0.0)), 0.0)
                for layer in active
            }

        quotas = {
            layer: remaining * active_weight[layer] / weight_sum
            for layer in active
        }
        for layer in active:
            residual_cap = max(int(capacities[layer]), 0) - budgets[layer]
            amount = min(residual_cap, int(math.floor(quotas[layer])))
            if amount > 0:
                budgets[layer] += amount

        remaining = target - sum(budgets.values())
        if remaining <= 0:
            break

        order = sorted(
            active,
            key=lambda layer: (
                quotas[layer] - math.floor(quotas[layer]),
                active_weight[layer],
                -layer,
            ),
            reverse=True,
        )
        progressed = False
        for layer in order:
            if remaining <= 0:
                break
            if budgets[layer] < max(int(capacities[layer]), 0):
                budgets[layer] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return budgets


def _allocate_layer_budgets_from_scores(
    scores: Dict[int, torch.Tensor],
    capacities: Dict[int, int],
    total_budget: int,
    alpha: float = 0.5,
    min_tokens: int = 0,
    eps: float = 1e-12,
    return_debug: bool = False,
):
    """Allocate integer layer budgets from scalar layer scores."""
    if alpha < 0:
        raise ValueError(f"layer budget alpha must be >= 0, got {alpha}")
    if min_tokens < 0:
        raise ValueError(f"layer budget min_tokens must be >= 0, got {min_tokens}")

    capacities = {int(layer): max(int(cap), 0) for layer, cap in capacities.items()}
    active_layers = [layer for layer, cap in capacities.items() if cap > 0]
    target = min(max(int(total_budget), 0), sum(capacities[layer] for layer in active_layers))
    budgets = {layer: 0 for layer in capacities}
    details = {
        "scores": {},
        "weights": {},
        "raw_budgets": {},
        "target_budget": target,
        "assigned_budget": 0,
        "total_capacity": sum(capacities[layer] for layer in active_layers),
        "fallback_uniform": False,
        "events": [],
    }
    if not active_layers or target <= 0:
        return (budgets, details) if return_debug else budgets
    if target >= details["total_capacity"]:
        for layer in active_layers:
            budgets[layer] = capacities[layer]
            details["raw_budgets"][layer] = float(capacities[layer])
        details["assigned_budget"] = sum(budgets.values())
        details["events"].append("total_budget>=total_capacity")
        return (budgets, details) if return_debug else budgets

    clean_scores = {
        layer: _as_nonnegative_float(scores.get(layer, 0.0), eps)
        for layer in active_layers
    }
    if alpha == 0.0 or all(score <= 0.0 for score in clean_scores.values()):
        weights = {layer: 1.0 for layer in active_layers}
        details["fallback_uniform"] = alpha != 0.0
        if details["fallback_uniform"]:
            details["events"].append("all_layer_scores_invalid_or_zero")
    else:
        weights = {
            layer: (clean_scores[layer] + eps) ** float(alpha)
            for layer in active_layers
        }
        if not any(math.isfinite(weight) and weight > 0.0 for weight in weights.values()):
            weights = {layer: 1.0 for layer in active_layers}
            details["fallback_uniform"] = True
            details["events"].append("all_layer_weights_invalid")

    for layer in active_layers:
        details["scores"][layer] = clean_scores[layer]
        details["weights"][layer] = weights[layer]

    min_tokens = int(min_tokens)
    base = {layer: 0 for layer in active_layers}
    if min_tokens > 0:
        base = {layer: min(min_tokens, capacities[layer]) for layer in active_layers}
        base_total = sum(base.values())
        if base_total >= target:
            details["events"].append("min_tokens_consumed_budget")
            budgets.update(
                _integer_allocate_by_weights(
                    {layer: base[layer] for layer in active_layers},
                    {layer: 1.0 for layer in active_layers},
                    target,
                )
            )
            base_weight_sum = sum(base.values())
            for layer in active_layers:
                details["raw_budgets"][layer] = (
                    target * base[layer] / base_weight_sum if base_weight_sum > 0 else 0.0
                )
            details["assigned_budget"] = sum(budgets.values())
            return (budgets, details) if return_debug else budgets
        budgets.update(base)

    remaining = target - sum(budgets.values())
    residual_caps = {
        layer: capacities[layer] - budgets[layer]
        for layer in active_layers
        if capacities[layer] - budgets[layer] > 0
    }
    residual_weight_sum = sum(weights[layer] for layer in residual_caps)
    if residual_weight_sum <= 0.0 or not math.isfinite(residual_weight_sum):
        residual_weights = {layer: 1.0 for layer in residual_caps}
        residual_weight_sum = float(len(residual_caps))
    else:
        residual_weights = {layer: weights[layer] for layer in residual_caps}
    for layer in active_layers:
        details["raw_budgets"][layer] = float(budgets[layer])
        if layer in residual_caps and residual_weight_sum > 0.0:
            details["raw_budgets"][layer] += remaining * residual_weights[layer] / residual_weight_sum

    residual_alloc = _integer_allocate_by_weights(residual_caps, residual_weights, remaining)
    for layer, amount in residual_alloc.items():
        budgets[layer] += amount
    details["assigned_budget"] = sum(budgets.values())
    if any(budgets[layer] >= capacities[layer] for layer in active_layers):
        details["events"].append("capacity_cap_applied")
    return (budgets, details) if return_debug else budgets


@dataclass
class EvictionResult:
    """Indices and scores produced by a head-wise eviction policy."""

    kept_candidate_indices: torch.Tensor
    policy_scores: torch.Tensor
    mean_scores: torch.Tensor
    summary_score: float
    leverage_basis: Optional["SvdLeverageBasis"] = None
    layer_budget_score: Optional[torch.Tensor] = None


@dataclass
class SvdLeverageBasis:
    """Reusable low-rank coordinates from the leverage computation."""

    q: torch.Tensor
    r_diag: torch.Tensor
    granularity: str
    basis_kind: str = "qr"


class EvictionManager:
    """Dispatches head-wise cache eviction policies."""

    VALID_POLICIES = ("mean", "baseline_mean", "svd_leverage", "dpp")
    VALID_LEVERAGE_EVICTION_RISK_MODES = ("low_leverage", "outlier_then_low")
    VALID_LEVERAGE_DPP_FEATURE_PROJECTIONS = ("raw", "random")
    VALID_LEVERAGE_SIMILARITY_FEATURE_PROJECTIONS = ("raw", "random")
    VALID_LEVERAGE_SIMILARITY_GRANULARITIES = ("layer", "head")
    VALID_LEVERAGE_APPROX_METHODS = (
        "exact_qr",
        "right_sketch",        "full_d_ridge",
        "right_sketch_ridge",
    )
    VALID_LEVERAGE_EVICTION_SELECTORS = ("topk", "fast_dpp", "layer_head_fast_dpp", "similarity_topk")

    def __init__(
        self,
        policy: str = "mean",
        profile: bool = False,
        debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_normalize_before_projection: bool = False,
        leverage_normalize_before_projection_headwise: bool = False,
        leverage_projected_key_cache: bool = False,
        leverage_approx_method: str = "right_sketch",
        leverage_ridge_lambda: float = 1e-3,
        leverage_ridge_lambda_mode: str = "relative",
        leverage_ridge_score_chunk_size: int = 4096,
        leverage_ridge_jitter: float = 1e-6,
        leverage_ridge_dim: Optional[int] = None,
        rls_refresh_interval: int = 1,
        leverage_diag: bool = False,
        leverage_diag_interval: int = 0,
        leverage_random_seed: int = 0,
        leverage_eviction_selector: str = "topk",
        leverage_similarity_granularity: str = "layer",
        leverage_similarity_feature_projection: str = "raw",
        leverage_similarity_leverage_gamma: float = 1.0,
        leverage_eviction_risk_mode: str = "low_leverage",
        leverage_high_outlier_z: float = 3.0,
        leverage_dpp_candidate_multiplier: int = 2,
        leverage_dpp_greedy_block_size: int = 32,
        leverage_dpp_quality_beta: float = 1.0,
        leverage_dpp_diversity_beta: float = 1.0,
        leverage_dpp_feature_projection: str = "raw",
        leverage_dpp_recency_bonus: bool = False,
        leverage_dpp_recency_lambda: float = 0.2,
        leverage_dpp_recency_window: int = 5,
        leverage_dpp_recency_gate_power: float = 1.0,
        leverage_dpp_recency_debug: bool = False,
        leverage_conf_gate: bool = False,
        leverage_conf_gate_floor: float = 0.2,
        leverage_conf_gate_depth_alpha: float = 1.0,
        leverage_conf_gate_point_beta: float = 1.0,
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.5,
        layer_budget_value_norm_type: str = "rms",
        layer_budget_norm_source: str = "value",
        layer_budget_eps: float = 1e-12,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
    ) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Unknown eviction policy '{policy}'. Valid policies: {self.VALID_POLICIES}")
        if leverage_sketch_dim is not None and leverage_sketch_dim < 0:
            raise ValueError(f"leverage_sketch_dim must be >= 0 or None, got {leverage_sketch_dim}")
        if leverage_head_mean_dim < 1:
            raise ValueError(f"leverage_head_mean_dim must be >= 1, got {leverage_head_mean_dim}")
        if leverage_granularity not in ("head", "layer"):
            raise ValueError("leverage_granularity must be 'head' or 'layer', got " f"{leverage_granularity!r}")
        if leverage_feature not in ("key", "key_value", "key_value_lowdim_concat"):
            raise ValueError(
                "leverage_feature must be 'key', 'key_value', or "
                f"'key_value_lowdim_concat', got {leverage_feature!r}"
            )
        if leverage_projection not in ("random", "head_mean"):
            raise ValueError(
                "leverage_projection must be 'random' or 'head_mean', got "
                f"{leverage_projection!r}"
            )
        if leverage_projection == "head_mean" and leverage_granularity != "layer":
            raise ValueError("leverage_projection='head_mean' requires leverage_granularity='layer'")
        if leverage_projection == "head_mean" and leverage_feature != "key":
            raise ValueError("leverage_projection='head_mean' requires leverage_feature='key'")
        if leverage_feature == "key_value_lowdim_concat":
            if leverage_granularity != "layer":
                raise ValueError("leverage_feature='key_value_lowdim_concat' requires leverage_granularity='layer'")
            if leverage_projection != "random":
                raise ValueError("leverage_feature='key_value_lowdim_concat' requires leverage_projection='random'")
            if leverage_approx_method not in ("right_sketch", "right_sketch_ridge"):
                raise ValueError(
                    "leverage_feature='key_value_lowdim_concat' requires "
                    "leverage_approx_method='right_sketch' or 'right_sketch_ridge'"
                )
            if leverage_ridge_dim is None:
                raise ValueError("leverage_feature='key_value_lowdim_concat' requires leverage_ridge_dim >= 1")
        if leverage_approx_method not in self.VALID_LEVERAGE_APPROX_METHODS:
            raise ValueError(
                "leverage_approx_method must be one of "
                f"{self.VALID_LEVERAGE_APPROX_METHODS}, got {leverage_approx_method!r}"
            )
        if leverage_eviction_selector not in self.VALID_LEVERAGE_EVICTION_SELECTORS:
            raise ValueError(
                "leverage_eviction_selector must be one of "
                f"{self.VALID_LEVERAGE_EVICTION_SELECTORS}, got {leverage_eviction_selector!r}"
            )
        if leverage_eviction_risk_mode not in self.VALID_LEVERAGE_EVICTION_RISK_MODES:
            raise ValueError(
                "leverage_eviction_risk_mode must be one of "
                f"{self.VALID_LEVERAGE_EVICTION_RISK_MODES}, got {leverage_eviction_risk_mode!r}"
            )
        if leverage_dpp_feature_projection not in self.VALID_LEVERAGE_DPP_FEATURE_PROJECTIONS:
            raise ValueError(
                "leverage_dpp_feature_projection must be one of "
                f"{self.VALID_LEVERAGE_DPP_FEATURE_PROJECTIONS}, got {leverage_dpp_feature_projection!r}"
            )
        if leverage_similarity_granularity not in self.VALID_LEVERAGE_SIMILARITY_GRANULARITIES:
            raise ValueError(
                "leverage_similarity_granularity must be one of "
                f"{self.VALID_LEVERAGE_SIMILARITY_GRANULARITIES}, got {leverage_similarity_granularity!r}"
            )
        if leverage_similarity_feature_projection not in self.VALID_LEVERAGE_SIMILARITY_FEATURE_PROJECTIONS:
            raise ValueError(
                "leverage_similarity_feature_projection must be one of "
                f"{self.VALID_LEVERAGE_SIMILARITY_FEATURE_PROJECTIONS}, "
                f"got {leverage_similarity_feature_projection!r}"
            )
        if leverage_similarity_leverage_gamma < 0:
            raise ValueError(
                "leverage_similarity_leverage_gamma must be >= 0, got "
                f"{leverage_similarity_leverage_gamma}"
            )
        if leverage_high_outlier_z < 0:
            raise ValueError(f"leverage_high_outlier_z must be >= 0, got {leverage_high_outlier_z}")
        if leverage_eviction_selector == "layer_head_fast_dpp" and (
            policy != "svd_leverage" or leverage_granularity != "layer"
        ):
            raise ValueError(
                "leverage_eviction_selector='layer_head_fast_dpp' requires "
                "policy='svd_leverage' and leverage_granularity='layer'"
            )
        if leverage_dpp_candidate_multiplier < 1:
            raise ValueError(
                "leverage_dpp_candidate_multiplier must be >= 1, got "
                f"{leverage_dpp_candidate_multiplier}"
            )
        if leverage_dpp_greedy_block_size < 1:
            raise ValueError(
                "leverage_dpp_greedy_block_size must be >= 1, got "
                f"{leverage_dpp_greedy_block_size}"
            )
        if leverage_dpp_quality_beta < 0:
            raise ValueError(
                "leverage_dpp_quality_beta must be >= 0, got "
                f"{leverage_dpp_quality_beta}"
            )
        if leverage_dpp_diversity_beta < 0:
            raise ValueError(
                "leverage_dpp_diversity_beta must be >= 0, got "
                f"{leverage_dpp_diversity_beta}"
            )
        if leverage_dpp_recency_lambda < 0:
            raise ValueError(
                "leverage_dpp_recency_lambda must be >= 0, got "
                f"{leverage_dpp_recency_lambda}"
            )
        if leverage_dpp_recency_window < 1:
            raise ValueError(
                "leverage_dpp_recency_window must be >= 1, got "
                f"{leverage_dpp_recency_window}"
            )
        if leverage_dpp_recency_gate_power < 0:
            raise ValueError(
                "leverage_dpp_recency_gate_power must be >= 0, got "
                f"{leverage_dpp_recency_gate_power}"
            )
        if not (0.0 <= float(leverage_conf_gate_floor) <= 1.0):
            raise ValueError(f"leverage_conf_gate_floor must be in [0, 1], got {leverage_conf_gate_floor}")
        if leverage_conf_gate_depth_alpha < 0:
            raise ValueError(
                "leverage_conf_gate_depth_alpha must be >= 0, got "
                f"{leverage_conf_gate_depth_alpha}"
            )
        if leverage_conf_gate_point_beta < 0:
            raise ValueError(
                "leverage_conf_gate_point_beta must be >= 0, got "
                f"{leverage_conf_gate_point_beta}"
            )
        if layer_budget_strategy not in VALID_LAYER_BUDGET_STRATEGIES:
            raise ValueError(
                "layer_budget_strategy must be one of "
                f"{VALID_LAYER_BUDGET_STRATEGIES}, got {layer_budget_strategy!r}"
            )
        if layer_budget_strategy in LAYER_BUDGET_SCORE_STRATEGIES and (
            policy != "svd_leverage" or leverage_granularity != "layer"
        ):
            raise ValueError(
                "leverage-based layer_budget_strategy requires "
                "eviction_policy='svd_leverage' and leverage_granularity='layer'"
            )
        if layer_budget_value_gamma < 0:
            raise ValueError(f"layer_budget_value_gamma must be >= 0, got {layer_budget_value_gamma}")
        if layer_budget_value_norm_type not in ("mean", "rms"):
            raise ValueError(
                "layer_budget_value_norm_type must be 'mean' or 'rms', got "
                f"{layer_budget_value_norm_type!r}"
            )
        if layer_budget_norm_source not in ("value", "key"):
            raise ValueError(
                "layer_budget_norm_source must be 'value' or 'key', got "
                f"{layer_budget_norm_source!r}"
            )
        if not math.isfinite(float(slots_per_direction)) or float(slots_per_direction) <= 0.0:
            raise ValueError(f"slots_per_direction must be finite and > 0, got {slots_per_direction}")
        if not math.isfinite(float(hybrid_beta)) or not (0.0 <= float(hybrid_beta) <= 1.0):
            raise ValueError(f"hybrid_beta must be finite and in [0, 1], got {hybrid_beta}")
        if leverage_ridge_lambda < 0:
            raise ValueError(f"leverage_ridge_lambda must be >= 0, got {leverage_ridge_lambda}")
        if leverage_ridge_lambda_mode not in ("relative", "absolute"):
            raise ValueError(
                "leverage_ridge_lambda_mode must be 'relative' or 'absolute', got "
                f"{leverage_ridge_lambda_mode!r}"
            )
        if leverage_ridge_score_chunk_size < 1:
            raise ValueError(
                "leverage_ridge_score_chunk_size must be >= 1, got "
                f"{leverage_ridge_score_chunk_size}"
            )
        # if leverage_ridge_jitter <= 0:
        #     raise ValueError(f"leverage_ridge_jitter must be > 0, got {leverage_ridge_jitter}")
        if leverage_ridge_dim is not None and leverage_ridge_dim < 1:
            raise ValueError(f"leverage_ridge_dim must be >= 1 or None, got {leverage_ridge_dim}")
        if rls_refresh_interval <= 0:
            raise ValueError(f"rls_refresh_interval must be >= 1, got {rls_refresh_interval}")
        if leverage_diag_interval < 0:
            raise ValueError(f"leverage_diag_interval must be >= 0, got {leverage_diag_interval}")
        if leverage_approx_method == "right_sketch_ridge" and leverage_ridge_dim is None:
            raise ValueError("right_sketch_ridge requires leverage_ridge_dim >= 1")
        if leverage_dpp_feature_projection == "random" and leverage_ridge_dim is None:
            raise ValueError("leverage_dpp_feature_projection='random' requires leverage_ridge_dim >= 1")
        if (
            leverage_similarity_feature_projection == "random"
            and leverage_approx_method not in ("right_sketch", "right_sketch_ridge")
        ):
            raise ValueError(
                "leverage_similarity_feature_projection='random' reuses projected leverage features "
                "and requires leverage_approx_method='right_sketch' or 'right_sketch_ridge'"
            )
        if leverage_normalize_before_projection:
            if policy != "svd_leverage":
                raise ValueError("leverage_normalize_before_projection requires policy='svd_leverage'")
            if leverage_granularity != "layer":
                raise ValueError("leverage_normalize_before_projection requires leverage_granularity='layer'")
            if leverage_feature != "key":
                raise ValueError("leverage_normalize_before_projection requires leverage_feature='key'")
            if leverage_projection != "random":
                raise ValueError("leverage_normalize_before_projection requires leverage_projection='random'")
            if leverage_approx_method not in ("exact_qr", "right_sketch", "right_sketch_ridge"):
                raise ValueError(
                    "leverage_normalize_before_projection requires "
                    "leverage_approx_method='exact_qr', 'right_sketch', or 'right_sketch_ridge'"
                )
        if leverage_normalize_before_projection_headwise and not leverage_normalize_before_projection:
            raise ValueError(
                "leverage_normalize_before_projection_headwise requires "
                "leverage_normalize_before_projection=True"
            )

        if leverage_projected_key_cache:
            if policy != "svd_leverage":
                raise ValueError("leverage_projected_key_cache requires policy='svd_leverage'")
            if leverage_granularity != "layer":
                raise ValueError("leverage_projected_key_cache requires leverage_granularity='layer'")
            if leverage_feature != "key":
                raise ValueError("leverage_projected_key_cache requires leverage_feature='key'")
            if leverage_projection != "random":
                raise ValueError("leverage_projected_key_cache requires leverage_projection='random'")
            if leverage_approx_method not in ("right_sketch", "right_sketch_ridge"):
                raise ValueError(
                    "leverage_projected_key_cache requires "
                    "leverage_approx_method='right_sketch' or 'right_sketch_ridge'"
                )
            if leverage_eviction_selector == "layer_head_fast_dpp" or (
                leverage_eviction_selector == "similarity_topk"
                and leverage_similarity_granularity == "head"
            ):
                raise ValueError(
                    "leverage_projected_key_cache only supports shared layer keep-set selectors"
                )
        self.policy = policy
        self.profile = bool(profile)
        self.debug = bool(debug)
        self.leverage_sketch_dim = leverage_sketch_dim
        self.leverage_granularity = leverage_granularity
        self.leverage_feature = leverage_feature
        self.leverage_projection = leverage_projection
        self.leverage_head_mean_dim = int(leverage_head_mean_dim)
        self.leverage_normalize_rows = bool(leverage_normalize_rows)
        self.leverage_normalize_before_projection = bool(leverage_normalize_before_projection)
        self.leverage_normalize_before_projection_headwise = bool(leverage_normalize_before_projection_headwise)
        self.leverage_projected_key_cache = bool(leverage_projected_key_cache)
        self.leverage_approx_method = leverage_approx_method
        self.leverage_ridge_lambda = float(leverage_ridge_lambda)
        self.leverage_ridge_lambda_mode = leverage_ridge_lambda_mode
        self.leverage_ridge_score_chunk_size = int(leverage_ridge_score_chunk_size)
        self.leverage_ridge_jitter = float(leverage_ridge_jitter)
        self.leverage_ridge_dim = leverage_ridge_dim
        self.rls_refresh_interval = int(rls_refresh_interval)
        self.leverage_diag = bool(leverage_diag)
        self.leverage_diag_interval = int(leverage_diag_interval)
        self.leverage_random_seed = int(leverage_random_seed)
        self.leverage_eviction_selector = leverage_eviction_selector
        self.leverage_similarity_granularity = leverage_similarity_granularity
        self.leverage_similarity_feature_projection = leverage_similarity_feature_projection
        self.leverage_similarity_leverage_gamma = float(leverage_similarity_leverage_gamma)
        self.leverage_eviction_risk_mode = leverage_eviction_risk_mode
        self.leverage_high_outlier_z = float(leverage_high_outlier_z)
        self.leverage_dpp_candidate_multiplier = int(leverage_dpp_candidate_multiplier)
        self.leverage_dpp_greedy_block_size = int(leverage_dpp_greedy_block_size)
        self.leverage_dpp_quality_beta = float(leverage_dpp_quality_beta)
        self.leverage_dpp_diversity_beta = float(leverage_dpp_diversity_beta)
        self.leverage_dpp_feature_projection = leverage_dpp_feature_projection
        self.leverage_dpp_recency_bonus = bool(leverage_dpp_recency_bonus)
        self.leverage_dpp_recency_lambda = float(leverage_dpp_recency_lambda)
        self.leverage_dpp_recency_window = int(leverage_dpp_recency_window)
        self.leverage_dpp_recency_gate_power = float(leverage_dpp_recency_gate_power)
        self.leverage_dpp_recency_debug = bool(leverage_dpp_recency_debug)
        self.leverage_conf_gate = bool(leverage_conf_gate)
        self.leverage_conf_gate_floor = float(leverage_conf_gate_floor)
        self.leverage_conf_gate_depth_alpha = float(leverage_conf_gate_depth_alpha)
        self.leverage_conf_gate_point_beta = float(leverage_conf_gate_point_beta)
        self.layer_budget_strategy = layer_budget_strategy
        self.layer_budget_value_gamma = float(layer_budget_value_gamma)
        self.layer_budget_value_norm_type = layer_budget_value_norm_type
        self.layer_budget_norm_source = layer_budget_norm_source
        self.layer_budget_eps = float(layer_budget_eps)
        self.slots_per_direction = float(slots_per_direction)
        self.hybrid_beta = float(hybrid_beta)
        self.leverage_dpp_full_sim_max_elements = 4_000_000
        self._leverage_right_sketch_cache = {}
        self._leverage_left_srht_cache = {}
        self._last_leverage_profile: Dict[str, float] = {}
        self._profile_totals: Dict[str, float] = {}
        self._profile_count = 0
        self._last_layer_feature_shape: Optional[tuple[int, int]] = None
        self._last_layer_covariance_pr: Optional[torch.Tensor] = None
        self._last_dpp_recency_debug: Dict[str, float] = {}
        self._last_similarity_layer_features: Optional[torch.Tensor] = None
        self._last_similarity_head_features: Optional[torch.Tensor] = None
        self._leverage_diag_context: Dict[str, object] = {}
        self._leverage_diag_emitted_once = False
        self._projected_key_cache: Optional[torch.Tensor] = None
        self._projected_key_head_cache: Optional[torch.Tensor] = None
        self._projected_key_pre_norm_cache: Optional[torch.Tensor] = None
        self._projected_key_cache_meta: Optional[Dict[str, object]] = None
        self._last_projected_key_features: Optional[torch.Tensor] = None
        self._last_projected_key_head_features: Optional[torch.Tensor] = None
        self._last_projected_key_pre_norms: Optional[torch.Tensor] = None
        self._last_projected_key_cache_meta: Optional[Dict[str, object]] = None
        self.reset_rls_cache()

    def reset_profile_stats(self) -> None:
        self._profile_totals = {}
        self._profile_count = 0

    def reset_rls_cache(self) -> None:
        self.cached_ktk: Optional[torch.Tensor] = None
        self.cached_rls_chol: Optional[torch.Tensor] = None
        self.cached_rls_inv: Optional[torch.Tensor] = None
        self.cached_rls_lam: Optional[torch.Tensor] = None
        self.cached_rls_meta: Optional[Dict[str, object]] = None
        self.last_rls_refresh_frame: Optional[int] = None
        self.rls_refresh_count = 0
        self.rls_cache_hit_count = 0

    def _record_profile_event(self, profile: Dict[str, float]) -> None:
        if not profile:
            return
        self._profile_count += 1
        for name, value in profile.items():
            if isinstance(value, (int, float)):
                self._profile_totals[name] = self._profile_totals.get(name, 0.0) + float(value)

    def get_profile_stats(self) -> Dict[str, object]:
        return {"count": self._profile_count, "totals": dict(self._profile_totals)}

    def select(
        self,
        k: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        *,
        v: Optional[torch.Tensor] = None,
        need_summary: bool = False,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        current_frame_idx: Optional[int] = None,
        protect_recent_frames: int = 0,
        candidate_frame_ids: Optional[torch.Tensor] = None,
        candidate_token_indices: Optional[torch.Tensor] = None,
        candidate_depth_confidence: Optional[torch.Tensor] = None,
        candidate_point_confidence: Optional[torch.Tensor] = None,
        candidate_conf_gate: Optional[torch.Tensor] = None,
        candidate_evictable_mask: Optional[torch.Tensor] = None,
        need_leverage_basis: bool = False,
        capture_projected_norms: bool = False,
        history_anchor_frame_ids=None,
        history_anchor_patch_topk_per_frame: int = 0,
        history_anchor_max_frames: int = 0,
        special_token_count: int = 0,
    ) -> EvictionResult:
        """Select candidate-local indices to retain.

        Args:
            k: Key cache shaped ``[B, H, N, D]``.
            cache_budget: Final number of tokens to retain.
            num_anchor_tokens: Initial tokens that are always preserved.

        Returns:
            EvictionResult with candidate-local kept indices shaped
            ``[B, H, K]``. ``K`` may exceed ``cache_budget - num_anchor_tokens``
            when token protection leaves too few evictable candidates.
        """
        B, H, N, D = k.shape
        if protect_recent_frames < 0:
            raise ValueError(f"protect_recent_frames must be >= 0, got {protect_recent_frames}")
        num_candidates = N - num_anchor_tokens
        num_to_keep = cache_budget - num_anchor_tokens
        if num_to_keep < 0:
            raise ValueError(
                f"cache_budget ({cache_budget}) must be >= num_anchor_tokens ({num_anchor_tokens})"
            )
        if num_to_keep > num_candidates:
            raise ValueError(f"Cannot keep {num_to_keep} candidates from {num_candidates} candidates")

        candidate_k = k[:, :, num_anchor_tokens:, :]
        candidate_v = v[:, :, num_anchor_tokens:, :] if v is not None else None
        self._set_leverage_diag_context(
            layer_id=layer_id,
            step_idx=step_idx,
            current_frame_idx=current_frame_idx,
            granularity=self.leverage_granularity,
            batch_size=B,
            num_heads=H,
        )
        self._last_leverage_profile = {}
        self._last_layer_feature_shape = None
        self._last_dpp_recency_debug = {}
        self._last_similarity_layer_features = None
        self._last_similarity_head_features = None
        self._last_projected_pre_norms = None
        self._last_projected_key_features = None
        self._last_projected_key_head_features = None
        self._last_projected_key_pre_norms = None
        self._last_projected_key_cache_meta = None
        self._capture_projected_norms = bool(capture_projected_norms)
        need_mean_scores = self.policy in ("mean", "baseline_mean") or need_summary or self.debug
        mean_scores = self._mean_scores(candidate_k) if need_mean_scores else None

        leverage_basis = None
        layer_budget_score = None
        raw_policy_scores = None
        if self.policy in ("mean", "baseline_mean"):
            policy_scores = mean_scores
            if protect_recent_frames > 0 or candidate_evictable_mask is not None:
                kept, protection_debug = self._keep_with_recent_protection(
                    policy_scores,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
                    candidate_evictable_mask,
                    evict_highest=True,
                    shared_across_heads=False,
                )
            else:
                kept = self._keep_lowest_scores(policy_scores, num_to_keep)
                protection_debug = None
        elif self.policy == "svd_leverage":
            if self.leverage_granularity == "head":
                if need_leverage_basis:
                    policy_scores, leverage_basis = self._svd_leverage_scores(candidate_k, return_basis=True)
                else:
                    policy_scores = self._svd_leverage_scores(candidate_k)
                raw_policy_scores = policy_scores
                policy_scores = self._apply_confidence_gate(
                    policy_scores,
                    candidate_depth_confidence,
                    candidate_point_confidence,
                    candidate_conf_gate,
                    candidate_frame_ids,
                    current_frame_idx,
                    shared_across_heads=False,
                )
                policy_scores = self._apply_recency_score_bonus(
                    policy_scores,
                    candidate_frame_ids,
                    current_frame_idx,
                    shared_across_heads=False,
                )
                candidate_evictable_mask = self._combine_history_anchor_patch_topk_mask(
                    raw_policy_scores,
                    candidate_frame_ids,
                    candidate_token_indices,
                    candidate_evictable_mask,
                    history_anchor_frame_ids,
                    history_anchor_patch_topk_per_frame,
                    history_anchor_max_frames,
                    special_token_count,
                )
                kept, protection_debug = self._select_svd_leverage_kept(
                    policy_scores,
                    candidate_k,
                    candidate_v,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
                    candidate_evictable_mask,
                    shared_across_heads=False,
                    raw_outlier_scores=raw_policy_scores,
                )
            else:
                if need_leverage_basis:
                    policy_scores, leverage_basis = self._layer_svd_leverage_scores(
                        candidate_k,
                        candidate_v,
                        return_basis=True,
                    )
                else:
                    policy_scores = self._layer_svd_leverage_scores(candidate_k, candidate_v)
                raw_policy_scores = policy_scores
                layer_budget_score = self._compute_layer_budget_score(raw_policy_scores, candidate_k, candidate_v)
                candidate_evictable_mask = self._combine_history_anchor_patch_topk_mask(
                    raw_policy_scores,
                    candidate_frame_ids,
                    candidate_token_indices,
                    candidate_evictable_mask,
                    history_anchor_frame_ids,
                    history_anchor_patch_topk_per_frame,
                    history_anchor_max_frames,
                    special_token_count,
                )
                policy_scores = self._apply_confidence_gate(
                    policy_scores,
                    candidate_depth_confidence,
                    candidate_point_confidence,
                    candidate_conf_gate,
                    candidate_frame_ids,
                    current_frame_idx,
                    shared_across_heads=True,
                )
                policy_scores = self._apply_recency_score_bonus(
                    policy_scores,
                    candidate_frame_ids,
                    current_frame_idx,
                    shared_across_heads=True,
                )
                if (
                    self.leverage_eviction_selector == "layer_head_fast_dpp"
                    and candidate_evictable_mask is not None
                    and candidate_evictable_mask.ndim == 2
                ):
                    candidate_evictable_mask = candidate_evictable_mask.unsqueeze(1).expand(B, H, num_candidates)
                if self.leverage_eviction_selector == "layer_head_fast_dpp":
                    kept, protection_debug = self._select_layer_scores_head_dpp_kept(
                        policy_scores,
                        candidate_k,
                        candidate_v,
                        num_to_keep,
                        candidate_frame_ids,
                        current_frame_idx,
                        protect_recent_frames,
                        candidate_evictable_mask,
                        raw_outlier_scores=raw_policy_scores,
                    )
                else:
                    kept, protection_debug = self._select_svd_leverage_kept(
                        policy_scores,
                        candidate_k,
                        candidate_v,
                        num_to_keep,
                        candidate_frame_ids,
                        current_frame_idx,
                        protect_recent_frames,
                        candidate_evictable_mask,
                        shared_across_heads=True,
                        num_heads=H,
                        raw_outlier_scores=raw_policy_scores,
                    )
        elif self.policy == "dpp":
            if self.leverage_granularity == "head":
                policy_scores = torch.zeros(B, H, num_candidates, device=k.device, dtype=torch.float32)
                kept, protection_debug = self._select_svd_leverage_kept(
                    policy_scores,
                    candidate_k,
                    candidate_v,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
                    candidate_evictable_mask,
                    shared_across_heads=False,
                    use_full_dpp_pool=True,
                )
            else:
                policy_scores = torch.zeros(B, num_candidates, device=k.device, dtype=torch.float32)
                kept, protection_debug = self._select_svd_leverage_kept(
                    policy_scores,
                    candidate_k,
                    candidate_v,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
                    candidate_evictable_mask,
                    shared_across_heads=True,
                    num_heads=H,
                    use_full_dpp_pool=True,
                )
        else:
            raise AssertionError(f"Unhandled eviction policy: {self.policy}")

        summary_score = mean_scores.mean().item() if need_summary and mean_scores is not None else 0.0
        if self.debug:
            sketch_label = (
                "exact"
                if self.leverage_sketch_dim in (None, 0)
                else str(self.leverage_sketch_dim)
            )
            requested_evicted = num_candidates - num_to_keep
            actual_evicted = num_candidates - int(kept.shape[-1])
            feature_dim = D
            if self.policy in ("svd_leverage", "dpp") and self.leverage_granularity == "layer":
                if self.leverage_projection == "head_mean":
                    feature_dim = H * self.leverage_head_mean_dim
                else:
                    if self.leverage_feature == "key_value_lowdim_concat" and self._last_layer_feature_shape is not None:
                        feature_dim = self._last_layer_feature_shape[-1]
                    else:
                        feature_dim = H * D * (2 if self.leverage_feature == "key_value" else 1)
            msg = (
                f"[EvictionManager] policy={self.policy} layer={layer_id} step={step_idx} "
                f"cache={N} budget={cache_budget} keep_candidates={kept.shape[-1]} "
                f"requested_evicted={requested_evicted} evicted={actual_evicted} "
                f"scores={tuple(policy_scores.shape)}"
            )
            if protection_debug is not None:
                msg += (
                    f" current_frame_idx={protection_debug['current_frame_idx']} "
                    f"protect_recent_frames={protection_debug['protect_recent_frames']} "
                    f"protected_special_tokens={protection_debug['protected_special_tokens']} "
                    f"protected_tokens={protection_debug['protected_tokens']} "
                    f"candidate_tokens={protection_debug['candidate_tokens']} "
                    f"limited_by_protection={protection_debug['limited_by_protection']}"
                )
            if self.policy in ("svd_leverage", "dpp"):
                msg += (
                    f" layer_budget_strategy={self.layer_budget_strategy} "
                    f"leverage_approx_method={self.leverage_approx_method} "
                    f"leverage_normalize_before_projection={self.leverage_normalize_before_projection} "
                    f"leverage_normalize_before_projection_headwise={self.leverage_normalize_before_projection_headwise} "
                    f" leverage_sketch_dim={sketch_label} "
                    f"leverage_ridge_lambda={self.leverage_ridge_lambda} "
                    f"leverage_ridge_lambda_mode={self.leverage_ridge_lambda_mode} "
                    f"leverage_ridge_score_chunk_size={self.leverage_ridge_score_chunk_size} "
                    f"leverage_ridge_jitter={self.leverage_ridge_jitter} "
                    f"leverage_ridge_dim={self.leverage_ridge_dim} "
                    f"rls_refresh_interval={self.rls_refresh_interval} "
                    f"last_rls_refresh_frame={self.last_rls_refresh_frame} "
                    f"rls_refresh_count={self.rls_refresh_count} "
                    f"rls_cache_hit_count={self.rls_cache_hit_count} "
                    f"leverage_random_seed={self.leverage_random_seed} "
                    f"leverage_granularity={self.leverage_granularity} leverage_feature={self.leverage_feature} "
                    f"leverage_projection={self.leverage_projection} "
                    f"leverage_head_mean_dim={self.leverage_head_mean_dim} "
                    f"leverage_normalize_rows={self.leverage_normalize_rows} "
                    f"leverage_projected_key_cache={self.leverage_projected_key_cache} "
                    f"leverage_eviction_selector={self.leverage_eviction_selector} "
                    f"leverage_similarity_granularity={self.leverage_similarity_granularity} "
                    f"leverage_similarity_feature_projection={self.leverage_similarity_feature_projection} "
                    f"leverage_similarity_leverage_gamma={self.leverage_similarity_leverage_gamma} "
                    f"leverage_dpp_candidate_multiplier={self.leverage_dpp_candidate_multiplier} "
                    f"leverage_dpp_greedy_block_size={self.leverage_dpp_greedy_block_size} "
                    f"leverage_dpp_quality_beta={self.leverage_dpp_quality_beta} "
                    f"leverage_dpp_diversity_beta={self.leverage_dpp_diversity_beta} "
                    f"leverage_dpp_feature_projection={self.leverage_dpp_feature_projection} "
                    f"leverage_dpp_recency_bonus={self.leverage_dpp_recency_bonus} "
                    f"leverage_dpp_recency_lambda={self.leverage_dpp_recency_lambda} "
                    f"leverage_dpp_recency_window={self.leverage_dpp_recency_window} "
                    f"leverage_dpp_recency_gate_power={self.leverage_dpp_recency_gate_power} "
                    f"leverage_conf_gate={self.leverage_conf_gate} "
                    f"leverage_conf_gate_floor={self.leverage_conf_gate_floor} "
                    f"leverage_conf_gate_depth_alpha={self.leverage_conf_gate_depth_alpha} "
                    f"leverage_conf_gate_point_beta={self.leverage_conf_gate_point_beta} "
                    f"layer_budget_value_gamma={self.layer_budget_value_gamma} "
                    f"layer_budget_value_norm_type={self.layer_budget_value_norm_type} "
                    f"layer_budget_norm_source={self.layer_budget_norm_source} "
                    f"slots_per_direction={self.slots_per_direction} "
                    f"hybrid_beta={self.hybrid_beta} "
                    f"num_heads={H} num_tokens={num_candidates} head_dim={D} feature_dim={feature_dim}"
                )
                if self.policy == "dpp":
                    msg += " dpp_pool=full"
                elif self.leverage_eviction_selector == "layer_head_fast_dpp":
                    msg += (
                        " dpp_selection_granularity=head "
                        "layer_feature_alignment=approximate_after_head_specific_selection"
                    )
            print(msg)
            if protection_debug is not None and protection_debug["limited_by_protection"]:
                print(
                    "[EvictionManager] token protection limited eviction; "
                    "cache may temporarily exceed budget"
                )
            if self.policy == "svd_leverage" and self.leverage_granularity == "layer":
                print(f"[EvictionManager] layer-wise SVD leverage: X shape={self._last_layer_feature_shape}")
            elif self.policy == "dpp" and self.leverage_granularity == "layer":
                print(f"[EvictionManager] layer-wise DPP features: X shape={self._last_layer_feature_shape}")
            if self.policy == "svd_leverage" and self._last_leverage_profile:
                if "scoring" in self._last_leverage_profile and "score_calc" not in self._last_leverage_profile:
                    self._last_leverage_profile["score_calc"] = self._last_leverage_profile["scoring"]
                profile_items = []
                time_fields = {
                    "candidate_matrix_preparation",
                    "feature",
                    "sketch",
                    "sketch_matrix_retrieval",
                    "projection_matmul",
                    "normalization",
                    "pre_projection_normalization",
                    "post_projection_normalization",
                    "pre_score_normalization",
                    "qr",
                    "small_qr",
                    "left_sketch",
                    "right_jl_solve",
                    "omega_gemm",
                    "gram",
                    "cholesky",
                    "inverse_build",
                    "score_solve",
                    "scoring",
                    "score_calc",
                    "total",
                }
                int_fields = {"fallback", "N", "D", "sketch_dim", "cholesky_retries", "rls_refresh_interval", "rls_frame_idx", "rls_cache_refreshed", "rls_refresh_frame", "rls_refresh_count", "rls_cache_hit_count"}
                for name, value in self._last_leverage_profile.items():
                    if isinstance(value, str):
                        profile_items.append(f"{name}={value}")
                    elif name in int_fields:
                        profile_items.append(f"{name}={int(value)}")
                    elif name in time_fields:
                        profile_items.append(f"{name}={value * 1000.0:.3f}ms")
                    else:
                        profile_items.append(f"{name}={float(value):.6g}")
                profile = " ".join(profile_items)
                self._record_profile_event(self._last_leverage_profile)
        if self.profile and not self.debug and self.policy == "svd_leverage" and self._last_leverage_profile:
            if "scoring" in self._last_leverage_profile and "score_calc" not in self._last_leverage_profile:
                self._last_leverage_profile["score_calc"] = self._last_leverage_profile["scoring"]
            profile_items = []
            time_fields = {
                "candidate_matrix_preparation",
                "feature",
                "sketch",
                "sketch_matrix_retrieval",
                "projection_matmul",
                "normalization",
                "pre_projection_normalization",
                "post_projection_normalization",
                "pre_score_normalization",
                "qr",
                "small_qr",
                "left_sketch",
                "right_jl_solve",
                "omega_gemm",
                "gram",
                "cholesky",
                "inverse_build",
                "score_solve",
                "scoring",
                "score_calc",
                "total",
            }
            int_fields = {"fallback", "N", "D", "sketch_dim", "cholesky_retries", "rls_refresh_interval", "rls_frame_idx", "rls_cache_refreshed", "rls_refresh_frame", "rls_refresh_count", "rls_cache_hit_count"}
            for name, value in self._last_leverage_profile.items():
                if isinstance(value, str):
                    profile_items.append(f"{name}={value}")
                elif name in int_fields:
                    profile_items.append(f"{name}={int(value)}")
                elif name in time_fields:
                    profile_items.append(f"{name}={value * 1000.0:.3f}ms")
                else:
                    profile_items.append(f"{name}={float(value):.6g}")
            profile = " ".join(profile_items)
            self._record_profile_event(self._last_leverage_profile)
        if self.leverage_dpp_recency_debug and self._last_dpp_recency_debug.get("count", 0.0) > 0.0:
            print(
                self._format_dpp_recency_debug(
                    self._last_dpp_recency_debug,
                    self.leverage_dpp_quality_beta,
                    self.leverage_dpp_recency_lambda,
                    self.leverage_dpp_recency_window,
                    self.leverage_dpp_recency_gate_power,
                )
            )
        return EvictionResult(
            kept_candidate_indices=kept,
            policy_scores=policy_scores,
            mean_scores=mean_scores
            if mean_scores is not None
            else torch.empty(B, H, 0, device=k.device, dtype=torch.float32),
            summary_score=summary_score,
            leverage_basis=leverage_basis,
            layer_budget_score=layer_budget_score,
        )

    def _compute_layer_budget_score(
        self,
        policy_scores: torch.Tensor,
        candidate_k: Optional[torch.Tensor] = None,
        candidate_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.layer_budget_strategy in ("uniform", "cosine_precomputed"):
            return None
        with torch.no_grad():
            if policy_scores.ndim == 1:
                score_input = policy_scores.unsqueeze(0)
            elif policy_scores.ndim == 2:
                score_input = policy_scores
            else:
                raise ValueError(
                    "Layer budget scores require layer-wise leverage scores shaped [B, N], "
                    f"got {tuple(policy_scores.shape)}"
                )
            layer_scores = []
            layer_value_norms = []
            covariance_pr = self._last_layer_covariance_pr
            if covariance_pr is not None:
                covariance_pr = torch.as_tensor(covariance_pr, device=policy_scores.device).detach().float().reshape(-1)
            for batch_idx, batch_scores in enumerate(score_input):
                if self.layer_budget_strategy == "leverage_entropy":
                    layer_scores.append(_layer_score_leverage_entropy(batch_scores, self.layer_budget_eps))
                elif self.layer_budget_strategy == "key_norm":
                    if candidate_k is None:
                        layer_scores.append(
                            torch.zeros((), device=policy_scores.device, dtype=torch.float32)
                        )
                    else:
                        if candidate_k.ndim != 4:
                            raise ValueError(
                                "Layer budget key_norm requires candidate_k shaped [B, H, N, D], "
                                f"got {tuple(candidate_k.shape)}"
                            )
                        _, num_heads, num_tokens, head_dim = candidate_k.shape
                        key_rows = (
                            candidate_k[batch_idx]
                            .transpose(0, 1)
                            .reshape(num_tokens, num_heads * head_dim)
                        )
                        layer_scores.append(
                            _layer_value_norm_score(
                                key_rows,
                                norm_type="mean",
                                eps=self.layer_budget_eps,
                            )
                        )
                elif self.layer_budget_strategy in LAYER_BUDGET_SCORE_STRATEGIES:
                    leverage_pr = _layer_score_leverage_pr(batch_scores, self.layer_budget_eps)
                    batch_covariance_pr = None
                    if covariance_pr is not None and covariance_pr.numel() > 0:
                        batch_covariance_pr = covariance_pr[min(batch_idx, covariance_pr.numel() - 1)]
                    layer_scores.append(
                        _combine_layer_budget_pr_base(
                            leverage_pr,
                            batch_covariance_pr,
                            self.layer_budget_strategy,
                            self.slots_per_direction,
                            self.hybrid_beta,
                            self.layer_budget_eps,
                        )
                    )
                    if self.layer_budget_strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                        norm_source = candidate_v if self.layer_budget_norm_source == "value" else candidate_k
                        if norm_source is None:
                            layer_value_norms.append(
                                torch.zeros((), device=policy_scores.device, dtype=torch.float32)
                            )
                        else:
                            if norm_source.ndim != 4:
                                raise ValueError(
                                    "Layer budget norm source requires tensor shaped [B, H, N, D], "
                                    f"got {tuple(norm_source.shape)}"
                                )
                            _, num_heads, num_tokens, head_dim = norm_source.shape
                            norm_rows = (
                                norm_source[batch_idx]
                                .transpose(0, 1)
                                .reshape(num_tokens, num_heads * head_dim)
                            )
                            layer_value_norms.append(
                                _layer_value_norm_score(
                                    norm_rows,
                                    norm_type=self.layer_budget_value_norm_type,
                                    eps=self.layer_budget_eps,
                                )
                            )
                else:
                    raise AssertionError(f"Unhandled layer budget strategy: {self.layer_budget_strategy}")
            base_score = torch.stack(layer_scores).mean()
            if self.layer_budget_strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                value_norm = torch.stack(layer_value_norms).mean()
                return torch.stack([base_score, value_norm]).detach()
            return base_score.detach()

    @staticmethod
    def _mean_scores(candidate_k: torch.Tensor) -> torch.Tensor:
        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)
        return torch.sum(candidate_k_norm * mean_vector, dim=-1)

    @staticmethod
    def _keep_lowest_scores(scores: torch.Tensor, num_to_keep: int) -> torch.Tensor:
        _, kept = torch.topk(-scores, k=num_to_keep, dim=-1)
        return kept.sort(dim=-1).values

    @staticmethod
    def _robust_high_outlier_z_scores(scores: torch.Tensor) -> torch.Tensor:
        clean = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        clean = torch.clamp(clean, min=0.0)
        center = clean.median(dim=-1, keepdim=True).values
        abs_dev = (clean - center).abs()
        mad = abs_dev.median(dim=-1, keepdim=True).values
        scale = 1.4826 * mad
        std = clean.std(dim=-1, keepdim=True, unbiased=False)
        scale = torch.where(scale > 1e-12, scale, std)
        valid_scale = torch.isfinite(scale) & (scale > 1e-12)
        z_scores = torch.zeros_like(clean)
        return torch.where(valid_scale, (clean - center) / scale.clamp_min(1e-12), z_scores)

    @staticmethod
    def _low_score_evicted_indices(
        row_scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
    ) -> torch.Tensor:
        if num_to_evict <= 0:
            return torch.empty(0, device=row_scores.device, dtype=torch.long)
        all_indices = torch.arange(row_scores.shape[-1], device=row_scores.device, dtype=torch.long)
        evictable = all_indices[evictable_mask]
        if evictable.numel() < num_to_evict:
            raise ValueError(
                f"Cannot evict {num_to_evict} tokens from only {int(evictable.numel())} evictable candidates"
            )
        clean_scores = torch.nan_to_num(row_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        order = torch.argsort(clean_scores.index_select(0, evictable), stable=True)
        return evictable.index_select(0, order[: int(num_to_evict)])

    def _outlier_then_low_evicted_indices(
        self,
        row_scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        low_eviction_fn,
        *,
        outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_to_evict <= 0:
            return torch.empty(0, device=row_scores.device, dtype=torch.long)
        outlier_source = row_scores if outlier_scores is None else outlier_scores
        z_scores = self._robust_high_outlier_z_scores(outlier_source.unsqueeze(0)).squeeze(0)
        all_indices = torch.arange(row_scores.shape[-1], device=row_scores.device, dtype=torch.long)
        high_outlier_mask = evictable_mask & (z_scores > self.leverage_high_outlier_z)
        high_outliers = all_indices[high_outlier_mask]
        direct_count = min(int(num_to_evict), int(high_outliers.numel()))
        if direct_count > 0:
            high_order = torch.argsort(z_scores.index_select(0, high_outliers), descending=True, stable=True)
            direct_evicted = high_outliers.index_select(0, high_order[:direct_count])
        else:
            direct_evicted = torch.empty(0, device=row_scores.device, dtype=torch.long)
        remaining = int(num_to_evict) - int(direct_evicted.numel())
        if remaining <= 0:
            return direct_evicted
        low_evictable = evictable_mask.clone()
        if direct_evicted.numel() > 0:
            low_evictable[direct_evicted] = False
        low_evicted = low_eviction_fn(low_evictable, remaining)
        return torch.cat([direct_evicted, low_evicted], dim=0)

    def _keep_after_outlier_then_low_topk(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        leading_shape = scores.shape[:-1]
        num_candidates = int(scores.shape[-1])
        keep_count = num_candidates - int(num_to_evict)
        all_indices = torch.arange(num_candidates, device=scores.device, dtype=torch.long)
        if num_to_evict <= 0:
            return all_indices.expand(*leading_shape, num_candidates)
        flat_scores = scores.reshape(-1, num_candidates)
        flat_mask = evictable_mask.reshape(-1, num_candidates)
        flat_outlier_scores = (
            raw_outlier_scores.reshape(-1, num_candidates)
            if raw_outlier_scores is not None
            else None
        )
        kept_rows = []
        for row_idx in range(flat_scores.shape[0]):
            evicted = self._outlier_then_low_evicted_indices(
                flat_scores[row_idx],
                flat_mask[row_idx],
                int(num_to_evict),
                lambda low_mask, remaining, row_idx=row_idx: self._low_score_evicted_indices(
                    flat_scores[row_idx], low_mask, remaining
                ),
                outlier_scores=flat_outlier_scores[row_idx] if flat_outlier_scores is not None else None,
            )
            keep_mask = torch.ones(num_candidates, device=scores.device, dtype=torch.bool)
            keep_mask[evicted] = False
            kept_rows.append(all_indices[keep_mask].sort(dim=-1).values)
        return torch.stack(kept_rows, dim=0).reshape(*leading_shape, keep_count)

    def _select_svd_leverage_kept(
        self,
        scores: torch.Tensor,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        num_to_keep: int,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_idx: Optional[int],
        protect_recent_frames: int,
        candidate_evictable_mask: Optional[torch.Tensor],
        *,
        shared_across_heads: bool,
        num_heads: Optional[int] = None,
        use_full_dpp_pool: bool = False,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[Dict[str, int]]]:
        num_candidates = int(scores.shape[-1])
        requested_evict = num_candidates - int(num_to_keep)
        evictable_mask, protection_debug = self._build_evictable_mask(
            scores,
            candidate_frame_ids,
            current_frame_idx,
            protect_recent_frames,
            candidate_evictable_mask,
            shared_across_heads=shared_across_heads,
        )
        actual_evict = min(requested_evict, int(evictable_mask.sum(dim=-1).min().item()))
        if protection_debug is not None:
            protection_debug["requested_eviction_count"] = int(requested_evict)
            protection_debug["actual_eviction_count"] = int(actual_evict)
            protection_debug["limited_by_protection"] = int(actual_evict < requested_evict)

        if use_full_dpp_pool:
            if shared_across_heads:
                kept_2d = self._keep_after_layer_fast_dpp(
                    scores,
                    evictable_mask,
                    actual_evict,
                    candidate_k,
                    candidate_v,
                    use_full_pool=True,
                )
                H = int(num_heads) if num_heads is not None else int(candidate_k.shape[1])
                return kept_2d.unsqueeze(1).expand(scores.shape[0], H, kept_2d.shape[-1]), protection_debug
            return self._keep_after_head_fast_dpp(
                scores,
                evictable_mask,
                actual_evict,
                candidate_k,
                use_full_pool=True,
            ), protection_debug

        if self.leverage_eviction_selector == "topk":
            if shared_across_heads:
                if self.leverage_eviction_risk_mode == "outlier_then_low":
                    kept_2d = self._keep_after_outlier_then_low_topk(
                        scores, evictable_mask, actual_evict, raw_outlier_scores
                    )
                else:
                    kept_2d = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest=False)
                H = int(num_heads) if num_heads is not None else int(candidate_k.shape[1])
                return kept_2d.unsqueeze(1).expand(scores.shape[0], H, kept_2d.shape[-1]), protection_debug
            if self.leverage_eviction_risk_mode == "outlier_then_low":
                return self._keep_after_outlier_then_low_topk(
                    scores, evictable_mask, actual_evict, raw_outlier_scores
                ), protection_debug
            return self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest=False), protection_debug

        if self.leverage_eviction_selector == "fast_dpp":
            if shared_across_heads:
                kept_2d = self._keep_after_layer_fast_dpp(
                    scores,
                    evictable_mask,
                    actual_evict,
                    candidate_k,
                    candidate_v,
                    candidate_frame_ids=candidate_frame_ids,
                    current_frame_id=current_frame_idx,
                    outlier_then_low=self.leverage_eviction_risk_mode == "outlier_then_low",
                    raw_outlier_scores=raw_outlier_scores,
                )

                H = int(num_heads) if num_heads is not None else int(candidate_k.shape[1])
                return kept_2d.unsqueeze(1).expand(scores.shape[0], H, kept_2d.shape[-1]), protection_debug
            return self._keep_after_head_fast_dpp(
                scores,
                evictable_mask,
                actual_evict,
                candidate_k,
                candidate_frame_ids=candidate_frame_ids,
                current_frame_id=current_frame_idx,
                outlier_then_low=self.leverage_eviction_risk_mode == "outlier_then_low",
                raw_outlier_scores=raw_outlier_scores,
            ), protection_debug

        if self.leverage_eviction_selector == "similarity_topk":
            if shared_across_heads:
                if self.leverage_similarity_granularity == "head":
                    return self._keep_after_layer_head_similarity_topk(
                        scores,
                        evictable_mask,
                        actual_evict,
                        candidate_k,
                        outlier_then_low=self.leverage_eviction_risk_mode == "outlier_then_low",
                        raw_outlier_scores=raw_outlier_scores,
                    ), protection_debug
                kept_2d = self._keep_after_layer_similarity_topk(
                    scores,
                    evictable_mask,
                    actual_evict,
                    candidate_k,
                    candidate_v,
                    outlier_then_low=self.leverage_eviction_risk_mode == "outlier_then_low",
                    raw_outlier_scores=raw_outlier_scores,
                )
                H = int(num_heads) if num_heads is not None else int(candidate_k.shape[1])
                return kept_2d.unsqueeze(1).expand(scores.shape[0], H, kept_2d.shape[-1]), protection_debug
            return self._keep_after_head_similarity_topk(
                scores,
                evictable_mask,
                actual_evict,
                candidate_k,
                outlier_then_low=self.leverage_eviction_risk_mode == "outlier_then_low",
                raw_outlier_scores=raw_outlier_scores,
            ), protection_debug

        raise AssertionError(f"Unhandled leverage eviction selector: {self.leverage_eviction_selector}")

    def _select_layer_scores_head_dpp_kept(
        self,
        scores: torch.Tensor,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        num_to_keep: int,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_idx: Optional[int],
        protect_recent_frames: int,
        candidate_evictable_mask: Optional[torch.Tensor],
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[Dict[str, int]]]:
        """Apply shared layer-wise importance scores with head-wise DPP diversity."""
        B, H, N, _ = candidate_k.shape
        if scores.shape != (B, N):
            raise ValueError(
                "layer_head_fast_dpp requires layer-wise scores shaped [B, N], "
                f"got {tuple(scores.shape)} for candidate shape {(B, H, N)}"
            )
        head_scores = scores.unsqueeze(1).expand(B, H, N)
        evictable_mask, protection_debug = self._build_evictable_mask(
            head_scores,
            candidate_frame_ids,
            current_frame_idx,
            protect_recent_frames,
            candidate_evictable_mask,
            shared_across_heads=False,
        )
        requested_evict = N - int(num_to_keep)
        actual_evict = min(requested_evict, int(evictable_mask.sum(dim=-1).min().item()))
        if protection_debug is not None:
            protection_debug["requested_eviction_count"] = int(requested_evict)
            protection_debug["actual_eviction_count"] = int(actual_evict)
            protection_debug["limited_by_protection"] = int(actual_evict < requested_evict)
        if self.leverage_eviction_risk_mode == "outlier_then_low":
            kept = self._keep_after_layer_head_outlier_then_low_dpp(
                scores,
                evictable_mask,
                actual_evict,
                candidate_k,
                candidate_v,
                candidate_frame_ids=candidate_frame_ids,
                current_frame_id=current_frame_idx,
                raw_outlier_scores=raw_outlier_scores,
            )
        else:
            kept = self._keep_after_layer_head_fast_dpp(
                scores,
                evictable_mask,
                actual_evict,
                candidate_k,
                candidate_v,
                candidate_frame_ids=candidate_frame_ids,
                current_frame_id=current_frame_idx,
            )
        return kept, protection_debug

    def _apply_confidence_gate(
        self,
        scores: torch.Tensor,
        candidate_depth_confidence: Optional[torch.Tensor],
        candidate_point_confidence: Optional[torch.Tensor],
        candidate_conf_gate: Optional[torch.Tensor],
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_id: Optional[int],
        *,
        shared_across_heads: bool,
    ) -> torch.Tensor:
        if not self.leverage_conf_gate:
            return scores

        B = int(scores.shape[0])
        N = int(scores.shape[-1])
        if shared_across_heads:
            if scores.ndim != 2:
                raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
            if candidate_conf_gate is not None and candidate_conf_gate.ndim == 3:
                H = int(candidate_conf_gate.shape[1])
            elif candidate_depth_confidence is not None and candidate_depth_confidence.ndim == 3:
                H = int(candidate_depth_confidence.shape[1])
            elif candidate_point_confidence is not None and candidate_point_confidence.ndim == 3:
                H = int(candidate_point_confidence.shape[1])
            elif candidate_frame_ids is not None and candidate_frame_ids.ndim == 3:
                H = int(candidate_frame_ids.shape[1])
            else:
                H = 1
        else:
            if scores.ndim != 3:
                raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")
            H = int(scores.shape[1])

        if candidate_conf_gate is not None:
            gate = self._confidence_gate_tensor(
                candidate_conf_gate, B, H, N, scores.device, shared_across_heads=shared_across_heads
            )
            gate = torch.nan_to_num(gate.float(), nan=1.0, posinf=1.0, neginf=0.0)
        else:
            depth_conf = self._confidence_gate_tensor(
                candidate_depth_confidence, B, H, N, scores.device, shared_across_heads=shared_across_heads
            )
            point_conf = self._confidence_gate_tensor(
                candidate_point_confidence, B, H, N, scores.device, shared_across_heads=shared_across_heads
            )
            depth_conf = torch.nan_to_num(depth_conf.float(), nan=1.0, posinf=1.0, neginf=0.0)
            point_conf = torch.nan_to_num(point_conf.float(), nan=1.0, posinf=1.0, neginf=0.0)

            if self.leverage_conf_gate_depth_alpha != 1.0:
                depth_conf = depth_conf.pow(float(self.leverage_conf_gate_depth_alpha))
            if self.leverage_conf_gate_point_beta != 1.0:
                point_conf = point_conf.pow(float(self.leverage_conf_gate_point_beta))
            floor = float(self.leverage_conf_gate_floor)
            gate = floor + (1.0 - floor) * depth_conf * point_conf

        if candidate_frame_ids is not None and current_frame_id is not None:
            if shared_across_heads:
                frame_ids = self._layer_candidate_frame_ids(candidate_frame_ids, B, H, N, scores.device)
            else:
                frame_ids = self._expand_candidate_frame_ids(candidate_frame_ids, B, H, N, scores.device)
            gate = torch.where(frame_ids == int(current_frame_id), torch.ones_like(gate), gate)

        return scores * gate.to(device=scores.device, dtype=scores.dtype)

    @staticmethod
    def _confidence_gate_tensor(
        confidence: Optional[torch.Tensor],
        B: int,
        H: int,
        N: int,
        device: torch.device,
        *,
        shared_across_heads: bool,
    ) -> torch.Tensor:
        if confidence is None:
            shape = (B, N) if shared_across_heads else (B, H, N)
            return torch.ones(shape, device=device, dtype=torch.float32)
        confidence = confidence.to(device=device)
        if confidence.dim() == 2:
            if tuple(confidence.shape) != (B, N):
                raise ValueError(f"candidate confidence must have shape [B, N] or [B, H, N], got {tuple(confidence.shape)}")
            if shared_across_heads:
                return confidence
            return confidence.unsqueeze(1).expand(B, H, N)
        if confidence.dim() == 3:
            if tuple(confidence.shape) != (B, H, N):
                raise ValueError(f"candidate confidence must have shape [B, N] or [B, H, N], got {tuple(confidence.shape)}")
            if shared_across_heads:
                return confidence.float().mean(dim=1)
            return confidence
        raise ValueError(f"candidate confidence must have shape [B, N] or [B, H, N], got {tuple(confidence.shape)}")

    def _apply_recency_score_bonus(
        self,
        scores: torch.Tensor,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_id: Optional[int],
        *,
        shared_across_heads: bool,
    ) -> torch.Tensor:
        if (
            not self.leverage_dpp_recency_bonus
            or self.leverage_dpp_recency_lambda <= 0.0
            or candidate_frame_ids is None
            or current_frame_id is None
        ):
            return scores

        clean_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        score_min = clean_scores.min(dim=-1, keepdim=True).values
        score_range = (clean_scores.max(dim=-1, keepdim=True).values - score_min).clamp_min(1e-12)
        score01 = ((clean_scores - score_min) / score_range).clamp(0.0, 1.0)

        B = int(scores.shape[0])
        N = int(scores.shape[-1])
        if shared_across_heads:
            if scores.ndim != 2:
                raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
            H = int(candidate_frame_ids.shape[1]) if candidate_frame_ids.ndim == 3 else 1
            frame_ids = self._layer_candidate_frame_ids(candidate_frame_ids, B, H, N, scores.device)
        else:
            if scores.ndim != 3:
                raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")
            H = int(scores.shape[1])
            frame_ids = self._expand_candidate_frame_ids(candidate_frame_ids, B, H, N, scores.device)

        age = (int(current_frame_id) - frame_ids).float().clamp_min(0.0)
        window = max(1.0, float(self.leverage_dpp_recency_window))
        freshness = (1.0 - age / window).clamp(0.0, 1.0)
        low_score_gate = (1.0 - score01).clamp(0.0, 1.0)
        gate_power = float(self.leverage_dpp_recency_gate_power)
        if gate_power == 0.0:
            low_score_gate = torch.ones_like(low_score_gate)
        elif gate_power != 1.0:
            low_score_gate = low_score_gate.pow(gate_power)
        recency_bonus = float(self.leverage_dpp_recency_lambda) * freshness * low_score_gate
        self._record_dpp_recency_debug(freshness, low_score_gate, recency_bonus)
        return score01 + recency_bonus

    def _dpp_quality_log(
        self,
        pool_scores: torch.Tensor,
        pool_frame_ids: Optional[torch.Tensor] = None,
        current_frame_id: Optional[int] = None,
    ) -> torch.Tensor:
        score_min = pool_scores.min(dim=-1, keepdim=True).values
        score_range = (pool_scores.max(dim=-1, keepdim=True).values - score_min).clamp_min(1e-12)
        score01 = ((pool_scores - score_min) / score_range).clamp(0.0, 1.0)
        leverage_log_quality = torch.log(score01 + 1e-6)
        return self.leverage_dpp_quality_beta * leverage_log_quality

    def _record_dpp_recency_debug(
        self,
        freshness: torch.Tensor,
        low_score_gate: torch.Tensor,
        recency_bonus: torch.Tensor,
    ) -> None:
        if not self.leverage_dpp_recency_debug:
            return
        with torch.no_grad():
            count = int(recency_bonus.numel())
            if count <= 0:
                return
            stats = self._last_dpp_recency_debug
            stats["count"] = stats.get("count", 0.0) + float(count)
            stats["freshness_sum"] = stats.get("freshness_sum", 0.0) + float(freshness.detach().float().sum().item())
            stats["gate_sum"] = stats.get("gate_sum", 0.0) + float(low_score_gate.detach().float().sum().item())
            stats["bonus_sum"] = stats.get("bonus_sum", 0.0) + float(recency_bonus.detach().float().sum().item())
            stats["freshness_max"] = max(stats.get("freshness_max", 0.0), float(freshness.detach().float().max().item()))
            stats["bonus_max"] = max(stats.get("bonus_max", 0.0), float(recency_bonus.detach().float().max().item()))
    @staticmethod
    def _format_dpp_recency_debug(stats: Dict[str, float], quality_beta: float, recency_lambda: float, window: int, gate_power: float) -> str:
        count = max(float(stats.get("count", 0.0)), 1.0)
        bonus_max = float(stats.get("bonus_max", 0.0))
        return (
            "[EvictionScore/recency] "
            f"lambda={recency_lambda} window={window} gate_power={gate_power} "
            f"freshness_mean={stats.get('freshness_sum', 0.0) / count:.4f} "
            f"freshness_max={stats.get('freshness_max', 0.0):.4f} "
            f"gate_mean={stats.get('gate_sum', 0.0) / count:.4f} "
            f"bonus_mean={stats.get('bonus_sum', 0.0) / count:.4f} "
            f"bonus_max={bonus_max:.4f} quality_beta={quality_beta} "
            f"score_bonus_max={bonus_max:.4f}"
        )
    @staticmethod
    def _expand_candidate_frame_ids(
        candidate_frame_ids: torch.Tensor,
        B: int,
        H: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        if candidate_frame_ids.dim() == 2:
            if tuple(candidate_frame_ids.shape) != (B, N):
                raise ValueError(
                    f"candidate_frame_ids must have shape [B, N] or [B, H, N], got {tuple(candidate_frame_ids.shape)}"
                )
            return candidate_frame_ids.to(device=device).unsqueeze(1).expand(B, H, N)
        if candidate_frame_ids.dim() == 3:
            if tuple(candidate_frame_ids.shape) != (B, H, N):
                raise ValueError(
                    f"candidate_frame_ids must have shape [B, N] or [B, H, N], got {tuple(candidate_frame_ids.shape)}"
                )
            return candidate_frame_ids.to(device=device)
        raise ValueError(
            f"candidate_frame_ids must have shape [B, N] or [B, H, N], got {tuple(candidate_frame_ids.shape)}"
        )
    @staticmethod
    def _layer_candidate_frame_ids(
        candidate_frame_ids: torch.Tensor,
        B: int,
        H: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        if candidate_frame_ids.dim() == 2:
            if tuple(candidate_frame_ids.shape) != (B, N):
                raise ValueError(
                    f"candidate_frame_ids must have shape [B, N] or [B, H, N], got {tuple(candidate_frame_ids.shape)}"
                )
            return candidate_frame_ids.to(device=device)
        if candidate_frame_ids.dim() == 3:
            if tuple(candidate_frame_ids.shape) != (B, H, N):
                raise ValueError(
                    f"candidate_frame_ids must have shape [B, N] or [B, H, N], got {tuple(candidate_frame_ids.shape)}"
                )
            return candidate_frame_ids[:, 0].to(device=device)
        raise ValueError(
            f"candidate_frame_ids must have shape [B, N] or [B, H, N], got {tuple(candidate_frame_ids.shape)}"
        )

    def _keep_after_layer_head_fast_dpp(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        candidate_frame_ids: Optional[torch.Tensor] = None,
        current_frame_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Vectorized head-wise Fast DPP using one layer-wise score vector."""
        B, H, N, D = candidate_k.shape
        keep_count = N - int(num_to_evict)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        expanded_indices = all_indices.view(1, 1, N).expand(B, H, N)
        if num_to_evict <= 0:
            return expanded_indices

        min_evictable = int(evictable_mask.sum(dim=-1).min().item())
        pool_size = min(
            min_evictable,
            max(int(num_to_evict), int(num_to_evict) * self.leverage_dpp_candidate_multiplier),
        )
        if pool_size < num_to_evict:
            raise ValueError(
                f"Cannot evict {num_to_evict} tokens from a vectorized DPP pool of size {pool_size}"
            )

        shared_mask = bool((evictable_mask == evictable_mask[:, :1]).all().item())
        if shared_mask:
            masked_scores = scores.masked_fill(~evictable_mask[:, 0], torch.inf)
            shared_pool = torch.argsort(masked_scores, dim=-1, stable=True)[:, :pool_size]
            pool = shared_pool.unsqueeze(1).expand(B, H, pool_size)
        else:
            head_scores = scores.unsqueeze(1).expand(B, H, N)
            masked_scores = head_scores.masked_fill(~evictable_mask, torch.inf)
            pool = torch.argsort(masked_scores, dim=-1, stable=True)[..., :pool_size]

        retain_count = pool_size - int(num_to_evict)
        if retain_count <= 0:
            evicted = pool
        else:
            gather_index = pool.unsqueeze(-1).expand(B, H, pool_size, D)
            features = torch.gather(candidate_k, dim=2, index=gather_index).float()
            if self.leverage_feature in KV_LEVERAGE_FEATURES:
                if candidate_v is None:
                    raise ValueError(
                        f"leverage_feature={self.leverage_feature!r} requires value cache tensor for layer_head_fast_dpp"
                    )
                value_features = torch.gather(candidate_v, dim=2, index=gather_index).float()
                features = torch.cat([features, value_features], dim=-1)
            features = F.normalize(features, p=2, dim=-1, eps=1e-12)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            pool_scores = torch.gather(
                scores.unsqueeze(1).expand(B, H, N),
                dim=-1,
                index=pool,
            ).float()
            pool_frame_ids = None
            if candidate_frame_ids is not None:
                frame_ids = self._expand_candidate_frame_ids(candidate_frame_ids, B, H, N, pool.device)
                pool_frame_ids = torch.gather(frame_ids, dim=-1, index=pool)
            log_quality = self._dpp_quality_log(pool_scores, pool_frame_ids, current_frame_id)

            selected_mask = torch.zeros(B, H, pool_size, device=scores.device, dtype=torch.bool)
            max_similarity_sq = torch.zeros(B, H, pool_size, device=scores.device, dtype=torch.float32)
            first_local = torch.argmax(log_quality, dim=-1, keepdim=True)
            selected_mask.scatter_(dim=-1, index=first_local, value=True)
            max_similarity_sq = torch.maximum(
                max_similarity_sq,
                self._batched_dpp_similarity_sq_to_selected(features, first_local),
            )

            selected_count = 1
            block_size = max(1, int(self.leverage_dpp_greedy_block_size))
            while selected_count < retain_count:
                current_block = min(block_size, retain_count - selected_count)
                diversity = (1.0 - max_similarity_sq).clamp_min(1e-12)
                greedy_scores = log_quality + self.leverage_dpp_diversity_beta * torch.log(diversity)
                greedy_scores = greedy_scores.masked_fill(selected_mask, -torch.inf)
                block_local = torch.topk(greedy_scores, k=current_block, dim=-1).indices
                selected_mask.scatter_(dim=-1, index=block_local, value=True)
                max_similarity_sq = torch.maximum(
                    max_similarity_sq,
                    self._batched_dpp_similarity_sq_to_selected(features, block_local),
                )
                selected_count += current_block
            evicted = pool[~selected_mask].reshape(B, H, int(num_to_evict))

        keep_mask = torch.ones(B, H, N, device=scores.device, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=evicted, value=False)
        return expanded_indices[keep_mask].reshape(B, H, keep_count)

    def _keep_after_layer_head_outlier_then_low_dpp(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        candidate_frame_ids: Optional[torch.Tensor] = None,
        current_frame_id: Optional[int] = None,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, N, _ = candidate_k.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, H, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        frame_ids = None
        if candidate_frame_ids is not None:
            frame_ids = self._expand_candidate_frame_ids(candidate_frame_ids, B, H, N, scores.device)
        for b in range(B):
            for h in range(H):
                evicted = self._outlier_then_low_evicted_indices(
                    scores[b],
                    evictable_mask[b, h],
                    int(num_to_evict),
                    lambda low_mask, remaining, b=b, h=h: self._fast_dpp_evicted_indices(
                        scores[b],
                        low_mask,
                        remaining,
                        lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                        row_frame_ids=frame_ids[b, h] if frame_ids is not None else None,
                        current_frame_id=current_frame_id,
                        use_full_pool=False,
                    ),
                    outlier_scores=raw_outlier_scores[b] if raw_outlier_scores is not None else None,
                )
                keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
                keep_mask[evicted] = False
                kept[b, h] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    @staticmethod
    def _batched_dpp_similarity_sq_to_selected(
        features: torch.Tensor,
        selected_local: torch.Tensor,
    ) -> torch.Tensor:
        feature_dim = int(features.shape[-1])
        selected_features = torch.gather(
            features,
            dim=2,
            index=selected_local.unsqueeze(-1).expand(*selected_local.shape, feature_dim),
        )
        return torch.matmul(features, selected_features.transpose(-1, -2)).square().amax(dim=-1)

    def _keep_after_head_fast_dpp(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        *,
        candidate_frame_ids: Optional[torch.Tensor] = None,
        current_frame_id: Optional[int] = None,
        use_full_pool: bool = False,
        outlier_then_low: bool = False,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, N = scores.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, H, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        frame_ids = None
        if candidate_frame_ids is not None:
            frame_ids = self._expand_candidate_frame_ids(candidate_frame_ids, B, H, N, scores.device)
        for b in range(B):
            for h in range(H):
                if outlier_then_low:
                    evicted = self._outlier_then_low_evicted_indices(
                        scores[b, h],
                        evictable_mask[b, h],
                        int(num_to_evict),
                        lambda low_mask, remaining, b=b, h=h: self._fast_dpp_evicted_indices(
                            scores[b, h],
                            low_mask,
                            remaining,
                            lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                            row_frame_ids=frame_ids[b, h] if frame_ids is not None else None,
                            current_frame_id=current_frame_id,
                            use_full_pool=False,
                        ),
                        outlier_scores=raw_outlier_scores[b, h] if raw_outlier_scores is not None else None,
                    )
                else:
                    evicted = self._fast_dpp_evicted_indices(
                        scores[b, h],
                        evictable_mask[b, h],
                        int(num_to_evict),
                        lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                        row_frame_ids=frame_ids[b, h] if frame_ids is not None else None,
                        current_frame_id=current_frame_id,
                        use_full_pool=use_full_pool,
                    )
                keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
                keep_mask[evicted] = False
                kept[b, h] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    def _keep_after_layer_fast_dpp(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        *,
        candidate_frame_ids: Optional[torch.Tensor] = None,
        current_frame_id: Optional[int] = None,
        use_full_pool: bool = False,
        outlier_then_low: bool = False,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = scores.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        frame_ids = None
        if candidate_frame_ids is not None:
            frame_ids = self._layer_candidate_frame_ids(candidate_frame_ids, B, candidate_k.shape[1], N, scores.device)
        for b in range(B):
            if outlier_then_low:
                evicted = self._outlier_then_low_evicted_indices(
                    scores[b],
                    evictable_mask[b],
                    int(num_to_evict),
                    lambda low_mask, remaining, b=b: self._fast_dpp_evicted_indices(
                        scores[b],
                        low_mask,
                        remaining,
                        lambda idx, b=b: self._layer_dpp_features(candidate_k, candidate_v, b, idx),
                        row_frame_ids=frame_ids[b] if frame_ids is not None else None,
                        current_frame_id=current_frame_id,
                        use_full_pool=False,
                    ),
                    outlier_scores=raw_outlier_scores[b] if raw_outlier_scores is not None else None,
                )
            else:
                evicted = self._fast_dpp_evicted_indices(
                    scores[b],
                    evictable_mask[b],
                    int(num_to_evict),
                    lambda idx, b=b: self._layer_dpp_features(candidate_k, candidate_v, b, idx),
                    row_frame_ids=frame_ids[b] if frame_ids is not None else None,
                    current_frame_id=current_frame_id,
                    use_full_pool=use_full_pool,
                )
            keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
            keep_mask[evicted] = False
            kept[b] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    def reset_projected_key_cache(self) -> None:
        self._projected_key_cache = None
        self._projected_key_head_cache = None
        self._projected_key_pre_norm_cache = None
        self._projected_key_cache_meta = None
        self._last_projected_key_features = None
        self._last_projected_key_head_features = None
        self._last_projected_key_pre_norms = None
        self._last_projected_key_cache_meta = None

    def _projected_key_cache_meta_for(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        sketch_dim: int,
        device: torch.device,
    ) -> Dict[str, object]:
        return {
            "batch_size": int(batch_size),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
            "sketch_dim": int(sketch_dim),
            "device": str(device),
            "normalize_before_projection": bool(self.leverage_normalize_before_projection),
            "normalize_before_projection_headwise": bool(self.leverage_normalize_before_projection_headwise),
            "projected_head_features": bool(self._needs_projected_head_features()),
        }

    def _needs_projected_head_features(self) -> bool:
        return (
            self.leverage_similarity_feature_projection == "random"
            and self.leverage_similarity_granularity == "head"
        )

    def _projected_key_cache_length(
        self,
        *,
        batch_size: int,
        num_heads: int,
        num_tokens: int,
        head_dim: int,
        sketch_dim: int,
        device: torch.device,
    ) -> int:
        if not self.leverage_projected_key_cache:
            return 0
        cache = self._projected_key_cache
        head_cache = self._projected_key_head_cache
        norm_cache = self._projected_key_pre_norm_cache
        meta = self._projected_key_cache_meta
        expected = self._projected_key_cache_meta_for(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            sketch_dim=sketch_dim,
            device=device,
        )
        need_head_features = self._needs_projected_head_features()
        if cache is None or norm_cache is None or meta != expected:
            return 0
        if need_head_features and head_cache is None:
            return 0
        cache_len = int(cache.shape[1])
        if cache_len > int(num_tokens):
            self.reset_projected_key_cache()
            return 0
        if cache.shape != (batch_size, cache_len, sketch_dim):
            self.reset_projected_key_cache()
            return 0
        if head_cache is not None and head_cache.shape != (batch_size, num_heads, cache_len, sketch_dim):
            self.reset_projected_key_cache()
            return 0
        if norm_cache.shape != (batch_size, cache_len):
            self.reset_projected_key_cache()
            return 0
        return cache_len

    def _normalize_layer_key_before_projection(self, mat_k: torch.Tensor) -> torch.Tensor:
        if not self.leverage_normalize_before_projection:
            return mat_k
        if self.leverage_normalize_before_projection_headwise:
            norm = mat_k.square().sum(dim=3, keepdim=True).sqrt().clamp_min(1e-12)
        else:
            norm = mat_k.square().sum(dim=(1, 3), keepdim=True).sqrt().clamp_min(1e-12)
        return mat_k / norm

    def _project_key_with_omega(
        self,
        mat_k: torch.Tensor,
        omega_key: torch.Tensor,
        *,
        need_head_features: bool = False,
    ):
        mat_k = self._normalize_layer_key_before_projection(mat_k)
        B, H, N, D = mat_k.shape
        sketch_dim = int(omega_key.shape[-1])
        if need_head_features:
            head_projected_raw = torch.einsum("bhnd,hds->bhns", mat_k, omega_key)
            layer_projected_raw = head_projected_raw.sum(dim=1)
        else:
            flat_k = mat_k.permute(0, 2, 1, 3).reshape(B, N, H * D)
            omega_flat = omega_key.reshape(H * D, sketch_dim)
            layer_projected_raw = torch.matmul(flat_k, omega_flat)
            head_projected_raw = None
        pre_norms = torch.linalg.vector_norm(layer_projected_raw.detach(), ord=2, dim=-1)
        layer_projected = self._maybe_normalize_rows(layer_projected_raw)
        head_projected = self._maybe_normalize_rows(head_projected_raw) if head_projected_raw is not None else None
        return layer_projected, head_projected, pre_norms

    def _capture_projected_pre_normalization_norm_values(self, norms: torch.Tensor) -> None:
        if getattr(self, "_capture_projected_norms", False):
            self._last_projected_pre_norms = norms.detach()

    def _layer_key_projected_features(
        self,
        mat_k: torch.Tensor,
        omega_key: torch.Tensor,
        sketch_dim: int,
        profile: Dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, N, D = mat_k.shape
        meta = self._projected_key_cache_meta_for(
            batch_size=B,
            num_heads=H,
            head_dim=D,
            sketch_dim=sketch_dim,
            device=mat_k.device,
        )
        need_head_features = self._needs_projected_head_features()
        cache_len = self._projected_key_cache_length(
            batch_size=B,
            num_heads=H,
            num_tokens=N,
            head_dim=D,
            sketch_dim=sketch_dim,
            device=mat_k.device,
        )
        if cache_len > 0:
            layer_parts = [self._projected_key_cache[:, :cache_len].to(device=mat_k.device)]
            head_parts = (
                [self._projected_key_head_cache[:, :, :cache_len].to(device=mat_k.device)]
                if need_head_features
                else None
            )
            norm_parts = [self._projected_key_pre_norm_cache[:, :cache_len].to(device=mat_k.device)]
            if cache_len < N:
                suffix_layer, suffix_head, suffix_norms = self._project_key_with_omega(
                    mat_k[:, :, cache_len:, :],
                    omega_key,
                    need_head_features=need_head_features,
                )
                layer_parts.append(suffix_layer)
                if head_parts is not None and suffix_head is not None:
                    head_parts.append(suffix_head)
                norm_parts.append(suffix_norms)
            leverage_matrix = torch.cat(layer_parts, dim=1)
            head_leverage_matrix = torch.cat(head_parts, dim=2) if head_parts is not None else None
            pre_norms = torch.cat(norm_parts, dim=1)
            profile["projection_cache_hits"] = float(cache_len)
            profile["projection_cache_misses"] = float(N - cache_len)
        else:
            leverage_matrix, head_leverage_matrix, pre_norms = self._project_key_with_omega(
                mat_k,
                omega_key,
                need_head_features=need_head_features,
            )
            profile["projection_cache_hits"] = 0.0
            profile["projection_cache_misses"] = float(N)

        self._capture_projected_pre_normalization_norm_values(pre_norms)
        self._last_projected_key_features = leverage_matrix.detach()
        self._last_projected_key_head_features = head_leverage_matrix.detach() if head_leverage_matrix is not None else None
        self._last_projected_key_pre_norms = pre_norms.detach()
        self._last_projected_key_cache_meta = meta
        return leverage_matrix, head_leverage_matrix

    def update_projected_key_cache_after_eviction(
        self,
        kept_candidate_indices: torch.Tensor,
        *,
        tail_k: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.leverage_projected_key_cache:
            return
        features = self._last_projected_key_features
        head_features = self._last_projected_key_head_features
        pre_norms = self._last_projected_key_pre_norms
        meta = self._last_projected_key_cache_meta
        need_head_features = self._needs_projected_head_features()
        if features is None or pre_norms is None or meta is None:
            self.reset_projected_key_cache()
            return
        if need_head_features and head_features is None:
            self.reset_projected_key_cache()
            return
        if kept_candidate_indices.ndim != 3 or kept_candidate_indices.shape[0] != features.shape[0]:
            self.reset_projected_key_cache()
            return
        B, N, S = features.shape
        H = int(meta["num_heads"])
        if kept_candidate_indices.shape[1] < 1:
            self.reset_projected_key_cache()
            return
        if head_features is not None and head_features.shape[:3] != (B, H, N):
            self.reset_projected_key_cache()
            return
        row_indices = kept_candidate_indices[:, 0, :].to(device=features.device, dtype=torch.long)
        if row_indices.numel() > 0 and (int(row_indices.min().item()) < 0 or int(row_indices.max().item()) >= N):
            self.reset_projected_key_cache()
            return
        gather_layer = row_indices.unsqueeze(-1).expand(B, row_indices.shape[1], S)
        next_features = torch.gather(features, 1, gather_layer)
        next_norms = torch.gather(pre_norms, 1, row_indices)
        if head_features is not None:
            gather_head = row_indices[:, None, :, None].expand(B, H, row_indices.shape[1], S)
            next_head_features = torch.gather(head_features, 2, gather_head)
        else:
            next_head_features = None

        if tail_k is not None and int(tail_k.shape[2]) > 0:
            with torch.cuda.amp.autocast(enabled=False):
                tail = torch.nan_to_num(tail_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                omega = self._get_leverage_right_sketch(
                    H * int(meta["head_dim"]),
                    int(meta["sketch_dim"]),
                    device=tail.device,
                    seed=self.leverage_random_seed,
                )
                omega_key = omega[: H * int(meta["head_dim"])].view(H, int(meta["head_dim"]), int(meta["sketch_dim"]))
                tail_features, tail_head_features, tail_norms = self._project_key_with_omega(
                    tail,
                    omega_key,
                    need_head_features=need_head_features,
                )
            next_features = torch.cat([next_features, tail_features.detach()], dim=1)
            if next_head_features is not None and tail_head_features is not None:
                next_head_features = torch.cat([next_head_features, tail_head_features.detach()], dim=2)
            next_norms = torch.cat([next_norms, tail_norms.detach()], dim=1)

        self._projected_key_cache = next_features.detach()
        self._projected_key_head_cache = next_head_features.detach() if next_head_features is not None else None
        self._projected_key_pre_norm_cache = next_norms.detach()
        self._projected_key_cache_meta = dict(meta)

    def _similarity_projection_error(self, granularity: str) -> ValueError:
        return ValueError(
            "leverage_similarity_feature_projection='random' requires reusable projected "
            f"{granularity}-wise leverage features from right_sketch or right_sketch_ridge"
        )

    def _capture_projected_pre_normalization_norms(self, projected: torch.Tensor) -> None:
        if getattr(self, "_capture_projected_norms", False):
            self._last_projected_pre_norms = torch.linalg.vector_norm(projected.detach(), ord=2, dim=-1)

    def _store_projected_similarity_features(self, projected: torch.Tensor, granularity: str) -> None:
        if self.leverage_similarity_feature_projection != "random":
            return
        stored = torch.nan_to_num(
            projected.detach().to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if granularity == "head" and stored.ndim == 4:
            self._last_similarity_head_features = stored
        elif granularity == "layer" and stored.ndim == 3:
            self._last_similarity_layer_features = stored

    def _similarity_feature_fn(self, raw_feature_fn, *, batch_idx: int, head_idx: Optional[int] = None):
        if self.leverage_similarity_feature_projection == "raw":
            return raw_feature_fn
        if head_idx is None:
            features = self._last_similarity_layer_features
            if features is None or features.ndim != 3:
                raise self._similarity_projection_error("layer")
            row_features = features[int(batch_idx)]
        else:
            features = self._last_similarity_head_features
            if features is None or features.ndim != 4:
                raise self._similarity_projection_error("head")
            row_features = features[int(batch_idx), int(head_idx)]
        return lambda idx, row_features=row_features: torch.nan_to_num(
            row_features.index_select(0, idx).to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _keep_after_layer_head_similarity_topk(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        *,
        outlier_then_low: bool = False,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, N, _ = candidate_k.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, H, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        for b in range(B):
            for h in range(H):
                if outlier_then_low:
                    evicted = self._outlier_then_low_evicted_indices(
                        scores[b],
                        evictable_mask[b],
                        int(num_to_evict),
                        lambda low_mask, remaining, b=b, h=h: self._similarity_topk_evicted_indices(
                            scores[b],
                            low_mask,
                            remaining,
                            self._similarity_feature_fn(
                                lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                                batch_idx=b,
                                head_idx=h,
                            ),
                        ),
                        outlier_scores=raw_outlier_scores[b] if raw_outlier_scores is not None else None,
                    )
                else:
                    evicted = self._similarity_topk_evicted_indices(
                        scores[b],
                        evictable_mask[b],
                        int(num_to_evict),
                        self._similarity_feature_fn(
                                lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                                batch_idx=b,
                                head_idx=h,
                            ),
                    )
                keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
                keep_mask[evicted] = False
                kept[b, h] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    def _keep_after_head_similarity_topk(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        *,
        outlier_then_low: bool = False,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, N = scores.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, H, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        for b in range(B):
            for h in range(H):
                if outlier_then_low:
                    evicted = self._outlier_then_low_evicted_indices(
                        scores[b, h],
                        evictable_mask[b, h],
                        int(num_to_evict),
                        lambda low_mask, remaining, b=b, h=h: self._similarity_topk_evicted_indices(
                            scores[b, h],
                            low_mask,
                            remaining,
                            self._similarity_feature_fn(
                                lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                                batch_idx=b,
                                head_idx=h,
                            ),
                        ),
                        outlier_scores=raw_outlier_scores[b, h] if raw_outlier_scores is not None else None,
                    )
                else:
                    evicted = self._similarity_topk_evicted_indices(
                        scores[b, h],
                        evictable_mask[b, h],
                        int(num_to_evict),
                        self._similarity_feature_fn(
                                lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
                                batch_idx=b,
                                head_idx=h,
                            ),
                    )
                keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
                keep_mask[evicted] = False
                kept[b, h] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    def _keep_after_layer_similarity_topk(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        *,
        outlier_then_low: bool = False,
        raw_outlier_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = scores.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        for b in range(B):
            if outlier_then_low:
                evicted = self._outlier_then_low_evicted_indices(
                    scores[b],
                    evictable_mask[b],
                    int(num_to_evict),
                    lambda low_mask, remaining, b=b: self._similarity_topk_evicted_indices(
                        scores[b],
                        low_mask,
                        remaining,
                        self._similarity_feature_fn(
                            lambda idx, b=b: self._layer_dpp_features(candidate_k, candidate_v, b, idx),
                            batch_idx=b,
                        ),
                    ),
                    outlier_scores=raw_outlier_scores[b] if raw_outlier_scores is not None else None,
                )
            else:
                evicted = self._similarity_topk_evicted_indices(
                    scores[b],
                    evictable_mask[b],
                    int(num_to_evict),
                    self._similarity_feature_fn(
                            lambda idx, b=b: self._layer_dpp_features(candidate_k, candidate_v, b, idx),
                            batch_idx=b,
                        ),
                )
            keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
            keep_mask[evicted] = False
            kept[b] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    def _maybe_project_fast_dpp_features(self, features: torch.Tensor) -> torch.Tensor:
        if (
            self.leverage_dpp_feature_projection != "random"
            or self.leverage_eviction_selector != "fast_dpp"
        ):
            return features
        feature_dim = int(features.shape[-1])
        sketch_dim = min(int(self.leverage_ridge_dim), feature_dim)
        omega = self._get_leverage_right_sketch(
            feature_dim,
            sketch_dim,
            device=features.device,
            seed=self.leverage_random_seed,
        )
        projected = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).matmul(omega)
        self._last_layer_feature_shape = tuple(projected.shape)
        return projected

    def _fast_dpp_evicted_indices(
        self,
        row_scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        feature_fn,
        *,
        row_frame_ids: Optional[torch.Tensor] = None,
        current_frame_id: Optional[int] = None,
        use_full_pool: bool = False,
    ) -> torch.Tensor:
        if num_to_evict <= 0:
            return torch.empty(0, device=row_scores.device, dtype=torch.long)

        all_indices = torch.arange(row_scores.shape[-1], device=row_scores.device, dtype=torch.long)
        evictable = all_indices[evictable_mask]
        if evictable.numel() < num_to_evict:
            raise ValueError(
                f"Cannot evict {num_to_evict} tokens from only {int(evictable.numel())} evictable candidates"
            )

        row_scores = torch.nan_to_num(row_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if use_full_pool:
            pool = evictable
            pool_size = int(pool.numel())
        else:
            pool_size = min(
                int(evictable.numel()),
                max(int(num_to_evict), int(num_to_evict) * self.leverage_dpp_candidate_multiplier),
            )
            order = torch.argsort(row_scores.index_select(0, evictable), stable=True)
            pool = evictable.index_select(0, order[:pool_size])
        num_to_retain_from_pool = int(pool_size) - int(num_to_evict)
        if num_to_retain_from_pool <= 0:
            return pool

        features = feature_fn(pool).float()
        features = self._maybe_project_fast_dpp_features(features)
        features = F.normalize(features, p=2, dim=-1, eps=1e-12)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if use_full_pool:
            pool_scores = torch.ones(pool_size, device=row_scores.device, dtype=torch.float32)
        else:
            # The low-score pool is vulnerable; DPP selects representatives to retain.
            pool_scores = row_scores.index_select(0, pool)
        pool_frame_ids = row_frame_ids.to(device=row_scores.device).index_select(0, pool) if row_frame_ids is not None else None
        log_quality = self._dpp_quality_log(pool_scores, pool_frame_ids, current_frame_id)

        block_size = max(1, int(self.leverage_dpp_greedy_block_size))
        selected = []
        selected_count = 0
        available = torch.ones(pool_size, device=row_scores.device, dtype=torch.bool)
        max_similarity_sq = torch.zeros(pool_size, device=row_scores.device, dtype=torch.float32)

        full_sim_sq = None
        if pool_size * pool_size <= self.leverage_dpp_full_sim_max_elements:
            full_sim_sq = torch.mm(features, features.transpose(0, 1)).square()

        first_local = torch.argmax(log_quality, dim=0, keepdim=True)
        selected.append(first_local)
        selected_count += 1
        available[first_local] = False
        if full_sim_sq is None:
            first_similarity_sq = self._dpp_similarity_sq_to_selected(features, first_local)
        else:
            first_similarity_sq = full_sim_sq.index_select(1, first_local).max(dim=1).values
        max_similarity_sq = torch.maximum(max_similarity_sq, first_similarity_sq)

        while selected_count < int(num_to_retain_from_pool):
            remaining = int(num_to_retain_from_pool) - selected_count
            current_block = min(block_size, remaining)
            diversity = (1.0 - max_similarity_sq).clamp_min(1e-12)
            greedy_scores = log_quality + self.leverage_dpp_diversity_beta * torch.log(diversity)
            greedy_scores = greedy_scores.masked_fill(~available, -torch.inf)
            _, block_local = torch.topk(greedy_scores, k=current_block, dim=0)
            selected.append(block_local)
            selected_count += int(block_local.numel())
            available[block_local] = False
            if full_sim_sq is None:
                block_similarity_sq = self._dpp_similarity_sq_to_selected(features, block_local)
            else:
                block_similarity_sq = full_sim_sq.index_select(1, block_local).max(dim=1).values
            max_similarity_sq = torch.maximum(max_similarity_sq, block_similarity_sq)

        selected_local = torch.cat([idx.reshape(-1) for idx in selected], dim=0)
        retain_mask = torch.zeros(pool_size, device=row_scores.device, dtype=torch.bool)
        retain_mask[selected_local] = True
        return pool[~retain_mask]

    def _similarity_topk_evicted_indices(
        self,
        row_scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        feature_fn,
    ) -> torch.Tensor:
        if num_to_evict <= 0:
            return torch.empty(0, device=row_scores.device, dtype=torch.long)

        all_indices = torch.arange(row_scores.shape[-1], device=row_scores.device, dtype=torch.long)
        evictable = all_indices[evictable_mask]
        if evictable.numel() < num_to_evict:
            raise ValueError(
                f"Cannot evict {num_to_evict} tokens from only {int(evictable.numel())} evictable candidates"
            )

        row_scores = torch.nan_to_num(row_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pool_size = min(
            int(evictable.numel()),
            max(int(num_to_evict), int(num_to_evict) * self.leverage_dpp_candidate_multiplier),
        )
        order = torch.argsort(row_scores.index_select(0, evictable), stable=True)
        pool = evictable.index_select(0, order[:pool_size])
        if pool_size <= int(num_to_evict):
            return pool

        retained_mask = torch.ones(row_scores.shape[-1], device=row_scores.device, dtype=torch.bool)
        retained_mask[pool] = False
        retained = all_indices[retained_mask]
        if retained.numel() == 0:
            return pool[: int(num_to_evict)]

        pool_features = feature_fn(pool).float()
        retained_features = feature_fn(retained).float()
        pool_features = F.normalize(pool_features, p=2, dim=-1, eps=1e-12)
        retained_features = F.normalize(retained_features, p=2, dim=-1, eps=1e-12)
        pool_features = torch.nan_to_num(pool_features, nan=0.0, posinf=0.0, neginf=0.0)
        retained_features = torch.nan_to_num(retained_features, nan=0.0, posinf=0.0, neginf=0.0)

        max_similarity = torch.mm(pool_features, retained_features.transpose(0, 1)).max(dim=1).values
        pool_leverage = row_scores.index_select(0, pool).clamp_min(1e-12)
        gamma = float(self.leverage_similarity_leverage_gamma)
        leverage_denom = pool_leverage.pow(gamma) if gamma != 1.0 else pool_leverage
        eviction_scores = max_similarity / leverage_denom.clamp_min(1e-12)
        local_evicted = torch.topk(eviction_scores, k=int(num_to_evict), dim=0).indices
        return pool.index_select(0, local_evicted)

    @staticmethod
    def _dpp_similarity_sq_to_selected(features: torch.Tensor, selected_local: torch.Tensor) -> torch.Tensor:
        selected_features = features.index_select(0, selected_local.reshape(-1))
        return torch.mm(features, selected_features.transpose(0, 1)).square().max(dim=1).values

    @staticmethod
    def _head_dpp_features(candidate_k: torch.Tensor, batch_idx: int, head_idx: int, indices: torch.Tensor) -> torch.Tensor:
        features = candidate_k[batch_idx, head_idx].index_select(0, indices)
        return torch.nan_to_num(features.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _layer_dpp_features(
        self,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        batch_idx: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        _, H, _, D = candidate_k.shape
        mat_k = candidate_k[batch_idx].index_select(1, indices).to(dtype=torch.float32)
        mat_k = torch.nan_to_num(mat_k, nan=0.0, posinf=0.0, neginf=0.0)
        if self.leverage_projection == "head_mean":
            head_chunks = torch.tensor_split(mat_k, self.leverage_head_mean_dim, dim=-1)
            head_features = torch.stack([chunk.mean(dim=-1) for chunk in head_chunks], dim=-1)
            features = head_features.permute(1, 0, 2).reshape(indices.numel(), H * self.leverage_head_mean_dim)
        else:
            features = mat_k.transpose(0, 1).reshape(indices.numel(), H * D)
            if self.leverage_feature in KV_LEVERAGE_FEATURES:
                if candidate_v is None:
                    raise ValueError(f"leverage_feature={self.leverage_feature!r} requires value cache tensor")
                mat_v = candidate_v[batch_idx].index_select(1, indices).to(dtype=torch.float32)
                mat_v = torch.nan_to_num(mat_v, nan=0.0, posinf=0.0, neginf=0.0)
                value_features = mat_v.transpose(0, 1).reshape(indices.numel(), H * D)
                features = torch.cat([features, value_features], dim=-1)
        self._last_layer_feature_shape = tuple(features.shape)
        return features

    def _combine_history_anchor_patch_topk_mask(
        self,
        scores: torch.Tensor,
        candidate_frame_ids: Optional[torch.Tensor],
        candidate_token_indices: Optional[torch.Tensor],
        candidate_evictable_mask: Optional[torch.Tensor],
        history_anchor_frame_ids,
        patch_topk_per_frame: int,
        max_anchor_frames: int,
        special_token_count: int,
    ) -> Optional[torch.Tensor]:
        patch_topk_per_frame = int(patch_topk_per_frame)
        max_anchor_frames = int(max_anchor_frames)
        special_token_count = int(special_token_count)
        if (
            patch_topk_per_frame <= 0
            or max_anchor_frames <= 0
            or history_anchor_frame_ids is None
            or candidate_frame_ids is None
            or candidate_token_indices is None
            or special_token_count <= 0
        ):
            return candidate_evictable_mask

        if not isinstance(history_anchor_frame_ids, torch.Tensor):
            if len(history_anchor_frame_ids) == 0:
                return candidate_evictable_mask
            anchor_ids = torch.as_tensor(history_anchor_frame_ids, device=scores.device, dtype=torch.long)
        else:
            anchor_ids = history_anchor_frame_ids.to(device=scores.device, dtype=torch.long).reshape(-1)
        if anchor_ids.numel() == 0:
            return candidate_evictable_mask
        anchor_ids = torch.unique(anchor_ids, sorted=False)

        score_rows = torch.nan_to_num(scores.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        leading_shape = score_rows.shape[:-1]
        num_candidates = int(score_rows.shape[-1])
        flat_scores = score_rows.reshape(-1, num_candidates)
        flat_frames = self._metadata_like_scores(
            candidate_frame_ids,
            scores,
            name="candidate_frame_ids",
            dtype=torch.long,
        ).reshape(-1, num_candidates)
        flat_tokens = self._metadata_like_scores(
            candidate_token_indices,
            scores,
            name="candidate_token_indices",
            dtype=torch.long,
        ).reshape(-1, num_candidates)

        membership = flat_frames.unsqueeze(1).eq(anchor_ids.view(1, -1, 1))
        patch_mask = membership & flat_tokens.unsqueeze(1).ge(special_token_count)
        valid_patch_counts = patch_mask.sum(dim=-1)
        if not bool(valid_patch_counts.gt(0).any().item()):
            return candidate_evictable_mask

        patch_k = min(max(patch_topk_per_frame, 1), num_candidates)
        masked_patch_scores = flat_scores.unsqueeze(1).masked_fill(~patch_mask, -torch.inf)
        patch_top_values, patch_top_indices = torch.topk(masked_patch_scores, k=patch_k, dim=-1)
        finite_patch_values = torch.isfinite(patch_top_values)
        frame_den = valid_patch_counts.clamp(max=patch_k).clamp_min(1).to(dtype=flat_scores.dtype)
        frame_scores = patch_top_values.masked_fill(~finite_patch_values, 0.0).sum(dim=-1) / frame_den
        frame_scores = frame_scores.masked_fill(valid_patch_counts <= 0, -torch.inf)

        frame_k = min(max_anchor_frames, int(anchor_ids.numel()))
        selected_frame_scores, selected_frame_indices = torch.topk(frame_scores, k=frame_k, dim=-1)
        selected_anchor_rows = torch.zeros_like(frame_scores, dtype=torch.bool)
        selected_anchor_rows.scatter_(1, selected_frame_indices, torch.isfinite(selected_frame_scores))

        selected_membership = membership & selected_anchor_rows.unsqueeze(-1)
        protect_special = selected_membership.any(dim=1) & flat_tokens.lt(special_token_count)

        patch_top_rows = torch.zeros_like(patch_mask, dtype=torch.bool)
        patch_top_rows.scatter_(2, patch_top_indices, finite_patch_values)
        protect_patch = (patch_top_rows & selected_anchor_rows.unsqueeze(-1) & patch_mask).any(dim=1)
        protected = protect_special | protect_patch
        anchor_evictable_mask = (~protected).reshape(*leading_shape, num_candidates)

        if candidate_evictable_mask is None:
            return anchor_evictable_mask
        existing = self._metadata_like_scores(
            candidate_evictable_mask,
            scores,
            name="candidate_evictable_mask",
            dtype=torch.bool,
        )
        return existing & anchor_evictable_mask

    @staticmethod
    def _metadata_like_scores(
        metadata: torch.Tensor,
        scores: torch.Tensor,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        B = int(scores.shape[0])
        N = int(scores.shape[-1])
        data = metadata.to(device=scores.device, dtype=dtype)
        if scores.ndim == 2:
            if data.ndim == 2 and tuple(data.shape) == (B, N):
                return data
            if data.ndim == 3 and data.shape[0] == B and data.shape[2] == N:
                if dtype == torch.bool:
                    return data.all(dim=1)
                return data[:, 0]
        elif scores.ndim == 3:
            H = int(scores.shape[1])
            if data.ndim == 3 and tuple(data.shape) == (B, H, N):
                return data
            if data.ndim == 2 and tuple(data.shape) == (B, N):
                return data.unsqueeze(1).expand(B, H, N)
        raise ValueError(f"{name} shape {tuple(data.shape)} is incompatible with scores {tuple(scores.shape)}")

    def _keep_with_recent_protection(
        self,
        scores: torch.Tensor,
        num_to_keep: int,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_idx: Optional[int],
        protect_recent_frames: int,
        candidate_evictable_mask: Optional[torch.Tensor],
        *,
        evict_highest: bool,
        shared_across_heads: bool,
        num_heads: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        """Keep tokens after excluding protected tokens from eviction candidates."""
        evictable_mask, debug = self._build_evictable_mask(
            scores,
            candidate_frame_ids,
            current_frame_idx,
            protect_recent_frames,
            candidate_evictable_mask,
            shared_across_heads=shared_across_heads,
        )
        if debug is None:
            raise ValueError("Protected-token selection requires at least one protection mask")

        requested_evict = scores.shape[-1] - int(num_to_keep)
        actual_evict = min(requested_evict, int(evictable_mask.sum(dim=-1).min().item()))
        if shared_across_heads:
            B = int(scores.shape[0])
            source_mask = candidate_evictable_mask if candidate_evictable_mask is not None else candidate_frame_ids
            if num_heads is not None:
                H = int(num_heads)
            elif source_mask is not None and source_mask.ndim == 3:
                H = int(source_mask.shape[1])
            else:
                H = 1
            kept_2d = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest)
            kept = kept_2d.unsqueeze(1).expand(B, H, kept_2d.shape[-1])
        else:
            kept = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest)

        debug["requested_eviction_count"] = int(requested_evict)
        debug["actual_eviction_count"] = int(actual_evict)
        debug["limited_by_protection"] = int(actual_evict < requested_evict)
        return kept, debug

    @staticmethod
    def _build_evictable_mask(
        scores: torch.Tensor,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_idx: Optional[int],
        protect_recent_frames: int,
        candidate_evictable_mask: Optional[torch.Tensor],
        *,
        shared_across_heads: bool,
    ) -> tuple[torch.Tensor, Optional[Dict[str, int]]]:
        if shared_across_heads:
            if scores.ndim != 2:
                raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
        elif scores.ndim != 3:
            raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")

        evictable_mask = torch.ones_like(scores, dtype=torch.bool)
        protected_special_tokens = 0
        protection_enabled = False

        if candidate_evictable_mask is not None:
            special_mask = candidate_evictable_mask.to(device=scores.device, dtype=torch.bool)
            if shared_across_heads:
                if special_mask.shape == scores.shape:
                    pass
                elif special_mask.ndim == 3 and special_mask.shape[0] == scores.shape[0] and special_mask.shape[2] == scores.shape[1]:
                    special_mask = special_mask.all(dim=1)
                else:
                    raise ValueError(
                        "Expected candidate_evictable_mask [B, N] or [B, H, N] for layer-wise protection, "
                        f"got {tuple(special_mask.shape)} for scores {tuple(scores.shape)}"
                    )
            elif special_mask.shape != scores.shape:
                raise ValueError(
                    "candidate_evictable_mask must match head-wise score shape, "
                    f"got {tuple(special_mask.shape)} vs {tuple(scores.shape)}"
                )
            evictable_mask &= special_mask
            protected_special_tokens = int((~special_mask).sum().item())
            protection_enabled = True

        if protect_recent_frames > 0:
            if current_frame_idx is None:
                raise ValueError("current_frame_idx is required when protect_recent_frames > 0")
            if candidate_frame_ids is None:
                raise ValueError("candidate_frame_ids is required when protect_recent_frames > 0")
            threshold = int(current_frame_idx) - int(protect_recent_frames) + 1
            frame_ids = candidate_frame_ids.to(device=scores.device, dtype=torch.long)
            if shared_across_heads:
                if frame_ids.ndim == 2 and frame_ids.shape == scores.shape:
                    recent_mask = (frame_ids < 0) | (frame_ids < threshold)
                elif frame_ids.ndim == 3 and frame_ids.shape[0] == scores.shape[0] and frame_ids.shape[2] == scores.shape[1]:
                    recent_mask = ((frame_ids < 0) | (frame_ids < threshold)).all(dim=1)
                else:
                    raise ValueError(
                        "Expected candidate_frame_ids [B, N] or [B, H, N] for layer-wise protection, "
                        f"got {tuple(frame_ids.shape)} for scores {tuple(scores.shape)}"
                    )
            else:
                if frame_ids.shape == scores.shape:
                    recent_mask = (frame_ids < 0) | (frame_ids < threshold)
                elif frame_ids.ndim == 2 and frame_ids.shape[0] == scores.shape[0] and frame_ids.shape[1] == scores.shape[2]:
                    recent_mask = ((frame_ids < 0) | (frame_ids < threshold)).unsqueeze(1).expand_as(scores)
                else:
                    raise ValueError(
                        "candidate_frame_ids must match head-wise score shape or be [B, N], "
                        f"got {tuple(frame_ids.shape)} vs {tuple(scores.shape)}"
                    )
            evictable_mask &= recent_mask
            protection_enabled = True

        if not protection_enabled:
            return evictable_mask, None

        return evictable_mask, {
            "current_frame_idx": -1 if current_frame_idx is None else int(current_frame_idx),
            "protect_recent_frames": int(protect_recent_frames),
            "protected_special_tokens": protected_special_tokens,
            "protected_tokens": int((~evictable_mask).sum().item()),
            "candidate_tokens": int(evictable_mask.sum().item()),
        }

    @staticmethod
    def _keep_after_eviction(
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        evict_highest: bool,
    ) -> torch.Tensor:
        leading_shape = scores.shape[:-1]
        num_candidates = int(scores.shape[-1])
        keep_count = num_candidates - int(num_to_evict)
        all_indices = torch.arange(num_candidates, device=scores.device, dtype=torch.long)
        if num_to_evict <= 0:
            return all_indices.expand(*leading_shape, num_candidates)

        selection_scores = scores if evict_highest else -scores
        selection_scores = selection_scores.masked_fill(~evictable_mask, -torch.inf)
        _, evicted = torch.topk(selection_scores, k=int(num_to_evict), dim=-1)

        keep_mask = torch.ones_like(evictable_mask, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=evicted, value=False)
        expanded_indices = all_indices.expand(*leading_shape, num_candidates)
        return expanded_indices[keep_mask].reshape(*leading_shape, keep_count)

    def _get_leverage_right_sketch(
        self,
        embed_dim: int,
        sketch_dim: int,
        *,
        device: torch.device,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        active_seed = self.leverage_random_seed if seed is None else int(seed)
        key = (str(device), embed_dim, sketch_dim, active_seed)
        omega = self._leverage_right_sketch_cache.get(key)
        if omega is not None:
            return omega

        generator = torch.Generator()
        generator.manual_seed(active_seed)
        omega = torch.randn(
            embed_dim,
            sketch_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(device=device)
        omega = omega / math.sqrt(float(sketch_dim))
        self._leverage_right_sketch_cache[key] = omega
        return omega

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (int(value) - 1).bit_length()

    def _resolve_leverage_approx_method(self, sketch_dim: Optional[int]) -> str:
        if self.leverage_approx_method in ("exact_qr", "full_d_ridge", "right_sketch_ridge"):
            return self.leverage_approx_method
        if sketch_dim in (None, 0):
            return "exact_qr"
        return "right_sketch"

    def _resolve_ridge_sketch_dim(self, feature_dim: int, num_tokens: int) -> int:
        requested = self.leverage_ridge_dim
        if requested is None or int(requested) < 1:
            raise ValueError("right_sketch_ridge requires leverage_ridge_dim >= 1")
        return min(int(requested), int(feature_dim), int(num_tokens))

    def _empty_leverage_basis(self, mat: torch.Tensor, granularity: str, *, basis_kind: str) -> SvdLeverageBasis:
        prefix = mat.shape[:-2]
        num_tokens = int(mat.shape[-2])
        return SvdLeverageBasis(
            q=torch.empty(*prefix, num_tokens, 0, device=mat.device, dtype=torch.float32),
            r_diag=torch.empty(*prefix, 0, device=mat.device, dtype=torch.float32),
            granularity=granularity,
            basis_kind=basis_kind,
        )

    def _profile_finish(
        self,
        profile: Dict[str, float],
        total_start: float,
        tensor: torch.Tensor,
        *,
        do_profile: bool,
    ) -> None:
        if do_profile:
            self._sync_for_timing(tensor)
            profile["total"] = time.perf_counter() - total_start
            self._last_leverage_profile = profile

    @staticmethod
    def _sync_for_timing(tensor: torch.Tensor) -> None:
        if tensor.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(tensor.device)

    @staticmethod
    def _profile_add(profile: Dict[str, float], name: str, elapsed: float) -> None:
        profile[name] = profile.get(name, 0.0) + elapsed

    def _profile_normalize_rows(
        self,
        x: torch.Tensor,
        profile: Optional[Dict[str, float]],
        name: str,
    ) -> torch.Tensor:
        if not self.leverage_normalize_rows:
            return x
        do_profile = profile is not None and self.profile
        start = time.perf_counter() if do_profile else 0.0
        normalized = F.normalize(x, p=2, dim=-1, eps=1e-12)
        if do_profile:
            self._sync_for_timing(normalized)
            elapsed = time.perf_counter() - start
            self._profile_add(profile, name, elapsed)
            if name != "normalization":
                self._profile_add(profile, "normalization", elapsed)
        return normalized

    def _profile_normalize_layer_key_before_projection(
        self,
        mat_k: torch.Tensor,
        profile: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        if not self.leverage_normalize_before_projection:
            return mat_k
        do_profile = profile is not None and self.profile
        start = time.perf_counter() if do_profile else 0.0
        normalized = self._normalize_layer_key_before_projection(mat_k)
        if do_profile:
            self._sync_for_timing(normalized)
            elapsed = time.perf_counter() - start
            self._profile_add(profile, "pre_projection_normalization", elapsed)
            self._profile_add(profile, "normalization", elapsed)
        return normalized

    def _maybe_normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        if not self.leverage_normalize_rows:
            return x
        return F.normalize(x, p=2, dim=-1, eps=1e-12)

    @staticmethod
    def _normalizes_before_projection(method: str) -> bool:
        return method in ("exact_qr", "full_d_ridge")

    def _needs_layer_covariance_pr(self) -> bool:
        return self.layer_budget_strategy in COVARIANCE_LAYER_BUDGET_STRATEGIES

    def _resolve_rls_frame_idx(self) -> int:
        for key in ("current_frame_idx", "step_idx"):
            value = self._leverage_diag_context.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _rls_cache_meta_for(
        self,
        mat: torch.Tensor,
        work: torch.Tensor,
        *,
        granularity: str,
        basis_kind: str,
        sketch_dim: Optional[int],
    ) -> Dict[str, object]:
        device = work.device
        return {
            "leading_shape": tuple(int(x) for x in work.shape[:-2]),
            "feature_dim": int(work.shape[-1]),
            "device_type": device.type,
            "device_index": device.index,
            "source_dtype": str(mat.dtype),
            "work_dtype": str(work.dtype),
            "granularity": granularity,
            "basis_kind": basis_kind,
            "sketch_dim": None if sketch_dim is None else int(sketch_dim),
            "ridge_lambda": float(self.leverage_ridge_lambda),
            "ridge_lambda_mode": self.leverage_ridge_lambda_mode,
            "ridge_jitter": float(self.leverage_ridge_jitter),
        }

    def _should_refresh_rls_cache(self, frame_idx: int, meta: Dict[str, object]) -> bool:
        if self.cached_rls_meta != meta:
            return True
        if self.cached_ktk is None or self.cached_rls_lam is None:
            return True
        if self.cached_rls_chol is None and self.cached_rls_inv is None:
            return True
        interval = int(self.rls_refresh_interval)
        if int(frame_idx) % interval == 0:
            return True
        if self.last_rls_refresh_frame is None:
            return True
        return int(frame_idx) - int(self.last_rls_refresh_frame) >= interval

    def _set_last_layer_covariance_pr_from_gram(self, gram: torch.Tensor) -> None:
        if self._needs_layer_covariance_pr():
            self._last_layer_covariance_pr = _covariance_participation_ratio(gram, self.layer_budget_eps).detach()

    def _set_last_layer_covariance_pr_from_matrix(self, mat: torch.Tensor) -> None:
        if self._needs_layer_covariance_pr():
            gram = mat.transpose(-2, -1) @ mat
            self._set_last_layer_covariance_pr_from_gram(gram)

    def _set_leverage_diag_context(
        self,
        *,
        layer_id: Optional[int],
        step_idx: Optional[int],
        current_frame_idx: Optional[int],
        granularity: str,
        batch_size: int,
        num_heads: int,
    ) -> None:
        self._leverage_diag_context = {
            "layer_id": layer_id,
            "step_idx": step_idx,
            "current_frame_idx": current_frame_idx,
            "granularity": granularity,
            "batch_size": int(batch_size),
            "num_heads": int(num_heads),
        }

    def _should_emit_leverage_diag(self) -> bool:
        if not self.leverage_diag:
            return False
        interval = int(self.leverage_diag_interval)
        if interval == 0:
            if self._leverage_diag_emitted_once:
                return False
            self._leverage_diag_emitted_once = True
            return True
        step_idx = self._leverage_diag_context.get("step_idx")
        if step_idx is None:
            return False
        try:
            return int(step_idx) % interval == 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_ratio_for_diag(numer: float, denom: float, eps: float = 1e-12) -> float:
        if not math.isfinite(numer):
            return float("nan")
        if not math.isfinite(denom) or abs(denom) <= eps:
            return float("inf") if numer >= 0.0 else float("-inf")
        return numer / denom

    @staticmethod
    def _fmt_diag_float(value: float) -> str:
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"

    def _diag_slice_labels(self, flat_idx: int, leading_shape: tuple[int, ...], granularity: str) -> tuple[int, str]:
        if not leading_shape:
            return 0, "all"
        if granularity == "head" and len(leading_shape) >= 2:
            heads = int(leading_shape[-1])
            batch = int(flat_idx // max(heads, 1))
            head = int(flat_idx % max(heads, 1))
            return batch, str(head)
        return int(flat_idx), "all"

    def _maybe_print_leverage_diag(
        self,
        *,
        scores: torch.Tensor,
        gram: torch.Tensor,
        lam: torch.Tensor,
        granularity: str,
        basis_kind: str,
        feature_dim: int,
        sketch_dim: Optional[int],
    ) -> None:
        if not self._should_emit_leverage_diag():
            return
        with torch.no_grad():
            if scores.numel() == 0 or gram.numel() == 0:
                return
            num_tokens = int(scores.shape[-1])
            d_sketch = int(sketch_dim) if sketch_dim is not None else int(feature_dim)
            leading_shape = tuple(int(x) for x in scores.shape[:-1])
            scores_flat = scores.detach().float().reshape(-1, num_tokens)
            gram_flat = gram.detach().float().reshape(-1, int(feature_dim), int(feature_dim))
            lam_flat = lam.detach().float().reshape(-1)
            q_levels = torch.tensor(
                [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
                device=scores.device,
                dtype=torch.float32,
            )
            layer_id = self._leverage_diag_context.get("layer_id")
            step_idx = self._leverage_diag_context.get("step_idx")
            for flat_idx in range(scores_flat.shape[0]):
                score_i = torch.nan_to_num(scores_flat[flat_idx], nan=0.0, posinf=0.0, neginf=0.0)
                gram_i = torch.nan_to_num(gram_flat[flat_idx], nan=0.0, posinf=0.0, neginf=0.0)
                eig = torch.linalg.eigvalsh(gram_i)
                eig = torch.nan_to_num(eig, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                lam_i = float(torch.nan_to_num(lam_flat[flat_idx], nan=0.0, posinf=0.0, neginf=0.0).item())
                score_min = float(score_i.min().item())
                score_max = float(score_i.max().item())
                score_mean = float(score_i.mean().item())
                score_std = float(score_i.std(unbiased=False).item())
                score_sum = float(score_i.sum().item())
                cv = self._safe_ratio_for_diag(score_std, score_mean)
                quantiles = torch.quantile(score_i, q_levels).detach().cpu().tolist()
                eig_min = float(eig.min().item()) if eig.numel() > 0 else 0.0
                eig_median = float(torch.quantile(eig, 0.5).item()) if eig.numel() > 0 else 0.0
                eig_mean = float(eig.mean().item()) if eig.numel() > 0 else 0.0
                eig_max = float(eig.max().item()) if eig.numel() > 0 else 0.0
                effective_dim_ratio = self._safe_ratio_for_diag(score_sum, float(max(d_sketch, 1)))
                lambda_over_mean = self._safe_ratio_for_diag(lam_i, eig_mean)
                lambda_over_median = self._safe_ratio_for_diag(lam_i, eig_median)
                base_lam = lam_i if lam_i != 0.0 else 1e-8
                sweep_lams = [base_lam * factor for factor in (0.01, 0.1, 1.0, 10.0, 100.0)]
                sweep = []
                for sweep_lam in sweep_lams:
                    sweep_tensor = eig / (eig + float(sweep_lam)).clamp_min(1e-30)
                    sweep.append((sweep_lam, float(sweep_tensor.sum().item())))
                batch_label, head_label = self._diag_slice_labels(flat_idx, leading_shape, granularity)
                q_text = " / ".join(self._fmt_diag_float(float(v)) for v in quantiles)
                lines = [
                    (
                        f"[LeverageDiag] layer={layer_id} batch={batch_label} head={head_label} "
                        f"step={step_idx} granularity={granularity} method={basis_kind}"
                    ),
                    f"N={num_tokens}, D_sketch={d_sketch}",
                    (
                        "score min/mean/max="
                        f"{self._fmt_diag_float(score_min)} / {self._fmt_diag_float(score_mean)} / "
                        f"{self._fmt_diag_float(score_max)}"
                    ),
                    f"score std={self._fmt_diag_float(score_std)}, cv={self._fmt_diag_float(cv)}",
                    (
                        "score sum effective_dim="
                        f"{self._fmt_diag_float(score_sum)}, effective_dim_ratio="
                        f"{self._fmt_diag_float(effective_dim_ratio)}"
                    ),
                    f"quantiles p01/p05/p10/p25/p50/p75/p90/p95/p99={q_text}",
                    (
                        "eig min/median/mean/max="
                        f"{self._fmt_diag_float(eig_min)} / {self._fmt_diag_float(eig_median)} / "
                        f"{self._fmt_diag_float(eig_mean)} / {self._fmt_diag_float(eig_max)}"
                    ),
                    (
                        f"lambda={self._fmt_diag_float(lam_i)}, "
                        f"lambda/mean_eig={self._fmt_diag_float(lambda_over_mean)}, "
                        f"lambda/median_eig={self._fmt_diag_float(lambda_over_median)}"
                    ),
                    "deff sweep:",
                ]
                lines.extend(
                    f"  lam={self._fmt_diag_float(sweep_lam)} -> deff={self._fmt_diag_float(deff)}"
                    for sweep_lam, deff in sweep
                )
                print("\n".join(lines))

    def _ridge_leverage_scores_from_matrix(
        self,
        mat: torch.Tensor,
        *,
        return_basis: bool,
        granularity: str,
        profile: Dict[str, float],
        total_start: float,
        do_profile: bool,
        basis_kind: str = "full_d_ridge",
        sketch_dim: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        num_tokens = int(mat.shape[-2])
        feature_dim = int(mat.shape[-1])
        frame_idx = self._resolve_rls_frame_idx()
        profile["method"] = basis_kind
        profile["N"] = float(num_tokens)
        profile.setdefault("D", float(feature_dim))
        profile["rls_refresh_interval"] = float(self.rls_refresh_interval)
        profile["rls_frame_idx"] = float(frame_idx)
        if sketch_dim is not None:
            profile["sketch_dim"] = float(sketch_dim)

        with torch.cuda.amp.autocast(enabled=False):
            work = torch.nan_to_num(mat.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            eye = torch.eye(feature_dim, device=work.device, dtype=torch.float32)
            eye = eye.view(*((1,) * (work.ndim - 2)), feature_dim, feature_dim)
            meta = self._rls_cache_meta_for(
                mat,
                work,
                granularity=granularity,
                basis_kind=basis_kind,
                sketch_dim=sketch_dim,
            )
            refresh = self._should_refresh_rls_cache(frame_idx, meta)

            # rls_refresh_interval controls how often the key covariance inverse used for ridge leverage
            # score estimation is refreshed. When set to 1, the covariance inverse is recomputed every
            # frame. When set to r > 1, the inverse from the last refresh frame is reused for the
            # intermediate frames, reducing the cost of repeated K^T K and inverse computations at the
            # expense of using a slightly stale RLS estimator. The current stream API carries one
            # current_frame_idx per call, so mixed video streams in one batch share this refresh cadence.
            gram = None
            lam = None
            chol = None
            inv_a = None
            fallback_state = 0.0
            retries = 0
            if refresh:
                gram_start = time.perf_counter() if do_profile else 0.0
                gram = work.transpose(-2, -1) @ work
                self._set_last_layer_covariance_pr_from_gram(gram)
                if do_profile:
                    self._sync_for_timing(gram)
                    profile["gram"] = time.perf_counter() - gram_start

                trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1)
                scale = trace / float(max(feature_dim, 1))
                eps = torch.finfo(torch.float32).eps
                if self.leverage_ridge_lambda_mode == "absolute":
                    lam = torch.full_like(scale, float(self.leverage_ridge_lambda))
                else:
                    lam = float(self.leverage_ridge_lambda) * scale.clamp_min(eps)
                jitter_scale = torch.ones_like(scale)

                chol_start = time.perf_counter() if do_profile else 0.0
                last_a = None
                for retry_idx, jitter_multiplier in enumerate((1.0, 10.0, 100.0, 1000.0)):
                    diag_add = lam + float(self.leverage_ridge_jitter) * float(jitter_multiplier) * jitter_scale
                    last_a = gram + diag_add.unsqueeze(-1).unsqueeze(-1) * eye
                    chol, info = torch.linalg.cholesky_ex(last_a)
                    if not bool((info != 0).any().item()):
                        retries = retry_idx
                        break
                    try:
                        info_max = int(info.detach().max().item())
                        diag_mean = float(torch.nan_to_num(diag_add.detach().float()).mean().item())
                    except RuntimeError:
                        info_max = -1
                        diag_mean = float("nan")
                    print(
                        "[RLS][cholesky_retry] "
                        f"method={basis_kind} granularity={granularity} frame={frame_idx} "
                        f"N={num_tokens} D={feature_dim} retry={retry_idx} "
                        f"jitter_multiplier={jitter_multiplier:g} diag_add_mean={diag_mean:.6g} "
                        f"info_max={info_max}"
                    )
                    chol = None
                    retries = retry_idx + 1
                if chol is None and last_a is not None:
                    print(
                        "[RLS][pinv_fallback] "
                        f"method={basis_kind} granularity={granularity} frame={frame_idx} "
                        f"N={num_tokens} D={feature_dim} retries={retries}"
                    )
                    try:
                        inv_a = torch.linalg.pinv(last_a)
                        fallback_state = 1.0
                    except RuntimeError as exc:
                        inv_a = None
                        fallback_state = 2.0
                        print(
                            "[RLS][pinv_failed] "
                            f"method={basis_kind} granularity={granularity} frame={frame_idx} "
                            f"N={num_tokens} D={feature_dim} error={exc}"
                        )
                if do_profile:
                    if chol is not None:
                        self._sync_for_timing(chol)
                    elif inv_a is not None:
                        self._sync_for_timing(inv_a)
                    elif last_a is not None:
                        self._sync_for_timing(last_a)
                    profile["cholesky"] = time.perf_counter() - chol_start
                    profile["lambda_value"] = float(torch.nan_to_num(lam.detach().float()).mean().item())
                    profile["cholesky_retries"] = float(retries)

                if chol is not None:
                    inverse_start = time.perf_counter() if do_profile else 0.0
                    try:
                        inv_a = torch.cholesky_inverse(chol)
                    except RuntimeError:
                        inv_a = None
                    if do_profile:
                        sync_tensor = inv_a if inv_a is not None else chol
                        self._sync_for_timing(sync_tensor)
                        profile["inverse_build"] = time.perf_counter() - inverse_start

                if chol is not None or inv_a is not None:
                    self.cached_ktk = gram.detach()
                    self.cached_rls_chol = chol.detach() if chol is not None else None
                    self.cached_rls_inv = inv_a.detach() if inv_a is not None else None
                    self.cached_rls_lam = lam.detach()
                    self.cached_rls_meta = dict(meta)
                    self.last_rls_refresh_frame = int(frame_idx)
                else:
                    self.cached_ktk = None
                    self.cached_rls_chol = None
                    self.cached_rls_inv = None
                    self.cached_rls_lam = None
                    self.cached_rls_meta = None
                    self.last_rls_refresh_frame = None
                self.rls_refresh_count += 1
            else:
                gram = self.cached_ktk
                lam = self.cached_rls_lam
                chol = self.cached_rls_chol
                inv_a = self.cached_rls_inv
                self.rls_cache_hit_count += 1
                if gram is None or lam is None or (chol is None and inv_a is None):
                    raise RuntimeError("RLS cache was selected for reuse but no valid cached factorization exists")
                self._set_last_layer_covariance_pr_from_gram(gram)
                if chol is None:
                    fallback_state = 1.0
                if do_profile:
                    profile["gram"] = 0.0
                    profile["cholesky"] = 0.0
                    profile["inverse_build"] = 0.0
                    profile["lambda_value"] = float(torch.nan_to_num(lam.detach().float()).mean().item())
                    profile["cholesky_retries"] = 0.0

            profile["rls_cache_refreshed"] = 1.0 if refresh else 0.0
            profile["rls_refresh_frame"] = float(-1 if self.last_rls_refresh_frame is None else self.last_rls_refresh_frame)
            profile["rls_refresh_count"] = float(self.rls_refresh_count)
            profile["rls_cache_hit_count"] = float(self.rls_cache_hit_count)

            score_start = time.perf_counter() if do_profile else 0.0
            if inv_a is not None:
                transformed = work @ inv_a
                scores = (transformed * work).sum(dim=-1)
                fallback = fallback_state
                score_backend = "pinv" if chol is None else "inverse"
            elif chol is not None:
                chunks = []
                chunk_size = int(self.leverage_ridge_score_chunk_size)
                for start in range(0, num_tokens, chunk_size):
                    end = min(start + chunk_size, num_tokens)
                    rhs = work[..., start:end, :].transpose(-2, -1).contiguous()
                    solved = torch.cholesky_solve(rhs, chol)
                    chunks.append((rhs * solved).sum(dim=-2))
                scores = torch.cat(chunks, dim=-1) if chunks else torch.empty(*work.shape[:-1], device=work.device, dtype=torch.float32)
                fallback = 0.0
                score_backend = "chol_solve"
            else:
                scores = work.square().sum(dim=-1)
                fallback = max(fallback_state, 2.0)
                score_backend = "norm_fallback"
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            self._maybe_print_leverage_diag(
                scores=scores,
                gram=gram,
                lam=lam,
                granularity=granularity,
                basis_kind=basis_kind,
                feature_dim=feature_dim,
                sketch_dim=sketch_dim,
            )
            if do_profile:
                self._sync_for_timing(scores)
                profile["score_solve"] = time.perf_counter() - score_start
                profile["scoring"] = profile["score_solve"]
                profile["fallback"] = fallback
                profile["score_backend"] = score_backend
                profile["score_min"] = float(scores.min().item()) if scores.numel() > 0 else 0.0
                profile["score_max"] = float(scores.max().item()) if scores.numel() > 0 else 0.0
                profile["score_mean"] = float(scores.mean().item()) if scores.numel() > 0 else 0.0

        self._profile_finish(profile, total_start, scores, do_profile=do_profile)
        if return_basis:
            return scores, self._empty_leverage_basis(mat, granularity, basis_kind=basis_kind)
        return scores

    def _right_sketch_ridge_leverage_scores_from_matrix(
        self,
        mat: torch.Tensor,
        *,
        return_basis: bool,
        granularity: str,
        profile: Dict[str, float],
        total_start: float,
        do_profile: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        num_tokens = int(mat.shape[-2])
        feature_dim = int(mat.shape[-1])
        sketch_dim = self._resolve_ridge_sketch_dim(feature_dim, num_tokens)
        profile["D"] = float(feature_dim)
        profile["sketch_dim"] = float(sketch_dim)
        with torch.cuda.amp.autocast(enabled=False):
            sketch_retrieval_start = time.perf_counter() if do_profile else 0.0
            omega = self._get_leverage_right_sketch(
                feature_dim,
                sketch_dim,
                device=mat.device,
                seed=self.leverage_random_seed,
            )
            if do_profile:
                self._sync_for_timing(omega)
                profile["sketch_matrix_retrieval"] = time.perf_counter() - sketch_retrieval_start
            projection_start = time.perf_counter() if do_profile else 0.0
            projected = mat @ omega
            if do_profile:
                self._sync_for_timing(projected)
                profile["projection_matmul"] = time.perf_counter() - projection_start
            self._capture_projected_pre_normalization_norms(projected)
            projected = self._profile_normalize_rows(projected, profile if do_profile else None, "post_projection_normalization")
            self._store_projected_similarity_features(projected, granularity)
            if do_profile:
                profile["sketch"] = profile["sketch_matrix_retrieval"] + profile["projection_matmul"]
        return self._ridge_leverage_scores_from_matrix(
            projected,
            return_basis=return_basis,
            granularity=granularity,
            profile=profile,
            total_start=total_start,
            do_profile=do_profile,
            basis_kind="right_sketch_ridge",
            sketch_dim=sketch_dim,
        )

    def _qr_leverage_scores_from_matrix(
        self,
        mat: torch.Tensor,
        active_sketch_dim: Optional[int],
        eps: float,
        *,
        return_basis: bool,
        granularity: str,
        profile: Dict[str, float],
        total_start: float,
        do_profile: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        num_tokens = int(mat.shape[-2])
        feature_dim = int(mat.shape[-1])
        method = self._resolve_leverage_approx_method(active_sketch_dim)
        if method == "full_d_ridge":
            return self._ridge_leverage_scores_from_matrix(
                mat,
                return_basis=return_basis,
                granularity=granularity,
                profile=profile,
                total_start=total_start,
                do_profile=do_profile,
                basis_kind="full_d_ridge",
            )
        if method == "right_sketch_ridge":
            return self._right_sketch_ridge_leverage_scores_from_matrix(
                mat,
                return_basis=return_basis,
                granularity=granularity,
                profile=profile,
                total_start=total_start,
                do_profile=do_profile,
            )
        with torch.cuda.amp.autocast(enabled=False):
            if method == "exact_qr":
                leverage_matrix = mat
                if do_profile:
                    profile["sketch"] = 0.0
            else:
                # Right-sketched leverage / Compactor-style approximation.
                # QR is applied to K Phi, so scores are leverage scores of the
                # compressed key matrix.
                sketch_dim = min(int(active_sketch_dim), feature_dim, num_tokens)
                sketch_retrieval_start = time.perf_counter() if do_profile else 0.0
                omega = self._get_leverage_right_sketch(
                    feature_dim,
                    sketch_dim,
                    device=mat.device,
                    seed=self.leverage_random_seed,
                )
                if do_profile:
                    self._sync_for_timing(omega)
                    profile["sketch_matrix_retrieval"] = time.perf_counter() - sketch_retrieval_start
                projection_start = time.perf_counter() if do_profile else 0.0
                leverage_matrix = mat @ omega
                if do_profile:
                    self._sync_for_timing(leverage_matrix)
                    profile["projection_matmul"] = time.perf_counter() - projection_start
                self._capture_projected_pre_normalization_norms(leverage_matrix)
                leverage_matrix = self._profile_normalize_rows(leverage_matrix, profile if do_profile else None, "post_projection_normalization")
                self._store_projected_similarity_features(leverage_matrix, granularity)
                if do_profile:
                    profile["sketch"] = profile["sketch_matrix_retrieval"] + profile["projection_matmul"]

            self._set_last_layer_covariance_pr_from_matrix(leverage_matrix)
            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = torch.nan_to_num(leverage_matrix.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
                profile["fallback"] = 1.0
                self._profile_finish(profile, total_start, scores, do_profile=do_profile)
                if return_basis:
                    return scores, self._empty_leverage_basis(mat, granularity, basis_kind=method)
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        max_diag = diag.max(dim=-1, keepdim=True).values.clamp_min(float(eps)) if diag.ndim > 1 else diag.max().clamp_min(float(eps))
        active = (diag > max_diag * float(eps)).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(-2)).sum(dim=-1)
        scores_sq = torch.nan_to_num(scores_sq, nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["fallback"] = profile.get("fallback", 0.0)
        self._profile_finish(profile, total_start, scores_sq, do_profile=do_profile)
        if return_basis:
            return scores_sq, SvdLeverageBasis(
                q=q.detach(),
                r_diag=diag.detach(),
                granularity=granularity,
                basis_kind=method,
            )
        return scores_sq

    def compute_svd_leverage_scores(
        self,
        x: torch.Tensor,
        sketch_dim: Optional[int] = None,
        eps: float = 1e-6,
        *,
        return_basis: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        """Compute row leverage scores for a 2D token-feature matrix."""
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D token-feature matrix, got shape {tuple(x.shape)}")
        num_tokens, feature_dim = x.shape
        if num_tokens <= 0:
            scores = torch.empty(0, device=x.device, dtype=torch.float32)
            if return_basis:
                return scores, SvdLeverageBasis(
                    q=torch.empty(0, 0, device=x.device, dtype=torch.float32),
                    r_diag=torch.empty(0, device=x.device, dtype=torch.float32),
                    granularity="layer",
                )
            return scores
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0 for SVD leverage, got {feature_dim}")

        active_sketch_dim = self.leverage_sketch_dim if sketch_dim is None else sketch_dim
        profile: Dict[str, float] = {}
        do_profile = self.profile

        if do_profile:
            self._sync_for_timing(x)
        total_start = time.perf_counter() if do_profile else 0.0
        prep_start = time.perf_counter() if do_profile else 0.0
        with torch.cuda.amp.autocast(enabled=False):
            mat = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            method = self._resolve_leverage_approx_method(active_sketch_dim)
            if self._normalizes_before_projection(method):
                mat = self._profile_normalize_rows(mat, profile if do_profile else None, "pre_score_normalization")
        if do_profile:
            self._sync_for_timing(mat)
            profile["candidate_matrix_preparation"] = time.perf_counter() - prep_start
        return self._qr_leverage_scores_from_matrix(
            mat,
            active_sketch_dim,
            eps,
            return_basis=return_basis,
            granularity="layer",
            profile=profile,
            total_start=total_start,
            do_profile=do_profile,
        )

    def _svd_leverage_scores(
        self,
        candidate_k: torch.Tensor,
        *,
        return_basis: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        """Compute row leverage scores per batch/head."""
        B, H, N, D = candidate_k.shape
        if N <= 0:
            scores = torch.empty(B, H, 0, device=candidate_k.device, dtype=torch.float32)
            if return_basis:
                return scores, SvdLeverageBasis(
                    q=torch.empty(B, H, 0, 0, device=candidate_k.device, dtype=torch.float32),
                    r_diag=torch.empty(B, H, 0, device=candidate_k.device, dtype=torch.float32),
                    granularity="head",
                )
            return scores

        profile: Dict[str, float] = {}
        do_profile = self.profile
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0
        prep_start = time.perf_counter() if do_profile else 0.0
        with torch.cuda.amp.autocast(enabled=False):
            mat = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            method = self._resolve_leverage_approx_method(self.leverage_sketch_dim)
            if self._normalizes_before_projection(method):
                mat = self._profile_normalize_rows(mat, profile if do_profile else None, "pre_score_normalization")
        if do_profile:
            self._sync_for_timing(mat)
            profile["candidate_matrix_preparation"] = time.perf_counter() - prep_start
        return self._qr_leverage_scores_from_matrix(
            mat,
            self.leverage_sketch_dim,
            1e-6,
            return_basis=return_basis,
            granularity="head",
            profile=profile,
            total_start=total_start,
            do_profile=do_profile,
        )

    def _layer_svd_leverage_scores(
        self,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor] = None,
        *,
        return_basis: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        """Compute one leverage-score vector per batch by concatenating heads."""
        B, H, N, D = candidate_k.shape
        if N <= 0:
            scores = torch.empty(B, 0, device=candidate_k.device, dtype=torch.float32)
            if return_basis:
                return scores, SvdLeverageBasis(
                    q=torch.empty(B, 0, 0, device=candidate_k.device, dtype=torch.float32),
                    r_diag=torch.empty(B, 0, device=candidate_k.device, dtype=torch.float32),
                    granularity="layer",
                )
            return scores
        if D <= 0:
            raise ValueError(f"head_dim must be > 0 for layer-wise SVD leverage, got {D}")
        if self.leverage_projection == "head_mean":
            return self._layer_svd_leverage_scores_head_mean(candidate_k, return_basis=return_basis)
        uses_value_feature = self.leverage_feature in KV_LEVERAGE_FEATURES
        feature_dim = H * D * (2 if self.leverage_feature == "key_value" else 1)
        self._last_layer_feature_shape = (int(N), int(feature_dim))
        if uses_value_feature:
            if candidate_v is None:
                raise ValueError(f"leverage_feature={self.leverage_feature!r} requires value cache tensor")
            if candidate_v.shape != candidate_k.shape:
                raise ValueError(
                    "candidate_v must match candidate_k for key/value leverage, "
                    f"got {tuple(candidate_v.shape)} vs {tuple(candidate_k.shape)}"
                )

        if (
            self.leverage_feature == "key_value_lowdim_concat"
            or self._resolve_leverage_approx_method(self.leverage_sketch_dim) in ("right_sketch", "right_sketch_ridge")
        ):
            return self._layer_svd_leverage_scores_sketched(
                candidate_k,
                candidate_v,
                feature_dim,
                return_basis=return_basis,
            )

        scores = []
        covariance_prs = []
        qs = []
        diags = []
        basis_kind = self._resolve_leverage_approx_method(self.leverage_sketch_dim)
        aggregate_profile: Dict[str, float] = {
            "feature": 0.0,
            "sketch": 0.0,
            "qr": 0.0,
            "scoring": 0.0,
            "total": 0.0,
            "fallback": 0.0,
        }
        if self.leverage_normalize_before_projection:
            candidate_k = self._profile_normalize_layer_key_before_projection(
                torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0),
                aggregate_profile,
            )

        for batch_idx in range(B):
            feature_start = time.perf_counter() if self.profile else 0.0
            x_key = candidate_k[batch_idx].transpose(0, 1).reshape(N, H * D)
            if self.leverage_feature in KV_LEVERAGE_FEATURES:
                assert candidate_v is not None
                x_value = candidate_v[batch_idx].transpose(0, 1).reshape(N, H * D)
                x_layer = torch.cat([x_key, x_value], dim=-1)
            else:
                x_layer = x_key
            if self.profile:
                self._sync_for_timing(x_layer)
            feature_time = time.perf_counter() - feature_start if self.profile else 0.0
            if return_basis:
                score, basis = self.compute_svd_leverage_scores(
                    x_layer,
                    self.leverage_sketch_dim,
                    return_basis=True,
                )
                qs.append(basis.q)
                diags.append(basis.r_diag)
                basis_kind = basis.basis_kind
            else:
                score = self.compute_svd_leverage_scores(x_layer, self.leverage_sketch_dim)
            scores.append(score)
            if self._needs_layer_covariance_pr() and self._last_layer_covariance_pr is not None:
                covariance_prs.append(self._last_layer_covariance_pr.reshape(()))
            if self.profile:
                aggregate_profile["feature"] += feature_time
                for name, value in self._last_leverage_profile.items():
                    if isinstance(value, (int, float)):
                        aggregate_profile[name] = aggregate_profile.get(name, 0.0) + value
                    else:
                        aggregate_profile[name] = value
        if self.profile:
            self._last_leverage_profile = aggregate_profile
        if covariance_prs:
            self._last_layer_covariance_pr = torch.stack(covariance_prs, dim=0).detach()
        stacked = torch.stack(scores, dim=0)
        if return_basis:
            return stacked, SvdLeverageBasis(
                q=torch.stack(qs, dim=0),
                r_diag=torch.stack(diags, dim=0),
                granularity="layer",
                basis_kind=basis_kind,
            )
        return stacked

    def _layer_svd_leverage_scores_head_mean(
        self,
        candidate_k: torch.Tensor,
        *,
        return_basis: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        """Layer-wise leverage from deterministic per-head mean features."""
        B, H, N, D = candidate_k.shape
        if self.leverage_head_mean_dim > D:
            raise ValueError(
                "leverage_head_mean_dim must be <= head_dim for head_mean projection, "
                f"got {self.leverage_head_mean_dim} > {D}"
            )
        feature_dim = H * self.leverage_head_mean_dim
        self._last_layer_feature_shape = (int(N), int(feature_dim))

        profile: Dict[str, float] = {}
        do_profile = self.profile
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0

        with torch.cuda.amp.autocast(enabled=False):
            feature_start = time.perf_counter() if do_profile else 0.0
            mat_k = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            head_chunks = torch.tensor_split(mat_k, self.leverage_head_mean_dim, dim=-1)
            head_features = torch.stack([chunk.mean(dim=-1) for chunk in head_chunks], dim=-1)
            leverage_matrix = head_features.permute(0, 2, 1, 3).reshape(B, N, feature_dim).contiguous()
            method = self._resolve_leverage_approx_method(self.leverage_sketch_dim)
            if self._normalizes_before_projection(method):
                leverage_matrix = self._profile_normalize_rows(leverage_matrix, profile if do_profile else None, "pre_score_normalization")
            if do_profile:
                self._sync_for_timing(leverage_matrix)
                profile["feature"] = time.perf_counter() - feature_start
                profile["sketch"] = 0.0

            if method == "full_d_ridge":
                return self._ridge_leverage_scores_from_matrix(
                    leverage_matrix,
                    return_basis=return_basis,
                    granularity="layer",
                    profile=profile,
                    total_start=total_start,
                    do_profile=do_profile,
                    basis_kind="full_d_ridge",
                )
            if method == "right_sketch_ridge":
                return self._right_sketch_ridge_leverage_scores_from_matrix(
                    leverage_matrix,
                    return_basis=return_basis,
                    granularity="layer",
                    profile=profile,
                    total_start=total_start,
                    do_profile=do_profile,
                )

            self._set_last_layer_covariance_pr_from_matrix(leverage_matrix)
            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = torch.nan_to_num(leverage_matrix.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
                if do_profile:
                    self._sync_for_timing(scores)
                    profile["total"] = time.perf_counter() - total_start
                    self._last_leverage_profile = profile
                if return_basis:
                    return scores, SvdLeverageBasis(
                        q=torch.empty(B, N, 0, device=candidate_k.device, dtype=torch.float32),
                        r_diag=torch.empty(B, 0, device=candidate_k.device, dtype=torch.float32),
                        granularity="layer",
                    )
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        max_diag = diag.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        active = (diag > max_diag * 1e-6).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(1)).sum(dim=-1)
        scores_sq = torch.nan_to_num(scores_sq, nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["total"] = time.perf_counter() - total_start
            self._last_leverage_profile = profile
        if return_basis:
            return scores_sq, SvdLeverageBasis(q=q.detach(), r_diag=diag.detach(), granularity="layer")
        return scores_sq

    def _layer_svd_leverage_scores_sketched(
        self,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        feature_dim: int,
        *,
        return_basis: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        """Layer-wise sketched leverage without materializing ``[B, N, H * D]``."""
        B, H, N, D = candidate_k.shape
        lowdim_concat = self.leverage_feature == "key_value_lowdim_concat"
        method = self.leverage_approx_method if lowdim_concat else self._resolve_leverage_approx_method(self.leverage_sketch_dim)
        if method == "right_sketch_ridge" or lowdim_concat:
            sketch_dim = self._resolve_ridge_sketch_dim(H * D if lowdim_concat else feature_dim, N)
        else:
            sketch_dim = min(int(self.leverage_sketch_dim), int(feature_dim), int(N))
        if sketch_dim <= 0:
            scores = torch.empty(B, N, device=candidate_k.device, dtype=torch.float32)
            if return_basis:
                return scores, SvdLeverageBasis(
                    q=torch.empty(B, N, 0, device=candidate_k.device, dtype=torch.float32),
                    r_diag=torch.empty(B, 0, device=candidate_k.device, dtype=torch.float32),
                    granularity="layer",
                )
            return scores

        profile: Dict[str, float] = {}
        do_profile = self.profile
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0
        prep_start = time.perf_counter() if do_profile else 0.0

        with torch.cuda.amp.autocast(enabled=False):
            mat_k = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mat_v = (
                torch.nan_to_num(candidate_v.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if self.leverage_feature in KV_LEVERAGE_FEATURES and candidate_v is not None
                else None
            )
            if do_profile:
                self._sync_for_timing(mat_k)
                profile["candidate_matrix_preparation"] = time.perf_counter() - prep_start

            # Right-sketched leverage / Compactor-style approximation for the
            # concatenated layer feature matrix, applied without materializing
            # the full [B, N, H * D] matrix first.
            sketch_retrieval_start = time.perf_counter() if do_profile else 0.0
            omega = self._get_leverage_right_sketch(
                H * D if lowdim_concat else feature_dim,
                sketch_dim,
                device=mat_k.device,
                seed=self.leverage_random_seed,
            )
            omega_value = None
            if lowdim_concat:
                omega_value = self._get_leverage_right_sketch(
                    H * D,
                    sketch_dim,
                    device=mat_k.device,
                    seed=self.leverage_random_seed + 1,
                )
            if do_profile:
                sync_tensor = omega_value if omega_value is not None else omega
                self._sync_for_timing(sync_tensor)
                profile["sketch_matrix_retrieval"] = time.perf_counter() - sketch_retrieval_start
            omega_key = omega[: H * D].view(H, D, sketch_dim)
            need_head_features = self._needs_projected_head_features()
            if self.leverage_projected_key_cache and not lowdim_concat and mat_v is None:
                projection_start = time.perf_counter() if do_profile else 0.0
                leverage_matrix, head_leverage_matrix = self._layer_key_projected_features(
                    mat_k,
                    omega_key,
                    sketch_dim,
                    profile,
                )
                if do_profile and "projection_matmul" not in profile:
                    self._sync_for_timing(leverage_matrix)
                    profile["projection_matmul"] = time.perf_counter() - projection_start
            else:
                mat_k = self._profile_normalize_layer_key_before_projection(mat_k, profile if do_profile else None)
                projection_start = time.perf_counter() if do_profile else 0.0
                if not need_head_features and not lowdim_concat and mat_v is None:
                    flat_k = mat_k.permute(0, 2, 1, 3).reshape(B, N, H * D)
                    leverage_matrix = torch.matmul(flat_k, omega_key.reshape(H * D, sketch_dim))
                    head_leverage_matrix = None
                else:
                    head_key_matrix = torch.matmul(mat_k, omega_key.unsqueeze(0))
                    if lowdim_concat:
                        if mat_v is None or omega_value is None:
                            raise ValueError("leverage_feature='key_value_lowdim_concat' requires value cache tensor")
                        omega_value_view = omega_value.view(H, D, sketch_dim)
                        head_value_matrix = torch.matmul(mat_v, omega_value_view.unsqueeze(0))
                        head_leverage_matrix = torch.cat([head_key_matrix, head_value_matrix], dim=-1)
                        leverage_matrix = torch.cat([head_key_matrix.sum(dim=1), head_value_matrix.sum(dim=1)], dim=-1)
                        self._last_layer_feature_shape = (int(N), int(leverage_matrix.shape[-1]))
                    else:
                        head_leverage_matrix = head_key_matrix
                        if mat_v is not None:
                            omega_value_view = omega[H * D :].view(H, D, sketch_dim)
                            head_leverage_matrix = head_leverage_matrix + torch.matmul(mat_v, omega_value_view.unsqueeze(0))
                        leverage_matrix = head_leverage_matrix.sum(dim=1)
                if do_profile:
                    self._sync_for_timing(leverage_matrix)
                    profile["projection_matmul"] = time.perf_counter() - projection_start
                self._capture_projected_pre_normalization_norms(leverage_matrix)
                leverage_matrix = self._profile_normalize_rows(leverage_matrix, profile if do_profile else None, "post_projection_normalization")
                if head_leverage_matrix is not None:
                    head_leverage_matrix = self._profile_normalize_rows(head_leverage_matrix, profile if do_profile else None, "post_projection_normalization")
            self._store_projected_similarity_features(leverage_matrix, "layer")
            if head_leverage_matrix is not None:
                self._store_projected_similarity_features(head_leverage_matrix, "head")
            if do_profile:
                profile["feature"] = 0.0
                profile["sketch"] = profile["sketch_matrix_retrieval"] + profile["projection_matmul"]

            if method == "right_sketch_ridge":
                profile["D"] = float(leverage_matrix.shape[-1])
                profile["sketch_dim"] = float(sketch_dim)
                return self._ridge_leverage_scores_from_matrix(
                    leverage_matrix,
                    return_basis=return_basis,
                    granularity="layer",
                    profile=profile,
                    total_start=total_start,
                    do_profile=do_profile,
                    basis_kind="right_sketch_ridge",
                    sketch_dim=sketch_dim,
                )

            self._set_last_layer_covariance_pr_from_matrix(leverage_matrix)
            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = mat_k.square().sum(dim=(1, 3))
                if mat_v is not None:
                    scores = scores + mat_v.square().sum(dim=(1, 3))
                if self.leverage_normalize_rows:
                    scores = (scores > 1e-24).to(dtype=torch.float32)
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                if do_profile:
                    self._sync_for_timing(scores)
                    profile["total"] = time.perf_counter() - total_start
                    profile["fallback"] = 1.0
                    self._last_leverage_profile = profile
                if return_basis:
                    return scores, SvdLeverageBasis(
                        q=torch.empty(B, N, 0, device=candidate_k.device, dtype=torch.float32),
                        r_diag=torch.empty(B, 0, device=candidate_k.device, dtype=torch.float32),
                        granularity="layer",
                        basis_kind="right_sketch",
                    )
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        max_diag = diag.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        active = (diag > max_diag * 1e-6).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(1)).sum(dim=-1)
        scores_sq = torch.nan_to_num(scores_sq, nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["total"] = time.perf_counter() - total_start
            profile["fallback"] = profile.get("fallback", 0.0)
            self._last_leverage_profile = profile
        if return_basis:
            return scores_sq, SvdLeverageBasis(
                q=q.detach(),
                r_diag=diag.detach(),
                granularity="layer",
                basis_kind="right_sketch",
            )
        return scores_sq
