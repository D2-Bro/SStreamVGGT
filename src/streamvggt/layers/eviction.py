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
    "leverage_entropy",
    "value_weighted_leverage_pr",
)


LAYER_BUDGET_SCORE_STRATEGIES = (
    "leverage_pr",
    "leverage_entropy",
    "value_weighted_leverage_pr",
)


def _layer_score_leverage_pr(lev: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Effective count from the participation ratio of QR leverage mass."""
    if lev is None or lev.numel() == 0:
        device = lev.device if lev is not None else "cpu"
        return torch.tensor(0.0, device=device)
    lev = lev.float()
    lev = torch.nan_to_num(lev, nan=0.0, posinf=0.0, neginf=0.0)
    lev = torch.clamp(lev, min=0.0)
    sum_lev = lev.sum()
    if sum_lev <= eps:
        return torch.zeros((), device=lev.device, dtype=lev.dtype)
    return sum_lev.square() / lev.square().sum().clamp_min(eps)


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
    if eps <= 0:
        raise ValueError(f"layer budget eps must be > 0, got {eps}")

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
    VALID_LEVERAGE_APPROX_METHODS = (
        "exact_qr",
        "right_sketch",
        "drineas_srht",
        "full_d_ridge",
        "right_sketch_ridge",
    )
    VALID_LEVERAGE_EVICTION_SELECTORS = ("topk", "fast_dpp")

    def __init__(
        self,
        policy: str = "mean",
        debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_approx_method: str = "right_sketch",
        leverage_left_sketch_dim: Optional[int] = 2048,
        leverage_right_jl_dim: Optional[int] = 64,
        leverage_ridge_lambda: float = 1e-3,
        leverage_ridge_lambda_mode: str = "relative",
        leverage_ridge_score_chunk_size: int = 4096,
        leverage_ridge_jitter: float = 1e-6,
        leverage_ridge_dim: Optional[int] = None,
        leverage_random_seed: int = 0,
        leverage_eviction_selector: str = "topk",
        leverage_dpp_candidate_multiplier: int = 2,
        leverage_dpp_greedy_block_size: int = 32,
        leverage_dpp_diversity_beta: float = 1.0,
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.5,
        layer_budget_value_norm_type: str = "rms",
        layer_budget_norm_source: str = "value",
        layer_budget_eps: float = 1e-12,
    ) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Unknown eviction policy '{policy}'. Valid policies: {self.VALID_POLICIES}")
        if leverage_sketch_dim is not None and leverage_sketch_dim < 0:
            raise ValueError(f"leverage_sketch_dim must be >= 0 or None, got {leverage_sketch_dim}")
        if leverage_head_mean_dim < 1:
            raise ValueError(f"leverage_head_mean_dim must be >= 1, got {leverage_head_mean_dim}")
        if leverage_granularity not in ("head", "layer"):
            raise ValueError("leverage_granularity must be 'head' or 'layer', got " f"{leverage_granularity!r}")
        if leverage_feature not in ("key", "key_value"):
            raise ValueError("leverage_feature must be 'key' or 'key_value', got " f"{leverage_feature!r}")
        if leverage_projection not in ("random", "head_mean"):
            raise ValueError(
                "leverage_projection must be 'random' or 'head_mean', got "
                f"{leverage_projection!r}"
            )
        if leverage_projection == "head_mean" and leverage_granularity != "layer":
            raise ValueError("leverage_projection='head_mean' requires leverage_granularity='layer'")
        if leverage_projection == "head_mean" and leverage_feature != "key":
            raise ValueError("leverage_projection='head_mean' requires leverage_feature='key'")
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
        if leverage_dpp_diversity_beta < 0:
            raise ValueError(
                "leverage_dpp_diversity_beta must be >= 0, got "
                f"{leverage_dpp_diversity_beta}"
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
        if layer_budget_eps <= 0:
            raise ValueError(f"layer_budget_eps must be > 0, got {layer_budget_eps}")
        if leverage_left_sketch_dim is not None and leverage_left_sketch_dim <= 0:
            raise ValueError(
                "leverage_left_sketch_dim must be > 0 or None, got "
                f"{leverage_left_sketch_dim}"
            )
        if leverage_right_jl_dim is not None and leverage_right_jl_dim < 0:
            raise ValueError(
                "leverage_right_jl_dim must be >= 0 or None, got "
                f"{leverage_right_jl_dim}"
            )
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
        if leverage_ridge_jitter <= 0:
            raise ValueError(f"leverage_ridge_jitter must be > 0, got {leverage_ridge_jitter}")
        if leverage_ridge_dim is not None and leverage_ridge_dim < 1:
            raise ValueError(f"leverage_ridge_dim must be >= 1 or None, got {leverage_ridge_dim}")
        if leverage_approx_method == "right_sketch_ridge":
            resolved_ridge_dim = leverage_ridge_dim if leverage_ridge_dim is not None else leverage_right_jl_dim
            if resolved_ridge_dim is None or int(resolved_ridge_dim) < 1:
                raise ValueError(
                    "right_sketch_ridge requires leverage_ridge_dim >= 1 or "
                    "leverage_right_jl_dim >= 1"
                )
        self.policy = policy
        self.debug = debug
        self.leverage_sketch_dim = leverage_sketch_dim
        self.leverage_granularity = leverage_granularity
        self.leverage_feature = leverage_feature
        self.leverage_projection = leverage_projection
        self.leverage_head_mean_dim = int(leverage_head_mean_dim)
        self.leverage_normalize_rows = bool(leverage_normalize_rows)
        self.leverage_approx_method = leverage_approx_method
        self.leverage_left_sketch_dim = leverage_left_sketch_dim
        self.leverage_right_jl_dim = leverage_right_jl_dim
        self.leverage_ridge_lambda = float(leverage_ridge_lambda)
        self.leverage_ridge_lambda_mode = leverage_ridge_lambda_mode
        self.leverage_ridge_score_chunk_size = int(leverage_ridge_score_chunk_size)
        self.leverage_ridge_jitter = float(leverage_ridge_jitter)
        self.leverage_ridge_dim = leverage_ridge_dim
        self.leverage_random_seed = int(leverage_random_seed)
        self.leverage_eviction_selector = leverage_eviction_selector
        self.leverage_dpp_candidate_multiplier = int(leverage_dpp_candidate_multiplier)
        self.leverage_dpp_greedy_block_size = int(leverage_dpp_greedy_block_size)
        self.leverage_dpp_diversity_beta = float(leverage_dpp_diversity_beta)
        self.layer_budget_strategy = layer_budget_strategy
        self.layer_budget_value_gamma = float(layer_budget_value_gamma)
        self.layer_budget_value_norm_type = layer_budget_value_norm_type
        self.layer_budget_norm_source = layer_budget_norm_source
        self.layer_budget_eps = float(layer_budget_eps)
        self.leverage_dpp_full_sim_max_elements = 4_000_000
        self._leverage_right_sketch_cache = {}
        self._leverage_left_srht_cache = {}
        self._last_leverage_profile: Dict[str, float] = {}
        self._last_layer_feature_shape: Optional[tuple[int, int]] = None

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
        candidate_evictable_mask: Optional[torch.Tensor] = None,
        need_leverage_basis: bool = False,
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
        B, H, N, _ = k.shape
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
        self._last_leverage_profile = {}
        self._last_layer_feature_shape = None
        need_mean_scores = self.policy in ("mean", "baseline_mean") or need_summary or self.debug
        mean_scores = self._mean_scores(candidate_k) if need_mean_scores else None

        leverage_basis = None
        layer_budget_score = None
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
                layer_budget_score = self._compute_layer_budget_score(policy_scores, candidate_k, candidate_v)
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
                    f" leverage_sketch_dim={sketch_label} "
                    f"leverage_left_sketch_dim={self.leverage_left_sketch_dim} "
                    f"leverage_right_jl_dim={self.leverage_right_jl_dim} "
                    f"leverage_ridge_lambda={self.leverage_ridge_lambda} "
                    f"leverage_ridge_lambda_mode={self.leverage_ridge_lambda_mode} "
                    f"leverage_ridge_score_chunk_size={self.leverage_ridge_score_chunk_size} "
                    f"leverage_ridge_jitter={self.leverage_ridge_jitter} "
                    f"leverage_ridge_dim={self.leverage_ridge_dim} "
                    f"leverage_random_seed={self.leverage_random_seed} "
                    f"leverage_granularity={self.leverage_granularity} leverage_feature={self.leverage_feature} "
                    f"leverage_projection={self.leverage_projection} "
                    f"leverage_head_mean_dim={self.leverage_head_mean_dim} "
                    f"leverage_normalize_rows={self.leverage_normalize_rows} "
                    f"leverage_eviction_selector={self.leverage_eviction_selector} "
                    f"leverage_dpp_candidate_multiplier={self.leverage_dpp_candidate_multiplier} "
                    f"leverage_dpp_greedy_block_size={self.leverage_dpp_greedy_block_size} "
                    f"leverage_dpp_diversity_beta={self.leverage_dpp_diversity_beta} "
                    f"layer_budget_value_gamma={self.layer_budget_value_gamma} "
                    f"layer_budget_value_norm_type={self.layer_budget_value_norm_type} "
                    f"layer_budget_norm_source={self.layer_budget_norm_source} "
                    f"num_heads={H} num_tokens={num_candidates} head_dim={D} feature_dim={feature_dim}"
                )
                if self.policy == "dpp":
                    msg += " dpp_pool=full"
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
                profile_items = []
                time_fields = {
                    "candidate_matrix_preparation",
                    "feature",
                    "sketch",
                    "sketch_matrix_retrieval",
                    "projection_matmul",
                    "qr",
                    "small_qr",
                    "left_sketch",
                    "right_jl_solve",
                    "omega_gemm",
                    "gram",
                    "cholesky",
                    "score_solve",
                    "scoring",
                    "total",
                }
                int_fields = {"fallback", "N", "D", "sketch_dim", "cholesky_retries"}
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
                print(f"[EvictionManager] svd_leverage_profile {profile}")
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
            for batch_idx, batch_scores in enumerate(score_input):
                if self.layer_budget_strategy == "leverage_pr":
                    layer_scores.append(_layer_score_leverage_pr(batch_scores, self.layer_budget_eps))
                elif self.layer_budget_strategy == "leverage_entropy":
                    layer_scores.append(_layer_score_leverage_entropy(batch_scores, self.layer_budget_eps))
                elif self.layer_budget_strategy == "value_weighted_leverage_pr":
                    layer_scores.append(_layer_score_leverage_pr(batch_scores, self.layer_budget_eps))
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
            if self.layer_budget_strategy == "value_weighted_leverage_pr":
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
    def _keep_highest_scores(scores: torch.Tensor, num_to_keep: int) -> torch.Tensor:
        _, kept = torch.topk(scores, k=num_to_keep, dim=-1)
        return kept.sort(dim=-1).values

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
                kept_2d = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest=False)
                H = int(num_heads) if num_heads is not None else int(candidate_k.shape[1])
                return kept_2d.unsqueeze(1).expand(scores.shape[0], H, kept_2d.shape[-1]), protection_debug
            return self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest=False), protection_debug

        if self.leverage_eviction_selector == "fast_dpp":
            if shared_across_heads:
                kept_2d = self._keep_after_layer_fast_dpp(
                    scores,
                    evictable_mask,
                    actual_evict,
                    candidate_k,
                    candidate_v,
                )
                H = int(num_heads) if num_heads is not None else int(candidate_k.shape[1])
                return kept_2d.unsqueeze(1).expand(scores.shape[0], H, kept_2d.shape[-1]), protection_debug
            return self._keep_after_head_fast_dpp(
                scores,
                evictable_mask,
                actual_evict,
                candidate_k,
            ), protection_debug

        raise AssertionError(f"Unhandled leverage eviction selector: {self.leverage_eviction_selector}")

    def _keep_after_head_fast_dpp(
        self,
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        candidate_k: torch.Tensor,
        *,
        use_full_pool: bool = False,
    ) -> torch.Tensor:
        B, H, N = scores.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, H, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        for b in range(B):
            for h in range(H):
                evicted = self._fast_dpp_evicted_indices(
                    scores[b, h],
                    evictable_mask[b, h],
                    int(num_to_evict),
                    lambda idx, b=b, h=h: self._head_dpp_features(candidate_k, b, h, idx),
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
        use_full_pool: bool = False,
    ) -> torch.Tensor:
        B, N = scores.shape
        keep_count = N - int(num_to_evict)
        kept = torch.empty(B, keep_count, device=scores.device, dtype=torch.long)
        all_indices = torch.arange(N, device=scores.device, dtype=torch.long)
        for b in range(B):
            evicted = self._fast_dpp_evicted_indices(
                scores[b],
                evictable_mask[b],
                int(num_to_evict),
                lambda idx, b=b: self._layer_dpp_features(candidate_k, candidate_v, b, idx),
                use_full_pool=use_full_pool,
            )
            keep_mask = torch.ones(N, device=scores.device, dtype=torch.bool)
            keep_mask[evicted] = False
            kept[b] = all_indices[keep_mask].sort(dim=-1).values
        return kept

    def _fast_dpp_evicted_indices(
        self,
        row_scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        feature_fn,
        *,
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

        features = F.normalize(feature_fn(pool).float(), p=2, dim=-1, eps=1e-12)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if use_full_pool:
            quality = torch.ones(pool_size, device=row_scores.device, dtype=torch.float32)
        else:
            # The low-score pool is vulnerable; DPP selects representatives to retain.
            pool_scores = row_scores.index_select(0, pool)
            score_min = pool_scores.min()
            score_range = (pool_scores.max() - score_min).clamp_min(1e-12)
            quality = ((pool_scores - score_min) / score_range).clamp_min(0.0) + 1e-6
        log_quality = torch.log(quality)

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
            self._last_layer_feature_shape = tuple(features.shape)
            return features

        key_features = mat_k.transpose(0, 1).reshape(indices.numel(), H * D)
        if self.leverage_feature == "key_value":
            if candidate_v is None:
                raise ValueError("leverage_feature=key_value requires value cache tensor")
            mat_v = candidate_v[batch_idx].index_select(1, indices).to(dtype=torch.float32)
            mat_v = torch.nan_to_num(mat_v, nan=0.0, posinf=0.0, neginf=0.0)
            value_features = mat_v.transpose(0, 1).reshape(indices.numel(), H * D)
            features = torch.cat([key_features, value_features], dim=-1)
            self._last_layer_feature_shape = tuple(features.shape)
            return features
        self._last_layer_feature_shape = tuple(key_features.shape)
        return key_features

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
            H = int(num_heads) if num_heads is not None else int(source_mask.shape[1])
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
                if special_mask.ndim != 3 or special_mask.shape[0] != scores.shape[0] or special_mask.shape[2] != scores.shape[1]:
                    raise ValueError(
                        "Expected candidate_evictable_mask [B, H, N] for layer-wise protection, "
                        f"got {tuple(special_mask.shape)} for scores {tuple(scores.shape)}"
                    )
                special_mask = special_mask.all(dim=1)
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
                if frame_ids.ndim != 3 or frame_ids.shape[0] != scores.shape[0] or frame_ids.shape[2] != scores.shape[1]:
                    raise ValueError(
                        "Expected candidate_frame_ids [B, H, N] for layer-wise protection, "
                        f"got {tuple(frame_ids.shape)} for scores {tuple(scores.shape)}"
                    )
                recent_mask = ((frame_ids < 0) | (frame_ids < threshold)).all(dim=1)
            else:
                if frame_ids.shape != scores.shape:
                    raise ValueError(
                        "candidate_frame_ids must match head-wise score shape, "
                        f"got {tuple(frame_ids.shape)} vs {tuple(scores.shape)}"
                    )
                recent_mask = (frame_ids < 0) | (frame_ids < threshold)
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

    def _get_left_srht_factors(
        self,
        num_tokens: int,
        padded_tokens: int,
        sketch_dim: int,
        *,
        device: torch.device,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), int(num_tokens), int(padded_tokens), int(sketch_dim), int(seed), "left_srht")
        cached = self._leverage_left_srht_cache.get(key)
        if cached is not None:
            return cached

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        signs = torch.randint(
            0,
            2,
            (int(padded_tokens),),
            dtype=torch.int8,
            generator=generator,
        ).to(dtype=torch.float32)
        signs = signs.mul_(2.0).sub_(1.0).to(device=device)
        sample = torch.randperm(int(padded_tokens), generator=generator, dtype=torch.long)[: int(sketch_dim)]
        sample = sample.to(device=device)
        cached = (signs, sample)
        self._leverage_left_srht_cache[key] = cached
        return cached

    @staticmethod
    def _fwht_token_dim(x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """Pure PyTorch FWHT over the token dimension ``-2``."""
        num_tokens = int(x.shape[-2])
        if num_tokens <= 0 or num_tokens & (num_tokens - 1):
            raise ValueError(f"FWHT token dimension must be a power of two, got {num_tokens}")

        y = x.transpose(-2, -1).contiguous()
        prefix = y.shape[:-1]
        stride = 1
        while stride < num_tokens:
            y = y.reshape(*prefix, num_tokens // (2 * stride), 2, stride)
            left = y[..., 0, :].clone()
            right = y[..., 1, :].clone()
            y = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, num_tokens)
            stride *= 2
        if normalize:
            y = y / math.sqrt(float(num_tokens))
        return y.transpose(-2, -1).contiguous()

    def _apply_left_srht(
        self,
        mat: torch.Tensor,
        sketch_dim: int,
        *,
        seed: Optional[int] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Apply ``sqrt(N_pad / r1) * S H D`` to token rows without forming it."""
        num_tokens = int(mat.shape[-2])
        padded_tokens = self._next_power_of_two(num_tokens)
        active_sketch_dim = min(int(sketch_dim), padded_tokens)
        active_seed = self.leverage_random_seed if seed is None else int(seed)
        signs, sample = self._get_left_srht_factors(
            num_tokens,
            padded_tokens,
            active_sketch_dim,
            device=mat.device,
            seed=active_seed,
        )

        if padded_tokens != num_tokens:
            work = F.pad(mat, (0, 0, 0, padded_tokens - num_tokens))
        else:
            work = mat
        work = work * signs.view(*((1,) * (work.ndim - 2)), padded_tokens, 1)
        work = self._fwht_token_dim(work, normalize=normalize)
        work = work.index_select(-2, sample)
        scale = math.sqrt(float(padded_tokens) / float(active_sketch_dim))
        return work * scale

    def _resolve_leverage_approx_method(self, sketch_dim: Optional[int]) -> str:
        if self.leverage_approx_method in ("exact_qr", "drineas_srht", "full_d_ridge", "right_sketch_ridge"):
            return self.leverage_approx_method
        if sketch_dim in (None, 0):
            return "exact_qr"
        return "right_sketch"

    def _resolve_ridge_sketch_dim(self, feature_dim: int, num_tokens: int) -> int:
        requested = self.leverage_ridge_dim if self.leverage_ridge_dim is not None else self.leverage_right_jl_dim
        if requested is None or int(requested) < 1:
            raise ValueError(
                "right_sketch_ridge requires leverage_ridge_dim >= 1 or "
                "leverage_right_jl_dim >= 1"
            )
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

    def _maybe_normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        if not self.leverage_normalize_rows:
            return x
        return F.normalize(x, p=2, dim=-1, eps=1e-12)

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
        profile["method"] = basis_kind
        profile["N"] = float(num_tokens)
        profile.setdefault("D", float(feature_dim))
        if sketch_dim is not None:
            profile["sketch_dim"] = float(sketch_dim)

        with torch.cuda.amp.autocast(enabled=False):
            work = torch.nan_to_num(mat.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            eye = torch.eye(feature_dim, device=work.device, dtype=torch.float32)
            eye = eye.view(*((1,) * (work.ndim - 2)), feature_dim, feature_dim)

            gram_start = time.perf_counter() if do_profile else 0.0
            gram = work.transpose(-2, -1) @ work
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
            jitter_scale = torch.maximum(scale, torch.ones_like(scale))

            chol_start = time.perf_counter() if do_profile else 0.0
            chol = None
            last_a = None
            retries = 0
            for retry_idx, jitter_multiplier in enumerate((1.0, 10.0, 100.0, 1000.0)):
                diag_add = lam + float(self.leverage_ridge_jitter) * float(jitter_multiplier) * jitter_scale
                last_a = gram + diag_add.unsqueeze(-1).unsqueeze(-1) * eye
                chol, info = torch.linalg.cholesky_ex(last_a)
                if not bool((info != 0).any().item()):
                    retries = retry_idx
                    break
                chol = None
                retries = retry_idx + 1
            if do_profile:
                if chol is not None:
                    self._sync_for_timing(chol)
                elif last_a is not None:
                    self._sync_for_timing(last_a)
                profile["cholesky"] = time.perf_counter() - chol_start
                profile["lambda_value"] = float(torch.nan_to_num(lam.detach().float()).mean().item())
                profile["cholesky_retries"] = float(retries)

            score_start = time.perf_counter() if do_profile else 0.0
            fallback = 0.0
            if chol is not None:
                chunks = []
                chunk_size = int(self.leverage_ridge_score_chunk_size)
                for start in range(0, num_tokens, chunk_size):
                    end = min(start + chunk_size, num_tokens)
                    rhs = work[..., start:end, :].transpose(-2, -1).contiguous()
                    solved = torch.cholesky_solve(rhs, chol)
                    chunks.append((rhs * solved).sum(dim=-2))
                scores = torch.cat(chunks, dim=-1) if chunks else torch.empty(*work.shape[:-1], device=work.device, dtype=torch.float32)
            else:
                try:
                    assert last_a is not None
                    inv_a = torch.linalg.pinv(last_a)
                    transformed = work @ inv_a
                    scores = (transformed * work).sum(dim=-1)
                    fallback = 1.0
                except RuntimeError:
                    scores = work.square().sum(dim=-1)
                    fallback = 2.0
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            if do_profile:
                self._sync_for_timing(scores)
                profile["score_solve"] = time.perf_counter() - score_start
                profile["scoring"] = profile["score_solve"]
                profile["fallback"] = fallback
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
        if method == "drineas_srht":
            return self._drineas_srht_leverage_scores_from_matrix(
                mat,
                eps,
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
                # Right-sketched leverage / Compactor-style approximation:
                # QR is applied to K Phi, so scores are leverage scores of the
                # compressed key matrix rather than Drineas full-space scores.
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
                    profile["sketch"] = profile["sketch_matrix_retrieval"] + profile["projection_matmul"]

            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = torch.nan_to_num(mat.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
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

    def _drineas_srht_leverage_scores_from_matrix(
        self,
        mat: torch.Tensor,
        eps: float,
        *,
        return_basis: bool,
        granularity: str,
        profile: Dict[str, float],
        total_start: float,
        do_profile: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, SvdLeverageBasis]:
        # Left-sketch Drineas-style approximate full leverage:
        # Pi1 K -> R^{-1}, Omega = K R^{-1} Pi2, scores = ||Omega_i||_2^2.
        num_tokens = int(mat.shape[-2])
        feature_dim = int(mat.shape[-1])
        requested_r1 = self.leverage_left_sketch_dim if self.leverage_left_sketch_dim is not None else self._next_power_of_two(num_tokens)
        padded_tokens = self._next_power_of_two(num_tokens)
        r1 = min(int(requested_r1), padded_tokens)
        r2 = self.leverage_right_jl_dim

        with torch.cuda.amp.autocast(enabled=False):
            left_start = time.perf_counter() if do_profile else 0.0
            b_mat = self._apply_left_srht(mat, r1, seed=self.leverage_random_seed, normalize=True)
            if do_profile:
                self._sync_for_timing(b_mat)
                profile["left_sketch"] = time.perf_counter() - left_start

            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                _q_b, r = torch.linalg.qr(b_mat, mode="reduced")
                if do_profile:
                    self._sync_for_timing(r)
                    profile["small_qr"] = time.perf_counter() - qr_start
                diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
                square_r = r.shape[-2] == r.shape[-1]
                diag_max = diag.max(dim=-1, keepdim=True).values.clamp_min(float(eps)) if diag.ndim > 1 else diag.max().clamp_min(float(eps))
                ill_conditioned = (not square_r) or bool((diag <= diag_max * float(eps)).any().item())
                if ill_conditioned:
                    raise RuntimeError("Drineas SRHT QR produced singular or non-square R")

                solve_start = time.perf_counter() if do_profile else 0.0
                if r2 is None or int(r2) <= 0 or int(r2) >= feature_dim:
                    rhs = torch.eye(feature_dim, device=mat.device, dtype=torch.float32)
                    rhs = rhs.expand(*r.shape[:-2], feature_dim, feature_dim)
                else:
                    sketch_retrieval_start = time.perf_counter() if do_profile else 0.0
                    pi2 = self._get_leverage_right_sketch(
                        feature_dim,
                        int(r2),
                        device=mat.device,
                        seed=self.leverage_random_seed + 1,
                    )
                    if do_profile:
                        self._sync_for_timing(pi2)
                        profile["sketch_matrix_retrieval"] = (
                            profile.get("sketch_matrix_retrieval", 0.0)
                            + time.perf_counter()
                            - sketch_retrieval_start
                        )
                    rhs = pi2.expand(*r.shape[:-2], feature_dim, int(r2))
                c = torch.linalg.solve_triangular(r, rhs, upper=True)
                if do_profile:
                    self._sync_for_timing(c)
                    profile["right_jl_solve"] = time.perf_counter() - solve_start

                gemm_start = time.perf_counter() if do_profile else 0.0
                omega = mat @ c
                if do_profile:
                    self._sync_for_timing(omega)
                    profile["omega_gemm"] = time.perf_counter() - gemm_start
                profile["fallback"] = 0.0
            except RuntimeError:
                scores, basis = self._drineas_svd_fallback(
                    mat,
                    b_mat,
                    eps,
                    return_basis=return_basis,
                    granularity=granularity,
                    profile=profile,
                    do_profile=do_profile,
                )
                self._profile_finish(profile, total_start, scores, do_profile=do_profile)
                if return_basis:
                    return scores, basis
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        scores_sq = torch.nan_to_num(omega.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
        self._profile_finish(profile, total_start, scores_sq, do_profile=do_profile)
        if return_basis:
            # q is not exact QR Q here; it is the Drineas score-space matrix Omega.
            return scores_sq, SvdLeverageBasis(
                q=omega.detach(),
                r_diag=diag.detach(),
                granularity=granularity,
                basis_kind="drineas_srht",
            )
        return scores_sq

    def _drineas_svd_fallback(
        self,
        mat: torch.Tensor,
        b_mat: torch.Tensor,
        eps: float,
        *,
        return_basis: bool,
        granularity: str,
        profile: Dict[str, float],
        do_profile: bool,
    ) -> tuple[torch.Tensor, SvdLeverageBasis]:
        try:
            svd_start = time.perf_counter() if do_profile else 0.0
            _u, s, vh = torch.linalg.svd(b_mat, full_matrices=False)
            max_s = s.max(dim=-1, keepdim=True).values.clamp_min(float(eps)) if s.ndim > 1 else s.max().clamp_min(float(eps))
            active = s > max_s * float(eps)
            if not bool(active.any().item()):
                raise RuntimeError("Drineas SRHT SVD fallback found zero numerical rank")
            if do_profile:
                self._sync_for_timing(s)
                profile["small_qr"] = profile.get("small_qr", 0.0) + time.perf_counter() - svd_start

            coords = mat @ vh.transpose(-2, -1)
            coords = coords / s.clamp_min(float(eps)).unsqueeze(-2)
            coords = coords * active.to(dtype=coords.dtype).unsqueeze(-2)
            r2 = self.leverage_right_jl_dim
            if r2 is not None and int(r2) > 0 and int(r2) < coords.shape[-1]:
                solve_start = time.perf_counter() if do_profile else 0.0
                sketch_retrieval_start = time.perf_counter() if do_profile else 0.0
                pi2 = self._get_leverage_right_sketch(
                    int(coords.shape[-1]),
                    int(r2),
                    device=mat.device,
                    seed=self.leverage_random_seed + 1,
                )
                if do_profile:
                    self._sync_for_timing(pi2)
                    profile["sketch_matrix_retrieval"] = (
                        profile.get("sketch_matrix_retrieval", 0.0)
                        + time.perf_counter()
                        - sketch_retrieval_start
                    )
                coords = coords @ pi2
                if do_profile:
                    self._sync_for_timing(coords)
                    profile["right_jl_solve"] = time.perf_counter() - solve_start
            scores = torch.nan_to_num(coords.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            profile["fallback"] = 1.0
            basis = SvdLeverageBasis(
                q=coords.detach() if return_basis else torch.empty(*mat.shape[:-1], 0, device=mat.device, dtype=torch.float32),
                r_diag=s.detach(),
                granularity=granularity,
                basis_kind="drineas_srht_svd_fallback",
            )
            return scores, basis
        except RuntimeError:
            scores = torch.nan_to_num(mat.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            profile["fallback"] = 2.0
            return scores, self._empty_leverage_basis(mat, granularity, basis_kind="row_norm_fallback")

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
        do_profile = self.debug

        if do_profile:
            self._sync_for_timing(x)
        total_start = time.perf_counter() if do_profile else 0.0
        prep_start = time.perf_counter() if do_profile else 0.0
        with torch.cuda.amp.autocast(enabled=False):
            mat = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mat = self._maybe_normalize_rows(mat)
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
        do_profile = self.debug
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0
        prep_start = time.perf_counter() if do_profile else 0.0
        with torch.cuda.amp.autocast(enabled=False):
            mat = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mat = self._maybe_normalize_rows(mat)
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
        feature_dim = H * D * (2 if self.leverage_feature == "key_value" else 1)
        self._last_layer_feature_shape = (int(N), int(feature_dim))
        if self.leverage_feature == "key_value":
            if candidate_v is None:
                raise ValueError("leverage_feature='key_value' requires value cache tensor")
            if candidate_v.shape != candidate_k.shape:
                raise ValueError(
                    "candidate_v must match candidate_k for key_value leverage, "
                    f"got {tuple(candidate_v.shape)} vs {tuple(candidate_k.shape)}"
                )

        if self._resolve_leverage_approx_method(self.leverage_sketch_dim) in ("right_sketch", "right_sketch_ridge"):
            return self._layer_svd_leverage_scores_sketched(
                candidate_k,
                candidate_v,
                feature_dim,
                return_basis=return_basis,
            )

        scores = []
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
        for batch_idx in range(B):
            feature_start = time.perf_counter() if self.debug else 0.0
            x_key = candidate_k[batch_idx].transpose(0, 1).reshape(N, H * D)
            if self.leverage_feature == "key_value":
                assert candidate_v is not None
                x_value = candidate_v[batch_idx].transpose(0, 1).reshape(N, H * D)
                x_layer = torch.cat([x_key, x_value], dim=-1)
            else:
                x_layer = x_key
            if self.debug:
                self._sync_for_timing(x_layer)
            feature_time = time.perf_counter() - feature_start if self.debug else 0.0
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
            if self.debug:
                aggregate_profile["feature"] += feature_time
                for name, value in self._last_leverage_profile.items():
                    if isinstance(value, (int, float)):
                        aggregate_profile[name] = aggregate_profile.get(name, 0.0) + value
                    else:
                        aggregate_profile[name] = value
        if self.debug:
            self._last_leverage_profile = aggregate_profile
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
        do_profile = self.debug
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0

        with torch.cuda.amp.autocast(enabled=False):
            feature_start = time.perf_counter() if do_profile else 0.0
            mat_k = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            head_chunks = torch.tensor_split(mat_k, self.leverage_head_mean_dim, dim=-1)
            head_features = torch.stack([chunk.mean(dim=-1) for chunk in head_chunks], dim=-1)
            leverage_matrix = head_features.permute(0, 2, 1, 3).reshape(B, N, feature_dim).contiguous()
            leverage_matrix = self._maybe_normalize_rows(leverage_matrix)
            if do_profile:
                self._sync_for_timing(leverage_matrix)
                profile["feature"] = time.perf_counter() - feature_start
                profile["sketch"] = 0.0

            method = self._resolve_leverage_approx_method(self.leverage_sketch_dim)
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
        method = self._resolve_leverage_approx_method(self.leverage_sketch_dim)
        if method == "right_sketch_ridge":
            sketch_dim = self._resolve_ridge_sketch_dim(feature_dim, N)
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
        do_profile = self.debug
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0
        prep_start = time.perf_counter() if do_profile else 0.0

        with torch.cuda.amp.autocast(enabled=False):
            mat_k = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mat_v = (
                torch.nan_to_num(candidate_v.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if self.leverage_feature == "key_value" and candidate_v is not None
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
                feature_dim,
                sketch_dim,
                device=mat_k.device,
                seed=self.leverage_random_seed,
            )
            if do_profile:
                self._sync_for_timing(omega)
                profile["sketch_matrix_retrieval"] = time.perf_counter() - sketch_retrieval_start
            projection_start = time.perf_counter() if do_profile else 0.0
            omega_key = omega[: H * D].view(H, D, sketch_dim)
            leverage_matrix = torch.einsum("bhnd,hds->bns", mat_k, omega_key)
            if mat_v is not None:
                omega_value = omega[H * D :].view(H, D, sketch_dim)
                leverage_matrix = leverage_matrix + torch.einsum("bhnd,hds->bns", mat_v, omega_value)
            if self.leverage_normalize_rows:
                row_norm_sq = mat_k.square().sum(dim=(1, 3))
                if mat_v is not None:
                    row_norm_sq = row_norm_sq + mat_v.square().sum(dim=(1, 3))
                leverage_matrix = leverage_matrix / row_norm_sq.sqrt().clamp_min(1e-12).unsqueeze(-1)
            if do_profile:
                self._sync_for_timing(leverage_matrix)
                profile["feature"] = 0.0
                profile["projection_matmul"] = time.perf_counter() - projection_start
                profile["sketch"] = profile["sketch_matrix_retrieval"] + profile["projection_matmul"]

            if method == "right_sketch_ridge":
                profile["D"] = float(feature_dim)
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
