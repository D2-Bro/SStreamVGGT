"""Head-wise KV cache eviction policies for streaming attention."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


VALID_LAYER_BUDGET_STRATEGIES = ("uniform", "leverage_pr", "leverage_entropy")


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
    VALID_LEVERAGE_APPROX_METHODS = ("exact_qr", "right_sketch", "drineas_srht")
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
        leverage_random_seed: int = 0,
        leverage_eviction_selector: str = "topk",
        leverage_dpp_candidate_multiplier: int = 2,
        leverage_dpp_greedy_block_size: int = 32,
        layer_budget_strategy: str = "uniform",
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
        if layer_budget_strategy not in VALID_LAYER_BUDGET_STRATEGIES:
            raise ValueError(
                "layer_budget_strategy must be one of "
                f"{VALID_LAYER_BUDGET_STRATEGIES}, got {layer_budget_strategy!r}"
            )
        if layer_budget_strategy != "uniform" and (policy != "svd_leverage" or leverage_granularity != "layer"):
            raise ValueError(
                "layer_budget_strategy requires eviction_policy='svd_leverage' "
                "and leverage_granularity='layer'"
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
        self.leverage_random_seed = int(leverage_random_seed)
        self.leverage_eviction_selector = leverage_eviction_selector
        self.leverage_dpp_candidate_multiplier = int(leverage_dpp_candidate_multiplier)
        self.leverage_dpp_greedy_block_size = int(leverage_dpp_greedy_block_size)
        self.layer_budget_strategy = layer_budget_strategy
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
            when recent-frame protection leaves too few evictable candidates.
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
            if protect_recent_frames > 0:
                kept, protection_debug = self._keep_with_recent_protection(
                    policy_scores,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
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
                layer_budget_score = self._compute_layer_budget_score(policy_scores)
                kept, protection_debug = self._select_svd_leverage_kept(
                    policy_scores,
                    candidate_k,
                    candidate_v,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
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
                    f"leverage_random_seed={self.leverage_random_seed} "
                    f"leverage_granularity={self.leverage_granularity} leverage_feature={self.leverage_feature} "
                    f"leverage_projection={self.leverage_projection} "
                    f"leverage_head_mean_dim={self.leverage_head_mean_dim} "
                    f"leverage_normalize_rows={self.leverage_normalize_rows} "
                    f"leverage_eviction_selector={self.leverage_eviction_selector} "
                    f"leverage_dpp_candidate_multiplier={self.leverage_dpp_candidate_multiplier} "
                    f"leverage_dpp_greedy_block_size={self.leverage_dpp_greedy_block_size} "
                    f"num_heads={H} num_tokens={num_candidates} head_dim={D} feature_dim={feature_dim}"
                )
                if self.policy == "dpp":
                    msg += " dpp_pool=full"
            print(msg)
            if protection_debug is not None and protection_debug["limited_by_protection"]:
                print(
                    "[EvictionManager] recent-frame protection limited eviction; "
                    "cache may temporarily exceed budget"
                )
            if self.policy == "svd_leverage" and self.leverage_granularity == "layer":
                print(f"[EvictionManager] layer-wise SVD leverage: X shape={self._last_layer_feature_shape}")
            elif self.policy == "dpp" and self.leverage_granularity == "layer":
                print(f"[EvictionManager] layer-wise DPP features: X shape={self._last_layer_feature_shape}")
            if self.policy == "svd_leverage" and self._last_leverage_profile:
                profile_items = []
                for name, value in self._last_leverage_profile.items():
                    if name == "fallback":
                        profile_items.append(f"{name}={int(value)}")
                    else:
                        profile_items.append(f"{name}={value * 1000.0:.3f}ms")
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

    def _compute_layer_budget_score(self, policy_scores: torch.Tensor) -> torch.Tensor:
        if self.layer_budget_strategy == "uniform":
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
            for batch_scores in score_input:
                if self.layer_budget_strategy == "leverage_pr":
                    layer_scores.append(_layer_score_leverage_pr(batch_scores, self.layer_budget_eps))
                elif self.layer_budget_strategy == "leverage_entropy":
                    layer_scores.append(_layer_score_leverage_entropy(batch_scores, self.layer_budget_eps))
                else:
                    raise AssertionError(f"Unhandled layer budget strategy: {self.layer_budget_strategy}")
            return torch.stack(layer_scores).mean().detach()

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
        *,
        shared_across_heads: bool,
        num_heads: Optional[int] = None,
        use_full_dpp_pool: bool = False,
    ) -> tuple[torch.Tensor, Optional[Dict[str, int]]]:
        num_candidates = int(scores.shape[-1])
        requested_evict = num_candidates - int(num_to_keep)
        if protect_recent_frames > 0:
            if current_frame_idx is None:
                raise ValueError("current_frame_idx is required when protect_recent_frames > 0")
            if candidate_frame_ids is None:
                raise ValueError("candidate_frame_ids is required when protect_recent_frames > 0")
            threshold = int(current_frame_idx) - int(protect_recent_frames) + 1
            frame_ids = candidate_frame_ids.to(device=scores.device, dtype=torch.long)
            if shared_across_heads:
                if scores.ndim != 2:
                    raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
                if frame_ids.ndim != 3:
                    raise ValueError(
                        "Expected candidate_frame_ids [B, H, N] for layer-wise protection, "
                        f"got {tuple(frame_ids.shape)}"
                    )
                evictable_mask = ((frame_ids < 0) | (frame_ids < threshold)).all(dim=1)
                protected_tokens = int((~evictable_mask).sum().item())
                candidate_tokens = int(evictable_mask.sum().item())
            else:
                if scores.ndim != 3:
                    raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")
                if frame_ids.shape != scores.shape:
                    raise ValueError(
                        "candidate_frame_ids must match head-wise score shape, "
                        f"got {tuple(frame_ids.shape)} vs {tuple(scores.shape)}"
                    )
                evictable_mask = (frame_ids < 0) | (frame_ids < threshold)
                protected_tokens = int((~evictable_mask).sum().item())
                candidate_tokens = int(evictable_mask.sum().item())
            actual_evict = min(requested_evict, int(evictable_mask.sum(dim=-1).min().item()))
            protection_debug = {
                "current_frame_idx": int(current_frame_idx),
                "protect_recent_frames": int(protect_recent_frames),
                "protected_tokens": protected_tokens,
                "candidate_tokens": candidate_tokens,
                "requested_eviction_count": int(requested_evict),
                "actual_eviction_count": int(actual_evict),
                "limited_by_protection": int(actual_evict < requested_evict),
            }
        else:
            evictable_mask = torch.ones_like(scores, dtype=torch.bool)
            actual_evict = requested_evict
            protection_debug = None

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
            greedy_scores = log_quality + torch.log(diversity)
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
        *,
        evict_highest: bool,
        shared_across_heads: bool,
        num_heads: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        """Keep tokens after excluding recent frames only from eviction candidates.

        SVD/QR leverage scores are computed before this method is called; this
        mask only limits which scored tokens may be selected for eviction.
        """
        if current_frame_idx is None:
            raise ValueError("current_frame_idx is required when protect_recent_frames > 0")
        if candidate_frame_ids is None:
            raise ValueError("candidate_frame_ids is required when protect_recent_frames > 0")

        threshold = int(current_frame_idx) - int(protect_recent_frames) + 1
        if shared_across_heads:
            if scores.ndim != 2:
                raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
            if candidate_frame_ids.ndim != 3:
                raise ValueError(
                    "Expected candidate_frame_ids [B, H, N] for layer-wise protection, "
                    f"got {tuple(candidate_frame_ids.shape)}"
                )
            B, _, N = candidate_frame_ids.shape
            H = int(num_heads) if num_heads is not None else int(candidate_frame_ids.shape[1])
            frame_ids = candidate_frame_ids.to(device=scores.device, dtype=torch.long)
            evictable_mask = ((frame_ids < 0) | (frame_ids < threshold)).all(dim=1)
            actual_evict = min(N - int(num_to_keep), int(evictable_mask.sum(dim=-1).min().item()))
            kept_2d = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest)
            kept = kept_2d.unsqueeze(1).expand(B, H, kept_2d.shape[-1])
            protected_tokens = int((~evictable_mask).sum().item())
            candidate_tokens = int(evictable_mask.sum().item())
        else:
            if scores.ndim != 3:
                raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")
            if candidate_frame_ids.shape != scores.shape:
                raise ValueError(
                    "candidate_frame_ids must match head-wise score shape, "
                    f"got {tuple(candidate_frame_ids.shape)} vs {tuple(scores.shape)}"
                )
            _, _, N = scores.shape
            frame_ids = candidate_frame_ids.to(device=scores.device, dtype=torch.long)
            evictable_mask = (frame_ids < 0) | (frame_ids < threshold)
            actual_evict = min(N - int(num_to_keep), int(evictable_mask.sum(dim=-1).min().item()))
            kept = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest)
            protected_tokens = int((~evictable_mask).sum().item())
            candidate_tokens = int(evictable_mask.sum().item())

        requested_evict = scores.shape[-1] - int(num_to_keep)
        debug = {
            "current_frame_idx": int(current_frame_idx),
            "protect_recent_frames": int(protect_recent_frames),
            "protected_tokens": protected_tokens,
            "candidate_tokens": candidate_tokens,
            "requested_eviction_count": int(requested_evict),
            "actual_eviction_count": int(actual_evict),
            "limited_by_protection": int(actual_evict < requested_evict),
        }
        return kept, debug

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
        if self.leverage_approx_method == "exact_qr":
            return "exact_qr"
        if self.leverage_approx_method == "drineas_srht":
            return "drineas_srht"
        if sketch_dim in (None, 0):
            return "exact_qr"
        return "right_sketch"

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

        if self._resolve_leverage_approx_method(self.leverage_sketch_dim) == "right_sketch":
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
                    aggregate_profile[name] = aggregate_profile.get(name, 0.0) + value
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
