"""Head-wise KV cache eviction policies for streaming attention."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


VALID_LAYER_BUDGET_STRATEGIES = (
    "uniform",
    "leverage_pr",
    "key_norm",
    "value_weighted_leverage_pr",
)


LAYER_BUDGET_SCORE_STRATEGIES = (
    "leverage_pr",
    "key_norm",
    "value_weighted_leverage_pr",
)


VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES = (
    "value_weighted_leverage_pr",
)


def _participation_ratio_from_values(values: Optional[torch.Tensor], eps: float = 0) -> torch.Tensor:
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


def _layer_score_leverage_pr(lev: Optional[torch.Tensor], eps: float = 0) -> torch.Tensor:
    """Effective count from the participation ratio of QR leverage mass."""
    return _participation_ratio_from_values(lev, eps)


def _combine_layer_budget_pr_base(
    leverage_pr: torch.Tensor,
    strategy: str,
    eps: float,
) -> torch.Tensor:
    if strategy in ("leverage_pr", "value_weighted_leverage_pr"):
        return leverage_pr

    raise AssertionError(f"Unhandled layer budget strategy: {strategy}")

def _combine_value_weighted_leverage_pr_scores(
    base_scores,
    value_norms,
    gamma: float,
    eps: float = 0,
    active_mask: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError(f"value-weighted base alpha must be finite and >= 0, got {alpha}")
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
    if float(gamma) != 0.0:
        active_values = values[active]
        if active_values.numel():
            value_mean = active_values.mean()
            valid_mean = torch.isfinite(value_mean) & (value_mean > eps)
            safe_mean = torch.where(valid_mean, value_mean, torch.ones_like(value_mean))
            prior_active = (active_values / safe_mean).clamp_min(0.0).pow(float(gamma))
            prior_active = torch.nan_to_num(prior_active, nan=1.0, posinf=1.0, neginf=1.0)
            prior[active] = torch.where(valid_mean, prior_active, torch.ones_like(prior_active))
    return base.pow(float(alpha)) * prior


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
    eps: float = 0,
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


@dataclass
class LayerBudgetScoreResult:
    """Raw scores produced without selecting or removing cache tokens."""

    policy_scores: torch.Tensor
    layer_budget_score: torch.Tensor


class EvictionManager:
    """Dispatches head-wise cache eviction policies."""

    VALID_POLICIES = ("mean", "baseline_mean", "svd_leverage")
    VALID_LEVERAGE_EVICTION_RISK_MODES = ("low_leverage")
    VALID_LEVERAGE_APPROX_METHODS = ("right_sketch_ridge",)
    VALID_LEVERAGE_EVICTION_SELECTORS = ("topk")

    def __init__(
        self,
        policy: str = "svd_leverage",
        profile: bool = False,
        debug: bool = False,
        leverage_granularity: str = "layer",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_normalize_rows: bool = False,
        leverage_normalize_before_projection: bool = False,
        leverage_normalize_before_projection_headwise: bool = False,
        leverage_projected_key_cache: bool = False,
        leverage_approx_method: str = "right_sketch_ridge",
        leverage_ridge_lambda: float = 0,
        leverage_ridge_lambda_mode: str = "absolute",
        leverage_ridge_score_chunk_size: int = 4096,
        leverage_ridge_jitter: float = 1e-6,
        leverage_ridge_dim: int = 256,
        rls_refresh_interval: int = 8,
        leverage_random_seed: int = 42,
        leverage_eviction_selector: str = "topk",
        leverage_conf_gate: bool = False,
        leverage_conf_gate_floor: float = 0.0,
        leverage_conf_gate_depth_alpha: float = 1.0,
        leverage_conf_gate_point_beta: float = 0.0,
        layer_budget_strategy: str = "value_weighted_leverage_pr",
        layer_budget_value_gamma: float = 0.7,
        layer_budget_value_norm_type: str = "mean",
        layer_budget_norm_source: str = "key",
        layer_budget_eps: float = 0,
    ) -> None:
        self.policy = policy
        self.profile = bool(profile)
        self.debug = bool(debug)
        self.leverage_granularity = leverage_granularity
        self.leverage_feature = leverage_feature
        self.leverage_projection = leverage_projection
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
        if self.rls_refresh_interval < 1:
            raise ValueError("rls_refresh_interval must be >= 1")
        self.leverage_random_seed = int(leverage_random_seed)
        self.leverage_eviction_selector = leverage_eviction_selector
        self.leverage_conf_gate = bool(leverage_conf_gate)
        self.leverage_conf_gate_floor = float(leverage_conf_gate_floor)
        self.leverage_conf_gate_depth_alpha = float(leverage_conf_gate_depth_alpha)
        self.leverage_conf_gate_point_beta = float(leverage_conf_gate_point_beta)
        self.layer_budget_strategy = layer_budget_strategy
        self.layer_budget_value_gamma = float(layer_budget_value_gamma)
        self.layer_budget_value_norm_type = layer_budget_value_norm_type
        self.layer_budget_norm_source = layer_budget_norm_source
        self.layer_budget_eps = float(layer_budget_eps)
        self._leverage_right_sketch_cache = {}
        self._last_leverage_profile: Dict[str, float] = {}
        self._profile_totals: Dict[str, float] = {}
        self._profile_count = 0
        self._last_layer_feature_shape: Optional[tuple[int, int]] = None
        self._projected_key_cache: Optional[torch.Tensor] = None
        self._projected_key_pre_norm_cache: Optional[torch.Tensor] = None
        self._projected_key_cache_meta: Optional[Dict[str, object]] = None
        self._last_projected_key_features: Optional[torch.Tensor] = None
        self._last_projected_key_pre_norms: Optional[torch.Tensor] = None
        self._last_projected_key_cache_meta: Optional[Dict[str, object]] = None
        self._perf_trace_enabled = False
        self._perf_trace_events = []
        self._rls_context: Dict[str, Optional[int]] = {
            "step_idx": None,
            "current_frame_idx": None,
        }
        self.reset_rls_cache()

    def reset_profile_stats(self) -> None:
        self._profile_totals = {}
        self._profile_count = 0

    def reset_perf_trace(self) -> None:
        self._perf_trace_events = []

    def _perf_trace_start(self, tensor: torch.Tensor):
        if not self._perf_trace_enabled or not tensor.is_cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _perf_trace_end(self, name: str, start_event) -> None:
        if start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._perf_trace_events.append((name, start_event, end_event))

    def get_perf_trace_stats(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for name, start_event, end_event in self._perf_trace_events:
            totals[name] = totals.get(name, 0.0) + float(start_event.elapsed_time(end_event))
        return totals

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

    def _record_last_leverage_profile(self) -> None:
        if not self._last_leverage_profile:
            return
        if "scoring" in self._last_leverage_profile and "score_calc" not in self._last_leverage_profile:
            self._last_leverage_profile["score_calc"] = self._last_leverage_profile["scoring"]
        self._record_profile_event(self._last_leverage_profile)

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
        current_frame_ids: Optional[Sequence[int]] = None,
        candidate_frame_ids: Optional[torch.Tensor] = None,
        candidate_depth_confidence: Optional[torch.Tensor] = None,
        candidate_point_confidence: Optional[torch.Tensor] = None,
        candidate_conf_gate: Optional[torch.Tensor] = None,
        candidate_attention_utility: Optional[torch.Tensor] = None,
        candidate_attention_observed: Optional[torch.Tensor] = None,
        attention_utility_beta: float = 0.0,
        need_leverage_basis: bool = False,
        capture_projected_norms: bool = False,
    ) -> EvictionResult:
        """Select candidate-local indices to retain.

        Args:
            k: Key cache shaped ``[B, H, N, D]``.
            cache_budget: Final number of tokens to retain.
            num_anchor_tokens: Initial tokens that are always preserved.

        Returns:
            EvictionResult with candidate-local kept indices shaped
            ``[B, H, K]``, where ``K = cache_budget - num_anchor_tokens``.
        """
        B, H, N, D = k.shape
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
        self._set_rls_context(step_idx=step_idx, current_frame_idx=current_frame_idx)
        self._last_projected_pre_norms = None
        self._last_projected_key_features = None
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
            kept = self._keep_lowest_scores(policy_scores, num_to_keep)
        elif self.policy == "svd_leverage":
            rls_refresh_before_score = self.rls_refresh_count
            score_trace_start = self._perf_trace_start(candidate_k)
            raw_policy_scores, leverage_basis, layer_budget_score = self._score_svd_leverage_candidates(
                candidate_k,
                candidate_v,
                need_leverage_basis=need_leverage_basis,
            )
            score_trace_name = (
                "score_rls_refresh"
                if self.rls_refresh_count > rls_refresh_before_score
                else "score_rls_cache_hit"
            )
            self._perf_trace_end(score_trace_name, score_trace_start)
            policy_scores = raw_policy_scores
            confidence_trace_start = self._perf_trace_start(policy_scores)
            policy_scores = self._apply_confidence_gate(
                policy_scores,
                candidate_depth_confidence,
                candidate_point_confidence,
                candidate_conf_gate,
                candidate_frame_ids,
                current_frame_idx,
                shared_across_heads=True,
            )
            self._perf_trace_end("confidence_gate", confidence_trace_start)
            utility_trace_start = self._perf_trace_start(policy_scores)
            policy_scores = self._blend_attention_utility(
                policy_scores,
                candidate_attention_utility,
                attention_utility_beta,
                shared_across_heads=True,
                attention_observed=candidate_attention_observed,
            )
            selection_scores = self._prioritize_current_frame_for_attention_only(
                policy_scores,
                candidate_frame_ids=candidate_frame_ids,
                current_frame_idx=current_frame_idx,
                current_frame_ids=current_frame_ids,
                attention_utility_beta=attention_utility_beta,
                attention_utility_enabled=candidate_attention_utility is not None,
            )
            self._perf_trace_end("attention_utility", utility_trace_start)
            topk_trace_start = self._perf_trace_start(selection_scores)
            kept = self._select_svd_leverage_kept(
                selection_scores,
                num_to_keep,
                num_heads=H,
            )
            self._perf_trace_end("topk", topk_trace_start)
        else:
            raise AssertionError(f"Unhandled eviction policy: {self.policy}")

        summary_score = mean_scores.mean().item() if need_summary and mean_scores is not None else 0.0
        if self.debug:
            sketch_label = str(self.leverage_ridge_dim)
            requested_evicted = num_candidates - num_to_keep
            actual_evicted = num_candidates - int(kept.shape[-1])
            feature_dim = D
            if self.policy in ("svd_leverage") and self.leverage_granularity == "layer":
                feature_dim = H * D
            msg = (
                f"[EvictionManager] policy={self.policy} layer={layer_id} step={step_idx} "
                f"cache={N} budget={cache_budget} keep_candidates={kept.shape[-1]} "
                f"requested_evicted={requested_evicted} evicted={actual_evicted} "
                f"scores={tuple(policy_scores.shape)}"
            )
            if self.policy in ("svd_leverage"):
                msg += (
                    f" layer_budget_strategy={self.layer_budget_strategy} "
                    f"leverage_approx_method={self.leverage_approx_method} "
                    f"leverage_normalize_before_projection={self.leverage_normalize_before_projection} "
                    f"leverage_normalize_before_projection_headwise={self.leverage_normalize_before_projection_headwise} "
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
                    f"leverage_normalize_rows={self.leverage_normalize_rows} "
                    f"leverage_projected_key_cache={self.leverage_projected_key_cache} "
                    f"leverage_eviction_selector={self.leverage_eviction_selector} "
                    f"leverage_conf_gate={self.leverage_conf_gate} "
                    f"leverage_conf_gate_floor={self.leverage_conf_gate_floor} "
                    f"leverage_conf_gate_depth_alpha={self.leverage_conf_gate_depth_alpha} "
                    f"leverage_conf_gate_point_beta={self.leverage_conf_gate_point_beta} "
                    f"layer_budget_value_gamma={self.layer_budget_value_gamma} "
                    f"layer_budget_value_norm_type={self.layer_budget_value_norm_type} "
                    f"layer_budget_norm_source={self.layer_budget_norm_source} "
                    f"num_heads={H} num_tokens={num_candidates} head_dim={D} feature_dim={feature_dim}"
                )
            print(msg)
            if self.policy == "svd_leverage" and self.leverage_granularity == "layer":
                print(f"[EvictionManager] layer-wise SVD leverage: X shape={self._last_layer_feature_shape}")
        if (self.debug or self.profile) and self.policy == "svd_leverage":
            self._record_last_leverage_profile()
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

    def score_layer_budget(
        self,
        k: torch.Tensor,
        num_anchor_tokens: int,
        *,
        v: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        current_frame_idx: Optional[int] = None,
        capture_projected_norms: bool = False,
    ) -> LayerBudgetScoreResult:
        """Compute the value-weighted leverage payload without selecting tokens."""
        if self.policy != "svd_leverage":
            raise ValueError("Layer-budget score-only mode requires policy='svd_leverage'")
        if self.leverage_granularity != "layer":
            raise ValueError("Layer-budget score-only mode requires leverage_granularity='layer'")
        if self.layer_budget_strategy != "value_weighted_leverage_pr":
            raise ValueError(
                "Layer-budget score-only mode requires "
                "layer_budget_strategy='value_weighted_leverage_pr'"
            )

        _, _, num_tokens, _ = k.shape
        num_anchor_tokens = max(0, min(int(num_anchor_tokens), num_tokens))
        candidate_k = k[:, :, num_anchor_tokens:, :]
        candidate_v = v[:, :, num_anchor_tokens:, :] if v is not None else None
        self._last_leverage_profile = {}
        self._last_layer_feature_shape = None
        self._set_rls_context(step_idx=step_idx, current_frame_idx=current_frame_idx)
        self._last_projected_pre_norms = None
        self._last_projected_key_features = None
        self._last_projected_key_pre_norms = None
        self._last_projected_key_cache_meta = None
        self._capture_projected_norms = bool(capture_projected_norms)

        policy_scores, _, layer_budget_score = self._score_svd_leverage_candidates(
            candidate_k,
            candidate_v,
            need_leverage_basis=False,
        )
        if layer_budget_score is None:
            raise RuntimeError("value_weighted_leverage_pr did not produce a layer-budget score")
        self.commit_projected_key_cache_without_eviction()

        if self.debug:
            print(
                f"[EvictionManager][score-only] layer={layer_id} step={step_idx} "
                f"cache={num_tokens} anchors={num_anchor_tokens} "
                f"candidates={candidate_k.shape[2]} scores={tuple(policy_scores.shape)}"
            )
        if self.debug or self.profile:
            self._record_last_leverage_profile()
        return LayerBudgetScoreResult(
            policy_scores=policy_scores,
            layer_budget_score=layer_budget_score,
        )

    def _score_svd_leverage_candidates(
        self,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        *,
        need_leverage_basis: bool,
    ) -> tuple[torch.Tensor, Optional[SvdLeverageBasis], Optional[torch.Tensor]]:
        if need_leverage_basis:
            policy_scores, leverage_basis = self._layer_svd_leverage_scores(
                candidate_k,
                candidate_v,
                return_basis=True,
            )
        else:
            policy_scores = self._layer_svd_leverage_scores(candidate_k, candidate_v)
            leverage_basis = None
        layer_budget_score = None
        if self.layer_budget_strategy in LAYER_BUDGET_SCORE_STRATEGIES:
            layer_budget_score = self._compute_layer_budget_score(
                policy_scores,
                candidate_k,
                candidate_v,
            )
        return policy_scores, leverage_basis, layer_budget_score

    def _compute_layer_budget_score(
        self,
        policy_scores: torch.Tensor,
        candidate_k: Optional[torch.Tensor] = None,
        candidate_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            if self.layer_budget_strategy == "key_norm":
                if candidate_k is None or candidate_k.numel() == 0:
                    return torch.zeros((), device=policy_scores.device, dtype=torch.float32)
                if candidate_k.ndim != 4:
                    raise ValueError(
                        "Layer budget key_norm requires candidate_k shaped [B, H, N, D], "
                        f"got {tuple(candidate_k.shape)}"
                    )
                key_sq_norms = torch.nan_to_num(
                    candidate_k.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).square().sum(dim=(1, 3))
                return key_sq_norms.sqrt().mean().detach()
            if self.layer_budget_strategy not in LAYER_BUDGET_SCORE_STRATEGIES:
                raise AssertionError(f"Unhandled layer budget strategy: {self.layer_budget_strategy}")

            layer_scores = _combine_layer_budget_pr_base(
                _layer_score_leverage_pr(score_input, self.layer_budget_eps),
                self.layer_budget_strategy,
                self.layer_budget_eps,
            )
            base_score = layer_scores.mean()
            if self.layer_budget_strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                norm_source = candidate_v if self.layer_budget_norm_source == "value" else candidate_k
                if norm_source is None or norm_source.numel() == 0:
                    value_norm = torch.zeros((), device=policy_scores.device, dtype=torch.float32)
                else:
                    if norm_source.ndim != 4:
                        raise ValueError(
                            "Layer budget norm source requires tensor shaped [B, H, N, D], "
                            f"got {tuple(norm_source.shape)}"
                        )
                    token_sq_norms = torch.nan_to_num(
                        norm_source.float(), nan=0.0, posinf=0.0, neginf=0.0
                    ).square().sum(dim=(1, 3))
                    if self.layer_budget_value_norm_type == "mean":
                        value_norm = token_sq_norms.sqrt().mean()
                    elif self.layer_budget_value_norm_type == "rms":
                        value_norm = token_sq_norms.mean(dim=1).clamp_min(self.layer_budget_eps).sqrt().mean()
                    else:
                        raise ValueError(
                            "layer_budget_value_norm_type must be 'mean' or 'rms', "
                            f"got {self.layer_budget_value_norm_type!r}"
                        )
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


    def _select_svd_leverage_kept(
        self,
        scores: torch.Tensor,
        num_to_keep: int,
        *,
        num_heads: int,
    ) -> torch.Tensor:
        if self.leverage_eviction_selector != "topk":
            raise AssertionError(f"Unhandled leverage eviction selector: {self.leverage_eviction_selector}")
        if scores.ndim != 2:
            raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
        _, kept_2d = torch.topk(scores, k=int(num_to_keep), dim=-1, sorted=False)
        kept_2d = kept_2d.sort(dim=-1).values
        return kept_2d.unsqueeze(1).expand(scores.shape[0], int(num_heads), kept_2d.shape[-1])

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

        return scores * gate.to(device=scores.device, dtype=scores.dtype)

    @staticmethod
    def _blend_attention_utility(
        scores: torch.Tensor,
        attention_utility: Optional[torch.Tensor],
        beta: float,
        *,
        shared_across_heads: bool,
        attention_observed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        beta = float(beta)
        if attention_utility is None or beta == 0.0:
            return scores
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"attention_utility_beta must be in [0, 1], got {beta}")

        utility = attention_utility.to(device=scores.device, dtype=torch.float32)
        if shared_across_heads:
            if scores.ndim != 2 or tuple(utility.shape) != tuple(scores.shape):
                raise ValueError(
                    "Layer-shared attention utility and scores must both have shape [B, N], "
                    f"got scores={tuple(scores.shape)} utility={tuple(utility.shape)}"
                )
        else:
            if scores.ndim != 3:
                raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")
            if utility.ndim == 2:
                utility = utility.unsqueeze(1).expand_as(scores)
            elif tuple(utility.shape) != tuple(scores.shape):
                raise ValueError(
                    "Head-wise attention utility must have shape [B, N] or [B, H, N], "
                    f"got {tuple(utility.shape)}"
                )

        if attention_observed is None:
            observed = torch.ones_like(utility, dtype=torch.bool)
        else:
            observed = attention_observed.to(device=scores.device, dtype=torch.bool)
            if shared_across_heads:
                if tuple(observed.shape) != tuple(scores.shape):
                    raise ValueError(
                        "Layer-shared attention observation mask and scores must have the same shape, "
                        f"got scores={tuple(scores.shape)} observed={tuple(observed.shape)}"
                    )
            elif observed.ndim == 2:
                observed = observed.unsqueeze(1).expand_as(scores)
            elif tuple(observed.shape) != tuple(scores.shape):
                raise ValueError(
                    "Head-wise attention observation mask must have shape [B, N] or [B, H, N], "
                    f"got {tuple(observed.shape)}"
                )

        def mean_normalize(
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            clean = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
            clean = clean.clamp_min(0.0)
            if mask is None:
                value_mean = clean.mean(dim=-1, keepdim=True)
            else:
                mask_float = mask.to(dtype=clean.dtype)
                value_sum = (clean * mask_float).sum(dim=-1, keepdim=True)
                value_count = mask_float.sum(dim=-1, keepdim=True)
                value_mean = value_sum / value_count.clamp_min(1.0)
            normalized = clean / value_mean.clamp_min(1e-12)
            return torch.where(value_mean > 1e-12, normalized, torch.zeros_like(normalized))

        normalized_scores = mean_normalize(scores)
        normalized_utility = mean_normalize(utility, observed)
        blended = (1.0 - beta) * normalized_scores + beta * normalized_utility
        result = torch.where(observed, blended, normalized_scores)
        any_observed = observed.any(dim=-1, keepdim=True)
        return torch.where(any_observed, result, scores.float())

    @staticmethod
    def _prioritize_current_frame_for_attention_only(
        scores: torch.Tensor,
        *,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_idx: Optional[int],
        attention_utility_beta: float,
        attention_utility_enabled: bool,
        current_frame_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if (
            not attention_utility_enabled
            or float(attention_utility_beta) != 1.0
            or candidate_frame_ids is None
        ):
            return scores
        if scores.ndim != 2 or tuple(candidate_frame_ids.shape) != tuple(scores.shape):
            raise ValueError(
                "Current-frame attention-only prioritization requires scores and "
                "candidate_frame_ids shaped [B, N], "
                f"got scores={tuple(scores.shape)} frame_ids={tuple(candidate_frame_ids.shape)}"
            )

        if current_frame_ids is None:
            if current_frame_idx is None:
                return scores
            resolved_current_frame_ids = [int(current_frame_idx)]
        else:
            resolved_current_frame_ids = [int(frame_id) for frame_id in current_frame_ids]
            if not resolved_current_frame_ids:
                if current_frame_idx is None:
                    return scores
                resolved_current_frame_ids = [int(current_frame_idx)]

        current_ids = torch.as_tensor(
            resolved_current_frame_ids,
            device=scores.device,
            dtype=candidate_frame_ids.dtype,
        )
        current_mask = candidate_frame_ids.to(device=scores.device).unsqueeze(-1).eq(current_ids).any(dim=-1)
        prioritized = scores.float()
        max_score = torch.finfo(prioritized.dtype).max
        return prioritized.masked_fill(current_mask, max_score)

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

    def reset_projected_key_cache(self) -> None:
        self._projected_key_cache = None
        self._projected_key_pre_norm_cache = None
        self._projected_key_cache_meta = None
        self._last_projected_key_features = None
        self._last_projected_key_pre_norms = None
        self._last_projected_key_cache_meta = None

    def _projected_key_cache_meta_for(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        sketch_dim: int,
        projection_bypassed: bool,
        device: torch.device,
    ) -> Dict[str, object]:
        return {
            "batch_size": int(batch_size),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
            "sketch_dim": int(sketch_dim),
            "projection_bypassed": bool(projection_bypassed),
            "device": str(device),
            "normalize_before_projection": bool(self.leverage_normalize_before_projection),
            "normalize_before_projection_headwise": bool(self.leverage_normalize_before_projection_headwise),
        }

    def _projected_key_cache_length(
        self,
        *,
        batch_size: int,
        num_heads: int,
        num_tokens: int,
        head_dim: int,
        sketch_dim: int,
        projection_bypassed: bool,
        device: torch.device,
    ) -> int:
        if not self.leverage_projected_key_cache:
            return 0
        cache = self._projected_key_cache
        norm_cache = self._projected_key_pre_norm_cache
        meta = self._projected_key_cache_meta
        expected = self._projected_key_cache_meta_for(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            sketch_dim=sketch_dim,
            projection_bypassed=projection_bypassed,
            device=device,
        )
        if cache is None or norm_cache is None or meta != expected:
            return 0
        cache_len = int(cache.shape[1])
        if cache_len > int(num_tokens):
            self.reset_projected_key_cache()
            return 0
        if cache.shape != (batch_size, cache_len, sketch_dim):
            self.reset_projected_key_cache()
            return 0
        if norm_cache.shape != (batch_size, cache_len):
            self.reset_projected_key_cache()
            return 0
        return cache_len

    def _normalize_layer_key_before_projection(
        self,
        mat_k: torch.Tensor,
        profile: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if not self.leverage_normalize_before_projection:
            return mat_k
        do_profile = profile is not None and self.profile
        start = time.perf_counter() if do_profile else 0.0
        if self.leverage_normalize_before_projection_headwise:
            norm = mat_k.square().sum(dim=3, keepdim=True).sqrt().clamp_min(1e-12)
        else:
            norm = mat_k.square().sum(dim=(1, 3), keepdim=True).sqrt().clamp_min(1e-12)
        normalized = mat_k / norm
        if do_profile:
            self._sync_for_timing(normalized)
            elapsed = time.perf_counter() - start
            self._profile_add(profile, "pre_projection_normalization", elapsed)
            self._profile_add(profile, "normalization", elapsed)
        return normalized

    def _normalize_rows(
        self,
        x: torch.Tensor,
        profile: Optional[Dict[str, float]] = None,
        name: str = "normalization",
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

    def _project_key_with_omega(
        self,
        mat_k: torch.Tensor,
        omega_key: torch.Tensor,
        profile: Optional[Dict[str, float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        do_profile = profile is not None and self.profile
        prep_start = time.perf_counter() if do_profile else 0.0
        mat_k = torch.nan_to_num(mat_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(mat_k)
            self._profile_add(profile, "candidate_matrix_preparation", time.perf_counter() - prep_start)
        mat_k = self._normalize_layer_key_before_projection(mat_k, profile)
        B, H, N, D = mat_k.shape
        sketch_dim = int(omega_key.shape[-1])
        flat_k = mat_k.permute(0, 2, 1, 3).reshape(B, N, H * D)
        omega_flat = omega_key.reshape(H * D, sketch_dim)
        projection_start = time.perf_counter() if do_profile else 0.0
        layer_projected_raw = torch.matmul(flat_k, omega_flat)
        if do_profile:
            self._sync_for_timing(layer_projected_raw)
            self._profile_add(profile, "projection_matmul", time.perf_counter() - projection_start)
        pre_norms = torch.linalg.vector_norm(layer_projected_raw.detach(), ord=2, dim=-1)
        layer_projected = self._normalize_rows(
            layer_projected_raw,
            profile,
            "post_projection_normalization",
        )
        return layer_projected, pre_norms

    def _flatten_key_without_projection(
        self,
        mat_k: torch.Tensor,
        profile: Optional[Dict[str, float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        do_profile = profile is not None and self.profile
        prep_start = time.perf_counter() if do_profile else 0.0
        mat_k = torch.nan_to_num(mat_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(mat_k)
            self._profile_add(profile, "candidate_matrix_preparation", time.perf_counter() - prep_start)
        mat_k = self._normalize_layer_key_before_projection(mat_k, profile)
        B, H, N, D = mat_k.shape
        flat_k = mat_k.permute(0, 2, 1, 3).reshape(B, N, H * D)
        pre_norms = torch.linalg.vector_norm(flat_k.detach(), ord=2, dim=-1)
        layer_features = self._normalize_rows(
            flat_k,
            profile,
            "post_projection_normalization",
        )
        return layer_features, pre_norms

    def _capture_projected_pre_norms(self, norms: torch.Tensor) -> None:
        if getattr(self, "_capture_projected_norms", False):
            self._last_projected_pre_norms = norms.detach()

    def _layer_key_projected_features(
        self,
        mat_k: torch.Tensor,
        omega_key: Optional[torch.Tensor],
        sketch_dim: int,
        profile: Dict[str, float],
    ) -> torch.Tensor:
        B, H, N, D = mat_k.shape
        meta = self._projected_key_cache_meta_for(
            batch_size=B,
            num_heads=H,
            head_dim=D,
            sketch_dim=sketch_dim,
            projection_bypassed=omega_key is None,
            device=mat_k.device,
        )
        cache_len = self._projected_key_cache_length(
            batch_size=B,
            num_heads=H,
            num_tokens=N,
            head_dim=D,
            sketch_dim=sketch_dim,
            projection_bypassed=omega_key is None,
            device=mat_k.device,
        )
        if cache_len > 0:
            layer_parts = [self._projected_key_cache[:, :cache_len].to(device=mat_k.device)]
            norm_parts = [self._projected_key_pre_norm_cache[:, :cache_len].to(device=mat_k.device)]
            if cache_len < N:
                if omega_key is None:
                    suffix_layer, suffix_norms = self._flatten_key_without_projection(
                        mat_k[:, :, cache_len:, :],
                        profile,
                    )
                else:
                    suffix_layer, suffix_norms = self._project_key_with_omega(
                        mat_k[:, :, cache_len:, :],
                        omega_key,
                        profile,
                    )
                layer_parts.append(suffix_layer)
                norm_parts.append(suffix_norms)
            leverage_matrix = torch.cat(layer_parts, dim=1)
            pre_norms = torch.cat(norm_parts, dim=1)
            profile["projection_cache_hits"] = float(cache_len)
            profile["projection_cache_misses"] = float(N - cache_len)
        else:
            if omega_key is None:
                leverage_matrix, pre_norms = self._flatten_key_without_projection(
                    mat_k,
                    profile,
                )
            else:
                leverage_matrix, pre_norms = self._project_key_with_omega(
                    mat_k,
                    omega_key,
                    profile,
                )
            profile["projection_cache_hits"] = 0.0
            profile["projection_cache_misses"] = float(N)

        self._capture_projected_pre_norms(pre_norms)
        self._last_projected_key_features = leverage_matrix.detach()
        self._last_projected_key_pre_norms = pre_norms.detach()
        self._last_projected_key_cache_meta = meta
        return leverage_matrix

    def update_projected_key_cache_after_eviction(
        self,
        kept_candidate_indices: torch.Tensor,
        *,
        tail_k: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.leverage_projected_key_cache:
            return
        features = self._last_projected_key_features
        pre_norms = self._last_projected_key_pre_norms
        meta = self._last_projected_key_cache_meta
        if features is None or pre_norms is None or meta is None:
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
        row_indices = kept_candidate_indices[:, 0, :].to(device=features.device, dtype=torch.long)
        gather_layer = row_indices.unsqueeze(-1).expand(B, row_indices.shape[1], S)
        next_features = torch.gather(features, 1, gather_layer)
        next_norms = torch.gather(pre_norms, 1, row_indices)

        if tail_k is not None and int(tail_k.shape[2]) > 0:
            with torch.amp.autocast("cuda", enabled=False):
                tail = tail_k
                feature_dim = H * int(meta["head_dim"])
                sketch_dim = int(meta["sketch_dim"])
                if bool(meta.get("projection_bypassed", False)):
                    tail_features, tail_norms = self._flatten_key_without_projection(tail)
                else:
                    omega = self._get_leverage_right_sketch(
                        feature_dim,
                        sketch_dim,
                        device=tail.device,
                        seed=self.leverage_random_seed,
                    )
                    omega_key = omega.view(H, int(meta["head_dim"]), sketch_dim)
                    tail_features, tail_norms = self._project_key_with_omega(
                        tail,
                        omega_key,
                    )
            next_features = torch.cat([next_features, tail_features.detach()], dim=1)
            next_norms = torch.cat([next_norms, tail_norms.detach()], dim=1)

        self._projected_key_cache = next_features.detach()
        self._projected_key_pre_norm_cache = next_norms.detach()
        self._projected_key_cache_meta = dict(meta)

    def commit_projected_key_cache_without_eviction(self) -> None:
        """Promote all most-recent projected candidates into the persistent cache."""
        if not self.leverage_projected_key_cache:
            return
        features = self._last_projected_key_features
        pre_norms = self._last_projected_key_pre_norms
        meta = self._last_projected_key_cache_meta
        if features is None or pre_norms is None or meta is None:
            self.reset_projected_key_cache()
            return
        self._projected_key_cache = features.detach()
        self._projected_key_pre_norm_cache = pre_norms.detach()
        self._projected_key_cache_meta = dict(meta)

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

    def _resolve_leverage_approx_method(self, sketch_dim: Optional[int]) -> str:
        if self.leverage_approx_method != "right_sketch_ridge":
            raise ValueError(
                f"Only leverage_approx_method='right_sketch_ridge' is supported, got {self.leverage_approx_method!r}"
            )
        return self.leverage_approx_method

    def _resolve_ridge_sketch_dim(self, feature_dim: int, num_tokens: int) -> int:
        requested = self.leverage_ridge_dim
        if requested is None or int(requested) < 1:
            raise ValueError("right_sketch_ridge requires leverage_ridge_dim >= 1")
        # return min(int(requested), int(feature_dim), int(num_tokens))
        return int(requested)

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

    def _resolve_rls_frame_idx(self) -> int:
        for key in ("current_frame_idx", "step_idx"):
            value = self._rls_context.get(key)
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

    def _set_rls_context(
        self,
        *,
        step_idx: Optional[int],
        current_frame_idx: Optional[int],
    ) -> None:
        self._rls_context = {
            "step_idx": step_idx,
            "current_frame_idx": current_frame_idx,
        }

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

        with torch.amp.autocast("cuda", enabled=False):
            work = mat
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

                eye = torch.eye(feature_dim, device=work.device, dtype=torch.float32)
                eye = eye.view(*((1,) * (work.ndim - 2)), feature_dim, feature_dim)
                chol_start = time.perf_counter() if do_profile else 0.0
                last_a = None
                jitter_multipliers = (
                    (1.0,)
                    if float(self.leverage_ridge_jitter) == 0.0
                    else (1.0, 10.0, 100.0, 1000.0)
                )
                for retry_idx, jitter_multiplier in enumerate(jitter_multipliers):
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
                        f"info_max={info_max}",
                        file=sys.stderr,
                        flush=True,
                    )
                    chol = None
                    retries = retry_idx + 1
                if chol is None and last_a is not None:
                    print(
                        "[RLS][pinv_fallback] "
                        f"method={basis_kind} granularity={granularity} frame={frame_idx} "
                        f"N={num_tokens} D={feature_dim} retries={retries}",
                        file=sys.stderr,
                        flush=True,
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
                            f"N={num_tokens} D={feature_dim} error={exc}",
                            file=sys.stderr,
                            flush=True,
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
                print(
                    "[RLS][norm_fallback] "
                    f"method={basis_kind} granularity={granularity} frame={frame_idx} "
                    f"N={num_tokens} D={feature_dim} reason=no_valid_cholesky_or_pinv",
                    file=sys.stderr,
                    flush=True,
                )
                scores = work.square().sum(dim=-1)
                fallback = max(fallback_state, 2.0)
                score_backend = "norm_fallback"
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
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
        feature_dim = H * D * (2 if self.leverage_feature == "key_value" else 1)
        self._last_layer_feature_shape = (int(N), int(feature_dim))

        self._resolve_leverage_approx_method(self.leverage_ridge_dim)
        return self._layer_svd_leverage_scores_sketched(
            candidate_k,
            candidate_v,
            feature_dim,
            return_basis=return_basis,
        )

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
        self._resolve_leverage_approx_method(self.leverage_ridge_dim)
        sketch_dim = self._resolve_ridge_sketch_dim(feature_dim, N)
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

        with torch.amp.autocast("cuda", enabled=False):
            bypass_projection = sketch_dim == feature_dim == H * D
            # Right-sketched leverage / Compactor-style approximation for the
            # concatenated layer feature matrix, applied without materializing
            # the full [B, N, H * D] matrix first.
            omega_key = None
            if bypass_projection:
                profile["sketch_matrix_retrieval"] = 0.0
                profile["projection_matmul"] = 0.0
                profile["projection_bypassed"] = 1.0
            else:
                sketch_retrieval_start = time.perf_counter() if do_profile else 0.0
                omega = self._get_leverage_right_sketch(
                    feature_dim,
                    sketch_dim,
                    device=candidate_k.device,
                    seed=self.leverage_random_seed,
                )
                if do_profile:
                    self._sync_for_timing(omega)
                    profile["sketch_matrix_retrieval"] = time.perf_counter() - sketch_retrieval_start
                omega_key = omega[: H * D].view(H, D, sketch_dim)
                profile["projection_bypassed"] = 0.0
            if self.leverage_projected_key_cache:
                projection_start = time.perf_counter() if do_profile else 0.0
                leverage_matrix = self._layer_key_projected_features(
                    candidate_k,
                    omega_key,
                    sketch_dim,
                    profile,
                )
                if do_profile and "projection_matmul" not in profile:
                    self._sync_for_timing(leverage_matrix)
                    profile["projection_matmul"] = time.perf_counter() - projection_start
            else:
                if bypass_projection:
                    leverage_matrix, pre_norms = self._flatten_key_without_projection(
                        candidate_k,
                        profile if do_profile else None,
                    )
                else:
                    leverage_matrix, pre_norms = self._project_key_with_omega(
                        candidate_k,
                        omega_key,
                        profile if do_profile else None,
                    )
                self._capture_projected_pre_norms(pre_norms)

            if do_profile:
                profile["feature"] = 0.0
                profile["sketch"] = profile["sketch_matrix_retrieval"] + profile["projection_matmul"]

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
