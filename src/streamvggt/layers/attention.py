import logging
import os
import time
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional, Sequence, Set

from streamvggt.utils.cache_analysis import (
    CacheAnalysisConfig,
    EvictionNNAnalysisConfig,
    LeverageScoreHistogramConfig,
    ProjectedNormHistogramConfig,
    PreEvictionSnapshotConfig,
    TokenOverlayDumpConfig,
    dump_eviction_nn_analysis,
    dump_eviction_snapshot,
    dump_pre_eviction_snapshot,
    dump_token_overlay_event,
)
from streamvggt.layers.eviction import EvictionManager
from streamvggt.layers.confidence_state import KVConfidenceState, pack_kv_cache, parse_confidence_gate_init, unpack_kv_cache

XFORMERS_AVAILABLE = False


def scale_stac_colsum(colsum: torch.Tensor, live_tokens: int, query_tokens: int) -> torch.Tensor:
    """Convert STAC per-head column sums into a cache-length-stable token score."""
    if colsum.ndim != 3:
        raise ValueError(f"colsum must have shape [B, H, N], got {tuple(colsum.shape)}")
    if int(live_tokens) != int(colsum.shape[-1]):
        raise ValueError(f"live_tokens={live_tokens} does not match colsum keys={colsum.shape[-1]}")
    if int(query_tokens) <= 0:
        raise ValueError(f"query_tokens must be positive, got {query_tokens}")
    return colsum.float().mean(dim=1) * (float(live_tokens) / float(query_tokens))


def _materialize_kv_by_keep_indices(
    k: torch.Tensor,
    v: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    head_shared: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, _, D = k.shape
    final_cache_size = keep_indices.shape[2]
    if head_shared and B == 1:
        indices = keep_indices[0, 0].to(device=k.device, dtype=torch.long)
        return k.index_select(2, indices), v.index_select(2, indices)

    expanded_indices = keep_indices.to(device=k.device, dtype=torch.long).unsqueeze(-1)
    expanded_indices = expanded_indices.expand(B, H, final_cache_size, D)
    return torch.gather(k, 2, expanded_indices), torch.gather(v, 2, expanded_indices)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0
        self._eviction_managers = {}
        self._eviction_profile_totals = {}
        self._eviction_profile_count = 0
        self._attention_forward_profile_totals = {}
        self._attention_forward_profile_count = 0
        self._perf_trace_enabled = False
        self._perf_trace_events = []
        self._perf_trace_totals = {}

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0
        self._eviction_profile_totals = {}
        self._eviction_profile_count = 0
        self._attention_forward_profile_totals = {}
        self._attention_forward_profile_count = 0
        self.reset_perf_trace()
        for eviction in self._eviction_managers.values():
            reset = getattr(eviction, "reset_projected_key_cache", None)
            if reset is not None:
                reset()
            reset_rls = getattr(eviction, "reset_rls_cache", None)
            if reset_rls is not None:
                reset_rls()
            reset_profile = getattr(eviction, "reset_profile_stats", None)
            if reset_profile is not None:
                reset_profile()

    def reset_profile_stats(self):
        self._eviction_profile_totals = {}
        self._eviction_profile_count = 0
        self._attention_forward_profile_totals = {}
        self._attention_forward_profile_count = 0
        for eviction in self._eviction_managers.values():
            reset_profile = getattr(eviction, "reset_profile_stats", None)
            if reset_profile is not None:
                reset_profile()

    def reset_perf_trace(self):
        self._perf_trace_events = []
        self._perf_trace_totals = {}
        for eviction in self._eviction_managers.values():
            reset = getattr(eviction, "reset_perf_trace", None)
            if reset is not None:
                reset()

    def _perf_trace_start(self, tensor: torch.Tensor):
        if not self._perf_trace_enabled or not tensor.is_cuda:
            return None, 0.0
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event, time.perf_counter()

    def _perf_trace_end(self, name: str, start_event, start_cpu: float, **counts) -> None:
        if start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._perf_trace_events.append((name, start_event, end_event))
        self._perf_trace_totals[f"{name}_cpu_enqueue"] = (
            self._perf_trace_totals.get(f"{name}_cpu_enqueue", 0.0)
            + (time.perf_counter() - start_cpu) * 1000.0
        )
        for key, value in counts.items():
            self._perf_trace_totals[key] = self._perf_trace_totals.get(key, 0.0) + float(value)

    def get_perf_trace_stats(self):
        totals = dict(self._perf_trace_totals)
        count = 0
        for name, start_event, end_event in self._perf_trace_events:
            totals[f"{name}_cuda"] = totals.get(f"{name}_cuda", 0.0) + float(start_event.elapsed_time(end_event))
            if name == "select":
                count += 1
        for eviction in self._eviction_managers.values():
            get_stats = getattr(eviction, "get_perf_trace_stats", None)
            if get_stats is None:
                continue
            for name, value in get_stats().items():
                key = f"manager_{name}"
                totals[key] = totals.get(key, 0.0) + float(value)
        return {"count": count, "totals": totals}

    def _profile_sync(self, tensor: Optional[torch.Tensor]) -> None:
        if tensor is not None and tensor.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(tensor.device)

    def _profile_start(self, tensor: Optional[torch.Tensor], enabled: bool) -> float:
        if not enabled:
            return 0.0
        self._profile_sync(tensor)
        return time.perf_counter()

    def _record_forward_profile(self, name: str, start: float, tensor: Optional[torch.Tensor], enabled: bool) -> None:
        if not enabled:
            return
        self._profile_sync(tensor)
        self._attention_forward_profile_totals[name] = (
            self._attention_forward_profile_totals.get(name, 0.0) + (time.perf_counter() - start)
        )

    def _record_eviction_profile(self, **metrics):
        self._eviction_profile_count += 1
        for name, value in metrics.items():
            self._eviction_profile_totals[name] = self._eviction_profile_totals.get(name, 0.0) + float(value)

    def get_eviction_profile_stats(self):
        leverage_totals = {}
        leverage_count = 0
        for eviction in self._eviction_managers.values():
            get_stats = getattr(eviction, "get_profile_stats", None)
            if get_stats is None:
                continue
            stats = get_stats()
            leverage_count += int(stats.get("count", 0))
            for name, value in stats.get("totals", {}).items():
                leverage_totals[name] = leverage_totals.get(name, 0.0) + float(value)
        return {
            "attention_count": self._eviction_profile_count,
            "attention_totals": dict(self._eviction_profile_totals),
            "attention_forward_count": self._attention_forward_profile_count,
            "attention_forward_totals": dict(self._attention_forward_profile_totals),
            "leverage_count": leverage_count,
            "leverage_totals": leverage_totals,
        }

    def eviction(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        confidence_state: Optional[KVConfidenceState] = None,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        leverage_score_histogram_config: Optional[LeverageScoreHistogramConfig] = None,
        projected_norm_histogram_config: Optional[ProjectedNormHistogramConfig] = None,
        token_overlay_dump_config: Optional[TokenOverlayDumpConfig] = None,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        current_frame_idx: Optional[int] = None,
        current_frame_ids: Optional[Sequence[int]] = None,
        tokens_per_frame: Optional[int] = None,
        eviction_policy: str = "mean",
        eviction_policy_layers: Optional[Set[int]] = None,
        profile_eviction: bool = False,
        eviction_debug: bool = False,
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
        leverage_ridge_dim: Optional[int] = None,
        rls_refresh_interval: int = 1,
        leverage_random_seed: int = 0,
        leverage_eviction_selector: str = "topk",
        leverage_conf_gate: bool = False,
        leverage_conf_gate_floor: float = 0.0,
        leverage_conf_gate_depth_alpha: float = 1.0,
        leverage_conf_gate_point_beta: float = 0.0,
        leverage_attention_beta: float = 0.3,
        layer_budget_strategy: str = "value_weighted_leverage_pr",
        layer_budget_value_gamma: float = 0.7,
        layer_budget_value_norm_type: str = "mean",
        layer_budget_norm_source: str = "key",
        layer_budget_eps: float = 0,
        layer_budget_score_only: bool = False,
    ):
        """
        Evicts tokens from the key-value cache based on key cosine similarity.

        Args:
            k (torch.Tensor): The key tensor of shape [B, H, N, D].
            v (torch.Tensor): The value tensor of shape [B, H, N, D].
            cache_budget (int): The maximum number of tokens to retain.
            num_anchor_tokens (int): The number of initial tokens to preserve.

        Returns:
            A tuple of pruned key and value tensors.
        """
        B, H, N, D = k.shape
        num_anchor_tokens = max(0, min(int(num_anchor_tokens), N))
        num_candidates = N - num_anchor_tokens
        num_to_keep = max(0, min(int(cache_budget) - num_anchor_tokens, num_candidates))

        if not layer_budget_score_only and (N <= cache_budget or N <= num_anchor_tokens):
            if confidence_state is not None:
                return k, v, confidence_state, 0.0
            return k, v, 0.0

        if not layer_budget_score_only and cache_budget <= num_anchor_tokens:
            keep_indices = torch.arange(num_anchor_tokens, device=k.device, dtype=torch.long)
            keep_indices = keep_indices.view(1, 1, num_anchor_tokens).expand(B, H, num_anchor_tokens)
            final_k, final_v = _materialize_kv_by_keep_indices(
                k, v, keep_indices, head_shared=True
            )
            final_confidence_state = (
                confidence_state.gather(keep_indices)
                if confidence_state is not None
                else None
            )
            if final_confidence_state is not None:
                return final_k, final_v, final_confidence_state, 0.0
            return final_k, final_v, 0.0

        select_k = k
        select_v = v
        select_confidence_state = confidence_state
        selection_budget = cache_budget
        select_trace_start, select_trace_cpu_start = self._perf_trace_start(k)

        profile_eviction = bool(profile_eviction)
        if profile_eviction and k.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(k.device)
        eviction_total_start = time.perf_counter() if profile_eviction else 0.0

        manager_key = (
            eviction_policy,
            profile_eviction,
            eviction_debug,
            leverage_granularity,
            leverage_feature,
            leverage_projection,
            leverage_normalize_rows,
            leverage_normalize_before_projection,
            leverage_normalize_before_projection_headwise,
            leverage_projected_key_cache,
            leverage_approx_method,
            leverage_ridge_lambda,
            leverage_ridge_lambda_mode,
            leverage_ridge_score_chunk_size,
            leverage_ridge_jitter,
            leverage_ridge_dim,
            rls_refresh_interval,
            leverage_random_seed,
            leverage_eviction_selector,
            leverage_conf_gate,
            leverage_conf_gate_floor,
            leverage_conf_gate_depth_alpha,
            leverage_conf_gate_point_beta,
            layer_budget_strategy,
            layer_budget_value_gamma,
            layer_budget_value_norm_type,
            layer_budget_norm_source,
            layer_budget_eps,
        )
        eviction = self._eviction_managers.get(manager_key)
        if eviction is None:
            eviction = EvictionManager(
                policy=eviction_policy,
                profile=profile_eviction,
                debug=eviction_debug,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
                leverage_projection=leverage_projection,
                leverage_normalize_rows=leverage_normalize_rows,
                leverage_normalize_before_projection=leverage_normalize_before_projection,
                leverage_normalize_before_projection_headwise=leverage_normalize_before_projection_headwise,
                leverage_projected_key_cache=leverage_projected_key_cache,
                leverage_approx_method=leverage_approx_method,
                leverage_ridge_lambda=leverage_ridge_lambda,
                leverage_ridge_lambda_mode=leverage_ridge_lambda_mode,
                leverage_ridge_score_chunk_size=leverage_ridge_score_chunk_size,
                leverage_ridge_jitter=leverage_ridge_jitter,
                leverage_ridge_dim=leverage_ridge_dim,
                rls_refresh_interval=rls_refresh_interval,
                leverage_random_seed=leverage_random_seed,
                leverage_eviction_selector=leverage_eviction_selector,
                leverage_conf_gate=leverage_conf_gate,
                leverage_conf_gate_floor=leverage_conf_gate_floor,
                leverage_conf_gate_depth_alpha=leverage_conf_gate_depth_alpha,
                leverage_conf_gate_point_beta=leverage_conf_gate_point_beta,
                layer_budget_strategy=layer_budget_strategy,
                layer_budget_value_gamma=layer_budget_value_gamma,
                layer_budget_value_norm_type=layer_budget_value_norm_type,
                layer_budget_norm_source=layer_budget_norm_source,
                layer_budget_eps=layer_budget_eps,
            )
            self._eviction_managers[manager_key] = eviction

        eviction._perf_trace_enabled = self._perf_trace_enabled

        rls_refresh_before = eviction.rls_refresh_count
        rls_cache_hits_before = eviction.rls_cache_hit_count

        if layer_budget_score_only:
            score_start = time.perf_counter() if profile_eviction else 0.0
            score_result = eviction.score_layer_budget(
                k,
                num_anchor_tokens,
                v=v,
                layer_id=layer_id,
                step_idx=step_idx,
                current_frame_idx=current_frame_idx if current_frame_idx is not None else step_idx,
                capture_projected_norms=projected_norm_histogram_config is not None,
            )
            if profile_eviction and k.is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize(k.device)
            score_time = time.perf_counter() - score_start if profile_eviction else 0.0
            if (
                leverage_score_histogram_config is not None
                and layer_id is not None
                and step_idx is not None
            ):
                leverage_score_histogram_config.record(
                    score_result.policy_scores,
                    layer_id=layer_id,
                    step_idx=step_idx,
                )
            if (
                projected_norm_histogram_config is not None
                and layer_id is not None
                and step_idx is not None
                and eviction._last_projected_pre_norms is not None
            ):
                projected_norm_histogram_config.record(
                    eviction._last_projected_pre_norms,
                    layer_id=layer_id,
                    step_idx=step_idx,
                )
            if profile_eviction:
                total_time = time.perf_counter() - eviction_total_start
                self._record_eviction_profile(
                    manager_score_only=score_time,
                    total_score_only=total_time,
                )
            score_payload = (0.0, score_result.layer_budget_score)
            if confidence_state is not None:
                return k, v, confidence_state, score_payload
            return k, v, score_payload

        selection_start = time.perf_counter() if profile_eviction else 0.0
        candidate_frame_ids = (
            select_confidence_state.frame_ids[:, num_anchor_tokens:]
            if select_confidence_state is not None
            else None
        )
        candidate_conf_gate = (
            select_confidence_state.confidence_gate[:, num_anchor_tokens:]
            if select_confidence_state is not None
            else None
        )
        candidate_attention_utility = (
            select_confidence_state.attention_utility()[:, num_anchor_tokens:]
            if select_confidence_state is not None and select_confidence_state.has_attention_utility
            else None
        )
        candidate_attention_observed = (
            select_confidence_state.attention_count[:, num_anchor_tokens:].gt(0)
            if select_confidence_state is not None
            and select_confidence_state.has_attention_utility
            and select_confidence_state.attention_count is not None
            else None
        )
        candidate_depth_confidence = None
        candidate_point_confidence = None
        eviction_result = eviction.select(
            select_k,
            selection_budget,
            num_anchor_tokens,
            v=select_v,
            need_summary=cache_analysis_config is not None or eviction_debug,
            layer_id=layer_id,
            step_idx=step_idx,
            current_frame_idx=current_frame_idx if current_frame_idx is not None else step_idx,
            current_frame_ids=current_frame_ids,
            candidate_frame_ids=candidate_frame_ids,
            candidate_depth_confidence=candidate_depth_confidence,
            candidate_point_confidence=candidate_point_confidence,
            candidate_conf_gate=candidate_conf_gate,
            candidate_attention_utility=candidate_attention_utility,
            candidate_attention_observed=candidate_attention_observed,
            attention_utility_beta=leverage_attention_beta,
            need_leverage_basis=(
                eviction_nn_analysis_config is not None
                and eviction_nn_analysis_config.wants_svd_coord()
            ),
            capture_projected_norms=projected_norm_histogram_config is not None,
        )
        self._perf_trace_end(
            "select",
            select_trace_start,
            select_trace_cpu_start,
            cache_tokens=N,
            candidate_tokens=num_candidates,
            kept_candidate_tokens=num_to_keep,
            rls_refreshes=eviction.rls_refresh_count - rls_refresh_before,
            rls_cache_hits=eviction.rls_cache_hit_count - rls_cache_hits_before,
        )
        if profile_eviction and k.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(k.device)
        selection_time = time.perf_counter() - selection_start if profile_eviction else 0.0
        top_indices = eviction_result.kept_candidate_indices
        avg_scores = eviction_result.summary_score
        layer_budget_score = eviction_result.layer_budget_score

        if (
            leverage_score_histogram_config is not None
            and layer_id is not None
            and step_idx is not None
            and eviction_policy == "svd_leverage"
        ):
            leverage_score_histogram_config.record(
                eviction_result.policy_scores,
                layer_id=layer_id,
                step_idx=step_idx,
            )

        if (
            projected_norm_histogram_config is not None
            and layer_id is not None
            and step_idx is not None
            and eviction_policy == "svd_leverage"
            and eviction._last_projected_pre_norms is not None
        ):
            projected_norm_histogram_config.record(
                eviction._last_projected_pre_norms,
                layer_id=layer_id,
                step_idx=step_idx,
            )

        if (
            token_overlay_dump_config is not None
            and layer_id is not None
            and step_idx is not None
            and eviction_policy == "svd_leverage"
        ):
            dump_token_overlay_event(
                token_overlay_dump_config,
                kept_candidate_indices=top_indices,
                policy_scores=eviction_result.policy_scores,
                metadata=select_confidence_state,
                layer_id=layer_id,
                step_idx=step_idx,
                cache_budget=cache_budget,
                num_anchor_tokens=num_anchor_tokens,
                tokens_per_frame=tokens_per_frame,
                eviction_policy=eviction_policy,
                leverage_granularity=leverage_granularity,
                selection_granularity=leverage_granularity,
            )

        if eviction_nn_analysis_config is not None and layer_id is not None and step_idx is not None:
            dump_eviction_nn_analysis(
                eviction_nn_analysis_config,
                k_before=select_k,
                v_before=select_v,
                kept_candidate_indices=top_indices,
                policy_scores=eviction_result.policy_scores,
                leverage_basis=eviction_result.leverage_basis,
                metadata=None,
                layer_id=layer_id,
                step_idx=step_idx,
                cache_budget=cache_budget,
                num_anchor_tokens=num_anchor_tokens,
                eviction_policy=eviction_policy,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
                selection_granularity=leverage_granularity,
            )

        if cache_analysis_config is not None and layer_id is not None and step_idx is not None:
            dump_eviction_snapshot(
                cache_analysis_config,
                k_before=select_k,
                scores=eviction_result.mean_scores,
                kept_candidate_indices=top_indices,
                layer_id=layer_id,
                step_idx=step_idx,
                cache_budget=cache_budget,
                num_anchor_tokens=num_anchor_tokens,
                tokens_per_frame=tokens_per_frame,
                eviction_policy=eviction_policy,
                leverage_sketch_dim=leverage_ridge_dim,
            )

        index_update_start = time.perf_counter() if profile_eviction else 0.0
        update_trace_start, update_trace_cpu_start = self._perf_trace_start(k)
        anchor_indices = torch.arange(num_anchor_tokens, device=k.device, dtype=torch.long)
        anchor_indices = anchor_indices.view(1, 1, num_anchor_tokens).expand(B, H, num_anchor_tokens)
        keep_indices = torch.cat([anchor_indices, top_indices + num_anchor_tokens,], dim=2,)
        final_k, final_v = _materialize_kv_by_keep_indices(
            k, v, keep_indices, head_shared=True
        )
        final_confidence_state = (
            confidence_state.gather(keep_indices)
            if confidence_state is not None
            else None
        )
        eviction.update_projected_key_cache_after_eviction(
            top_indices,
            tail_k= None,
        )
        projected_cache = getattr(eviction, "_projected_key_cache", None)
        projected_cache_tokens = int(projected_cache.shape[1]) if projected_cache is not None else 0
        self._perf_trace_end(
            "index_update",
            update_trace_start,
            update_trace_cpu_start,
            projected_cache_tokens=projected_cache_tokens,
        )
        if profile_eviction:
            if final_k.is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize(final_k.device)
            index_update_time = time.perf_counter() - index_update_start
            total_time = time.perf_counter() - eviction_total_start
            self._record_eviction_profile(
                manager_select=selection_time,
                cache_index_update=index_update_time,
                total_eviction=total_time,
            )

        if layer_budget_score is not None:
            score_payload = (avg_scores, layer_budget_score)
        else:
            score_payload = avg_scores
        if final_confidence_state is not None:
            return final_k, final_v, final_confidence_state, score_payload
        return final_k, final_v, score_payload

    def forward(self,
        x: torch.Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_budget = None,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        leverage_score_histogram_config: Optional[LeverageScoreHistogramConfig] = None,
        projected_norm_histogram_config: Optional[ProjectedNormHistogramConfig] = None,
        token_overlay_dump_config: Optional[TokenOverlayDumpConfig] = None,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        current_frame_ids: Optional[Sequence[int]] = None,
        current_frame_idx: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        eviction_policy: str = "mean",
        eviction_policy_layers: Optional[Set[int]] = None,
        profile_eviction: bool = False,
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_normalize_before_projection: bool = False,
        leverage_normalize_before_projection_headwise: bool = False,
        leverage_projected_key_cache: bool = False,
        leverage_approx_method: str = "right_sketch",        leverage_ridge_lambda: float = 1e-3,
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
        leverage_conf_gate_init: str = "mean",
        leverage_attention_utility: bool = False,
        leverage_attention_beta: float = 0.2,
        leverage_attention_ema_decay: float = 0.9,
        leverage_attention_freeze_updates: int = 5,
        leverage_attention_colsum_subsample_ratio: float = 1.0,
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.5,
        layer_budget_value_norm_type: str = "rms",
        layer_budget_norm_source: str = "value",
        layer_budget_eps: float = 1e-12,
        layer_budget_score_only: bool = False,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        anchor_token_count: Optional[int] = None,
        cache_write_current_frame: bool = True,
        cache_evict_current_frame: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        B, N, C = x.shape
        layer_budget_score_only = bool(layer_budget_score_only)
        if layer_budget_score_only:
            if not use_cache:
                raise ValueError("Layer-budget score-only mode requires use_cache=True")
            if eviction_policy != "svd_leverage" or leverage_granularity != "layer":
                raise ValueError(
                    "Layer-budget score-only mode requires eviction_policy='svd_leverage' "
                    "and leverage_granularity='layer'"
                )
            if layer_budget_strategy != "value_weighted_leverage_pr":
                raise ValueError(
                    "Layer-budget score-only mode requires "
                    "layer_budget_strategy='value_weighted_leverage_pr'"
                )
            if leverage_attention_utility:
                raise ValueError("Layer-budget score-only mode is incompatible with leverage_attention_utility")
            incompatible_outputs = []
            if cache_analysis_config is not None:
                incompatible_outputs.append("cache_analysis_config")
            if eviction_nn_analysis_config is not None:
                incompatible_outputs.append("eviction_nn_analysis_config")
            if token_overlay_dump_config is not None:
                incompatible_outputs.append("token_overlay_dump_config")
            if incompatible_outputs:
                raise ValueError(
                    "Layer-budget score-only mode does not produce kept/evicted indices and is "
                    f"incompatible with: {', '.join(incompatible_outputs)}"
                )
        leverage_attention_utility = bool(leverage_attention_utility)
        if leverage_attention_utility:
            if self.training:
                raise RuntimeError("Frozen early-attention utility is inference-only")
            if not use_cache:
                raise ValueError("Frozen early-attention utility requires use_cache=True")
            if self.head_dim != 64:
                raise ValueError(f"STAC CUDA attention requires head_dim=64, got {self.head_dim}")
            if attn_mask is not None:
                raise ValueError("Frozen early-attention utility does not support attention masks")
            if eviction_policy != "svd_leverage" or leverage_granularity != "layer":
                raise ValueError("Frozen early-attention utility requires layer-wise svd_leverage eviction")
            if leverage_eviction_selector != "topk":
                raise ValueError("Frozen early-attention utility initially supports only layer-shared Top-K")
            if not cache_write_current_frame or not cache_evict_current_frame:
                raise ValueError("Frozen early-attention utility requires cache write and eviction on every current chunk")
            if not 0.0 <= float(leverage_attention_beta) <= 1.0:
                raise ValueError("leverage_attention_beta must be in [0, 1]")
            if not 0.0 <= float(leverage_attention_ema_decay) <= 1.0:
                raise ValueError("leverage_attention_ema_decay must be in [0, 1]")
            if not 1 <= int(leverage_attention_freeze_updates) <= 255:
                raise ValueError("leverage_attention_freeze_updates must be in [1, 255]")
            if not 0.0 < float(leverage_attention_colsum_subsample_ratio) <= 1.0:
                raise ValueError("leverage_attention_colsum_subsample_ratio must be in (0, 1]")
        profile_forward = bool(profile_eviction)
        forward_start = self._profile_start(x, profile_forward)
        qkv_start = self._profile_start(x, profile_forward)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        self._record_forward_profile("qkv_projection", qkv_start, q, profile_forward)
        scores = None
        norm_rope_start = self._profile_start(q, profile_forward)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        attention_output_dtype = q.dtype
        if leverage_attention_utility:
            attention_compute_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float16
            # Materialize only the current-chunk Q/K/V in SStream's native BHND
            # layout. The persistent K/V cache then stays BHND-contiguous, so
            # the full live cache never needs a transpose-contiguous copy.
            q = q.to(dtype=attention_compute_dtype).contiguous()
            k = k.to(dtype=attention_compute_dtype).contiguous()
            v = v.to(dtype=attention_compute_dtype).contiguous()
        self._record_forward_profile("qk_norm_rope", norm_rope_start, q, profile_forward)

        if use_cache and self.num_anchor_tokens == 0:
            if tokens_per_frame is not None and int(tokens_per_frame) > 0:
                # The first frame remains the permanent anchor even when the
                # initial streaming forward contains multiple frames.
                self.num_anchor_tokens = min(int(tokens_per_frame), int(k.shape[2]))
            else:
                # Preserve the legacy fallback for callers that do not provide
                # a frame layout.
                self.num_anchor_tokens = k.shape[2]

        if use_cache:
            cache_concat_start = self._profile_start(k, profile_forward)
            cache_write_current_frame = bool(cache_write_current_frame)
            cache_evict_current_frame = cache_write_current_frame and bool(cache_evict_current_frame)
            original_past_key_values = past_key_values
            current_k = k
            current_v = v
            confidence_state = None
            confidence_needed = (
                bool(leverage_conf_gate)
                or leverage_attention_utility
                or token_overlay_dump_config is not None
            )
            if current_frame_ids is None:
                resolved_current_frame_ids = [step_idx if step_idx is not None else 0]
            else:
                resolved_current_frame_ids = [int(frame_id) for frame_id in current_frame_ids]
            if len(resolved_current_frame_ids) <= 0:
                resolved_current_frame_ids = [step_idx if step_idx is not None else 0]
            if leverage_attention_utility and len(resolved_current_frame_ids) > 1:
                if (
                    tokens_per_frame is None
                    or int(tokens_per_frame) <= 0
                    or int(tokens_per_frame) * len(resolved_current_frame_ids) != int(current_k.shape[2])
                ):
                    raise ValueError(
                        "Chunked early-attention utility requires current K/V to have "
                        "tokens_per_frame * len(current_frame_ids) tokens, "
                        f"got tokens={current_k.shape[2]}, tokens_per_frame={tokens_per_frame}, "
                        f"frames={len(resolved_current_frame_ids)}"
                    )
            resolved_current_frame_idx = (
                int(current_frame_idx)
                if current_frame_idx is not None
                else int(resolved_current_frame_ids[-1])
            )

            def _current_confidence(num_tokens: int, initial_gate_value=None) -> KVConfidenceState:
                if (
                    tokens_per_frame is not None
                    and int(tokens_per_frame) > 0
                    and int(tokens_per_frame) * len(resolved_current_frame_ids) == int(num_tokens)
                ):
                    return KVConfidenceState.for_frame_chunk(
                        batch_size=B,
                        num_heads=self.num_heads,
                        tokens_per_frame=int(tokens_per_frame),
                        frame_ids=resolved_current_frame_ids,
                        device=current_k.device,
                        initial_gate=initial_gate_value,
                        initialize_attention=leverage_attention_utility,
                    )
                return KVConfidenceState.for_current_frame(
                    batch_size=B,
                    num_heads=self.num_heads,
                    num_tokens=num_tokens,
                    frame_id=resolved_current_frame_idx,
                    device=current_k.device,
                    initial_gate=initial_gate_value,
                    initialize_attention=leverage_attention_utility,
                )

            initial_confidence_gate = None
            if confidence_needed:
                parsed_conf_gate_init = parse_confidence_gate_init(leverage_conf_gate_init)
                if parsed_conf_gate_init == "mean":
                    if past_key_values is not None:
                        _, _, past_confidence_for_init = unpack_kv_cache(past_key_values)
                        if past_confidence_for_init is not None:
                            initial_confidence_gate = past_confidence_for_init.mean_gate().to(
                                device=current_k.device,
                                dtype=torch.float32,
                            )
                else:
                    initial_confidence_gate = parsed_conf_gate_init
                confidence_state = _current_confidence(current_k.shape[2], initial_confidence_gate)

            write_k = current_k
            write_v = current_v
            write_confidence_state = confidence_state

            if past_key_values is not None:
                past_k, past_v, past_confidence_state = unpack_kv_cache(past_key_values)
                if cache_write_current_frame:
                    k = torch.cat([past_k, write_k], dim=2)
                    v = torch.cat([past_v, write_v], dim=2)
                else:
                    k = past_k
                    v = past_v
                if write_confidence_state is not None and past_confidence_state is None:
                    past_confidence_state = KVConfidenceState.for_current_frame(
                        batch_size=B,
                        num_heads=self.num_heads,
                        num_tokens=past_k.shape[2],
                        frame_id=-1,
                        device=past_k.device,
                        initialize_attention=leverage_attention_utility,
                    )
                if leverage_attention_utility and past_confidence_state is not None:
                    past_confidence_state.ensure_attention_utility()
                if (
                    cache_write_current_frame
                    and write_confidence_state is not None
                    and past_confidence_state is not None
                ):
                    confidence_state = past_confidence_state.concat(write_confidence_state)
                elif cache_write_current_frame and write_confidence_state is not None:
                    confidence_state = None
                else:
                    confidence_state = past_confidence_state
            else:
                k = write_k if cache_write_current_frame else current_k
                v = write_v if cache_write_current_frame else current_v
                confidence_state = (
                    write_confidence_state if cache_write_current_frame else confidence_state
                )
            self._record_forward_profile("cache_concat_update", cache_concat_start, k, profile_forward)
            if (
                cache_write_current_frame
                and pre_eviction_snapshot_config is not None
                and layer_id is not None
                and step_idx is not None
            ):
                dump_pre_eviction_snapshot(
                    pre_eviction_snapshot_config,
                    k_cache=k,
                    v_cache=v,
                    layer_id=layer_id,
                    step_idx=step_idx,
                    cache_budget=cache_budget,
                    num_anchor_tokens=(
                        int(anchor_token_count)
                        if anchor_token_count is not None
                        else self.num_anchor_tokens
                    ),
                    tokens_per_frame=tokens_per_frame,
                )
            eviction_deferred_for_snapshot = (
                pre_eviction_snapshot_config is not None
                and step_idx is not None
                and step_idx <= pre_eviction_snapshot_config.target_step_idx
            )
            if (
                (layer_budget_score_only or cache_evict_current_frame)
                and cache_write_current_frame
                and cache_budget is not None
                and (layer_budget_score_only or k.shape[2] > cache_budget)
                and (layer_budget_score_only or not eviction_deferred_for_snapshot)
            ):
                eviction_kwargs = {
                    "cache_analysis_config": cache_analysis_config,
                    "eviction_nn_analysis_config": eviction_nn_analysis_config,
                    "leverage_score_histogram_config": leverage_score_histogram_config,
                    "projected_norm_histogram_config": projected_norm_histogram_config,
                    "token_overlay_dump_config": token_overlay_dump_config,
                    "layer_id": layer_id,
                    "step_idx": step_idx,
                    "current_frame_idx": resolved_current_frame_idx,
                    "current_frame_ids": resolved_current_frame_ids,
                    "tokens_per_frame": tokens_per_frame,
                    "eviction_policy": eviction_policy,
                    "eviction_policy_layers": eviction_policy_layers,
                    "profile_eviction": profile_eviction,
                    "eviction_debug": eviction_debug,
                    "leverage_granularity": leverage_granularity,
                    "leverage_feature": leverage_feature,
                    "leverage_projection": leverage_projection,
                    "leverage_normalize_before_projection": leverage_normalize_before_projection,
                    "leverage_normalize_before_projection_headwise": leverage_normalize_before_projection_headwise,
                    "leverage_projected_key_cache": leverage_projected_key_cache,
                    "leverage_approx_method": leverage_approx_method,                    "leverage_ridge_lambda": leverage_ridge_lambda,
                    "leverage_ridge_lambda_mode": leverage_ridge_lambda_mode,
                    "leverage_ridge_score_chunk_size": leverage_ridge_score_chunk_size,
                    "leverage_ridge_jitter": leverage_ridge_jitter,
                    "leverage_ridge_dim": leverage_ridge_dim,
                    "rls_refresh_interval": rls_refresh_interval,
                    "leverage_random_seed": leverage_random_seed,
                    "leverage_eviction_selector": leverage_eviction_selector,
                    "leverage_conf_gate": leverage_conf_gate,
                    "leverage_conf_gate_floor": leverage_conf_gate_floor,
                    "leverage_conf_gate_depth_alpha": leverage_conf_gate_depth_alpha,
                    "leverage_conf_gate_point_beta": leverage_conf_gate_point_beta,
                    "leverage_attention_beta": leverage_attention_beta,
                    "layer_budget_strategy": layer_budget_strategy,
                    "layer_budget_value_gamma": layer_budget_value_gamma,
                    "layer_budget_value_norm_type": layer_budget_value_norm_type,
                    "layer_budget_norm_source": layer_budget_norm_source,
                    "layer_budget_eps": layer_budget_eps,
                    "layer_budget_score_only": layer_budget_score_only,
                }
                if leverage_normalize_rows:
                    eviction_kwargs["leverage_normalize_rows"] = leverage_normalize_rows
                eviction_kwargs["leverage_normalize_before_projection"] = leverage_normalize_before_projection
                eviction_kwargs["leverage_normalize_before_projection_headwise"] = leverage_normalize_before_projection_headwise
                effective_anchor_count = (
                    int(anchor_token_count)
                    if anchor_token_count is not None
                    else self.num_anchor_tokens
                )
                eviction_result = self.eviction(
                    k,
                    v,
                    cache_budget,
                    effective_anchor_count,
                    confidence_state=confidence_state,
                    **eviction_kwargs,
                )
                if confidence_state is not None:
                    k, v, confidence_state, scores = eviction_result
                else:
                    k, v, scores = eviction_result

            cache_read_start = self._profile_start(k, profile_forward)
            if cache_write_current_frame and not leverage_attention_utility:
                new_kv = pack_kv_cache(k, v, confidence_state)
            elif leverage_attention_utility:
                new_kv = None
            else:
                new_kv = original_past_key_values

            read_k = k
            read_v = v
            read_confidence_state = confidence_state
            if not cache_write_current_frame:
                if past_key_values is not None:
                    read_k = torch.cat([k, current_k], dim=2)
                    read_v = torch.cat([v, current_v], dim=2)
                    if confidence_state is not None:
                        current_read_confidence = _current_confidence(current_k.shape[2])
                        read_confidence_state = confidence_state.concat(current_read_confidence)
                else:
                    read_k = current_k
                    read_v = current_v

            self._record_forward_profile("cache_read_prepare", cache_read_start, read_k, profile_forward)
            k_for_attn = read_k
            v_for_attn = read_v
        else:
            k_for_attn = k
            v_for_attn = v

        attention_kernel_start = self._profile_start(q, profile_forward)
        attention_colsum = None
        if leverage_attention_utility:
            if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    "STAC CUDA attention requires CUDA FP16/BF16 Q/K/V, "
                    f"got device={q.device} dtype={q.dtype}"
                )
            try:
                import attn_cuda
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Frozen early-attention utility requires the local attn-cuda extension. "
                    "Build it with 'pip install -e ./attn-cuda --no-build-isolation'."
                ) from exc
            if not attn_cuda.is_available():
                raise RuntimeError("attn-cuda reports that the CUDA extension is unavailable")
            if not q.is_contiguous() or not k_for_attn.is_contiguous() or not v_for_attn.is_contiguous():
                raise RuntimeError("SStream STAC Q/K/V cache tensors must be contiguous in [B, H, N, D] layout")
            x_stac, _, attention_colsum = attn_cuda.flash_attn_bias_colsum_bhnd(
                q,
                k_for_attn,
                v_for_attn,
                softmax_scale=self.scale,
                return_colsum=True,
                subsample_ratio=float(leverage_attention_colsum_subsample_ratio),
            )
            x = x_stac.to(dtype=attention_output_dtype)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k_for_attn.transpose(-2, -1)
            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, k_for_attn.shape[2]), (
                    f"Expected mask shape [..., {N}, {k_for_attn.shape[2]}], got {attn_mask.shape}"
                )
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v_for_attn

        if leverage_attention_utility:
            if confidence_state is None or attention_colsum is None:
                raise RuntimeError("Attention utility state or CUDA column-sum is missing")
            if k_for_attn.shape[2] != confidence_state.frame_ids.shape[1]:
                raise RuntimeError(
                    "Attention utility state must align with the live KV cache, "
                    f"got keys={k_for_attn.shape[2]} state={confidence_state.frame_ids.shape[1]}"
                )
            attention_score = scale_stac_colsum(
                attention_colsum,
                live_tokens=k_for_attn.shape[2],
                query_tokens=q.shape[2],
            )
            confidence_state.update_attention_utility(
                attention_score,
                ema_decay=float(leverage_attention_ema_decay),
                freeze_updates=int(leverage_attention_freeze_updates),
            )
            new_kv = pack_kv_cache(k, v, confidence_state)

        self._record_forward_profile("attention_kernel", attention_kernel_start, x, profile_forward)
        output_projection_start = self._profile_start(x, profile_forward)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self._record_forward_profile("output_projection", output_projection_start, x, profile_forward)
        self._record_forward_profile("forward_total", forward_start, x, profile_forward)
        self._attention_forward_profile_count += 1 if profile_forward else 0
        if use_cache:
            return x, new_kv, scores
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
