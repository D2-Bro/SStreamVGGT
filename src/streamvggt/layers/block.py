import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Union, Optional, Set
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        x: Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_budget=None,
        cache_analysis_config=None,
        pre_eviction_snapshot_config=None,
        eviction_nn_analysis_config=None,
        leverage_score_histogram_config=None,
        token_overlay_dump_config=None,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        eviction_policy: str = "mean",
        eviction_policy_layers: Optional[Set[int]] = None,
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_approx_method: str = "right_sketch",        leverage_ridge_lambda: float = 1e-3,
        leverage_ridge_lambda_mode: str = "relative",
        leverage_ridge_score_chunk_size: int = 4096,
        leverage_ridge_jitter: float = 1e-6,
        leverage_ridge_dim: Optional[int] = None,
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
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.5,
        layer_budget_value_norm_type: str = "rms",
        layer_budget_norm_source: str = "value",
        layer_budget_eps: float = 1e-12,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        eviction_protect_recent_frames: int = 0,
        eviction_protect_special_tokens: bool = False,
        eviction_protect_special_token_interval: int = 1,
        special_token_count: int = 0,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config=None,
        svd_eviction_merge_config=None,
        voxel_covis_frame_ids=None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        cache_write_current_frame: bool = True,
        cache_evict_current_frame: bool = True,
        global_cache_history_anchor_special_tokens_only: bool = False,
        history_anchor_frame_ids=None,
        history_anchor_patch_topk_per_frame: int = 0,
        history_anchor_max_frames: int = 0,
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
            
        def attn_residual_func(
            x: Tensor,
            pos=None,
            attn_mask=None,
            past_key_values=None,
            use_cache=False,
            cache_budget=None,
        ) -> Union[Tensor, Tuple[Tensor, Dict]]:
            if use_cache:
                attn_result = self.attn(
                    self.norm1(x),
                    pos=pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_budget=cache_budget,
                    cache_analysis_config=cache_analysis_config,
                    pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                    eviction_nn_analysis_config=eviction_nn_analysis_config,
                    leverage_score_histogram_config=leverage_score_histogram_config,
                    token_overlay_dump_config=token_overlay_dump_config,
                    layer_id=layer_id,
                    step_idx=step_idx,
                    tokens_per_frame=tokens_per_frame,
                    eviction_policy=eviction_policy,
                    eviction_policy_layers=eviction_policy_layers,
                    eviction_debug=eviction_debug,
                    leverage_sketch_dim=leverage_sketch_dim,
                    leverage_granularity=leverage_granularity,
                    leverage_feature=leverage_feature,
                    leverage_projection=leverage_projection,
                    leverage_head_mean_dim=leverage_head_mean_dim,
                    leverage_normalize_rows=leverage_normalize_rows,
                    leverage_approx_method=leverage_approx_method,                    leverage_ridge_lambda=leverage_ridge_lambda,
                    leverage_ridge_lambda_mode=leverage_ridge_lambda_mode,
                    leverage_ridge_score_chunk_size=leverage_ridge_score_chunk_size,
                    leverage_ridge_jitter=leverage_ridge_jitter,
                    leverage_ridge_dim=leverage_ridge_dim,
                    leverage_diag=leverage_diag,
                    leverage_diag_interval=leverage_diag_interval,
                    leverage_random_seed=leverage_random_seed,
                    leverage_eviction_selector=leverage_eviction_selector,
                    leverage_similarity_granularity=leverage_similarity_granularity,
                leverage_similarity_feature_projection=leverage_similarity_feature_projection,
                leverage_similarity_leverage_gamma=leverage_similarity_leverage_gamma,
                leverage_eviction_risk_mode=leverage_eviction_risk_mode,
                    leverage_high_outlier_z=leverage_high_outlier_z,
                    leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                    leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                    leverage_dpp_quality_beta=leverage_dpp_quality_beta,
                    leverage_dpp_diversity_beta=leverage_dpp_diversity_beta,
                    leverage_dpp_feature_projection=leverage_dpp_feature_projection,
                    leverage_dpp_recency_bonus=leverage_dpp_recency_bonus,
                    leverage_dpp_recency_lambda=leverage_dpp_recency_lambda,
                    leverage_dpp_recency_window=leverage_dpp_recency_window,
                    leverage_dpp_recency_gate_power=leverage_dpp_recency_gate_power,
                    leverage_dpp_recency_debug=leverage_dpp_recency_debug,
                    leverage_conf_gate=leverage_conf_gate,
                    leverage_conf_gate_floor=leverage_conf_gate_floor,
                    leverage_conf_gate_depth_alpha=leverage_conf_gate_depth_alpha,
                    leverage_conf_gate_point_beta=leverage_conf_gate_point_beta,
                    leverage_conf_gate_init=leverage_conf_gate_init,
                    layer_budget_strategy=layer_budget_strategy,
                    layer_budget_value_gamma=layer_budget_value_gamma,
                    layer_budget_value_norm_type=layer_budget_value_norm_type,
                    layer_budget_norm_source=layer_budget_norm_source,
                    layer_budget_eps=layer_budget_eps,
                    slots_per_direction=slots_per_direction,
                    hybrid_beta=hybrid_beta,
                    eviction_protect_recent_frames=eviction_protect_recent_frames,
                    eviction_protect_special_tokens=eviction_protect_special_tokens,
                    eviction_protect_special_token_interval=eviction_protect_special_token_interval,
                    special_token_count=special_token_count,
                    anchor_token_count=anchor_token_count,
                    window_token_count=window_token_count,
                    recent_merge_config=recent_merge_config,
                    svd_eviction_merge_config=svd_eviction_merge_config,
                    voxel_covis_frame_ids=voxel_covis_frame_ids,
                    voxel_covis_enabled=voxel_covis_enabled,
                    voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                    cache_write_current_frame=cache_write_current_frame,
                    cache_evict_current_frame=cache_evict_current_frame,
                    global_cache_history_anchor_special_tokens_only=global_cache_history_anchor_special_tokens_only,
                    history_anchor_frame_ids=history_anchor_frame_ids,
                    history_anchor_patch_topk_per_frame=history_anchor_patch_topk_per_frame,
                    history_anchor_max_frames=history_anchor_max_frames,
                )
                if len(attn_result) == 4:
                    output, new_kv, scores, special_kv_sidecar = attn_result
                    return self.ls1(output), new_kv, scores, special_kv_sidecar
                output, new_kv, scores = attn_result
                return self.ls1(output), new_kv, scores
            else:
                if attn_mask is not None:
                    return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
                else:
                    return self.ls1(self.attn(self.norm1(x), pos=pos))
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        if use_cache:
            cache_result = attn_residual_func(x, pos=pos, past_key_values=past_key_values, use_cache=True, cache_budget=cache_budget)
            if len(cache_result) == 4:
                attn_output, new_kv, scores, special_kv_sidecar = cache_result
                x = x + attn_output
                x = x + ffn_residual_func(x)
                return x, new_kv, scores, special_kv_sidecar
            attn_output, new_kv, scores = cache_result
            x = x + attn_output
            x = x + ffn_residual_func(x)
            return x, new_kv, scores

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + ffn_residual_func(x)
        return x

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
