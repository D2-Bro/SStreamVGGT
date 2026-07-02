# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any, Set

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.confidence_state import pack_kv_cache, unpack_kv_cache
from streamvggt.layers.recent_merge import KVCacheMetadata, RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.eviction import (
    DEPTH_WEIGHTED_LAYER_BUDGET_STRATEGIES,
    LAYER_BUDGET_SCORE_STRATEGIES,
    VALID_LAYER_BUDGET_STRATEGIES,
    VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES,
    _allocate_layer_budgets_from_scores,
    _combine_value_weighted_leverage_pr_scores,
)
from streamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from streamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from streamvggt.utils.cache_analysis import (
    CacheAnalysisConfig,
    EvictionNNAnalysisConfig,
    LeverageScoreHistogramConfig,
    ProjectedNormHistogramConfig,
    PreEvictionSnapshotConfig,
    TokenOverlayDumpConfig,
)
from streamvggt.utils.global_attn_ranges import (
    GlobalAttnIdxRange,
    is_global_idx_enabled,
    parse_global_attn_idx_ranges,
)

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )
        self.last_scores = torch.zeros(self.depth)
        self.last_layer_budget_scores = torch.zeros(self.depth)
        self.last_layer_budget_base_scores = torch.zeros(self.depth)
        self.last_layer_budget_value_norms = torch.zeros(self.depth)
        self.last_global_attn_debug_trace = []
        self.last_global_special_kv_sidecars = [None] * self.depth
        self.register_buffer("layer_budget_proportions", None, persistent=False)

    def reset_stream_state(self) -> None:
        """Clear sequence-local streaming state before a new inference sequence."""
        self.last_scores.zero_()
        self.last_layer_budget_scores.zero_()
        self.last_layer_budget_base_scores.zero_()
        self.last_layer_budget_value_norms.zero_()
        self.last_global_attn_debug_trace = []
        self.last_global_special_kv_sidecars = [None] * self.depth
        for block in list(self.frame_blocks) + list(self.global_blocks):
            reset = getattr(getattr(block, "attn", None), "_reset_cache_state", None)
            if reset is not None:
                reset()


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        total_budget=0,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        leverage_score_histogram_config: Optional[LeverageScoreHistogramConfig] = None,
        projected_norm_histogram_config: Optional[ProjectedNormHistogramConfig] = None,
        token_overlay_dump_config: Optional[TokenOverlayDumpConfig] = None,
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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_depth_mu: float = 0.5,
        layer_budget_depth_sigma: float = 0.2,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        layer_budget_log_path: Optional[str] = None,
        eviction_protect_recent_frames: int = 0,
        eviction_protect_special_tokens: bool = False,
        eviction_protect_special_token_interval: int = 1,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[List[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        global_attn_idx_ranges: Optional[Union[str, List[GlobalAttnIdxRange]]] = None,
        global_attn_debug: bool = False,
        cache_write_current_frame: bool = True,
        cache_evict_current_frame: bool = True,
        global_cache_history_anchor_special_tokens_only: bool = False,
        history_anchor_frame_ids: Optional[List[int]] = None,
        history_anchor_patch_topk_per_frame: int = 0,
        history_anchor_max_frames: int = 0,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        cache_write_current_frame = bool(cache_write_current_frame)
        cache_evict_current_frame = cache_write_current_frame and bool(cache_evict_current_frame)
        if layer_budget_strategy not in VALID_LAYER_BUDGET_STRATEGIES:
            raise ValueError(
                "layer_budget_strategy must be one of "
                f"{VALID_LAYER_BUDGET_STRATEGIES}, got {layer_budget_strategy!r}"
            )
        if layer_budget_strategy in LAYER_BUDGET_SCORE_STRATEGIES and (
            eviction_policy != "svd_leverage" or leverage_granularity != "layer"
        ):
            raise ValueError(
                "leverage-based layer_budget_strategy requires "
                "eviction_policy='svd_leverage' and leverage_granularity='layer'"
            )
        if not (0.0 <= float(layer_budget_depth_mu) <= 1.0):
            raise ValueError(f"layer_budget_depth_mu must be in [0, 1], got {layer_budget_depth_mu}")
        if float(layer_budget_depth_sigma) <= 0.0:
            raise ValueError(f"layer_budget_depth_sigma must be > 0, got {layer_budget_depth_sigma}")
        parsed_global_attn_idx_ranges = self._normalize_global_attn_idx_ranges(global_attn_idx_ranges)
        range_mode_enabled = parsed_global_attn_idx_ranges is not None

        has_past_cache = False
        if use_cache and past_key_values is not None:
            if range_mode_enabled:
                has_past_cache = any(kv is not None for kv in past_key_values)
            else:
                has_past_cache = past_key_values[0] is not None

        if use_cache and has_past_cache:
            # _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true = past_frame_idx + 1
        else:
            S_true = S
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        current_cache_token_count = S * P if cache_write_current_frame else 0
        if (
            global_cache_history_anchor_special_tokens_only
            and cache_write_current_frame
            and int(past_frame_idx) != 0
        ):
            current_cache_token_count = S * max(P - int(self.patch_start_idx), 0)
        current_budgets = self._calculate_dynamic_budgets(
            total_budget,
            enabled_global_idx_ranges=parsed_global_attn_idx_ranges,
            layer_budget_strategy=layer_budget_strategy,
            layer_budget_value_gamma=layer_budget_value_gamma,
            layer_budget_value_norm_type=layer_budget_value_norm_type,
            layer_budget_alpha=layer_budget_alpha,
            layer_budget_min_tokens=layer_budget_min_tokens,
            layer_budget_eps=layer_budget_eps,
            layer_budget_depth_mu=layer_budget_depth_mu,
            layer_budget_depth_sigma=layer_budget_depth_sigma,
            slots_per_direction=slots_per_direction,
            hybrid_beta=hybrid_beta,
            layer_budget_log_path=layer_budget_log_path,
            layer_budget_step_idx=past_frame_idx,
            past_key_values=past_key_values,
            current_token_count=current_cache_token_count,
        )
        scores = []
        layer_budget_scores = []
        layer_budget_base_scores = []
        layer_budget_value_norms = []
        updated_scores = self.last_scores.clone()
        updated_layer_budget_scores = self.last_layer_budget_scores.clone()
        updated_layer_budget_base_scores = self.last_layer_budget_base_scores.clone()
        updated_layer_budget_value_norms = self.last_layer_budget_value_norms.clone()
        self.last_global_attn_debug_trace = []
        self.last_global_special_kv_sidecars = [None] * self.depth
        raw_block_idx = 0

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    for _ in frame_intermediates:
                        self._record_global_attn_debug(
                            enabled=global_attn_debug,
                            raw_block_idx=raw_block_idx,
                            original_attention_type="frame",
                            global_idx=None,
                            global_enabled=None,
                            g2f_conversion=False,
                            kv_read=False,
                            kv_write=False,
                        )
                        raw_block_idx += 1
                elif attn_type == "global":
                    if range_mode_enabled:
                        (
                            tokens,
                            global_idx,
                            global_intermediates,
                            updated_scores,
                            updated_layer_budget_scores,
                            updated_layer_budget_base_scores,
                            updated_layer_budget_value_norms,
                            raw_block_idx,
                        ) = self._process_global_attention_with_ranges(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            ranges=parsed_global_attn_idx_ranges,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            past_frame_idx=past_frame_idx,
                            current_budgets=current_budgets,
                            updated_scores=updated_scores,
                            updated_layer_budget_scores=updated_layer_budget_scores,
                            updated_layer_budget_base_scores=updated_layer_budget_base_scores,
                            updated_layer_budget_value_norms=updated_layer_budget_value_norms,
                            raw_block_idx=raw_block_idx,
                            cache_analysis_config=cache_analysis_config,
                            pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                            eviction_nn_analysis_config=eviction_nn_analysis_config,
                            leverage_score_histogram_config=leverage_score_histogram_config,
                            projected_norm_histogram_config=projected_norm_histogram_config,
                            token_overlay_dump_config=token_overlay_dump_config,
                            eviction_policy=eviction_policy,
                            eviction_policy_layers=eviction_policy_layers,
                            eviction_debug=eviction_debug,
                            leverage_sketch_dim=leverage_sketch_dim,
                            leverage_granularity=leverage_granularity,
                            leverage_feature=leverage_feature,
                            leverage_projection=leverage_projection,
                            leverage_head_mean_dim=leverage_head_mean_dim,
                            leverage_normalize_rows=leverage_normalize_rows,
                            leverage_approx_method=leverage_approx_method,                            leverage_ridge_lambda=leverage_ridge_lambda,
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
                            anchor_token_count=anchor_token_count,
                            window_token_count=window_token_count,
                            recent_merge_config=recent_merge_config,
                            svd_eviction_merge_config=svd_eviction_merge_config,
                            voxel_covis_frame_ids=voxel_covis_frame_ids,
                            voxel_covis_enabled=voxel_covis_enabled,
                            voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                            global_attn_debug=global_attn_debug,
                            cache_write_current_frame=cache_write_current_frame,
                            cache_evict_current_frame=cache_evict_current_frame,
                            global_cache_history_anchor_special_tokens_only=global_cache_history_anchor_special_tokens_only,
                            history_anchor_frame_ids=history_anchor_frame_ids,
                            history_anchor_patch_topk_per_frame=history_anchor_patch_topk_per_frame,
                            history_anchor_max_frames=history_anchor_max_frames,
                        )
                    elif use_cache:
                        old_global_idx = global_idx
                        old_kv_read = past_key_values[old_global_idx] is not None
                        tokens, global_idx, global_intermediates, new_kv, current_scores, special_kv_sidecar = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=current_budgets[global_idx].item(),
                            cache_analysis_config=cache_analysis_config,
                            pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                            eviction_nn_analysis_config=eviction_nn_analysis_config,
                            leverage_score_histogram_config=leverage_score_histogram_config,
                            projected_norm_histogram_config=projected_norm_histogram_config,
                            token_overlay_dump_config=token_overlay_dump_config,
                            eviction_policy=eviction_policy,
                            eviction_policy_layers=eviction_policy_layers,
                            eviction_debug=eviction_debug,
                            leverage_sketch_dim=leverage_sketch_dim,
                            leverage_granularity=leverage_granularity,
                            leverage_feature=leverage_feature,
                            leverage_projection=leverage_projection,
                            leverage_head_mean_dim=leverage_head_mean_dim,
                            leverage_normalize_rows=leverage_normalize_rows,
                            leverage_approx_method=leverage_approx_method,                            leverage_ridge_lambda=leverage_ridge_lambda,
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
                        past_key_values[global_idx - 1] = new_kv
                        self.last_global_special_kv_sidecars[global_idx - 1] = special_kv_sidecar
                        current_score, current_layer_budget_score = self._split_score_payload(current_scores)
                        if current_score is not None: # pruning happened
                            scores.append(current_score)
                        else:
                            scores.append(self.last_scores[global_idx-1].item())
                        score_value, base_value, value_norm = self._coerce_layer_budget_payload(
                            current_layer_budget_score,
                            fallback_score=self.last_layer_budget_scores[global_idx-1],
                            fallback_base=self.last_layer_budget_base_scores[global_idx-1],
                            fallback_value_norm=self.last_layer_budget_value_norms[global_idx-1],
                            strategy=layer_budget_strategy,
                        )
                        layer_budget_scores.append(score_value)
                        layer_budget_base_scores.append(base_value)
                        layer_budget_value_norms.append(value_norm)
                        self._record_global_attn_debug(
                            enabled=global_attn_debug,
                            raw_block_idx=raw_block_idx,
                            original_attention_type="global",
                            global_idx=old_global_idx,
                            global_enabled=True,
                            g2f_conversion=False,
                            kv_read=old_kv_read,
                            kv_write=cache_write_current_frame,
                        )
                        raw_block_idx += len(global_intermediates)
                    else: 
                        old_global_idx = global_idx
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                        self._record_global_attn_debug(
                            enabled=global_attn_debug,
                            raw_block_idx=raw_block_idx,
                            original_attention_type="global",
                            global_idx=old_global_idx,
                            global_enabled=True,
                            g2f_conversion=False,
                            kv_read=False,
                            kv_write=False,
                        )
                        raw_block_idx += len(global_intermediates)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        assert len(output_list) == self.depth, f"Expected {self.depth} outputs, got {len(output_list)}"
        if range_mode_enabled and use_cache:
            self.last_scores = updated_scores.to(device=self.last_scores.device, dtype=self.last_scores.dtype)
            self.last_layer_budget_base_scores = updated_layer_budget_base_scores.to(
                device=self.last_layer_budget_base_scores.device,
                dtype=self.last_layer_budget_base_scores.dtype,
            )
            self.last_layer_budget_value_norms = updated_layer_budget_value_norms.to(
                device=self.last_layer_budget_value_norms.device,
                dtype=self.last_layer_budget_value_norms.dtype,
            )
            if layer_budget_strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                self.last_layer_budget_scores = self._combine_value_weighted_layer_scores(
                    self.last_layer_budget_base_scores,
                    self.last_layer_budget_value_norms,
                    layer_budget_value_gamma,
                    layer_budget_eps,
                ).to(device=self.last_layer_budget_scores.device, dtype=self.last_layer_budget_scores.dtype)
            else:
                self.last_layer_budget_scores = updated_layer_budget_scores.to(
                    device=self.last_layer_budget_scores.device,
                    dtype=self.last_layer_budget_scores.dtype,
                )
        elif scores: # update scores
            self.last_scores = torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype)
            self.last_layer_budget_base_scores = torch.tensor(
                layer_budget_base_scores,
                device=self.last_layer_budget_base_scores.device,
                dtype=self.last_layer_budget_base_scores.dtype,
            )
            self.last_layer_budget_value_norms = torch.tensor(
                layer_budget_value_norms,
                device=self.last_layer_budget_value_norms.device,
                dtype=self.last_layer_budget_value_norms.dtype,
            )
            if layer_budget_strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                self.last_layer_budget_scores = self._combine_value_weighted_layer_scores(
                    self.last_layer_budget_base_scores,
                    self.last_layer_budget_value_norms,
                    layer_budget_value_gamma,
                    layer_budget_eps,
                ).to(device=self.last_layer_budget_scores.device, dtype=self.last_layer_budget_scores.dtype)
            else:
                self.last_layer_budget_scores = torch.tensor(
                    layer_budget_scores,
                    device=self.last_layer_budget_scores.device,
                    dtype=self.last_layer_budget_scores.dtype,
                )
        if global_attn_debug:
            self._print_global_attn_debug_trace()

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):

            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention_as_frame(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Run an original global block independently per current frame/token group.
        This intentionally bypasses any streaming KV cache reads/writes.
        """
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        assert tokens.shape == (B * S, P, C), f"Expected disabled global input {(B * S, P, C)}, got {tokens.shape}"
        if pos is not None:
            assert pos.shape == (B * S, P, 2), f"Expected disabled global pos {(B * S, P, 2)}, got {pos.shape}"

        tokens = self.global_blocks[global_idx](tokens, pos=pos)
        assert tokens.shape == (B * S, P, C), f"Expected disabled global output {(B * S, P, C)}, got {tokens.shape}"
        intermediate = tokens.reshape(B, S, P, C)
        assert intermediate.shape == (B, S, P, C), f"Expected disabled global intermediate {(B, S, P, C)}, got {intermediate.shape}"
        return tokens, global_idx + 1, [intermediate]

    def _process_global_attention_with_ranges(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        ranges=None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        current_budgets=None,
        updated_scores=None,
        updated_layer_budget_scores=None,
        updated_layer_budget_base_scores=None,
        updated_layer_budget_value_norms=None,
        raw_block_idx=0,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        leverage_score_histogram_config: Optional[LeverageScoreHistogramConfig] = None,
        projected_norm_histogram_config: Optional[ProjectedNormHistogramConfig] = None,
        token_overlay_dump_config: Optional[TokenOverlayDumpConfig] = None,
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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_depth_mu: float = 0.5,
        layer_budget_depth_sigma: float = 0.2,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        eviction_protect_recent_frames: int = 0,
        eviction_protect_special_tokens: bool = False,
        eviction_protect_special_token_interval: int = 1,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[List[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        global_attn_debug: bool = False,
        cache_write_current_frame: bool = True,
        cache_evict_current_frame: bool = True,
        global_cache_history_anchor_special_tokens_only: bool = False,
        history_anchor_frame_ids: Optional[List[int]] = None,
        history_anchor_patch_topk_per_frame: int = 0,
        history_anchor_max_frames: int = 0,
    ):
        intermediates = []

        for _ in range(self.aa_block_size):
            current_global_idx = global_idx
            global_enabled = is_global_idx_enabled(current_global_idx, ranges)
            kv_read = False
            kv_write = False

            if global_enabled:
                if use_cache:
                    past_key_values_block = None
                    if past_key_values is not None and past_key_values[current_global_idx] is not None:
                        past_key_values_block = past_key_values[current_global_idx]
                        kv_read = True
                    before_cache_shape = _cache_shape(past_key_values_block)
                    tokens, global_idx, block_intermediates, new_kv, current_scores, special_kv_sidecar = self._process_global_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        global_idx,
                        pos=pos,
                        past_key_values_block=past_key_values_block,
                        use_cache=True,
                        past_frame_idx=past_frame_idx,
                        cache_budget=current_budgets[current_global_idx].item(),
                        cache_analysis_config=cache_analysis_config,
                        pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                        eviction_nn_analysis_config=eviction_nn_analysis_config,
                        leverage_score_histogram_config=leverage_score_histogram_config,
                        projected_norm_histogram_config=projected_norm_histogram_config,
                        token_overlay_dump_config=token_overlay_dump_config,
                        eviction_policy=eviction_policy,
                        eviction_policy_layers=eviction_policy_layers,
                        eviction_debug=eviction_debug,
                        leverage_sketch_dim=leverage_sketch_dim,
                        leverage_granularity=leverage_granularity,
                        leverage_feature=leverage_feature,
                        leverage_projection=leverage_projection,
                        leverage_head_mean_dim=leverage_head_mean_dim,
                        leverage_normalize_rows=leverage_normalize_rows,
                        leverage_approx_method=leverage_approx_method,                        leverage_ridge_lambda=leverage_ridge_lambda,
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
                        anchor_token_count=anchor_token_count,
                        window_token_count=window_token_count,
                        recent_merge_config=recent_merge_config,
                        svd_eviction_merge_config=svd_eviction_merge_config,
                        voxel_covis_frame_ids=voxel_covis_frame_ids,
                        voxel_covis_enabled=voxel_covis_enabled,
                        voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                        block_count=1,
                        cache_write_current_frame=cache_write_current_frame,
                        cache_evict_current_frame=cache_evict_current_frame,
                        global_cache_history_anchor_special_tokens_only=global_cache_history_anchor_special_tokens_only,
                        history_anchor_frame_ids=history_anchor_frame_ids,
                        history_anchor_patch_topk_per_frame=history_anchor_patch_topk_per_frame,
                        history_anchor_max_frames=history_anchor_max_frames,
                    )
                    kv_write = cache_write_current_frame
                    past_key_values[current_global_idx] = new_kv
                    self.last_global_special_kv_sidecars[current_global_idx] = special_kv_sidecar
                    if cache_write_current_frame:
                        self._assert_enabled_cache_compatible(before_cache_shape, new_kv, current_global_idx)
                    current_score, current_layer_budget_score = self._split_score_payload(current_scores)
                    if current_score is not None:
                        updated_scores[current_global_idx] = torch.as_tensor(
                            current_score,
                            device=updated_scores.device,
                            dtype=updated_scores.dtype,
                        )
                    if current_layer_budget_score is not None and updated_layer_budget_scores is not None:
                        score_value, base_value, value_norm = self._coerce_layer_budget_payload(
                            current_layer_budget_score,
                            fallback_score=updated_layer_budget_scores[current_global_idx],
                            fallback_base=(
                                updated_layer_budget_base_scores[current_global_idx]
                                if updated_layer_budget_base_scores is not None
                                else updated_layer_budget_scores[current_global_idx]
                            ),
                            fallback_value_norm=(
                                updated_layer_budget_value_norms[current_global_idx]
                                if updated_layer_budget_value_norms is not None
                                else 0.0
                            ),
                            strategy=layer_budget_strategy,
                        )
                        updated_layer_budget_scores[current_global_idx] = torch.as_tensor(
                            score_value,
                            device=updated_layer_budget_scores.device,
                            dtype=updated_layer_budget_scores.dtype,
                        )
                        if updated_layer_budget_base_scores is not None:
                            updated_layer_budget_base_scores[current_global_idx] = torch.as_tensor(
                                base_value,
                                device=updated_layer_budget_base_scores.device,
                                dtype=updated_layer_budget_base_scores.dtype,
                            )
                        if updated_layer_budget_value_norms is not None:
                            updated_layer_budget_value_norms[current_global_idx] = torch.as_tensor(
                                value_norm,
                                device=updated_layer_budget_value_norms.device,
                                dtype=updated_layer_budget_value_norms.dtype,
                            )
                else:
                    tokens, global_idx, block_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, block_count=1
                    )
            else:
                tokens, global_idx, block_intermediates = self._process_global_attention_as_frame(
                    tokens, B, S, P, C, global_idx, pos=pos
                )

            intermediates.extend(block_intermediates)
            self._record_global_attn_debug(
                enabled=global_attn_debug,
                raw_block_idx=raw_block_idx,
                original_attention_type="global",
                global_idx=current_global_idx,
                global_enabled=global_enabled,
                g2f_conversion=not global_enabled,
                kv_read=kv_read,
                kv_write=kv_write,
                cache_budget=(
                    None
                    if not use_cache or not global_enabled or current_budgets is None
                    else int(current_budgets[current_global_idx].item())
                ),
            )
            raw_block_idx += 1

        return (
            tokens,
            global_idx,
            intermediates,
            updated_scores,
            updated_layer_budget_scores,
            updated_layer_budget_base_scores,
            updated_layer_budget_value_norms,
            raw_block_idx,
        )

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0,
        cache_budget=None,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        leverage_score_histogram_config: Optional[LeverageScoreHistogramConfig] = None,
        projected_norm_histogram_config: Optional[ProjectedNormHistogramConfig] = None,
        token_overlay_dump_config: Optional[TokenOverlayDumpConfig] = None,
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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_depth_mu: float = 0.5,
        layer_budget_depth_sigma: float = 0.2,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        eviction_protect_recent_frames: int = 0,
        eviction_protect_special_tokens: bool = False,
        eviction_protect_special_token_interval: int = 1,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[List[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        block_count: Optional[int] = None,
        cache_write_current_frame: bool = True,
        cache_evict_current_frame: bool = True,
        global_cache_history_anchor_special_tokens_only: bool = False,
        history_anchor_frame_ids: Optional[List[int]] = None,
        history_anchor_patch_topk_per_frame: int = 0,
        history_anchor_max_frames: int = 0,
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        if block_count is None:
            block_count = self.aa_block_size

        for _ in range(block_count):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None
            
            scores = None
            if use_cache:
                special_kv_sidecar = None
                block_result = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget,
                    cache_analysis_config=cache_analysis_config,
                    pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                    eviction_nn_analysis_config=eviction_nn_analysis_config,
                    leverage_score_histogram_config=leverage_score_histogram_config,
                    projected_norm_histogram_config=projected_norm_histogram_config,
                    token_overlay_dump_config=token_overlay_dump_config,
                    layer_id=global_idx,
                    step_idx=past_frame_idx,
                    tokens_per_frame=P,
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
                    special_token_count=self.patch_start_idx,
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
                if len(block_result) == 4:
                    tokens, block_kv, scores, special_kv_sidecar = block_result
                else:
                    tokens, block_kv, scores = block_result
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv, scores, special_kv_sidecar
        return tokens, global_idx, intermediates

    def _normalize_global_attn_idx_ranges(self, ranges):
        if ranges is None:
            return None
        if isinstance(ranges, str):
            return parse_global_attn_idx_ranges(ranges, num_global_blocks=len(self.global_blocks))
        normalized = []
        for start, end in ranges:
            if start < 0:
                raise ValueError(f"global attention range start must be >= 0, got {start}")
            if end is not None and start >= end:
                raise ValueError(f"invalid global attention range ({start}, {end}): start must be < end")
            if start >= len(self.global_blocks):
                raise ValueError(
                    f"global attention range starts at {start}, but only {len(self.global_blocks)} global blocks exist"
                )
            if end is not None and end > len(self.global_blocks):
                raise ValueError(
                    f"global attention range ends at {end}, but only {len(self.global_blocks)} global blocks exist"
                )
            normalized.append((int(start), None if end is None else int(end)))
        return normalized

    def _record_global_attn_debug(
        self,
        enabled,
        raw_block_idx,
        original_attention_type,
        global_idx,
        global_enabled,
        g2f_conversion,
        kv_read,
        kv_write,
        cache_budget=None,
    ):
        self.last_global_attn_debug_trace.append(
            {
                "raw_block_idx": raw_block_idx,
                "original_attention_type": original_attention_type,
                "global_idx": global_idx,
                "global_enabled": global_enabled,
                "g2f_conversion": g2f_conversion,
                "kv_read": kv_read,
                "kv_write": kv_write,
                "cache_budget": cache_budget,
            }
        )

    def _print_global_attn_debug_trace(self):
        for entry in self.last_global_attn_debug_trace:
            print(
                "[global-attn-debug] "
                f"raw_block_idx={entry['raw_block_idx']} "
                f"original_attention_type={entry['original_attention_type']} "
                f"global_idx={entry['global_idx']} "
                f"global_enabled={entry['global_enabled']} "
                f"g2f_conversion={entry['g2f_conversion']} "
                f"kv_read={entry['kv_read']} "
                f"kv_write={entry['kv_write']} "
                f"cache_budget={entry['cache_budget']}"
            )

    def _assert_enabled_cache_compatible(self, before_cache_shape, new_kv, global_idx):
        after_cache_shape = _cache_shape(new_kv)
        if after_cache_shape is None:
            raise AssertionError(f"Enabled global layer {global_idx} did not return a KV cache")
        if before_cache_shape is not None:
            before_b, before_h, _, before_d = before_cache_shape
            after_b, after_h, _, after_d = after_cache_shape
            assert (before_b, before_h, before_d) == (after_b, after_h, after_d), (
                f"Cache shape changed incompatibly for global layer {global_idx}: "
                f"before={before_cache_shape}, after={after_cache_shape}"
            )
        

    def sync_anchor_special_tokens_from_sidecars(
        self,
        past_key_values,
        special_kv_sidecars,
        anchor_token_count: int,
        tokens_per_frame: int,
        is_fifo: bool = False,
        global_anchor_token_count: Optional[int] = None,
    ):
        if past_key_values is None or special_kv_sidecars is None or anchor_token_count is None:
            return past_key_values
        anchor_token_count = int(anchor_token_count)
        tokens_per_frame = int(tokens_per_frame)
        global_anchor_token_count = (
            tokens_per_frame
            if global_anchor_token_count is None
            else min(max(int(global_anchor_token_count), 0), tokens_per_frame)
        )
        if anchor_token_count <= global_anchor_token_count or tokens_per_frame <= 0:
            return past_key_values

        for idx in range(min(self.depth, len(past_key_values), len(special_kv_sidecars))):
            layer_kv = past_key_values[idx]
            sidecar = special_kv_sidecars[idx]
            if layer_kv is None or sidecar is None:
                continue
            k, v, metadata, confidence_state = unpack_kv_cache(layer_kv)
            side_k, side_v, side_metadata, side_confidence_state = unpack_kv_cache(sidecar)
            chunk = int(side_k.shape[2])
            if chunk <= 0:
                continue
            B, H, N, D = k.shape
            if side_k.shape[:2] != (B, H) or side_v.shape[:2] != (B, H):
                continue

            global_anchor_end = min(global_anchor_token_count, N)
            if is_fifo:
                remove_start = global_anchor_end
                remove_end = min(global_anchor_end + chunk, anchor_token_count, N)
                if remove_end <= remove_start:
                    continue
                k_parts = [k[:, :, :remove_start, :], k[:, :, remove_end:anchor_token_count, :], side_k, k[:, :, anchor_token_count:, :]]
                v_parts = [v[:, :, :remove_start, :], v[:, :, remove_end:anchor_token_count, :], side_v, v[:, :, anchor_token_count:, :]]
                sidecar_part_specs = [(0, remove_start), (remove_end, min(anchor_token_count, N)), None, (anchor_token_count, N)]
            else:
                insert_pos = max(anchor_token_count - chunk, global_anchor_end)
                insert_pos = min(insert_pos, N)
                k_parts = [k[:, :, :insert_pos, :], side_k, k[:, :, insert_pos:, :]]
                v_parts = [v[:, :, :insert_pos, :], side_v, v[:, :, insert_pos:, :]]
                sidecar_part_specs = [(0, insert_pos), None, (insert_pos, N)]

            k_new = torch.cat(k_parts, dim=2)
            v_new = torch.cat(v_parts, dim=2)
            merged_metadata = None
            if metadata is not None and side_metadata is not None:
                meta_parts = []
                for spec in sidecar_part_specs:
                    if spec is None:
                        meta_parts.append(side_metadata)
                        continue
                    start_pos, end_pos = spec
                    if end_pos > start_pos:
                        meta_parts.append(Aggregator._metadata_slice(metadata, start_pos, end_pos))
                if meta_parts:
                    merged_metadata = meta_parts[0]
                    for part in meta_parts[1:]:
                        merged_metadata = merged_metadata.concat(part)
                else:
                    merged_metadata = metadata

            merged_confidence_state = None
            if confidence_state is not None and side_confidence_state is not None:
                conf_parts = []
                for spec in sidecar_part_specs:
                    if spec is None:
                        conf_parts.append(side_confidence_state)
                        continue
                    start_pos, end_pos = spec
                    if end_pos > start_pos:
                        conf_parts.append(confidence_state.slice(start_pos, end_pos))
                if conf_parts:
                    merged_confidence_state = conf_parts[0]
                    for part in conf_parts[1:]:
                        merged_confidence_state = merged_confidence_state.concat(part)
                else:
                    merged_confidence_state = confidence_state

            past_key_values[idx] = pack_kv_cache(k_new, v_new, merged_metadata, merged_confidence_state)
        return past_key_values


    def _metadata_slice(metadata, start: int, end: int):
        B, H, _ = metadata.frame_ids.shape
        indices = torch.arange(int(start), int(end), dtype=torch.long).view(1, 1, -1)
        indices = indices.expand(B, H, -1)
        return metadata.gather(indices)


    def sync_anchor_change(
        self,
        past_key_values,
        anchor_token_count: int,
        tokens_per_frame: int,
        anchor_keep_ratio: float,
        anchor_token_indices: Optional[torch.Tensor] = None,
        is_fifo: bool = False,
        global_anchor_token_count: Optional[int] = None,
    ):
        """Promote newest-frame tokens into the protected anchor zone."""
        if past_key_values is None or anchor_token_count is None:
            return past_key_values

        anchor_token_count = int(anchor_token_count)
        tokens_per_frame = int(tokens_per_frame)
        global_anchor_token_count = (
            tokens_per_frame
            if global_anchor_token_count is None
            else min(max(int(global_anchor_token_count), 0), tokens_per_frame)
        )
        if tokens_per_frame <= 0 or anchor_token_count <= global_anchor_token_count:
            return past_key_values

        if anchor_token_indices is not None:
            anchor_chunk = int(anchor_token_indices.shape[-1])
        else:
            anchor_chunk = max(int(tokens_per_frame * anchor_keep_ratio), 1)
        anchor_chunk = min(anchor_chunk, tokens_per_frame)
        if anchor_chunk <= 0:
            return past_key_values

        global_anchor_end = global_anchor_token_count

        for idx in range(self.depth):
            if past_key_values[idx] is None:
                continue

            layer_kv = past_key_values[idx]
            k, v, metadata, confidence_state = unpack_kv_cache(layer_kv)

            B, H, N, D = k.shape
            if N <= anchor_token_count:
                continue

            new_frame_start = N - tokens_per_frame
            if new_frame_start < 0 or new_frame_start < anchor_token_count:
                continue

            frame_indices = torch.arange(tokens_per_frame, device=k.device, dtype=torch.long)
            if anchor_token_indices is None:
                selected_by_batch = frame_indices[:anchor_chunk].view(1, anchor_chunk).expand(B, anchor_chunk)
            else:
                selected_by_batch = anchor_token_indices.to(device=k.device, dtype=torch.long)
                if selected_by_batch.dim() == 1:
                    selected_by_batch = selected_by_batch.view(1, -1).expand(B, -1)
                selected_by_batch = selected_by_batch[:, :anchor_chunk]

            batch_orders = []
            for b in range(B):
                selected = selected_by_batch[b]
                selected = selected[(selected >= 0) & (selected < tokens_per_frame)]
                if selected.numel() != anchor_chunk:
                    fallback = frame_indices[:anchor_chunk]
                    selected = fallback
                mask = torch.ones(tokens_per_frame, device=k.device, dtype=torch.bool)
                mask[selected] = False
                remaining = frame_indices[mask]
                selected_abs = selected + new_frame_start
                remaining_abs = remaining + new_frame_start

                if is_fifo:
                    demote_start = global_anchor_end
                    demote_end = min(global_anchor_end + anchor_chunk, anchor_token_count)
                    if demote_end <= demote_start:
                        break
                    order = torch.cat(
                        [
                            torch.arange(0, demote_start, device=k.device, dtype=torch.long),
                            torch.arange(demote_end, anchor_token_count, device=k.device, dtype=torch.long),
                            selected_abs,
                            torch.arange(anchor_token_count, new_frame_start, device=k.device, dtype=torch.long),
                            remaining_abs,
                            torch.arange(new_frame_start + tokens_per_frame, N, device=k.device, dtype=torch.long),
                            torch.arange(demote_start, demote_end, device=k.device, dtype=torch.long),
                        ],
                        dim=0,
                    )
                else:
                    old_anchor_end = max(anchor_token_count - anchor_chunk, global_anchor_end)
                    if old_anchor_end > new_frame_start:
                        break
                    order = torch.cat(
                        [
                            torch.arange(0, old_anchor_end, device=k.device, dtype=torch.long),
                            selected_abs,
                            torch.arange(old_anchor_end, new_frame_start, device=k.device, dtype=torch.long),
                            remaining_abs,
                            torch.arange(new_frame_start + tokens_per_frame, N, device=k.device, dtype=torch.long),
                        ],
                        dim=0,
                    )

                if order.numel() != N:
                    break
                batch_orders.append(order)

            if len(batch_orders) != B:
                continue

            gather_indices = torch.stack(batch_orders, dim=0).view(B, 1, N).expand(B, H, N)
            expanded_indices = gather_indices.unsqueeze(-1).expand(B, H, N, D)
            k_new = torch.gather(k, 2, expanded_indices)
            v_new = torch.gather(v, 2, expanded_indices)
            if metadata is not None:
                metadata = metadata.gather(gather_indices.detach().cpu())
            if confidence_state is not None:
                confidence_state = confidence_state.gather(gather_indices)
            past_key_values[idx] = pack_kv_cache(k_new, v_new, metadata, confidence_state)

        return past_key_values

    @staticmethod
    def _split_score_payload(score_payload):
        if isinstance(score_payload, tuple):
            if len(score_payload) != 2:
                raise ValueError(f"Unexpected score payload length: {len(score_payload)}")
            return score_payload
        return score_payload, None

    @staticmethod
    def _coerce_layer_budget_payload(
        layer_budget_payload,
        fallback_score,
        fallback_base,
        fallback_value_norm,
        strategy: str,
    ):
        if layer_budget_payload is None:
            return (
                float(torch.as_tensor(fallback_score).detach().float().item()),
                float(torch.as_tensor(fallback_base).detach().float().item()),
                float(torch.as_tensor(fallback_value_norm).detach().float().item()),
            )
        payload = torch.as_tensor(layer_budget_payload).detach().float().reshape(-1)
        if strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES and payload.numel() >= 2:
            base_value = float(torch.nan_to_num(payload[0], nan=0.0, posinf=0.0, neginf=0.0).item())
            value_norm = float(torch.nan_to_num(payload[1], nan=0.0, posinf=0.0, neginf=0.0).item())
            return base_value, base_value, value_norm
        score_value = float(torch.nan_to_num(payload.mean(), nan=0.0, posinf=0.0, neginf=0.0).item())
        return score_value, score_value, 0.0

    @staticmethod
    def _combine_value_weighted_layer_scores(
        base_scores,
        value_norms,
        gamma: float,
        eps: float,
        capacities: Optional[Dict[int, int]] = None,
    ):
        base = torch.as_tensor(base_scores).detach().float()
        active = None
        if capacities is not None:
            active = torch.zeros_like(base, dtype=torch.bool)
            for layer, capacity in capacities.items():
                layer = int(layer)
                if 0 <= layer < active.numel() and int(capacity) > 0:
                    active[layer] = True
        return _combine_value_weighted_leverage_pr_scores(
            base,
            value_norms,
            gamma=gamma,
            eps=eps,
            active_mask=active,
        )

    @staticmethod
    def _combine_depth_weighted_layer_scores(
        base_scores,
        mu: float,
        sigma: float,
        capacities: Optional[Dict[int, int]] = None,
    ):
        base = torch.nan_to_num(torch.as_tensor(base_scores).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        base = torch.clamp(base, min=0.0)
        depth = int(base.numel())
        if depth <= 0:
            return base
        denom = float(max(depth - 1, 1))
        positions = torch.arange(depth, device=base.device, dtype=base.dtype) / denom
        sigma_value = max(float(sigma), 1e-12)
        prior = torch.exp(-((positions - float(mu)) ** 2) / (2.0 * sigma_value * sigma_value))
        if capacities is not None:
            active = torch.zeros_like(base, dtype=torch.bool)
            for layer, capacity in capacities.items():
                layer = int(layer)
                if 0 <= layer < active.numel() and int(capacity) > 0:
                    active[layer] = True
            prior = torch.where(active, prior, torch.ones_like(prior))
        return base * prior

    def _layer_budget_capacities(self, past_key_values, current_token_count, enabled_global_idx_ranges=None):
        capacities = {}
        current_token_count = max(int(current_token_count or 0), 0)
        for idx in range(self.depth):
            if enabled_global_idx_ranges is not None and not is_global_idx_enabled(idx, enabled_global_idx_ranges):
                capacities[idx] = 0
                continue
            past_tokens = 0
            if past_key_values is not None and idx < len(past_key_values) and past_key_values[idx] is not None:
                shape = _cache_shape(past_key_values[idx])
                past_tokens = 0 if shape is None else int(shape[2])
            capacities[idx] = past_tokens + current_token_count
        return capacities

    def _append_layer_budget_log(
        self,
        log_path,
        step_idx,
        strategy,
        total_budget,
        capacities,
        budgets,
        details,
    ):
        if not log_path:
            return
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        events = "|".join(details.get("events", []))
        active_layers = [layer for layer, cap in capacities.items() if cap > 0]
        with open(log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "step,strategy,total_budget,assigned_budget,total_capacity,"
                    "active_layers,layer,capacity,score,weight,raw_budget,final_budget,events\n"
                )
            for layer in active_layers:
                f.write(
                    f"{int(step_idx)},{strategy},{int(total_budget)},"
                    f"{details.get('assigned_budget', sum(budgets.values()))},"
                    f"{details.get('total_capacity', sum(capacities.values()))},"
                    f"{len(active_layers)},{layer},{capacities[layer]},"
                    f"{details.get('scores', {}).get(layer, 0.0):.9g},"
                    f"{details.get('weights', {}).get(layer, 0.0):.9g},"
                    f"{details.get('raw_budgets', {}).get(layer, 0.0):.9g},"
                    f"{budgets.get(layer, 0)},{events}\n"
                )

    def _calculate_dynamic_budgets(
        self,
        total_budget,
        enabled_global_idx_ranges=None,
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.5,
        layer_budget_value_norm_type: str = "rms",
        layer_budget_norm_source: str = "value",
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_depth_mu: float = 0.5,
        layer_budget_depth_sigma: float = 0.2,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        layer_budget_log_path: Optional[str] = None,
        layer_budget_step_idx: int = 0,
        past_key_values=None,
        current_token_count: int = 0,
    ):

        with torch.no_grad():
            if total_budget < 0:
                total_budget = 0
            if layer_budget_strategy == "uniform":
                if enabled_global_idx_ranges is None:
                    diversity_scores = 1.0 - self.last_scores
                    scaled_scores = diversity_scores / 0.5
                    proportions = torch.softmax(scaled_scores, dim=0)
                    budgets = proportions * total_budget
                else:
                    budgets = torch.zeros_like(self.last_scores)
                    enabled_indices = [
                        idx
                        for idx in range(len(self.last_scores))
                        if is_global_idx_enabled(idx, enabled_global_idx_ranges)
                    ]
                    if enabled_indices:
                        enabled_idx_tensor = torch.tensor(
                            enabled_indices,
                            device=self.last_scores.device,
                            dtype=torch.long,
                        )
                        enabled_scores = self.last_scores.index_select(0, enabled_idx_tensor)
                        diversity_scores = 1.0 - enabled_scores
                        scaled_scores = diversity_scores / 0.5
                        proportions = torch.softmax(scaled_scores, dim=0)
                        budgets[enabled_idx_tensor] = proportions * total_budget
                return budgets.int()

            capacities = self._layer_budget_capacities(
                past_key_values,
                current_token_count,
                enabled_global_idx_ranges=enabled_global_idx_ranges,
            )
            if layer_budget_strategy == "cosine_precomputed":
                if self.layer_budget_proportions is None:
                    raise ValueError(
                        "cosine_precomputed layer budget requires precomputed proportions; "
                        "call StreamVGGT.set_layer_budget_proportions(...) first"
                    )
                active_layer_scores = self.layer_budget_proportions
                allocation_alpha = 1.0
                allocation_min_tokens = 0
            elif layer_budget_strategy in DEPTH_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                active_layer_scores = self._combine_depth_weighted_layer_scores(
                    self.last_layer_budget_scores,
                    layer_budget_depth_mu,
                    layer_budget_depth_sigma,
                    capacities=capacities,
                )
                allocation_alpha = layer_budget_alpha
                allocation_min_tokens = layer_budget_min_tokens
            elif layer_budget_strategy in VALUE_WEIGHTED_LAYER_BUDGET_STRATEGIES:
                active_layer_scores = self._combine_value_weighted_layer_scores(
                    self.last_layer_budget_base_scores,
                    self.last_layer_budget_value_norms,
                    layer_budget_value_gamma,
                    layer_budget_eps,
                    capacities=capacities,
                )
                allocation_alpha = layer_budget_alpha
                allocation_min_tokens = layer_budget_min_tokens
            else:
                active_layer_scores = self.last_layer_budget_scores
                allocation_alpha = layer_budget_alpha
                allocation_min_tokens = layer_budget_min_tokens
            score_dict = {
                idx: active_layer_scores[idx]
                for idx in range(len(active_layer_scores))
            }
            budget_dict, details = _allocate_layer_budgets_from_scores(
                score_dict,
                capacities,
                total_budget,
                alpha=allocation_alpha,
                min_tokens=allocation_min_tokens,
                eps=layer_budget_eps,
                return_debug=True,
            )
            budgets = torch.zeros_like(self.last_scores, dtype=torch.int32)
            for idx, budget in budget_dict.items():
                budgets[idx] = int(budget)
            self._append_layer_budget_log(
                layer_budget_log_path,
                layer_budget_step_idx,
                layer_budget_strategy,
                total_budget,
                capacities,
                budget_dict,
                details,
            )
            return budgets.int()


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined


def _cache_shape(kv):
    if kv is None:
        return None
    if len(kv) < 2:
        return None
    return tuple(kv[0].shape)
