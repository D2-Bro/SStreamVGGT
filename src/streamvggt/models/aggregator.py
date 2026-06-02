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
from typing import Optional, Tuple, Union, List, Dict, Any

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.recent_merge import KVCacheMetadata, RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.eviction import (
    VALID_LAYER_BUDGET_STRATEGIES,
    _allocate_layer_budgets_from_scores,
)
from streamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from streamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from streamvggt.utils.cache_analysis import CacheAnalysisConfig, EvictionNNAnalysisConfig, PreEvictionSnapshotConfig
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
        self.last_global_attn_debug_trace = []


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
        eviction_policy: str = "mean",
        eviction_debug: bool = False,
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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_debug: bool = False,
        layer_budget_log_path: Optional[str] = None,
        eviction_protect_recent_frames: int = 0,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[List[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        global_attn_idx_ranges: Optional[Union[str, List[GlobalAttnIdxRange]]] = None,
        global_attn_debug: bool = False,
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
        if layer_budget_strategy not in VALID_LAYER_BUDGET_STRATEGIES:
            raise ValueError(
                "layer_budget_strategy must be one of "
                f"{VALID_LAYER_BUDGET_STRATEGIES}, got {layer_budget_strategy!r}"
            )
        if layer_budget_strategy != "uniform" and (
            eviction_policy != "svd_leverage" or leverage_granularity != "layer"
        ):
            raise ValueError(
                "layer_budget_strategy requires eviction_policy='svd_leverage' "
                "and leverage_granularity='layer'"
            )
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
        current_budgets = self._calculate_dynamic_budgets(
            total_budget,
            enabled_global_idx_ranges=parsed_global_attn_idx_ranges,
            layer_budget_strategy=layer_budget_strategy,
            layer_budget_alpha=layer_budget_alpha,
            layer_budget_min_tokens=layer_budget_min_tokens,
            layer_budget_eps=layer_budget_eps,
            layer_budget_debug=layer_budget_debug,
            layer_budget_log_path=layer_budget_log_path,
            layer_budget_step_idx=past_frame_idx,
            past_key_values=past_key_values,
            current_token_count=S * P,
        )
        scores = []
        layer_budget_scores = []
        updated_scores = self.last_scores.clone()
        updated_layer_budget_scores = self.last_layer_budget_scores.clone()
        self.last_global_attn_debug_trace = []
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
                            raw_block_idx=raw_block_idx,
                            cache_analysis_config=cache_analysis_config,
                            pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                            eviction_nn_analysis_config=eviction_nn_analysis_config,
                            eviction_policy=eviction_policy,
                            eviction_debug=eviction_debug,
                            leverage_sketch_dim=leverage_sketch_dim,
                            leverage_granularity=leverage_granularity,
                            leverage_feature=leverage_feature,
                            leverage_projection=leverage_projection,
                            leverage_head_mean_dim=leverage_head_mean_dim,
                            leverage_normalize_rows=leverage_normalize_rows,
                            leverage_approx_method=leverage_approx_method,
                            leverage_left_sketch_dim=leverage_left_sketch_dim,
                            leverage_right_jl_dim=leverage_right_jl_dim,
                            leverage_random_seed=leverage_random_seed,
                            leverage_eviction_selector=leverage_eviction_selector,
                            leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                            leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                            layer_budget_strategy=layer_budget_strategy,
                            layer_budget_eps=layer_budget_eps,
                            eviction_protect_recent_frames=eviction_protect_recent_frames,
                            anchor_token_count=anchor_token_count,
                            window_token_count=window_token_count,
                            recent_merge_config=recent_merge_config,
                            svd_eviction_merge_config=svd_eviction_merge_config,
                            voxel_covis_frame_ids=voxel_covis_frame_ids,
                            voxel_covis_enabled=voxel_covis_enabled,
                            voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                            global_attn_debug=global_attn_debug,
                        )
                    elif use_cache:
                        old_global_idx = global_idx
                        old_kv_read = past_key_values[old_global_idx] is not None
                        tokens, global_idx, global_intermediates, new_kv, current_scores = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=current_budgets[global_idx].item(),
                            cache_analysis_config=cache_analysis_config,
                            pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                            eviction_nn_analysis_config=eviction_nn_analysis_config,
                            eviction_policy=eviction_policy,
                            eviction_debug=eviction_debug,
                            leverage_sketch_dim=leverage_sketch_dim,
                            leverage_granularity=leverage_granularity,
                            leverage_feature=leverage_feature,
                            leverage_projection=leverage_projection,
                            leverage_head_mean_dim=leverage_head_mean_dim,
                            leverage_normalize_rows=leverage_normalize_rows,
                            leverage_approx_method=leverage_approx_method,
                            leverage_left_sketch_dim=leverage_left_sketch_dim,
                            leverage_right_jl_dim=leverage_right_jl_dim,
                            leverage_random_seed=leverage_random_seed,
                            leverage_eviction_selector=leverage_eviction_selector,
                            leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                            leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                            layer_budget_strategy=layer_budget_strategy,
                            layer_budget_eps=layer_budget_eps,
                            eviction_protect_recent_frames=eviction_protect_recent_frames,
                            anchor_token_count=anchor_token_count,
                            window_token_count=window_token_count,
                            recent_merge_config=recent_merge_config,
                            svd_eviction_merge_config=svd_eviction_merge_config,
                            voxel_covis_frame_ids=voxel_covis_frame_ids,
                            voxel_covis_enabled=voxel_covis_enabled,
                            voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                        )
                        past_key_values[global_idx - 1] = new_kv
                        current_score, current_layer_budget_score = self._split_score_payload(current_scores)
                        if current_score is not None: # pruning happened
                            scores.append(current_score)
                        else:
                            scores.append(self.last_scores[global_idx-1].item())
                        if current_layer_budget_score is not None:
                            layer_budget_scores.append(float(torch.as_tensor(current_layer_budget_score).detach().float().item()))
                        else:
                            layer_budget_scores.append(self.last_layer_budget_scores[global_idx-1].item())
                        self._record_global_attn_debug(
                            enabled=global_attn_debug,
                            raw_block_idx=raw_block_idx,
                            original_attention_type="global",
                            global_idx=old_global_idx,
                            global_enabled=True,
                            g2f_conversion=False,
                            kv_read=old_kv_read,
                            kv_write=True,
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
            self.last_layer_budget_scores = updated_layer_budget_scores.to(
                device=self.last_layer_budget_scores.device,
                dtype=self.last_layer_budget_scores.dtype,
            )
        elif scores: # update scores
            self.last_scores = torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype)
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
        raw_block_idx=0,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        eviction_policy: str = "mean",
        eviction_debug: bool = False,
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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_debug: bool = False,
        eviction_protect_recent_frames: int = 0,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[List[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        global_attn_debug: bool = False,
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
                    tokens, global_idx, block_intermediates, new_kv, current_scores = self._process_global_attention(
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
                        eviction_policy=eviction_policy,
                        eviction_debug=eviction_debug,
                        leverage_sketch_dim=leverage_sketch_dim,
                        leverage_granularity=leverage_granularity,
                        leverage_feature=leverage_feature,
                        leverage_projection=leverage_projection,
                        leverage_head_mean_dim=leverage_head_mean_dim,
                        leverage_normalize_rows=leverage_normalize_rows,
                        leverage_approx_method=leverage_approx_method,
                        leverage_left_sketch_dim=leverage_left_sketch_dim,
                        leverage_right_jl_dim=leverage_right_jl_dim,
                        leverage_random_seed=leverage_random_seed,
                        leverage_eviction_selector=leverage_eviction_selector,
                        leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                        leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                        layer_budget_strategy=layer_budget_strategy,
                        layer_budget_eps=layer_budget_eps,
                        eviction_protect_recent_frames=eviction_protect_recent_frames,
                        anchor_token_count=anchor_token_count,
                        window_token_count=window_token_count,
                        recent_merge_config=recent_merge_config,
                        svd_eviction_merge_config=svd_eviction_merge_config,
                        voxel_covis_frame_ids=voxel_covis_frame_ids,
                        voxel_covis_enabled=voxel_covis_enabled,
                        voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                        block_count=1,
                    )
                    kv_write = True
                    past_key_values[current_global_idx] = new_kv
                    self._assert_enabled_cache_compatible(before_cache_shape, new_kv, current_global_idx)
                    current_score, current_layer_budget_score = self._split_score_payload(current_scores)
                    if current_score is not None:
                        updated_scores[current_global_idx] = torch.as_tensor(
                            current_score,
                            device=updated_scores.device,
                            dtype=updated_scores.dtype,
                        )
                    if current_layer_budget_score is not None and updated_layer_budget_scores is not None:
                        updated_layer_budget_scores[current_global_idx] = torch.as_tensor(
                            current_layer_budget_score,
                            device=updated_layer_budget_scores.device,
                            dtype=updated_layer_budget_scores.dtype,
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

        return tokens, global_idx, intermediates, updated_scores, updated_layer_budget_scores, raw_block_idx

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
        eviction_policy: str = "mean",
        eviction_debug: bool = False,
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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_debug: bool = False,
        eviction_protect_recent_frames: int = 0,
        anchor_token_count: Optional[int] = None,
        window_token_count: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[List[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
        block_count: Optional[int] = None,
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
                tokens, block_kv, scores = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget,
                    cache_analysis_config=cache_analysis_config,
                    pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                    eviction_nn_analysis_config=eviction_nn_analysis_config,
                    layer_id=global_idx,
                    step_idx=past_frame_idx,
                    tokens_per_frame=P,
                    eviction_policy=eviction_policy,
                    eviction_debug=eviction_debug,
                    leverage_sketch_dim=leverage_sketch_dim,
                    leverage_granularity=leverage_granularity,
                    leverage_feature=leverage_feature,
                    leverage_projection=leverage_projection,
                    leverage_head_mean_dim=leverage_head_mean_dim,
                    leverage_normalize_rows=leverage_normalize_rows,
                    leverage_approx_method=leverage_approx_method,
                    leverage_left_sketch_dim=leverage_left_sketch_dim,
                    leverage_right_jl_dim=leverage_right_jl_dim,
                    leverage_random_seed=leverage_random_seed,
                    leverage_eviction_selector=leverage_eviction_selector,
                    leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                    leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                    layer_budget_strategy=layer_budget_strategy,
                    layer_budget_eps=layer_budget_eps,
                    eviction_protect_recent_frames=eviction_protect_recent_frames,
                    anchor_token_count=anchor_token_count,
                    window_token_count=window_token_count,
                    recent_merge_config=recent_merge_config,
                    svd_eviction_merge_config=svd_eviction_merge_config,
                    voxel_covis_frame_ids=voxel_covis_frame_ids,
                    voxel_covis_enabled=voxel_covis_enabled,
                    voxel_covis_fallback_recent=voxel_covis_fallback_recent,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv, scores
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
        

    def sync_anchor_change(
        self,
        past_key_values,
        anchor_token_count: int,
        tokens_per_frame: int,
        anchor_keep_ratio: float,
        anchor_token_indices: Optional[torch.Tensor] = None,
        is_fifo: bool = False,
    ):
        """Promote newest-frame tokens into the protected anchor zone."""
        if past_key_values is None or anchor_token_count is None:
            return past_key_values

        anchor_token_count = int(anchor_token_count)
        tokens_per_frame = int(tokens_per_frame)
        if tokens_per_frame <= 0 or anchor_token_count <= tokens_per_frame:
            return past_key_values

        if anchor_token_indices is not None:
            anchor_chunk = int(anchor_token_indices.shape[-1])
        else:
            anchor_chunk = max(int(tokens_per_frame * anchor_keep_ratio), 1)
        anchor_chunk = min(anchor_chunk, tokens_per_frame)
        if anchor_chunk <= 0:
            return past_key_values

        global_anchor_end = tokens_per_frame

        for idx in range(self.depth):
            if past_key_values[idx] is None:
                continue

            layer_kv = past_key_values[idx]
            if len(layer_kv) == 3:
                k, v, metadata = layer_kv
            else:
                k, v = layer_kv
                metadata = None

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
                past_key_values[idx] = (k_new, v_new, metadata)
            else:
                past_key_values[idx] = (k_new, v_new)

        return past_key_values

    @staticmethod
    def _split_score_payload(score_payload):
        if isinstance(score_payload, tuple):
            if len(score_payload) != 2:
                raise ValueError(f"Unexpected score payload length: {len(score_payload)}")
            return score_payload
        return score_payload, None

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

    def _print_layer_budget_debug(self, strategy, total_budget, alpha, min_tokens, capacities, budgets, details):
        active_layers = [layer for layer, cap in capacities.items() if cap > 0]
        print(
            f"[LayerBudget/{strategy}] "
            f"total_budget={int(total_budget)} "
            f"assigned_budget={details.get('assigned_budget', sum(budgets.values()))} "
            f"active_layers={len(active_layers)} alpha={alpha} min_tokens={min_tokens}"
        )
        for event in details.get("events", []):
            print(f"[LayerBudget/{strategy}] {event}")
        for layer in active_layers:
            print(
                f"[LayerBudget/{strategy}] "
                f"layer={layer:02d} N={capacities[layer]} "
                f"score={details.get('scores', {}).get(layer, 0.0):.6g} "
                f"weight={details.get('weights', {}).get(layer, 0.0):.6g} "
                f"raw_budget={details.get('raw_budgets', {}).get(layer, 0.0):.3f} "
                f"final_budget={budgets.get(layer, 0)}"
            )

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
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_debug: bool = False,
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
            score_dict = {
                idx: self.last_layer_budget_scores[idx]
                for idx in range(len(self.last_layer_budget_scores))
            }
            budget_dict, details = _allocate_layer_budgets_from_scores(
                score_dict,
                capacities,
                total_budget,
                alpha=layer_budget_alpha,
                min_tokens=layer_budget_min_tokens,
                eps=layer_budget_eps,
                return_debug=True,
            )
            budgets = torch.zeros_like(self.last_scores, dtype=torch.int32)
            for idx, budget in budget_dict.items():
                budgets[idx] = int(budget)
            if layer_budget_debug:
                self._print_layer_budget_debug(
                    layer_budget_strategy,
                    total_budget,
                    layer_budget_alpha,
                    layer_budget_min_tokens,
                    capacities,
                    budget_dict,
                    details,
                )
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
