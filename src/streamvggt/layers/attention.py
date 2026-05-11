import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

from streamvggt.utils.cache_analysis import (
    CacheAnalysisConfig,
    PreEvictionSnapshotConfig,
    dump_eviction_snapshot,
    dump_pre_eviction_snapshot,
)
from streamvggt.layers.eviction import EvictionManager
from streamvggt.layers.recent_merge import KVCacheMetadata, RecentMergeConfig

XFORMERS_AVAILABLE = False



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

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0

    def eviction(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        metadata: Optional[KVCacheMetadata],
        cache_budget: int,
        num_anchor_tokens: int,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        eviction_policy: str = "mean",
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[KVCacheMetadata], float]:
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

        if N <= cache_budget:
            return k, v, metadata, 0.0

        manager_key = (
            eviction_policy,
            eviction_debug,
            leverage_sketch_dim,
            leverage_granularity,
            leverage_feature,
        )
        eviction = self._eviction_managers.get(manager_key)
        if eviction is None:
            eviction = EvictionManager(
                policy=eviction_policy,
                debug=eviction_debug,
                leverage_sketch_dim=leverage_sketch_dim,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
            )
            self._eviction_managers[manager_key] = eviction
        eviction_result = eviction.select(
            k,
            cache_budget,
            num_anchor_tokens,
            v=v,
            need_summary=cache_analysis_config is not None or eviction_debug,
            layer_id=layer_id,
            step_idx=step_idx,
        )
        top_indices = eviction_result.kept_candidate_indices
        avg_scores = eviction_result.summary_score

        if cache_analysis_config is not None and layer_id is not None and step_idx is not None:
            dump_eviction_snapshot(
                cache_analysis_config,
                k_before=k,
                scores=eviction_result.mean_scores,
                kept_candidate_indices=top_indices,
                layer_id=layer_id,
                step_idx=step_idx,
                cache_budget=cache_budget,
                num_anchor_tokens=num_anchor_tokens,
                tokens_per_frame=tokens_per_frame,
                eviction_policy=eviction_policy,
                leverage_sketch_dim=leverage_sketch_dim,
            )
        
        anchor_indices = torch.arange(num_anchor_tokens, device=k.device, dtype=torch.long)
        anchor_indices = anchor_indices.view(1, 1, num_anchor_tokens).expand(B, H, num_anchor_tokens)
        keep_indices = torch.cat([anchor_indices, top_indices + int(num_anchor_tokens)], dim=2)
        expanded_indices = keep_indices.unsqueeze(-1).expand(B, H, cache_budget, D)
        final_k = torch.gather(k, 2, expanded_indices)
        final_v = torch.gather(v, 2, expanded_indices)
        final_metadata = (
            metadata.prune_after_eviction(top_indices, num_anchor_tokens)
            if metadata is not None
            else None
        )

        return final_k, final_v, final_metadata, avg_scores

    def forward(self, 
        x: torch.Tensor, 
        pos=None, 
        attn_mask=None, 
        past_key_values=None, 
        use_cache=False,
        cache_budget = None,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        eviction_policy: str = "mean",
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        recent_merge_config: Optional[RecentMergeConfig] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = None
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if use_cache and self.num_anchor_tokens == 0:
            self.num_anchor_tokens = k.shape[2] 

        if use_cache:
            metadata = None
            if recent_merge_config is not None and recent_merge_config.enabled:
                metadata = KVCacheMetadata.for_current_frame(
                    batch_size=B,
                    num_heads=self.num_heads,
                    num_tokens=k.shape[2],
                    frame_id=step_idx if step_idx is not None else 0,
                )
            if past_key_values is not None:
                if len(past_key_values) == 3:
                    past_k, past_v, past_metadata = past_key_values
                else:
                    past_k, past_v = past_key_values
                    past_metadata = None
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
                if metadata is not None and past_metadata is not None:
                    metadata = past_metadata.concat(metadata)
                elif metadata is not None:
                    metadata = None
            if (
                pre_eviction_snapshot_config is not None
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
                    num_anchor_tokens=self.num_anchor_tokens,
                    tokens_per_frame=tokens_per_frame,
                )
            eviction_deferred_for_snapshot = (
                pre_eviction_snapshot_config is not None
                and step_idx is not None
                and step_idx <= pre_eviction_snapshot_config.target_step_idx
            )
            if cache_budget is not None and k.shape[2] > cache_budget and not eviction_deferred_for_snapshot:
                k, v, metadata, scores = self.eviction(
                    k,
                    v,
                    metadata,
                    cache_budget,
                    self.num_anchor_tokens,
                    cache_analysis_config=cache_analysis_config,
                    layer_id=layer_id,
                    step_idx=step_idx,
                    tokens_per_frame=tokens_per_frame,
                    eviction_policy=eviction_policy,
                    eviction_debug=eviction_debug,
                    leverage_sketch_dim=leverage_sketch_dim,
                    leverage_granularity=leverage_granularity,
                    leverage_feature=leverage_feature,
                )

            new_kv = (k, v, metadata) if metadata is not None else (k, v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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
