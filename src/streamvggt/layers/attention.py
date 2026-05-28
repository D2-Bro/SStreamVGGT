import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional, Sequence

from streamvggt.utils.cache_analysis import (
    CacheAnalysisConfig,
    PreEvictionSnapshotConfig,
    dump_eviction_snapshot,
    dump_pre_eviction_snapshot,
)
from streamvggt.layers.eviction import EvictionManager
from streamvggt.layers.recent_merge import KVCacheMetadata, RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig, SvdEvictionMerger

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
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        eviction_protect_recent_frames: int = 0,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
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
            leverage_projection,
            leverage_head_mean_dim,
            leverage_normalize_rows,
        )
        eviction = self._eviction_managers.get(manager_key)
        if eviction is None:
            eviction = EvictionManager(
                policy=eviction_policy,
                debug=eviction_debug,
                leverage_sketch_dim=leverage_sketch_dim,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
                leverage_projection=leverage_projection,
                leverage_head_mean_dim=leverage_head_mean_dim,
                leverage_normalize_rows=leverage_normalize_rows,
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
            current_frame_idx=step_idx,
            protect_recent_frames=eviction_protect_recent_frames,
            candidate_frame_ids=(
                metadata.frame_ids[:, :, num_anchor_tokens:]
                if metadata is not None
                else None
            ),
            need_leverage_basis=(
                svd_eviction_merge_config is not None
                and svd_eviction_merge_config.enabled
                and eviction_policy == "svd_leverage"
            ),
        )
        top_indices = eviction_result.kept_candidate_indices
        avg_scores = eviction_result.summary_score

        if (
            svd_eviction_merge_config is not None
            and svd_eviction_merge_config.enabled
            and eviction_policy == "svd_leverage"
        ):
            merger = SvdEvictionMerger(svd_eviction_merge_config, num_anchor_tokens=num_anchor_tokens)
            merger.merge(k, v, metadata, eviction_result, layer_id=layer_id, step_idx=step_idx)

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
        final_cache_size = keep_indices.shape[2]
        expanded_indices = keep_indices.unsqueeze(-1).expand(B, H, final_cache_size, D)
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
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        eviction_protect_recent_frames: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_frame_ids: Optional[Sequence[int]] = None,
        voxel_covis_enabled: bool = False,
        voxel_covis_fallback_recent: int = 0,
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
            metadata_needed = (
                (recent_merge_config is not None and recent_merge_config.enabled)
                or (svd_eviction_merge_config is not None and svd_eviction_merge_config.enabled and eviction_policy == "svd_leverage")
                or int(eviction_protect_recent_frames) > 0
                or bool(voxel_covis_enabled)
            )
            if metadata_needed:
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
                if metadata is not None and past_metadata is None:
                    past_metadata = KVCacheMetadata.for_current_frame(
                        batch_size=B,
                        num_heads=self.num_heads,
                        num_tokens=past_k.shape[2],
                        frame_id=-1,
                    )
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
                eviction_kwargs = {
                    "cache_analysis_config": cache_analysis_config,
                    "layer_id": layer_id,
                    "step_idx": step_idx,
                    "tokens_per_frame": tokens_per_frame,
                    "eviction_policy": eviction_policy,
                    "eviction_debug": eviction_debug,
                    "leverage_sketch_dim": leverage_sketch_dim,
                    "leverage_granularity": leverage_granularity,
                    "leverage_feature": leverage_feature,
                    "leverage_projection": leverage_projection,
                    "leverage_head_mean_dim": leverage_head_mean_dim,
                    "eviction_protect_recent_frames": eviction_protect_recent_frames,
                    "svd_eviction_merge_config": svd_eviction_merge_config,
                }
                if leverage_normalize_rows:
                    eviction_kwargs["leverage_normalize_rows"] = leverage_normalize_rows
                k, v, metadata, scores = self.eviction(
                    k,
                    v,
                    metadata,
                    cache_budget,
                    self.num_anchor_tokens,
                    **eviction_kwargs,
                )

            new_kv = (k, v, metadata) if metadata is not None else (k, v)
            if voxel_covis_enabled and metadata is not None and voxel_covis_frame_ids is not None:
                k_read, v_read, covis_attn_mask = _filter_kv_for_voxel_covis(
                    k,
                    v,
                    metadata,
                    selected_frame_ids=voxel_covis_frame_ids,
                    current_frame_id=step_idx if step_idx is not None else 0,
                    query_len=N,
                    fallback_recent=voxel_covis_fallback_recent,
                )
                k_for_attn = k_read
                v_for_attn = v_read
                attn_mask = _merge_attn_masks(attn_mask, covis_attn_mask, q.dtype)
            else:
                k_for_attn = k
                v_for_attn = v
        else:
            k_for_attn = k
            v_for_attn = v

        if self.fused_attn:
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

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
                return x, new_kv, scores
        return x


def _filter_kv_for_voxel_covis(
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: KVCacheMetadata,
    selected_frame_ids: Sequence[int],
    current_frame_id: int,
    query_len: int,
    fallback_recent: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a padded read-only cache containing selected past frames plus current tokens."""
    B, H, _, D = k.shape
    frame_ids = metadata.frame_ids
    selected = torch.as_tensor(list(selected_frame_ids), dtype=torch.long)
    keep_masks = []
    max_count = 0
    for b in range(B):
        row = []
        for h in range(H):
            head_frames = frame_ids[b, h]
            if selected.numel() > 0:
                selected_mask = torch.isin(head_frames, selected)
            else:
                selected_mask = torch.zeros_like(head_frames, dtype=torch.bool)
            has_selected_past = bool((selected_mask & (head_frames != int(current_frame_id))).any().item())
            if not has_selected_past and int(fallback_recent) > 0:
                fallback_frames = _recent_cached_frame_ids(
                    head_frames,
                    int(current_frame_id),
                    int(fallback_recent),
                )
                if fallback_frames:
                    fallback = torch.as_tensor(fallback_frames, dtype=head_frames.dtype)
                    selected_mask = selected_mask | torch.isin(head_frames, fallback)
            mask = selected_mask | (head_frames == int(current_frame_id))
            count = int(mask.sum().item())
            max_count = max(max_count, count)
            row.append(mask)
        keep_masks.append(row)

    if max_count == 0:
        return k, v, torch.zeros((B, H, query_len, k.shape[2]), device=k.device, dtype=k.dtype)

    if max_count == k.shape[2] and all(bool(mask.all()) for row in keep_masks for mask in row):
        return k, v, torch.zeros((B, H, query_len, k.shape[2]), device=k.device, dtype=k.dtype)

    k_read = torch.zeros((B, H, max_count, D), device=k.device, dtype=k.dtype)
    v_read = torch.zeros((B, H, max_count, D), device=v.device, dtype=v.dtype)
    valid = torch.zeros((B, H, max_count), device=k.device, dtype=torch.bool)

    for b in range(B):
        for h in range(H):
            indices = torch.nonzero(keep_masks[b][h], as_tuple=False).flatten().to(device=k.device)
            count = int(indices.numel())
            if count == 0:
                continue
            k_read[b, h, :count] = k[b, h].index_select(0, indices)
            v_read[b, h, :count] = v[b, h].index_select(0, indices)
            valid[b, h, :count] = True

    min_value = torch.finfo(k.dtype).min
    mask = torch.zeros((B, H, 1, max_count), device=k.device, dtype=k.dtype)
    mask = mask.masked_fill(~valid.unsqueeze(2), min_value)
    mask = mask.expand(B, H, query_len, max_count)
    return k_read, v_read, mask


def _recent_cached_frame_ids(head_frames: torch.Tensor, current_frame_id: int, count: int) -> list[int]:
    if count <= 0:
        return []
    past = head_frames[head_frames < int(current_frame_id)]
    if past.numel() == 0:
        return []
    return sorted({int(fid) for fid in past.tolist()}, reverse=True)[:count]


def _merge_attn_masks(
    existing_mask: Optional[torch.Tensor],
    covis_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    covis_mask = covis_mask.to(dtype=dtype)
    if existing_mask is None:
        return covis_mask
    return existing_mask.to(device=covis_mask.device, dtype=dtype) + covis_mask


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
