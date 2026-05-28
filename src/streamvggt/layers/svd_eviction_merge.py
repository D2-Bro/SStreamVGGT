"""Feature-first SVD-guided merge for tokens selected by cache eviction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from streamvggt.layers.eviction import EvictionResult, SvdLeverageBasis
from streamvggt.layers.recent_merge import KVCacheMetadata

_GEOMETRY_GATES = {"none", "voxel_neighbor"}
_MERGE_MODES = {"head", "layer_candidates", "layer"}


@dataclass
class SvdEvictionMergeConfig:
    enabled: bool = False
    mode: str = "head"
    candidate_axes: int = 2
    reps_per_axis: int = 8
    similarity_threshold: float = 0.9
    use_u_sigma: bool = True
    geometry_gate: str = "voxel_neighbor"
    voxel_neighbor_radius: int = 1
    allow_missing_geometry: bool = True
    ema_decay: float = 0.5
    use_depth_confidence: bool = True
    max_candidates_per_token: int = 32
    chunk_size: int = 512
    debug: bool = False
    profile: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _MERGE_MODES:
            raise ValueError(f"mode must be one of {sorted(_MERGE_MODES)}, got {self.mode!r}")
        if self.candidate_axes < 1:
            raise ValueError(f"candidate_axes must be >= 1, got {self.candidate_axes}")
        if self.reps_per_axis < 1:
            raise ValueError(f"reps_per_axis must be >= 1, got {self.reps_per_axis}")
        if not (0.0 <= float(self.similarity_threshold) <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        if self.geometry_gate not in _GEOMETRY_GATES:
            raise ValueError(f"geometry_gate must be one of {sorted(_GEOMETRY_GATES)}, got {self.geometry_gate!r}")
        if self.voxel_neighbor_radius < 0:
            raise ValueError(f"voxel_neighbor_radius must be >= 0, got {self.voxel_neighbor_radius}")
        if not (0.0 <= float(self.ema_decay) <= 1.0):
            raise ValueError("ema_decay must be in [0, 1]")
        if self.max_candidates_per_token < 1:
            raise ValueError(f"max_candidates_per_token must be >= 1, got {self.max_candidates_per_token}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")


@dataclass
class SvdEvictionMergeStats:
    evicted_tokens: int = 0
    retained_tokens: int = 0
    candidate_pairs: int = 0
    attempted_pairs: int = 0
    accepted_pairs: int = 0
    rejected_similarity: int = 0
    rejected_geometry: int = 0
    missing_geometry: int = 0
    profile_times: Optional[Dict[str, float]] = None


class SvdEvictionMerger:
    """Merge evicted KV entries into retained entries using low-rank leverage coordinates."""

    def __init__(self, config: SvdEvictionMergeConfig, num_anchor_tokens: int) -> None:
        self.config = config
        self.num_anchor_tokens = int(num_anchor_tokens)

    def merge(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        eviction_result: EvictionResult,
        *,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
    ) -> SvdEvictionMergeStats:
        stats = SvdEvictionMergeStats(profile_times={} if self.config.profile else None)
        if not self.config.enabled or eviction_result.leverage_basis is None:
            return stats
        basis = eviction_result.leverage_basis
        if basis.q.shape[-1] == 0:
            return stats

        total_start = self._profile_start(k)
        B, H, N, _ = k.shape
        candidate_count = N - self.num_anchor_tokens
        if candidate_count <= 0:
            return stats

        all_candidates = torch.arange(candidate_count, device=k.device, dtype=torch.long)
        if self.config.mode == "head":
            for b in range(B):
                for h in range(H):
                    kept = eviction_result.kept_candidate_indices[b, h].to(device=k.device, dtype=torch.long)
                    keep_mask = torch.zeros(candidate_count, device=k.device, dtype=torch.bool)
                    keep_mask[kept] = True
                    evicted = all_candidates[~keep_mask]
                    stats.evicted_tokens += int(evicted.numel())
                    stats.retained_tokens += int(kept.numel())
                    if evicted.numel() == 0 or kept.numel() == 0:
                        continue
                    self._merge_one_head(k, v, metadata, basis, b, h, kept, evicted, stats)
        else:
            self._validate_layer_mode_inputs(eviction_result, basis)
            for b in range(B):
                kept, evicted = self._shared_kept_evicted(eviction_result, all_candidates, candidate_count, b, k.device)
                if evicted.numel() == 0 or kept.numel() == 0:
                    continue
                if self.config.mode == "layer_candidates":
                    self._merge_layer_candidates(k, v, metadata, basis, b, kept, evicted, stats)
                else:
                    self._merge_one_layer(k, v, metadata, basis, b, kept, evicted, stats)

        self._profile_add(stats, "total", total_start, k)
        if self.config.debug:
            profile = ""
            if stats.profile_times:
                profile = " " + " ".join(f"{name}={value * 1000.0:.3f}ms" for name, value in stats.profile_times.items())
            print(
                "[SvdEvictionMerger] "
                f"mode={self.config.mode} layer={layer_id} step={step_idx} evicted={stats.evicted_tokens} "
                f"retained={stats.retained_tokens} candidates={stats.candidate_pairs} "
                f"attempted={stats.attempted_pairs} accepted={stats.accepted_pairs} "
                f"rejected_similarity={stats.rejected_similarity} "
                f"rejected_geometry={stats.rejected_geometry} missing_geometry={stats.missing_geometry}"
                f"{profile}"
            )
        return stats

    def _merge_one_head(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        basis: SvdLeverageBasis,
        b: int,
        h: int,
        kept: torch.Tensor,
        evicted: torch.Tensor,
        stats: SvdEvictionMergeStats,
    ) -> None:
        candidate_start = self._profile_start(k)
        coords = self._basis_for_head(basis, b, h, device=k.device)
        if coords.numel() == 0:
            return
        axis_ids = self._axis_ids(basis, b, h, coords.shape[-1], device=k.device)
        coords = coords[:, axis_ids]
        retained_coords = coords[kept]
        evicted_coords = coords[evicted]
        candidate_cols = self._build_candidate_matrix(retained_coords, evicted_coords)
        self._profile_add(stats, "candidate_build", candidate_start, k)
        self._merge_one_head_with_candidates(k, v, metadata, b, h, kept, evicted, candidate_cols, stats)

    def _merge_one_head_with_candidates(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        b: int,
        h: int,
        kept: torch.Tensor,
        evicted: torch.Tensor,
        candidate_cols: torch.Tensor,
        stats: SvdEvictionMergeStats,
    ) -> None:
        if candidate_cols.numel() == 0:
            return

        cosine_start = self._profile_start(k)
        retained_abs = kept + self.num_anchor_tokens
        evicted_abs = evicted + self.num_anchor_tokens
        retained_k = F.normalize(k[b, h, retained_abs].float(), dim=-1)
        evicted_k = F.normalize(k[b, h, evicted_abs].float(), dim=-1)
        accepted_scores = []
        accepted_src = []
        accepted_dst = []
        threshold = float(self.config.similarity_threshold)
        max_candidates = int(candidate_cols.shape[1])
        stats.candidate_pairs += int(candidate_cols.numel())
        stats.attempted_pairs += int(candidate_cols.numel())
        for start in range(0, evicted.numel(), int(self.config.chunk_size)):
            end = min(start + int(self.config.chunk_size), evicted.numel())
            chunk_cols = candidate_cols[start:end]
            local_cand = retained_k[chunk_cols.reshape(-1)].view(
                chunk_cols.shape[0],
                max_candidates,
                retained_k.shape[-1],
            )
            sim = (evicted_k[start:end].unsqueeze(1) * local_cand).sum(dim=-1)
            values, indices = sim.max(dim=1)
            above = values >= threshold
            stats.rejected_similarity += int((~above).sum().item())
            if not bool(above.any()):
                continue
            rows = torch.nonzero(above, as_tuple=False).flatten()
            accepted_value = values[rows]
            dst_cols = chunk_cols[rows, indices[rows]]
            src_abs = evicted_abs[start:end][rows]
            dst_abs = retained_abs[dst_cols]
            if self.config.geometry_gate != "none":
                geom_mask, missing = self._geometry_mask(metadata, b, h, src_abs, dst_abs)
                stats.missing_geometry += int(missing)
                stats.rejected_geometry += int((~geom_mask).sum().item())
                if not bool(geom_mask.any()):
                    continue
                accepted_value = accepted_value[geom_mask]
                src_abs = src_abs[geom_mask]
                dst_abs = dst_abs[geom_mask]
            accepted_scores.append(accepted_value)
            accepted_src.append(src_abs)
            accepted_dst.append(dst_abs)
        self._profile_add(stats, "cosine_match", cosine_start, k)

        if not accepted_src:
            return
        scores = torch.cat(accepted_scores, dim=0)
        src_abs = torch.cat(accepted_src, dim=0).to(dtype=torch.long)
        dst_abs = torch.cat(accepted_dst, dim=0).to(dtype=torch.long)
        if src_abs.numel() == 0:
            return

        # Keep the strongest source per retained destination. This avoids slow repeated
        # scalar GPU writes and makes the merge explicitly one retained token update per pass.
        order = torch.argsort(scores, descending=True)
        src_abs = src_abs[order]
        dst_abs = dst_abs[order]
        if dst_abs.numel() > 1:
            unique_dst, inverse = torch.unique(dst_abs, sorted=False, return_inverse=True)
            first = torch.full((unique_dst.numel(),), dst_abs.numel(), device=dst_abs.device, dtype=torch.long)
            positions = torch.arange(dst_abs.numel(), device=dst_abs.device, dtype=torch.long)
            first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
            keep_pair = torch.zeros(dst_abs.numel(), device=dst_abs.device, dtype=torch.bool)
            keep_pair[first] = True
            src_abs = src_abs[keep_pair]
            dst_abs = dst_abs[keep_pair]

        update_start = self._profile_start(k)
        with torch.no_grad():
            self._apply_merges(k, v, metadata, b, h, src_abs, dst_abs)
            stats.accepted_pairs += int(src_abs.numel())
        self._profile_add(stats, "ema_update", update_start, k)

    def _validate_layer_mode_inputs(self, eviction_result: EvictionResult, basis: SvdLeverageBasis) -> None:
        if basis.granularity != "layer":
            raise ValueError(
                f"svd_eviction_merge mode={self.config.mode!r} requires layer-wise leverage_basis, "
                f"got granularity={basis.granularity!r}"
            )
        kept = eviction_result.kept_candidate_indices
        if kept.ndim != 3:
            raise ValueError(f"kept_candidate_indices must be [B, H, K], got shape={tuple(kept.shape)}")
        if kept.shape[1] > 1 and not bool((kept == kept[:, :1, :]).all().item()):
            raise ValueError(
                f"svd_eviction_merge mode={self.config.mode!r} requires identical kept indices across heads"
            )

    def _shared_kept_evicted(
        self,
        eviction_result: EvictionResult,
        all_candidates: torch.Tensor,
        candidate_count: int,
        b: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kept = eviction_result.kept_candidate_indices[b, 0].to(device=device, dtype=torch.long)
        keep_mask = torch.zeros(candidate_count, device=device, dtype=torch.bool)
        keep_mask[kept] = True
        evicted = all_candidates[~keep_mask]
        return kept, evicted

    def _build_layer_candidates(
        self,
        k: torch.Tensor,
        basis: SvdLeverageBasis,
        b: int,
        kept: torch.Tensor,
        evicted: torch.Tensor,
        stats: SvdEvictionMergeStats,
    ) -> torch.Tensor:
        candidate_start = self._profile_start(k)
        coords = self._basis_for_head(basis, b, 0, device=k.device)
        if coords.numel() == 0:
            return torch.empty(evicted.shape[0], 0, device=k.device, dtype=torch.long)
        axis_ids = self._axis_ids(basis, b, 0, coords.shape[-1], device=k.device)
        coords = coords[:, axis_ids]
        retained_coords = coords[kept]
        evicted_coords = coords[evicted]
        candidate_cols = self._build_candidate_matrix(retained_coords, evicted_coords)
        self._profile_add(stats, "candidate_build", candidate_start, k)
        return candidate_cols

    def _merge_layer_candidates(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        basis: SvdLeverageBasis,
        b: int,
        kept: torch.Tensor,
        evicted: torch.Tensor,
        stats: SvdEvictionMergeStats,
    ) -> None:
        candidate_cols = self._build_layer_candidates(k, basis, b, kept, evicted, stats)
        if candidate_cols.numel() == 0:
            return
        H = k.shape[1]
        for h in range(H):
            stats.evicted_tokens += int(evicted.numel())
            stats.retained_tokens += int(kept.numel())
            self._merge_one_head_with_candidates(k, v, metadata, b, h, kept, evicted, candidate_cols, stats)

    def _merge_one_layer(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        basis: SvdLeverageBasis,
        b: int,
        kept: torch.Tensor,
        evicted: torch.Tensor,
        stats: SvdEvictionMergeStats,
    ) -> None:
        candidate_cols = self._build_layer_candidates(k, basis, b, kept, evicted, stats)
        if candidate_cols.numel() == 0:
            return

        cosine_start = self._profile_start(k)
        retained_abs = kept + self.num_anchor_tokens
        evicted_abs = evicted + self.num_anchor_tokens
        retained_k = F.normalize(k[b, :, retained_abs].float().permute(1, 0, 2).flatten(1), dim=-1)
        evicted_k = F.normalize(k[b, :, evicted_abs].float().permute(1, 0, 2).flatten(1), dim=-1)
        accepted_scores = []
        accepted_src = []
        accepted_dst = []
        threshold = float(self.config.similarity_threshold)
        max_candidates = int(candidate_cols.shape[1])
        stats.evicted_tokens += int(evicted.numel())
        stats.retained_tokens += int(kept.numel())
        stats.candidate_pairs += int(candidate_cols.numel())
        stats.attempted_pairs += int(candidate_cols.numel())
        for start in range(0, evicted.numel(), int(self.config.chunk_size)):
            end = min(start + int(self.config.chunk_size), evicted.numel())
            chunk_cols = candidate_cols[start:end]
            local_cand = retained_k[chunk_cols.reshape(-1)].view(
                chunk_cols.shape[0],
                max_candidates,
                retained_k.shape[-1],
            )
            sim = (evicted_k[start:end].unsqueeze(1) * local_cand).sum(dim=-1)
            values, indices = sim.max(dim=1)
            above = values >= threshold
            stats.rejected_similarity += int((~above).sum().item())
            if not bool(above.any()):
                continue
            rows = torch.nonzero(above, as_tuple=False).flatten()
            accepted_value = values[rows]
            dst_cols = chunk_cols[rows, indices[rows]]
            src_abs = evicted_abs[start:end][rows]
            dst_abs = retained_abs[dst_cols]
            if self.config.geometry_gate != "none":
                geom_mask, missing = self._geometry_mask(metadata, b, 0, src_abs, dst_abs)
                stats.missing_geometry += int(missing)
                stats.rejected_geometry += int((~geom_mask).sum().item())
                if not bool(geom_mask.any()):
                    continue
                accepted_value = accepted_value[geom_mask]
                src_abs = src_abs[geom_mask]
                dst_abs = dst_abs[geom_mask]
            accepted_scores.append(accepted_value)
            accepted_src.append(src_abs)
            accepted_dst.append(dst_abs)
        self._profile_add(stats, "cosine_match", cosine_start, k)

        if not accepted_src:
            return
        scores = torch.cat(accepted_scores, dim=0)
        src_abs = torch.cat(accepted_src, dim=0).to(dtype=torch.long)
        dst_abs = torch.cat(accepted_dst, dim=0).to(dtype=torch.long)
        if src_abs.numel() == 0:
            return

        order = torch.argsort(scores, descending=True)
        src_abs = src_abs[order]
        dst_abs = dst_abs[order]
        if dst_abs.numel() > 1:
            unique_dst, inverse = torch.unique(dst_abs, sorted=False, return_inverse=True)
            first = torch.full((unique_dst.numel(),), dst_abs.numel(), device=dst_abs.device, dtype=torch.long)
            positions = torch.arange(dst_abs.numel(), device=dst_abs.device, dtype=torch.long)
            first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
            keep_pair = torch.zeros(dst_abs.numel(), device=dst_abs.device, dtype=torch.bool)
            keep_pair[first] = True
            src_abs = src_abs[keep_pair]
            dst_abs = dst_abs[keep_pair]

        update_start = self._profile_start(k)
        with torch.no_grad():
            self._apply_layer_merges(k, v, metadata, b, src_abs, dst_abs)
            stats.accepted_pairs += int(src_abs.numel())
        self._profile_add(stats, "ema_update", update_start, k)

    def _basis_for_head(self, basis: SvdLeverageBasis, b: int, h: int, *, device: torch.device) -> torch.Tensor:
        if basis.granularity == "head":
            q = basis.q[b, h]
            diag = basis.r_diag[b, h]
        else:
            q = basis.q[b]
            diag = basis.r_diag[b]
        coords = torch.nan_to_num(q.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.config.use_u_sigma and diag.numel() > 0:
            coords = coords * torch.nan_to_num(diag.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).view(1, -1)
        return coords

    def _axis_ids(self, basis: SvdLeverageBasis, b: int, h: int, rank: int, *, device: torch.device) -> torch.Tensor:
        axes = min(int(self.config.candidate_axes), int(rank))
        if axes <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        diag = basis.r_diag[b, h] if basis.granularity == "head" else basis.r_diag[b]
        diag = torch.nan_to_num(diag.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if diag.numel() >= axes:
            return torch.topk(diag.abs(), k=axes, dim=-1).indices.sort().values
        return torch.arange(axes, device=device, dtype=torch.long)

    def _build_candidate_matrix(self, retained_coords: torch.Tensor, evicted_coords: torch.Tensor) -> torch.Tensor:
        if retained_coords.numel() == 0 or evicted_coords.numel() == 0:
            return torch.empty(evicted_coords.shape[0], 0, device=evicted_coords.device, dtype=torch.long)
        reps = max(int(self.config.reps_per_axis), 1)
        half = max(reps // 2, 1)
        max_candidates = max(int(self.config.max_candidates_per_token), 1)
        offsets = torch.arange(reps, device=evicted_coords.device, dtype=torch.long) - int(half)
        per_axis = []
        retained_count = int(retained_coords.shape[0])
        for axis in range(retained_coords.shape[1]):
            sorted_vals, order = torch.sort(retained_coords[:, axis].contiguous())
            insert = torch.searchsorted(sorted_vals, evicted_coords[:, axis].contiguous())
            cols = (insert.unsqueeze(1) + offsets.view(1, -1)).clamp(0, retained_count - 1)
            per_axis.append(order[cols])
        candidate_cols = torch.cat(per_axis, dim=1)
        if candidate_cols.shape[1] > max_candidates:
            local_coords = retained_coords[candidate_cols.reshape(-1)].view(
                candidate_cols.shape[0],
                candidate_cols.shape[1],
                retained_coords.shape[-1],
            )
            dist = (local_coords - evicted_coords.unsqueeze(1)).abs().sum(dim=-1)
            _, chosen = torch.topk(-dist, k=max_candidates, dim=1)
            candidate_cols = torch.gather(candidate_cols, 1, chosen)
        return candidate_cols.contiguous()

    def _geometry_mask(
        self,
        metadata: Optional[KVCacheMetadata],
        b: int,
        h: int,
        src_abs: torch.Tensor,
        dst_abs: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if self.config.geometry_gate == "none":
            return torch.ones_like(src_abs, dtype=torch.bool), 0
        if metadata is None or metadata.voxel_ids is None or metadata.voxel_valid is None:
            return torch.full_like(src_abs, bool(self.config.allow_missing_geometry), dtype=torch.bool), int(src_abs.numel())
        src_cpu = src_abs.detach().cpu().long()
        dst_cpu = dst_abs.detach().cpu().long()
        src_valid = metadata.voxel_valid[b, h, src_cpu]
        dst_valid = metadata.voxel_valid[b, h, dst_cpu]
        has_geometry = src_valid & dst_valid
        missing = int((~has_geometry).sum().item())
        if not bool(has_geometry.any()):
            return torch.full_like(src_abs, bool(self.config.allow_missing_geometry), dtype=torch.bool), missing
        mask_cpu = torch.full((src_abs.numel(),), bool(self.config.allow_missing_geometry), dtype=torch.bool)
        src_vox = metadata.voxel_ids[b, h, src_cpu[has_geometry]].to(torch.int64)
        dst_vox = metadata.voxel_ids[b, h, dst_cpu[has_geometry]].to(torch.int64)
        dist = (src_vox - dst_vox).abs().max(dim=-1).values
        mask_cpu[has_geometry] = dist <= int(self.config.voxel_neighbor_radius)
        return mask_cpu.to(device=src_abs.device), missing

    def _apply_merges(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        b: int,
        h: int,
        src_abs: torch.Tensor,
        dst_abs: torch.Tensor,
    ) -> None:
        if src_abs.numel() == 0:
            return
        if self.config.use_depth_confidence and metadata is not None:
            src_cpu = src_abs.detach().cpu().long()
            dst_cpu = dst_abs.detach().cpu().long()
            old_conf_cpu = metadata.accumulated_confidence[b, h, dst_cpu]
            new_conf_cpu = metadata.accumulated_confidence[b, h, src_cpu]
            old_w = torch.nan_to_num(old_conf_cpu.to(device=k.device, dtype=k.dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            new_w = torch.nan_to_num(new_conf_cpu.to(device=k.device, dtype=k.dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            alpha = (old_w / (old_w + new_w + 1e-6)).view(-1, 1)
        else:
            src_cpu = dst_cpu = None
            old_conf_cpu = new_conf_cpu = None
            alpha = torch.full((src_abs.numel(), 1), float(self.config.ema_decay), device=k.device, dtype=k.dtype)
        merged_k = alpha * k[b, h, dst_abs] + (1.0 - alpha) * k[b, h, src_abs]
        alpha_v = alpha.to(dtype=v.dtype)
        merged_v = alpha_v * v[b, h, dst_abs] + (1.0 - alpha_v) * v[b, h, src_abs]
        k[b, h, dst_abs] = torch.nan_to_num(merged_k).to(dtype=k.dtype)
        v[b, h, dst_abs] = torch.nan_to_num(merged_v).to(dtype=v.dtype)
        if metadata is not None:
            if src_cpu is None:
                src_cpu = src_abs.detach().cpu().long()
                dst_cpu = dst_abs.detach().cpu().long()
            if old_conf_cpu is not None and new_conf_cpu is not None:
                metadata.accumulated_confidence[b, h, dst_cpu] = torch.nan_to_num(
                    old_conf_cpu + new_conf_cpu,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            metadata.merge_counts[b, h, dst_cpu] += 1
            metadata.last_updated_frame[b, h, dst_cpu] = metadata.frame_ids[b, h, src_cpu].to(dtype=metadata.last_updated_frame.dtype)

    def _apply_layer_merges(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Optional[KVCacheMetadata],
        b: int,
        src_abs: torch.Tensor,
        dst_abs: torch.Tensor,
    ) -> None:
        if src_abs.numel() == 0:
            return
        if self.config.use_depth_confidence and metadata is not None:
            src_cpu = src_abs.detach().cpu().long()
            dst_cpu = dst_abs.detach().cpu().long()
            old_conf_cpu = metadata.accumulated_confidence[b, :, dst_cpu]
            new_conf_cpu = metadata.accumulated_confidence[b, :, src_cpu]
            old_w = torch.nan_to_num(old_conf_cpu.to(device=k.device, dtype=k.dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            new_w = torch.nan_to_num(new_conf_cpu.to(device=k.device, dtype=k.dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            alpha = (old_w / (old_w + new_w + 1e-6)).unsqueeze(-1)
        else:
            src_cpu = dst_cpu = None
            old_conf_cpu = new_conf_cpu = None
            alpha = torch.full((1, src_abs.numel(), 1), float(self.config.ema_decay), device=k.device, dtype=k.dtype)
        merged_k = alpha * k[b, :, dst_abs] + (1.0 - alpha) * k[b, :, src_abs]
        alpha_v = alpha.to(dtype=v.dtype)
        merged_v = alpha_v * v[b, :, dst_abs] + (1.0 - alpha_v) * v[b, :, src_abs]
        k[b, :, dst_abs] = torch.nan_to_num(merged_k).to(dtype=k.dtype)
        v[b, :, dst_abs] = torch.nan_to_num(merged_v).to(dtype=v.dtype)
        if metadata is not None:
            if src_cpu is None:
                src_cpu = src_abs.detach().cpu().long()
                dst_cpu = dst_abs.detach().cpu().long()
            if old_conf_cpu is not None and new_conf_cpu is not None:
                metadata.accumulated_confidence[b, :, dst_cpu] = torch.nan_to_num(
                    old_conf_cpu + new_conf_cpu,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            metadata.merge_counts[b, :, dst_cpu] += 1
            metadata.last_updated_frame[b, :, dst_cpu] = metadata.frame_ids[b, :, src_cpu].to(dtype=metadata.last_updated_frame.dtype)

    def _profile_start(self, tensor: torch.Tensor) -> Optional[float]:
        if not self.config.profile:
            return None
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        return time.perf_counter()

    def _profile_add(self, stats: SvdEvictionMergeStats, name: str, start: Optional[float], tensor: torch.Tensor) -> None:
        if start is None or stats.profile_times is None:
            return
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        stats.profile_times[name] = stats.profile_times.get(name, 0.0) + (time.perf_counter() - start)
