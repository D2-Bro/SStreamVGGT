"""Sliding-window similarity merge for recent streaming KV cache entries."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from streamvggt.utils.geometry import closed_form_inverse_se3
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


_INVALID_VOXEL = -2**30
_CANDIDATE_MODES = {"full", "spatial", "voxel", "voxel_spatial"}


@dataclass
class RecentMergeConfig:
    """Runtime options for geometry-validated recent KV merging."""

    enabled: bool = False
    window: int = 3
    similarity_threshold: float = 0.9
    # 5cm is a conservative default for reconstructed indoor/desktop scale scenes:
    # it absorbs sub-patch depth noise while still requiring close 3D agreement.
    voxel_size: float = 0.05
    use_depth_confidence: bool = True
    debug: bool = False
    chunk_size: int = 512
    disable_geometry_check: bool = False
    candidate_mode: str = "full"
    patch_radius: int = 1
    voxel_neighbor_radius: int = 0
    max_candidates_per_token: int = 64
    local_fallback: bool = True
    profile: bool = False
    recall_debug: bool = False
    recall_debug_max_tokens: int = 1024


@dataclass
class KVCacheMetadata:
    """CPU-side token provenance aligned with a per-layer KV cache."""

    frame_ids: torch.Tensor
    token_indices: torch.Tensor
    accumulated_confidence: torch.Tensor
    merge_counts: torch.Tensor
    last_updated_frame: torch.Tensor

    @classmethod
    def for_current_frame(
        cls,
        batch_size: int,
        num_heads: int,
        num_tokens: int,
        frame_id: int,
    ) -> "KVCacheMetadata":
        shape = (batch_size, num_heads, num_tokens)
        token_indices = torch.arange(num_tokens, dtype=torch.int32).view(1, 1, num_tokens)
        return cls(
            frame_ids=torch.full(shape, int(frame_id), dtype=torch.long),
            token_indices=token_indices.expand(shape).clone(),
            accumulated_confidence=torch.ones(shape, dtype=torch.float32),
            merge_counts=torch.zeros(shape, dtype=torch.int16),
            last_updated_frame=torch.full(shape, int(frame_id), dtype=torch.int32),
        )

    def concat(self, other: "KVCacheMetadata") -> "KVCacheMetadata":
        return KVCacheMetadata(
            frame_ids=torch.cat([self.frame_ids, other.frame_ids], dim=2),
            token_indices=torch.cat([self.token_indices, other.token_indices], dim=2),
            accumulated_confidence=torch.cat(
                [self.accumulated_confidence, other.accumulated_confidence], dim=2
            ),
            merge_counts=torch.cat([self.merge_counts, other.merge_counts], dim=2),
            last_updated_frame=torch.cat([self.last_updated_frame, other.last_updated_frame], dim=2),
        )

    def gather(self, indices: torch.Tensor) -> "KVCacheMetadata":
        indices = indices.to(device="cpu", dtype=torch.long)
        return KVCacheMetadata(
            frame_ids=torch.gather(self.frame_ids, 2, indices),
            token_indices=torch.gather(self.token_indices, 2, indices),
            accumulated_confidence=torch.gather(self.accumulated_confidence, 2, indices),
            merge_counts=torch.gather(self.merge_counts, 2, indices),
            last_updated_frame=torch.gather(self.last_updated_frame, 2, indices),
        )

    def prune_after_eviction(self, kept_candidate_indices: torch.Tensor, num_anchor_tokens: int) -> "KVCacheMetadata":
        B, H, _ = self.frame_ids.shape
        anchor = torch.arange(num_anchor_tokens, dtype=torch.long).view(1, 1, num_anchor_tokens)
        anchor = anchor.expand(B, H, num_anchor_tokens)
        kept = kept_candidate_indices.detach().cpu().to(torch.long) + int(num_anchor_tokens)
        return self.gather(torch.cat([anchor, kept], dim=2))

    def update_frame_confidence(self, frame_id: int, token_confidence: torch.Tensor) -> None:
        token_confidence = token_confidence.detach().cpu().float()
        B, H, _ = self.frame_ids.shape
        max_tokens = token_confidence.shape[1]
        mask = self.frame_ids == int(frame_id)
        if not bool(mask.any()):
            return

        batch_ids = torch.arange(B, dtype=torch.long).view(B, 1, 1).expand(B, H, self.frame_ids.shape[2])
        token_ids = self.token_indices.long()
        valid = mask & (token_ids >= 0) & (token_ids < max_tokens)

        values = torch.ones_like(self.accumulated_confidence)
        values[valid] = token_confidence[batch_ids[valid], token_ids[valid]]
        self.accumulated_confidence[mask] = values[mask]


@dataclass
class FrameGeometry:
    """Recent per-frame patch geometry used only for merge validation."""

    voxel_ids: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    patch_height: int
    patch_width: int
    patch_tokens: int
    tokens_per_frame: int


@dataclass
class RecentMergeStats:
    current_tokens: int = 0
    candidate_tokens: int = 0
    threshold_pairs: int = 0
    voxel_pairs: int = 0
    local_candidate_pairs: int = 0
    max_candidates_per_src: int = 0
    attempted_matches: int = 0
    rejected_similarity: int = 0
    rejected_geometry: int = 0
    fallback_count: int = 0
    merged_tokens: int = 0
    similarities: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    frame_gaps: list[int] = field(default_factory=list)
    profile_times: Dict[str, float] = field(default_factory=dict)
    recall_samples: int = 0
    recall_hits: int = 0
    recall_full_similarity_sum: float = 0.0
    recall_local_similarity_sum: float = 0.0
    cache_before: int = 0
    cache_after: int = 0


class RecentSimilarityMerge:
    """Geometry-validated greedy merge over only the recent frame window.

    Matching is deterministic and conservative: each layer/head chooses the best
    previous-window candidate for each current patch token, then a current token
    is physically removed only if every batch/head has a valid match for that
    token index. This keeps the dense KV tensor shape consistent across heads.
    """

    def __init__(self, config: RecentMergeConfig, patch_start_idx: int, patch_size: int) -> None:
        if config.candidate_mode not in _CANDIDATE_MODES:
            raise ValueError(
                f"candidate_mode must be one of {sorted(_CANDIDATE_MODES)}, got {config.candidate_mode!r}"
            )
        self.config = config
        self.patch_start_idx = int(patch_start_idx)
        self.patch_size = int(patch_size)
        self._geometry: Dict[int, FrameGeometry] = {}
        self._warned_missing_confidence = False

    def record_frame_geometry(
        self,
        frame_id: int,
        depth: torch.Tensor,
        depth_conf: Optional[torch.Tensor],
        pose_enc: torch.Tensor,
        image_hw: Tuple[int, int],
        tokens_per_frame: int,
    ) -> Optional[FrameGeometry]:
        """Compute transient patch-token voxels for one completed frame."""
        if not self.config.enabled:
            return None
        if depth is None or pose_enc is None or image_hw is None:
            if self.config.debug:
                print(f"[RecentSimilarityMerge] frame={frame_id} missing geometry; merge disabled for this frame")
            return None

        H_img, W_img = int(image_hw[0]), int(image_hw[1])
        depth = depth.detach()
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3:
            if self.config.debug:
                print(f"[RecentSimilarityMerge] frame={frame_id} unexpected depth shape {tuple(depth.shape)}")
            return None

        B = depth.shape[0]
        if depth_conf is None:
            if self.config.use_depth_confidence and not self._warned_missing_confidence and self.config.debug:
                print("[RecentSimilarityMerge] depth_conf missing; falling back to confidence=1.0")
                self._warned_missing_confidence = True
            depth_conf = torch.ones_like(depth)
        else:
            depth_conf = depth_conf.detach()
            if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
                depth_conf = depth_conf[..., 0]

        intrinsic = None
        cam_to_world = None
        if not self.config.disable_geometry_check:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.unsqueeze(1), (H_img, W_img))
            if intrinsic is None:
                if self.config.debug:
                    print(f"[RecentSimilarityMerge] frame={frame_id} missing intrinsics; merge disabled")
                return None

            extrinsic = extrinsic[:, 0]
            intrinsic = intrinsic[:, 0]
            extrinsic_h = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device).repeat(B, 1, 1)
            extrinsic_h[:, :3, :4] = extrinsic
            cam_to_world = closed_form_inverse_se3(extrinsic_h)

        patch_h = H_img // self.patch_size
        patch_w = W_img // self.patch_size
        patch_tokens = max(int(tokens_per_frame) - self.patch_start_idx, 0)
        expected_patch_tokens = patch_h * patch_w
        patch_tokens = min(patch_tokens, expected_patch_tokens)

        voxel_ids = torch.full((B, tokens_per_frame, 3), _INVALID_VOXEL, dtype=torch.int32)
        confidence = torch.zeros((B, tokens_per_frame), dtype=torch.float32)
        valid = torch.zeros((B, tokens_per_frame), dtype=torch.bool)
        if patch_tokens <= 0:
            self._geometry[int(frame_id)] = FrameGeometry(
                voxel_ids,
                confidence,
                valid,
                patch_height=patch_h,
                patch_width=patch_w,
                patch_tokens=patch_tokens,
                tokens_per_frame=int(tokens_per_frame),
            )
            return self._geometry[int(frame_id)]

        patch_ids = torch.arange(patch_tokens, device=depth.device)
        rows = torch.div(patch_ids, patch_w, rounding_mode="floor")
        cols = patch_ids % patch_w
        ys = torch.clamp((rows.float() + 0.5) * self.patch_size, 0, H_img - 1).round().long()
        xs = torch.clamp((cols.float() + 0.5) * self.patch_size, 0, W_img - 1).round().long()

        sampled_depth = depth[:, ys, xs].float()
        sampled_conf = depth_conf[:, ys, xs].float()
        finite = torch.isfinite(sampled_depth) & (sampled_depth > 1e-8)
        if self.config.use_depth_confidence:
            finite = finite & torch.isfinite(sampled_conf) & (sampled_conf > 0)

        ones = torch.ones_like(sampled_depth)
        pix = torch.stack(
            [
                xs.float().view(1, -1).expand(B, -1),
                ys.float().view(1, -1).expand(B, -1),
                ones,
            ],
            dim=-1,
        )
        if self.config.disable_geometry_check:
            patch_voxels = torch.zeros((B, patch_tokens, 3), device=depth.device, dtype=torch.int32)
        else:
            inv_intrinsic = torch.linalg.inv(intrinsic.float())
            rays = torch.bmm(pix.float(), inv_intrinsic.transpose(1, 2))
            cam_xyz = rays * sampled_depth.unsqueeze(-1)
            world_xyz = torch.bmm(cam_xyz, cam_to_world[:, :3, :3].float().transpose(1, 2))
            world_xyz = world_xyz + cam_to_world[:, :3, 3].float().unsqueeze(1)
            patch_voxels = torch.floor(world_xyz / float(self.config.voxel_size)).to(torch.int32)
        token_slice = slice(self.patch_start_idx, self.patch_start_idx + patch_tokens)
        voxel_ids[:, token_slice] = patch_voxels.detach().cpu()
        confidence[:, token_slice] = sampled_conf.detach().cpu().float().clamp_min(0)
        valid[:, token_slice] = finite.detach().cpu()

        geom = FrameGeometry(
            voxel_ids=voxel_ids,
            confidence=confidence,
            valid=valid,
            patch_height=patch_h,
            patch_width=patch_w,
            patch_tokens=patch_tokens,
            tokens_per_frame=int(tokens_per_frame),
        )
        self._geometry[int(frame_id)] = geom
        min_frame = int(frame_id) - int(self.config.window)
        for old_frame in list(self._geometry):
            if old_frame < min_frame:
                del self._geometry[old_frame]
        return geom

    def update_metadata_for_frame(self, metadata: KVCacheMetadata, frame_id: int) -> None:
        geom = self._geometry.get(int(frame_id))
        if geom is not None:
            metadata.update_frame_confidence(frame_id, geom.confidence)

    def merge_layer(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: KVCacheMetadata,
        layer_id: int,
        frame_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCacheMetadata, RecentMergeStats]:
        """Merge current-frame tokens into recent-window candidates for one layer."""
        stats = RecentMergeStats(cache_before=int(k.shape[2]))
        total_start = self._profile_start(k)
        if not self.config.enabled or int(frame_id) <= 0:
            stats.cache_after = int(k.shape[2])
            return k, v, metadata, stats
        if int(frame_id) not in self._geometry:
            if self.config.debug:
                print(f"[RecentSimilarityMerge] layer={layer_id} frame={frame_id} no geometry; merge skipped")
            stats.cache_after = int(k.shape[2])
            return k, v, metadata, stats

        B, H, N, D = k.shape
        per_head_matches: Dict[Tuple[int, int], Dict[int, Tuple[int, int, float, float]]] = {}
        matched_token_sets = []

        for b in range(B):
            for h in range(H):
                matches, head_stats = self._match_one_head(k, metadata, b, h, frame_id)
                per_head_matches[(b, h)] = matches
                matched_token_sets.append(set(matches.keys()))
                stats.current_tokens += head_stats.current_tokens
                stats.candidate_tokens += head_stats.candidate_tokens
                stats.threshold_pairs += head_stats.threshold_pairs
                stats.voxel_pairs += head_stats.voxel_pairs
                stats.local_candidate_pairs += head_stats.local_candidate_pairs
                stats.max_candidates_per_src = max(
                    stats.max_candidates_per_src, head_stats.max_candidates_per_src
                )
                stats.attempted_matches += head_stats.attempted_matches
                stats.rejected_similarity += head_stats.rejected_similarity
                stats.rejected_geometry += head_stats.rejected_geometry
                stats.fallback_count += head_stats.fallback_count
                stats.recall_samples += head_stats.recall_samples
                stats.recall_hits += head_stats.recall_hits
                stats.recall_full_similarity_sum += head_stats.recall_full_similarity_sum
                stats.recall_local_similarity_sum += head_stats.recall_local_similarity_sum
                for name, value in head_stats.profile_times.items():
                    stats.profile_times[name] = stats.profile_times.get(name, 0.0) + value

        if not matched_token_sets:
            stats.cache_after = int(k.shape[2])
            return k, v, metadata, stats

        common_token_ids = sorted(set.intersection(*matched_token_sets)) if matched_token_sets else []
        if not common_token_ids:
            stats.cache_after = int(k.shape[2])
            self._profile_add(stats, "total", total_start, k)
            self._debug(layer_id, frame_id, stats)
            return k, v, metadata, stats

        ema_start = self._profile_start(k)
        with torch.no_grad():
            for b in range(B):
                for h in range(H):
                    matches = per_head_matches[(b, h)]
                    rows = [matches[token_id] for token_id in common_token_ids]
                    cur_pos = torch.tensor([row[0] for row in rows], device=k.device, dtype=torch.long)
                    cand_pos = torch.tensor([row[1] for row in rows], device=k.device, dtype=torch.long)
                    new_conf = torch.tensor(
                        [row[3] for row in rows],
                        device=metadata.accumulated_confidence.device,
                        dtype=metadata.accumulated_confidence.dtype,
                    )
                    cand_pos_cpu = cand_pos.cpu()
                    old_conf = metadata.accumulated_confidence[b, h, cand_pos_cpu]
                    if self.config.use_depth_confidence:
                        old_w = torch.nan_to_num(
                            old_conf.to(device=k.device, dtype=k.dtype), nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp_min(1e-6)
                        new_w = torch.nan_to_num(
                            new_conf.to(device=k.device, dtype=k.dtype), nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp_min(1e-6)
                    else:
                        old_w = torch.ones_like(old_conf, device=k.device, dtype=k.dtype)
                        new_w = torch.ones_like(old_conf, device=k.device, dtype=k.dtype)
                    alpha = (old_w / (old_w + new_w + 1e-6)).view(-1, 1).to(dtype=k.dtype)
                    one = torch.ones_like(alpha)
                    merged_k = alpha * k[b, h, cand_pos] + (one - alpha) * k[b, h, cur_pos]
                    alpha_v = alpha.to(dtype=v.dtype)
                    one_v = one.to(dtype=v.dtype)
                    merged_v = alpha_v * v[b, h, cand_pos] + (one_v - alpha_v) * v[b, h, cur_pos]

                    k[b, h, cand_pos] = torch.nan_to_num(merged_k).to(dtype=k.dtype)
                    v[b, h, cand_pos] = torch.nan_to_num(merged_v).to(dtype=v.dtype)
                    metadata.accumulated_confidence[b, h, cand_pos_cpu] = torch.nan_to_num(
                        old_conf + new_conf.cpu(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    metadata.merge_counts[b, h, cand_pos_cpu] += 1
                    metadata.last_updated_frame[b, h, cand_pos_cpu] = int(frame_id)

                    if self.config.debug:
                        stats.similarities.extend(float(row[2]) for row in rows)
                        stats.confidences.extend(float(row[3]) for row in rows)
                        cand_frames = metadata.frame_ids[b, h, cand_pos_cpu]
                        stats.frame_gaps.extend((int(frame_id) - cand_frames).tolist())
        self._profile_add(stats, "ema_update", ema_start, k)

        keep_indices_cpu = self._build_keep_indices(metadata, frame_id, common_token_ids)
        keep_indices = keep_indices_cpu.to(device=k.device, dtype=torch.long)
        k = torch.gather(k, 2, keep_indices.unsqueeze(-1).expand(B, H, keep_indices.shape[2], D))
        v = torch.gather(v, 2, keep_indices.unsqueeze(-1).expand(B, H, keep_indices.shape[2], D))
        metadata = metadata.gather(keep_indices_cpu)
        stats.merged_tokens = len(common_token_ids)
        stats.cache_after = int(k.shape[2])
        self._profile_add(stats, "total", total_start, k)
        self._debug(layer_id, frame_id, stats)
        return k, v, metadata, stats

    def _match_one_head(
        self,
        k: torch.Tensor,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        frame_id: int,
    ) -> Tuple[Dict[int, Tuple[int, int, float, float]], RecentMergeStats]:
        if self.config.candidate_mode == "full":
            return self._match_one_head_full(k, metadata, b, h, frame_id)
        return self._match_one_head_local(k, metadata, b, h, frame_id)

    def _match_one_head_full(
        self,
        k: torch.Tensor,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        frame_id: int,
    ) -> Tuple[Dict[int, Tuple[int, int, float, float]], RecentMergeStats]:
        stats = RecentMergeStats()
        candidate_start = self._profile_start(k)
        frame_ids = metadata.frame_ids[b, h]
        token_ids = metadata.token_indices[b, h]
        current_mask = (frame_ids == int(frame_id)) & (token_ids >= self.patch_start_idx)
        min_frame = int(frame_id) - int(self.config.window)
        candidate_mask = (
            (frame_ids >= min_frame)
            & (frame_ids < int(frame_id))
            & (token_ids >= self.patch_start_idx)
        )

        cur_pos = torch.nonzero(current_mask, as_tuple=False).flatten()
        cand_pos = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        stats.current_tokens = int(cur_pos.numel())
        stats.candidate_tokens = int(cand_pos.numel())
        if cur_pos.numel() == 0 or cand_pos.numel() == 0:
            return {}, stats

        cur_vox, cur_valid, cur_conf = self._lookup_geometry(metadata, b, h, cur_pos)
        cand_vox, cand_valid, _ = self._lookup_geometry(metadata, b, h, cand_pos)
        valid_cur = cur_valid
        valid_cand = cand_valid
        if not bool(valid_cur.any()) or not bool(valid_cand.any()):
            return {}, stats

        cur_pos_cpu = cur_pos[valid_cur]
        cand_pos_cpu = cand_pos[valid_cand]
        cur_vox = cur_vox[valid_cur].to(device=k.device)
        cand_vox = cand_vox[valid_cand].to(device=k.device)
        cur_conf_cpu = cur_conf[valid_cur]
        if cur_pos_cpu.numel() == 0 or cand_pos_cpu.numel() == 0:
            return {}, stats
        self._profile_add(stats, "candidate_build", candidate_start, k)

        cur_pos = cur_pos_cpu.to(device=k.device)
        cand_pos = cand_pos_cpu.to(device=k.device)
        cur_k = F.normalize(k[b, h, cur_pos].float(), dim=-1)
        cand_k = F.normalize(k[b, h, cand_pos].float(), dim=-1)
        stats.local_candidate_pairs = int(cur_k.shape[0] * cand_k.shape[0])
        stats.max_candidates_per_src = int(cand_k.shape[0])
        if self.config.disable_geometry_check:
            same_voxel = torch.ones(
                (cur_vox.shape[0], cand_vox.shape[0]),
                dtype=torch.bool,
                device=k.device,
            )
        else:
            same_voxel = (cur_vox[:, None, :] == cand_vox[None, :, :]).all(dim=-1)

        best: Dict[int, Tuple[int, int, float, float]] = {}
        used_candidates: set[int] = set()
        best_rows = []
        chunk = max(int(self.config.chunk_size), 1)
        cosine_start = self._profile_start(k)
        stats.attempted_matches = int(cur_k.shape[0] * cand_k.shape[0])
        if not self.config.disable_geometry_check:
            stats.rejected_geometry = int((~same_voxel).sum().item())
        for start in range(0, cur_k.shape[0], chunk):
            end = min(start + chunk, cur_k.shape[0])
            sim = cur_k[start:end] @ cand_k.T
            threshold_mask = sim >= float(self.config.similarity_threshold)
            voxel_mask = same_voxel[start:end]
            valid_mask = threshold_mask & voxel_mask
            if self.config.debug:
                stats.threshold_pairs += int(threshold_mask.sum().item())
                stats.voxel_pairs += int(valid_mask.sum().item())
            stats.rejected_similarity += int((~threshold_mask & voxel_mask).sum().item())
            masked = sim.masked_fill(~valid_mask, -float("inf"))
            values, indices = masked.max(dim=1)
            finite = torch.isfinite(values)
            valid_rows = torch.nonzero(finite, as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                continue

            global_rows = valid_rows.detach().cpu() + int(start)
            cand_cols = indices[finite].detach().cpu().long()
            scores = values[finite].detach().cpu()
            cur_cache_pos = cur_pos_cpu[global_rows].long()
            cand_cache_pos = cand_pos_cpu[cand_cols].long()
            token_ids = metadata.token_indices[b, h, cur_cache_pos].long()
            confidences = cur_conf_cpu[global_rows]
            best_rows.extend(
                zip(
                    scores.tolist(),
                    cur_cache_pos.tolist(),
                    cand_cache_pos.tolist(),
                    token_ids.tolist(),
                    confidences.tolist(),
                )
            )
        self._profile_add(stats, "cosine_match", cosine_start, k)

        best_rows.sort(key=lambda row: (-row[0], row[3], row[2]))
        for score, cur_cache_pos, cand_cache_pos, token_id, conf in best_rows:
            if token_id in best or cand_cache_pos in used_candidates:
                continue
            best[token_id] = (cur_cache_pos, cand_cache_pos, score, conf)
            used_candidates.add(cand_cache_pos)
        return best, stats

    def _match_one_head_local(
        self,
        k: torch.Tensor,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        frame_id: int,
    ) -> Tuple[Dict[int, Tuple[int, int, float, float]], RecentMergeStats]:
        stats = RecentMergeStats()
        candidate_start = self._profile_start(k)
        frame_ids = metadata.frame_ids[b, h]
        token_ids = metadata.token_indices[b, h]
        current_mask = (frame_ids == int(frame_id)) & (token_ids >= self.patch_start_idx)
        min_frame = int(frame_id) - int(self.config.window)
        candidate_mask = (
            (frame_ids >= min_frame)
            & (frame_ids < int(frame_id))
            & (token_ids >= self.patch_start_idx)
        )

        cur_pos_cpu = torch.nonzero(current_mask, as_tuple=False).flatten()
        cand_pos_cpu = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        stats.current_tokens = int(cur_pos_cpu.numel())
        stats.candidate_tokens = int(cand_pos_cpu.numel())
        if cur_pos_cpu.numel() == 0 or cand_pos_cpu.numel() == 0:
            return {}, stats

        cur_vox, cur_valid, cur_conf_cpu = self._lookup_geometry(metadata, b, h, cur_pos_cpu)
        cand_vox, cand_valid, cand_conf = self._lookup_geometry(metadata, b, h, cand_pos_cpu)
        use_spatial_fast_path = self.config.candidate_mode == "spatial" and not self.config.recall_debug
        if use_spatial_fast_path:
            (
                local_cols_cpu,
                local_valid_cpu,
                local_candidate_pairs,
                max_candidates_per_src,
                rejected_geometry,
            ) = self._build_spatial_candidate_lists(
                metadata,
                b,
                h,
                cur_pos_cpu,
                cand_pos_cpu,
                cur_vox,
                cand_vox,
                cur_valid,
                cand_valid,
                cand_conf,
                frame_id,
            )
            stats.fallback_count = 0
            stats.local_candidate_pairs = int(local_candidate_pairs)
            stats.max_candidates_per_src = int(max_candidates_per_src)
            stats.rejected_geometry += int(rejected_geometry)
            stats.attempted_matches = int(local_valid_cpu.sum().item())
        else:
            candidate_mask_cpu, geometry_mask_cpu, fallback_count = self._build_local_candidate_masks(
                metadata,
                b,
                h,
                cur_pos_cpu,
                cand_pos_cpu,
                cur_vox,
                cand_vox,
                cur_valid,
                cand_valid,
                cand_conf,
            )
            stats.fallback_count = int(fallback_count)
            stats.local_candidate_pairs = int(candidate_mask_cpu.sum().item())
            stats.max_candidates_per_src = (
                int(candidate_mask_cpu.sum(dim=1).max().item()) if candidate_mask_cpu.numel() else 0
            )
            pair_mask_cpu = candidate_mask_cpu & geometry_mask_cpu
            if not self.config.disable_geometry_check:
                stats.rejected_geometry += int((candidate_mask_cpu & ~geometry_mask_cpu).sum().item())
            stats.attempted_matches = int(pair_mask_cpu.sum().item())
            pair_counts = pair_mask_cpu.sum(dim=1)
            max_pairs = int(pair_counts.max().item()) if pair_counts.numel() else 0
            local_cols_cpu = torch.zeros((pair_mask_cpu.shape[0], max_pairs), dtype=torch.long)
            local_valid_cpu = torch.zeros((pair_mask_cpu.shape[0], max_pairs), dtype=torch.bool)
            for row in range(pair_mask_cpu.shape[0]):
                cols = torch.nonzero(pair_mask_cpu[row], as_tuple=False).flatten()
                if cols.numel() == 0:
                    continue
                local_cols_cpu[row, : cols.numel()] = cols
                local_valid_cpu[row, : cols.numel()] = True

        if stats.local_candidate_pairs == 0:
            self._profile_add(stats, "candidate_build", candidate_start, k)
            return {}, stats
        self._profile_add(stats, "candidate_build", candidate_start, k)

        cur_pos = cur_pos_cpu.to(device=k.device)
        cand_pos = cand_pos_cpu.to(device=k.device)
        cur_k = F.normalize(k[b, h, cur_pos].float(), dim=-1)
        cand_k = F.normalize(k[b, h, cand_pos].float(), dim=-1)

        if self.config.recall_debug:
            candidate_mask_gpu = candidate_mask_cpu.to(device=k.device)
            geometry_mask_gpu = geometry_mask_cpu.to(device=k.device)
            self._accumulate_recall_debug(
                stats,
                cur_k,
                cand_k,
                candidate_mask_gpu,
                geometry_mask_gpu,
                cur_vox,
                cand_vox,
                cur_valid,
                cand_valid,
            )

        best: Dict[int, Tuple[int, int, float, float]] = {}
        used_candidates: set[int] = set()
        best_rows = []
        cosine_start = self._profile_start(k)
        threshold = float(self.config.similarity_threshold)
        if stats.attempted_matches == 0:
            self._profile_add(stats, "cosine_match", cosine_start, k)
            return {}, stats

        local_cols = local_cols_cpu.to(device=k.device)
        local_valid = local_valid_cpu.to(device=k.device)
        chunk = max(int(self.config.chunk_size), 1)
        for start in range(0, cur_k.shape[0], chunk):
            end = min(start + chunk, cur_k.shape[0])
            chunk_cols = local_cols[start:end]
            valid_pairs = local_valid[start:end]
            local_cand = cand_k[chunk_cols.reshape(-1)].view(
                chunk_cols.shape[0],
                chunk_cols.shape[1],
                cand_k.shape[-1],
            )
            sim = (cur_k[start:end].unsqueeze(1) * local_cand).sum(dim=-1)
            threshold_mask = sim >= threshold
            valid_mask = valid_pairs & threshold_mask
            stats.threshold_pairs += int(valid_mask.sum().item())
            stats.rejected_similarity += int((valid_pairs & ~threshold_mask).sum().item())
            masked = sim.masked_fill(~valid_mask, -float("inf"))
            values, indices = masked.max(dim=1)
            finite = torch.isfinite(values)
            valid_rows = torch.nonzero(finite, as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                continue

            global_rows = valid_rows.detach().cpu() + int(start)
            cand_cols = chunk_cols[finite, indices[finite]].detach().cpu().long()
            scores = values[finite].detach().cpu()
            cur_cache_pos = cur_pos_cpu[global_rows].long()
            cand_cache_pos = cand_pos_cpu[cand_cols].long()
            token_ids = metadata.token_indices[b, h, cur_cache_pos].long()
            confidences = cur_conf_cpu[global_rows]
            best_rows.extend(
                zip(
                    scores.tolist(),
                    cur_cache_pos.tolist(),
                    cand_cache_pos.tolist(),
                    token_ids.tolist(),
                    confidences.tolist(),
                )
            )
        self._profile_add(stats, "cosine_match", cosine_start, k)

        stats.voxel_pairs = len(best_rows)
        best_rows.sort(key=lambda row: (-row[0], row[3], row[2]))
        for score, cur_cache_pos, cand_cache_pos, token_id, conf in best_rows:
            if token_id in best or cand_cache_pos in used_candidates:
                continue
            best[token_id] = (cur_cache_pos, cand_cache_pos, score, conf)
            used_candidates.add(cand_cache_pos)
        return best, stats

    def _build_spatial_candidate_lists(
        self,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        cur_pos: torch.Tensor,
        cand_pos: torch.Tensor,
        cur_vox: torch.Tensor,
        cand_vox: torch.Tensor,
        cur_valid: torch.Tensor,
        cand_valid: torch.Tensor,
        cand_conf: torch.Tensor,
        frame_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        geom = self._geometry.get(int(frame_id))
        if geom is None or geom.patch_width <= 0 or geom.patch_tokens <= 0:
            empty_cols = torch.zeros((cur_pos.numel(), 0), dtype=torch.long)
            empty_valid = torch.zeros((cur_pos.numel(), 0), dtype=torch.bool)
            return empty_cols, empty_valid, 0, 0, 0

        window = max(int(self.config.window), 1)
        min_frame = int(frame_id) - window
        tokens_per_frame = int(geom.tokens_per_frame)
        lookup = torch.full((window, tokens_per_frame), -1, dtype=torch.long)

        cand_frame_ids = metadata.frame_ids[b, h, cand_pos].long()
        cand_token_ids = metadata.token_indices[b, h, cand_pos].long()
        cand_slots = cand_frame_ids - int(min_frame)
        cand_cols = torch.arange(cand_pos.numel(), dtype=torch.long)
        cand_lookup_valid = (
            (cand_slots >= 0)
            & (cand_slots < window)
            & (cand_token_ids >= 0)
            & (cand_token_ids < tokens_per_frame)
        )
        if bool(cand_lookup_valid.any()):
            lookup[cand_slots[cand_lookup_valid], cand_token_ids[cand_lookup_valid]] = cand_cols[cand_lookup_valid]

        cur_token_ids = metadata.token_indices[b, h, cur_pos].long()
        cur_patch_ids = cur_token_ids - int(self.patch_start_idx)
        cur_patch_valid = (cur_patch_ids >= 0) & (cur_patch_ids < int(geom.patch_tokens))
        cur_rows = torch.div(cur_patch_ids.clamp_min(0), int(geom.patch_width), rounding_mode="floor")
        cur_cols = cur_patch_ids.clamp_min(0) % int(geom.patch_width)

        patch_radius = max(int(self.config.patch_radius), 0)
        offsets = torch.arange(-patch_radius, patch_radius + 1, dtype=torch.long)
        dr, dc = torch.meshgrid(offsets, offsets, indexing="ij")
        dr = dr.reshape(-1)
        dc = dc.reshape(-1)
        neigh_rows = cur_rows[:, None] + dr[None, :]
        neigh_cols = cur_cols[:, None] + dc[None, :]
        in_bounds = (
            cur_patch_valid[:, None]
            & (neigh_rows >= 0)
            & (neigh_rows < int(geom.patch_height))
            & (neigh_cols >= 0)
            & (neigh_cols < int(geom.patch_width))
        )
        neigh_patch_ids = neigh_rows.clamp(0, max(int(geom.patch_height) - 1, 0)) * int(geom.patch_width)
        neigh_patch_ids = neigh_patch_ids + neigh_cols.clamp(0, max(int(geom.patch_width) - 1, 0))
        neigh_token_ids = neigh_patch_ids + int(self.patch_start_idx)
        neigh_token_ids = torch.where(in_bounds, neigh_token_ids, torch.zeros_like(neigh_token_ids))

        local_cols = lookup[:, neigh_token_ids.reshape(-1)].view(window, cur_pos.numel(), -1)
        local_cols = local_cols.permute(1, 0, 2).reshape(cur_pos.numel(), -1)
        local_pre_valid = in_bounds[:, None, :].expand(cur_pos.numel(), window, in_bounds.shape[1])
        local_pre_valid = local_pre_valid.reshape(cur_pos.numel(), -1) & (local_cols >= 0)
        local_candidate_pairs = int(local_pre_valid.sum().item())
        max_candidates_per_src = int(local_pre_valid.sum(dim=1).max().item()) if local_pre_valid.numel() else 0
        if local_candidate_pairs == 0:
            return local_cols.clamp_min(0), local_pre_valid, 0, 0, 0

        local_cols = local_cols.clamp_min(0)
        max_candidates = max(int(self.config.max_candidates_per_token), 1)
        if local_cols.shape[1] > max_candidates and max_candidates_per_src > max_candidates:
            patch_distance = (dr.abs() + dc.abs()).repeat(window)
            local_frame_ids = cand_frame_ids[local_cols]
            local_conf = cand_conf[local_cols]
            bounded_valid = torch.zeros_like(local_pre_valid)
            for row in range(local_pre_valid.shape[0]):
                cols = torch.nonzero(local_pre_valid[row], as_tuple=False).flatten()
                if cols.numel() == 0:
                    continue
                if cols.numel() > max_candidates:
                    ordered = sorted(
                        cols.tolist(),
                        key=lambda col: (
                            int(patch_distance[col].item()),
                            -int(local_frame_ids[row, col].item()),
                            -float(local_conf[row, col].item()),
                            int(local_cols[row, col].item()),
                        ),
                    )
                    cols = torch.tensor(ordered[:max_candidates], dtype=torch.long)
                bounded_valid[row, cols] = True
            local_pre_valid = bounded_valid
            local_candidate_pairs = int(local_pre_valid.sum().item())
            max_candidates_per_src = (
                int(local_pre_valid.sum(dim=1).max().item()) if local_pre_valid.numel() else 0
            )

        if self.config.disable_geometry_check:
            local_valid = local_pre_valid
        else:
            gathered_cand_vox = cand_vox[local_cols.reshape(-1)].view(local_cols.shape[0], local_cols.shape[1], 3)
            gathered_cand_valid = cand_valid[local_cols.reshape(-1)].view(local_cols.shape[0], local_cols.shape[1])
            same_voxel = (cur_vox[:, None, :] == gathered_cand_vox).all(dim=-1)
            local_valid = local_pre_valid & cur_valid[:, None] & gathered_cand_valid & same_voxel

        rejected_geometry = local_candidate_pairs - int(local_valid.sum().item())
        return local_cols, local_valid, local_candidate_pairs, max_candidates_per_src, rejected_geometry

    def _build_local_candidate_masks(
        self,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        cur_pos: torch.Tensor,
        cand_pos: torch.Tensor,
        cur_vox: torch.Tensor,
        cand_vox: torch.Tensor,
        cur_valid: torch.Tensor,
        cand_valid: torch.Tensor,
        cand_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        cur_rows, cur_cols, cur_patch_valid = self._lookup_patch_coords(metadata, b, h, cur_pos)
        cand_rows, cand_cols, cand_patch_valid = self._lookup_patch_coords(metadata, b, h, cand_pos)
        if cur_pos.numel() == 0 or cand_pos.numel() == 0:
            empty = torch.zeros((cur_pos.numel(), cand_pos.numel()), dtype=torch.bool)
            return empty, empty, 0

        patch_radius = max(int(self.config.patch_radius), 0)
        row_dist = (cur_rows[:, None] - cand_rows[None, :]).abs()
        col_dist = (cur_cols[:, None] - cand_cols[None, :]).abs()
        patch_distance = row_dist + col_dist
        spatial_mask = (
            cur_patch_valid[:, None]
            & cand_patch_valid[None, :]
            & (row_dist <= patch_radius)
            & (col_dist <= patch_radius)
        )

        cur_vox_i64 = cur_vox.to(torch.int64)
        cand_vox_i64 = cand_vox.to(torch.int64)
        voxel_distance = (cur_vox_i64[:, None, :] - cand_vox_i64[None, :, :]).abs().max(dim=-1).values
        same_voxel = voxel_distance == 0
        voxel_radius = max(int(self.config.voxel_neighbor_radius), 0)
        voxel_mask = cur_valid[:, None] & cand_valid[None, :] & (voxel_distance <= voxel_radius)
        exact_voxel_mask = cur_valid[:, None] & cand_valid[None, :] & same_voxel

        mode = self.config.candidate_mode
        fallback_count = 0
        if mode == "spatial":
            local_mask = spatial_mask
        elif mode == "voxel":
            local_mask = voxel_mask.clone()
            if self.config.local_fallback:
                empty_rows = ~local_mask.any(dim=1)
                fallback_rows = empty_rows & spatial_mask.any(dim=1)
                local_mask[fallback_rows] = spatial_mask[fallback_rows]
                fallback_count = int(fallback_rows.sum().item())
        elif mode == "voxel_spatial":
            intersection = voxel_mask & spatial_mask
            local_mask = intersection.clone()
            if self.config.local_fallback:
                empty_rows = ~local_mask.any(dim=1)
                voxel_rows = empty_rows & voxel_mask.any(dim=1)
                local_mask[voxel_rows] = voxel_mask[voxel_rows]
                empty_rows = ~local_mask.any(dim=1)
                spatial_rows = empty_rows & spatial_mask.any(dim=1)
                local_mask[spatial_rows] = spatial_mask[spatial_rows]
                fallback_count = int(voxel_rows.sum().item() + spatial_rows.sum().item())
        else:
            local_mask = torch.ones((cur_pos.numel(), cand_pos.numel()), dtype=torch.bool)

        if self.config.disable_geometry_check:
            geometry_mask = torch.ones_like(local_mask)
        elif mode in {"voxel", "voxel_spatial"}:
            geometry_mask = voxel_mask
        else:
            geometry_mask = exact_voxel_mask

        bounded = self._bound_candidate_mask(
            local_mask,
            same_voxel=same_voxel,
            patch_distance=patch_distance,
            candidate_frame_ids=metadata.frame_ids[b, h, cand_pos].long(),
            candidate_confidence=cand_conf,
        )
        return bounded, geometry_mask & bounded, fallback_count

    def _bound_candidate_mask(
        self,
        candidate_mask: torch.Tensor,
        same_voxel: torch.Tensor,
        patch_distance: torch.Tensor,
        candidate_frame_ids: torch.Tensor,
        candidate_confidence: torch.Tensor,
    ) -> torch.Tensor:
        max_candidates = max(int(self.config.max_candidates_per_token), 1)
        if candidate_mask.numel() == 0:
            return candidate_mask
        bounded = torch.zeros_like(candidate_mask)
        voxel_mode_active = self.config.candidate_mode in {"voxel", "voxel_spatial"}
        for row in range(candidate_mask.shape[0]):
            cols = torch.nonzero(candidate_mask[row], as_tuple=False).flatten()
            if cols.numel() == 0:
                continue
            if cols.numel() > max_candidates:
                ordered = sorted(
                    cols.tolist(),
                    key=lambda col: (
                        0 if (voxel_mode_active and bool(same_voxel[row, col])) else 1,
                        int(patch_distance[row, col].item()),
                        -int(candidate_frame_ids[col].item()),
                        -float(candidate_confidence[col].item()),
                        int(col),
                    ),
                )
                cols = torch.tensor(ordered[:max_candidates], dtype=torch.long)
            bounded[row, cols] = True
        return bounded

    def _lookup_geometry(
        self,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_ids = metadata.frame_ids[b, h, positions].long()
        token_ids = metadata.token_indices[b, h, positions].long()
        voxels = torch.full((positions.numel(), 3), _INVALID_VOXEL, dtype=torch.int32)
        valid = torch.zeros((positions.numel(),), dtype=torch.bool)
        confidence = torch.zeros((positions.numel(),), dtype=torch.float32)
        for frame_id in torch.unique(frame_ids).tolist():
            geom = self._geometry.get(int(frame_id))
            if geom is None:
                continue
            if b >= geom.valid.shape[0]:
                continue
            frame_mask = frame_ids == int(frame_id)
            frame_token_ids = token_ids[frame_mask]
            frame_valid = (frame_token_ids >= 0) & (frame_token_ids < geom.valid.shape[1])
            if not bool(frame_valid.any()):
                continue
            rows = torch.nonzero(frame_mask, as_tuple=False).flatten()[frame_valid]
            token_rows = frame_token_ids[frame_valid]
            valid[rows] = geom.valid[b, token_rows]
            voxels[rows] = geom.voxel_ids[b, token_rows]
            confidence[rows] = geom.confidence[b, token_rows]
        return voxels, valid, confidence

    def _lookup_patch_coords(
        self,
        metadata: KVCacheMetadata,
        b: int,
        h: int,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return row-major patch coordinates for patch tokens.

        Token order is assumed to match ``record_frame_geometry``: patch tokens
        are contiguous after ``patch_start_idx`` and laid out row-major over the
        frame patch grid. Non-patch tokens and frames without geometry are marked
        invalid for local spatial candidate search.
        """
        frame_ids = metadata.frame_ids[b, h, positions].long()
        token_ids = metadata.token_indices[b, h, positions].long()
        rows = torch.full((positions.numel(),), -1, dtype=torch.long)
        cols = torch.full((positions.numel(),), -1, dtype=torch.long)
        valid = torch.zeros((positions.numel(),), dtype=torch.bool)
        for frame_id in torch.unique(frame_ids).tolist():
            geom = self._geometry.get(int(frame_id))
            if geom is None or geom.patch_width <= 0:
                continue
            frame_mask = frame_ids == int(frame_id)
            frame_token_ids = token_ids[frame_mask]
            patch_ids = frame_token_ids.long() - int(self.patch_start_idx)
            frame_valid = (patch_ids >= 0) & (patch_ids < int(geom.patch_tokens))
            if not bool(frame_valid.any()):
                continue
            out_rows = torch.nonzero(frame_mask, as_tuple=False).flatten()[frame_valid]
            patch_ids = patch_ids[frame_valid]
            rows[out_rows] = torch.div(patch_ids, int(geom.patch_width), rounding_mode="floor")
            cols[out_rows] = patch_ids % int(geom.patch_width)
            valid[out_rows] = True
        return rows, cols, valid

    def _accumulate_recall_debug(
        self,
        stats: RecentMergeStats,
        cur_k: torch.Tensor,
        cand_k: torch.Tensor,
        local_candidate_mask: torch.Tensor,
        local_geometry_mask: torch.Tensor,
        cur_vox: torch.Tensor,
        cand_vox: torch.Tensor,
        cur_valid: torch.Tensor,
        cand_valid: torch.Tensor,
    ) -> None:
        if cur_k.shape[0] == 0 or cand_k.shape[0] == 0:
            return
        limit = min(int(self.config.recall_debug_max_tokens), int(cur_k.shape[0]))
        if limit <= 0:
            return
        device = cur_k.device
        cur_valid_gpu = cur_valid[:limit].to(device=device)
        cand_valid_gpu = cand_valid.to(device=device)
        cur_vox_gpu = cur_vox[:limit].to(device=device)
        cand_vox_gpu = cand_vox.to(device=device)
        if self.config.disable_geometry_check:
            full_geometry_mask = cur_valid_gpu[:, None].new_ones((limit, cand_k.shape[0]), dtype=torch.bool)
        else:
            full_geometry_mask = (
                cur_valid_gpu[:, None]
                & cand_valid_gpu[None, :]
                & (cur_vox_gpu[:, None, :] == cand_vox_gpu[None, :, :]).all(dim=-1)
            )

        for row in range(limit):
            full_cols = torch.nonzero(full_geometry_mask[row], as_tuple=False).flatten()
            if full_cols.numel() == 0:
                continue
            full_sim = (cur_k[row : row + 1] @ cand_k[full_cols].T).flatten()
            full_value, full_idx = full_sim.max(dim=0)
            full_col = full_cols[full_idx]

            local_mask = local_candidate_mask[row] & local_geometry_mask[row]
            local_cols = torch.nonzero(local_mask, as_tuple=False).flatten()
            local_value = torch.tensor(float("nan"), device=device)
            if local_cols.numel() > 0:
                local_sim = (cur_k[row : row + 1] @ cand_k[local_cols].T).flatten()
                local_value = local_sim.max()

            stats.recall_samples += 1
            stats.recall_hits += int(bool(local_candidate_mask[row, full_col].item()))
            stats.recall_full_similarity_sum += float(full_value.detach().cpu().item())
            if torch.isfinite(local_value):
                stats.recall_local_similarity_sum += float(local_value.detach().cpu().item())

    def _profile_start(self, tensor: torch.Tensor) -> Optional[float]:
        if not self.config.profile:
            return None
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        return time.perf_counter()

    def _profile_add(
        self,
        stats: RecentMergeStats,
        name: str,
        start: Optional[float],
        tensor: torch.Tensor,
    ) -> None:
        if start is None:
            return
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        stats.profile_times[name] = stats.profile_times.get(name, 0.0) + (time.perf_counter() - start)

    def _build_keep_indices(
        self,
        metadata: KVCacheMetadata,
        frame_id: int,
        remove_token_ids: list[int],
    ) -> torch.Tensor:
        B, H, N = metadata.frame_ids.shape
        remove = torch.tensor(remove_token_ids, dtype=metadata.token_indices.dtype)
        remove_mask = (metadata.frame_ids == int(frame_id)) & torch.isin(metadata.token_indices, remove)
        keep_mask = ~remove_mask
        keep = torch.nonzero(keep_mask, as_tuple=False)[:, 2].view(B, H, N - len(remove_token_ids))
        return keep.to(dtype=torch.long)

    def _debug(self, layer_id: int, frame_id: int, stats: RecentMergeStats) -> None:
        if not self.config.debug and not self.config.profile and not self.config.recall_debug:
            return
        sim = torch.tensor(stats.similarities, dtype=torch.float32)
        conf = torch.tensor(stats.confidences, dtype=torch.float32)
        gaps = {}
        for gap in stats.frame_gaps:
            gaps[gap] = gaps.get(gap, 0) + 1
        sim_mean = float(sim.mean().item()) if sim.numel() else 0.0
        sim_med = float(sim.median().item()) if sim.numel() else 0.0
        conf_mean = float(conf.mean().item()) if conf.numel() else 0.0
        conf_med = float(conf.median().item()) if conf.numel() else 0.0
        avg_candidates = (
            float(stats.local_candidate_pairs) / float(stats.current_tokens)
            if stats.current_tokens
            else 0.0
        )
        # ``current_tokens`` is accumulated over batch/head, while merged_tokens is
        # shared-token removals. The ratio below is intentionally conservative and
        # mainly useful for relative per-layer diagnostics.
        accepted_ratio = float(stats.merged_tokens) / float(max(stats.current_tokens, 1))
        print(
            f"[RecentSimilarityMerge] layer={layer_id} frame={frame_id} "
            f"mode={self.config.candidate_mode} current={stats.current_tokens} "
            f"candidates_before_local={stats.candidate_tokens} "
            f"avg_candidates_per_src={avg_candidates:.2f} "
            f"max_candidates_per_src={stats.max_candidates_per_src} "
            f"attempted={stats.attempted_matches} merged={stats.merged_tokens} "
            f"accepted_ratio={accepted_ratio:.4f} "
            f"rejected_similarity={stats.rejected_similarity} "
            f"rejected_geometry={stats.rejected_geometry} "
            f"fallback={stats.fallback_count} "
            f"threshold_pairs={stats.threshold_pairs} voxel_pairs={stats.voxel_pairs} "
            f"gap_hist={gaps} "
            f"sim_mean={sim_mean:.4f} sim_median={sim_med:.4f} "
            f"conf_mean={conf_mean:.4f} conf_median={conf_med:.4f} "
            f"cache={stats.cache_before}->{stats.cache_after}"
        )
        if self.config.profile and stats.profile_times:
            timings = " ".join(f"{name}={value * 1000.0:.2f}ms" for name, value in sorted(stats.profile_times.items()))
            print(f"[RecentSimilarityMergeProfile] layer={layer_id} frame={frame_id} {timings}")
        if self.config.recall_debug and stats.recall_samples:
            hit_rate = float(stats.recall_hits) / float(stats.recall_samples)
            full_sim = stats.recall_full_similarity_sum / float(stats.recall_samples)
            local_sim = stats.recall_local_similarity_sum / float(stats.recall_samples)
            print(
                f"[RecentSimilarityMergeRecall] layer={layer_id} frame={frame_id} "
                f"samples={stats.recall_samples} full_best_in_local={hit_rate:.4f} "
                f"full_best_sim={full_sim:.4f} local_best_sim={local_sim:.4f}"
            )
