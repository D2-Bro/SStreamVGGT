"""Sliding-window similarity merge for recent streaming KV cache entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from streamvggt.utils.geometry import closed_form_inverse_se3
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


_INVALID_VOXEL = -2**30


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
            frame_ids=torch.full(shape, int(frame_id), dtype=torch.int32),
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
        for b in range(B):
            for h in range(H):
                mask = self.frame_ids[b, h] == int(frame_id)
                if not bool(mask.any()):
                    continue
                token_ids = self.token_indices[b, h, mask].long()
                valid = (token_ids >= 0) & (token_ids < max_tokens)
                values = torch.ones_like(token_ids, dtype=torch.float32)
                values[valid] = token_confidence[b, token_ids[valid]]
                self.accumulated_confidence[b, h, mask] = values


@dataclass
class FrameGeometry:
    """Recent per-frame patch geometry used only for merge validation."""

    voxel_ids: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


@dataclass
class RecentMergeStats:
    current_tokens: int = 0
    candidate_tokens: int = 0
    threshold_pairs: int = 0
    voxel_pairs: int = 0
    merged_tokens: int = 0
    similarities: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    frame_gaps: list[int] = field(default_factory=list)
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
            self._geometry[int(frame_id)] = FrameGeometry(voxel_ids, confidence, valid)
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

        geom = FrameGeometry(voxel_ids=voxel_ids, confidence=confidence, valid=valid)
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

        if not matched_token_sets:
            stats.cache_after = int(k.shape[2])
            return k, v, metadata, stats

        common_token_ids = sorted(set.intersection(*matched_token_sets)) if matched_token_sets else []
        if not common_token_ids:
            stats.cache_after = int(k.shape[2])
            self._debug(layer_id, frame_id, stats)
            return k, v, metadata, stats

        with torch.no_grad():
            for b in range(B):
                for h in range(H):
                    matches = per_head_matches[(b, h)]
                    for token_id in common_token_ids:
                        cur_pos, cand_pos, sim, new_conf = matches[token_id]
                        old_conf = float(metadata.accumulated_confidence[b, h, cand_pos].item())
                        old_w = old_conf if self.config.use_depth_confidence else 1.0
                        new_w = new_conf if self.config.use_depth_confidence else 1.0
                        denom = old_w + new_w + 1e-6
                        alpha = old_w / denom
                        k[b, h, cand_pos] = alpha * k[b, h, cand_pos] + (1.0 - alpha) * k[b, h, cur_pos]
                        v[b, h, cand_pos] = alpha * v[b, h, cand_pos] + (1.0 - alpha) * v[b, h, cur_pos]
                        metadata.accumulated_confidence[b, h, cand_pos] = old_conf + new_conf
                        metadata.merge_counts[b, h, cand_pos] += 1
                        metadata.last_updated_frame[b, h, cand_pos] = int(frame_id)
                        stats.similarities.append(float(sim))
                        stats.confidences.append(float(new_conf))
                        cand_frame = int(metadata.frame_ids[b, h, cand_pos].item())
                        stats.frame_gaps.append(int(frame_id) - cand_frame)

        keep_indices_cpu = self._build_keep_indices(metadata, frame_id, common_token_ids)
        keep_indices = keep_indices_cpu.to(device=k.device, dtype=torch.long)
        k = torch.gather(k, 2, keep_indices.unsqueeze(-1).expand(B, H, keep_indices.shape[2], D))
        v = torch.gather(v, 2, keep_indices.unsqueeze(-1).expand(B, H, keep_indices.shape[2], D))
        metadata = metadata.gather(keep_indices_cpu)
        stats.merged_tokens = len(common_token_ids)
        stats.cache_after = int(k.shape[2])
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
        stats = RecentMergeStats()
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

        cur_pos = cur_pos[valid_cur]
        cand_pos = cand_pos[valid_cand]
        cur_vox = cur_vox[valid_cur]
        cand_vox = cand_vox[valid_cand]
        cur_conf = cur_conf[valid_cur]
        if cur_pos.numel() == 0 or cand_pos.numel() == 0:
            return {}, stats

        cur_k = F.normalize(k[b, h, cur_pos.to(k.device)].float(), dim=-1)
        cand_k = F.normalize(k[b, h, cand_pos.to(k.device)].float(), dim=-1)
        if self.config.disable_geometry_check:
            same_voxel = torch.ones(
                (cur_vox.shape[0], cand_vox.shape[0]),
                dtype=torch.bool,
                device=cur_vox.device,
            )
        else:
            same_voxel = (cur_vox[:, None, :] == cand_vox[None, :, :]).all(dim=-1)

        best: Dict[int, Tuple[int, int, float, float]] = {}
        used_candidates: set[int] = set()
        best_rows = []
        chunk = max(int(self.config.chunk_size), 1)
        for start in range(0, cur_k.shape[0], chunk):
            end = min(start + chunk, cur_k.shape[0])
            sim = cur_k[start:end] @ cand_k.T
            threshold_mask = sim >= float(self.config.similarity_threshold)
            stats.threshold_pairs += int(threshold_mask.sum().item())
            voxel_mask = same_voxel[start:end].to(device=sim.device)
            valid_mask = threshold_mask & voxel_mask
            stats.voxel_pairs += int(valid_mask.sum().item())
            masked = sim.masked_fill(~valid_mask, -float("inf"))
            values, indices = masked.max(dim=1)
            for local_row, (score, cand_col) in enumerate(zip(values.detach().cpu(), indices.detach().cpu())):
                if not torch.isfinite(score):
                    continue
                global_row = start + local_row
                best_rows.append(
                    (
                        float(score.item()),
                        int(cur_pos[global_row].item()),
                        int(cand_pos[int(cand_col.item())].item()),
                        int(metadata.token_indices[b, h, cur_pos[global_row]].item()),
                        float(cur_conf[global_row].item()),
                    )
                )

        best_rows.sort(key=lambda row: (-row[0], row[3], row[2]))
        for score, cur_cache_pos, cand_cache_pos, token_id, conf in best_rows:
            if token_id in best or cand_cache_pos in used_candidates:
                continue
            best[token_id] = (cur_cache_pos, cand_cache_pos, score, conf)
            used_candidates.add(cand_cache_pos)
        return best, stats

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
        for row, (frame_id, token_id) in enumerate(zip(frame_ids.tolist(), token_ids.tolist())):
            geom = self._geometry.get(int(frame_id))
            if geom is None:
                continue
            if b >= geom.valid.shape[0] or token_id < 0 or token_id >= geom.valid.shape[1]:
                continue
            valid[row] = bool(geom.valid[b, token_id].item())
            voxels[row] = geom.voxel_ids[b, token_id]
            confidence[row] = geom.confidence[b, token_id]
        return voxels, valid, confidence

    def _build_keep_indices(
        self,
        metadata: KVCacheMetadata,
        frame_id: int,
        remove_token_ids: list[int],
    ) -> torch.Tensor:
        remove = set(int(x) for x in remove_token_ids)
        B, H, N = metadata.frame_ids.shape
        keep = torch.empty((B, H, N - len(remove)), dtype=torch.long)
        for b in range(B):
            for h in range(H):
                rows = []
                for idx in range(N):
                    is_removed = (
                        int(metadata.frame_ids[b, h, idx].item()) == int(frame_id)
                        and int(metadata.token_indices[b, h, idx].item()) in remove
                    )
                    if not is_removed:
                        rows.append(idx)
                keep[b, h] = torch.tensor(rows, dtype=torch.long)
        return keep

    def _debug(self, layer_id: int, frame_id: int, stats: RecentMergeStats) -> None:
        if not self.config.debug:
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
        print(
            f"[RecentSimilarityMerge] layer={layer_id} frame={frame_id} "
            f"current={stats.current_tokens} candidates={stats.candidate_tokens} "
            f"threshold_pairs={stats.threshold_pairs} voxel_pairs={stats.voxel_pairs} "
            f"merged={stats.merged_tokens} gap_hist={gaps} "
            f"sim_mean={sim_mean:.4f} sim_median={sim_med:.4f} "
            f"conf_mean={conf_mean:.4f} conf_median={conf_med:.4f} "
            f"cache={stats.cache_before}->{stats.cache_after}"
        )
