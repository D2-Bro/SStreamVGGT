"""Voxel covisibility frame selection for streaming KV cache reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from streamvggt.utils.geometry import closed_form_inverse_se3
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


@dataclass
class VoxelCovisConfig:
    enabled: bool = False
    voxel_size: float = 0.05
    min_shared_voxels: int = 20
    min_overlap: float = 0.05
    max_covis_frames: int = 8
    fallback_recent: int = 1
    cache_miss_fallback_recent: int = 1
    debug: bool = False


@dataclass
class VoxelCovisScore:
    frame_id: int
    shared_voxels: int
    overlap_ratio: float


@dataclass
class VoxelCovisSelection:
    current_frame_id: int
    reference_frame_id: Optional[int]
    selected_frame_ids: List[int]
    fallback_used: bool
    scores: List[VoxelCovisScore] = field(default_factory=list)


class VoxelCovisibilityGraph:
    """Stores compact per-frame voxel sets and selects covisible cache frames."""

    def __init__(self, config: VoxelCovisConfig, patch_size: int = 14) -> None:
        self.config = config
        self.patch_size = int(patch_size)
        self._frame_voxels: Dict[int, torch.Tensor] = {}

    def set_frame_voxels(self, frame_id: int, voxel_ids: torch.Tensor) -> None:
        voxel_ids = voxel_ids.detach().cpu().to(torch.int64).reshape(-1)
        voxel_ids = voxel_ids[voxel_ids != _INVALID_HASH]
        self._frame_voxels[int(frame_id)] = torch.unique(voxel_ids, sorted=True)

    def record_frame_geometry(
        self,
        frame_id: int,
        depth: torch.Tensor,
        depth_conf: Optional[torch.Tensor],
        pose_enc: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if not self.config.enabled:
            return None
        voxel_ids = depth_pose_to_voxel_ids(
            depth=depth,
            depth_conf=depth_conf,
            pose_enc=pose_enc,
            image_hw=image_hw,
            patch_size=self.patch_size,
            voxel_size=self.config.voxel_size,
        )
        self.set_frame_voxels(frame_id, voxel_ids)
        return self._frame_voxels.get(int(frame_id))

    def select_for_frame(self, current_frame_id: int) -> VoxelCovisSelection:
        current_frame_id = int(current_frame_id)
        reference_frame_id = current_frame_id - 1 if current_frame_id > 0 else None
        if reference_frame_id is None or reference_frame_id not in self._frame_voxels:
            return VoxelCovisSelection(current_frame_id, reference_frame_id, [], False, [])

        query = self._frame_voxels[reference_frame_id]
        scores: List[VoxelCovisScore] = []
        for frame_id in sorted(fid for fid in self._frame_voxels if fid < current_frame_id):
            candidate = self._frame_voxels[frame_id]
            shared = _intersection_count(query, candidate)
            denom = max(min(int(query.numel()), int(candidate.numel())), 1)
            overlap = float(shared) / float(denom)
            if (
                shared >= int(self.config.min_shared_voxels)
                and overlap >= float(self.config.min_overlap)
            ):
                scores.append(VoxelCovisScore(frame_id, shared, overlap))

        mandatory_frames = []
        if 0 in self._frame_voxels:
            mandatory_frames.append(0)
        if reference_frame_id not in mandatory_frames:
            mandatory_frames.append(reference_frame_id)

        selected = _rank_scores(scores)
        fallback_used = len(selected) == 0
        if fallback_used and int(self.config.fallback_recent) > 0:
            selected.extend(
                _recent_frame_ids(
                    self._frame_voxels,
                    current_frame_id,
                    int(self.config.fallback_recent),
                )
            )
        for frame_id in mandatory_frames:
            if frame_id not in selected:
                selected.append(frame_id)


        max_frames = int(self.config.max_covis_frames)
        if max_frames > 0 and len(selected) > max_frames:
            mandatory_kept = [fid for fid in mandatory_frames if fid in selected][:max_frames]
            remaining_slots = max(max_frames - len(mandatory_kept), 0)
            mandatory_set = set(mandatory_kept)
            ranked = [
                fid
                for fid in _rank_scores(scores)
                if fid in selected and fid not in mandatory_set
            ]
            selected = mandatory_kept + ranked[:remaining_slots]

        selected = sorted(set(selected))
        return VoxelCovisSelection(
            current_frame_id=current_frame_id,
            reference_frame_id=reference_frame_id,
            selected_frame_ids=selected,
            fallback_used=fallback_used,
            scores=scores,
        )


def depth_pose_to_voxel_ids(
    depth: torch.Tensor,
    depth_conf: Optional[torch.Tensor],
    pose_enc: torch.Tensor,
    image_hw: tuple[int, int],
    patch_size: int,
    voxel_size: float,
) -> torch.Tensor:
    H_img, W_img = int(image_hw[0]), int(image_hw[1])
    depth = depth.detach()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        return torch.empty(0, dtype=torch.int64)

    B = depth.shape[0]
    if depth_conf is None:
        depth_conf = torch.ones_like(depth)
    else:
        depth_conf = depth_conf.detach()
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.unsqueeze(1), (H_img, W_img))
    if intrinsic is None:
        return torch.empty(0, dtype=torch.int64)

    extrinsic = extrinsic[:, 0]
    intrinsic = intrinsic[:, 0]
    extrinsic_h = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device).repeat(B, 1, 1)
    extrinsic_h[:, :3, :4] = extrinsic
    cam_to_world = closed_form_inverse_se3(extrinsic_h)

    patch_h = H_img // int(patch_size)
    patch_w = W_img // int(patch_size)
    patch_tokens = patch_h * patch_w
    if patch_tokens <= 0:
        return torch.empty(0, dtype=torch.int64)

    patch_ids = torch.arange(patch_tokens, device=depth.device)
    rows = torch.div(patch_ids, patch_w, rounding_mode="floor")
    cols = patch_ids % patch_w
    ys = torch.clamp((rows.float() + 0.5) * int(patch_size), 0, H_img - 1).round().long()
    xs = torch.clamp((cols.float() + 0.5) * int(patch_size), 0, W_img - 1).round().long()

    sampled_depth = depth[:, ys, xs].float()
    sampled_conf = depth_conf[:, ys, xs].float()
    valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth > 1e-8)
        & torch.isfinite(sampled_conf)
        & (sampled_conf > 0)
    )

    ones = torch.ones_like(sampled_depth)
    pix = torch.stack(
        [
            xs.float().view(1, -1).expand(B, -1),
            ys.float().view(1, -1).expand(B, -1),
            ones,
        ],
        dim=-1,
    )
    inv_intrinsic = torch.linalg.inv(intrinsic.float())
    rays = torch.bmm(pix.float(), inv_intrinsic.transpose(1, 2))
    cam_xyz = rays * sampled_depth.unsqueeze(-1)
    world_xyz = torch.bmm(cam_xyz, cam_to_world[:, :3, :3].float().transpose(1, 2))
    world_xyz = world_xyz + cam_to_world[:, :3, 3].float().unsqueeze(1)
    voxels = torch.floor(world_xyz / float(voxel_size)).to(torch.int64)
    hashes = hash_voxels(voxels)
    return hashes[valid].detach().cpu()


def hash_voxels(voxels: torch.Tensor) -> torch.Tensor:
    voxels = voxels.to(torch.int64)
    x, y, z = voxels.unbind(dim=-1)
    return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)


def _intersection_count(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.numel() == 0 or b.numel() == 0:
        return 0
    return int(torch.isin(a, b).sum().item())


def _rank_scores(scores: List[VoxelCovisScore]) -> List[int]:
    return [
        score.frame_id
        for score in sorted(
            scores,
            key=lambda item: (item.overlap_ratio, item.shared_voxels, item.frame_id),
            reverse=True,
        )
    ]


def _recent_frame_ids(frame_voxels: Dict[int, torch.Tensor], current_frame_id: int, count: int) -> List[int]:
    if count <= 0:
        return []
    return sorted(
        (fid for fid in frame_voxels if fid < current_frame_id),
        reverse=True,
    )[:count]


_INVALID_HASH = torch.iinfo(torch.int64).min
