"""History Anchor Manager for streaming KV cache protection."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class HistoryAnchorConfig:
    """Configuration for History Anchor selection."""

    strategy: str = "none"
    interval: int = 50
    min_anchor_interval: Optional[int] = None
    max_anchors: int = 3
    coverage_threshold: float = 0.4
    camera_motion_threshold: float = 0.2
    sample_ratio: float = 0.1
    anchor_keep_ratio: float = 1.0
    history_protect_token_count: int = 5
    global_protect_token_count: Optional[int] = None
    patch_topk_per_anchor_frame: int = 0


def compute_coverage(
    anchor_depth: torch.Tensor,
    anchor_pose: torch.Tensor,
    current_pose: torch.Tensor,
    image_size_hw: Tuple[int, int],
    sample_ratio: float = 0.1,
) -> float:
    """Project anchor-frame depth points into the current view and measure coverage."""
    from .geometry import closed_form_inverse_se3
    from .pose_enc import pose_encoding_to_extri_intri

    H, W = image_size_hw
    device = anchor_depth.device
    dtype = anchor_depth.dtype

    if anchor_depth.dim() == 3:
        anchor_depth = anchor_depth.squeeze(-1)

    anchor_pose_batched = anchor_pose.unsqueeze(0).unsqueeze(0)
    current_pose_batched = current_pose.unsqueeze(0).unsqueeze(0)
    anchor_extri, anchor_intri = pose_encoding_to_extri_intri(
        anchor_pose_batched,
        image_size_hw,
    )
    current_extri, current_intri = pose_encoding_to_extri_intri(
        current_pose_batched,
        image_size_hw,
    )

    anchor_extri = anchor_extri[0, 0]
    anchor_intri = anchor_intri[0, 0]
    current_extri = current_extri[0, 0]
    current_intri = current_intri[0, 0]

    total_pixels = H * W
    num_samples = max(1, int(total_pixels * sample_ratio))
    valid_indices = torch.nonzero(anchor_depth.flatten() > 1e-6, as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return 0.0

    if valid_indices.numel() > num_samples:
        perm = torch.randperm(valid_indices.numel(), device=device)[:num_samples]
        sample_indices = valid_indices[perm]
    else:
        sample_indices = valid_indices

    v_coords = sample_indices // W
    u_coords = sample_indices % W
    depths = anchor_depth[v_coords, u_coords]

    fx = anchor_intri[0, 0]
    fy = anchor_intri[1, 1]
    cx = anchor_intri[0, 2]
    cy = anchor_intri[1, 2]
    x_cam = (u_coords.float() - cx) * depths / fx
    y_cam = (v_coords.float() - cy) * depths / fy
    z_cam = depths
    cam_coords = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    anchor_extri_4x4 = torch.eye(4, device=device, dtype=dtype)
    anchor_extri_4x4[:3, :4] = anchor_extri
    anchor_cam_to_world = closed_form_inverse_se3(anchor_extri_4x4.unsqueeze(0))[0]
    world_coords = cam_coords @ anchor_cam_to_world[:3, :3].T + anchor_cam_to_world[:3, 3]

    current_cam_coords = world_coords @ current_extri[:3, :3].T + current_extri[:3, 3]
    z_current = current_cam_coords[:, 2]
    valid_z = z_current > 1e-6

    u_proj = torch.zeros_like(z_current)
    v_proj = torch.zeros_like(z_current)
    u_proj[valid_z] = (
        current_cam_coords[valid_z, 0] / z_current[valid_z]
    ) * current_intri[0, 0] + current_intri[0, 2]
    v_proj[valid_z] = (
        current_cam_coords[valid_z, 1] / z_current[valid_z]
    ) * current_intri[1, 1] + current_intri[1, 2]

    in_bounds = (
        valid_z
        & (u_proj >= 0)
        & (u_proj < W)
        & (v_proj >= 0)
        & (v_proj < H)
    )
    return in_bounds.float().mean().item()


def compute_camera_motion_score(
    previous_pose: torch.Tensor,
    current_pose: torch.Tensor,
    eps: float = 1e-12,
    translation_scale: Optional[float] = None,
    rotation_scale: Optional[float] = None,
) -> float:
    """Compute pose-encoding camera motion between two frames."""
    previous = previous_pose.detach().float()
    current = current_pose.detach().float()
    translation_score = torch.linalg.vector_norm(current[..., :3] - previous[..., :3], dim=-1)
    if translation_scale is not None:
        scale = max(float(translation_scale), eps)
        translation_score = translation_score / scale

    previous_q = previous[..., 3:7]
    current_q = current[..., 3:7]
    previous_q = previous_q / previous_q.norm(dim=-1, keepdim=True).clamp_min(eps)
    current_q = current_q / current_q.norm(dim=-1, keepdim=True).clamp_min(eps)
    quat_dot = (current_q * previous_q).sum(dim=-1).abs().clamp(max=1.0)
    rotation_score = 2.0 * torch.acos(quat_dot)
    if rotation_scale is not None:
        scale = max(float(rotation_scale), eps)
        rotation_score = rotation_score / scale
    score = translation_score + rotation_score
    return float(torch.nan_to_num(score.mean(), nan=0.0, posinf=0.0, neginf=0.0).item())


class HistoryAnchorManager:
    """Manage fixed-interval and coverage-based history anchors."""

    def __init__(self, config: HistoryAnchorConfig, tokens_per_frame: int):
        self.config = config
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_history_anchors = 0
        self.history_anchor_frames: List[int] = []
        self.next_anchor_frame = int(config.interval)
        self.last_anchor_frame: Optional[int] = None
        self.latest_anchor_depth: Optional[torch.Tensor] = None
        self.latest_anchor_pose: Optional[torch.Tensor] = None
        self.latest_camera_motion_anchor_pose: Optional[torch.Tensor] = None
        self.latest_camera_motion_previous_pose: Optional[torch.Tensor] = None
        self.camera_motion_step_translations: List[float] = []
        self.camera_motion_step_rotations: List[float] = []
        self.camera_motion_step_window = 20
        self.image_size_hw: Optional[Tuple[int, int]] = None

    def is_eviction_paused(self) -> bool:
        return False

    def should_become_anchor(self, frame_idx: int) -> Tuple[bool, bool, str]:
        if self.config.strategy != "fixed_interval":
            return False, False, "disabled"
        if self.config.max_anchors <= 0:
            return False, False, "max_anchors_disabled"
        if frame_idx != self.next_anchor_frame:
            return False, False, f"not_target_frame_{frame_idx}"

        self.next_anchor_frame += self.config.interval
        is_fifo = self.num_history_anchors >= self.config.max_anchors
        return True, is_fifo, f"interval_anchor_at_frame_{frame_idx}"

    def register_anchor(self, frame_idx: int) -> None:
        if self.config.max_anchors <= 0:
            return
        if self.num_history_anchors < self.config.max_anchors:
            self.num_history_anchors += 1
        if (
            self.config.patch_topk_per_anchor_frame <= 0
            and len(self.history_anchor_frames) >= self.config.max_anchors
        ):
            self.history_anchor_frames.pop(0)
        self.history_anchor_frames.append(frame_idx)

    def get_protected_token_count(self) -> int:
        global_anchor_tokens = self.tokens_per_frame
        if self.config.global_protect_token_count is not None:
            global_anchor_tokens = min(
                max(int(self.config.global_protect_token_count), 0),
                self.tokens_per_frame,
            )
        history_anchor_tokens = self.num_history_anchors * min(
            int(self.config.history_protect_token_count),
            self.tokens_per_frame,
        )
        return global_anchor_tokens + history_anchor_tokens

    def get_num_anchors(self) -> int:
        return 1 + self.num_history_anchors

    def should_become_anchor_coverage(
        self,
        frame_idx: int,
        current_depth: torch.Tensor,
        current_pose: torch.Tensor,
    ) -> Tuple[bool, bool, str, float]:
        if self.config.strategy != "coverage":
            return False, False, "coverage_disabled", 1.0

        if frame_idx == 0:
            self.latest_anchor_depth = current_depth.clone()
            self.latest_anchor_pose = current_pose.clone()
            return False, False, "frame_0_global_anchor", 1.0

        if self.latest_anchor_depth is None or self.latest_anchor_pose is None:
            return True, False, "no_anchor_info", 0.0

        if self.image_size_hw is None:
            if current_depth.dim() == 3:
                self.image_size_hw = (current_depth.shape[0], current_depth.shape[1])
            else:
                self.image_size_hw = tuple(current_depth.shape)

        coverage = compute_coverage(
            self.latest_anchor_depth,
            self.latest_anchor_pose,
            current_pose,
            self.image_size_hw,
            self.config.sample_ratio,
        )
        if coverage >= self.config.coverage_threshold:
            return (
                False,
                False,
                f"coverage_{coverage:.3f}>=threshold_{self.config.coverage_threshold}",
                coverage,
            )

        if self.last_anchor_frame is not None:
            base_interval = self.config.min_anchor_interval
            if base_interval is None:
                base_interval = self.config.interval
            min_interval = max(base_interval, 0)
            if (frame_idx - self.last_anchor_frame) < min_interval:
                return False, False, f"min_interval_{min_interval}_not_met", coverage

        is_fifo = self.num_history_anchors >= self.config.max_anchors
        return (
            self.config.max_anchors > 0,
            is_fifo,
            f"coverage_{coverage:.3f}<threshold_{self.config.coverage_threshold}",
            coverage,
        )

    def register_anchor_coverage(
        self,
        frame_idx: int,
        depth: torch.Tensor,
        pose: torch.Tensor,
    ) -> None:
        self.register_anchor(frame_idx)
        self.last_anchor_frame = frame_idx
        self.latest_anchor_depth = depth.clone()
        self.latest_anchor_pose = pose.clone()

    def should_become_anchor_camera_motion(
        self,
        frame_idx: int,
        current_pose: torch.Tensor,
    ) -> Tuple[bool, bool, str, float]:
        if self.config.strategy != "camera_motion":
            return False, False, "camera_motion_disabled", 0.0

        self._update_camera_motion_step_scale(current_pose)

        if self.latest_camera_motion_anchor_pose is None:
            self.latest_camera_motion_anchor_pose = current_pose.clone()
            return False, False, "frame_0_global_anchor", 0.0

        score = compute_camera_motion_score(
            self.latest_camera_motion_anchor_pose,
            current_pose,
            translation_scale=self._camera_motion_translation_step_median(),
            rotation_scale=self._camera_motion_rotation_step_median(),
        )
        if score < self.config.camera_motion_threshold:
            return (
                False,
                False,
                f"camera_motion_{score:.3f}<threshold_{self.config.camera_motion_threshold}",
                score,
            )

        if self.last_anchor_frame is not None:
            base_interval = self.config.min_anchor_interval
            if base_interval is None:
                base_interval = self.config.interval
            min_interval = max(base_interval, 0)
            if (frame_idx - self.last_anchor_frame) < min_interval:
                return False, False, f"min_interval_{min_interval}_not_met", score

        is_fifo = self.num_history_anchors >= self.config.max_anchors
        return (
            self.config.max_anchors > 0,
            is_fifo,
            f"camera_motion_{score:.3f}>=threshold_{self.config.camera_motion_threshold}",
            score,
        )

    def _update_camera_motion_step_scale(self, current_pose: torch.Tensor) -> None:
        current = current_pose.detach().float()
        if self.latest_camera_motion_previous_pose is not None:
            previous = self.latest_camera_motion_previous_pose.detach().float()
            step = torch.linalg.vector_norm(current[..., :3] - previous[..., :3], dim=-1)
            step_value = float(torch.nan_to_num(step.mean(), nan=0.0, posinf=0.0, neginf=0.0).item())
            if step_value > 0.0:
                self.camera_motion_step_translations.append(step_value)
                if len(self.camera_motion_step_translations) > self.camera_motion_step_window:
                    self.camera_motion_step_translations = self.camera_motion_step_translations[-self.camera_motion_step_window:]

            previous_q = previous[..., 3:7]
            current_q = current[..., 3:7]
            previous_q = previous_q / previous_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            current_q = current_q / current_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            quat_dot = (current_q * previous_q).sum(dim=-1).abs().clamp(max=1.0)
            rot_step = 2.0 * torch.acos(quat_dot)
            rot_value = float(torch.nan_to_num(rot_step.mean(), nan=0.0, posinf=0.0, neginf=0.0).item())
            if rot_value > 0.0:
                self.camera_motion_step_rotations.append(rot_value)
                if len(self.camera_motion_step_rotations) > self.camera_motion_step_window:
                    self.camera_motion_step_rotations = self.camera_motion_step_rotations[-self.camera_motion_step_window:]
        self.latest_camera_motion_previous_pose = current_pose.clone()

    def _camera_motion_translation_step_median(self) -> Optional[float]:
        return self._median_or_none(self.camera_motion_step_translations)

    def _camera_motion_rotation_step_median(self) -> Optional[float]:
        return self._median_or_none(self.camera_motion_step_rotations)

    @staticmethod
    def _median_or_none(values: List[float]) -> Optional[float]:
        if not values:
            return None
        tensor_values = torch.tensor(values, dtype=torch.float32)
        median = torch.median(tensor_values)
        return float(torch.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0).item())

    def register_anchor_camera_motion(
        self,
        frame_idx: int,
        pose: torch.Tensor,
    ) -> None:
        self.register_anchor(frame_idx)
        self.last_anchor_frame = frame_idx
        self.latest_camera_motion_anchor_pose = pose.clone()

    def __repr__(self) -> str:
        base_repr = (
            "HistoryAnchorManager("
            f"strategy={self.config.strategy}, "
            f"num_anchors={self.get_num_anchors()}, "
            f"history_frames={self.history_anchor_frames}"
        )
        if self.config.strategy == "fixed_interval":
            return base_repr + f", next_target={self.next_anchor_frame})"
        if self.config.strategy == "coverage":
            return base_repr + f", threshold={self.config.coverage_threshold})"
        if self.config.strategy == "camera_motion":
            return base_repr + f", threshold={self.config.camera_motion_threshold})"
        return base_repr + ")"
