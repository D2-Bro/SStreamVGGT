"""Lightweight confidence sidecar aligned with streaming KV caches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def parse_confidence_gate_init(value: str | float | int) -> str | float:
    if isinstance(value, str) and value == "mean":
        return "mean"
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("leverage_conf_gate_init must be 'mean' or a finite non-negative float") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("leverage_conf_gate_init must be 'mean' or a finite non-negative float")
    return parsed


@dataclass
class KVConfidenceState:
    """GPU-side token provenance and confidence aligned with a KV cache."""

    frame_ids: torch.Tensor
    token_indices: torch.Tensor
    confidence_gate: torch.Tensor
    gate_sum: torch.Tensor
    gate_count: torch.Tensor

    @staticmethod
    def _stats_from_gate(confidence_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(confidence_gate)
        gate_sum = torch.where(finite, confidence_gate, torch.zeros_like(confidence_gate)).sum(dim=1)
        gate_count = finite.sum(dim=1)
        return gate_sum, gate_count

    def mean_gate(self) -> torch.Tensor:
        return torch.where(
            self.gate_count > 0,
            self.gate_sum / self.gate_count.clamp_min(1).to(dtype=self.gate_sum.dtype),
            torch.ones_like(self.gate_sum),
        )

    @classmethod
    def for_current_frame(
        cls,
        batch_size: int,
        num_heads: int,
        num_tokens: int,
        frame_id: int,
        device: torch.device,
        initial_gate: Optional[torch.Tensor | float] = None,
    ) -> "KVConfidenceState":
        del num_heads  # Confidence is shared across heads; keep the arg for call-site compatibility.
        shape = (int(batch_size), int(num_tokens))
        token_indices = torch.arange(num_tokens, device=device, dtype=torch.long).view(1, num_tokens)
        if initial_gate is None:
            confidence_gate = torch.ones(shape, device=device, dtype=torch.float32)
        else:
            confidence_gate = torch.as_tensor(initial_gate, device=device, dtype=torch.float32)
            if confidence_gate.ndim == 0:
                confidence_gate = confidence_gate.view(1, 1).expand(shape).clone()
            elif confidence_gate.ndim == 1:
                if confidence_gate.shape[0] != shape[0]:
                    raise ValueError(f"initial_gate [B] must match batch size {shape[0]}, got {tuple(confidence_gate.shape)}")
                confidence_gate = confidence_gate.view(shape[0], 1).expand(shape).clone()
            elif confidence_gate.ndim == 2:
                if tuple(confidence_gate.shape) != shape:
                    raise ValueError(f"initial_gate [B, N] must have shape {shape}, got {tuple(confidence_gate.shape)}")
                confidence_gate = confidence_gate.clone()
            else:
                raise ValueError(f"initial_gate must be scalar, [B], or [B, N], got {tuple(confidence_gate.shape)}")
            confidence_gate = torch.nan_to_num(confidence_gate, nan=1.0, posinf=1.0, neginf=1.0)
        gate_sum, gate_count = cls._stats_from_gate(confidence_gate)
        return cls(
            frame_ids=torch.full(shape, int(frame_id), device=device, dtype=torch.long),
            token_indices=token_indices.expand(shape).clone(),
            confidence_gate=confidence_gate,
            gate_sum=gate_sum,
            gate_count=gate_count,
        )

    @classmethod
    def for_frame_chunk(
        cls,
        batch_size: int,
        num_heads: int,
        tokens_per_frame: int,
        frame_ids,
        device: torch.device,
        initial_gate: Optional[torch.Tensor | float] = None,
    ) -> "KVConfidenceState":
        del num_heads
        frame_ids = torch.as_tensor(frame_ids, device=device, dtype=torch.long).reshape(-1)
        tokens_per_frame = int(tokens_per_frame)
        if tokens_per_frame <= 0:
            raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
        if frame_ids.numel() <= 0:
            raise ValueError("frame_ids must contain at least one frame")

        B = int(batch_size)
        S = int(frame_ids.numel())
        N = S * tokens_per_frame
        shape = (B, N)
        chunk_frame_ids = frame_ids.view(1, S, 1).expand(B, S, tokens_per_frame).reshape(shape).clone()
        token_indices = torch.arange(tokens_per_frame, device=device, dtype=torch.long).view(1, 1, tokens_per_frame)
        token_indices = token_indices.expand(B, S, tokens_per_frame).reshape(shape).clone()
        if initial_gate is None:
            confidence_gate = torch.ones(shape, device=device, dtype=torch.float32)
        else:
            confidence_gate = torch.as_tensor(initial_gate, device=device, dtype=torch.float32)
            if confidence_gate.ndim == 0:
                confidence_gate = confidence_gate.view(1, 1).expand(shape).clone()
            elif confidence_gate.ndim == 1:
                if confidence_gate.shape[0] != shape[0]:
                    raise ValueError(f"initial_gate [B] must match batch size {shape[0]}, got {tuple(confidence_gate.shape)}")
                confidence_gate = confidence_gate.view(shape[0], 1).expand(shape).clone()
            elif confidence_gate.ndim == 2:
                if tuple(confidence_gate.shape) != shape:
                    raise ValueError(f"initial_gate [B, N] must have shape {shape}, got {tuple(confidence_gate.shape)}")
                confidence_gate = confidence_gate.clone()
            else:
                raise ValueError(f"initial_gate must be scalar, [B], or [B, N], got {tuple(confidence_gate.shape)}")
            confidence_gate = torch.nan_to_num(confidence_gate, nan=1.0, posinf=1.0, neginf=1.0)
        gate_sum, gate_count = cls._stats_from_gate(confidence_gate)
        return cls(
            frame_ids=chunk_frame_ids,
            token_indices=token_indices,
            confidence_gate=confidence_gate,
            gate_sum=gate_sum,
            gate_count=gate_count,
        )

    def concat(self, other: "KVConfidenceState") -> "KVConfidenceState":
        other_gate_sum = other.gate_sum.to(self.gate_sum.device)
        other_gate_count = other.gate_count.to(self.gate_count.device)
        return KVConfidenceState(
            frame_ids=torch.cat([self.frame_ids, other.frame_ids.to(self.frame_ids.device)], dim=1),
            token_indices=torch.cat([self.token_indices, other.token_indices.to(self.token_indices.device)], dim=1),
            confidence_gate=torch.cat(
                [self.confidence_gate, other.confidence_gate.to(self.confidence_gate.device)],
                dim=1,
            ),
            gate_sum=self.gate_sum + other_gate_sum,
            gate_count=self.gate_count + other_gate_count,
        )

    def _head_shared_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.frame_ids.device, dtype=torch.long)
        if indices.dim() == 3:
            return indices[:, 0, :]
        if indices.dim() == 2:
            return indices
        raise ValueError(f"Expected gather indices [B, N] or [B, H, N], got {tuple(indices.shape)}")

    def gather(self, indices: torch.Tensor) -> "KVConfidenceState":
        indices = self._head_shared_indices(indices)
        confidence_gate = torch.gather(self.confidence_gate, 1, indices)
        gate_sum, gate_count = self._stats_from_gate(confidence_gate)
        return KVConfidenceState(
            frame_ids=torch.gather(self.frame_ids, 1, indices),
            token_indices=torch.gather(self.token_indices, 1, indices),
            confidence_gate=confidence_gate,
            gate_sum=gate_sum,
            gate_count=gate_count,
        )

    def slice(self, start: int, end: int) -> "KVConfidenceState":
        B, _ = self.frame_ids.shape
        indices = torch.arange(int(start), int(end), device=self.frame_ids.device, dtype=torch.long)
        indices = indices.view(1, -1).expand(B, -1)
        return self.gather(indices)

    def update_frame_gate(
        self,
        frame_id: int,
        token_confidence_gate: torch.Tensor,
    ) -> None:
        token_confidence_gate = token_confidence_gate.to(
            device=self.confidence_gate.device,
            dtype=torch.float32,
        )
        token_confidence_gate = torch.nan_to_num(token_confidence_gate, nan=1.0, posinf=1.0, neginf=1.0)
        B, N = self.frame_ids.shape
        if token_confidence_gate.dim() != 2 or token_confidence_gate.shape[0] != B:
            raise ValueError(
                "token_confidence_gate must have shape [B, tokens_per_frame], "
                f"got {tuple(token_confidence_gate.shape)} for batch {B}"
            )

        is_frame = self.frame_ids == int(frame_id)
        suffix_mask = is_frame.flip(1).cumprod(dim=1).flip(1).bool()

        token_ids = self.token_indices.long()
        valid = suffix_mask & (token_ids >= 0) & (token_ids < token_confidence_gate.shape[1])
        batch_ids = torch.arange(B, device=self.frame_ids.device, dtype=torch.long).view(B, 1).expand(B, N)

        old_batch_ids = batch_ids[suffix_mask]
        old_values = self.confidence_gate[suffix_mask]
        old_finite = torch.isfinite(old_values)
        old_sum = torch.zeros((B,), device=self.confidence_gate.device, dtype=self.confidence_gate.dtype)
        old_count = torch.zeros((B,), device=self.confidence_gate.device, dtype=torch.long)
        old_sum.scatter_add_(0, old_batch_ids[old_finite], old_values[old_finite])
        old_count.scatter_add_(0, old_batch_ids[old_finite], torch.ones_like(old_batch_ids[old_finite]))

        self.confidence_gate[suffix_mask] = 1.0
        valid_batch_ids = batch_ids[valid]
        valid_values = token_confidence_gate[valid_batch_ids, token_ids[valid]]
        self.confidence_gate[valid] = valid_values
        valid_delta = torch.zeros((B,), device=self.confidence_gate.device, dtype=self.confidence_gate.dtype)
        valid_delta.scatter_add_(0, valid_batch_ids, valid_values - 1.0)

        suffix_count = suffix_mask.sum(dim=1)
        new_sum = suffix_count.to(dtype=self.confidence_gate.dtype) + valid_delta
        self.gate_sum = self.gate_sum - old_sum + new_sum
        self.gate_count = self.gate_count - old_count + suffix_count


def make_token_confidence_gate(
    token_depth_confidence: torch.Tensor,
    token_point_confidence: Optional[torch.Tensor],
    *,
    floor: float,
    depth_alpha: float,
    point_beta: float,
    normalizer_k: float = 1.0,
    normalize_confidence: bool = True,
    preserve_prefix_tokens: int = 0,
    prefix_token_mode: str = "mean",
) -> torch.Tensor:
    if prefix_token_mode not in ("mean", "one"):
        raise ValueError(f"prefix_token_mode must be 'mean' or 'one', got {prefix_token_mode!r}")

    depth_conf = torch.nan_to_num(token_depth_confidence.float(), nan=1.0, posinf=1.0e6, neginf=0.0).clamp_min(0.0)
    if token_point_confidence is None:
        point_conf = torch.ones_like(depth_conf)
    else:
        point_conf = torch.nan_to_num(
            token_point_confidence.to(device=depth_conf.device, dtype=torch.float32),
            nan=1.0,
            posinf=1.0e6,
            neginf=0.0,
        ).clamp_min(0.0)

    prefix = max(int(preserve_prefix_tokens), 0)
    if normalize_confidence:
        normalizer_k = float(normalizer_k)
        if normalizer_k <= 0.0:
            raise ValueError(f"normalizer_k must be > 0, got {normalizer_k}")
        depth_conf = depth_conf / (depth_conf + normalizer_k)
        if token_point_confidence is not None:
            point_conf = point_conf / (point_conf + normalizer_k)

    if float(depth_alpha) != 1.0:
        depth_conf = depth_conf.pow(float(depth_alpha))
    if float(point_beta) != 1.0:
        point_conf = point_conf.pow(float(point_beta))
    floor = float(floor)
    gate = floor + (1.0 - floor) * depth_conf * point_conf

    prefix = min(prefix, gate.shape[1])
    if prefix > 0:
        if prefix_token_mode == "mean" and gate.shape[1] > prefix:
            patch_gate = gate[:, prefix:]
            finite = torch.isfinite(patch_gate)
            patch_sum = torch.where(finite, patch_gate, torch.zeros_like(patch_gate)).sum(dim=1)
            patch_count = finite.sum(dim=1).clamp_min(1)
            prefix_gate = patch_sum / patch_count.to(dtype=patch_sum.dtype)
        else:
            prefix_gate = torch.ones((gate.shape[0],), device=gate.device, dtype=gate.dtype)
        gate[:, :prefix] = prefix_gate.view(-1, 1)
    return gate


def pack_kv_cache(k: torch.Tensor, v: torch.Tensor, metadata=None, confidence_state: Optional[KVConfidenceState] = None):
    if confidence_state is not None:
        return k, v, metadata, confidence_state
    if metadata is not None:
        return k, v, metadata
    return k, v


def unpack_kv_cache(kv) -> Tuple[torch.Tensor, torch.Tensor, object, Optional[KVConfidenceState]]:
    if kv is None:
        raise ValueError("Cannot unpack a None KV cache")
    if len(kv) == 4:
        k, v, metadata, confidence_state = kv
        return k, v, metadata, confidence_state
    if len(kv) == 3:
        k, v, metadata = kv
        return k, v, metadata, None
    if len(kv) == 2:
        k, v = kv
        return k, v, None, None
    raise ValueError(f"Expected KV tuple of length 2, 3, or 4, got {len(kv)}")


def sample_token_confidence(
    confidence: Optional[torch.Tensor],
    image_hw: Tuple[int, int],
    tokens_per_frame: int,
    patch_start_idx: int,
    patch_size: int,
    *,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if confidence is None:
        if batch_size is None or device is None:
            raise ValueError("batch_size and device are required when confidence is None")
        return torch.ones(
            (int(batch_size), int(tokens_per_frame)),
            device=device,
            dtype=torch.float32,
        )

    confidence = confidence.detach()
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim == 4 and confidence.shape[1] == 1:
        confidence = confidence[:, 0]
    if confidence.ndim != 3:
        raise ValueError(f"Expected confidence map [B,H,W], got {tuple(confidence.shape)}")

    B = int(confidence.shape[0])
    H_img, W_img = int(image_hw[0]), int(image_hw[1])
    tokens_per_frame = int(tokens_per_frame)
    patch_start_idx = int(patch_start_idx)
    patch_size = int(patch_size)
    token_confidence = torch.ones(
        (B, tokens_per_frame),
        device=confidence.device,
        dtype=torch.float32,
    )

    patch_h = H_img // patch_size
    patch_w = W_img // patch_size
    patch_tokens = min(max(tokens_per_frame - patch_start_idx, 0), patch_h * patch_w)
    if patch_tokens <= 0:
        return token_confidence

    pooled = F.adaptive_avg_pool2d(confidence.float().unsqueeze(1), (patch_h, patch_w))
    pooled = pooled[:, 0].reshape(B, patch_h * patch_w)
    token_confidence[:, patch_start_idx : patch_start_idx + patch_tokens] = pooled[:, :patch_tokens]
    return token_confidence
