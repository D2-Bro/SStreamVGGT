"""Head-wise KV cache eviction policies for streaming attention."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class EvictionResult:
    """Indices and scores produced by a head-wise eviction policy."""

    kept_candidate_indices: torch.Tensor
    policy_scores: torch.Tensor
    mean_scores: torch.Tensor
    summary_score: float


class EvictionManager:
    """Dispatches head-wise cache eviction policies."""

    VALID_POLICIES = ("mean", "baseline_mean", "svd_leverage")

    def __init__(
        self,
        policy: str = "mean",
        debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
    ) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Unknown eviction policy '{policy}'. Valid policies: {self.VALID_POLICIES}")
        if leverage_sketch_dim is not None and leverage_sketch_dim < 0:
            raise ValueError(f"leverage_sketch_dim must be >= 0 or None, got {leverage_sketch_dim}")
        if leverage_head_mean_dim < 1:
            raise ValueError(f"leverage_head_mean_dim must be >= 1, got {leverage_head_mean_dim}")
        if leverage_granularity not in ("head", "layer"):
            raise ValueError("leverage_granularity must be 'head' or 'layer', got " f"{leverage_granularity!r}")
        if leverage_feature not in ("key", "key_value"):
            raise ValueError("leverage_feature must be 'key' or 'key_value', got " f"{leverage_feature!r}")
        if leverage_projection not in ("random", "head_mean"):
            raise ValueError(
                "leverage_projection must be 'random' or 'head_mean', got "
                f"{leverage_projection!r}"
            )
        if leverage_projection == "head_mean" and leverage_granularity != "layer":
            raise ValueError("leverage_projection='head_mean' requires leverage_granularity='layer'")
        if leverage_projection == "head_mean" and leverage_feature != "key":
            raise ValueError("leverage_projection='head_mean' requires leverage_feature='key'")
        self.policy = policy
        self.debug = debug
        self.leverage_sketch_dim = leverage_sketch_dim
        self.leverage_granularity = leverage_granularity
        self.leverage_feature = leverage_feature
        self.leverage_projection = leverage_projection
        self.leverage_head_mean_dim = int(leverage_head_mean_dim)
        self._leverage_right_sketch_cache = {}
        self._last_leverage_profile: Dict[str, float] = {}
        self._last_layer_feature_shape: Optional[tuple[int, int]] = None

    def select(
        self,
        k: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        *,
        v: Optional[torch.Tensor] = None,
        need_summary: bool = False,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
        current_frame_idx: Optional[int] = None,
        protect_recent_frames: int = 0,
        candidate_frame_ids: Optional[torch.Tensor] = None,
    ) -> EvictionResult:
        """Select candidate-local indices to retain.

        Args:
            k: Key cache shaped ``[B, H, N, D]``.
            cache_budget: Final number of tokens to retain.
            num_anchor_tokens: Initial tokens that are always preserved.

        Returns:
            EvictionResult with candidate-local kept indices shaped
            ``[B, H, K]``. ``K`` may exceed ``cache_budget - num_anchor_tokens``
            when recent-frame protection leaves too few evictable candidates.
        """
        B, H, N, _ = k.shape
        if protect_recent_frames < 0:
            raise ValueError(f"protect_recent_frames must be >= 0, got {protect_recent_frames}")
        num_candidates = N - num_anchor_tokens
        num_to_keep = cache_budget - num_anchor_tokens
        if num_to_keep < 0:
            raise ValueError(
                f"cache_budget ({cache_budget}) must be >= num_anchor_tokens ({num_anchor_tokens})"
            )
        if num_to_keep > num_candidates:
            raise ValueError(f"Cannot keep {num_to_keep} candidates from {num_candidates} candidates")

        candidate_k = k[:, :, num_anchor_tokens:, :]
        candidate_v = v[:, :, num_anchor_tokens:, :] if v is not None else None
        self._last_leverage_profile = {}
        self._last_layer_feature_shape = None
        need_mean_scores = self.policy in ("mean", "baseline_mean") or need_summary or self.debug
        mean_scores = self._mean_scores(candidate_k) if need_mean_scores else None

        if self.policy in ("mean", "baseline_mean"):
            policy_scores = mean_scores
            if protect_recent_frames > 0:
                kept, protection_debug = self._keep_with_recent_protection(
                    policy_scores,
                    num_to_keep,
                    candidate_frame_ids,
                    current_frame_idx,
                    protect_recent_frames,
                    evict_highest=True,
                    shared_across_heads=False,
                )
            else:
                kept = self._keep_lowest_scores(policy_scores, num_to_keep)
                protection_debug = None
        elif self.policy == "svd_leverage":
            if self.leverage_granularity == "head":
                policy_scores = self._svd_leverage_scores(candidate_k)
                if protect_recent_frames > 0:
                    kept, protection_debug = self._keep_with_recent_protection(
                        policy_scores,
                        num_to_keep,
                        candidate_frame_ids,
                        current_frame_idx,
                        protect_recent_frames,
                        evict_highest=False,
                        shared_across_heads=False,
                    )
                else:
                    kept = self._keep_highest_scores(policy_scores, num_to_keep)
                    protection_debug = None
            else:
                policy_scores = self._layer_svd_leverage_scores(candidate_k, candidate_v)
                if protect_recent_frames > 0:
                    kept, protection_debug = self._keep_with_recent_protection(
                        policy_scores,
                        num_to_keep,
                        candidate_frame_ids,
                        current_frame_idx,
                        protect_recent_frames,
                        evict_highest=False,
                        shared_across_heads=True,
                        num_heads=H,
                    )
                else:
                    kept = self._keep_highest_scores(policy_scores.unsqueeze(1), num_to_keep)
                    kept = kept.expand(B, H, num_to_keep)
                    protection_debug = None
        else:
            raise AssertionError(f"Unhandled eviction policy: {self.policy}")

        summary_score = mean_scores.mean().item() if need_summary and mean_scores is not None else 0.0
        if self.debug:
            sketch_label = (
                "exact"
                if self.leverage_sketch_dim in (None, 0)
                else str(self.leverage_sketch_dim)
            )
            requested_evicted = num_candidates - num_to_keep
            actual_evicted = num_candidates - int(kept.shape[-1])
            feature_dim = D
            if self.policy == "svd_leverage" and self.leverage_granularity == "layer":
                if self.leverage_projection == "head_mean":
                    feature_dim = H * self.leverage_head_mean_dim
                else:
                    feature_dim = H * D * (2 if self.leverage_feature == "key_value" else 1)
            msg = (
                f"[EvictionManager] policy={self.policy} layer={layer_id} step={step_idx} "
                f"cache={N} budget={cache_budget} keep_candidates={kept.shape[-1]} "
                f"requested_evicted={requested_evicted} evicted={actual_evicted} "
                f"scores={tuple(policy_scores.shape)}"
            )
            if protection_debug is not None:
                msg += (
                    f" current_frame_idx={protection_debug['current_frame_idx']} "
                    f"protect_recent_frames={protection_debug['protect_recent_frames']} "
                    f"protected_tokens={protection_debug['protected_tokens']} "
                    f"candidate_tokens={protection_debug['candidate_tokens']} "
                    f"limited_by_protection={protection_debug['limited_by_protection']}"
                )
            if self.policy == "svd_leverage":
                msg += (
                    f" leverage_sketch_dim={sketch_label} "
                    f"leverage_granularity={self.leverage_granularity} leverage_feature={self.leverage_feature} "
                    f"leverage_projection={self.leverage_projection} "
                    f"leverage_head_mean_dim={self.leverage_head_mean_dim} "
                    f"num_heads={H} num_tokens={num_candidates} head_dim={D} feature_dim={feature_dim}"
                )
            print(msg)
            if protection_debug is not None and protection_debug["limited_by_protection"]:
                print(
                    "[EvictionManager] recent-frame protection limited eviction; "
                    "cache may temporarily exceed budget"
                )
            if self.policy == "svd_leverage" and self.leverage_granularity == "layer":
                print(f"[EvictionManager] layer-wise SVD leverage: X shape={self._last_layer_feature_shape}")
            if self.policy == "svd_leverage" and self._last_leverage_profile:
                profile = " ".join(f"{name}={value * 1000.0:.3f}ms" for name, value in self._last_leverage_profile.items())
                print(f"[EvictionManager] svd_leverage_profile {profile}")
        return EvictionResult(
            kept_candidate_indices=kept,
            policy_scores=policy_scores,
            mean_scores=mean_scores
            if mean_scores is not None
            else torch.empty(B, H, 0, device=k.device, dtype=torch.float32),
            summary_score=summary_score,
        )

    @staticmethod
    def _mean_scores(candidate_k: torch.Tensor) -> torch.Tensor:
        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)
        return torch.sum(candidate_k_norm * mean_vector, dim=-1)

    @staticmethod
    def _keep_lowest_scores(scores: torch.Tensor, num_to_keep: int) -> torch.Tensor:
        _, kept = torch.topk(-scores, k=num_to_keep, dim=-1)
        return kept.sort(dim=-1).values

    @staticmethod
    def _keep_highest_scores(scores: torch.Tensor, num_to_keep: int) -> torch.Tensor:
        _, kept = torch.topk(scores, k=num_to_keep, dim=-1)
        return kept.sort(dim=-1).values

    def _keep_with_recent_protection(
        self,
        scores: torch.Tensor,
        num_to_keep: int,
        candidate_frame_ids: Optional[torch.Tensor],
        current_frame_idx: Optional[int],
        protect_recent_frames: int,
        *,
        evict_highest: bool,
        shared_across_heads: bool,
        num_heads: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        """Keep tokens after excluding recent frames only from eviction candidates.

        SVD/QR leverage scores are computed before this method is called; this
        mask only limits which scored tokens may be selected for eviction.
        """
        if current_frame_idx is None:
            raise ValueError("current_frame_idx is required when protect_recent_frames > 0")
        if candidate_frame_ids is None:
            raise ValueError("candidate_frame_ids is required when protect_recent_frames > 0")

        threshold = int(current_frame_idx) - int(protect_recent_frames) + 1
        if shared_across_heads:
            if scores.ndim != 2:
                raise ValueError(f"Expected layer-wise scores [B, N], got {tuple(scores.shape)}")
            if candidate_frame_ids.ndim != 3:
                raise ValueError(
                    "Expected candidate_frame_ids [B, H, N] for layer-wise protection, "
                    f"got {tuple(candidate_frame_ids.shape)}"
                )
            B, _, N = candidate_frame_ids.shape
            H = int(num_heads) if num_heads is not None else int(candidate_frame_ids.shape[1])
            frame_ids = candidate_frame_ids.to(device=scores.device, dtype=torch.long)
            evictable_mask = ((frame_ids < 0) | (frame_ids < threshold)).all(dim=1)
            actual_evict = min(N - int(num_to_keep), int(evictable_mask.sum(dim=-1).min().item()))
            kept_2d = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest)
            kept = kept_2d.unsqueeze(1).expand(B, H, kept_2d.shape[-1])
            protected_tokens = int((~evictable_mask).sum().item())
            candidate_tokens = int(evictable_mask.sum().item())
        else:
            if scores.ndim != 3:
                raise ValueError(f"Expected head-wise scores [B, H, N], got {tuple(scores.shape)}")
            if candidate_frame_ids.shape != scores.shape:
                raise ValueError(
                    "candidate_frame_ids must match head-wise score shape, "
                    f"got {tuple(candidate_frame_ids.shape)} vs {tuple(scores.shape)}"
                )
            _, _, N = scores.shape
            frame_ids = candidate_frame_ids.to(device=scores.device, dtype=torch.long)
            evictable_mask = (frame_ids < 0) | (frame_ids < threshold)
            actual_evict = min(N - int(num_to_keep), int(evictable_mask.sum(dim=-1).min().item()))
            kept = self._keep_after_eviction(scores, evictable_mask, actual_evict, evict_highest)
            protected_tokens = int((~evictable_mask).sum().item())
            candidate_tokens = int(evictable_mask.sum().item())

        requested_evict = scores.shape[-1] - int(num_to_keep)
        debug = {
            "current_frame_idx": int(current_frame_idx),
            "protect_recent_frames": int(protect_recent_frames),
            "protected_tokens": protected_tokens,
            "candidate_tokens": candidate_tokens,
            "requested_eviction_count": int(requested_evict),
            "actual_eviction_count": int(actual_evict),
            "limited_by_protection": int(actual_evict < requested_evict),
        }
        return kept, debug

    @staticmethod
    def _keep_after_eviction(
        scores: torch.Tensor,
        evictable_mask: torch.Tensor,
        num_to_evict: int,
        evict_highest: bool,
    ) -> torch.Tensor:
        leading_shape = scores.shape[:-1]
        num_candidates = int(scores.shape[-1])
        keep_count = num_candidates - int(num_to_evict)
        kept_flat = torch.empty(
            int(math.prod(leading_shape)),
            keep_count,
            device=scores.device,
            dtype=torch.long,
        )
        all_indices = torch.arange(num_candidates, device=scores.device, dtype=torch.long)
        scores_flat = scores.reshape(-1, num_candidates)
        mask_flat = evictable_mask.reshape(-1, num_candidates)

        for row_idx in range(scores_flat.shape[0]):
            evictable = all_indices[mask_flat[row_idx]]
            if num_to_evict > 0:
                row_scores = scores_flat[row_idx][evictable]
                selection_scores = row_scores if evict_highest else -row_scores
                _, local_evict = torch.topk(selection_scores, k=num_to_evict, dim=-1)
                evicted = evictable[local_evict]
                keep_mask = torch.ones(num_candidates, device=scores.device, dtype=torch.bool)
                keep_mask[evicted] = False
                row_kept = all_indices[keep_mask]
            else:
                row_kept = all_indices
            kept_flat[row_idx] = row_kept.sort(dim=-1).values
        return kept_flat.reshape(*leading_shape, keep_count)

    def _get_leverage_right_sketch(
        self,
        embed_dim: int,
        sketch_dim: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        key = (str(device), embed_dim, sketch_dim)
        omega = self._leverage_right_sketch_cache.get(key)
        if omega is not None:
            return omega

        generator = torch.Generator()
        generator.manual_seed(0)
        omega = torch.randn(
            embed_dim,
            sketch_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(device=device)
        omega = omega / math.sqrt(float(sketch_dim))
        self._leverage_right_sketch_cache[key] = omega
        return omega

    @staticmethod
    def _sync_for_timing(tensor: torch.Tensor) -> None:
        if tensor.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(tensor.device)

    def compute_svd_leverage_scores(
        self,
        x: torch.Tensor,
        sketch_dim: Optional[int] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute row leverage scores for a 2D token-feature matrix."""
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D token-feature matrix, got shape {tuple(x.shape)}")
        num_tokens, feature_dim = x.shape
        if num_tokens <= 0:
            return torch.empty(0, device=x.device, dtype=torch.float32)
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0 for SVD leverage, got {feature_dim}")

        active_sketch_dim = self.leverage_sketch_dim if sketch_dim is None else sketch_dim
        profile: Dict[str, float] = {}
        do_profile = self.debug

        if do_profile:
            self._sync_for_timing(x)
        total_start = time.perf_counter() if do_profile else 0.0
        with torch.cuda.amp.autocast(enabled=False):
            mat = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if active_sketch_dim in (None, 0):
                leverage_matrix = mat
                if do_profile:
                    profile["sketch"] = 0.0
            else:
                sketch_start = time.perf_counter() if do_profile else 0.0
                active_sketch_dim = min(int(active_sketch_dim), int(feature_dim), int(num_tokens))
                omega = self._get_leverage_right_sketch(
                    feature_dim,
                    active_sketch_dim,
                    device=mat.device,
                )
                leverage_matrix = mat @ omega
                if do_profile:
                    self._sync_for_timing(leverage_matrix)
                    profile["sketch"] = time.perf_counter() - sketch_start
            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = mat.square().sum(dim=-1)
                if do_profile:
                    self._sync_for_timing(scores)
                    profile["total"] = time.perf_counter() - total_start
                    self._last_leverage_profile = profile
                return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        max_diag = diag.max().clamp_min(float(eps))
        active = (diag > max_diag * float(eps)).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(0)).sum(dim=-1)
        scores_sq = torch.nan_to_num(scores_sq, nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["total"] = time.perf_counter() - total_start
            self._last_leverage_profile = profile
        return scores_sq

    def _svd_leverage_scores(self, candidate_k: torch.Tensor) -> torch.Tensor:
        """Compute row leverage scores per batch/head.

        With ``leverage_sketch_dim > 0``, each head matrix ``K_h`` shaped
        ``[tokens, features]`` is projected with a fixed random right sketch
        ``Omega`` before QR, and row norms in ``Q`` are used as approximate
        leverage scores. With ``leverage_sketch_dim`` set to ``0`` or ``None``,
        QR is applied to the original key matrix for exact row leverage scores
        in the full column space. The output stays ``[B, H, N]`` so each head
        evicts independently.
        """
        B, H, N, D = candidate_k.shape
        if N <= 0:
            return torch.empty(B, H, 0, device=candidate_k.device, dtype=torch.float32)

        profile: Dict[str, float] = {}
        do_profile = self.debug
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0
        with torch.cuda.amp.autocast(enabled=False):
            mat = candidate_k.to(dtype=torch.float32)
            sketch_dim = self.leverage_sketch_dim
            if sketch_dim in (None, 0):
                leverage_matrix = mat
                if do_profile:
                    profile["sketch"] = 0.0
            else:
                sketch_start = time.perf_counter() if do_profile else 0.0
                sketch_dim = min(int(sketch_dim), int(D), int(N))
                omega = self._get_leverage_right_sketch(
                    D,
                    sketch_dim,
                    device=mat.device,
                )
                leverage_matrix = mat @ omega
                if do_profile:
                    self._sync_for_timing(leverage_matrix)
                    profile["sketch"] = time.perf_counter() - sketch_start
            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = mat.square().sum(dim=-1)
                if do_profile:
                    self._sync_for_timing(scores)
                    profile["total"] = time.perf_counter() - total_start
                    self._last_leverage_profile = profile
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        tol = diag.max(dim=-1, keepdim=True).values * 1e-6
        active = (diag > tol).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(-2)).sum(dim=-1)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["total"] = time.perf_counter() - total_start
            self._last_leverage_profile = profile
        return scores_sq

    def _layer_svd_leverage_scores(
        self,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute one leverage-score vector per batch by concatenating heads."""
        B, H, N, D = candidate_k.shape
        if N <= 0:
            return torch.empty(B, 0, device=candidate_k.device, dtype=torch.float32)
        if D <= 0:
            raise ValueError(f"head_dim must be > 0 for layer-wise SVD leverage, got {D}")
        if self.leverage_projection == "head_mean":
            return self._layer_svd_leverage_scores_head_mean(candidate_k)
        feature_dim = H * D * (2 if self.leverage_feature == "key_value" else 1)
        self._last_layer_feature_shape = (int(N), int(feature_dim))
        if self.leverage_feature == "key_value":
            if candidate_v is None:
                raise ValueError("leverage_feature='key_value' requires value cache tensor")
            if candidate_v.shape != candidate_k.shape:
                raise ValueError(
                    "candidate_v must match candidate_k for key_value leverage, "
                    f"got {tuple(candidate_v.shape)} vs {tuple(candidate_k.shape)}"
                )

        if self.leverage_sketch_dim not in (None, 0):
            return self._layer_svd_leverage_scores_sketched(candidate_k, candidate_v, feature_dim)

        scores = []
        aggregate_profile: Dict[str, float] = {"feature": 0.0, "sketch": 0.0, "qr": 0.0, "scoring": 0.0, "total": 0.0}
        for batch_idx in range(B):
            feature_start = time.perf_counter() if self.debug else 0.0
            x_key = candidate_k[batch_idx].transpose(0, 1).reshape(N, H * D)
            if self.leverage_feature == "key_value":
                assert candidate_v is not None
                x_value = candidate_v[batch_idx].transpose(0, 1).reshape(N, H * D)
                x_layer = torch.cat([x_key, x_value], dim=-1)
            else:
                x_layer = x_key
            if self.debug:
                self._sync_for_timing(x_layer)
            feature_time = time.perf_counter() - feature_start if self.debug else 0.0
            score = self.compute_svd_leverage_scores(x_layer, self.leverage_sketch_dim)
            scores.append(score)
            if self.debug:
                aggregate_profile["feature"] += feature_time
                for name, value in self._last_leverage_profile.items():
                    aggregate_profile[name] = aggregate_profile.get(name, 0.0) + value
        if self.debug:
            self._last_leverage_profile = aggregate_profile
        return torch.stack(scores, dim=0)

    def _layer_svd_leverage_scores_head_mean(self, candidate_k: torch.Tensor) -> torch.Tensor:
        """Layer-wise leverage from deterministic per-head mean features."""
        B, H, N, D = candidate_k.shape
        if self.leverage_head_mean_dim > D:
            raise ValueError(
                "leverage_head_mean_dim must be <= head_dim for head_mean projection, "
                f"got {self.leverage_head_mean_dim} > {D}"
            )
        feature_dim = H * self.leverage_head_mean_dim
        self._last_layer_feature_shape = (int(N), int(feature_dim))

        profile: Dict[str, float] = {}
        do_profile = self.debug
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0

        with torch.cuda.amp.autocast(enabled=False):
            feature_start = time.perf_counter() if do_profile else 0.0
            mat_k = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            head_chunks = torch.tensor_split(mat_k, self.leverage_head_mean_dim, dim=-1)
            head_features = torch.stack([chunk.mean(dim=-1) for chunk in head_chunks], dim=-1)
            leverage_matrix = head_features.permute(0, 2, 1, 3).reshape(B, N, feature_dim).contiguous()
            if do_profile:
                self._sync_for_timing(leverage_matrix)
                profile["feature"] = time.perf_counter() - feature_start
                profile["sketch"] = 0.0

            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = leverage_matrix.square().sum(dim=-1)
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                if do_profile:
                    self._sync_for_timing(scores)
                    profile["total"] = time.perf_counter() - total_start
                    self._last_leverage_profile = profile
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        max_diag = diag.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        active = (diag > max_diag * 1e-6).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(1)).sum(dim=-1)
        scores_sq = torch.nan_to_num(scores_sq, nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["total"] = time.perf_counter() - total_start
            self._last_leverage_profile = profile
        return scores_sq

    def _layer_svd_leverage_scores_sketched(
        self,
        candidate_k: torch.Tensor,
        candidate_v: Optional[torch.Tensor],
        feature_dim: int,
    ) -> torch.Tensor:
        """Layer-wise sketched leverage without materializing ``[B, N, H * D]``."""
        B, H, N, D = candidate_k.shape
        sketch_dim = min(int(self.leverage_sketch_dim), int(feature_dim), int(N))
        if sketch_dim <= 0:
            return torch.empty(B, N, device=candidate_k.device, dtype=torch.float32)

        profile: Dict[str, float] = {}
        do_profile = self.debug
        if do_profile:
            self._sync_for_timing(candidate_k)
        total_start = time.perf_counter() if do_profile else 0.0

        with torch.cuda.amp.autocast(enabled=False):
            mat_k = torch.nan_to_num(candidate_k.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mat_v = (
                torch.nan_to_num(candidate_v.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if self.leverage_feature == "key_value" and candidate_v is not None
                else None
            )

            sketch_start = time.perf_counter() if do_profile else 0.0
            omega = self._get_leverage_right_sketch(
                feature_dim,
                sketch_dim,
                device=mat_k.device,
            )
            omega_key = omega[: H * D].view(H, D, sketch_dim)
            leverage_matrix = torch.einsum("bhnd,hds->bns", mat_k, omega_key)
            if mat_v is not None:
                omega_value = omega[H * D :].view(H, D, sketch_dim)
                leverage_matrix = leverage_matrix + torch.einsum("bhnd,hds->bns", mat_v, omega_value)
            if do_profile:
                self._sync_for_timing(leverage_matrix)
                profile["feature"] = 0.0
                profile["sketch"] = time.perf_counter() - sketch_start

            try:
                qr_start = time.perf_counter() if do_profile else 0.0
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
                if do_profile:
                    self._sync_for_timing(q)
                    profile["qr"] = time.perf_counter() - qr_start
            except RuntimeError:
                scores = mat_k.square().sum(dim=(1, 3))
                if mat_v is not None:
                    scores = scores + mat_v.square().sum(dim=(1, 3))
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                if do_profile:
                    self._sync_for_timing(scores)
                    profile["total"] = time.perf_counter() - total_start
                    self._last_leverage_profile = profile
                return scores

        score_start = time.perf_counter() if do_profile else 0.0
        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        max_diag = diag.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        active = (diag > max_diag * 1e-6).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(1)).sum(dim=-1)
        scores_sq = torch.nan_to_num(scores_sq, nan=0.0, posinf=0.0, neginf=0.0)
        if do_profile:
            self._sync_for_timing(scores_sq)
            profile["scoring"] = time.perf_counter() - score_start
            profile["total"] = time.perf_counter() - total_start
            self._last_leverage_profile = profile
        return scores_sq
