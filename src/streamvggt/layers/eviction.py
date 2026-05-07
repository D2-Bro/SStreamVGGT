"""Head-wise KV cache eviction policies for streaming attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

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
    ) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Unknown eviction policy '{policy}'. Valid policies: {self.VALID_POLICIES}")
        if leverage_sketch_dim is not None and leverage_sketch_dim < 0:
            raise ValueError(f"leverage_sketch_dim must be >= 0 or None, got {leverage_sketch_dim}")
        self.policy = policy
        self.debug = debug
        self.leverage_sketch_dim = leverage_sketch_dim
        self._leverage_right_sketch_cache = {}

    def select(
        self,
        k: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        *,
        need_summary: bool = False,
        layer_id: Optional[int] = None,
        step_idx: Optional[int] = None,
    ) -> EvictionResult:
        """Select candidate-local indices to retain.

        Args:
            k: Key cache shaped ``[B, H, N, D]``.
            cache_budget: Final number of tokens to retain.
            num_anchor_tokens: Initial tokens that are always preserved.

        Returns:
            EvictionResult with candidate-local kept indices shaped
            ``[B, H, cache_budget - num_anchor_tokens]``.
        """
        B, H, N, _ = k.shape
        num_candidates = N - num_anchor_tokens
        num_to_keep = cache_budget - num_anchor_tokens
        if num_to_keep < 0:
            raise ValueError(
                f"cache_budget ({cache_budget}) must be >= num_anchor_tokens ({num_anchor_tokens})"
            )
        if num_to_keep > num_candidates:
            raise ValueError(f"Cannot keep {num_to_keep} candidates from {num_candidates} candidates")

        candidate_k = k[:, :, num_anchor_tokens:, :]
        need_mean_scores = self.policy in ("mean", "baseline_mean") or need_summary or self.debug
        mean_scores = self._mean_scores(candidate_k) if need_mean_scores else None

        if self.policy in ("mean", "baseline_mean"):
            policy_scores = mean_scores
            kept = self._keep_lowest_scores(policy_scores, num_to_keep)
        elif self.policy == "svd_leverage":
            policy_scores = self._svd_leverage_scores(candidate_k)
            kept = self._keep_highest_scores(policy_scores, num_to_keep)
        else:
            raise AssertionError(f"Unhandled eviction policy: {self.policy}")

        summary_score = mean_scores.mean().item() if need_summary and mean_scores is not None else 0.0
        if self.debug:
            sketch_label = (
                "exact"
                if self.leverage_sketch_dim in (None, 0)
                else str(self.leverage_sketch_dim)
            )
            print(
                f"[EvictionManager] policy={self.policy} layer={layer_id} step={step_idx} "
                f"cache={N} budget={cache_budget} keep_candidates={num_to_keep} "
                f"scores={tuple(policy_scores.shape)} leverage_sketch_dim={sketch_label}"
            )
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

        with torch.cuda.amp.autocast(enabled=False):
            mat = candidate_k.to(dtype=torch.float32)
            sketch_dim = self.leverage_sketch_dim
            if sketch_dim in (None, 0):
                leverage_matrix = mat
            else:
                sketch_dim = min(int(sketch_dim), int(D), int(N))
                omega = self._get_leverage_right_sketch(
                    D,
                    sketch_dim,
                    device=mat.device,
                )
                leverage_matrix = mat @ omega
            try:
                q, r = torch.linalg.qr(leverage_matrix, mode="reduced")
            except RuntimeError:
                return mat.square().sum(dim=-1)

        diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1))
        tol = diag.max(dim=-1, keepdim=True).values * 1e-6
        active = (diag > tol).to(dtype=q.dtype)
        scores_sq = (q.square() * active.unsqueeze(-2)).sum(dim=-1)
        return scores_sq
