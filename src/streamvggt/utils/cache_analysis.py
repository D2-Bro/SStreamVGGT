"""Lightweight cache snapshot dumping for eviction analysis experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch


def parse_index_filter(spec: Optional[str]) -> Optional[set[int]]:
    """Parse comma/range filters such as ``"0,3,8-10"``.

    ``None``, ``""``, and ``"all"`` mean no filtering.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec or spec.lower() == "all":
        return None

    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return values


@dataclass
class CacheAnalysisConfig:
    """Opt-in controls for dumping per-head old-key-cache snapshots."""

    output_dir: str
    layers: Optional[set[int]] = None
    heads: Optional[set[int]] = None
    steps: Optional[set[int]] = None
    max_snapshots: Optional[int] = None

    def __post_init__(self) -> None:
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self._num_snapshots = 0

    @classmethod
    def from_cli(
        cls,
        output_dir: Optional[str],
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        steps: Optional[str] = None,
        max_snapshots: Optional[int] = None,
    ) -> Optional["CacheAnalysisConfig"]:
        if not output_dir:
            return None
        return cls(
            output_dir=output_dir,
            layers=parse_index_filter(layers),
            heads=parse_index_filter(heads),
            steps=parse_index_filter(steps),
            max_snapshots=max_snapshots,
        )

    def should_dump(self, layer_id: int, head_id: int, step_idx: int) -> bool:
        if self.max_snapshots is not None and self._num_snapshots >= self.max_snapshots:
            return False
        if self.layers is not None and layer_id not in self.layers:
            return False
        if self.heads is not None and head_id not in self.heads:
            return False
        if self.steps is not None and step_idx not in self.steps:
            return False
        return True

    def record_dump(self) -> None:
        self._num_snapshots += 1


def dump_eviction_snapshot(
    config: CacheAnalysisConfig,
    *,
    k_before: torch.Tensor,
    scores: torch.Tensor,
    kept_candidate_indices: torch.Tensor,
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    tokens_per_frame: Optional[int] = None,
) -> None:
    """Save selected per-head cache snapshots immediately before eviction.

    Args:
        k_before: Key cache before eviction, shaped ``[B, H, N, D]``.
        scores: Existing mean-similarity eviction scores for candidate tokens,
            shaped ``[B, H, N - num_anchor_tokens]``. Higher scores are closer
            to the per-head candidate mean; the current policy retains the
            lowest-scoring candidates and evicts the remaining candidates.
        kept_candidate_indices: Candidate-local indices retained by the
            current policy, shaped ``[B, H, K]``.
    """
    B, H, N, _ = k_before.shape
    candidate_count = N - num_anchor_tokens
    device = k_before.device
    all_candidate_indices = torch.arange(candidate_count, device=device)

    for batch_id in range(B):
        for head_id in range(H):
            if not config.should_dump(layer_id, head_id, step_idx):
                continue

            kept_local = kept_candidate_indices[batch_id, head_id].detach()
            evict_mask = torch.ones(candidate_count, dtype=torch.bool, device=device)
            evict_mask[kept_local] = False
            evicted_local = all_candidate_indices[evict_mask]

            kept_global = kept_local + num_anchor_tokens
            evicted_global = evicted_local + num_anchor_tokens

            token_index = torch.arange(N, dtype=torch.long)
            token_frame_index = None
            provenance_note = "not_tracked_by_cache"
            if tokens_per_frame is not None and tokens_per_frame > 0:
                token_frame_index = token_index // int(tokens_per_frame)
                provenance_note = "approximate_from_current_cache_order_only"

            payload = {
                "old_key": k_before[batch_id, head_id].detach().to("cpu", torch.float32),
                "mean_scores": scores[batch_id, head_id].detach().to("cpu", torch.float32),
                "kept_candidate_indices": kept_local.detach().cpu().long(),
                "kept_token_indices": kept_global.detach().cpu().long(),
                "evicted_candidate_indices": evicted_local.detach().cpu().long(),
                "evicted_token_indices": evicted_global.detach().cpu().long(),
                "anchor_token_indices": torch.arange(num_anchor_tokens, dtype=torch.long),
                "token_indices": token_index,
                "token_frame_indices": token_frame_index,
                "meta": {
                    "layer_id": int(layer_id),
                    "head_id": int(head_id),
                    "batch_id": int(batch_id),
                    "step_idx": int(step_idx),
                    "cache_size": int(N),
                    "cache_budget": int(cache_budget),
                    "num_anchor_tokens": int(num_anchor_tokens),
                    "candidate_count": int(candidate_count),
                    "tokens_per_frame": None if tokens_per_frame is None else int(tokens_per_frame),
                    "provenance": provenance_note,
                    "score_definition": "dot(normalized_candidate_key, mean(normalized_candidate_keys))",
                    "policy": "anchors preserved; candidates with lowest mean_scores are retained",
                },
            }

            stem = f"step{step_idx:06d}_layer{layer_id:02d}_head{head_id:02d}_batch{batch_id:02d}"
            torch.save(payload, os.path.join(config.output_dir, f"{stem}.pt"), pickle_protocol=4)
            with open(os.path.join(config.output_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(payload["meta"], f, indent=2)
            config.record_dump()
