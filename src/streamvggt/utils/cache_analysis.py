"""Lightweight cache snapshot dumping for eviction analysis experiments."""

from __future__ import annotations

import atexit
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F


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



@dataclass
class LeverageScoreHistogramConfig:
    """Opt-in cumulative histograms for raw SVD leverage policy scores."""

    output_dir: str
    bins: int = 100
    min_value: float = 0.0
    max_value: float = 1.0
    layers: Optional[set[int]] = None
    steps: Optional[set[int]] = None

    def __post_init__(self) -> None:
        if int(self.bins) < 1:
            raise ValueError(f"bins must be >= 1, got {self.bins}")
        if not math.isfinite(float(self.min_value)) or not math.isfinite(float(self.max_value)):
            raise ValueError("histogram min/max must be finite")
        if float(self.max_value) <= float(self.min_value):
            raise ValueError(
                f"histogram max must be greater than min, got min={self.min_value}, max={self.max_value}"
            )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.bins = int(self.bins)
        self.min_value = float(self.min_value)
        self.max_value = float(self.max_value)
        self.edges = torch.linspace(self.min_value, self.max_value, self.bins + 1, dtype=torch.float64)
        self._counts: dict[int, torch.Tensor] = {}
        self._summary: dict[int, dict[str, float | int | None]] = {}
        self._flushed = False
        atexit.register(self.flush)

    @classmethod
    def from_cli(
        cls,
        output_dir: Optional[str],
        bins: int = 100,
        min_value: float = 0.0,
        max_value: float = 1.0,
        layers: Optional[str] = None,
        steps: Optional[str] = None,
    ) -> Optional["LeverageScoreHistogramConfig"]:
        if not output_dir:
            return None
        return cls(
            output_dir=output_dir,
            bins=bins,
            min_value=min_value,
            max_value=max_value,
            layers=parse_index_filter(layers),
            steps=parse_index_filter(steps),
        )

    def should_record(self, layer_id: int, step_idx: int) -> bool:
        if self.layers is not None and int(layer_id) not in self.layers:
            return False
        if self.steps is not None and int(step_idx) not in self.steps:
            return False
        return True

    def record(self, policy_scores: torch.Tensor, *, layer_id: int, step_idx: int) -> None:
        if policy_scores is None or not self.should_record(layer_id, step_idx):
            return
        with torch.no_grad():
            values = policy_scores.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return

            layer_id = int(layer_id)
            if layer_id not in self._counts:
                self._counts[layer_id] = torch.zeros(self.bins, dtype=torch.int64)
                self._summary[layer_id] = {
                    "total_tokens": 0,
                    "records": 0,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "min": None,
                    "max": None,
                    "underflow": 0,
                    "overflow": 0,
                }

            in_range = (values >= self.min_value) & (values <= self.max_value)
            underflow = values < self.min_value
            overflow = values > self.max_value
            if bool(in_range.any().item()):
                bin_ids = torch.bucketize(values[in_range], self.edges[1:-1], right=False)
                self._counts[layer_id] += torch.bincount(bin_ids, minlength=self.bins).to(torch.int64)

            summary = self._summary[layer_id]
            count = int(values.numel())
            summary["total_tokens"] = int(summary["total_tokens"]) + count
            summary["records"] = int(summary["records"]) + 1
            summary["sum"] = float(summary["sum"]) + float(values.sum().item())
            summary["sum_sq"] = float(summary["sum_sq"]) + float((values * values).sum().item())
            vmin = float(values.min().item())
            vmax = float(values.max().item())
            summary["min"] = vmin if summary["min"] is None else min(float(summary["min"]), vmin)
            summary["max"] = vmax if summary["max"] is None else max(float(summary["max"]), vmax)
            summary["underflow"] = int(summary["underflow"]) + int(underflow.sum().item())
            summary["overflow"] = int(summary["overflow"]) + int(overflow.sum().item())
            self._flushed = False

    def flush(self) -> None:
        if self._flushed:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        hist_path = os.path.join(self.output_dir, "leverage_histograms.csv")
        summary_path = os.path.join(self.output_dir, "leverage_histogram_summary.csv")
        layer_ids = sorted(self._counts)

        with open(hist_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_id", "bin_left", "bin_right", "count"])
            for layer_id in layer_ids:
                counts = self._counts[layer_id].tolist()
                for bin_idx, count in enumerate(counts):
                    writer.writerow([
                        layer_id,
                        float(self.edges[bin_idx].item()),
                        float(self.edges[bin_idx + 1].item()),
                        int(count),
                    ])

        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_id", "min", "max", "mean", "std", "total_tokens", "underflow", "overflow", "records"])
            for layer_id in layer_ids:
                summary = self._summary[layer_id]
                total = int(summary["total_tokens"])
                mean = None
                std = None
                if total > 0:
                    mean = float(summary["sum"]) / float(total)
                    variance = max(float(summary["sum_sq"]) / float(total) - mean * mean, 0.0)
                    std = math.sqrt(variance)
                writer.writerow([
                    layer_id,
                    summary["min"],
                    summary["max"],
                    mean,
                    std,
                    total,
                    int(summary["underflow"]),
                    int(summary["overflow"]),
                    int(summary["records"]),
                ])

        self._write_plots(layer_ids)
        self._flushed = True

    def _write_plots(self, layer_ids: list[int]) -> None:
        if not layer_ids:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        edges = self.edges.numpy()
        left = edges[:-1]
        width = edges[1:] - edges[:-1]
        for layer_id in layer_ids:
            counts = self._counts[layer_id].numpy()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(left, counts, width=width, align="edge")
            ax.set_xlabel("Leverage score bin")
            ax.set_ylabel("Token count")
            ax.set_title(f"Layer {layer_id:02d} leverage score histogram")
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"layer_{layer_id:02d}_hist.png"), dpi=150)
            plt.close(fig)

        heat = torch.stack([self._counts[layer_id] for layer_id in layer_ids], dim=0).numpy()
        fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(layer_ids))))
        image = ax.imshow(heat, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_xlabel("Leverage score bin")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layer_ids)))
        ax.set_yticklabels([str(layer_id) for layer_id in layer_ids])
        ax.set_title("Leverage score histogram by layer")
        fig.colorbar(image, ax=ax, label="Token count")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "all_layers_heatmap.png"), dpi=150)
        plt.close(fig)


def add_leverage_score_histogram_args(parser) -> None:
    parser.add_argument("--leverage_score_histogram_dir", "--leverage-score-histogram-dir", type=str, default=None)
    parser.add_argument("--leverage_score_histogram_bins", "--leverage-score-histogram-bins", type=int, default=100)
    parser.add_argument("--leverage_score_histogram_min", "--leverage-score-histogram-min", type=float, default=0.0)
    parser.add_argument("--leverage_score_histogram_max", "--leverage-score-histogram-max", type=float, default=1.0)
    parser.add_argument("--leverage_score_histogram_layers", "--leverage-score-histogram-layers", type=str, default="all")
    parser.add_argument("--leverage_score_histogram_steps", "--leverage-score-histogram-steps", type=str, default="all")


def leverage_score_histogram_config_from_args(
    args: Any,
    output_dir: Optional[str] = None,
) -> Optional[LeverageScoreHistogramConfig]:
    return LeverageScoreHistogramConfig.from_cli(
        output_dir if output_dir is not None else getattr(args, "leverage_score_histogram_dir", None),
        bins=getattr(args, "leverage_score_histogram_bins", 100),
        min_value=getattr(args, "leverage_score_histogram_min", 0.0),
        max_value=getattr(args, "leverage_score_histogram_max", 1.0),
        layers=getattr(args, "leverage_score_histogram_layers", "all"),
        steps=getattr(args, "leverage_score_histogram_steps", "all"),
    )


@dataclass
class TokenOverlayDumpConfig:
    """Opt-in full token dump for step-wise eviction/leverage overlays."""

    output_dir: str
    layers: Optional[set[int]] = None
    heads: Optional[set[int]] = None
    steps: Optional[set[int]] = None
    max_events: Optional[int] = None

    def __post_init__(self) -> None:
        self.output_dir = os.path.abspath(self.output_dir)
        self.events_dir = os.path.join(self.output_dir, "events")
        os.makedirs(self.events_dir, exist_ok=True)
        self._num_events = 0

    @classmethod
    def from_cli(
        cls,
        output_dir: Optional[str],
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        steps: Optional[str] = None,
        max_events: Optional[int] = None,
    ) -> Optional["TokenOverlayDumpConfig"]:
        if not output_dir:
            return None
        return cls(
            output_dir=output_dir,
            layers=parse_index_filter(layers),
            heads=parse_index_filter(heads),
            steps=parse_index_filter(steps),
            max_events=max_events,
        )

    def should_dump(self, layer_id: int, head_id: Optional[int], step_idx: int) -> bool:
        if self.max_events is not None and self._num_events >= self.max_events:
            return False
        if self.layers is not None and int(layer_id) not in self.layers:
            return False
        if self.heads is not None and head_id is not None and int(head_id) not in self.heads:
            return False
        if self.steps is not None and int(step_idx) not in self.steps:
            return False
        return True

    def next_path(self, *, step_idx: int, layer_id: int, head_id: Optional[int], batch_id: int) -> Path:
        head_label = "layer_shared" if head_id is None else f"{int(head_id):02d}"
        stem = f"step_{int(step_idx):06d}_layer_{int(layer_id):02d}_head_{head_label}_batch_{int(batch_id):02d}"
        return Path(self.events_dir) / f"{stem}.pt"

    def record_dump(self) -> None:
        self._num_events += 1


def add_token_overlay_dump_args(parser) -> None:
    parser.add_argument("--token_overlay_dump_dir", "--token-overlay-dump-dir", type=str, default=None)
    parser.add_argument("--token_overlay_dump_layers", "--token-overlay-dump-layers", type=str, default="all")
    parser.add_argument("--token_overlay_dump_heads", "--token-overlay-dump-heads", type=str, default="all")
    parser.add_argument("--token_overlay_dump_steps", "--token-overlay-dump-steps", type=str, default="all")
    parser.add_argument("--token_overlay_dump_max_events", "--token-overlay-dump-max-events", type=int, default=None)


def token_overlay_dump_config_from_args(
    args: Any,
    output_dir: Optional[str] = None,
) -> Optional[TokenOverlayDumpConfig]:
    return TokenOverlayDumpConfig.from_cli(
        output_dir if output_dir is not None else getattr(args, "token_overlay_dump_dir", None),
        layers=getattr(args, "token_overlay_dump_layers", "all"),
        heads=getattr(args, "token_overlay_dump_heads", "all"),
        steps=getattr(args, "token_overlay_dump_steps", "all"),
        max_events=getattr(args, "token_overlay_dump_max_events", None),
    )


def dump_token_overlay_event(
    config: TokenOverlayDumpConfig,
    *,
    kept_candidate_indices: torch.Tensor,
    policy_scores: torch.Tensor,
    metadata: Optional[Any],
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    tokens_per_frame: Optional[int],
    eviction_policy: str,
    leverage_granularity: str,
    selection_granularity: Optional[str] = None,
) -> None:
    """Save full candidate leverage scores and current-step evicted token ids."""
    if policy_scores is None or kept_candidate_indices is None or metadata is None:
        return
    with torch.no_grad():
        scores = policy_scores.detach()
        if scores.ndim not in (2, 3):
            return

        B = int(metadata.frame_ids.shape[0])
        H = int(metadata.frame_ids.shape[1])
        candidate_count = int(scores.shape[-1])
        if candidate_count <= 0:
            return
        active_selection_granularity = leverage_granularity if selection_granularity is None else selection_granularity
        all_candidates_cpu = torch.arange(candidate_count, dtype=torch.long)
        all_candidates_device = all_candidates_cpu.to(device=kept_candidate_indices.device)

        def candidate_metadata(batch_id: int, head_id: int) -> tuple[torch.Tensor, torch.Tensor]:
            start = int(num_anchor_tokens)
            end = start + candidate_count
            frame_ids = metadata.frame_ids[batch_id, head_id, start:end]
            token_indices = metadata.token_indices[batch_id, head_id, start:end]
            return frame_ids.detach().cpu().long(), token_indices.detach().cpu().long()

        def anchor_metadata(batch_id: int, head_id: int) -> tuple[torch.Tensor, torch.Tensor]:
            if int(num_anchor_tokens) <= 0:
                empty = torch.empty(0, dtype=torch.long)
                return empty, empty
            frame_ids = metadata.frame_ids[batch_id, head_id, :int(num_anchor_tokens)]
            token_indices = metadata.token_indices[batch_id, head_id, :int(num_anchor_tokens)]
            return frame_ids.detach().cpu().long(), token_indices.detach().cpu().long()

        if active_selection_granularity == "layer" or scores.ndim == 2:
            for batch_id in range(B):
                if not config.should_dump(layer_id, None, step_idx):
                    continue
                kept_local, evicted_local = _shared_local_indices(
                    kept_candidate_indices,
                    all_candidates_device,
                    candidate_count,
                    batch_id,
                    device=kept_candidate_indices.device,
                )
                kept_local = kept_local.detach().cpu().long()
                evicted_local = evicted_local.detach().cpu().long()
                candidate_scores = _scores_for_batch(scores, batch_id, None).detach().cpu().float()
                candidate_frame_ids, candidate_token_indices = candidate_metadata(batch_id, 0)
                anchor_frame_ids, anchor_token_indices = anchor_metadata(batch_id, 0)
                _write_token_overlay_payload(
                    config,
                    batch_id=batch_id,
                    head_id=None,
                    layer_id=layer_id,
                    step_idx=step_idx,
                    cache_budget=cache_budget,
                    num_anchor_tokens=num_anchor_tokens,
                    tokens_per_frame=tokens_per_frame,
                    candidate_frame_ids=candidate_frame_ids,
                    candidate_token_indices=candidate_token_indices,
                    candidate_scores=candidate_scores,
                    kept_local=kept_local,
                    evicted_local=evicted_local,
                    anchor_frame_ids=anchor_frame_ids,
                    anchor_token_indices=anchor_token_indices,
                    eviction_policy=eviction_policy,
                    leverage_granularity=leverage_granularity,
                    selection_granularity=active_selection_granularity,
                )
        else:
            for batch_id in range(B):
                for head_id in range(H):
                    if not config.should_dump(layer_id, head_id, step_idx):
                        continue
                    kept_local = kept_candidate_indices[batch_id, head_id].detach().cpu().long()
                    evicted_local = _evicted_from_kept(
                        kept_local.to(device=all_candidates_cpu.device),
                        all_candidates_cpu,
                        candidate_count,
                    ).detach().cpu().long()
                    candidate_scores = _scores_for_batch(scores, batch_id, head_id).detach().cpu().float()
                    candidate_frame_ids, candidate_token_indices = candidate_metadata(batch_id, head_id)
                    anchor_frame_ids, anchor_token_indices = anchor_metadata(batch_id, head_id)
                    _write_token_overlay_payload(
                        config,
                        batch_id=batch_id,
                        head_id=head_id,
                        layer_id=layer_id,
                        step_idx=step_idx,
                        cache_budget=cache_budget,
                        num_anchor_tokens=num_anchor_tokens,
                        tokens_per_frame=tokens_per_frame,
                        candidate_frame_ids=candidate_frame_ids,
                        candidate_token_indices=candidate_token_indices,
                        candidate_scores=candidate_scores,
                        kept_local=kept_local,
                        evicted_local=evicted_local,
                        anchor_frame_ids=anchor_frame_ids,
                        anchor_token_indices=anchor_token_indices,
                        eviction_policy=eviction_policy,
                        leverage_granularity=leverage_granularity,
                        selection_granularity=active_selection_granularity,
                    )


def _write_token_overlay_payload(
    config: TokenOverlayDumpConfig,
    *,
    batch_id: int,
    head_id: Optional[int],
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    tokens_per_frame: Optional[int],
    candidate_frame_ids: torch.Tensor,
    candidate_token_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    kept_local: torch.Tensor,
    evicted_local: torch.Tensor,
    anchor_frame_ids: torch.Tensor,
    anchor_token_indices: torch.Tensor,
    eviction_policy: str,
    leverage_granularity: str,
    selection_granularity: str,
) -> None:
    event_path = config.next_path(step_idx=step_idx, layer_id=layer_id, head_id=head_id, batch_id=batch_id)
    empty_float = torch.empty(0, dtype=torch.float32)
    empty_long = torch.empty(0, dtype=torch.long)
    evicted_scores = candidate_scores.index_select(0, evicted_local) if evicted_local.numel() else empty_float
    payload = {
        "candidate_frame_ids": candidate_frame_ids,
        "candidate_token_indices": candidate_token_indices,
        "candidate_leverage_scores": candidate_scores,
        "kept_candidate_indices": kept_local,
        "evicted_candidate_indices": evicted_local,
        "evicted_frame_ids": candidate_frame_ids.index_select(0, evicted_local) if evicted_local.numel() else empty_long,
        "evicted_token_indices": candidate_token_indices.index_select(0, evicted_local) if evicted_local.numel() else empty_long,
        "evicted_leverage_scores": evicted_scores,
        "anchor_frame_ids": anchor_frame_ids,
        "anchor_token_indices": anchor_token_indices,
        "meta": {
            "layer_id": int(layer_id),
            "step_idx": int(step_idx),
            "batch_id": int(batch_id),
            "head_id": None if head_id is None else int(head_id),
            "head_label": "layer_shared" if head_id is None else f"{int(head_id):02d}",
            "num_anchor_tokens": int(num_anchor_tokens),
            "cache_budget": int(cache_budget),
            "candidate_count": int(candidate_scores.numel()),
            "tokens_per_frame": None if tokens_per_frame is None else int(tokens_per_frame),
            "evicted_count": int(evicted_local.numel()),
            "kept_count": int(kept_local.numel()),
            "eviction_policy": str(eviction_policy),
            "leverage_granularity": str(leverage_granularity),
            "selection_granularity": str(selection_granularity),
        },
    }
    torch.save(payload, event_path, pickle_protocol=4)
    with open(event_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload["meta"]), f, indent=2)
    config.record_dump()


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
    eviction_policy: str = "mean",
    leverage_sketch_dim: Optional[int] = None,
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
                    "eviction_policy": str(eviction_policy),
                    "leverage_sketch_dim": leverage_sketch_dim,
                    "score_definition": "dot(normalized_candidate_key, mean(normalized_candidate_keys))",
                    "policy": "anchors preserved; evicted/kept indices follow eviction_policy",
                },
            }

            stem = f"step{step_idx:06d}_layer{layer_id:02d}_head{head_id:02d}_batch{batch_id:02d}"
            torch.save(payload, os.path.join(config.output_dir, f"{stem}.pt"), pickle_protocol=4)
            with open(os.path.join(config.output_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(payload["meta"], f, indent=2)
            config.record_dump()


@dataclass
class EvictionNNAnalysisConfig:
    """Opt-in nearest-retained diagnostic for KV-cache eviction events."""

    output_dir: str
    layers: Optional[set[int]] = None
    heads: Optional[set[int]] = None
    steps: Optional[set[int]] = None
    space: str = "full_key"
    max_evicted: int = 8192
    chunk_size: int = 2048
    save_topk_pairs: int = 256
    save_hist: bool = False
    seed: int = 0
    knn_k: int = 5
    max_snapshots: Optional[int] = None

    def __post_init__(self) -> None:
        if self.space not in ("full_key", "svd_coord", "both"):
            raise ValueError(f"space must be one of ('full_key', 'svd_coord', 'both'), got {self.space!r}")
        if int(self.max_evicted) < 1:
            raise ValueError(f"max_evicted must be >= 1, got {self.max_evicted}")
        if int(self.chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if int(self.save_topk_pairs) < 0:
            raise ValueError(f"save_topk_pairs must be >= 0, got {self.save_topk_pairs}")
        if int(self.knn_k) < 1:
            raise ValueError(f"knn_k must be >= 1, got {self.knn_k}")
        self.output_dir = os.path.abspath(self.output_dir)
        self.events_dir = os.path.join(self.output_dir, "events")
        os.makedirs(self.events_dir, exist_ok=True)
        self.summary_path = os.path.join(self.output_dir, "summary.jsonl")
        self._num_snapshots = 0

    @classmethod
    def from_cli(
        cls,
        output_dir: Optional[str],
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        steps: Optional[str] = None,
        space: str = "full_key",
        max_evicted: int = 8192,
        chunk_size: int = 2048,
        save_topk_pairs: int = 256,
        save_hist: bool = False,
        seed: int = 0,
        knn_k: int = 5,
        max_snapshots: Optional[int] = None,
    ) -> Optional["EvictionNNAnalysisConfig"]:
        if not output_dir:
            return None
        return cls(
            output_dir=output_dir,
            layers=parse_index_filter(layers),
            heads=parse_index_filter(heads),
            steps=parse_index_filter(steps),
            space=space,
            max_evicted=max_evicted,
            chunk_size=chunk_size,
            save_topk_pairs=save_topk_pairs,
            save_hist=save_hist,
            seed=seed,
            knn_k=knn_k,
            max_snapshots=max_snapshots,
        )

    def spaces(self) -> tuple[str, ...]:
        return ("full_key", "svd_coord") if self.space == "both" else (self.space,)

    def wants_svd_coord(self) -> bool:
        return self.space in ("svd_coord", "both")

    def should_dump(self, layer_id: int, head_id: Optional[int], step_idx: int) -> bool:
        if self.max_snapshots is not None and self._num_snapshots >= self.max_snapshots:
            return False
        if self.layers is not None and int(layer_id) not in self.layers:
            return False
        if self.heads is not None and head_id is not None and int(head_id) not in self.heads:
            return False
        if self.steps is not None and int(step_idx) not in self.steps:
            return False
        return True

    def next_path(self, *, step_idx: int, layer_id: int, head_id: Optional[int], batch_id: int, space: str) -> Path:
        head_label = "layer_shared" if head_id is None else f"{int(head_id):02d}"
        stem = f"step_{int(step_idx):06d}_layer_{int(layer_id):02d}_head_{head_label}_batch_{int(batch_id):02d}_{space}"
        return Path(self.events_dir) / f"{stem}.json"

    def record(self, summary: dict[str, Any], event_path: Path, top_pairs: Optional[list[dict[str, Any]]] = None) -> None:
        payload = dict(summary)
        if top_pairs is not None:
            payload["top_pairs_preview"] = top_pairs[: min(len(top_pairs), 16)]
        with open(event_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2)
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(summary), separators=(",", ":")) + "\n")
        self._num_snapshots += 1


def add_eviction_nn_analysis_args(parser) -> None:
    parser.add_argument("--eviction_nn_analysis_dir", "--eviction-nn-analysis-dir", type=str, default=None)
    parser.add_argument("--eviction_nn_analysis_layers", "--eviction-nn-analysis-layers", type=str, default="all")
    parser.add_argument("--eviction_nn_analysis_heads", "--eviction-nn-analysis-heads", type=str, default="all")
    parser.add_argument("--eviction_nn_analysis_steps", "--eviction-nn-analysis-steps", type=str, default="all")
    parser.add_argument(
        "--eviction_nn_analysis_space",
        "--eviction-nn-analysis-space",
        choices=("full_key", "svd_coord", "both"),
        default="full_key",
    )
    parser.add_argument("--eviction_nn_analysis_max_evicted", "--eviction-nn-analysis-max-evicted", type=int, default=8192)
    parser.add_argument("--eviction_nn_analysis_chunk_size", "--eviction-nn-analysis-chunk-size", type=int, default=2048)
    parser.add_argument("--eviction_nn_analysis_save_topk_pairs", "--eviction-nn-analysis-save-topk-pairs", type=int, default=256)
    parser.add_argument("--eviction_nn_analysis_save_hist", "--eviction-nn-analysis-save-hist", action="store_true")
    parser.add_argument("--eviction_nn_analysis_seed", "--eviction-nn-analysis-seed", type=int, default=0)
    parser.add_argument("--eviction_nn_analysis_knn_k", "--eviction-nn-analysis-knn-k", type=int, default=5)
    parser.add_argument("--eviction_nn_analysis_max_snapshots", "--eviction-nn-analysis-max-snapshots", type=int, default=None)


def eviction_nn_config_from_args(args: Any, output_dir: Optional[str] = None) -> Optional[EvictionNNAnalysisConfig]:
    return EvictionNNAnalysisConfig.from_cli(
        output_dir if output_dir is not None else getattr(args, "eviction_nn_analysis_dir", None),
        layers=getattr(args, "eviction_nn_analysis_layers", "all"),
        heads=getattr(args, "eviction_nn_analysis_heads", "all"),
        steps=getattr(args, "eviction_nn_analysis_steps", "all"),
        space=getattr(args, "eviction_nn_analysis_space", "full_key"),
        max_evicted=getattr(args, "eviction_nn_analysis_max_evicted", 8192),
        chunk_size=getattr(args, "eviction_nn_analysis_chunk_size", 2048),
        save_topk_pairs=getattr(args, "eviction_nn_analysis_save_topk_pairs", 256),
        save_hist=getattr(args, "eviction_nn_analysis_save_hist", False),
        seed=getattr(args, "eviction_nn_analysis_seed", 0),
        knn_k=getattr(args, "eviction_nn_analysis_knn_k", 5),
        max_snapshots=getattr(args, "eviction_nn_analysis_max_snapshots", None),
    )


def dump_eviction_nn_analysis(
    config: EvictionNNAnalysisConfig,
    *,
    k_before: torch.Tensor,
    v_before: Optional[torch.Tensor],
    kept_candidate_indices: torch.Tensor,
    policy_scores: torch.Tensor,
    leverage_basis: Optional[Any],
    metadata: Optional[Any],
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    eviction_policy: str,
    leverage_granularity: str,
    leverage_feature: str,
    selection_granularity: Optional[str] = None,
) -> None:
    """Dump nearest-retained diagnostics for one pre-eviction cache state."""
    with torch.no_grad():
        B, H, N, D = k_before.shape
        candidate_count = int(N) - int(num_anchor_tokens)
        if candidate_count < 0:
            return
        active_selection_granularity = leverage_granularity if selection_granularity is None else selection_granularity
        if active_selection_granularity not in ("head", "layer"):
            raise ValueError(
                "selection_granularity must be 'head' or 'layer', got "
                f"{active_selection_granularity!r}"
            )
        device = k_before.device
        all_candidates = torch.arange(candidate_count, device=device, dtype=torch.long)

        for space in config.spaces():
            if space == "full_key":
                if active_selection_granularity == "layer":
                    for batch_id in range(B):
                        if not config.should_dump(layer_id, None, step_idx):
                            continue
                        kept_local, evicted_local = _shared_local_indices(kept_candidate_indices, all_candidates, candidate_count, batch_id, device)
                        if leverage_feature in ("key_value", "key_value_lowdim_concat") and v_before is not None:
                            features = torch.cat(
                                [
                                    k_before[batch_id].float().permute(1, 0, 2).reshape(N, H * D),
                                    v_before[batch_id].float().permute(1, 0, 2).reshape(N, H * D),
                                ],
                                dim=-1,
                            )
                            analysis_feature = leverage_feature
                        else:
                            features = k_before[batch_id].float().permute(1, 0, 2).reshape(N, H * D)
                            analysis_feature = "key_fallback" if leverage_feature in ("key_value", "key_value_lowdim_concat") else "key"
                        _dump_one_eviction_nn_event(
                            config,
                            features=features,
                            retained_cache_idx=kept_local + int(num_anchor_tokens),
                            evicted_cache_idx=evicted_local + int(num_anchor_tokens),
                            policy_scores=_scores_for_batch(policy_scores, batch_id, None),
                            metadata=metadata,
                            batch_id=batch_id,
                            head_id=None,
                            layer_id=layer_id,
                            step_idx=step_idx,
                            cache_budget=cache_budget,
                            num_anchor_tokens=num_anchor_tokens,
                            eviction_policy=eviction_policy,
                            leverage_granularity=leverage_granularity,
                            selection_granularity=active_selection_granularity,
                            analysis_space=space,
                            analysis_feature=analysis_feature,
                            total_tokens_before_eviction=N,
                            original_evicted_count=int(evicted_local.numel()),
                        )
                else:
                    for batch_id in range(B):
                        for head_id in range(H):
                            if not config.should_dump(layer_id, head_id, step_idx):
                                continue
                            kept_local = kept_candidate_indices[batch_id, head_id].to(device=device, dtype=torch.long)
                            evicted_local = _evicted_from_kept(kept_local, all_candidates, candidate_count)
                            features = k_before[batch_id, head_id].float()
                            analysis_feature = "key"
                            if leverage_feature in ("key_value", "key_value_lowdim_concat") and v_before is not None:
                                features = torch.cat([features, v_before[batch_id, head_id].float()], dim=-1)
                                analysis_feature = leverage_feature
                            elif leverage_feature in ("key_value", "key_value_lowdim_concat"):
                                analysis_feature = "key_fallback"
                            _dump_one_eviction_nn_event(
                                config,
                                features=features,
                                retained_cache_idx=kept_local + int(num_anchor_tokens),
                                evicted_cache_idx=evicted_local + int(num_anchor_tokens),
                                policy_scores=_scores_for_batch(policy_scores, batch_id, head_id),
                                metadata=metadata,
                                batch_id=batch_id,
                                head_id=head_id,
                                layer_id=layer_id,
                                step_idx=step_idx,
                                cache_budget=cache_budget,
                                num_anchor_tokens=num_anchor_tokens,
                                eviction_policy=eviction_policy,
                                leverage_granularity=leverage_granularity,
                                selection_granularity=active_selection_granularity,
                                analysis_space=space,
                                analysis_feature=analysis_feature,
                                total_tokens_before_eviction=N,
                                original_evicted_count=int(evicted_local.numel()),
                            )
            elif space == "svd_coord":
                _dump_svd_coord_events(
                    config,
                    leverage_basis=leverage_basis,
                    kept_candidate_indices=kept_candidate_indices,
                    all_candidates=all_candidates,
                    candidate_count=candidate_count,
                    policy_scores=policy_scores,
                    metadata=metadata,
                    layer_id=layer_id,
                    step_idx=step_idx,
                    cache_budget=cache_budget,
                    num_anchor_tokens=num_anchor_tokens,
                    eviction_policy=eviction_policy,
                    leverage_granularity=leverage_granularity,
                    selection_granularity=active_selection_granularity,
                    total_tokens_before_eviction=N,
                    device=device,
                )


def _dump_svd_coord_events(
    config: EvictionNNAnalysisConfig,
    *,
    leverage_basis: Optional[Any],
    kept_candidate_indices: torch.Tensor,
    all_candidates: torch.Tensor,
    candidate_count: int,
    policy_scores: torch.Tensor,
    metadata: Optional[Any],
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    eviction_policy: str,
    leverage_granularity: str,
    selection_granularity: str,
    total_tokens_before_eviction: int,
    device: torch.device,
) -> None:
    B, H, _ = kept_candidate_indices.shape
    basis_q = None if leverage_basis is None else getattr(leverage_basis, "q", None)
    basis_granularity = None if leverage_basis is None else getattr(leverage_basis, "granularity", None)
    unavailable = basis_q is None or basis_q.numel() == 0

    if selection_granularity == "layer":
        for batch_id in range(B):
            if not config.should_dump(layer_id, None, step_idx):
                continue
            kept_local, evicted_local = _shared_local_indices(kept_candidate_indices, all_candidates, candidate_count, batch_id, device)
            if unavailable or basis_granularity != "layer":
                _record_skipped_svd_event(config, batch_id, None, layer_id, step_idx, cache_budget, num_anchor_tokens, eviction_policy, leverage_granularity, selection_granularity, total_tokens_before_eviction, int(kept_local.numel()), int(evicted_local.numel()), "coordinates_not_available")
                continue
            features = torch.nan_to_num(basis_q[batch_id].to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            _dump_one_eviction_nn_event(
                config,
                features=features,
                retained_cache_idx=kept_local + int(num_anchor_tokens),
                evicted_cache_idx=evicted_local + int(num_anchor_tokens),
                feature_row_cache_offset=int(num_anchor_tokens),
                policy_scores=_scores_for_batch(policy_scores, batch_id, None),
                metadata=metadata,
                batch_id=batch_id,
                head_id=None,
                layer_id=layer_id,
                step_idx=step_idx,
                cache_budget=cache_budget,
                num_anchor_tokens=num_anchor_tokens,
                eviction_policy=eviction_policy,
                leverage_granularity=leverage_granularity,
                selection_granularity=selection_granularity,
                analysis_space="svd_coord",
                analysis_feature="signed_q",
                total_tokens_before_eviction=total_tokens_before_eviction,
                original_evicted_count=int(evicted_local.numel()),
                svd_coord_available=True,
            )
    else:
        for batch_id in range(B):
            for head_id in range(H):
                if not config.should_dump(layer_id, head_id, step_idx):
                    continue
                kept_local = kept_candidate_indices[batch_id, head_id].to(device=device, dtype=torch.long)
                evicted_local = _evicted_from_kept(kept_local, all_candidates, candidate_count)
                if unavailable or basis_granularity not in ("head", "layer"):
                    _record_skipped_svd_event(config, batch_id, head_id, layer_id, step_idx, cache_budget, num_anchor_tokens, eviction_policy, leverage_granularity, selection_granularity, total_tokens_before_eviction, int(kept_local.numel()), int(evicted_local.numel()), "coordinates_not_available")
                    continue
                basis_features = basis_q[batch_id, head_id] if basis_granularity == "head" else basis_q[batch_id]
                features = torch.nan_to_num(basis_features.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _dump_one_eviction_nn_event(
                    config,
                    features=features,
                    retained_cache_idx=kept_local + int(num_anchor_tokens),
                    evicted_cache_idx=evicted_local + int(num_anchor_tokens),
                    feature_row_cache_offset=int(num_anchor_tokens),
                    policy_scores=_scores_for_batch(policy_scores, batch_id, head_id),
                    metadata=metadata,
                    batch_id=batch_id,
                    head_id=head_id,
                    layer_id=layer_id,
                    step_idx=step_idx,
                    cache_budget=cache_budget,
                    num_anchor_tokens=num_anchor_tokens,
                    eviction_policy=eviction_policy,
                    leverage_granularity=leverage_granularity,
                    selection_granularity=selection_granularity,
                    analysis_space="svd_coord",
                    analysis_feature="signed_q",
                    total_tokens_before_eviction=total_tokens_before_eviction,
                    original_evicted_count=int(evicted_local.numel()),
                    svd_coord_available=True,
                )


def _dump_one_eviction_nn_event(
    config: EvictionNNAnalysisConfig,
    *,
    features: torch.Tensor,
    retained_cache_idx: torch.Tensor,
    evicted_cache_idx: torch.Tensor,
    policy_scores: Optional[torch.Tensor],
    metadata: Optional[Any],
    batch_id: int,
    head_id: Optional[int],
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    eviction_policy: str,
    leverage_granularity: str,
    selection_granularity: str,
    analysis_space: str,
    analysis_feature: str,
    total_tokens_before_eviction: int,
    original_evicted_count: int,
    feature_row_cache_offset: int = 0,
    svd_coord_available: Optional[bool] = None,
) -> None:
    sampled_cache_idx, sampled = _sample_evicted(
        evicted_cache_idx,
        max_evicted=int(config.max_evicted),
        seed=int(config.seed) + int(step_idx) * 100000 + int(layer_id) * 1000 + (-1 if head_id is None else int(head_id)),
    )
    event_path = config.next_path(step_idx=step_idx, layer_id=layer_id, head_id=head_id, batch_id=batch_id, space=analysis_space)
    summary = _base_summary(
        batch_id=batch_id,
        head_id=head_id,
        layer_idx=layer_id,
        step_idx=step_idx,
        cache_budget=cache_budget,
        num_anchor_tokens=num_anchor_tokens,
        eviction_policy=eviction_policy,
        leverage_granularity=leverage_granularity,
        selection_granularity=selection_granularity,
        analysis_space=analysis_space,
        analysis_feature=analysis_feature,
        total_tokens_before_eviction=total_tokens_before_eviction,
        retained_count=int(retained_cache_idx.numel()),
        evicted_count=int(original_evicted_count),
        analyzed_evicted_count=int(sampled_cache_idx.numel()),
        sampled=sampled,
    )
    if svd_coord_available is not None:
        summary["svd_coord_available"] = bool(svd_coord_available)

    if retained_cache_idx.numel() == 0 or sampled_cache_idx.numel() == 0:
        summary.update(_empty_distance_summary())
        config.record(summary, event_path, top_pairs=[])
        _print_eviction_nn_debug(summary)
        return

    retained_rows = retained_cache_idx.to(device=features.device, dtype=torch.long) - int(feature_row_cache_offset)
    evicted_rows = sampled_cache_idx.to(device=features.device, dtype=torch.long) - int(feature_row_cache_offset)
    valid_retained = (retained_rows >= 0) & (retained_rows < features.shape[0])
    valid_evicted = (evicted_rows >= 0) & (evicted_rows < features.shape[0])
    retained_cache_idx_device = retained_cache_idx.to(device=features.device, dtype=torch.long)
    sampled_cache_idx_device = sampled_cache_idx.to(device=features.device, dtype=torch.long)
    if not bool(valid_retained.all().item()):
        retained_rows = retained_rows[valid_retained]
        retained_cache_idx_device = retained_cache_idx_device[valid_retained]
    if not bool(valid_evicted.all().item()):
        evicted_rows = evicted_rows[valid_evicted]
        sampled_cache_idx_device = sampled_cache_idx_device[valid_evicted]
    if retained_rows.numel() == 0 or evicted_rows.numel() == 0:
        summary["analyzed_evicted_count"] = int(evicted_rows.numel())
        summary.update(_empty_distance_summary())
        summary["skip_reason"] = "indices_outside_feature_rows"
        config.record(summary, event_path, top_pairs=[])
        _print_eviction_nn_debug(summary)
        return

    retained_features = features.index_select(0, retained_rows).float()
    evicted_features = features.index_select(0, evicted_rows).float()
    nn = compute_nearest_retained(retained_features, evicted_features, retained_cache_idx_device, chunk_size=int(config.chunk_size))
    nearest_cache_idx = nn["nearest_retained_cache_idx"]
    cosine_similarity = nn["nearest_cosine_similarity"]
    cosine_distance = 1.0 - cosine_similarity
    summary["analyzed_evicted_count"] = int(sampled_cache_idx_device.numel())
    summary.update(_distance_summary(cosine_similarity, cosine_distance))
    summary.update(
        compute_knn_distance_ratio(
            retained_features=retained_features,
            pre_features=features.float(),
            evicted_features=evicted_features,
            evicted_pre_rows=evicted_rows,
            k=int(config.knn_k),
            chunk_size=int(config.chunk_size),
        )
    )

    evicted_leverage = _gather_scores(policy_scores, sampled_cache_idx_device, num_anchor_tokens)
    nearest_leverage = _gather_scores(policy_scores, nearest_cache_idx, num_anchor_tokens)
    _add_leverage_summary(summary, evicted_leverage, nearest_leverage, cosine_distance)
    frame_info = _frame_metadata(metadata, batch_id, head_id, sampled_cache_idx_device, nearest_cache_idx, step_idx)
    summary.update(frame_info["summary"])

    top_pairs = _top_pairs(
        config,
        evicted_cache_idx=sampled_cache_idx_device,
        nearest_cache_idx=nearest_cache_idx,
        cosine_similarity=cosine_similarity,
        cosine_distance=cosine_distance,
        evicted_leverage=evicted_leverage,
        nearest_leverage=nearest_leverage,
        frame_info=frame_info,
        metadata=metadata,
        batch_id=batch_id,
        head_id=head_id,
    )
    if top_pairs:
        top_path = event_path.with_name(event_path.stem + "_top_pairs.pt")
        torch.save(top_pairs, top_path, pickle_protocol=4)
        summary["top_pairs_path"] = str(top_path)
    if config.save_hist:
        hist_path = event_path.with_name(event_path.stem + "_hist.npz")
        _save_hist(hist_path, cosine_distance)
        summary["hist_path"] = str(hist_path)
    config.record(summary, event_path, top_pairs=top_pairs)
    _print_eviction_nn_debug(summary)


def compute_nearest_retained(retained_features: torch.Tensor, evicted_features: torch.Tensor, retained_cache_idx: torch.Tensor, *, chunk_size: int) -> dict[str, torch.Tensor]:
    """Return nearest retained token for each evicted feature by cosine similarity."""
    if int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    ret = F.normalize(torch.nan_to_num(retained_features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1, eps=1e-6)
    ev = F.normalize(torch.nan_to_num(evicted_features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1, eps=1e-6)
    nearest_sim = []
    nearest_idx = []
    for start in range(0, ev.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), ev.shape[0])
        sim = ev[start:end] @ ret.T
        max_sim, argmax_local = sim.max(dim=1)
        nearest_sim.append(max_sim)
        nearest_idx.append(retained_cache_idx[argmax_local])
        del sim
    return {"nearest_cosine_similarity": torch.cat(nearest_sim, dim=0), "nearest_retained_cache_idx": torch.cat(nearest_idx, dim=0)}




def compute_knn_distance_ratio(
    retained_features: torch.Tensor,
    pre_features: torch.Tensor,
    evicted_features: torch.Tensor,
    evicted_pre_rows: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Compute retained/pre KNN distance-sum ratio for analyzed evicted tokens."""
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    query_count = int(evicted_features.shape[0])
    retained_count = int(retained_features.shape[0])
    pre_count = int(pre_features.shape[0])
    ret_k = min(int(k), retained_count)
    pre_k = min(int(k), max(pre_count - 1, 0))
    base = {
        "knn_k": int(k),
        "retained_knn_effective_k": int(ret_k),
        "pre_knn_effective_k": int(pre_k),
        "r_knn_valid_count": int(query_count),
    }
    if query_count == 0 or ret_k <= 0 or pre_k <= 0:
        base.update(
            {
                "r_knn": None,
                "retained_knn_distance_sum": None,
                "pre_knn_distance_sum": None,
                "retained_knn_distance_mean": None,
                "pre_knn_distance_mean": None,
                "r_knn_skip_reason": "insufficient_neighbors",
            }
        )
        return base

    ret = F.normalize(torch.nan_to_num(retained_features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1, eps=1e-6)
    pre = F.normalize(torch.nan_to_num(pre_features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1, eps=1e-6)
    ev = F.normalize(torch.nan_to_num(evicted_features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1, eps=1e-6)
    evicted_rows = evicted_pre_rows.to(device=ev.device, dtype=torch.long)

    retained_sum = torch.zeros((), device=ev.device, dtype=torch.float32)
    pre_sum = torch.zeros((), device=ev.device, dtype=torch.float32)
    for start in range(0, query_count, int(chunk_size)):
        end = min(start + int(chunk_size), query_count)
        ev_chunk = ev[start:end]

        ret_sim = ev_chunk @ ret.T
        ret_top = torch.topk(ret_sim, k=ret_k, dim=1, largest=True).values
        retained_sum = retained_sum + (1.0 - ret_top).sum()
        del ret_sim, ret_top

        pre_sim = ev_chunk @ pre.T
        rows = evicted_rows[start:end]
        valid_self = (rows >= 0) & (rows < pre_count)
        if bool(valid_self.any().item()):
            pre_sim[torch.arange(end - start, device=pre_sim.device)[valid_self], rows[valid_self]] = -float("inf")
        pre_top = torch.topk(pre_sim, k=pre_k, dim=1, largest=True).values
        pre_sum = pre_sum + (1.0 - pre_top).sum()
        del pre_sim, pre_top

    retained_value = float(retained_sum.item())
    pre_value = float(pre_sum.item())
    retained_mean = retained_value / float(query_count * ret_k)
    pre_mean = pre_value / float(query_count * pre_k)
    if not math.isfinite(pre_value) or pre_value <= 1e-12:
        base.update(
            {
                "r_knn": None,
                "retained_knn_distance_sum": retained_value,
                "pre_knn_distance_sum": pre_value,
                "retained_knn_distance_mean": retained_mean,
                "pre_knn_distance_mean": pre_mean,
                "r_knn_skip_reason": "zero_pre_knn_distance_sum",
            }
        )
        return base

    base.update(
        {
            "r_knn": retained_value / pre_value,
            "retained_knn_distance_sum": retained_value,
            "pre_knn_distance_sum": pre_value,
            "retained_knn_distance_mean": retained_mean,
            "pre_knn_distance_mean": pre_mean,
            "r_knn_skip_reason": None,
        }
    )
    return base


def _shared_local_indices(kept_candidate_indices: torch.Tensor, all_candidates: torch.Tensor, candidate_count: int, batch_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    kept = kept_candidate_indices[batch_id, 0].to(device=device, dtype=torch.long)
    return kept, _evicted_from_kept(kept, all_candidates, candidate_count)


def _evicted_from_kept(kept_local: torch.Tensor, all_candidates: torch.Tensor, candidate_count: int) -> torch.Tensor:
    keep_mask = torch.zeros(candidate_count, device=all_candidates.device, dtype=torch.bool)
    if kept_local.numel() > 0:
        keep_mask[kept_local] = True
    return all_candidates[~keep_mask]


def _scores_for_batch(scores: torch.Tensor, batch_id: int, head_id: Optional[int]) -> Optional[torch.Tensor]:
    if scores is None or scores.numel() == 0:
        return None
    if scores.ndim == 3 and head_id is not None:
        return scores[batch_id, head_id]
    if scores.ndim == 2:
        return scores[batch_id]
    if scores.ndim == 3 and head_id is None:
        return scores[batch_id].mean(dim=0)
    return None


def _sample_evicted(evicted_cache_idx: torch.Tensor, *, max_evicted: int, seed: int) -> tuple[torch.Tensor, bool]:
    if evicted_cache_idx.numel() <= int(max_evicted):
        return evicted_cache_idx.detach(), False
    generator = torch.Generator(device=evicted_cache_idx.device)
    generator.manual_seed(int(seed))
    order = torch.randperm(evicted_cache_idx.numel(), device=evicted_cache_idx.device, generator=generator)[: int(max_evicted)]
    return evicted_cache_idx[order].sort().values.detach(), True


def _base_summary(**kwargs: Any) -> dict[str, Any]:
    head_id = kwargs.pop("head_id")
    summary = dict(kwargs)
    summary["head_idx"] = "layer_shared" if head_id is None else int(head_id)
    return summary


def _empty_distance_summary() -> dict[str, Any]:
    keys = (
        "cosine_distance_mean", "cosine_distance_std", "cosine_distance_min", "cosine_distance_p50",
        "cosine_distance_p75", "cosine_distance_p90", "cosine_distance_p95", "cosine_distance_p99",
        "cosine_distance_max", "cosine_similarity_mean", "frac_dist_gt_0_05", "frac_dist_gt_0_10",
        "frac_dist_gt_0_20", "frac_dist_gt_0_30", "frac_sim_lt_0_95", "frac_sim_lt_0_90",
        "frac_sim_lt_0_80", "knn_k", "r_knn", "retained_knn_distance_sum",
        "pre_knn_distance_sum", "retained_knn_distance_mean", "pre_knn_distance_mean",
        "r_knn_valid_count", "retained_knn_effective_k", "pre_knn_effective_k", "r_knn_skip_reason",
    )
    return {key: None for key in keys}


def _distance_summary(sim: torch.Tensor, dist: torch.Tensor) -> dict[str, Any]:
    dist_f = dist.detach().float()
    sim_f = sim.detach().float()
    quantile_q = torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99], device=dist_f.device, dtype=dist_f.dtype)
    quantiles = torch.quantile(dist_f, quantile_q)
    return {
        "cosine_distance_mean": float(dist_f.mean().item()),
        "cosine_distance_std": float(dist_f.std(unbiased=False).item()),
        "cosine_distance_min": float(dist_f.min().item()),
        "cosine_distance_p50": float(quantiles[0].item()),
        "cosine_distance_p75": float(quantiles[1].item()),
        "cosine_distance_p90": float(quantiles[2].item()),
        "cosine_distance_p95": float(quantiles[3].item()),
        "cosine_distance_p99": float(quantiles[4].item()),
        "cosine_distance_max": float(dist_f.max().item()),
        "cosine_similarity_mean": float(sim_f.mean().item()),
        "frac_dist_gt_0_05": float((dist_f > 0.05).float().mean().item()),
        "frac_dist_gt_0_10": float((dist_f > 0.10).float().mean().item()),
        "frac_dist_gt_0_20": float((dist_f > 0.20).float().mean().item()),
        "frac_dist_gt_0_30": float((dist_f > 0.30).float().mean().item()),
        "frac_sim_lt_0_95": float((sim_f < 0.95).float().mean().item()),
        "frac_sim_lt_0_90": float((sim_f < 0.90).float().mean().item()),
        "frac_sim_lt_0_80": float((sim_f < 0.80).float().mean().item()),
    }


def _gather_scores(scores: Optional[torch.Tensor], cache_idx: torch.Tensor, num_anchor_tokens: int) -> Optional[torch.Tensor]:
    if scores is None or scores.numel() == 0 or cache_idx.numel() == 0:
        return None
    local = cache_idx.to(device=scores.device, dtype=torch.long) - int(num_anchor_tokens)
    valid = (local >= 0) & (local < scores.shape[-1])
    out = torch.full((cache_idx.numel(),), float("nan"), device=scores.device, dtype=torch.float32)
    if bool(valid.any().item()):
        out[valid] = scores.float().index_select(0, local[valid])
    return out


def _add_leverage_summary(summary: dict[str, Any], evicted_leverage: Optional[torch.Tensor], nearest_leverage: Optional[torch.Tensor], cosine_distance: torch.Tensor) -> None:
    if evicted_leverage is None:
        return
    valid = torch.isfinite(evicted_leverage)
    if not bool(valid.any().item()):
        return
    ev = evicted_leverage[valid]
    summary["evicted_leverage_mean"] = float(ev.mean().item())
    summary["evicted_leverage_p50"] = float(torch.quantile(ev, torch.tensor(0.50, device=ev.device)).item())
    summary["evicted_leverage_p90"] = float(torch.quantile(ev, torch.tensor(0.90, device=ev.device)).item())
    if nearest_leverage is not None:
        nvalid = torch.isfinite(nearest_leverage)
        if bool(nvalid.any().item()):
            summary["nearest_retained_leverage_mean"] = float(nearest_leverage[nvalid].mean().item())
    if int(valid.sum().item()) >= 2:
        x = evicted_leverage[valid].float()
        y = cosine_distance[valid.to(device=cosine_distance.device)].float()
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        summary["corr_evicted_leverage_vs_nn_distance"] = None if float(denom.item()) == 0.0 else float((x * y).sum().div(denom).item())


def _frame_metadata(metadata: Optional[Any], batch_id: int, head_id: Optional[int], evicted_cache_idx: torch.Tensor, nearest_cache_idx: torch.Tensor, step_idx: int) -> dict[str, Any]:
    result = {"summary": {}, "evicted_frame_ids": None, "nearest_frame_ids": None, "frame_gaps": None}
    if metadata is None or not hasattr(metadata, "frame_ids"):
        return result
    h = 0 if head_id is None else int(head_id)
    ev_cpu = evicted_cache_idx.detach().cpu().long()
    near_cpu = nearest_cache_idx.detach().cpu().long()
    frame_ids = metadata.frame_ids
    ev_frames = frame_ids[batch_id, h, ev_cpu].to(torch.float32)
    near_frames = frame_ids[batch_id, h, near_cpu].to(torch.float32)
    gap = (ev_frames - near_frames).abs()
    latest = torch.max(frame_ids[batch_id, h].to(torch.float32))
    result["evicted_frame_ids"] = ev_frames
    result["nearest_frame_ids"] = near_frames
    result["frame_gaps"] = gap
    result["summary"] = {
        "evicted_frame_id_mean": float(ev_frames.mean().item()),
        "retained_nearest_frame_id_mean": float(near_frames.mean().item()),
        "frame_gap_mean": float(gap.mean().item()),
        "frame_gap_p50": float(torch.quantile(gap, torch.tensor(0.50)).item()),
        "frame_gap_p90": float(torch.quantile(gap, torch.tensor(0.90)).item()),
        "frac_same_frame": float((gap == 0).float().mean().item()),
        "frac_nearest_recent_frame": float((near_frames == latest).float().mean().item()),
        "recent_frame_id": int(latest.item()),
    }
    return result


def _top_pairs(
    config: EvictionNNAnalysisConfig,
    *,
    evicted_cache_idx: torch.Tensor,
    nearest_cache_idx: torch.Tensor,
    cosine_similarity: torch.Tensor,
    cosine_distance: torch.Tensor,
    evicted_leverage: Optional[torch.Tensor],
    nearest_leverage: Optional[torch.Tensor],
    frame_info: dict[str, Any],
    metadata: Optional[Any],
    batch_id: int,
    head_id: Optional[int],
) -> list[dict[str, Any]]:
    k = min(int(config.save_topk_pairs), int(cosine_distance.numel()))
    if k <= 0:
        return []
    _, order = torch.topk(cosine_distance.float(), k=k, largest=True)
    ev_cpu = evicted_cache_idx.detach().cpu().long()
    near_cpu = nearest_cache_idx.detach().cpu().long()
    sim_cpu = cosine_similarity.detach().cpu().float()
    dist_cpu = cosine_distance.detach().cpu().float()
    ev_lev = None if evicted_leverage is None else evicted_leverage.detach().cpu().float()
    near_lev = None if nearest_leverage is None else nearest_leverage.detach().cpu().float()
    ev_frames = frame_info.get("evicted_frame_ids")
    near_frames = frame_info.get("nearest_frame_ids")
    gaps = frame_info.get("frame_gaps")
    h = 0 if head_id is None else int(head_id)
    pairs = []
    for idx in order.detach().cpu().long().tolist():
        ev_idx = int(ev_cpu[idx].item())
        near_idx = int(near_cpu[idx].item())
        ev_voxel, near_voxel = _voxel_pair(metadata, batch_id, h, ev_idx, near_idx)
        pairs.append({
            "evicted_cache_idx": ev_idx,
            "nearest_retained_cache_idx": near_idx,
            "nearest_cosine_similarity": float(sim_cpu[idx].item()),
            "nearest_cosine_distance": float(dist_cpu[idx].item()),
            "evicted_leverage": _optional_float(ev_lev, idx),
            "nearest_retained_leverage": _optional_float(near_lev, idx),
            "evicted_frame_id": _optional_int(ev_frames, idx),
            "nearest_retained_frame_id": _optional_int(near_frames, idx),
            "frame_gap": _optional_int(gaps, idx),
            "evicted_token_idx": _metadata_token(metadata, batch_id, h, ev_idx),
            "nearest_retained_token_idx": _metadata_token(metadata, batch_id, h, near_idx),
            "evicted_uv": None,
            "nearest_retained_uv": None,
            "evicted_voxel": ev_voxel,
            "nearest_retained_voxel": near_voxel,
        })
    return pairs


def _metadata_token(metadata: Optional[Any], batch_id: int, head_id: int, cache_idx: int) -> Optional[int]:
    if metadata is None or not hasattr(metadata, "token_indices"):
        return None
    return int(metadata.token_indices[batch_id, head_id, int(cache_idx)].item())


def _voxel_pair(metadata: Optional[Any], batch_id: int, head_id: int, ev_idx: int, near_idx: int) -> tuple[Optional[list[int]], Optional[list[int]]]:
    if metadata is None or not hasattr(metadata, "voxel_ids") or metadata.voxel_ids is None:
        return None, None
    valid = getattr(metadata, "voxel_valid", None)
    if valid is not None:
        if not bool(valid[batch_id, head_id, ev_idx].item()) or not bool(valid[batch_id, head_id, near_idx].item()):
            return None, None
    return [int(x) for x in metadata.voxel_ids[batch_id, head_id, ev_idx].tolist()], [int(x) for x in metadata.voxel_ids[batch_id, head_id, near_idx].tolist()]


def _optional_float(values: Optional[torch.Tensor], idx: int) -> Optional[float]:
    if values is None:
        return None
    value = float(values[idx].item())
    return None if not math.isfinite(value) else value


def _optional_int(values: Optional[torch.Tensor], idx: int) -> Optional[int]:
    if values is None:
        return None
    return int(values[idx].item())


def _save_hist(path: Path, cosine_distance: torch.Tensor) -> None:
    try:
        import numpy as np
    except ImportError:
        return
    dist = cosine_distance.detach().cpu().float()
    counts = torch.histc(dist, bins=50, min=0.0, max=2.0).to(torch.int64).cpu().numpy()
    bins = torch.linspace(0.0, 2.0, steps=51).cpu().numpy()
    np.savez(path, bins=bins, counts=counts, cosine_distances_sampled=dist.numpy())


def _record_skipped_svd_event(
    config: EvictionNNAnalysisConfig,
    batch_id: int,
    head_id: Optional[int],
    layer_id: int,
    step_idx: int,
    cache_budget: int,
    num_anchor_tokens: int,
    eviction_policy: str,
    leverage_granularity: str,
    selection_granularity: str,
    total_tokens_before_eviction: int,
    retained_count: int,
    evicted_count: int,
    reason: str,
) -> None:
    event_path = config.next_path(step_idx=step_idx, layer_id=layer_id, head_id=head_id, batch_id=batch_id, space="svd_coord")
    summary = _base_summary(
        batch_id=batch_id,
        head_id=head_id,
        layer_idx=layer_id,
        step_idx=step_idx,
        cache_budget=cache_budget,
        num_anchor_tokens=num_anchor_tokens,
        eviction_policy=eviction_policy,
        leverage_granularity=leverage_granularity,
        selection_granularity=selection_granularity,
        analysis_space="svd_coord",
        analysis_feature="signed_q",
        total_tokens_before_eviction=total_tokens_before_eviction,
        retained_count=retained_count,
        evicted_count=evicted_count,
        analyzed_evicted_count=0,
        sampled=False,
    )
    summary.update(_empty_distance_summary())
    summary["svd_coord_available"] = False
    summary["skip_reason"] = reason
    config.record(summary, event_path, top_pairs=[])


def _print_eviction_nn_debug(summary: dict[str, Any]) -> None:
    p50 = summary.get("cosine_distance_p50")
    p90 = summary.get("cosine_distance_p90")
    frac = summary.get("frac_dist_gt_0_20")
    fmt = lambda value: "nan" if value is None else f"{float(value):.4f}"
    print(
        "[EvictionNN] "
        f"step={summary.get('step_idx')} layer={summary.get('layer_idx')} head={summary.get('head_idx')} "
        f"evicted={summary.get('evicted_count')} analyzed={summary.get('analyzed_evicted_count')} "
        f"dist_p50={fmt(p50)} dist_p90={fmt(p90)} frac_dist_gt_0.2={fmt(frac)}"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class PreEvictionSnapshotConfig:
    """Opt-in controls for dumping a shared cache snapshot before eviction."""

    output_dir: str
    frame_count: int = 40
    layers: Optional[set[int]] = None
    heads: Optional[set[int]] = None
    max_snapshots: Optional[int] = None

    def __post_init__(self) -> None:
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self._num_snapshots = 0
        self._dumped_steps: set[tuple[int, int, int, int]] = set()
        if self.frame_count < 1:
            raise ValueError(f"frame_count must be >= 1, got {self.frame_count}")

    @classmethod
    def from_cli(
        cls,
        enabled: bool,
        output_dir: Optional[str],
        frame_count: int = 40,
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        max_snapshots: Optional[int] = None,
    ) -> Optional["PreEvictionSnapshotConfig"]:
        if not enabled:
            return None
        if not output_dir:
            raise ValueError("--snapshot_output_dir is required when --snapshot_before_eviction is set")
        return cls(
            output_dir=output_dir,
            frame_count=frame_count,
            layers=parse_index_filter(layers),
            heads=parse_index_filter(heads),
            max_snapshots=max_snapshots,
        )

    @property
    def target_step_idx(self) -> int:
        return self.frame_count - 1

    def should_dump(self, layer_id: int, head_id: int, step_idx: int, batch_id: int) -> bool:
        if step_idx != self.target_step_idx:
            return False
        if self.max_snapshots is not None and self._num_snapshots >= self.max_snapshots:
            return False
        if self.layers is not None and layer_id not in self.layers:
            return False
        if self.heads is not None and head_id not in self.heads:
            return False
        key = (int(layer_id), int(head_id), int(step_idx), int(batch_id))
        if key in self._dumped_steps:
            return False
        return True

    def record_dump(self, layer_id: int, head_id: int, step_idx: int, batch_id: int) -> None:
        self._dumped_steps.add((int(layer_id), int(head_id), int(step_idx), int(batch_id)))
        self._num_snapshots += 1


def dump_pre_eviction_snapshot(
    config: PreEvictionSnapshotConfig,
    *,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    layer_id: int,
    step_idx: int,
    cache_budget: Optional[int],
    num_anchor_tokens: int,
    tokens_per_frame: Optional[int] = None,
) -> None:
    """Save per-head K/V cache before any eviction is applied."""
    B, H, N, _ = k_cache.shape
    token_index = torch.arange(N, dtype=torch.long)
    token_frame_index = None
    provenance_note = "not_tracked_by_cache"
    if tokens_per_frame is not None and tokens_per_frame > 0:
        token_frame_index = token_index // int(tokens_per_frame)
        provenance_note = "approximate_from_current_cache_order_only"

    for batch_id in range(B):
        for head_id in range(H):
            if not config.should_dump(layer_id, head_id, step_idx, batch_id):
                continue

            payload = {
                "old_key": k_cache[batch_id, head_id].detach().to("cpu", torch.float32),
                "old_value": None
                if v_cache is None
                else v_cache[batch_id, head_id].detach().to("cpu", torch.float32),
                "anchor_token_indices": torch.arange(num_anchor_tokens, dtype=torch.long),
                "token_indices": token_index,
                "token_frame_indices": token_frame_index,
                "meta": {
                    "snapshot_type": "pre_eviction_common_cache",
                    "layer_id": int(layer_id),
                    "head_id": int(head_id),
                    "batch_id": int(batch_id),
                    "step_idx": int(step_idx),
                    "frame_count": int(step_idx) + 1,
                    "cache_size": int(N),
                    "cache_budget": None if cache_budget is None else int(cache_budget),
                    "num_anchor_tokens": int(num_anchor_tokens),
                    "candidate_count": int(N - num_anchor_tokens),
                    "tokens_per_frame": None if tokens_per_frame is None else int(tokens_per_frame),
                    "provenance": provenance_note,
                    "policy": "snapshot saved before any eviction at this step",
                },
            }

            stem = f"pre_step{step_idx:06d}_layer{layer_id:02d}_head{head_id:02d}_batch{batch_id:02d}"
            torch.save(payload, os.path.join(config.output_dir, f"{stem}.pt"), pickle_protocol=4)
            with open(os.path.join(config.output_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(payload["meta"], f, indent=2)
            config.record_dump(layer_id, head_id, step_idx, batch_id)
