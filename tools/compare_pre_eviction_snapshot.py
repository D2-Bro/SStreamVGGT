#!/usr/bin/env python3
"""Compare mean and SVD-leverage eviction on the same pre-eviction snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from streamvggt.layers.eviction import EvictionManager


def _snapshot_group_key(meta: dict[str, Any]) -> tuple[int, int, int]:
    """Group per-head snapshots into one layer snapshot."""
    return (
        int(meta.get("step_idx", -1)),
        int(meta.get("layer_id", -1)),
        int(meta.get("batch_id", 0)),
    )


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _score_summary(scores: torch.Tensor) -> dict[str, float]:
    if scores.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    scores = scores.float()
    return {
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
        "mean": float(scores.mean().item()),
    }


def _select_policy(
    keys: torch.Tensor,
    *,
    policy: str,
    cache_budget: int,
    num_anchor_tokens: int,
) -> dict[str, torch.Tensor]:
    manager = EvictionManager(policy=policy)
    result = manager.select(
        keys.unsqueeze(0).unsqueeze(0),
        cache_budget=cache_budget,
        num_anchor_tokens=num_anchor_tokens,
    )
    kept_candidate = result.kept_candidate_indices[0, 0].cpu().long()
    policy_scores = result.policy_scores[0, 0].cpu().float()
    mean_scores = result.mean_scores[0, 0].cpu().float()

    candidate_count = keys.shape[0] - num_anchor_tokens
    all_candidate = torch.arange(candidate_count, dtype=torch.long)
    evict_mask = torch.ones(candidate_count, dtype=torch.bool)
    evict_mask[kept_candidate] = False
    evicted_candidate = all_candidate[evict_mask]

    anchors = torch.arange(num_anchor_tokens, dtype=torch.long)
    kept_token = torch.cat([anchors, kept_candidate + num_anchor_tokens])
    evicted_token = evicted_candidate + num_anchor_tokens

    return {
        "kept_candidate_indices": kept_candidate,
        "evicted_candidate_indices": evicted_candidate,
        "retain_indices": kept_token,
        "evict_indices": evicted_token,
        "policy_scores": policy_scores,
        "mean_scores": mean_scores,
    }


def _indices_from_candidate_scores(
    scores: torch.Tensor,
    *,
    num_anchor_tokens: int,
    num_to_keep: int,
    keep_highest: bool,
) -> dict[str, torch.Tensor]:
    """Convert candidate scores into global retain/evict indices."""
    candidate_count = int(scores.numel())
    if num_to_keep < 0:
        raise ValueError(f"num_to_keep must be >= 0, got {num_to_keep}")
    if num_to_keep > candidate_count:
        raise ValueError(f"Cannot keep {num_to_keep} candidates from {candidate_count}")

    ranked_scores = scores if keep_highest else -scores
    _, kept_candidate = torch.topk(ranked_scores, k=num_to_keep, dim=0)
    kept_candidate = kept_candidate.sort().values.cpu().long()

    all_candidate = torch.arange(candidate_count, dtype=torch.long)
    evict_mask = torch.ones(candidate_count, dtype=torch.bool)
    evict_mask[kept_candidate] = False
    evicted_candidate = all_candidate[evict_mask]

    anchors = torch.arange(num_anchor_tokens, dtype=torch.long)
    return {
        "kept_candidate_indices": kept_candidate,
        "evicted_candidate_indices": evicted_candidate,
        "retain_indices": torch.cat([anchors, kept_candidate + num_anchor_tokens]),
        "evict_indices": evicted_candidate + num_anchor_tokens,
        "policy_scores": scores.cpu().float(),
    }


def _svd_leverage_scores(matrix: torch.Tensor) -> torch.Tensor:
    """Compute row leverage scores for a 2D token-feature matrix."""
    if matrix.numel() == 0:
        return torch.empty(matrix.shape[0], dtype=torch.float32)
    mat = matrix.float()
    try:
        u, _, _ = torch.linalg.svd(mat, full_matrices=False)
        return u.square().sum(dim=1).cpu().float()
    except RuntimeError:
        return mat.square().sum(dim=1).cpu().float()


def build_layer_svd_results(
    paths: list[Path],
    args: argparse.Namespace,
) -> dict[Path, dict[str, torch.Tensor]]:
    """Compute one SVD-leverage eviction decision per layer across all heads."""
    snapshots_by_group: dict[tuple[int, int, int], list[tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        meta = torch.load(path, map_location="cpu").get("meta", {})
        snapshots_by_group.setdefault(_snapshot_group_key(meta), []).append((path, meta))

    results: dict[Path, dict[str, torch.Tensor]] = {}
    for group_key, group_items in sorted(snapshots_by_group.items()):
        group_items = sorted(group_items, key=lambda item: int(item[1].get("head_id", 0)))
        loaded = [(path, torch.load(path, map_location="cpu")) for path, _ in group_items]
        first_payload = loaded[0][1]
        first_meta = dict(first_payload.get("meta", {}))
        cache_size = int(first_payload["old_key"].shape[0])
        num_anchor_tokens = int(first_meta.get("num_anchor_tokens", 0))
        eviction_count = _resolve_eviction_count({**first_meta, "cache_size": cache_size}, args)
        num_to_keep = cache_size - num_anchor_tokens - eviction_count

        for path, payload in loaded:
            meta = dict(payload.get("meta", {}))
            if int(payload["old_key"].shape[0]) != cache_size:
                raise ValueError(f"Layer-wise SVD group {group_key} has mismatched cache sizes")
            if int(meta.get("num_anchor_tokens", 0)) != num_anchor_tokens:
                raise ValueError(f"Layer-wise SVD group {group_key} has mismatched anchor token counts")

        candidate_keys = [
            payload["old_key"][num_anchor_tokens:].float()
            for _, payload in loaded
        ]
        layer_matrix = torch.cat(candidate_keys, dim=1)
        scores = _svd_leverage_scores(layer_matrix)
        layer_result = _indices_from_candidate_scores(
            scores,
            num_anchor_tokens=num_anchor_tokens,
            num_to_keep=num_to_keep,
            keep_highest=True,
        )
        layer_result["score_definition"] = "row leverage scores from SVD over concatenated per-head keys in one layer"

        step_idx, layer_id, batch_id = group_key
        print(
            f"layer-wise svd layer={layer_id:02d} batch={batch_id:02d} step={step_idx:06d} "
            f"heads={len(loaded)} tokens={cache_size} evict={eviction_count}"
        )
        for path, _ in loaded:
            results[path] = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in layer_result.items()
            }
    return results


def _resolve_eviction_count(meta: dict[str, Any], args: argparse.Namespace) -> int:
    cache_size = int(meta["cache_size"])
    num_anchor_tokens = int(meta.get("num_anchor_tokens", 0))
    candidate_count = cache_size - num_anchor_tokens

    if args.eviction_count is not None:
        eviction_count = int(args.eviction_count)
    else:
        cache_budget = args.cache_budget
        if cache_budget is None:
            cache_budget = _safe_int(meta.get("cache_budget"), cache_size)
        eviction_count = max(0, cache_size - int(cache_budget))

    if eviction_count < 0:
        raise ValueError(f"eviction_count must be >= 0, got {eviction_count}")
    return min(eviction_count, candidate_count)


def _overlap_ratio(left: torch.Tensor, right: torch.Tensor) -> tuple[int, float]:
    left_set = set(left.tolist())
    right_set = set(right.tolist())
    denom = max(len(left_set), 1)
    overlap = len(left_set & right_set)
    return overlap, float(overlap / denom)


def compare_snapshot(
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    layer_svd_results: dict[Path, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    keys = payload["old_key"].float()
    values = payload.get("old_value")
    meta = dict(payload.get("meta", {}))
    cache_size = int(keys.shape[0])
    num_anchor_tokens = int(meta.get("num_anchor_tokens", 0))
    eviction_count = _resolve_eviction_count({**meta, "cache_size": cache_size}, args)
    cache_budget = cache_size - eviction_count

    mean = _select_policy(
        keys,
        policy="mean",
        cache_budget=cache_budget,
        num_anchor_tokens=num_anchor_tokens,
    )
    svd = _select_policy(
        keys,
        policy="svd_leverage",
        cache_budget=cache_budget,
        num_anchor_tokens=num_anchor_tokens,
    )
    layer_svd = None
    if layer_svd_results is not None:
        layer_svd = layer_svd_results[path]

    overlap, overlap_ratio = _overlap_ratio(mean["evict_indices"], svd["evict_indices"])

    result_payload: dict[str, Any] = {
        "source_snapshot": str(path),
        "meta": {
            **meta,
            "comparison_cache_budget": int(cache_budget),
            "comparison_eviction_count": int(eviction_count),
        },
        "mean": mean,
        "svd_leverage": svd,
        "summary": {
            "total_token_count": int(cache_size),
            "candidate_count": int(cache_size - num_anchor_tokens),
            "eviction_count": int(eviction_count),
            "mean_evicted_count": int(mean["evict_indices"].numel()),
            "svd_leverage_evicted_count": int(svd["evict_indices"].numel()),
            "evicted_overlap_count": int(overlap),
            "evicted_overlap_ratio": overlap_ratio,
            "mean_score_summary": _score_summary(mean["policy_scores"]),
            "svd_leverage_score_summary": _score_summary(svd["policy_scores"]),
        },
    }
    if layer_svd is not None:
        layer_overlap, layer_overlap_ratio = _overlap_ratio(mean["evict_indices"], layer_svd["evict_indices"])
        head_vs_layer_overlap, head_vs_layer_overlap_ratio = _overlap_ratio(
            svd["evict_indices"], layer_svd["evict_indices"]
        )
        result_payload["layer_svd_leverage"] = layer_svd
        result_payload["summary"].update(
            {
                "layer_svd_leverage_evicted_count": int(layer_svd["evict_indices"].numel()),
                "mean_vs_layer_svd_evicted_overlap_count": int(layer_overlap),
                "mean_vs_layer_svd_evicted_overlap_ratio": layer_overlap_ratio,
                "head_svd_vs_layer_svd_evicted_overlap_count": int(head_vs_layer_overlap),
                "head_svd_vs_layer_svd_evicted_overlap_ratio": head_vs_layer_overlap_ratio,
                "layer_svd_leverage_score_summary": _score_summary(layer_svd["policy_scores"]),
            }
        )

    if args.save_tensors:
        result_payload["mean"]["retained_key"] = keys[mean["retain_indices"]]
        result_payload["mean"]["evicted_key"] = keys[mean["evict_indices"]]
        result_payload["svd_leverage"]["retained_key"] = keys[svd["retain_indices"]]
        result_payload["svd_leverage"]["evicted_key"] = keys[svd["evict_indices"]]
        if layer_svd is not None:
            result_payload["layer_svd_leverage"]["retained_key"] = keys[layer_svd["retain_indices"]]
            result_payload["layer_svd_leverage"]["evicted_key"] = keys[layer_svd["evict_indices"]]
        if values is not None:
            result_payload["mean"]["retained_value"] = values[mean["retain_indices"]]
            result_payload["mean"]["evicted_value"] = values[mean["evict_indices"]]
            result_payload["svd_leverage"]["retained_value"] = values[svd["retain_indices"]]
            result_payload["svd_leverage"]["evicted_value"] = values[svd["evict_indices"]]
            if layer_svd is not None:
                result_payload["layer_svd_leverage"]["retained_value"] = values[layer_svd["retain_indices"]]
                result_payload["layer_svd_leverage"]["evicted_value"] = values[layer_svd["evict_indices"]]

    out_path = output_dir / f"{path.stem}_mean_vs_svd.pt"
    torch.save(result_payload, out_path, pickle_protocol=4)

    json_path = output_dir / f"{path.stem}_mean_vs_svd.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_snapshot": str(path),
                "meta": result_payload["meta"],
                "summary": result_payload["summary"],
                "result_file": str(out_path),
            },
            f,
            indent=2,
        )

    layer_id = int(meta.get("layer_id", -1))
    head_id = int(meta.get("head_id", -1))
    print(
        f"layer={layer_id:02d} head={head_id:02d} tokens={cache_size} "
        f"evict={eviction_count} mean={mean['evict_indices'].numel()} "
        f"svd={svd['evict_indices'].numel()} overlap={overlap_ratio:.4f}"
        + (
            f" layer_svd={layer_svd['evict_indices'].numel()} "
            f"mean_layer_overlap={result_payload['summary']['mean_vs_layer_svd_evicted_overlap_ratio']:.4f}"
            if layer_svd is not None
            else ""
        )
    )

    return {
        "snapshot": str(path),
        "result_file": str(out_path),
        "layer_id": layer_id,
        "head_id": head_id,
        **result_payload["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snapshot_dir",
        "--snapshot-dir",
        type=str,
        required=True,
        help="Directory with pre-eviction .pt snapshots",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        required=True,
        help="Directory for comparison result files",
    )
    parser.add_argument(
        "--eviction_count",
        "--eviction-count",
        type=int,
        default=None,
        help="Number of candidate tokens to evict per head; defaults to cache_size - snapshot cache_budget",
    )
    parser.add_argument(
        "--cache_budget",
        "--cache-budget",
        type=int,
        default=None,
        help="Override final cache budget; ignored when eviction_count is set",
    )
    parser.add_argument(
        "--max_snapshots",
        "--max-snapshots",
        type=int,
        default=None,
        help="Optional cap for smoke runs",
    )
    parser.add_argument(
        "--save_tensors",
        "--save-tensors",
        action="store_true",
        help="Also save evicted/retained K/V tensors",
    )
    parser.add_argument(
        "--include_layer_svd_leverage",
        "--include-layer-svd-leverage",
        action="store_true",
        help="Also compute one SVD-leverage eviction decision per layer by concatenating all head keys",
    )
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(snapshot_dir.glob("*.pt"))
    if args.max_snapshots is not None:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise FileNotFoundError(f"No .pt snapshots found in {snapshot_dir}")

    layer_svd_results = build_layer_svd_results(paths, args) if args.include_layer_svd_leverage else None
    rows = [compare_snapshot(path, args, output_dir, layer_svd_results) for path in paths]

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "snapshot",
        "result_file",
        "layer_id",
        "head_id",
        "total_token_count",
        "candidate_count",
        "eviction_count",
        "mean_evicted_count",
        "svd_leverage_evicted_count",
        "layer_svd_leverage_evicted_count",
        "evicted_overlap_count",
        "evicted_overlap_ratio",
        "mean_vs_layer_svd_evicted_overlap_count",
        "mean_vs_layer_svd_evicted_overlap_ratio",
        "head_svd_vs_layer_svd_evicted_overlap_count",
        "head_svd_vs_layer_svd_evicted_overlap_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} comparison results to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
