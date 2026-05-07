"""Analyze normalized coverage distance on pre-eviction comparison outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


POLICIES = ("mean", "svd_leverage", "layer_svd_leverage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def resolve_source(path: str | Path, compare_path: Path) -> Path:
    source = Path(path)
    candidates = [source]
    if not source.is_absolute():
        candidates.extend([Path.cwd() / source, compare_path.parent / source])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def ncd_at_k(
    keys: torch.Tensor,
    evicted: torch.Tensor,
    k: int,
    *,
    chunk_size: int,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute NCD@k for evicted tokens in original key space."""
    keys_norm = F.normalize(keys.float(), dim=-1)
    evicted = evicted.long()
    retained_mask = torch.ones(keys_norm.shape[0], dtype=torch.bool)
    retained_mask[evicted] = False

    all_t = keys_norm.T.contiguous()
    retained_t = keys_norm[retained_mask].T.contiguous()
    ratios = []
    after_dists = []
    local_scales = []

    for start in range(0, evicted.numel(), chunk_size):
        idx = evicted[start : start + chunk_size]
        q = keys_norm[idx]

        sim_all = q @ all_t
        sim_all[torch.arange(idx.numel()), idx] = -float("inf")
        topk = torch.topk(sim_all, k=min(k, keys_norm.shape[0] - 1), dim=1).values
        local_scale = (1.0 - topk).clamp_min(0.0).mean(dim=1)

        sim_retained = q @ retained_t
        after_dist = (1.0 - sim_retained.max(dim=1).values).clamp_min(0.0)

        ratios.append(after_dist / local_scale.clamp_min(eps))
        after_dists.append(after_dist)
        local_scales.append(local_scale)

    ratio = torch.cat(ratios)
    after = torch.cat(after_dists)
    local = torch.cat(local_scales)
    return {
        "mean": float(ratio.mean().item()),
        "median": float(torch.quantile(ratio, 0.50).item()),
        "p95": float(torch.quantile(ratio, 0.95).item()),
        "p99": float(torch.quantile(ratio, 0.99).item()),
        "max": float(ratio.max().item()),
        "frac_gt_2": float((ratio > 2.0).float().mean().item()),
        "frac_gt_5": float((ratio > 5.0).float().mean().item()),
        "frac_gt_10": float((ratio > 10.0).float().mean().item()),
        "after_dist_mean": float(after.mean().item()),
        "local_scale_mean": float(local.mean().item()),
    }


def stat(values: list[float]) -> dict[str, float] | None:
    values = sorted(v for v in values if isinstance(v, (int, float)) and not math.isnan(v))
    if not values:
        return None

    def quantile(p: float) -> float:
        return values[int(round((len(values) - 1) * p))]

    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p05": quantile(0.05),
        "p95": quantile(0.95),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:.6g} median={stats['median']:.6g} "
        f"p05={stats['p05']:.6g} p95={stats['p95']:.6g} "
        f"min={stats['min']:.6g} max={stats['max']:.6g}"
    )


def main() -> None:
    args = parse_args()
    paths = sorted(args.comparison_dir.glob("*_mean_vs_svd.pt"))
    if args.max_files is not None:
        paths = paths[: args.max_files]

    rows = []
    for n, path in enumerate(paths, 1):
        comparison = torch.load(path, map_location="cpu")
        source = torch.load(resolve_source(comparison["source_snapshot"], path), map_location="cpu")
        keys = source["old_key"].float()
        meta = comparison["meta"]

        for policy in POLICIES:
            if policy not in comparison:
                continue
            row = {
                "policy": policy,
                "layer_id": int(meta["layer_id"]),
                "head_id": int(meta["head_id"]),
            }
            evicted = comparison[policy]["evict_indices"].long()
            for k in args.k:
                metrics = ncd_at_k(keys, evicted, k, chunk_size=args.chunk_size)
                row.update({f"ncd{k}_{name}": value for name, value in metrics.items()})
            rows.append(row)

        if n % 16 == 0:
            print(f"processed {n}/{len(paths)}", flush=True)

    print(f"rows {len(rows)}")

    summary: dict[str, object] = {"rows": rows, "aggregates": {}}
    aggregates = summary["aggregates"]
    assert isinstance(aggregates, dict)

    for k in args.k:
        print(f"\n=== NCD@{k} token distribution summarized per head ===")
        aggregates[f"ncd{k}"] = {}
        for metric in ("mean", "median", "p95", "p99", "max", "frac_gt_2", "frac_gt_5", "frac_gt_10"):
            print(f"\nncd{k}_{metric}")
            aggregates[f"ncd{k}"][metric] = {}
            for policy in POLICIES:
                stats = stat([row[f"ncd{k}_{metric}"] for row in rows if row["policy"] == policy])
                if stats is None:
                    continue
                aggregates[f"ncd{k}"][metric][policy] = stats
                print(f"{policy}: {fmt_stats(stats)}")

    for k in args.k:
        for metric in ("p95", "frac_gt_5", "frac_gt_10"):
            print(f"\nlayer means ncd{k}_{metric}")
            for policy in POLICIES:
                by_layer = defaultdict(list)
                for row in rows:
                    if row["policy"] == policy:
                        by_layer[row["layer_id"]].append(row[f"ncd{k}_{metric}"])
                print(
                    policy,
                    " ".join(
                        f"{layer}:{sum(values) / len(values):.4g}"
                        for layer, values in sorted(by_layer.items())
                    ),
                )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
