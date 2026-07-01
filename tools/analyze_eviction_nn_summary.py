#!/usr/bin/env python3
"""Aggregate eviction nearest-retained analysis summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "cosine_distance_mean",
    "cosine_distance_p50",
    "cosine_distance_p90",
    "cosine_distance_p95",
    "frac_dist_gt_0_10",
    "frac_dist_gt_0_20",
    "frac_sim_lt_0_90",
    "r_knn",
    "retained_knn_distance_mean",
    "pre_knn_distance_mean",
    "r_knn_valid_count",
    "frame_gap_mean",
    "frac_same_frame",
)
QUANTILE_FIELDS = (
    "cosine_distance_p50",
    "cosine_distance_p75",
    "cosine_distance_p90",
    "cosine_distance_p95",
    "cosine_distance_p99",
)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _read_records(input_dirs: list[str], names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for input_dir, name in zip(input_dirs, names):
        root = Path(input_dir)
        for summary_path in sorted(root.rglob("summary.jsonl")):
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec["dataset"] = name
                    rec["summary_path"] = str(summary_path)
                    records.append(rec)
    return records


def _group_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[tuple(rec.get(key) for key in keys)].append(rec)
    return groups


def _summarize_group(group: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    row["events"] = len(group)
    row["evicted_tokens"] = sum(int(rec.get("evicted_count") or 0) for rec in group)
    row["analyzed_tokens"] = sum(int(rec.get("analyzed_evicted_count") or 0) for rec in group)
    for metric in METRICS:
        values = [_float_or_none(rec.get(metric)) for rec in group]
        values = [value for value in values if value is not None]
        if metric == "cosine_distance_p50":
            row[f"median_{metric}"] = _median(values)
        elif metric == "r_knn":
            row[f"mean_{metric}"] = _mean(values)
            row[f"median_{metric}"] = _median(values)
        else:
            row[f"mean_{metric}"] = _mean(values)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summaries(records: list[dict[str, Any]], output_dir: Path) -> None:
    overall = [
        _summarize_group(group, {"dataset": dataset})
        for (dataset,), group in sorted(_group_records(records, ("dataset",)).items())
    ]
    layer = [
        _summarize_group(group, {"dataset": dataset, "layer_idx": layer_idx})
        for (dataset, layer_idx), group in sorted(_group_records(records, ("dataset", "layer_idx")).items())
    ]
    head = [
        _summarize_group(group, {"dataset": dataset, "layer_idx": layer_idx, "head_idx": head_idx})
        for (dataset, layer_idx, head_idx), group in sorted(_group_records(records, ("dataset", "layer_idx", "head_idx")).items())
    ]
    quantiles = []
    for (dataset,), group in sorted(_group_records(records, ("dataset",)).items()):
        row = {"dataset": dataset, "events": len(group)}
        for field in QUANTILE_FIELDS:
            values = [_float_or_none(rec.get(field)) for rec in group]
            values = [value for value in values if value is not None]
            row[f"mean_{field}"] = _mean(values)
            row[f"median_{field}"] = _median(values)
        quantiles.append(row)
    _write_csv(output_dir / "overall_summary.csv", overall)
    _write_csv(output_dir / "layer_summary.csv", layer)
    _write_csv(output_dir / "head_summary.csv", head)
    _write_csv(output_dir / "distance_quantiles.csv", quantiles)


def _maybe_write_plots(records: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    by_dataset = _group_records(records, ("dataset",))
    labels = []
    box_values = []
    for (dataset,), group in sorted(by_dataset.items()):
        values = [_float_or_none(rec.get("cosine_distance_p50")) for rec in group]
        values = [value for value in values if value is not None]
        if values:
            labels.append(str(dataset))
            box_values.append(values)
    if box_values:
        plt.figure(figsize=(max(6, len(labels) * 1.5), 4))
        plt.boxplot(box_values, labels=labels, showfliers=False)
        plt.ylabel("Event p50 nearest cosine distance")
        plt.tight_layout()
        plt.savefig(output_dir / "compare_boxplot.png", dpi=160)
        plt.close()

    layer_groups = _group_records(records, ("dataset", "layer_idx"))
    datasets = sorted({rec.get("dataset") for rec in records})
    layers = sorted({rec.get("layer_idx") for rec in records if rec.get("layer_idx") is not None})
    if datasets and layers:
        plt.figure(figsize=(max(7, len(layers) * 0.45), 4))
        for dataset in datasets:
            y = []
            x = []
            for layer in layers:
                group = layer_groups.get((dataset, layer), [])
                values = [_float_or_none(rec.get("cosine_distance_p90")) for rec in group]
                values = [value for value in values if value is not None]
                if values:
                    x.append(layer)
                    y.append(sum(values) / len(values))
            if x:
                plt.plot(x, y, marker="o", label=str(dataset))
        plt.xlabel("Layer")
        plt.ylabel("Mean event p90 nearest cosine distance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "compare_p90_by_layer.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if len(args.input_dirs) != len(args.names):
        raise SystemExit("--input_dirs and --names must have the same length")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(args.input_dirs, args.names)
    _write_summaries(records, output_dir)
    _maybe_write_plots(records, output_dir)
    print(f"Loaded {len(records)} event summaries into {output_dir}")


if __name__ == "__main__":
    main()
