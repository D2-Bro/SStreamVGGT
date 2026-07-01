#!/usr/bin/env python3
"""Summarize layer-budget score logs into frame-wise tables and histograms."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_frame_wide(rows: list[dict[str, str]], output_path: Path) -> tuple[list[int], dict[int, list[float]], list[dict[str, object]]]:
    by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    layers = sorted({int(row["layer"]) for row in rows})
    for row in rows:
        by_step[int(row["step"])].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "strategy",
        "total_budget",
        "assigned_budget",
        "total_capacity",
        "active_layers",
        "mean_score",
    ] + [f"score_layer_{layer:02d}" for layer in layers]

    layer_scores: dict[int, list[float]] = {layer: [] for layer in layers}
    frame_rows: list[dict[str, object]] = []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(by_step):
            step_rows = by_step[step]
            row_by_layer = {int(row["layer"]): row for row in step_rows}
            scores = []
            out: dict[str, object] = {
                "step": step,
                "strategy": step_rows[0].get("strategy", ""),
                "total_budget": step_rows[0].get("total_budget", ""),
                "assigned_budget": step_rows[0].get("assigned_budget", ""),
                "total_capacity": step_rows[0].get("total_capacity", ""),
                "active_layers": step_rows[0].get("active_layers", ""),
            }
            for layer in layers:
                score = _float_or_zero(row_by_layer.get(layer, {}).get("score", "0"))
                out[f"score_layer_{layer:02d}"] = f"{score:.9g}"
                scores.append(score)
                layer_scores[layer].append(score)
            out["mean_score"] = f"{mean(scores):.9g}" if scores else "0"
            writer.writerow(out)
            frame_rows.append(out)
    return layers, layer_scores, frame_rows


def write_layer_means(layer_scores: dict[int, list[float]], output_path: Path) -> list[tuple[int, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_score", "num_frames"])
        writer.writeheader()
        for layer in sorted(layer_scores):
            values = layer_scores[layer]
            layer_mean = mean(values) if values else 0.0
            means.append((layer, layer_mean))
            writer.writerow({"layer": layer, "mean_score": f"{layer_mean:.9g}", "num_frames": len(values)})
    return means


def plot_layer_mean_bar(layer_means: list[tuple[int, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [layer for layer, _ in layer_means]
    values = [value for _, value in layer_means]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.bar(layers, values, color="#4C78A8", edgecolor="white")
    ax.set_title("Mean Layer-Budget Score by Layer")
    ax.set_xlabel("layer number")
    ax.set_ylabel("mean score")
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_frame_mean_hist(values: list[float], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    bins = min(24, max(5, len(values)))
    ax.hist(values, bins=bins, color="#72B7B2", edgecolor="white")
    ax.set_title("Frame Mean Layer-Budget Score Distribution")
    ax.set_xlabel("mean score across layers")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir or input_csv.parent
    rows = read_rows(input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {input_csv}")

    frame_csv = output_dir / "layer_budget_scores_by_frame.csv"
    layer_mean_csv = output_dir / "layer_budget_layer_mean_scores.csv"
    layer_bar = output_dir / "layer_budget_layer_mean_score_by_layer.png"
    frame_hist = output_dir / "layer_budget_frame_mean_score_hist.png"

    _, layer_scores, frame_rows = write_frame_wide(rows, frame_csv)
    layer_means = write_layer_means(layer_scores, layer_mean_csv)
    frame_means = [_float_or_zero(str(row["mean_score"])) for row in frame_rows]
    plot_layer_mean_bar(layer_means, layer_bar)
    plot_frame_mean_hist(frame_means, frame_hist)

    print(f"wrote {frame_csv}")
    print(f"wrote {layer_mean_csv}")
    print(f"wrote {layer_bar}")
    print(f"wrote {frame_hist}")


if __name__ == "__main__":
    main()
