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


def write_budget_frame_wide(
    rows: list[dict[str, str]], output_path: Path
) -> tuple[list[int], dict[int, list[float]], list[dict[str, object]]]:
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
        "mean_budget",
    ] + [f"budget_layer_{layer:02d}" for layer in layers]

    layer_budgets: dict[int, list[float]] = {layer: [] for layer in layers}
    frame_rows: list[dict[str, object]] = []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(by_step):
            step_rows = by_step[step]
            row_by_layer = {int(row["layer"]): row for row in step_rows}
            budgets = []
            out: dict[str, object] = {
                "step": step,
                "strategy": step_rows[0].get("strategy", ""),
                "total_budget": step_rows[0].get("total_budget", ""),
                "assigned_budget": step_rows[0].get("assigned_budget", ""),
                "total_capacity": step_rows[0].get("total_capacity", ""),
                "active_layers": step_rows[0].get("active_layers", ""),
            }
            for layer in layers:
                budget = _float_or_zero(row_by_layer.get(layer, {}).get("final_budget", "0"))
                out[f"budget_layer_{layer:02d}"] = f"{budget:.9g}"
                budgets.append(budget)
                layer_budgets[layer].append(budget)
            out["mean_budget"] = f"{mean(budgets):.9g}" if budgets else "0"
            writer.writerow(out)
            frame_rows.append(out)
    return layers, layer_budgets, frame_rows


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


def write_layer_mean_budgets(layer_budgets: dict[int, list[float]], output_path: Path) -> list[tuple[int, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_budget", "num_frames"])
        writer.writeheader()
        for layer in sorted(layer_budgets):
            values = layer_budgets[layer]
            layer_mean = mean(values) if values else 0.0
            means.append((layer, layer_mean))
            writer.writerow({"layer": layer, "mean_budget": f"{layer_mean:.9g}", "num_frames": len(values)})
    return means


def plot_layer_mean_bar(
    layer_means: list[tuple[int, float]],
    output_path: Path,
    title: str = "Mean Layer-Budget Score by Layer",
    ylabel: str = "mean score",
    color: str = "#F8A054",
    ymax: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [layer for layer, _ in layer_means]
    values = [value for _, value in layer_means]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.bar(layers, values, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("layer number")
    ax.set_ylabel(ylabel)
    ax.set_xticks(layers)
    ax.tick_params(axis="both", labelsize=18)
    if ymax is not None:
        ax.set_ylim(0, ymax)
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


def get_step_budgets(rows: list[dict[str, str]], step: int) -> list[tuple[int, float]]:
    step_rows = [row for row in rows if int(row["step"]) == step]
    if not step_rows:
        available = sorted({int(row["step"]) for row in rows})
        raise SystemExit(f"Step {step} not found. Available steps: {available}")
    return [
        (int(row["layer"]), _float_or_zero(row.get("final_budget", "0")))
        for row in sorted(step_rows, key=lambda row: int(row["layer"]))
    ]


def plot_all_frame_layer_bars(
    rows: list[dict[str, str]],
    layers: list[int],
    output_dir: Path,
    *,
    value_field: str,
    filename_suffix: str,
    title_label: str,
    ylabel: str,
    color: str=None,
) -> list[Path]:
    by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)

    frame_values: list[tuple[int, str, list[tuple[int, float]]]] = []
    global_max = 0.0
    for step in sorted(by_step):
        step_rows = by_step[step]
        row_by_layer = {int(row["layer"]): row for row in step_rows}
        values = [
            (layer, _float_or_zero(row_by_layer.get(layer, {}).get(value_field, "0")))
            for layer in layers
        ]
        global_max = max(global_max, *(value for _, value in values))
        frame_values.append((step, step_rows[0].get("strategy", ""), values))

    shared_ymax = global_max * 1.05 if global_max > 0 else 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for step, strategy, values in frame_values:
        output_path = output_dir / f"frame_{step:06d}_{filename_suffix}_by_layer.png"
        strategy_suffix = f" ({strategy})" if strategy else ""
        plot_layer_mean_bar(
            values,
            output_path,
            title=f"{title_label} at Frame {step}{strategy_suffix}",
            ylabel=ylabel,
            ymax=shared_ymax,
        )
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None, help="Plot final_budget by layer for one step")
    parser.add_argument(
        "--latest-step",
        action="store_true",
        help="Plot final_budget by layer for the latest logged step",
    )
    args = parser.parse_args()
    if args.step is not None and args.latest_step:
        raise SystemExit("--step and --latest-step cannot be used together")

    input_csv = args.input_csv
    output_dir = args.output_dir or input_csv.parent
    rows = read_rows(input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {input_csv}")
    if "final_budget" not in rows[0]:
        raise SystemExit(
            f"{input_csv} is missing final_budget; budget plots require "
            "the layer_budget_scores.csv schema written by StreamVGGT."
        )

    frame_csv = output_dir / "layer_budget_scores_by_frame.csv"
    layer_mean_csv = output_dir / "layer_budget_layer_mean_scores.csv"
    layer_bar = output_dir / "layer_budget_layer_mean_score_by_layer.png"
    frame_hist = output_dir / "layer_budget_frame_mean_score_hist.png"
    budget_frame_csv = output_dir / "layer_budget_budgets_by_frame.csv"
    budget_layer_mean_csv = output_dir / "layer_budget_layer_mean_budgets.csv"
    budget_layer_bar = output_dir / "layer_budget_layer_mean_budget_by_layer.png"

    layers, layer_scores, frame_rows = write_frame_wide(rows, frame_csv)
    layer_means = write_layer_means(layer_scores, layer_mean_csv)
    frame_means = [_float_or_zero(str(row["mean_score"])) for row in frame_rows]
    plot_layer_mean_bar(layer_means, layer_bar)
    plot_frame_mean_hist(frame_means, frame_hist)
    _, layer_budgets, _ = write_budget_frame_wide(rows, budget_frame_csv)
    budget_layer_means = write_layer_mean_budgets(layer_budgets, budget_layer_mean_csv)
    plot_layer_mean_bar(
        budget_layer_means,
        budget_layer_bar,
        title="Mean Layer Budget by Layer",
        ylabel="mean final_budget",
        color="#F58518",
    )
    frame_score_dir = output_dir / "layer_budget_frame_scores"
    frame_budget_dir = output_dir / "layer_budget_frame_budgets"
    frame_score_plots = plot_all_frame_layer_bars(
        rows,
        layers,
        frame_score_dir,
        value_field="score",
        filename_suffix="score",
        title_label="Layer-Budget Score by Layer",
        ylabel="score",
    )
    frame_budget_plots = plot_all_frame_layer_bars(
        rows,
        layers,
        frame_budget_dir,
        value_field="final_budget",
        filename_suffix="budget",
        title_label="Final Layer Budget by Layer",
        ylabel="final_budget",
    )

    step_plot = None
    if args.latest_step:
        selected_step = max(int(row["step"]) for row in rows)
    else:
        selected_step = args.step
    if selected_step is not None:
        step_plot = output_dir / f"layer_budget_step_{selected_step:06d}_budget_by_layer.png"
        plot_layer_mean_bar(
            get_step_budgets(rows, selected_step),
            step_plot,
            title=f"Layer Budget by Layer at Step {selected_step}",
            ylabel="final_budget",
            color="#E45756",
        )

    print(f"wrote {frame_csv}")
    print(f"wrote {layer_mean_csv}")
    print(f"wrote {layer_bar}")
    print(f"wrote {frame_hist}")
    print(f"wrote {budget_frame_csv}")
    print(f"wrote {budget_layer_mean_csv}")
    print(f"wrote {budget_layer_bar}")
    print(f"wrote {len(frame_score_plots)} frame score plots under {frame_score_dir}")
    print(f"wrote {len(frame_budget_plots)} frame budget plots under {frame_budget_dir}")
    if step_plot is not None:
        print(f"wrote {step_plot}")


if __name__ == "__main__":
    main()
