#!/usr/bin/env python3
"""Regression checks for per-frame layer-budget visualization outputs."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


FIELDNAMES = [
    "step",
    "strategy",
    "total_budget",
    "assigned_budget",
    "total_capacity",
    "active_layers",
    "layer",
    "capacity",
    "score",
    "weight",
    "raw_budget",
    "final_budget",
    "events",
]


def _write_test_log(path: Path) -> None:
    rows = [
        [7, "value_weighted_leverage_pr", 60, 60, 90, 2, 2, 30, 3.0, 0.6, 36, 30, ""],
        [2, "value_weighted_leverage_pr", 60, 60, 90, 3, 1, 30, 2.0, 0.3, 18, 18, ""],
        [7, "value_weighted_leverage_pr", 60, 60, 90, 2, 0, 30, 1.5, 0.4, 24, 30, ""],
        [2, "value_weighted_leverage_pr", 60, 60, 90, 3, 2, 30, 3.0, 0.5, 30, 30, ""],
        [2, "value_weighted_leverage_pr", 60, 60, 90, 3, 0, 30, 1.0, 0.2, 12, 12, ""],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)


def test_cli_outputs() -> None:
    script_path = Path(__file__).with_name("summarize_layer_budget_scores.py")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_csv = temp_path / "layer_budget_scores.csv"
        output_dir = temp_path / "summary"
        _write_test_log(input_csv)

        subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(input_csv),
                "--output_dir",
                str(output_dir),
                "--latest-step",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        score_dir = output_dir / "layer_budget_frame_scores"
        budget_dir = output_dir / "layer_budget_frame_budgets"
        assert sorted(path.name for path in score_dir.glob("*.png")) == [
            "frame_000002_score_by_layer.png",
            "frame_000007_score_by_layer.png",
        ]
        assert sorted(path.name for path in budget_dir.glob("*.png")) == [
            "frame_000002_budget_by_layer.png",
            "frame_000007_budget_by_layer.png",
        ]

        existing_outputs = [
            "layer_budget_scores_by_frame.csv",
            "layer_budget_layer_mean_scores.csv",
            "layer_budget_layer_mean_score_by_layer.png",
            "layer_budget_frame_mean_score_hist.png",
            "layer_budget_budgets_by_frame.csv",
            "layer_budget_layer_mean_budgets.csv",
            "layer_budget_layer_mean_budget_by_layer.png",
            "layer_budget_step_000007_budget_by_layer.png",
        ]
        for filename in existing_outputs:
            assert (output_dir / filename).is_file(), filename

        with (output_dir / "layer_budget_scores_by_frame.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            rows = {int(row["step"]): row for row in csv.DictReader(f)}
        assert list(rows) == [2, 7]
        assert rows[7]["score_layer_01"] == "0"

        with (output_dir / "layer_budget_budgets_by_frame.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            rows = {int(row["step"]): row for row in csv.DictReader(f)}
        assert rows[7]["budget_layer_01"] == "0"


if __name__ == "__main__":
    test_cli_outputs()
    print("layer-budget visualization checks passed")
