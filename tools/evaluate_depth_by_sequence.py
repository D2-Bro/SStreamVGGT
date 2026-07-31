#!/usr/bin/env python3
"""Evaluate saved video-depth predictions separately for each KITTI sequence.

The metric implementation is imported from ``src/eval/video_depth/tools.py`` so
the numbers use the same resizing, alignment, masking, and metric definitions as
the existing aggregate evaluator.

Example:
    python tools/evaluate_depth_by_sequence.py \
        eval_results/video_depth/Final_result_kitti \
        --align scale
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.video_depth.tools import depth_evaluation  # noqa: E402


DEFAULT_KITTI_GT_DIR = Path(
    "/home/dongjae/data/kitti_depth/depth_selection/"
    "val_selection_cropped/groundtruth_depth_gathered"
)
METRIC_NAMES = (
    "Abs Rel",
    "Sq Rel",
    "RMSE",
    "Log RMSE",
    "δ < 1.",
    "δ < 1.25",
    "δ < 1.25^2",
    "δ < 1.25^3",
)
SHEETS_COLUMNS = (
    "row_type",
    "sequence",
    "frames",
    "valid_pixels",
    "abs_rel",
    "sq_rel",
    "rmse",
    "log_rmse",
    "delta_lt_1",
    "delta_lt_1_25",
    "delta_lt_1_25_sq",
    "delta_lt_1_25_cu",
)
SHEETS_METRIC_COLUMNS = {
    "Abs Rel": "abs_rel",
    "Sq Rel": "sq_rel",
    "RMSE": "rmse",
    "Log RMSE": "log_rmse",
    "δ < 1.": "delta_lt_1",
    "δ < 1.25": "delta_lt_1_25",
    "δ < 1.25^2": "delta_lt_1_25_sq",
    "δ < 1.25^3": "delta_lt_1_25_cu",
}


def _natural_key(path: Path) -> list[object]:
    import re

    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def _load_kitti_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth_png = np.asarray(image, dtype=np.int64)
    if depth_png.size == 0 or int(depth_png.max()) <= 255:
        raise ValueError(f"{path} is not a valid 16-bit KITTI depth image")
    depth = depth_png.astype(np.float64) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def _load_predictions(
    paths: Sequence[Path], target_height: int, target_width: int
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for path in paths:
        prediction = np.asarray(np.load(path, allow_pickle=False))
        prediction = np.squeeze(prediction)
        if prediction.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2D prediction after squeeze, got "
                f"shape {prediction.shape}"
            )
        resized = cv2.resize(
            prediction,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frames.append(resized)
    return np.stack(frames, axis=0)


def _json_number(value: Any) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric is not finite: {number}")
    if number.is_integer() and abs(number) < 2**53:
        return int(number)
    return number


def evaluate_sequence(
    prediction_paths: Sequence[Path],
    gt_paths: Sequence[Path],
    align: str,
    use_gpu: bool,
) -> dict[str, int | float]:
    if len(prediction_paths) != len(gt_paths):
        raise ValueError(
            f"frame count mismatch: predictions={len(prediction_paths)}, "
            f"ground_truth={len(gt_paths)}"
        )
    if not prediction_paths:
        raise ValueError("no prediction frames")

    gt_depth = np.stack([_load_kitti_depth(path) for path in gt_paths], axis=0)
    prediction_depth = _load_predictions(
        prediction_paths,
        target_height=gt_depth.shape[1],
        target_width=gt_depth.shape[2],
    )

    alignment_args = {
        "align_with_lad2": align == "scale&shift",
        "align_with_scale": align == "scale",
        "metric_scale": align == "metric",
    }
    metrics, _, _, _ = depth_evaluation(
        prediction_depth,
        gt_depth,
        max_depth=None,
        use_gpu=use_gpu,
        **alignment_args,
    )
    return {key: _json_number(value) for key, value in metrics.items()}


def _weighted_average(
    sequence_metrics: dict[str, dict[str, int | float]]
) -> dict[str, float]:
    weights = np.asarray(
        [metrics["valid_pixels"] for metrics in sequence_metrics.values()],
        dtype=np.float64,
    )
    if weights.size == 0 or float(weights.sum()) <= 0:
        raise ValueError("no valid pixels found across evaluated sequences")
    return {
        metric: float(
            np.average(
                [metrics[metric] for metrics in sequence_metrics.values()],
                weights=weights,
            )
        )
        for metric in METRIC_NAMES
    }


def _write_outputs(
    output_dir: Path,
    align: str,
    sequence_metrics: dict[str, dict[str, int | float]],
    errors: dict[str, str],
) -> tuple[Path, Path, Path]:
    summary_path = output_dir / f"result_{align}_by_sequence.json"
    csv_path = output_dir / f"result_{align}_by_sequence.csv"
    tsv_path = output_dir / f"result_{align}_by_sequence.tsv"
    weighted_average = _weighted_average(sequence_metrics)
    summary = {
        "align": align,
        "sequences": sequence_metrics,
        "weighted_average": weighted_average,
        "errors": errors,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    tabular_rows: list[dict[str, Any]] = []
    for sequence, metrics in sequence_metrics.items():
        tabular_rows.append(
            {
                "row_type": "sequence",
                "sequence": sequence,
                "frames": metrics["frames"],
                "valid_pixels": metrics["valid_pixels"],
                **{
                    column: metrics[metric]
                    for metric, column in SHEETS_METRIC_COLUMNS.items()
                },
            }
        )
    tabular_rows.append(
        {
            "row_type": "weighted_average",
            "sequence": "ALL",
            "frames": sum(
                int(metrics["frames"]) for metrics in sequence_metrics.values()
            ),
            "valid_pixels": sum(
                int(metrics["valid_pixels"])
                for metrics in sequence_metrics.values()
            ),
            **{
                column: weighted_average[metric]
                for metric, column in SHEETS_METRIC_COLUMNS.items()
            },
        }
    )

    # UTF-8 BOM helps spreadsheet applications detect CSV encoding reliably.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEETS_COLUMNS)
        writer.writeheader()
        writer.writerows(tabular_rows)

    # The TSV can be copied in full and pasted directly into Google Sheets A1.
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SHEETS_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(tabular_rows)
    return summary_path, csv_path, tsv_path


def _print_metrics(sequence: str, metrics: dict[str, int | float]) -> None:
    print(
        f"{sequence}: "
        f"Abs Rel={metrics['Abs Rel']:.6f}, "
        f"Sq Rel={metrics['Sq Rel']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}, "
        f"Log RMSE={metrics['Log RMSE']:.6f}, "
        f"δ1={metrics['δ < 1.25']:.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frame_*.npy depth predictions separately for every "
            "KITTI sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory containing one prediction subdirectory per sequence",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_KITTI_GT_DIR,
        help="KITTI groundtruth_depth_gathered directory",
    )
    parser.add_argument(
        "--align",
        choices=("scale&shift", "scale", "metric"),
        default="scale",
        help="prediction alignment method",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        help="evaluate only this exact sequence name; repeat for multiple sequences",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="metric computation device",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record a failed sequence and continue evaluating the rest",
    )
    parser.add_argument(
        "--no-per-sequence-files",
        action="store_true",
        help="do not write result_<align>.json inside each sequence directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    if not output_dir.is_dir():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        return 2
    if not gt_dir.is_dir():
        print(f"Error: KITTI GT directory not found: {gt_dir}", file=sys.stderr)
        return 2

    import torch

    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        print("Error: --device cuda requested but CUDA is unavailable.", file=sys.stderr)
        return 2
    use_gpu = args.device == "cuda" or (args.device == "auto" and cuda_available)
    print(f"Metric device: {'cuda' if use_gpu else 'cpu'}")

    requested_sequences = set(args.sequence or [])
    prediction_dirs = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_dir() and any(path.glob("frame_*.npy"))
        ),
        key=_natural_key,
    )
    if requested_sequences:
        discovered_names = {path.name for path in prediction_dirs}
        missing = sorted(requested_sequences - discovered_names)
        if missing:
            print(
                "Error: requested sequence(s) not found: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        prediction_dirs = [
            path for path in prediction_dirs if path.name in requested_sequences
        ]
    if not prediction_dirs:
        print(
            f"Error: no sequence/frame_*.npy predictions found in {output_dir}",
            file=sys.stderr,
        )
        return 2

    sequence_metrics: dict[str, dict[str, int | float]] = {}
    errors: dict[str, str] = {}
    for index, prediction_dir in enumerate(prediction_dirs, start=1):
        sequence = prediction_dir.name
        print(f"[{index}/{len(prediction_dirs)}] Evaluating {sequence} ...")
        prediction_paths = sorted(prediction_dir.glob("frame_*.npy"), key=_natural_key)
        sequence_gt_dir = gt_dir / sequence
        gt_paths = (
            sorted(sequence_gt_dir.glob("*.png"), key=_natural_key)
            if sequence_gt_dir.is_dir()
            else []
        )
        try:
            metrics = evaluate_sequence(
                prediction_paths=prediction_paths,
                gt_paths=gt_paths,
                align=args.align,
                use_gpu=use_gpu,
            )
            metrics["frames"] = len(prediction_paths)
            sequence_metrics[sequence] = metrics
            _print_metrics(sequence, metrics)
            if not args.no_per_sequence_files:
                result_path = prediction_dir / f"result_{args.align}.json"
                with result_path.open("w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
        except Exception as exc:
            message = str(exc)
            errors[sequence] = message
            print(f"Error: {sequence}: {message}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    if not sequence_metrics:
        print("Error: every sequence evaluation failed.", file=sys.stderr)
        return 1
    summary_path, csv_path, tsv_path = _write_outputs(
        output_dir=output_dir,
        align=args.align,
        sequence_metrics=sequence_metrics,
        errors=errors,
    )
    print(f"Saved per-sequence JSON: {summary_path}")
    print(f"Saved Google Sheets CSV: {csv_path}")
    print(f"Saved copy/paste TSV:    {tsv_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
