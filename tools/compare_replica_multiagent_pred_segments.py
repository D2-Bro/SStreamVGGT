#!/usr/bin/env python3
"""Compare a ReplicaMultiagent pred_traj.txt against GT by trajectory segment."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.pose_evaluation.evo_utils import eval_metrics, plot_trajectory, replica_pose_rows_to_traj_tum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice a ReplicaMultiagent pose_evaluation pred_traj.txt and compare each "
            "slice against the matching GT frames."
        )
    )
    parser.add_argument("pred_traj", type=Path, help="Path to pose_evaluation pred_traj.txt.")
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene name, e.g. Apart-2. Defaults to pred_traj parent directory name.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get(
                "SSTREAMVGGT_REPLICA_MULTIAGENT_ROOT",
                "/home/dongjae/data/ReplicaMultiagent",
            )
        ),
        help="ReplicaMultiagent root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <pred_traj parent>/segment_eval.",
    )
    parser.add_argument(
        "--kf-every",
        type=int,
        default=5,
        help="kf_every used by pose_evaluation. Current termProject script uses 5.",
    )
    parser.add_argument(
        "--pose-eval-stride",
        type=int,
        default=1,
        help="pose_eval_stride used by pose_evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="max_frames used by pose_evaluation. Omit for full sequence.",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["parts", "chunks", "ranges"],
        default="parts",
        help="How to split pred_traj before comparing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of predicted poses per chunk when --segment-mode chunks.",
    )
    parser.add_argument(
        "--ranges",
        default=None,
        help="Comma-separated pred index ranges for --segment-mode ranges, e.g. 0:100,100:250.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save GT/pred overlay plots for each segment.",
    )
    parser.add_argument(
        "--min-poses",
        type=int,
        default=3,
        help="Skip segments shorter than this many poses.",
    )
    return parser.parse_args()


def natural_frame_key(path: str) -> tuple[int, str]:
    name = os.path.basename(path)
    match = re.search(r"frame(\d+)\.jpg$", name)
    return (int(match.group(1)), name) if match else (10**18, name)


def discover_parts(scene_dir: Path) -> list[Path]:
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Missing ReplicaMultiagent scene directory: {scene_dir}")
    parts = sorted(
        part
        for part in scene_dir.iterdir()
        if part.is_dir() and (part / "results").is_dir() and (part / "traj.txt").exists()
    )
    if len(parts) < 2:
        raise FileNotFoundError(f"Scene needs at least two part folders with results/ and traj.txt: {scene_dir}")
    return parts


def frame_index(path: str) -> int:
    match = re.search(r"frame(\d+)\.jpg$", os.path.basename(path))
    if not match:
        raise ValueError(f"Unsupported frame filename: {path}")
    return int(match.group(1))


def selected_frames(scene_dir: Path, kf_every: int, pose_eval_stride: int, max_frames: int | None) -> list[tuple[Path, str]]:
    if kf_every < 1 or pose_eval_stride < 1:
        raise ValueError("--kf-every and --pose-eval-stride must be >= 1")
    effective_stride = kf_every * pose_eval_stride

    frames: list[tuple[Path, str]] = []
    for part in discover_parts(scene_dir):
        part_frames = sorted(
            glob.glob(str(part / "results" / "frame*.jpg")),
            key=natural_frame_key,
        )
        if not part_frames:
            raise FileNotFoundError(f"No RGB frames found under {part / 'results'}")
        frames.extend((Path(path), part.name) for path in part_frames)

    frames = frames[::effective_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No selected frames for {scene_dir}")
    return frames


def load_gt_for_frames(frames: list[tuple[Path, str]]) -> tuple[np.ndarray, np.ndarray]:
    traj_cache: dict[Path, np.ndarray] = {}
    rows = []
    for image_path, _part_name in frames:
        part_dir = image_path.parent.parent
        traj_path = part_dir / "traj.txt"
        if traj_path not in traj_cache:
            traj_rows = np.loadtxt(traj_path, dtype=np.float64)
            if traj_rows.ndim == 1:
                traj_rows = traj_rows[None, :]
            if traj_rows.ndim != 2 or traj_rows.shape[1] not in (12, 16):
                raise ValueError(f"Unsupported trajectory shape {traj_rows.shape}: {traj_path}")
            traj_cache[traj_path] = traj_rows

        idx = frame_index(str(image_path))
        traj_rows = traj_cache[traj_path]
        if idx >= traj_rows.shape[0]:
            raise IndexError(f"Frame index {idx} is outside trajectory length {traj_rows.shape[0]}: {image_path}")
        rows.append(traj_rows[idx])

    return replica_pose_rows_to_traj_tum(np.stack(rows, axis=0))


def load_pred_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"Expected TUM trajectory with 8 columns at {path}, got {data.shape}")
    return data[:, 1:8], data[:, 0]


def part_segments(frames: list[tuple[Path, str]]) -> list[tuple[str, int, int]]:
    segments = []
    start = 0
    current = frames[0][1]
    for i, (_path, part_name) in enumerate(frames[1:], start=1):
        if part_name != current:
            segments.append((current, start, i))
            start = i
            current = part_name
    segments.append((current, start, len(frames)))
    return segments


def chunk_segments(length: int, chunk_size: int) -> list[tuple[str, int, int]]:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    return [
        (f"chunk_{start:05d}_{min(start + chunk_size, length):05d}", start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def range_segments(ranges: str | None, length: int) -> list[tuple[str, int, int]]:
    if not ranges:
        raise ValueError("--ranges is required when --segment-mode ranges")

    segments = []
    for item in ranges.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid range {item!r}; expected start:end")
        start_s, end_s = item.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if start < 0 or end > length or start >= end:
            raise ValueError(f"Invalid range {item!r} for trajectory length {length}")
        segments.append((f"range_{start:05d}_{end:05d}", start, end))
    if not segments:
        raise ValueError("No valid ranges were provided")
    return segments


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment",
                "start",
                "end",
                "num_poses",
                "ate_rmse_m",
                "rpe_trans_rmse_m",
                "rpe_rot_rmse_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scene = args.scene or args.pred_traj.parent.name
    output_dir = args.output_dir or (args.pred_traj.parent / "segment_eval")
    scene_dir = args.root / scene

    frames = selected_frames(scene_dir, args.kf_every, args.pose_eval_stride, args.max_frames)
    pred_traj = load_pred_tum(args.pred_traj)
    gt_traj = load_gt_for_frames(frames)

    pred_len = pred_traj[0].shape[0]
    gt_len = gt_traj[0].shape[0]
    if pred_len != gt_len:
        raise ValueError(
            f"Length mismatch: pred={pred_len}, gt={gt_len}. "
            "Check --kf-every, --pose-eval-stride, and --max-frames."
        )

    if args.segment_mode == "parts":
        segments = part_segments(frames)
    elif args.segment_mode == "chunks":
        segments = chunk_segments(pred_len, args.chunk_size)
    else:
        segments = range_segments(args.ranges, pred_len)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, start, end in segments:
        num_poses = end - start
        if num_poses < args.min_poses:
            print(f"skip {name}: only {num_poses} poses")
            continue

        pred_slice = (pred_traj[0][start:end], pred_traj[1][start:end])
        gt_slice = (gt_traj[0][start:end], gt_traj[1][start:end])
        metric_path = output_dir / f"{name}_eval_metric.txt"
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_slice,
            gt_slice,
            seq=f"{scene}:{name}",
            filename=str(metric_path),
        )
        if args.plot:
            plot_trajectory(
                pred_slice,
                gt_slice,
                title=f"{scene}:{name}",
                filename=str(output_dir / f"{name}.png"),
            )

        rows.append(
            {
                "segment": name,
                "start": start,
                "end": end,
                "num_poses": num_poses,
                "ate_rmse_m": ate,
                "rpe_trans_rmse_m": rpe_trans,
                "rpe_rot_rmse_deg": rpe_rot,
            }
        )

    summary_path = output_dir / "segment_metrics.csv"
    write_csv(summary_path, rows)
    print(summary_path)


if __name__ == "__main__":
    main()
