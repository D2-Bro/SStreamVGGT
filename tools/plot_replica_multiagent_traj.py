#!/usr/bin/env python3
"""Plot ReplicaMultiagent GT trajectories by part or as a virtual scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXIS_TO_COL = {"x": 0, "y": 1, "z": 2}
AXIS_LABELS = {"x": "x (m)", "y": "y (m)", "z": "z (m)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ReplicaMultiagent part-level traj.txt files without running evaluation."
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
        "--seq-list",
        nargs="*",
        default=None,
        help="Scene names to plot, e.g. Apart-1. Defaults to all detected scenes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/traj_plots.",
    )
    parser.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xz",
        help="2D projection to plot.",
    )
    parser.add_argument(
        "--mode",
        choices=["parts", "combined", "both"],
        default="both",
        help="Plot part trajectories, concatenated scene trajectories, or both.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def discover_scenes(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"ReplicaMultiagent root does not exist: {root}")
    return sorted(
        scene.name
        for scene in root.iterdir()
        if scene.is_dir() and len(discover_parts(scene)) >= 2
    )


def discover_parts(scene_dir: Path) -> list[Path]:
    if not scene_dir.is_dir():
        return []
    return sorted(
        part
        for part in scene_dir.iterdir()
        if part.is_dir() and (part / "traj.txt").exists()
    )


def load_replica_positions(traj_path: Path) -> np.ndarray:
    rows = np.loadtxt(traj_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] not in (12, 16):
        raise ValueError(f"Unsupported Replica trajectory shape {rows.shape}: {traj_path}")

    positions = np.stack([rows[:, 3], rows[:, 7], rows[:, 11]], axis=1)
    if not np.isfinite(positions).all():
        raise ValueError(f"Trajectory contains non-finite positions: {traj_path}")
    return positions


def equal_limits(positions: np.ndarray, x_idx: int, y_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = positions[:, [x_idx, y_idx]]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    half = 0.5 * span * 1.08
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def plot_lines(
    trajectories: list[tuple[str, np.ndarray]],
    output_path: Path,
    title: str,
    axes: str,
    dpi: int,
) -> None:
    x_axis, y_axis = axes
    x_idx = AXIS_TO_COL[x_axis]
    y_idx = AXIS_TO_COL[y_axis]
    all_positions = np.concatenate([positions for _, positions in trajectories], axis=0)
    xlim, ylim = equal_limits(all_positions, x_idx, y_idx)

    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=dpi)
    for name, positions in trajectories:
        xy = positions[:, [x_idx, y_idx]]
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.6, label=f"{name} ({len(positions)})")
        ax.scatter(xy[0, 0], xy[0, 1], s=22, marker="o")
        ax.scatter(xy[-1, 0], xy[-1, 1], s=28, marker="x")

    ax.set_title(title)
    ax.set_xlabel(AXIS_LABELS[x_axis])
    ax.set_ylabel(AXIS_LABELS[y_axis])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(output_path)


def plot_scene(scene_dir: Path, output_dir: Path, axes: str, mode: str, dpi: int) -> None:
    parts = discover_parts(scene_dir)
    if len(parts) < 2:
        raise FileNotFoundError(f"Scene needs at least two part folders with traj.txt: {scene_dir}")

    part_trajectories = [(part.name, load_replica_positions(part / "traj.txt")) for part in parts]
    if mode in ("parts", "both"):
        plot_lines(
            part_trajectories,
            output_dir / f"{scene_dir.name}_parts_traj.png",
            f"{scene_dir.name} GT by part",
            axes,
            dpi,
        )

    if mode in ("combined", "both"):
        combined = np.concatenate([positions for _, positions in part_trajectories], axis=0)
        plot_lines(
            [(scene_dir.name, combined)],
            output_dir / f"{scene_dir.name}_combined_traj.png",
            f"{scene_dir.name} GT combined",
            axes,
            dpi,
        )


def main() -> None:
    args = parse_args()
    scenes = args.seq_list if args.seq_list else discover_scenes(args.root)
    if not scenes:
        raise FileNotFoundError(f"No ReplicaMultiagent scenes found under {args.root}")

    output_dir = args.output_dir or (args.root / "traj_plots")
    for scene in scenes:
        plot_scene(args.root / scene, output_dir, args.axes, args.mode, args.dpi)


if __name__ == "__main__":
    main()
