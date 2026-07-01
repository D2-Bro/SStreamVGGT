#!/usr/bin/env python3
"""Plot pose_evaluation TUM trajectories with a fixed figure layout."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface, plot


AXIS_TO_COL = {"x": 1, "y": 2, "z": 3}
AXIS_LABELS = {"x": "x (m)", "y": "y (m)", "z": "z (m)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pred_traj.txt files produced by src/eval/pose_evaluation. "
            "The input trajectory must be TUM format: timestamp x y z qw qx qy qz."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Run directory, sequence directory, or a single pred_traj.txt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save plots. Defaults to <result_dir>/trajectory_plots.",
    )
    parser.add_argument(
        "--pattern",
        default="*/pred_traj.txt",
        help="Glob used when result_dir is a run directory. Default: */pred_traj.txt",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help=(
            "Plot only one sequence when result_dir is a run directory "
            "(for example: --sequence office4)."
        ),
    )
    parser.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xz",
        help="2D projection to plot. Default: xz, which is often most readable for Replica.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=6.0,
        help="Figure width in inches. Default: 6.0",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=6.0,
        help="Figure height in inches. Default: 6.0",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI. Default: 200, so the default output is 1200x1200 px.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format. Default: png",
    )
    parser.add_argument(
        "--global-limits",
        action="store_true",
        help="Use one shared equal-axis range for every trajectory in the run.",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Manual horizontal axis limits in plotted coordinates.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Manual vertical axis limits in plotted coordinates.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional padding around automatic limits. Default: 0.05",
    )
    parser.add_argument(
        "--start-at-origin",
        action="store_true",
        help="Subtract the first translation from each trajectory before plotting.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw sequence titles.",
    )
    parser.add_argument(
        "--gt-traj",
        type=Path,
        default=None,
        help="Optional ground-truth trajectory used to compute evo APE errors.",
    )
    parser.add_argument(
        "--gt-format",
        choices=["tum", "nrgbd"],
        default="tum",
        help="Ground-truth trajectory format. Default: tum. Use nrgbd for Neural RGB-D poses.txt.",
    )
    parser.add_argument(
        "--error-file",
        type=Path,
        default=None,
        help="Optional text file with one per-frame error value. Overrides --gt-traj errors.",
    )
    parser.add_argument(
        "--error-min",
        type=float,
        default=None,
        help="Minimum value for trajectory error color normalization.",
    )
    parser.add_argument(
        "--error-max",
        type=float,
        default=None,
        help="Maximum value for trajectory error color normalization.",
    )
    parser.add_argument(
        "--error-cmap",
        default="turbo",
        help="Matplotlib colormap for error-colored trajectory. Default: turbo",
    )
    parser.add_argument(
        "--error-label",
        default="translation error (m)",
        help="Colorbar label when error coloring is enabled.",
    )
    parser.add_argument(
        "--no-error-align",
        action="store_true",
        help="Do not Sim(3)-align predicted positions to GT before computing errors.",
    )
    return parser.parse_args()


def find_trajectory_files(result_dir: Path, pattern: str, sequence: str | None) -> list[Path]:
    if result_dir.is_file():
        if sequence is not None:
            raise ValueError("--sequence cannot be used when result_dir is a trajectory file.")
        return [result_dir]

    direct = result_dir / "pred_traj.txt"
    if direct.exists():
        if sequence is not None and result_dir.name != sequence:
            raise FileNotFoundError(
                f"{result_dir} is already a sequence directory, but its name is "
                f"{result_dir.name!r}, not {sequence!r}."
            )
        return [direct]

    if sequence is not None:
        sequence_file = result_dir / sequence / "pred_traj.txt"
        if not sequence_file.exists():
            raise FileNotFoundError(f"No trajectory file found at {sequence_file}.")
        return [sequence_file]

    files = sorted(result_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No trajectory files found under {result_dir} with pattern {pattern!r}."
        )
    return files


def load_tum_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least 4.")

    timestamps = data[:, 0].astype(np.float64)
    positions = data[:, 1:4].astype(np.float64)
    finite = np.isfinite(timestamps) & np.isfinite(positions).all(axis=1)
    timestamps = timestamps[finite]
    positions = positions[finite]
    if positions.shape[0] == 0:
        raise ValueError(f"{path} does not contain any finite positions.")
    return timestamps, positions


def load_tum_positions(path: Path, start_at_origin: bool) -> np.ndarray:
    _, positions = load_tum_data(path)
    if start_at_origin:
        positions = positions - positions[0]
    return positions


def load_error_values(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#", ndmin=1)
    if values.ndim > 1:
        values = values[:, -1]
    values = values.astype(np.float64)
    values = values[np.isfinite(values)]
    if values.shape[0] == 0:
        raise ValueError(f"{path} does not contain any finite error values.")
    return values


def umeyama_align_positions(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> np.ndarray:
    if source.shape != target.shape:
        raise ValueError(f"Cannot align arrays with shapes {source.shape} and {target.shape}.")
    if source.shape[0] < 3:
        return source.copy()

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    cov = (tgt_centered.T @ src_centered) / source.shape[0]
    u, singular_values, vh = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vh
    scale = 1.0
    if with_scale:
        src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
        if src_var > 0:
            scale = float(np.sum(singular_values * sign) / src_var)
    translation = tgt_mean - scale * (rotation @ src_mean)
    return (scale * (rotation @ source.T)).T + translation


def match_gt_positions(pred_path: Path, gt_path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    pred_timestamps, pred_positions = load_tum_data(pred_path)
    gt_timestamps, gt_positions = load_tum_data(gt_path)
    n = min(count, pred_positions.shape[0])
    pred_timestamps = pred_timestamps[:n]
    pred_positions = pred_positions[:n]

    if gt_positions.shape[0] == n:
        return pred_positions, gt_positions

    matched_gt = []
    for timestamp in pred_timestamps:
        idx = int(np.argmin(np.abs(gt_timestamps - timestamp)))
        matched_gt.append(gt_positions[idx])
    return pred_positions, np.asarray(matched_gt, dtype=np.float64)


def compute_translation_errors(
    pred_path: Path,
    pred_positions: np.ndarray,
    gt_path: Path | None,
    error_file: Path | None,
    align: bool,
) -> np.ndarray | None:
    if error_file is not None:
        errors = load_error_values(error_file)
        if errors.shape[0] < pred_positions.shape[0]:
            raise ValueError(
                f"{error_file} has {errors.shape[0]} error values, but "
                f"{pred_path} has {pred_positions.shape[0]} poses."
            )
        return errors[: pred_positions.shape[0]]
    if gt_path is None:
        return None

    raw_pred, gt_positions = match_gt_positions(pred_path, gt_path, pred_positions.shape[0])
    error_pred = umeyama_align_positions(raw_pred, gt_positions) if align else raw_pred
    errors = np.linalg.norm(error_pred - gt_positions, axis=1)
    return errors[: pred_positions.shape[0]]


def sequence_name(path: Path, root: Path) -> str:
    if path.name == "pred_traj.txt":
        return path.parent.name
    try:
        return path.relative_to(root).with_suffix("").as_posix().replace("/", "_")
    except ValueError:
        return path.stem


def equal_limits(points: np.ndarray, x_idx: int, y_idx: int, padding: float) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = points[:, [x_idx, y_idx]]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span == 0.0:
        span = 1.0
    half = span * (1.0 + padding) / 2.0
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def enforce_png_size(output_path: Path, width: float, height: float, dpi: int) -> None:
    if output_path.suffix.lower() != ".png":
        return
    target_size = (int(round(width * dpi)), int(round(height * dpi)))
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(output_path) as image:
        if image.size == target_size:
            return
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize(target_size, resampling)
        resized.save(output_path)


def plot_mode_from_axes(axes: str):
    return getattr(plot.PlotMode, axes)


def load_evo_tum_trajectory(path: Path):
    return file_interface.read_tum_trajectory_file(str(path))


def load_evo_nrgbd_trajectory(path: Path):
    pose_rows = np.loadtxt(path, dtype=np.float64)
    if pose_rows.ndim != 2 or pose_rows.shape[1] != 4 or pose_rows.shape[0] % 4 != 0:
        raise ValueError(f"Invalid NRGBD trajectory shape in {path}: {pose_rows.shape}")
    if np.isnan(pose_rows).any():
        raise ValueError(f"NRGBD trajectory contains nan poses: {path}")

    poses = pose_rows.reshape(-1, 4, 4).copy()
    poses[:, :, 1:3] *= -1.0
    pose_path = PosePath3D(poses_se3=list(poses))
    timestamps = np.arange(len(poses)).astype(float)
    return PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps)


def load_evo_gt_trajectory(path: Path, gt_format: str):
    if gt_format == "tum":
        return load_evo_tum_trajectory(path)
    if gt_format == "nrgbd":
        return load_evo_nrgbd_trajectory(path)
    raise ValueError(f"Unsupported GT format: {gt_format}")


def associated_evo_trajectories(pred_path: Path, gt_path: Path, gt_format: str):
    pred_traj = load_evo_tum_trajectory(pred_path)
    gt_traj = load_evo_gt_trajectory(gt_path, gt_format)
    if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
        pred_traj.timestamps = gt_traj.timestamps
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)
    return gt_traj, pred_traj


def draw_evo_ape_trajectory(
    pred_path: Path,
    gt_path: Path,
    gt_format: str,
    output_path: Path,
    name: str,
    axes: str,
    figsize: tuple[float, float],
    dpi: int,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    title: bool,
    align: bool,
    correct_scale: bool,
    error_min: float | None,
    error_max: float | None,
    error_cmap: str,
) -> None:
    gt_traj, pred_traj = associated_evo_trajectories(pred_path, gt_path, gt_format)
    ape_result = main_ape.ape(
        deepcopy(gt_traj),
        pred_traj,
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=correct_scale,
        est_name="traj",
    )
    errors = ape_result.np_arrays["error_array"]
    if errors.shape[0] != pred_traj.num_poses:
        raise ValueError(
            f"APE error length {errors.shape[0]} does not match trajectory length {pred_traj.num_poses}."
        )

    min_map = float(np.nanmin(errors)) if error_min is None else error_min
    max_map = float(np.nanmax(errors)) if error_max is None else error_max
    if max_map <= min_map:
        max_map = min_map + 1e-12

    plot.SETTINGS.plot_trajectory_cmap = error_cmap
    plot_mode = plot_mode_from_axes(axes)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plot.prepare_axis(fig, plot_mode)
    if title:
        ax.set_title(name, fontsize=12)
    plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj_colormap(
        ax,
        pred_traj,
        errors,
        plot_mode,
        min_map,
        max_map,
        title="",
        fig=fig,
        plot_start_end_markers=True,
    )
    ax.set_xlabel(AXIS_LABELS[axes[0]])
    ax.set_ylabel(AXIS_LABELS[axes[1]])
    if xlim is None or ylim is None:
        combined = np.concatenate([gt_traj.positions_xyz, pred_traj.positions_xyz], axis=0)
        x_idx = AXIS_TO_COL[axes[0]] - 1
        y_idx = AXIS_TO_COL[axes[1]] - 1
        auto_xlim, auto_ylim = equal_limits(combined, x_idx, y_idx, padding=0.05)
        xlim = xlim or auto_xlim
        ylim = ylim or auto_ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc="best", frameon=True, fontsize=9)
    fig.set_size_inches(figsize[0], figsize[1], forward=True)
    fig.set_dpi(dpi)
    fig.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    enforce_png_size(output_path, figsize[0], figsize[1], dpi)


def draw_trajectory(
    positions: np.ndarray,
    output_path: Path,
    name: str,
    axes: str,
    figsize: tuple[float, float],
    dpi: int,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    title: bool,
    errors: np.ndarray | None,
    error_min: float | None,
    error_max: float | None,
    error_cmap: str,
    error_label: str,
) -> None:
    x_axis, y_axis = axes
    x_idx = AXIS_TO_COL[x_axis] - 1
    y_idx = AXIS_TO_COL[y_axis] - 1
    xy = positions[:, [x_idx, y_idx]]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if errors is None:
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color="#1f77b4",
            linewidth=1.8,
            label="Predicted",
        )
    else:
        errors = np.asarray(errors, dtype=np.float64)[: xy.shape[0]]
        if errors.shape[0] != xy.shape[0]:
            raise ValueError(
                f"{name}: error length {errors.shape[0]} does not match trajectory length {xy.shape[0]}."
            )
        finite = np.isfinite(errors)
        if not finite.all():
            errors = errors.copy()
            errors[~finite] = np.nanmin(errors[finite]) if finite.any() else 0.0
        vmin = float(np.nanmin(errors)) if error_min is None else error_min
        vmax = float(np.nanmax(errors)) if error_max is None else error_max
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = plt.get_cmap(error_cmap)

        if xy.shape[0] > 1:
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            segment_errors = 0.5 * (errors[:-1] + errors[1:])
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=2.2,
                capstyle="round",
                label="Predicted",
            )
            collection.set_array(segment_errors)
            ax.add_collection(collection)
            mappable = collection
        else:
            mappable = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=errors,
                cmap=cmap,
                norm=norm,
                s=24,
                label="Predicted",
            )
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(error_label)

    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        s=28,
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Start",
    )
    ax.scatter(
        xy[-1, 0],
        xy[-1, 1],
        s=28,
        color="#d62728",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="End",
    )
    if title:
        ax.set_title(name, fontsize=12)
    ax.set_xlabel(AXIS_LABELS[x_axis])
    ax.set_ylabel(AXIS_LABELS[y_axis])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    enforce_png_size(output_path, figsize[0], figsize[1], dpi)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    traj_files = find_trajectory_files(result_dir, args.pattern, args.sequence)
    if (args.gt_traj is not None or args.error_file is not None) and len(traj_files) != 1:
        raise ValueError("--gt-traj and --error-file require plotting exactly one trajectory; use --sequence.")
    if args.error_min is not None and args.error_max is not None and args.error_max <= args.error_min:
        raise ValueError("--error-max must be greater than --error-min.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = result_dir.parent / "trajectory_plots" if result_dir.is_file() else result_dir / "trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []
    for path in traj_files:
        positions = load_tum_positions(path, start_at_origin=args.start_at_origin)
        errors = None
        if args.error_file is not None:
            errors = compute_translation_errors(
                pred_path=path,
                pred_positions=positions,
                gt_path=None,
                error_file=args.error_file,
                align=not args.no_error_align,
            )
        trajectories.append((path, positions, errors))

    x_axis, y_axis = args.axes
    x_idx = AXIS_TO_COL[x_axis] - 1
    y_idx = AXIS_TO_COL[y_axis] - 1
    shared_xlim = tuple(args.xlim) if args.xlim is not None else None
    shared_ylim = tuple(args.ylim) if args.ylim is not None else None
    if args.global_limits and (shared_xlim is None or shared_ylim is None):
        all_positions = np.concatenate([positions for _, positions, _ in trajectories], axis=0)
        auto_xlim, auto_ylim = equal_limits(all_positions, x_idx, y_idx, args.padding)
        shared_xlim = shared_xlim or auto_xlim
        shared_ylim = shared_ylim or auto_ylim

    for path, positions, errors in trajectories:
        name = sequence_name(path, result_dir)
        xlim = shared_xlim
        ylim = shared_ylim
        if xlim is None or ylim is None:
            auto_xlim, auto_ylim = equal_limits(positions, x_idx, y_idx, args.padding)
            xlim = xlim or auto_xlim
            ylim = ylim or auto_ylim

        output_path = output_dir / f"{name}_trajectory.{args.format}"
        if args.gt_traj is not None and args.error_file is None:
            draw_evo_ape_trajectory(
                pred_path=path,
                gt_path=args.gt_traj,
                gt_format=args.gt_format,
                output_path=output_path,
                name=name,
                axes=args.axes,
                figsize=(args.width, args.height),
                dpi=args.dpi,
                xlim=tuple(args.xlim) if args.xlim is not None else None,
                ylim=tuple(args.ylim) if args.ylim is not None else None,
                title=not args.no_title,
                align=not args.no_error_align,
                correct_scale=not args.no_error_align,
                error_min=args.error_min,
                error_max=args.error_max,
                error_cmap=args.error_cmap,
            )
        else:
            draw_trajectory(
                positions=positions,
                output_path=output_path,
                name=name,
                axes=args.axes,
                figsize=(args.width, args.height),
                dpi=args.dpi,
                xlim=xlim,
                ylim=ylim,
                title=not args.no_title,
                errors=errors,
                error_min=args.error_min,
                error_max=args.error_max,
                error_cmap=args.error_cmap,
                error_label=args.error_label,
            )
        print(output_path)


if __name__ == "__main__":
    main()
