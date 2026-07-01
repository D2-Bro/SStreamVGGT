#!/usr/bin/env python3
"""Run and visualize mv_recon leverage/eviction diagnostics.

This wrapper keeps the evaluation entrypoint unchanged. It enables the
existing leverage histogram, eviction-NN, and token-overlay dump hooks, then
converts the full token dumps into step-wise image overlays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any



REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results" / "mv_recon" / "SStreamVGGT_100_termProject_a0.7_SimTopK_ridge1e-5"
MV_RECON_RUN_DATASETS = ("7scenes", "NRGBD")
ACTIVE_DATASETS = ("7scenes", "NRGBD", "kitti_s1_500")
DATASET_ROOTS = {
    "7scenes": Path("/home/dongjae/data/7scenes_sfm"),
    "NRGBD": Path("/home/dongjae/data/neural_rgbd_data"),
    "Replica": Path(os.environ.get("SSTREAMVGGT_REPLICA_ROOT", "/home/dongjae/data/replica/Replica")),
    "kitti": Path(os.environ.get("SSTREAMVGGT_KITTI_DEPTH_ROOT", "/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/image_gathered")),
}
_KITTI_IMAGE_CACHE: dict[tuple[str, str], list[Path]] = {}


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def add_common_paths(parser: argparse.ArgumentParser, *, default_datasets: tuple[str, ...] = ACTIVE_DATASETS) -> None:
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", type=str, default=",".join(default_datasets))
    parser.add_argument("--kf_every", type=int, default=0, help="Frame stride for source image lookup; 0 uses dataset defaults")
    parser.add_argument("--image_width", type=int, default=518)
    parser.add_argument("--image_height", type=int, default=392)
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=14)


def build_run_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir
    hist_dir = output_dir / "leverage_score_histograms"
    nn_dir = output_dir / "eviction_nn_analysis"
    token_dump_dir = output_dir / "token_overlay_dumps"
    first_frame_special_args = ["--first_frame_special_tokens_only"] if args.first_frame_special_tokens_only else []

    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(args.num_processes),
        "--main_process_port",
        str(args.main_process_port),
        "./eval/mv_recon/launch.py",
        "--weights",
        args.weights,
        "--output_dir",
        str(output_dir.relative_to(SRC_DIR) if output_dir.is_relative_to(SRC_DIR) else output_dir),
        "--model_name",
        "StreamVGGT",
        "--max_frames",
        str(args.max_frames),
        "--eviction_policy",
        "svd_leverage",
        "--leverage_granularity",
        "layer",
        "--leverage_projection",
        "random",
        "--leverage_eviction_selector",
        "similarity_topk",
        "--leverage_similarity_granularity",
        "layer",
        "--leverage_similarity_feature_projection",
        "raw",
        "--leverage_eviction_risk_mode",
        "low_leverage",
        "--leverage_high_outlier_z",
        "3.0",
        "--leverage_dpp_candidate_multiplier",
        "3",
        "--leverage_dpp_greedy_block_size",
        "128",
        "--leverage_dpp_quality_beta",
        "0.0",
        "--leverage_dpp_diversity_beta",
        "1.0",
        "--leverage_dpp_feature_projection",
        "random",
        "--layer_budget_strategy",
        "value_weighted_leverage_pr",
        "--layer_budget_alpha",
        "0.7",
        "--layer_budget_min_tokens",
        "0",
        "--layer_budget_eps",
        "1e-12",
        "--leverage_approx_method",
        "right_sketch_ridge",
        "--leverage_ridge_lambda",
        "1e-5",
        "--leverage_ridge_lambda_mode",
        "relative",
        "--leverage_ridge_score_chunk_size",
        "4096",
        "--leverage_ridge_jitter",
        "1e-6",
        "--leverage_ridge_dim",
        "128",
        "--leverage_random_seed",
        "42",
        "--layer_budget_value_gamma",
        "0.7",
        "--layer_budget_value_norm_type",
        "mean",
        "--layer_budget_norm_source",
        "key",
        "--budget",
        "200000",
        "--history_anchor_strategy",
        "none",
        "--camera_motion_threshold",
        "0.2",
        "--max_anchors",
        "10",
        "--min_anchor_interval",
        "5",
        "--leverage_score_histogram_dir",
        str(hist_dir),
        "--leverage_score_histogram_bins",
        str(args.histogram_bins),
        "--leverage_score_histogram_min",
        str(args.histogram_min),
        "--leverage_score_histogram_max",
        str(args.histogram_max),
        "--leverage_score_histogram_layers",
        args.analysis_layers,
        "--leverage_score_histogram_steps",
        args.analysis_steps,
        "--eviction_nn_analysis_dir",
        str(nn_dir),
        "--eviction_nn_analysis_layers",
        args.analysis_layers,
        "--eviction_nn_analysis_heads",
        args.analysis_heads,
        "--eviction_nn_analysis_steps",
        args.analysis_steps,
        "--eviction_nn_analysis_space",
        args.analysis_space,
        "--eviction_nn_analysis_max_evicted",
        str(args.max_evicted_per_event),
        "--eviction_nn_analysis_save_topk_pairs",
        str(args.save_topk_pairs),
        "--token_overlay_dump_dir",
        str(token_dump_dir),
        "--token_overlay_dump_layers",
        args.analysis_layers,
        "--token_overlay_dump_heads",
        args.analysis_heads,
        "--token_overlay_dump_steps",
        args.analysis_steps,
    ]
    if args.token_overlay_dump_max_events is not None:
        command.extend(["--token_overlay_dump_max_events", str(args.token_overlay_dump_max_events)])
    command.extend(first_frame_special_args)
    command.extend(args.extra_launch_arg or [])
    return command


def run_command(args: argparse.Namespace) -> int:
    requested = tuple(parse_csv(args.datasets))
    unsupported = [name for name in requested if name not in MV_RECON_RUN_DATASETS]
    if unsupported:
        print(
            "Warning: mv_recon/launch.py does not expose a dataset selector. "
            f"Requested {unsupported}, but the unchanged launcher will run its active dataset scope: "
            f"7scenes, NRGBD."
        )

    command = build_run_command(args)
    print(shell_join(command))
    if args.dry_run or args.skip_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=str(SRC_DIR), check=False)
    return int(completed.returncode)


def safe_torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def event_metadata_from_path(path: Path, analysis_root: Path) -> dict[str, str]:
    rel = path.relative_to(analysis_root)
    parts = rel.parts
    dataset = parts[0] if len(parts) >= 1 else "unknown"
    rank = parts[1] if len(parts) >= 2 else "rank_unknown"
    scene_key = parts[2] if len(parts) >= 3 else "unknown_scene"
    safe_scene = scene_key.split("_", 1)[1] if "_" in scene_key else scene_key
    return {"dataset": dataset, "rank": rank, "scene_key": scene_key, "safe_scene": safe_scene}


def is_kitti_dataset(dataset: str) -> bool:
    return dataset == "kitti" or dataset.startswith("kitti_s1_")


def kitti_dataset_root(dataset: str) -> Path:
    if dataset == "kitti":
        return DATASET_ROOTS["kitti"]
    match = re.match(r"kitti_s1_(?P<count>\d+)$", dataset)
    if match:
        env_key = f"SSTREAMVGGT_KITTI_DEPTH_ROOT_{match.group('count')}"
        default = f"/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/image_gathered_{match.group('count')}"
        return Path(os.environ.get(env_key, default))
    return DATASET_ROOTS["kitti"]


def parse_scene(dataset: str, safe_scene: str) -> str:
    if dataset == "7scenes":
        match = re.match(r"(?P<scene>.+)_seq-(?P<seq>\d+)$", safe_scene)
        if match:
            return f"{match.group('scene')}/seq-{match.group('seq')}"
    return safe_scene


def resolve_kf_every(dataset: str, kf_every: int) -> int:
    if int(kf_every) > 0:
        return int(kf_every)
    return 1 if is_kitti_dataset(dataset) else 2


def kitti_frame_image_path(dataset: str, scene: str, frame_id: int, kf_every: int) -> Path | None:
    root = kitti_dataset_root(dataset)
    key = (dataset, scene)
    images = _KITTI_IMAGE_CACHE.get(key)
    if images is None:
        seq_dir = root / scene
        images = sorted(seq_dir.glob("*.png"))
        _KITTI_IMAGE_CACHE[key] = images
    frame_no = int(frame_id) * resolve_kf_every(dataset, kf_every)
    if frame_no < 0 or frame_no >= len(images):
        return None
    return images[frame_no]


def frame_image_path(dataset: str, scene: str, frame_id: int, kf_every: int) -> Path | None:
    frame_no = int(frame_id) * resolve_kf_every(dataset, kf_every)
    if dataset == "7scenes":
        return DATASET_ROOTS[dataset] / scene / f"frame-{frame_no:06d}.color.png"
    if dataset == "NRGBD":
        return DATASET_ROOTS[dataset] / scene / "images" / f"img{frame_no}.png"
    if dataset == "Replica":
        return DATASET_ROOTS[dataset] / scene / "results" / f"frame{frame_no}.jpg"
    if is_kitti_dataset(dataset):
        return kitti_frame_image_path(dataset, scene, frame_id, kf_every)
    return None


def load_image_for_overlay(path: Path, width: int, height: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.size != (int(width), int(height)):
        image = image.resize((int(width), int(height)))
    return image


def tensor_to_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def token_to_patch(token_idx: int, patch_start_idx: int, patch_count: int, patch_w: int) -> tuple[int, int, int] | None:
    if int(token_idx) < int(patch_start_idx):
        return None
    patch_id = int(token_idx) - int(patch_start_idx)
    if patch_id < 0 or patch_id >= int(patch_count):
        return None
    return patch_id, patch_id // int(patch_w), patch_id % int(patch_w)


def iter_token_overlay_events(args: argparse.Namespace):
    analysis_root = args.output_dir / "token_overlay_dumps"
    datasets = set(parse_csv(args.datasets))
    if not analysis_root.exists():
        print(f"Warning: token overlay dump root does not exist: {analysis_root}")
        return
    for event_path in sorted(analysis_root.rglob("events/step_*.pt")):
        meta = event_metadata_from_path(event_path, analysis_root)
        if datasets and meta["dataset"] not in datasets:
            continue
        try:
            payload = safe_torch_load(event_path)
        except Exception as exc:
            print(f"Warning: failed to read {event_path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        scene = parse_scene(meta["dataset"], meta["safe_scene"])
        event_meta = dict(payload.get("meta") or {})
        event_meta.update(meta)
        event_meta["scene"] = scene
        event_meta["event_file"] = str(event_path)
        yield event_meta, payload


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return None
    pos = (len(values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    weight = pos - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def save_evict_overlay(image_path: Path, heat, out_path: Path, title: str, alpha: float, patch_size: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    image = load_image_for_overlay(image_path, heat.shape[1] * int(patch_size), heat.shape[0] * int(patch_size))
    arr = np.asarray(image)
    masked = np.ma.masked_where(heat <= 0, heat)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(arr)
    overlay = ax.imshow(
        masked,
        cmap="hot",
        interpolation="nearest",
        alpha=float(alpha),
        extent=(0, arr.shape[1], arr.shape[0], 0),
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(overlay, ax=ax, fraction=0.035, pad=0.02, label="current-step evicted count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_leverage_overlay(
    image_path: Path,
    score_grid,
    missing_mask,
    current_evict_mask,
    out_path: Path,
    title: str,
    alpha: float,
    patch_size: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    image = load_image_for_overlay(image_path, score_grid.shape[1] * int(patch_size), score_grid.shape[0] * int(patch_size))
    arr = np.asarray(image)
    masked_scores = np.ma.masked_invalid(score_grid)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(arr)
    overlay = ax.imshow(
        masked_scores,
        cmap="viridis",
        interpolation="nearest",
        alpha=float(alpha),
        extent=(0, arr.shape[1], arr.shape[0], 0),
    )
    if bool(missing_mask.any()):
        dark = np.zeros((*missing_mask.shape, 4), dtype=np.float32)
        dark[..., 3] = missing_mask.astype(np.float32) * min(max(float(alpha), 0.25), 0.75)
        ax.imshow(dark, interpolation="nearest", extent=(0, arr.shape[1], arr.shape[0], 0))
    if bool(current_evict_mask.any()):
        rows, cols = np.where(current_evict_mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            rect = plt.Rectangle(
                (col * int(patch_size), row * int(patch_size)),
                int(patch_size),
                int(patch_size),
                fill=False,
                edgecolor="red",
                linewidth=1.2,
            )
            ax.add_patch(rect)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(overlay, ax=ax, fraction=0.035, pad=0.02, label="candidate leverage score")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def patch_maps_for_event(args: argparse.Namespace, event_meta: dict[str, Any], payload: dict[str, Any]):
    import numpy as np

    patch_w = int(args.image_width) // int(args.patch_size)
    patch_h = int(args.image_height) // int(args.patch_size)
    patch_count = patch_w * patch_h
    patch_start_idx = int(args.patch_start_idx)
    step_idx = int(event_meta.get("step_idx", 0))

    candidate_frames = tensor_to_list(payload, "candidate_frame_ids")
    candidate_tokens = tensor_to_list(payload, "candidate_token_indices")
    candidate_scores = tensor_to_list(payload, "candidate_leverage_scores")
    evicted_frames = tensor_to_list(payload, "evicted_frame_ids")
    evicted_tokens = tensor_to_list(payload, "evicted_token_indices")
    evicted_scores = tensor_to_list(payload, "evicted_leverage_scores")
    anchor_frames = tensor_to_list(payload, "anchor_frame_ids")
    anchor_tokens = tensor_to_list(payload, "anchor_token_indices")

    score_sum: dict[int, Any] = defaultdict(lambda: np.zeros((patch_h, patch_w), dtype=np.float64))
    score_count: dict[int, Any] = defaultdict(lambda: np.zeros((patch_h, patch_w), dtype=np.float64))
    present_by_frame: dict[int, set[tuple[int, int]]] = defaultdict(set)
    current_evicted_by_frame: dict[int, Any] = defaultdict(lambda: np.zeros((patch_h, patch_w), dtype=bool))
    evict_count_by_frame: dict[int, Any] = defaultdict(lambda: np.zeros((patch_h, patch_w), dtype=np.float32))
    evicted_rows: list[dict[str, Any]] = []
    leverage_rows: list[dict[str, Any]] = []
    non_patch_rows: list[dict[str, Any]] = []

    current_evicted_set: set[tuple[int, int]] = set()
    for frame_id, token_idx, score in zip(evicted_frames, evicted_tokens, evicted_scores):
        frame_id = int(frame_id)
        token_idx = int(token_idx)
        patch = token_to_patch(token_idx, patch_start_idx, patch_count, patch_w)
        base = {
            **event_meta,
            "frame_id": frame_id,
            "token_idx": token_idx,
            "leverage_score": None if score is None else float(score),
        }
        if patch is None:
            non_patch_rows.append({**base, "reason": "protected_or_non_patch"})
            continue
        patch_id, patch_row, patch_col = patch
        evict_count_by_frame[frame_id][patch_row, patch_col] += 1.0
        current_evicted_by_frame[frame_id][patch_row, patch_col] = True
        current_evicted_set.add((frame_id, token_idx))
        evicted_rows.append({**base, "patch_id": patch_id, "patch_row": patch_row, "patch_col": patch_col})

    for frame_id, token_idx in zip(anchor_frames, anchor_tokens):
        frame_id = int(frame_id)
        token_idx = int(token_idx)
        patch = token_to_patch(token_idx, patch_start_idx, patch_count, patch_w)
        if patch is None:
            non_patch_rows.append({**event_meta, "frame_id": frame_id, "token_idx": token_idx, "reason": "protected_or_non_patch"})
            continue
        _, patch_row, patch_col = patch
        present_by_frame[frame_id].add((patch_row, patch_col))

    for frame_id, token_idx, score in zip(candidate_frames, candidate_tokens, candidate_scores):
        frame_id = int(frame_id)
        token_idx = int(token_idx)
        patch = token_to_patch(token_idx, patch_start_idx, patch_count, patch_w)
        base = {
            **event_meta,
            "frame_id": frame_id,
            "token_idx": token_idx,
            "leverage_score": float(score),
            "current_step_evicted": (frame_id, token_idx) in current_evicted_set,
            "missing_before_step_eviction": False,
        }
        if patch is None:
            non_patch_rows.append({**base, "reason": "protected_or_non_patch"})
            continue
        patch_id, patch_row, patch_col = patch
        score_sum[frame_id][patch_row, patch_col] += float(score)
        score_count[frame_id][patch_row, patch_col] += 1.0
        present_by_frame[frame_id].add((patch_row, patch_col))
        leverage_rows.append({**base, "patch_id": patch_id, "patch_row": patch_row, "patch_col": patch_col})

    frame_maps = {}
    expected_frames = range(max(step_idx, 0) + 1)
    for frame_id in expected_frames:
        score_grid = np.full((patch_h, patch_w), np.nan, dtype=np.float32)
        count = score_count[frame_id]
        has_score = count > 0
        score_grid[has_score] = (score_sum[frame_id][has_score] / count[has_score]).astype(np.float32)
        present_mask = np.zeros((patch_h, patch_w), dtype=bool)
        for patch_row, patch_col in present_by_frame.get(frame_id, set()):
            present_mask[patch_row, patch_col] = True
        missing_mask = ~present_mask
        frame_maps[frame_id] = {
            "score_grid": score_grid,
            "missing_mask": missing_mask,
            "current_evict_mask": current_evicted_by_frame[frame_id],
            "evict_count": evict_count_by_frame[frame_id],
            "score_values": score_grid[np.isfinite(score_grid)].astype(float).tolist(),
        }
    return frame_maps, evicted_rows, leverage_rows, non_patch_rows


def visualize(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / "eviction_image_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_evicted_rows: list[dict[str, Any]] = []
    all_leverage_rows: list[dict[str, Any]] = []
    all_non_patch_rows: list[dict[str, Any]] = []
    all_frame_summary_rows: list[dict[str, Any]] = []
    saved = 0
    event_count = 0

    for event_meta, payload in iter_token_overlay_events(args) or []:
        event_count += 1
        dataset = str(event_meta["dataset"])
        rank = str(event_meta["rank"])
        scene_key = str(event_meta["scene_key"])
        scene = str(event_meta["scene"])
        step_idx = int(event_meta.get("step_idx", 0))
        layer_id = int(event_meta.get("layer_id", -1))
        layer_label = f"layer_{layer_id:02d}" if layer_id >= 0 else "layer_unknown"
        head_label = str(event_meta.get("head_label", "layer_shared"))
        step_dir = out_dir / dataset / rank / scene_key / f"step_{step_idx:06d}"
        evict_dir = step_dir / "evict_overlay" / layer_label
        leverage_dir = step_dir / "leverage_score_overlay" / layer_label
        csv_dir = step_dir / "csv" / layer_label
        if head_label != "layer_shared":
            evict_dir = evict_dir / head_label
            leverage_dir = leverage_dir / head_label
            csv_dir = csv_dir / head_label

        frame_maps, evicted_rows, leverage_rows, non_patch_rows = patch_maps_for_event(args, event_meta, payload)
        all_evicted_rows.extend(evicted_rows)
        all_leverage_rows.extend(leverage_rows)
        all_non_patch_rows.extend(non_patch_rows)

        frame_summary_rows: list[dict[str, Any]] = []
        for frame_id, maps in sorted(frame_maps.items()):
            evict_count = maps["evict_count"]
            score_values = maps["score_values"]
            row = {
                **event_meta,
                "frame_id": int(frame_id),
                "candidate_scored_patch_count": int((maps["score_grid"] == maps["score_grid"]).sum()),
                "missing_patch_count": int(maps["missing_mask"].sum()),
                "current_step_evicted_patch_count": int((evict_count > 0).sum()),
                "current_step_evicted_token_count": int(evict_count.sum()),
                "leverage_score_mean": None if not score_values else sum(score_values) / len(score_values),
                "leverage_score_p50": quantile(score_values, 0.50),
                "leverage_score_p90": quantile(score_values, 0.90),
                "leverage_score_p95": quantile(score_values, 0.95),
            }
            frame_summary_rows.append(row)
            all_frame_summary_rows.append(row)

            image_path = frame_image_path(dataset, scene, frame_id, args.kf_every)
            if image_path is None or not image_path.exists():
                print(f"Warning: missing source image for {dataset} {scene} frame {frame_id}: {image_path}")
                continue
            if int(evict_count.sum()) >= int(args.overlay_min_count):
                save_evict_overlay(
                    image_path,
                    evict_count,
                    evict_dir / f"frame_{frame_id:06d}.png",
                    f"{dataset} {scene} step={step_idx} layer={layer_id} frame={frame_id} evicted={int(evict_count.sum())}",
                    args.overlay_alpha,
                    args.patch_size,
                )
                saved += 1
            save_leverage_overlay(
                image_path,
                maps["score_grid"],
                maps["missing_mask"],
                maps["current_evict_mask"],
                leverage_dir / f"frame_{frame_id:06d}.png",
                f"{dataset} {scene} step={step_idx} layer={layer_id} frame={frame_id} leverage/missing/current-evict",
                args.overlay_alpha,
                args.patch_size,
            )
            saved += 1

        write_csv(csv_dir / "evicted_tokens.csv", evicted_rows)
        write_csv(csv_dir / "leverage_tokens.csv", leverage_rows)
        write_csv(csv_dir / "non_patch_or_protected_tokens.csv", non_patch_rows)
        write_csv(csv_dir / "frame_summary.csv", frame_summary_rows)

    write_csv(out_dir / "all_evicted_tokens.csv", all_evicted_rows)
    write_csv(out_dir / "all_leverage_tokens.csv", all_leverage_rows)
    write_csv(out_dir / "all_non_patch_or_protected_tokens.csv", all_non_patch_rows)
    write_csv(out_dir / "all_frame_summary.csv", all_frame_summary_rows)
    manifest = {
        "output_dir": str(out_dir),
        "analysis_root": str(args.output_dir / "token_overlay_dumps"),
        "events_processed": event_count,
        "evicted_patch_tokens": len(all_evicted_rows),
        "candidate_patch_tokens": len(all_leverage_rows),
        "non_patch_or_protected_tokens": len(all_non_patch_rows),
        "frame_rows": len(all_frame_summary_rows),
        "overlays_saved": saved,
        "overlay_layout": "<dataset>/<rank>/<scene>/step_<step>/evict_overlay|leverage_score_overlay/layer_<layer>/frame_<frame>.png",
        "csv_layout": "<dataset>/<rank>/<scene>/step_<step>/csv/layer_<layer>/*.csv",
        "patch_start_idx": int(args.patch_start_idx),
        "patch_size": int(args.patch_size),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run mv_recon with leverage diagnostics enabled")
    add_common_paths(run_parser, default_datasets=MV_RECON_RUN_DATASETS)
    run_parser.add_argument("--weights", type=str, default="../ckpt/checkpoints.pth")
    run_parser.add_argument("--max_frames", type=int, default=100)
    run_parser.add_argument("--num_processes", type=int, default=3)
    run_parser.add_argument("--main_process_port", type=int, default=29402)
    run_parser.add_argument("--analysis_layers", type=str, default="23")
    run_parser.add_argument("--analysis_steps", type=str, default="all")
    run_parser.add_argument("--analysis_heads", type=str, default="all")
    run_parser.add_argument("--analysis_space", choices=("full_key", "svd_coord", "both"), default="full_key")
    run_parser.add_argument("--max_evicted_per_event", type=int, default=4096)
    run_parser.add_argument("--save_topk_pairs", type=int, default=4096)
    run_parser.add_argument("--token_overlay_dump_max_events", type=int, default=None)
    run_parser.add_argument("--histogram_bins", type=int, default=100)
    run_parser.add_argument("--histogram_min", type=float, default=0.0)
    run_parser.add_argument("--histogram_max", type=float, default=1.0)
    run_parser.add_argument("--first_frame_special_tokens_only", action=argparse.BooleanOptionalAction, default=False)
    run_parser.add_argument("--dry_run", action="store_true")
    run_parser.add_argument("--skip_run", action="store_true")
    run_parser.add_argument("--skip_visualize", action="store_true", help="Accepted for shared experiment scripts; run subcommand never visualizes.")
    run_parser.add_argument("--extra_launch_arg", action="append", default=[], help="Extra argument forwarded to launch.py; repeat for multiple args.")
    run_parser.set_defaults(func=run_command)

    vis_parser = subparsers.add_parser("visualize", help="Build step-wise image overlays from saved full token diagnostics")
    add_common_paths(vis_parser, default_datasets=ACTIVE_DATASETS)
    vis_parser.add_argument("--overlay_alpha", type=float, default=0.55)
    vis_parser.add_argument("--overlay_top_frames", type=int, default=100)
    vis_parser.add_argument("--overlay_min_count", type=int, default=1)
    vis_parser.add_argument("--skip_visualize", action="store_true")
    vis_parser.add_argument("--skip_run", action="store_true", help="Accepted for shared experiment scripts; visualize subcommand never runs inference.")
    vis_parser.set_defaults(func=lambda args: 0 if args.skip_visualize else visualize(args))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
