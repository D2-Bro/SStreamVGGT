#!/usr/bin/env python3
"""Analyze VGGT depth confidence distributions from direct inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
NATURAL_SORT_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT/StreamVGGT inference and summarize predictions['depth_conf']."
    )
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image_dir", required=True, help="Directory containing RGB frames.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary outputs.")
    parser.add_argument(
        "--model_name",
        default="StreamVGGT",
        choices=("StreamVGGT", "VGGT"),
        help="Model class to instantiate.",
    )
    parser.add_argument("--max_frames", type=int, default=500, help="Maximum number of frames to analyze.")
    parser.add_argument("--frame_stride", type=int, default=2, help="Frame stride before max_frames is applied.")
    parser.add_argument("--size", type=int, default=518, help="Input size passed to load_images_for_eval.")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--budget", type=int, default=200000, help="StreamVGGT total token budget.")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins.")
    parser.add_argument(
        "--log_conf",
        action="store_true",
        help="Also write distribution files for log(depth_conf).",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable load_images_for_eval center crop.",
    )
    return parser.parse_args()


def resolve_device(device_spec: str) -> torch.device:
    if device_spec.startswith("cuda") and not torch.cuda.is_available():
        print("[depth-conf] CUDA is not available; falling back to CPU. This may be slow.", flush=True)
        return torch.device("cpu")
    return torch.device(device_spec)


def natural_sort_key(path: Path) -> tuple:
    parts = NATURAL_SORT_RE.split(path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def collect_image_paths(image_dir: str, frame_stride: int, max_frames: int | None) -> list[str]:
    if frame_stride < 1:
        raise ValueError(f"--frame_stride must be >= 1, got {frame_stride}")
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths = [str(path) for path in sorted(paths, key=natural_sort_key)]
    paths = paths[::frame_stride]
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError(f"--max_frames must be >= 1, got {max_frames}")
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No RGB images found in {root} with extensions {IMAGE_EXTENSIONS}")
    return paths


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(args.weights)
    if args.model_name == "StreamVGGT":
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT(total_budget=args.budget)
    elif args.model_name == "VGGT":
        from vggt.models.vggt import VGGT

        model = VGGT()
    else:
        raise ValueError(f"Unsupported --model_name: {args.model_name}")

    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def prepare_views(image_paths: list[str], size: int, crop: bool, device: torch.device) -> list[dict]:
    from dust3r.utils.image import load_images_for_eval

    views = load_images_for_eval(image_paths, size=size, crop=crop)
    prepared = []
    for view in views:
        prepared_view = dict(view)
        prepared_view["img"] = prepared_view["img"].to(device)
        if "true_shape" in prepared_view:
            prepared_view["true_shape"] = torch.from_numpy(prepared_view["true_shape"]).to(device)
        prepared.append(prepared_view)
    return prepared


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def as_frame_sequence(value, name: str) -> np.ndarray:
    arr = tensor_to_numpy(value)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim >= 4:
        height, width = arr.shape[-2:]
        return arr.reshape(-1, height, width)
    raise ValueError(f"Unsupported {name} shape: {arr.shape}")


def extract_conf_sequence(outputs, dict_key: str, ress_key: str) -> np.ndarray:
    if hasattr(outputs, "ress") and outputs.ress is not None:
        frames = []
        for idx, result in enumerate(outputs.ress):
            if ress_key not in result:
                raise KeyError(f"outputs.ress[{idx}] does not contain {ress_key!r}")
            frames.append(as_frame_sequence(result[ress_key], ress_key))
        return np.concatenate(frames, axis=0)

    if isinstance(outputs, dict):
        if dict_key not in outputs:
            raise KeyError(f"Model output dictionary does not contain {dict_key!r}")
        return as_frame_sequence(outputs[dict_key], dict_key)

    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def compute_stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("No finite confidence values to summarize")
    stats: dict[str, float | int] = {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }
    for percentile in PERCENTILES:
        stats[f"p{percentile}"] = float(np.percentile(flat, percentile))
    return stats


def frame_stats(conf_sequence: np.ndarray, names: Iterable[str]) -> list[dict[str, float | int | str]]:
    rows = []
    for idx, (frame_conf, name) in enumerate(zip(conf_sequence, names)):
        row = {"frame_index": idx, "image": name}
        row.update(compute_stats(frame_conf))
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_frame_stats_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_histogram_csv(path: Path, counts: np.ndarray, edges: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "count"])
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([float(left), float(right), int(count)])


def write_histogram_png(path: Path, values: np.ndarray, bins: int, title: str, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(flat, bins=bins, color="#3274a1", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_distribution_outputs(
    output_dir: Path,
    prefix: str,
    values: np.ndarray,
    image_names: list[str],
    bins: int,
    metadata: dict,
) -> dict:
    summary = {
        "metadata": metadata,
        "overall": compute_stats(values),
    }
    rows = frame_stats(values, image_names)
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    counts, edges = np.histogram(flat, bins=bins)

    write_json(output_dir / f"{prefix}_summary.json", summary)
    write_frame_stats_csv(output_dir / f"{prefix}_frame_stats.csv", rows)
    write_histogram_csv(output_dir / f"{prefix}_histogram.csv", counts, edges)
    write_histogram_png(
        output_dir / f"{prefix}_histogram.png",
        values,
        bins,
        title=prefix.replace("_", " ").title(),
        xlabel=prefix,
    )
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(args.image_dir, args.frame_stride, args.max_frames)
    image_names = [Path(path).name for path in image_paths]
    print(
        f"[depth-conf] Running {args.model_name} on {len(image_paths)} frames "
        f"(device={device}, size={args.size})",
        flush=True,
    )

    model = load_model(args, device)
    views = prepare_views(image_paths, args.size, crop=not args.no_crop, device=device)
    with torch.no_grad():
        outputs = model(views)
    conf_specs = [
        ("depth_conf", extract_conf_sequence(outputs, "depth_conf", "depth_conf"), "depth_conf"),
        (
            "world_points_conf",
            extract_conf_sequence(outputs, "world_points_conf", "conf"),
            "world_points_conf",
        ),
    ]

    base_metadata = {
        "weights": args.weights,
        "image_dir": args.image_dir,
        "model_name": args.model_name,
        "input_images": len(image_paths),
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "size": args.size,
        "device": str(device),
        "bins": args.bins,
    }
    summaries = {}
    for prefix, conf, conf_source in conf_specs:
        metadata = {
            **base_metadata,
            "conf_source": conf_source,
            "num_frames": int(conf.shape[0]),
            "conf_shape": list(conf.shape),
        }
        summaries[prefix] = write_distribution_outputs(
            output_dir=output_dir,
            prefix=prefix,
            values=conf,
            image_names=image_names,
            bins=args.bins,
            metadata=metadata,
        )

        if args.log_conf:
            positive = conf[np.isfinite(conf) & (conf > 0)]
            if positive.size == 0:
                raise ValueError(f"--log_conf requested, but no positive finite values found for {prefix}")
            log_conf = np.full(conf.shape, np.nan, dtype=np.float32)
            valid = np.isfinite(conf) & (conf > 0)
            log_conf[valid] = np.log(conf[valid])
            write_distribution_outputs(
                output_dir=output_dir,
                prefix=f"log_{prefix}",
                values=log_conf,
                image_names=image_names,
                bins=args.bins,
                metadata={**metadata, "source": f"np.log({prefix})"},
            )

    for prefix, summary in summaries.items():
        overall = summary["overall"]
        print(
            f"[{prefix}] "
            f"count={overall['count']} mean={overall['mean']:.6g} std={overall['std']:.6g} "
            f"min={overall['min']:.6g} p50={overall['p50']:.6g} "
            f"p90={overall['p90']:.6g} p95={overall['p95']:.6g} "
            f"p99={overall['p99']:.6g} max={overall['max']:.6g}",
            flush=True,
        )
    print(f"[depth-conf] Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
