#!/usr/bin/env python3
"""Run an image sequence without KV eviction and analyze keys by layer."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from streamvggt.layers.confidence_state import unpack_kv_cache  # noqa: E402
from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from streamvggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def uncentered_effective_dim(keys: torch.Tensor, eps: float = 1e-12) -> float:
    """Return ``(sum sigma^2)^2 / sum sigma^4`` without centering ``keys``."""
    if keys.ndim != 2:
        raise ValueError(f"keys must have shape [tokens, features], got {tuple(keys.shape)}")
    if keys.numel() == 0:
        return 0.0
    x = keys.float()
    if not torch.isfinite(x).all():
        raise ValueError("keys contain NaN or infinite values")
    energy = x.square().sum()
    if energy.item() <= eps:
        return 0.0
    gram = x.transpose(0, 1) @ x
    fourth_moment = gram.square().sum()
    return float((energy.square() / fourth_moment.clamp_min(eps)).item())


def exact_qr_leverage_pr(
    keys: torch.Tensor, eps: float = 1e-12, rank_tolerance: float = 1e-6
) -> tuple[float, int]:
    """Return participation ratio of exact-QR row leverage scores and QR rank.

    This follows the exact layer-leverage convention in ``EvictionManager``:
    columns whose absolute R diagonal is at most ``rank_tolerance`` times the
    maximum diagonal are excluded before summing squared rows of Q.
    """
    if keys.ndim != 2:
        raise ValueError(f"keys must have shape [tokens, features], got {tuple(keys.shape)}")
    if keys.numel() == 0:
        return 0.0, 0
    x = keys.float()
    if not torch.isfinite(x).all():
        raise ValueError("keys contain NaN or infinite values")
    q, r = torch.linalg.qr(x, mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    if diagonal.numel() == 0:
        return 0.0, 0
    active = diagonal > diagonal.max().clamp_min(eps) * rank_tolerance
    leverage_scores = (q.square() * active.to(dtype=q.dtype).unsqueeze(0)).sum(dim=1)
    total = leverage_scores.sum()
    leverage_pr = total.square() / leverage_scores.square().sum().clamp_min(eps)
    leverage_pr = torch.where(total > eps, leverage_pr, torch.zeros_like(leverage_pr))
    return float(leverage_pr.item()), int(active.sum().item())


def resolve_sequence_images(sequence_dir: Path, num_frames: int, frame_stride: int = 1) -> list[Path]:
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")
    image_paths = sorted(
        path for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    selected = image_paths[::frame_stride][:num_frames]
    if len(selected) < num_frames:
        raise ValueError(
            f"Requested {num_frames} frames from {sequence_dir}, but only {len(selected)} are available "
            f"after frame_stride={frame_stride}"
        )
    return selected


def load_model(checkpoint_path: Path, device: torch.device) -> StreamVGGT:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    print(f"Loading model from {checkpoint_path} ...")
    model = StreamVGGT()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    del state_dict
    return model


def collect_no_eviction_key_caches(
    model: StreamVGGT,
    images: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Process frames sequentially and return final global-attention key caches."""
    aggregator = model.aggregator
    aggregator.reset_stream_state()
    past_key_values = [None] * aggregator.depth

    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    else:
        autocast_dtype = None

    with torch.no_grad():
        for frame_idx in range(images.shape[0]):
            frame = images[frame_idx : frame_idx + 1].unsqueeze(0).to(device)
            autocast_context = (
                torch.cuda.amp.autocast(dtype=autocast_dtype) if device.type == "cuda" else nullcontext()
            )
            with autocast_context:
                _, _, past_key_values = aggregator(
                    frame,
                    past_key_values=past_key_values,
                    use_cache=True,
                    past_frame_idx=frame_idx,
                    current_frame_ids=[frame_idx],
                    current_frame_idx=frame_idx,
                    total_budget=1,
                    cache_write_current_frame=True,
                    cache_evict_current_frame=False,
                )
            print(f"Processed frame {frame_idx + 1}/{images.shape[0]}", flush=True)

    layer_caches: dict[int, torch.Tensor] = {}
    for layer, layer_kv in enumerate(past_key_values):
        if layer_kv is None:
            raise RuntimeError(f"Global attention layer {layer} did not return a KV cache")
        keys, _, _, _ = unpack_kv_cache(layer_kv)
        layer_caches[layer] = keys
    return layer_caches


def validate_layer_caches(layer_caches: dict[int, torch.Tensor]) -> None:
    if not layer_caches:
        raise ValueError("No layer key caches were collected")
    reference_shape = None
    for layer, keys in sorted(layer_caches.items()):
        if keys.ndim != 4:
            raise ValueError(f"Layer {layer} keys must have shape [B,H,N,D], got {tuple(keys.shape)}")
        leading_shape = tuple(keys.shape[:3])
        if reference_shape is None:
            reference_shape = leading_shape
        elif leading_shape != reference_shape:
            raise ValueError(
                f"Layer {layer} cache shape {leading_shape} is not comparable with {reference_shape}"
            )


def analyze_layer_caches(
    layer_caches: dict[int, torch.Tensor], num_frames: int
) -> tuple[
    list[dict[str, int | float]],
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, list[float]],
]:
    """Compute effective dimension and mean full-key norm for each layer and batch."""
    validate_layer_caches(layer_caches)
    rows: list[dict[str, int | float]] = []
    mean_norms_by_layer: dict[int, list[float]] = {}
    dimensions_by_layer: dict[int, list[float]] = {}
    leverage_pr_by_layer: dict[int, list[float]] = {}

    for layer, keys in sorted(layer_caches.items()):
        mean_norms_by_layer[layer] = []
        dimensions_by_layer[layer] = []
        leverage_pr_by_layer[layer] = []
        batch_size, num_heads, num_tokens, head_dim = keys.shape
        for batch in range(batch_size):
            full_keys = keys[batch].permute(1, 0, 2).reshape(num_tokens, num_heads * head_dim)
            key_norms = torch.linalg.vector_norm(full_keys.float(), ord=2, dim=1)
            effective_dim = uncentered_effective_dim(full_keys)
            leverage_pr, leverage_rank = exact_qr_leverage_pr(full_keys)
            mean_norm = float(key_norms.mean().item()) if key_norms.numel() else 0.0
            std_norm = float(key_norms.std(unbiased=False).item()) if key_norms.numel() else 0.0
            rows.append(
                {
                    "layer": layer,
                    "batch": batch,
                    "num_frames": num_frames,
                    "num_heads": num_heads,
                    "num_tokens": num_tokens,
                    "feature_dim": num_heads * head_dim,
                    "effective_dim": effective_dim,
                    "effective_dim_ratio": effective_dim / max(num_heads * head_dim, 1),
                    "leverage_pr": leverage_pr,
                    "leverage_rank": leverage_rank,
                    "mean_key_norm": mean_norm,
                    "std_key_norm": std_norm,
                }
            )
            mean_norms_by_layer[layer].append(mean_norm)
            dimensions_by_layer[layer].append(effective_dim)
            leverage_pr_by_layer[layer].append(leverage_pr)
    return rows, mean_norms_by_layer, dimensions_by_layer, leverage_pr_by_layer


def write_csv(rows: list[dict[str, int | float]], output_path: Path) -> None:
    fieldnames = [
        "layer",
        "batch",
        "num_frames",
        "num_heads",
        "num_tokens",
        "feature_dim",
        "effective_dim",
        "effective_dim_ratio",
        "leverage_pr",
        "leverage_rank",
        "mean_key_norm",
        "std_key_norm",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, int | float]]) -> None:
    header = (
        f"{'layer':>6} {'batch':>6} {'frames':>7} {'heads':>6} {'tokens':>8} "
        f"{'feat':>7} {'eff_dim':>11} {'eff/feat':>10} {'lev_pr':>11} {'lev_rank':>9} "
        f"{'mean_norm':>11} {'std_norm':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['layer']):6d} {int(row['batch']):6d} {int(row['num_frames']):7d} "
            f"{int(row['num_heads']):6d} {int(row['num_tokens']):8d} "
            f"{int(row['feature_dim']):7d} {float(row['effective_dim']):11.4f} "
            f"{float(row['effective_dim_ratio']):10.4f} "
            f"{float(row['leverage_pr']):11.4f} {int(row['leverage_rank']):9d} "
            f"{float(row['mean_key_norm']):11.4f} {float(row['std_key_norm']):10.4f}"
        )


def plot_layer_values(
    values_by_layer: dict[int, list[float]],
    *,
    ylabel: str,
    title: str,
    color: str,
    output_path: Path,
) -> None:
    layers = sorted(values_by_layer)
    means = np.asarray([np.mean(values_by_layer[layer]) for layer in layers], dtype=np.float64)
    stds = np.asarray([np.std(values_by_layer[layer]) for layer in layers], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(max(10.0, 0.5 * len(layers)), 5.5))
    axis.bar(layers, means, yerr=stds, capsize=3, color=color, alpha=0.85)
    axis.set_xticks(layers)
    axis.set_xlabel("Layer")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_results(
    layer_caches: dict[int, torch.Tensor], num_frames: int, output_dir: Path
) -> list[dict[str, int | float]]:
    rows, mean_norms_by_layer, dimensions_by_layer, leverage_pr_by_layer = analyze_layer_caches(
        layer_caches, num_frames
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "layer_key_stats.csv")
    plot_layer_values(
        mean_norms_by_layer,
        ylabel="Mean full-key L2 norm",
        title=f"Mean key norm by layer ({num_frames} frames, no eviction)",
        color="tab:blue",
        output_path=output_dir / "mean_key_norm_by_layer.png",
    )
    plot_layer_values(
        dimensions_by_layer,
        ylabel="Effective dimension",
        title=f"Uncentered effective dimensionality by layer ({num_frames} frames, no eviction)",
        color="tab:orange",
        output_path=output_dir / "effective_dim_by_layer.png",
    )
    plot_layer_values(
        leverage_pr_by_layer,
        ylabel="Leverage participation ratio",
        title=f"Exact-QR leverage PR by layer ({num_frames} frames, no eviction)",
        color="tab:green",
        output_path=output_dir / "leverage_pr_by_layer.png",
    )
    print_table(rows)
    print(f"\nResults written to {output_dir}")
    return rows


def run_sequence_analysis(
    sequence_dir: Path,
    checkpoint_path: Path,
    num_frames: int,
    output_dir: Path,
    *,
    frame_stride: int = 1,
    preprocess_mode: str = "crop",
    device_name: str = "cuda",
) -> list[dict[str, int | float]]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    image_paths = resolve_sequence_images(sequence_dir, num_frames, frame_stride)
    print(f"Using {len(image_paths)} frames from {sequence_dir}")
    images = load_and_preprocess_images([os.fspath(path) for path in image_paths], mode=preprocess_mode)
    model = load_model(checkpoint_path, device)
    layer_caches = collect_no_eviction_key_caches(model, images, device)
    rows = write_results(layer_caches, num_frames, output_dir)
    del layer_caches, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sequence_dir", "--sequence-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "checkpoints.pth",
    )
    parser.add_argument("--num_frames", "--num-frames", type=int, required=True)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--frame_stride", "--frame-stride", type=int, default=1)
    parser.add_argument("--preprocess_mode", "--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:1, or cpu")
    args = parser.parse_args()
    run_sequence_analysis(
        args.sequence_dir,
        args.checkpoint_path,
        args.num_frames,
        args.output_dir,
        frame_stride=args.frame_stride,
        preprocess_mode=args.preprocess_mode,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
