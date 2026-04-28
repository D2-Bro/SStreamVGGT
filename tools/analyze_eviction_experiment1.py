#!/usr/bin/env python3
"""Offline analysis for Experiment 1: mean-based eviction vs local structure.

The script loads per-head snapshots dumped by ``--cache_analysis_dir`` and
analyzes each attention head independently. It does not propose or integrate a
new eviction method; random eviction is included only as an offline reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def pca_project(keys: torch.Tensor, projection_dim: int) -> dict[str, torch.Tensor]:
    """Center keys and project them onto the top principal directions.

    If ``X`` is the centered ``N x D`` key matrix and ``X = U S V^T``, projected
    coordinates are ``X V[:, :d]``. Explained variance ratio is
    ``S_i^2 / sum_j S_j^2``.
    """
    x = keys.float()
    centered = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    dim = min(projection_dim, vh.shape[0])
    components = vh[:dim].T
    coords = centered @ components
    variance = s.square()
    total = variance.sum().clamp_min(1e-12)
    return {
        "coords": coords,
        "singular_values": s,
        "explained_variance_ratio": variance / total,
        "components": components,
        "mean": x.mean(dim=0),
    }


def make_retained_mask(num_tokens: int, evicted_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(num_tokens, dtype=torch.bool)
    mask[evicted_indices.long()] = False
    return mask


def bin_ids(coords2d: torch.Tensor, bins: int) -> tuple[np.ndarray, list[np.ndarray]]:
    xy = coords2d.cpu().numpy()
    edges = [
        np.linspace(xy[:, axis].min() - 1e-6, xy[:, axis].max() + 1e-6, bins + 1)
        for axis in range(2)
    ]
    ids = []
    for axis in range(2):
        axis_ids = np.digitize(xy[:, axis], edges[axis][1:-1], right=False)
        ids.append(axis_ids)
    return ids[0] * bins + ids[1], edges


def kmeans(coords: torch.Tensor, num_clusters: int, iterations: int = 40) -> torch.Tensor:
    """Small deterministic k-means for projected coordinates."""
    n = coords.shape[0]
    k = max(1, min(num_clusters, n))
    init_idx = torch.linspace(0, n - 1, steps=k).round().long()
    centers = coords[init_idx].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iterations):
        distances = torch.cdist(coords, centers)
        labels = distances.argmin(dim=1)
        new_centers = centers.clone()
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                new_centers[cluster_id] = coords[mask].mean(dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def knn_density(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Estimate local density as inverse mean kNN distance in projected space."""
    n = coords.shape[0]
    if n <= 1:
        return torch.ones(n)
    kk = max(1, min(k, n - 1))
    distances = torch.cdist(coords, coords)
    distances.fill_diagonal_(float("inf"))
    mean_knn = torch.topk(distances, k=kk, largest=False).values.mean(dim=1)
    return 1.0 / mean_knn.clamp_min(1e-12)


def group_survival(ids: np.ndarray, retained_mask: torch.Tensor) -> tuple[list[dict[str, float]], float, float]:
    """Return per-region survival and distortion.

    Survival ratio for region r is ``retained_count_r / original_count_r``.
    Occupancy distortion is total variation distance between region occupancy
    distributions before and after eviction:
    ``0.5 * sum_r |p_before(r) - p_after(r)|``.
    Imbalance is the occupancy-weighted standard deviation of survival ratios.
    """
    retained = retained_mask.cpu().numpy()
    rows = []
    before_counts = []
    after_counts = []
    for group_id in sorted(np.unique(ids).tolist()):
        group_mask = ids == group_id
        before = int(group_mask.sum())
        after = int((group_mask & retained).sum())
        survival = float(after / before) if before else 0.0
        rows.append({"id": int(group_id), "before": before, "after": after, "survival": survival})
        before_counts.append(before)
        after_counts.append(after)

    before_arr = np.asarray(before_counts, dtype=np.float64)
    after_arr = np.asarray(after_counts, dtype=np.float64)
    p_before = before_arr / max(before_arr.sum(), 1.0)
    p_after = after_arr / max(after_arr.sum(), 1.0)
    distortion = float(0.5 * np.abs(p_before - p_after).sum())
    survival_values = np.asarray([row["survival"] for row in rows], dtype=np.float64)
    imbalance = float(np.sqrt(np.sum(p_before * (survival_values - survival_values.mean()) ** 2)))
    return rows, distortion, imbalance


def effective_rank(keys: torch.Tensor) -> float:
    """Effective rank ``exp(H(p))`` where ``p_i = S_i^2 / sum_j S_j^2``."""
    if keys.shape[0] <= 1:
        return 0.0
    centered = keys.float() - keys.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    p = s.square()
    p = p / p.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


def coverage_metrics(keys: torch.Tensor, coords: torch.Tensor, retained_mask: torch.Tensor) -> dict[str, float]:
    """Coverage metrics comparing all original tokens to retained tokens.

    ``nearest_retained_cosine`` averages, over every original token, the maximum
    cosine similarity to any retained key. ``nearest_retained_projected_distance``
    averages the Euclidean distance from every original projected coordinate to
    its nearest retained projected coordinate. ``explained_variance_retained`` is
    the fraction of total projected squared energy retained after eviction.
    """
    retained_keys = keys[retained_mask]
    retained_coords = coords[retained_mask]
    if retained_keys.numel() == 0:
        return {
            "nearest_retained_cosine": float("nan"),
            "nearest_retained_projected_distance": float("nan"),
            "explained_variance_retained": 0.0,
        }
    sim = F.normalize(keys.float(), dim=-1) @ F.normalize(retained_keys.float(), dim=-1).T
    nearest_cosine = sim.max(dim=1).values.mean()
    distances = torch.cdist(coords.float(), retained_coords.float())
    nearest_distance = distances.min(dim=1).values.mean()
    retained_energy = retained_coords.square().sum()
    total_energy = coords.square().sum().clamp_min(1e-12)
    return {
        "nearest_retained_cosine": float(nearest_cosine.item()),
        "nearest_retained_projected_distance": float(nearest_distance.item()),
        "explained_variance_retained": float((retained_energy / total_energy).item()),
    }


def plot_scatter(coords2d: torch.Tensor, retained_mask: torch.Tensor, path: Path, mode: str) -> None:
    xy = coords2d.cpu().numpy()
    retained = retained_mask.cpu().numpy()
    plt.figure(figsize=(5, 4), dpi=160)
    if mode == "before":
        plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.75)
        plt.title("Before eviction")
    elif mode == "after":
        plt.scatter(xy[retained, 0], xy[retained, 1], s=8, alpha=0.75)
        plt.title("After mean-based eviction")
    else:
        plt.scatter(xy[retained, 0], xy[retained, 1], s=8, alpha=0.45, label="retained")
        plt.scatter(xy[~retained, 0], xy[~retained, 1], s=14, alpha=0.9, label="evicted")
        plt.legend(frameon=False)
        plt.title("Evicted tokens highlighted")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_survival(rows: list[dict[str, float]], path: Path, title: str) -> None:
    labels = [row["id"] for row in rows]
    survival = [row["survival"] for row in rows]
    plt.figure(figsize=(6, 3), dpi=160)
    plt.bar(range(len(labels)), survival)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.ylim(0, 1.05)
    plt.ylabel("survival ratio")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_spectrum(singular_values: torch.Tensor, explained: torch.Tensor, path: Path) -> None:
    s = singular_values.cpu().numpy()
    ev = explained.cpu().numpy()
    plt.figure(figsize=(5, 3), dpi=160)
    ax = plt.gca()
    ax.plot(np.arange(1, len(s) + 1), s, label="singular value")
    ax.set_xlabel("component")
    ax.set_ylabel("singular value")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(ev) + 1), np.cumsum(ev), color="tab:orange", label="cumulative explained variance")
    ax2.set_ylabel("cumulative explained variance")
    plt.title("Spectrum")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def analyze_snapshot(path: Path, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    snap = torch.load(path, map_location="cpu")
    keys = snap["old_key"].float()
    evicted = snap["evicted_token_indices"].long()
    retained_mask = make_retained_mask(keys.shape[0], evicted)

    pca = pca_project(keys, args.projection_dim)
    coords = pca["coords"]
    coords2d = coords[:, :2] if coords.shape[1] >= 2 else torch.cat([coords, torch.zeros_like(coords)], dim=1)

    bin_labels, _ = bin_ids(coords2d, args.bins)
    cluster_labels = kmeans(coords, args.clusters).cpu().numpy()
    density = knn_density(coords, args.knn_k)
    density_threshold = torch.quantile(density, args.top_density_quantile)
    top_density = density >= density_threshold
    evicted_mask = ~retained_mask
    evicted_top_density_fraction = float((evicted_mask & top_density).sum().item() / max(evicted_mask.sum().item(), 1))

    bin_rows, bin_distortion, _ = group_survival(bin_labels, retained_mask)
    cluster_rows, cluster_distortion, cluster_imbalance = group_survival(cluster_labels, retained_mask)

    metrics = coverage_metrics(keys, coords, retained_mask)
    eff_before = effective_rank(keys)
    eff_after = effective_rank(keys[retained_mask])

    meta = snap["meta"]
    stem = f"step{meta['step_idx']:06d}_layer{meta['layer_id']:02d}_head{meta['head_id']:02d}_batch{meta['batch_id']:02d}"
    head_dir = out_dir / "per_head" / stem
    head_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter(coords2d, retained_mask, head_dir / "scatter_before.png", "before")
    plot_scatter(coords2d, retained_mask, head_dir / "scatter_after.png", "after")
    plot_scatter(coords2d, retained_mask, head_dir / "scatter_evicted.png", "evicted")
    plot_survival(bin_rows, head_dir / "bin_survival.png", "Per-bin survival")
    plot_survival(cluster_rows, head_dir / "cluster_survival.png", "Per-cluster survival")
    plot_spectrum(pca["singular_values"], pca["explained_variance_ratio"], head_dir / "spectrum.png")

    result = {
        **meta,
        "snapshot": str(path),
        "num_evicted": int(evicted.numel()),
        "num_retained": int(retained_mask.sum().item()),
        "bin_occupancy_distortion": bin_distortion,
        "cluster_occupancy_distortion": cluster_distortion,
        "cluster_survival_imbalance": cluster_imbalance,
        "evicted_top_density_fraction": evicted_top_density_fraction,
        "effective_rank_before": eff_before,
        "effective_rank_after": eff_after,
        "effective_rank_drop": eff_before - eff_after,
        **metrics,
    }

    if args.random_baseline:
        generator = torch.Generator().manual_seed(args.random_seed + int(meta["layer_id"]) * 1009 + int(meta["head_id"]))
        random_evicted = torch.randperm(keys.shape[0], generator=generator)[: evicted.numel()]
        random_retained = make_retained_mask(keys.shape[0], random_evicted)
        random_metrics = coverage_metrics(keys, coords, random_retained)
        random_cluster_rows, random_cluster_distortion, random_cluster_imbalance = group_survival(
            cluster_labels, random_retained
        )
        result.update(
            {
                "random_nearest_retained_projected_distance": random_metrics[
                    "nearest_retained_projected_distance"
                ],
                "random_nearest_retained_cosine": random_metrics["nearest_retained_cosine"],
                "random_cluster_occupancy_distortion": random_cluster_distortion,
                "random_cluster_survival_imbalance": random_cluster_imbalance,
            }
        )
        with open(head_dir / "random_cluster_survival.json", "w", encoding="utf-8") as f:
            json.dump(random_cluster_rows, f, indent=2)

    with open(head_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(head_dir / "bin_survival.json", "w", encoding="utf-8") as f:
        json.dump(bin_rows, f, indent=2)
    with open(head_dir / "cluster_survival.json", "w", encoding="utf-8") as f:
        json.dump(cluster_rows, f, indent=2)
    return result


def plot_aggregate(results: list[dict[str, Any]], key: str, path: Path, title: str) -> None:
    values = np.asarray([row[key] for row in results if row.get(key) is not None and not math.isnan(row[key])])
    if values.size == 0:
        return
    plt.figure(figsize=(5, 3), dpi=160)
    plt.boxplot(values, vert=False)
    plt.xlabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    if not results:
        return
    csv_path = out_dir / "per_head_metrics.csv"
    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    numeric = {}
    for key in fieldnames:
        vals = [row[key] for row in results if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            numeric[key] = {"mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std())}
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(numeric, f, indent=2)

    plot_aggregate(results, "nearest_retained_projected_distance", out_dir / "coverage_distance_boxplot.png", "Coverage distance")
    plot_aggregate(results, "cluster_occupancy_distortion", out_dir / "occupancy_distortion_boxplot.png", "Cluster occupancy distortion")
    plot_aggregate(results, "cluster_survival_imbalance", out_dir / "cluster_imbalance_boxplot.png", "Cluster survival imbalance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--snapshot_dir", required=True, help="Directory containing per-head .pt snapshots")
    parser.add_argument("--output_dir", required=True, help="Directory for analysis metrics and figures")
    parser.add_argument("--projection_dim", type=int, default=8, help="PCA dimension used for metrics")
    parser.add_argument("--bins", type=int, default=8, help="Number of bins per PC axis for 2D occupancy")
    parser.add_argument("--clusters", type=int, default=12, help="Number of k-means clusters")
    parser.add_argument("--knn_k", type=int, default=16, help="k for local kNN density")
    parser.add_argument("--top_density_quantile", type=float, default=0.75, help="Quantile threshold for top-density tokens")
    parser.add_argument("--random_baseline", action="store_true", help="Add offline random eviction reference")
    parser.add_argument("--random_seed", type=int, default=0, help="Seed for random eviction reference")
    parser.add_argument("--max_snapshots", type=int, default=None, help="Optional cap for quick analysis")
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(snapshot_dir.glob("*.pt"))
    if args.max_snapshots is not None:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise SystemExit(f"No .pt snapshots found under {snapshot_dir}")

    results = [analyze_snapshot(path, args, out_dir) for path in paths]
    write_summary(results, out_dir)
    print(f"Analyzed {len(results)} snapshots. Results written to {out_dir}")


if __name__ == "__main__":
    main()
