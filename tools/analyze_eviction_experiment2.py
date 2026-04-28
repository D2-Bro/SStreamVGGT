#!/usr/bin/env python3
"""Experiment 2: compare mean eviction with two offline SVD references.

This script is analysis-only. It loads the per-head pre-eviction snapshots
produced by the Experiment 1 dump hooks and compares three eviction decisions
under the same budget:

1. The captured mean-based online decision.
2. SVD leverage-score eviction over uncentered candidate keys.
3. SVD projected-space cluster-balanced eviction over centered candidate keys.

No downstream reconstruction, pose, or task metrics are computed here.
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


METHODS = ("mean", "svd_leverage", "svd_projected")


def pca_project(keys: torch.Tensor, projection_dim: int) -> dict[str, torch.Tensor]:
    """Center keys and project them onto the top principal directions.

    For centered matrix ``X`` and SVD ``X = U S V^T``, coordinates are
    ``X @ V[:, :d]`` and explained variance ratio is
    ``S_i^2 / sum_j S_j^2``.
    """
    x = keys.float()
    centered = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    dim = min(projection_dim, vh.shape[0])
    coords = centered @ vh[:dim].T
    variance = s.square()
    total = variance.sum().clamp_min(1e-12)
    return {
        "coords": coords,
        "singular_values": s,
        "explained_variance_ratio": variance / total,
    }


def make_retained_mask(num_tokens: int, evicted_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(num_tokens, dtype=torch.bool)
    mask[evicted_indices.long()] = False
    return mask


def candidate_sort_indices(scores: torch.Tensor, candidate_offset: int, descending: bool = False) -> torch.Tensor:
    """Stable score sort with token-index tie break.

    Args:
        scores: Candidate-local scores.
        candidate_offset: Offset from candidate-local to global token index.
        descending: Sort high-to-low when true, low-to-high when false.
    """
    score_np = scores.detach().cpu().numpy()
    idx_np = np.arange(score_np.shape[0], dtype=np.int64) + int(candidate_offset)
    primary = -score_np if descending else score_np
    order = np.lexsort((idx_np, primary))
    return torch.from_numpy(order.astype(np.int64))


def recompute_mean_eviction(keys: torch.Tensor, num_anchor_tokens: int, num_evicted: int) -> torch.Tensor:
    """Recompute InfiniteVGGT's mean-based eviction decision offline.

    Candidate keys are L2-normalized, the candidate mean is computed, and
    score_i = dot(normalized_key_i, mean_vector). The online rule retains the
    lowest-scoring candidates, so the evicted set is the highest-scoring
    candidates under the same budget.
    """
    candidates = keys[num_anchor_tokens:]
    candidate_norm = F.normalize(candidates.float(), p=2, dim=-1)
    mean_vector = candidate_norm.mean(dim=0, keepdim=True)
    scores = (candidate_norm * mean_vector).sum(dim=-1)
    order = candidate_sort_indices(scores, candidate_offset=num_anchor_tokens, descending=True)
    return order[:num_evicted] + num_anchor_tokens


def load_mean_eviction(snap: dict[str, Any], keys: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    if "evicted_token_indices" in snap:
        return snap["evicted_token_indices"].long()
    num_anchor_tokens = int(meta["num_anchor_tokens"])
    num_evicted = int(meta["cache_size"]) - int(meta["cache_budget"])
    return recompute_mean_eviction(keys, num_anchor_tokens, num_evicted)


def svd_leverage_eviction(
    keys: torch.Tensor,
    num_anchor_tokens: int,
    num_evicted: int,
    leverage_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evict lowest row-leverage tokens from uncentered candidate keys.

    Convention: for uncentered candidate matrix ``K_c = U S V^T`` and rank
    ``r``, token leverage is ``l_i = ||U_i[:r]||_2^2``. Low leverage means the
    row contributes less to the retained principal row subspace, so this
    analysis reference evicts the lowest leverage scores.
    """
    candidates = keys[num_anchor_tokens:].float()
    if candidates.numel() == 0 or num_evicted <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    u, _, _ = torch.linalg.svd(candidates, full_matrices=False)
    rank = max(1, min(int(leverage_rank), u.shape[1]))
    leverage = u[:, :rank].square().sum(dim=1)
    order = candidate_sort_indices(leverage, candidate_offset=num_anchor_tokens, descending=False)
    evicted = order[:num_evicted] + num_anchor_tokens
    return evicted.long(), leverage


def kmeans(coords: torch.Tensor, num_clusters: int, iterations: int = 40) -> torch.Tensor:
    """Small deterministic k-means implementation for projected coordinates."""
    n = coords.shape[0]
    k = max(1, min(int(num_clusters), n))
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


def allocate_cluster_quotas(counts: np.ndarray, num_evicted: int) -> np.ndarray:
    """Allocate evictions proportional to cluster size while preserving coverage."""
    counts = counts.astype(np.int64)
    caps = np.maximum(counts - 1, 0)
    if num_evicted <= 0 or caps.sum() == 0:
        return np.zeros_like(counts)
    target = counts / max(counts.sum(), 1) * int(num_evicted)
    quotas = np.minimum(np.floor(target).astype(np.int64), caps)
    remaining = int(num_evicted - quotas.sum())
    fractional = target - np.floor(target)
    while remaining > 0:
        eligible = np.where(quotas < caps)[0]
        if eligible.size == 0:
            break
        best = eligible[np.lexsort((eligible, -fractional[eligible]))][0]
        quotas[best] += 1
        remaining -= 1
    return quotas


def svd_projected_eviction(
    keys: torch.Tensor,
    num_anchor_tokens: int,
    num_evicted: int,
    projection_dim: int,
    clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster-balanced eviction in centered SVD/PCA candidate coordinates.

    Candidate keys are centered, projected into the top SVD/PCA subspace, then
    clustered. Each cluster receives an eviction quota proportional to its
    candidate population while preserving at least one candidate whenever
    possible. Within each cluster, tokens nearest to the cluster centroid are
    evicted first, treating local centrality as redundancy.
    """
    candidates = keys[num_anchor_tokens:].float()
    if candidates.numel() == 0 or num_evicted <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0)

    projected = pca_project(candidates, projection_dim)["coords"]
    labels = kmeans(projected, clusters)
    label_np = labels.cpu().numpy()
    cluster_ids, counts = np.unique(label_np, return_counts=True)
    quotas_by_pos = allocate_cluster_quotas(counts, num_evicted)

    evicted_local: list[int] = []
    centroid_distances = torch.empty(projected.shape[0], dtype=torch.float32)
    for pos, cluster_id in enumerate(cluster_ids):
        member_idx = torch.nonzero(labels == int(cluster_id), as_tuple=False).flatten()
        member_coords = projected[member_idx]
        centroid = member_coords.mean(dim=0, keepdim=True)
        distances = torch.cdist(member_coords, centroid).flatten()
        centroid_distances[member_idx] = distances
        quota = int(quotas_by_pos[pos])
        if quota <= 0:
            continue
        member_global = member_idx + int(num_anchor_tokens)
        order_np = np.lexsort((member_global.cpu().numpy(), distances.cpu().numpy()))
        evicted_local.extend(member_idx[torch.from_numpy(order_np[:quota])].tolist())

    if len(evicted_local) < num_evicted:
        selected = set(evicted_local)
        all_order = candidate_sort_indices(centroid_distances, candidate_offset=num_anchor_tokens, descending=False)
        for idx in all_order.tolist():
            if idx not in selected:
                evicted_local.append(idx)
                selected.add(idx)
            if len(evicted_local) == num_evicted:
                break

    evicted = torch.tensor(sorted(evicted_local[:num_evicted]), dtype=torch.long) + num_anchor_tokens
    return evicted, labels, centroid_distances


def bin_ids(coords2d: torch.Tensor, bins: int) -> np.ndarray:
    xy = coords2d.cpu().numpy()
    edges = [
        np.linspace(xy[:, axis].min() - 1e-6, xy[:, axis].max() + 1e-6, int(bins) + 1)
        for axis in range(2)
    ]
    ids = []
    for axis in range(2):
        ids.append(np.digitize(xy[:, axis], edges[axis][1:-1], right=False))
    return ids[0] * int(bins) + ids[1]


def knn_density(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Density proxy ``1 / mean distance to k nearest neighbors``."""
    n = coords.shape[0]
    if n <= 1:
        return torch.ones(n)
    kk = max(1, min(int(k), n - 1))
    distances = torch.cdist(coords.float(), coords.float())
    distances.fill_diagonal_(float("inf"))
    mean_knn = torch.topk(distances, k=kk, largest=False).values.mean(dim=1)
    return 1.0 / mean_knn.clamp_min(1e-12)


def group_survival(ids: np.ndarray, retained_mask: torch.Tensor) -> tuple[list[dict[str, float]], float, float, float]:
    """Compute region survival and distortion.

    Survival ratio for region r is ``retained_count_r / original_count_r``.
    Occupancy distortion is total variation distance:
    ``0.5 * sum_r |p_before(r) - p_after(r)|``.
    Imbalance is occupancy-weighted standard deviation of survival ratios.
    Max suppression ratio is ``max_r (1 - survival_r)``.
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
    max_suppression = float(1.0 - survival_values.min()) if survival_values.size else 0.0
    return rows, distortion, imbalance, max_suppression


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
    """Coverage metrics comparing original tokens to retained tokens.

    ``nearest_retained_cosine`` averages max cosine similarity from each
    original key to the retained keys. ``nearest_retained_projected_distance``
    averages projected-space nearest retained Euclidean distance. 
    ``explained_variance_retained`` is retained projected squared energy over
    total projected squared energy.
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


def method_metrics(
    method: str,
    keys: torch.Tensor,
    coords: torch.Tensor,
    evicted: torch.Tensor,
    cluster_ids: np.ndarray,
    bin_labels: np.ndarray,
    top_density: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    retained_mask = make_retained_mask(keys.shape[0], evicted)
    cluster_rows, cluster_distortion, cluster_imbalance, max_cluster_suppression = group_survival(
        cluster_ids, retained_mask
    )
    bin_rows, bin_distortion, _, max_bin_suppression = group_survival(bin_labels, retained_mask)
    cov = coverage_metrics(keys, coords, retained_mask)
    eff_before = effective_rank(keys)
    eff_after = effective_rank(keys[retained_mask])
    evicted_mask = ~retained_mask
    top_density_fraction = float((evicted_mask & top_density).sum().item() / max(evicted_mask.sum().item(), 1))
    metrics = {
        f"{method}_num_evicted": int(evicted.numel()),
        f"{method}_num_retained": int(retained_mask.sum().item()),
        f"{method}_cluster_occupancy_distortion": cluster_distortion,
        f"{method}_bin_occupancy_distortion": bin_distortion,
        f"{method}_cluster_survival_imbalance": cluster_imbalance,
        f"{method}_max_cluster_suppression": max_cluster_suppression,
        f"{method}_max_bin_suppression": max_bin_suppression,
        f"{method}_evicted_top_density_fraction": top_density_fraction,
        f"{method}_effective_rank_before": eff_before,
        f"{method}_effective_rank_after": eff_after,
        f"{method}_effective_rank_drop": eff_before - eff_after,
        f"{method}_nearest_retained_cosine": cov["nearest_retained_cosine"],
        f"{method}_nearest_retained_projected_distance": cov["nearest_retained_projected_distance"],
        f"{method}_explained_variance_retained": cov["explained_variance_retained"],
    }
    return metrics, cluster_rows, bin_rows


def plot_scatter(coords2d: torch.Tensor, evicted: torch.Tensor | None, path: Path, title: str) -> None:
    xy = coords2d.cpu().numpy()
    plt.figure(figsize=(5, 4), dpi=160)
    if evicted is None:
        plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.75)
    else:
        retained = make_retained_mask(coords2d.shape[0], evicted).cpu().numpy()
        plt.scatter(xy[retained, 0], xy[retained, 1], s=8, alpha=0.4, label="retained")
        plt.scatter(xy[~retained, 0], xy[~retained, 1], s=14, alpha=0.9, label="evicted")
        plt.legend(frameon=False)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_survival_comparison(
    rows_by_method: dict[str, list[dict[str, float]]],
    path: Path,
    title: str,
) -> None:
    ids = sorted({row["id"] for rows in rows_by_method.values() for row in rows})
    x = np.arange(len(ids))
    width = 0.8 / len(rows_by_method)
    plt.figure(figsize=(8, 3.5), dpi=160)
    for offset, method in enumerate(METHODS):
        lookup = {row["id"]: row["survival"] for row in rows_by_method[method]}
        values = [lookup.get(i, 0.0) for i in ids]
        plt.bar(x + (offset - 1) * width, values, width=width, label=method)
    plt.xticks(x, ids, rotation=90, fontsize=6)
    plt.ylim(0, 1.05)
    plt.ylabel("survival ratio")
    plt.title(title)
    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_spectrum(singular_values: torch.Tensor, explained: torch.Tensor, path: Path) -> None:
    s = singular_values.cpu().numpy()
    ev = explained.cpu().numpy()
    plt.figure(figsize=(5, 3), dpi=160)
    ax = plt.gca()
    ax.plot(np.arange(1, len(s) + 1), s)
    ax.set_xlabel("component")
    ax.set_ylabel("singular value")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(ev) + 1), np.cumsum(ev), color="tab:orange")
    ax2.set_ylabel("cumulative explained variance")
    plt.title("Spectrum")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_method_boxplot(results: list[dict[str, Any]], metric_suffix: str, path: Path, title: str) -> None:
    values = []
    labels = []
    for method in METHODS:
        key = f"{method}_{metric_suffix}"
        vals = [row[key] for row in results if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])]
        values.append(vals)
        labels.append(method)
    plt.figure(figsize=(7, 3.5), dpi=160)
    plt.boxplot(values, labels=labels, vert=True)
    plt.ylabel(metric_suffix)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def validate_eviction(
    method: str,
    evicted: torch.Tensor,
    num_tokens: int,
    num_anchor_tokens: int,
    num_evicted: int,
    cache_budget: int,
) -> None:
    if evicted.numel() != num_evicted:
        raise ValueError(f"{method}: expected {num_evicted} evictions, got {evicted.numel()}")
    if evicted.numel() and int(evicted.min().item()) < num_anchor_tokens:
        raise ValueError(f"{method}: anchor token was evicted")
    if torch.unique(evicted).numel() != evicted.numel():
        raise ValueError(f"{method}: duplicate evicted indices")
    retained = num_tokens - evicted.numel()
    if retained != cache_budget:
        raise ValueError(f"{method}: retained count {retained} != cache_budget {cache_budget}")


def analyze_snapshot(path: Path, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    snap = torch.load(path, map_location="cpu")
    keys = snap["old_key"].float()
    meta = snap["meta"]
    num_tokens = keys.shape[0]
    num_anchor_tokens = int(meta["num_anchor_tokens"])
    cache_budget = int(meta["cache_budget"])

    mean_evicted = load_mean_eviction(snap, keys, meta)
    num_evicted = int(mean_evicted.numel())
    leverage_rank = args.leverage_rank if args.leverage_rank is not None else args.projection_dim
    svd_leverage_evicted, _ = svd_leverage_eviction(keys, num_anchor_tokens, num_evicted, leverage_rank)
    svd_projected_evicted, _, _ = svd_projected_eviction(
        keys, num_anchor_tokens, num_evicted, args.projection_dim, args.clusters
    )

    evictions = {
        "mean": mean_evicted.long(),
        "svd_leverage": svd_leverage_evicted.long(),
        "svd_projected": svd_projected_evicted.long(),
    }
    for method, evicted in evictions.items():
        validate_eviction(method, evicted, num_tokens, num_anchor_tokens, num_evicted, cache_budget)

    pca = pca_project(keys, args.projection_dim)
    coords = pca["coords"]
    coords2d = coords[:, :2] if coords.shape[1] >= 2 else torch.cat([coords, torch.zeros_like(coords)], dim=1)
    cluster_ids = kmeans(coords, args.clusters).cpu().numpy()
    bin_labels = bin_ids(coords2d, args.bins)
    density = knn_density(coords, args.knn_k)
    top_density = density >= torch.quantile(density, args.top_density_quantile)

    stem = f"step{meta['step_idx']:06d}_layer{meta['layer_id']:02d}_head{meta['head_id']:02d}_batch{meta['batch_id']:02d}"
    head_dir = out_dir / "per_head" / stem
    head_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        **meta,
        "snapshot": str(path),
        "projection_dim": int(args.projection_dim),
        "leverage_rank": int(leverage_rank),
        "clusters": int(args.clusters),
        "bins": int(args.bins),
        "knn_k": int(args.knn_k),
        "num_evicted": num_evicted,
        "leverage_convention": "uncentered candidate keys; leverage_i = ||U_i[:r]||_2^2; evict lowest leverage",
        "projected_convention": "centered candidate keys; PCA projection; k-means; proportional cluster quotas; evict nearest centroid tokens",
    }
    cluster_rows_by_method = {}
    bin_rows_by_method = {}
    for method in METHODS:
        metrics, cluster_rows, bin_rows = method_metrics(
            method,
            keys,
            coords,
            evictions[method],
            cluster_ids,
            bin_labels,
            top_density,
        )
        result.update(metrics)
        cluster_rows_by_method[method] = cluster_rows
        bin_rows_by_method[method] = bin_rows
        with open(head_dir / f"{method}_cluster_survival.json", "w", encoding="utf-8") as f:
            json.dump(cluster_rows, f, indent=2)
        with open(head_dir / f"{method}_bin_survival.json", "w", encoding="utf-8") as f:
            json.dump(bin_rows, f, indent=2)

    plot_scatter(coords2d, None, head_dir / "scatter_before.png", "Before eviction")
    plot_scatter(coords2d, evictions["mean"], head_dir / "scatter_mean_evicted.png", "Mean evicted")
    plot_scatter(
        coords2d,
        evictions["svd_leverage"],
        head_dir / "scatter_svd_leverage_evicted.png",
        "SVD leverage evicted",
    )
    plot_scatter(
        coords2d,
        evictions["svd_projected"],
        head_dir / "scatter_svd_projected_evicted.png",
        "SVD projected evicted",
    )
    plot_survival_comparison(
        cluster_rows_by_method,
        head_dir / "cluster_survival_comparison.png",
        "Per-cluster survival",
    )
    plot_survival_comparison(
        bin_rows_by_method,
        head_dir / "bin_survival_comparison.png",
        "Per-bin survival",
    )
    plot_spectrum(pca["singular_values"], pca["explained_variance_ratio"], head_dir / "spectrum.png")

    with open(head_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_summaries(results: list[dict[str, Any]], out_dir: Path) -> None:
    if not results:
        return
    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(out_dir / "per_head_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary: dict[str, Any] = {"num_snapshots": len(results)}
    aggregate_rows = []
    metric_suffixes = [
        "cluster_occupancy_distortion",
        "bin_occupancy_distortion",
        "cluster_survival_imbalance",
        "max_cluster_suppression",
        "evicted_top_density_fraction",
        "nearest_retained_cosine",
        "nearest_retained_projected_distance",
        "effective_rank_drop",
        "explained_variance_retained",
    ]
    for method in METHODS:
        for suffix in metric_suffixes:
            key = f"{method}_{suffix}"
            values = np.asarray(
                [row[key] for row in results if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            stats = {
                "method": method,
                "metric": suffix,
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
            }
            aggregate_rows.append(stats)
            summary[f"{method}_{suffix}"] = {
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "n": stats["n"],
            }

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "aggregate_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "metric", "n", "mean", "median", "std"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    plot_method_boxplot(
        results,
        "cluster_occupancy_distortion",
        out_dir / "occupancy_distortion_comparison.png",
        "Cluster occupancy distortion",
    )
    plot_method_boxplot(
        results,
        "nearest_retained_projected_distance",
        out_dir / "coverage_distance_comparison.png",
        "Nearest-retained projected distance",
    )
    plot_method_boxplot(
        results,
        "cluster_survival_imbalance",
        out_dir / "cluster_imbalance_comparison.png",
        "Cluster survival imbalance",
    )
    plot_method_boxplot(
        results,
        "evicted_top_density_fraction",
        out_dir / "top_density_suppression_comparison.png",
        "Top-density eviction fraction",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--snapshot_dir", required=True, help="Directory containing per-head .pt snapshots")
    parser.add_argument("--output_dir", required=True, help="Directory for Experiment 2 metrics and figures")
    parser.add_argument("--projection_dim", type=int, default=16, help="SVD/PCA dimension used for metrics")
    parser.add_argument("--leverage_rank", type=int, default=None, help="Rank for leverage scores; defaults to projection_dim")
    parser.add_argument("--clusters", type=int, default=12, help="Number of projected-space k-means clusters")
    parser.add_argument("--bins", type=int, default=8, help="Number of bins per PC axis for 2D occupancy")
    parser.add_argument("--knn_k", type=int, default=16, help="k for local kNN density")
    parser.add_argument("--top_density_quantile", type=float, default=0.75, help="Quantile for top-density tokens")
    parser.add_argument("--max_snapshots", type=int, default=None, help="Optional cap for smoke runs")
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
    write_summaries(results, out_dir)
    print(f"Analyzed {len(results)} snapshots. Results written to {out_dir}")


if __name__ == "__main__":
    main()
