#!/usr/bin/env python3
"""Analyze mean-vs-SVD results produced from common pre-eviction snapshots."""

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

PREFERRED_POLICIES = ("mean", "svd_leverage", "layer_svd_leverage")


def comparison_policies(comp: dict[str, Any]) -> tuple[str, ...]:
    """Return policy result keys present in a comparison payload."""
    return tuple(policy for policy in PREFERRED_POLICIES if policy in comp)


def pca_project(keys: torch.Tensor, projection_dim: int) -> dict[str, torch.Tensor]:
    """Center keys and project them onto top principal directions."""
    x = keys.float()
    centered = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    dim = vh.shape[0] if int(projection_dim) < 0 else min(int(projection_dim), vh.shape[0])
    coords = centered @ vh[:dim].T
    variance = s.square()
    total = variance.sum().clamp_min(1e-12)
    return {
        "coords": coords,
        "singular_values": s,
        "explained_variance_ratio": variance / total,
    }


def make_retained_mask(num_tokens: int, evicted_indices: torch.Tensor) -> torch.Tensor:
    """Build a retained-token mask from global evicted token indices."""
    mask = torch.ones(num_tokens, dtype=torch.bool)
    if evicted_indices.numel() > 0:
        mask[evicted_indices.long()] = False
    return mask


def bin_ids(coords2d: torch.Tensor, bins: int) -> np.ndarray:
    """Assign projected 2D coordinates to regular grid bins."""
    xy = coords2d.cpu().numpy()
    edges = [
        np.linspace(xy[:, axis].min() - 1e-6, xy[:, axis].max() + 1e-6, int(bins) + 1)
        for axis in range(2)
    ]
    ids = []
    for axis in range(2):
        ids.append(np.digitize(xy[:, axis], edges[axis][1:-1], right=False))
    return ids[0] * int(bins) + ids[1]


def kmeans(coords: torch.Tensor, num_clusters: int, iterations: int = 40) -> torch.Tensor:
    """Small deterministic k-means implementation."""
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
    """Compute survival, occupancy distortion, imbalance, and max suppression."""
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


def local_region_hole_metrics(
    rows: list[dict[str, float]],
    *,
    total_count: int,
    min_region_size: int,
    low_survival_threshold: float,
    prefix: str,
) -> dict[str, float]:
    """Summarize empty-vs-thinned local regions from survival rows."""
    valid_rows = [row for row in rows if int(row["before"]) >= int(min_region_size)]
    if not valid_rows:
        return {
            f"{prefix}_valid_region_count": 0,
            f"{prefix}_empty_region_fraction": 0.0,
            f"{prefix}_empty_region_mass": 0.0,
            f"{prefix}_low_survival_nonempty_region_fraction": 0.0,
            f"{prefix}_low_survival_nonempty_region_mass": 0.0,
            f"{prefix}_p05_region_survival": float("nan"),
            f"{prefix}_min_region_survival": float("nan"),
        }

    empty_rows = [row for row in valid_rows if int(row["after"]) == 0]
    low_nonempty_rows = [
        row
        for row in valid_rows
        if int(row["after"]) > 0 and float(row["survival"]) < float(low_survival_threshold)
    ]
    survival = np.asarray([float(row["survival"]) for row in valid_rows], dtype=np.float64)
    denom_regions = max(len(valid_rows), 1)
    denom_tokens = max(int(total_count), 1)
    return {
        f"{prefix}_valid_region_count": int(len(valid_rows)),
        f"{prefix}_empty_region_fraction": float(len(empty_rows) / denom_regions),
        f"{prefix}_empty_region_mass": float(sum(int(row["before"]) for row in empty_rows) / denom_tokens),
        f"{prefix}_low_survival_nonempty_region_fraction": float(len(low_nonempty_rows) / denom_regions),
        f"{prefix}_low_survival_nonempty_region_mass": float(
            sum(int(row["before"]) for row in low_nonempty_rows) / denom_tokens
        ),
        f"{prefix}_p05_region_survival": float(np.quantile(survival, 0.05)),
        f"{prefix}_min_region_survival": float(survival.min()),
    }


def evicted_key_coverage_metrics(
    keys: torch.Tensor,
    evicted: torch.Tensor,
    retained_mask: torch.Tensor,
    *,
    cosine_threshold: float,
) -> dict[str, float]:
    """Measure whether evicted 64D key directions have retained neighbors."""
    if evicted.numel() == 0:
        return {
            "key64_evicted_nearest_retained_cosine_mean": float("nan"),
            "key64_evicted_nearest_retained_cosine_p05": float("nan"),
            "key64_evicted_nearest_retained_cosine_min": float("nan"),
            "key64_uncovered_evicted_fraction": 0.0,
        }
    retained_keys = keys[retained_mask]
    if retained_keys.numel() == 0:
        return {
            "key64_evicted_nearest_retained_cosine_mean": float("nan"),
            "key64_evicted_nearest_retained_cosine_p05": float("nan"),
            "key64_evicted_nearest_retained_cosine_min": float("nan"),
            "key64_uncovered_evicted_fraction": 1.0,
        }
    evicted_keys = keys[evicted.long()]
    sim = F.normalize(evicted_keys.float(), dim=-1) @ F.normalize(retained_keys.float(), dim=-1).T
    nearest = sim.max(dim=1).values.float()
    return {
        "key64_evicted_nearest_retained_cosine_mean": float(nearest.mean().item()),
        "key64_evicted_nearest_retained_cosine_p05": float(torch.quantile(nearest, 0.05).item()),
        "key64_evicted_nearest_retained_cosine_min": float(nearest.min().item()),
        "key64_uncovered_evicted_fraction": float((nearest < float(cosine_threshold)).float().mean().item()),
    }


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
    """Nearest retained cosine/distance and projected energy retained."""
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


def plot_scatter(coords2d: torch.Tensor, evicted: torch.Tensor | None, path: Path, title: str) -> None:
    """Plot projected keys before eviction or with evicted tokens highlighted."""
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


def plot_policy_scatter(coords2d: torch.Tensor, policy_evictions: dict[str, torch.Tensor], path: Path) -> None:
    """Plot policy evictions side by side in the same PCA basis."""
    xy = coords2d.cpu().numpy()
    policies = list(policy_evictions.keys())
    fig, axes = plt.subplots(1, len(policies), figsize=(4.5 * len(policies), 4), dpi=160, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, policy in zip(axes, policies):
        evicted = policy_evictions[policy]
        retained = make_retained_mask(coords2d.shape[0], evicted).cpu().numpy()
        ax.scatter(xy[retained, 0], xy[retained, 1], s=7, alpha=0.35, label="retained")
        ax.scatter(xy[~retained, 0], xy[~retained, 1], s=13, alpha=0.9, label="evicted")
        ax.set_title(policy)
        ax.set_xlabel("PC1")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_survival(rows: list[dict[str, float]], path: Path, title: str) -> None:
    """Plot per-group survival ratios."""
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
    """Plot singular values and cumulative explained variance."""
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


def plot_policy_boxplot(results: list[dict[str, Any]], key: str, path: Path, title: str) -> None:
    """Plot aggregate metric distributions for mean and SVD policies."""
    values = []
    labels = []
    for policy in PREFERRED_POLICIES:
        policy_values = [
            row[key]
            for row in results
            if row.get("policy") == policy and isinstance(row.get(key), (int, float)) and not math.isnan(row[key])
        ]
        if policy_values:
            values.append(policy_values)
            labels.append(policy)
    if not values:
        return
    plt.figure(figsize=(6, 3), dpi=160)
    plt.boxplot(values, labels=labels, vert=False)
    plt.xlabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_delta_boxplot(rows: list[dict[str, Any]], key: str, path: Path, title: str) -> None:
    """Plot paired SVD-minus-mean metric deltas."""
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])]
    if not values:
        return
    plt.figure(figsize=(5, 3), dpi=160)
    plt.boxplot(values, vert=False)
    plt.axvline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def resolve_source_snapshot(path: str, compare_path: Path) -> Path:
    """Resolve source snapshot paths stored by the comparison script."""
    source = Path(path)
    candidates = [source]
    if not source.is_absolute():
        candidates.append(Path.cwd() / source)
        candidates.append(compare_path.parent / source)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve source snapshot '{path}' from {compare_path}")


def analyze_policy(
    *,
    policy: str,
    evicted: torch.Tensor,
    keys: torch.Tensor,
    coords: torch.Tensor,
    coords2d: torch.Tensor,
    cluster_labels: np.ndarray,
    bin_labels: np.ndarray,
    key64_cluster_labels: np.ndarray,
    top_density: torch.Tensor,
    meta: dict[str, Any],
    compare_path: Path,
    head_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Analyze one policy decision inside a shared comparison result."""
    retained_mask = make_retained_mask(keys.shape[0], evicted)
    evicted_mask = ~retained_mask
    cluster_rows, cluster_distortion, cluster_imbalance, max_cluster_suppression = group_survival(
        cluster_labels, retained_mask
    )
    bin_rows, bin_distortion, _, max_bin_suppression = group_survival(bin_labels, retained_mask)
    key64_cluster_rows, key64_cluster_distortion, key64_cluster_imbalance, max_key64_cluster_suppression = (
        group_survival(key64_cluster_labels, retained_mask)
    )
    cov = coverage_metrics(keys, coords, retained_mask)
    key64_holes = local_region_hole_metrics(
        key64_cluster_rows,
        total_count=keys.shape[0],
        min_region_size=args.key_min_region_size,
        low_survival_threshold=args.key_low_survival_threshold,
        prefix="key64_cluster",
    )
    key64_evicted_coverage = evicted_key_coverage_metrics(
        keys,
        evicted,
        retained_mask,
        cosine_threshold=args.key_uncovered_cosine_threshold,
    )
    eff_before = effective_rank(keys)
    eff_after = effective_rank(keys[retained_mask])
    evicted_top_density_fraction = float((evicted_mask & top_density).sum().item() / max(evicted_mask.sum().item(), 1))

    policy_dir = head_dir / policy
    policy_dir.mkdir(parents=True, exist_ok=True)
    result = {
        **meta,
        "comparison_file": str(compare_path),
        "policy": policy,
        "projection_dim": int(args.projection_dim),
        "clusters": int(args.clusters),
        "bins": int(args.bins),
        "knn_k": int(args.knn_k),
        "num_evicted": int(evicted.numel()),
        "num_retained": int(retained_mask.sum().item()),
        "cluster_occupancy_distortion": cluster_distortion,
        "bin_occupancy_distortion": bin_distortion,
        "cluster_survival_imbalance": cluster_imbalance,
        "key64_cluster_occupancy_distortion": key64_cluster_distortion,
        "key64_cluster_survival_imbalance": key64_cluster_imbalance,
        "max_cluster_suppression": max_cluster_suppression,
        "max_key64_cluster_suppression": max_key64_cluster_suppression,
        "max_bin_suppression": max_bin_suppression,
        "evicted_top_density_fraction": evicted_top_density_fraction,
        "effective_rank_before": eff_before,
        "effective_rank_after": eff_after,
        "effective_rank_drop": eff_before - eff_after,
        **cov,
        **key64_holes,
        **key64_evicted_coverage,
    }

    with (policy_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (policy_dir / "cluster_survival.json").open("w", encoding="utf-8") as f:
        json.dump(cluster_rows, f, indent=2)
    with (policy_dir / "bin_survival.json").open("w", encoding="utf-8") as f:
        json.dump(bin_rows, f, indent=2)
    with (policy_dir / "key64_cluster_survival.json").open("w", encoding="utf-8") as f:
        json.dump(key64_cluster_rows, f, indent=2)

    plot_scatter(coords2d, evicted, policy_dir / "scatter_evicted.png", f"{policy} evicted")
    plot_survival(cluster_rows, policy_dir / "cluster_survival.png", f"{policy} per-cluster survival")
    plot_survival(bin_rows, policy_dir / "bin_survival.png", f"{policy} per-bin survival")
    plot_survival(key64_cluster_rows, policy_dir / "key64_cluster_survival.png", f"{policy} 64D-key cluster survival")
    return result


def analyze_comparison(path: Path, args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze both policies from one comparison result file."""
    comp = torch.load(path, map_location="cpu")
    meta = dict(comp["meta"])
    source_path = resolve_source_snapshot(comp["source_snapshot"], path)
    source = torch.load(source_path, map_location="cpu")
    keys = source["old_key"].float()

    pca = pca_project(keys, args.projection_dim)
    coords = pca["coords"]
    coords2d = coords[:, :2] if coords.shape[1] >= 2 else torch.cat([coords, torch.zeros_like(coords)], dim=1)
    cluster_labels = kmeans(coords, args.clusters).cpu().numpy()
    bin_labels = bin_ids(coords2d, args.bins)
    key64_cluster_labels = kmeans(F.normalize(keys.float(), dim=-1), args.key_clusters).cpu().numpy()
    density = knn_density(coords, args.knn_k)
    top_density = density >= torch.quantile(density, args.top_density_quantile)

    stem = f"step{meta['step_idx']:06d}_layer{meta['layer_id']:02d}_head{meta['head_id']:02d}_batch{meta['batch_id']:02d}"
    head_dir = out_dir / "per_head" / stem
    head_dir.mkdir(parents=True, exist_ok=True)

    policy_evictions = {
        policy: comp[policy]["evict_indices"].long()
        for policy in comparison_policies(comp)
    }
    plot_scatter(coords2d, None, head_dir / "scatter_before.png", "Before eviction")
    plot_policy_scatter(coords2d, policy_evictions, head_dir / "scatter_mean_vs_svd.png")
    plot_spectrum(pca["singular_values"], pca["explained_variance_ratio"], head_dir / "spectrum.png")

    policy_results = [
        analyze_policy(
            policy=policy,
            evicted=policy_evictions[policy],
            keys=keys,
            coords=coords,
            coords2d=coords2d,
            cluster_labels=cluster_labels,
            bin_labels=bin_labels,
            key64_cluster_labels=key64_cluster_labels,
            top_density=top_density,
            meta=meta,
            compare_path=path,
            head_dir=head_dir,
            args=args,
        )
        for policy in policy_evictions.keys()
    ]

    mean_set = set(policy_evictions["mean"].tolist())
    svd_set = set(policy_evictions["svd_leverage"].tolist())
    eviction_count = max(len(mean_set), 1)
    overlap_ratio = float(len(mean_set & svd_set) / eviction_count)
    by_policy = {row["policy"]: row for row in policy_results}
    delta_keys = [
        "cluster_occupancy_distortion",
        "bin_occupancy_distortion",
        "cluster_survival_imbalance",
        "max_cluster_suppression",
        "max_bin_suppression",
        "key64_cluster_occupancy_distortion",
        "key64_cluster_survival_imbalance",
        "key64_cluster_empty_region_fraction",
        "key64_cluster_empty_region_mass",
        "key64_cluster_low_survival_nonempty_region_fraction",
        "key64_cluster_low_survival_nonempty_region_mass",
        "key64_cluster_p05_region_survival",
        "key64_evicted_nearest_retained_cosine_p05",
        "key64_uncovered_evicted_fraction",
        "evicted_top_density_fraction",
        "effective_rank_drop",
        "nearest_retained_cosine",
        "nearest_retained_projected_distance",
        "explained_variance_retained",
    ]
    delta = {
        **meta,
        "comparison_file": str(path),
        "source_snapshot": str(source_path),
        "eviction_count": int(comp["summary"]["eviction_count"]),
        "evicted_overlap_ratio": overlap_ratio,
    }
    for key in delta_keys:
        delta[f"{key}_svd_minus_mean"] = float(by_policy["svd_leverage"][key] - by_policy["mean"][key])
        if "layer_svd_leverage" in by_policy:
            delta[f"{key}_layer_svd_minus_mean"] = float(
                by_policy["layer_svd_leverage"][key] - by_policy["mean"][key]
            )
            delta[f"{key}_layer_svd_minus_head_svd"] = float(
                by_policy["layer_svd_leverage"][key] - by_policy["svd_leverage"][key]
            )

    with (head_dir / "comparison_delta.json").open("w", encoding="utf-8") as f:
        json.dump(delta, f, indent=2)

    print(
        f"layer={meta['layer_id']:02d} head={meta['head_id']:02d} "
        f"tokens={keys.shape[0]} evict={comp['summary']['eviction_count']} overlap={overlap_ratio:.4f}"
    )
    return policy_results, delta


def summarize_numeric(rows: list[dict[str, Any]], group_key: str | None = None) -> dict[str, Any]:
    """Summarize numeric columns globally or by a grouping key."""
    if not rows:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    if group_key is not None:
        groups = {}
        for row in rows:
            groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)

    summary: dict[str, Any] = {}
    for group, group_rows in groups.items():
        fieldnames = sorted({key for row in group_rows for key in row.keys()})
        group_summary: dict[str, Any] = {"n": len(group_rows)}
        for key in fieldnames:
            values = [
                row[key]
                for row in group_rows
                if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])
            ]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                group_summary[key] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "n": int(arr.size),
                }
        summary[group] = group_summary
    return summary


def write_outputs(policy_results: list[dict[str, Any]], deltas: list[dict[str, Any]], out_dir: Path) -> None:
    """Write CSV/JSON summaries and aggregate figures."""
    if policy_results:
        fieldnames = sorted({key for row in policy_results for key in row.keys()})
        with (out_dir / "per_head_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(policy_results)

    if deltas:
        fieldnames = sorted({key for row in deltas for key in row.keys()})
        with (out_dir / "comparison_deltas.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deltas)
        with (out_dir / "comparison_deltas.json").open("w", encoding="utf-8") as f:
            json.dump(deltas, f, indent=2)

    summary = {
        "num_policy_rows": len(policy_results),
        "num_comparisons": len(deltas),
        "by_policy": summarize_numeric(policy_results, group_key="policy"),
        "deltas": summarize_numeric(deltas).get("all", {}),
    }
    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_policy_boxplot(
        policy_results,
        "cluster_occupancy_distortion",
        out_dir / "occupancy_distortion_boxplot.png",
        "Cluster occupancy distortion",
    )
    plot_policy_boxplot(
        policy_results,
        "nearest_retained_projected_distance",
        out_dir / "coverage_distance_boxplot.png",
        "Coverage distance",
    )
    plot_policy_boxplot(
        policy_results,
        "cluster_survival_imbalance",
        out_dir / "cluster_imbalance_boxplot.png",
        "Cluster survival imbalance",
    )
    plot_policy_boxplot(
        policy_results,
        "evicted_top_density_fraction",
        out_dir / "top_density_suppression_boxplot.png",
        "Top-density eviction fraction",
    )
    plot_policy_boxplot(
        policy_results,
        "key64_cluster_empty_region_mass",
        out_dir / "key64_empty_region_mass_boxplot.png",
        "64D-key empty local-region mass",
    )
    plot_policy_boxplot(
        policy_results,
        "key64_cluster_low_survival_nonempty_region_mass",
        out_dir / "key64_low_survival_nonempty_mass_boxplot.png",
        "64D-key thinned local-region mass",
    )
    plot_policy_boxplot(
        policy_results,
        "key64_evicted_nearest_retained_cosine_p05",
        out_dir / "key64_evicted_nearest_cosine_p05_boxplot.png",
        "64D-key evicted nearest-retained cosine p05",
    )
    plot_delta_boxplot(
        deltas,
        "cluster_occupancy_distortion_svd_minus_mean",
        out_dir / "occupancy_distortion_delta_boxplot.png",
        "SVD minus mean cluster occupancy distortion",
    )
    plot_delta_boxplot(
        deltas,
        "nearest_retained_projected_distance_svd_minus_mean",
        out_dir / "coverage_distance_delta_boxplot.png",
        "SVD minus mean coverage distance",
    )
    plot_delta_boxplot(
        deltas,
        "key64_cluster_empty_region_mass_svd_minus_mean",
        out_dir / "key64_empty_region_mass_delta_boxplot.png",
        "SVD minus mean 64D-key empty-region mass",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--comparison_dir", required=True, help="Directory containing *_mean_vs_svd.pt files")
    parser.add_argument("--output_dir", required=True, help="Directory for mean-vs-SVD metrics and figures")
    parser.add_argument("--projection_dim", type=int, default=16, help="PCA dimension for metrics; -1 uses full rank")
    parser.add_argument("--clusters", type=int, default=12, help="Number of projected-space k-means clusters")
    parser.add_argument("--bins", type=int, default=8, help="Number of bins per PC axis")
    parser.add_argument("--knn_k", type=int, default=16, help="k for local kNN density")
    parser.add_argument("--top_density_quantile", type=float, default=0.75, help="Quantile for top-density tokens")
    parser.add_argument("--key_clusters", type=int, default=64, help="Number of normalized 64D-key clusters for local hole metrics")
    parser.add_argument(
        "--key_min_region_size",
        type=int,
        default=20,
        help="Minimum pre-eviction token count for a 64D-key cluster to count as a local region",
    )
    parser.add_argument(
        "--key_low_survival_threshold",
        type=float,
        default=0.1,
        help="Survival ratio below which a non-empty 64D-key cluster counts as thinned",
    )
    parser.add_argument(
        "--key_uncovered_cosine_threshold",
        type=float,
        default=0.99,
        help="Evicted token is uncovered when nearest retained original-key cosine is below this threshold",
    )
    parser.add_argument("--max_comparisons", type=int, default=None, help="Optional cap for quick checks")
    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(comparison_dir.glob("*_mean_vs_svd.pt"))
    if args.max_comparisons is not None:
        paths = paths[: args.max_comparisons]
    if not paths:
        raise SystemExit(f"No *_mean_vs_svd.pt files found under {comparison_dir}")

    policy_results: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for path in paths:
        rows, delta = analyze_comparison(path, args, out_dir)
        policy_results.extend(rows)
        deltas.append(delta)

    write_outputs(policy_results, deltas, out_dir)
    print(f"Analyzed {len(paths)} mean-vs-SVD comparison files. Results written to {out_dir}")


if __name__ == "__main__":
    main()
