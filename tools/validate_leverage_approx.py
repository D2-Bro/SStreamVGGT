#!/usr/bin/env python3
"""Compare exact, right-sketched, and Drineas SRHT leverage scores."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.eviction import EvictionManager


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a_rank = torch.argsort(torch.argsort(a.float()))
    b_rank = torch.argsort(torch.argsort(b.float()))
    a_rank = a_rank.float() - a_rank.float().mean()
    b_rank = b_rank.float() - b_rank.float().mean()
    denom = a_rank.norm() * b_rank.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a_rank * b_rank).sum() / denom)


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(int(k), int(a.numel()), int(b.numel()))
    if k <= 0:
        return float("nan")
    top_a = set(torch.topk(a, k=k).indices.cpu().tolist())
    top_b = set(torch.topk(b, k=k).indices.cpu().tolist())
    return len(top_a & top_b) / float(k)


def _relative_error(exact: torch.Tensor, approx: torch.Tensor) -> tuple[float, float]:
    mask = exact.abs() > max(float(exact.abs().max()) * 1e-6, 1e-12)
    if not bool(mask.any().item()):
        return float("nan"), float("nan")
    rel = (approx[mask] - exact[mask]).abs() / exact[mask].abs().clamp_min(1e-12)
    return float(rel.max()), float(rel.mean())


def _score_matrix(mat: torch.Tensor, method: str, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float], float]:
    manager = EvictionManager(
        policy="svd_leverage",
        debug=args.profile,
        leverage_sketch_dim=args.right_sketch_dim,
        leverage_granularity="layer",
        leverage_approx_method=method,
        leverage_left_sketch_dim=args.left_sketch_dim,
        leverage_right_jl_dim=args.right_jl_dim,
        leverage_random_seed=args.seed,
    )
    start = time.perf_counter()
    scores = manager.compute_svd_leverage_scores(mat, args.right_sketch_dim)
    if mat.is_cuda:
        torch.cuda.synchronize(mat.device)
    elapsed = time.perf_counter() - start
    return scores.detach().cpu(), dict(manager._last_leverage_profile), elapsed


def _run_shape(num_tokens: int, feature_dim: int, args: argparse.Namespace) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    mat = torch.randn(num_tokens, feature_dim, generator=generator, dtype=torch.float32).to(args.device)

    exact, exact_profile, exact_time = _score_matrix(mat, "exact_qr", args)
    right, right_profile, right_time = _score_matrix(mat, "right_sketch", args)
    drineas, drineas_profile, drineas_time = _score_matrix(mat, "drineas_srht", args)
    rank_est = int(torch.linalg.matrix_rank(mat.float().cpu()).item())

    print(f"\nshape=[{num_tokens}, {feature_dim}] rank_est={rank_est}")
    for name, scores, profile, elapsed in (
        ("exact_qr", exact, exact_profile, exact_time),
        ("right_sketch", right, right_profile, right_time),
        ("drineas_srht", drineas, drineas_profile, drineas_time),
    ):
        max_rel, mean_rel = _relative_error(exact, scores)
        print(
            f"{name:14s} sum={float(scores.sum()):.4f} time={elapsed * 1000.0:.2f}ms "
            f"spearman={_spearman(exact, scores):.4f} "
            f"top{args.topk}_overlap={_topk_overlap(exact, scores, args.topk):.4f} "
            f"max_rel={max_rel:.4f} mean_rel={mean_rel:.4f} profile={profile}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--right_sketch_dim", type=int, default=16)
    parser.add_argument("--left_sketch_dim", type=int, default=2048)
    parser.add_argument("--right_jl_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--include_large", action="store_true", help="Also run [9000, 1024] profiling comparison")
    args = parser.parse_args()

    shapes = [(512, 64), (2048, 128)]
    if args.include_large:
        shapes.append((9000, 1024))
    for shape in shapes:
        _run_shape(*shape, args=args)


if __name__ == "__main__":
    main()
