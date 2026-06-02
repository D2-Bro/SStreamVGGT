#!/usr/bin/env python3
"""Benchmark exact and approximate leverage-score eviction kernels."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.eviction import EvictionManager


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    r1: int | None = None
    r2: int | None = None
    right_sketch_dim: int | None = None


def _dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(table)}")
    return table[name]


def _exact_svd_scores(mat: torch.Tensor) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        work = torch.nan_to_num(mat.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        u, _s, _vh = torch.linalg.svd(work, full_matrices=False)
        return torch.nan_to_num(u.square().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def _manager_scores(mat: torch.Tensor, case: Case, seed: int, profile: bool) -> torch.Tensor:
    manager = EvictionManager(
        policy="svd_leverage",
        debug=profile,
        leverage_sketch_dim=case.right_sketch_dim,
        leverage_granularity="layer",
        leverage_approx_method=case.method,
        leverage_left_sketch_dim=case.r1,
        leverage_right_jl_dim=case.r2,
        leverage_random_seed=seed,
    )
    scores = manager.compute_svd_leverage_scores(mat, case.right_sketch_dim)
    if profile and manager._last_leverage_profile:
        profile_items = " ".join(
            f"{key}={value * 1000.0:.3f}ms" if key != "fallback" else f"{key}={int(value)}"
            for key, value in manager._last_leverage_profile.items()
        )
        print(f"    profile[{case.name}] {profile_items}")
    return scores


def _time_cuda(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int) -> tuple[list[float], int]:
    latencies = []
    peak = 0
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        del out
    return latencies, peak


def _time_cpu(fn: Callable[[], torch.Tensor], repeats: int) -> tuple[list[float], int]:
    import time

    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
        del out
    return latencies, 0


def _benchmark_case(
    mat: torch.Tensor,
    name: str,
    fn: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
) -> tuple[float, float, int]:
    for _ in range(warmup):
        out = fn()
        if mat.is_cuda:
            torch.cuda.synchronize(mat.device)
        del out
    if mat.is_cuda:
        latencies, peak = _time_cuda(fn, mat.device, repeats)
    else:
        latencies, peak = _time_cpu(fn, repeats)
    mean_ms = statistics.fmean(latencies)
    median_ms = statistics.median(latencies)
    print(f"{name:24s} mean={mean_ms:9.3f}ms median={median_ms:9.3f}ms peak={peak / (1024 ** 2):8.1f}MiB")
    return mean_ms, median_ms, peak


def _selector_fn(
    k: torch.Tensor,
    cache_budget: int,
    selector: str,
    block_size: int,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        manager = EvictionManager(
            policy="svd_leverage",
            leverage_sketch_dim=4,
            leverage_granularity="layer",
            leverage_projection="head_mean",
            leverage_head_mean_dim=4,
            leverage_eviction_selector=selector,
            leverage_dpp_candidate_multiplier=2,
            leverage_dpp_greedy_block_size=block_size,
        )
        result = manager.select(k, cache_budget=cache_budget, num_anchor_tokens=0)
        return result.kept_candidate_indices
    return run


def _benchmark_eviction_selectors(args, device: torch.device, dtype: torch.dtype) -> None:
    if args.evict < 1 or args.evict >= args.n:
        raise ValueError(f"--evict must be in [1, n), got evict={args.evict}, n={args.n}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 17)
    k = torch.randn(1, args.heads, args.n, args.head_dim, generator=gen, dtype=dtype).to(device)
    cache_budget = args.n - args.evict
    cases = [
        ("selector_topk", "topk", 32),
        ("selector_fast_dpp_b1", "fast_dpp", 1),
        ("selector_fast_dpp_b32", "fast_dpp", 32),
    ]
    print("\nselector benchmark")
    print(
        f"cache=[1, {args.heads}, {args.n}, {args.head_dim}] evict={args.evict} "
        f"dtype={dtype} device={device}"
    )
    rows = []
    for name, selector, block_size in cases:
        mean_ms, median_ms, peak = _benchmark_case(
            k,
            name,
            _selector_fn(k, cache_budget, selector, block_size),
            args.warmup,
            args.repeats,
        )
        rows.append((name, mean_ms, median_ms, peak))
    baseline = rows[0][1]
    print("\nselector summary")
    for name, mean_ms, median_ms, peak in rows:
        speedup = baseline / mean_ms if mean_ms > 0 else float("nan")
        print(
            f"{name:24s} mean={mean_ms:9.3f}ms median={median_ms:9.3f}ms "
            f"peak={peak / (1024 ** 2):8.1f}MiB speedup_vs_topk={speedup:6.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n", type=int, default=9000)
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", help="Input dtype; eviction code internally scores in float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--current_method", choices=("right_sketch", "drineas_srht"), default="right_sketch")
    parser.add_argument("--current_right_sketch_dim", type=int, default=16)
    parser.add_argument("--profile", action="store_true", help="Print EvictionManager substage timings")
    parser.add_argument("--benchmark_eviction_selectors", action="store_true", help="Also benchmark topk vs fast_dpp selection latency")
    parser.add_argument("--evict", type=int, default=512, help="Number of tokens to evict in selector benchmark")
    parser.add_argument("--heads", type=int, default=16, help="Number of synthetic heads for selector benchmark")
    parser.add_argument("--head_dim", type=int, default=64, help="Synthetic head dimension for selector benchmark")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    mat = torch.randn(args.n, args.d, generator=gen, dtype=dtype).to(device)

    cases = [
        Case("exact_svd_full", "exact_svd"),
        Case("exact_qr_impl", "exact_qr", right_sketch_dim=0),
        Case(f"current_{args.current_method}", args.current_method, r1=2048, r2=64, right_sketch_dim=args.current_right_sketch_dim),
        Case("drineas_r1_384_r2_64", "drineas_srht", r1=384, r2=64, right_sketch_dim=0),
        Case("drineas_r1_512_r2_128", "drineas_srht", r1=512, r2=128, right_sketch_dim=0),
        Case("drineas_r1_768_r2_128", "drineas_srht", r1=768, r2=128, right_sketch_dim=0),
        Case("drineas_r1_768_r2_256", "drineas_srht", r1=768, r2=256, right_sketch_dim=0),
    ]

    print(f"shape=[{args.n}, {args.d}] dtype={dtype} device={device} warmup={args.warmup} repeats={args.repeats}")
    exact_mean = None
    rows = []
    for case in cases:
        if case.method == "exact_svd":
            fn = lambda mat=mat: _exact_svd_scores(mat)
        else:
            fn = lambda case=case, mat=mat: _manager_scores(mat, case, args.seed, args.profile)
        mean_ms, median_ms, peak = _benchmark_case(mat, case.name, fn, args.warmup, args.repeats)
        if case.name == "exact_svd_full":
            exact_mean = mean_ms
        speedup = (exact_mean / mean_ms) if exact_mean and mean_ms > 0 else float("nan")
        rows.append((case.name, mean_ms, median_ms, peak, speedup))

    print("\nsummary")
    for name, mean_ms, median_ms, peak, speedup in rows:
        verdict = "faster" if speedup > 1.0 else "not_faster"
        print(
            f"{name:24s} mean={mean_ms:9.3f}ms median={median_ms:9.3f}ms "
            f"peak={peak / (1024 ** 2):8.1f}MiB speedup_vs_exact_svd={speedup:6.2f}x {verdict}"
        )

    if args.benchmark_eviction_selectors:
        _benchmark_eviction_selectors(args, device, dtype)


if __name__ == "__main__":
    main()
