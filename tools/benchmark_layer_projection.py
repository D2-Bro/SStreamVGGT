#!/usr/bin/env python3
"""Benchmark layer-wise leverage projection variants.

This isolates the hot path used by layer svd_leverage:
    torch.einsum("bhnd,hds->bhns", mat_k, omega_key)

Run the same command in two environments to see whether the slowdown is from
einsum lowering, tensor layout, TF32 policy, or the large [B, H, N, S]
intermediate that is later summed into [B, N, S].
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cuda(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int) -> list[float]:
    latencies = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
        del out
    return latencies


def _time_cpu(fn: Callable[[], torch.Tensor], repeats: int) -> list[float]:
    import time

    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
        del out
    return latencies


def _bench(
    name: str,
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        out = fn()
        _sync(device)
        del out
    if device.type == "cuda":
        latencies = _time_cuda(fn, device, repeats)
    else:
        latencies = _time_cpu(fn, repeats)
    mean = statistics.fmean(latencies)
    median = statistics.median(latencies)
    print(f"{name:28s} mean={mean:9.3f}ms median={median:9.3f}ms")
    return mean, median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tf32", choices=("default", "on", "off"), default="default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.tf32 != "default":
        enabled = args.tf32 == "on"
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enabled else "highest")

    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    mat_k = torch.randn(
        args.batch,
        args.heads,
        args.tokens,
        args.head_dim,
        dtype=dtype,
        generator=generator,
    ).to(device)
    omega_key = torch.randn(
        args.heads,
        args.head_dim,
        args.sketch_dim,
        dtype=torch.float32,
        generator=generator,
    ).to(device)
    if dtype != torch.float32:
        omega_key = omega_key.to(dtype=dtype)

    # Mirror eviction.py: projection runs under autocast(False), so fp32 input
    # is the most representative mode for the current hot path.
    mat_k = mat_k.to(dtype=torch.float32)
    omega_key = omega_key.to(dtype=torch.float32)
    omega_flat = omega_key.reshape(args.heads * args.head_dim, args.sketch_dim).contiguous()

    print("environment")
    print(f"  torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"  device={torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(f"  allow_tf32_matmul={torch.backends.cuda.matmul.allow_tf32}")
    print(f"  allow_tf32_cudnn={torch.backends.cudnn.allow_tf32}")
    if hasattr(torch, "get_float32_matmul_precision"):
        print(f"  matmul_precision={torch.get_float32_matmul_precision()}")
    print(
        "shape "
        f"B={args.batch} H={args.heads} N={args.tokens} D={args.head_dim} S={args.sketch_dim}"
    )
    print(f"mat_k stride={mat_k.stride()} omega stride={omega_key.stride()}")

    def original_einsum_head() -> torch.Tensor:
        return torch.einsum("bhnd,hds->bhns", mat_k, omega_key)

    def original_einsum_layer() -> torch.Tensor:
        return torch.einsum("bhnd,hds->bhns", mat_k, omega_key).sum(dim=1)

    def broadcast_matmul_head() -> torch.Tensor:
        return torch.matmul(mat_k, omega_key.unsqueeze(0))

    def bmm_head() -> torch.Tensor:
        x = mat_k.reshape(args.batch * args.heads, args.tokens, args.head_dim)
        w = omega_key.unsqueeze(0).expand(args.batch, -1, -1, -1)
        w = w.reshape(args.batch * args.heads, args.head_dim, args.sketch_dim)
        return torch.bmm(x, w).reshape(args.batch, args.heads, args.tokens, args.sketch_dim)

    def flat_layer_matmul() -> torch.Tensor:
        x = mat_k.permute(0, 2, 1, 3).reshape(args.batch, args.tokens, args.heads * args.head_dim)
        return torch.matmul(x, omega_flat)

    def flat_layer_matmul_contig() -> torch.Tensor:
        x = mat_k.permute(0, 2, 1, 3).contiguous().view(
            args.batch, args.tokens, args.heads * args.head_dim
        )
        return torch.matmul(x, omega_flat)

    print("benchmark")
    rows = []
    rows.append(("einsum_head", *_bench("einsum_head [B,H,N,S]", original_einsum_head, device, args.warmup, args.repeats)))
    rows.append(("einsum_layer", *_bench("einsum_head.sum layer", original_einsum_layer, device, args.warmup, args.repeats)))
    rows.append(("broadcast_matmul", *_bench("broadcast matmul head", broadcast_matmul_head, device, args.warmup, args.repeats)))
    rows.append(("bmm_head", *_bench("reshape bmm head", bmm_head, device, args.warmup, args.repeats)))
    rows.append(("flat_layer", *_bench("flat layer matmul", flat_layer_matmul, device, args.warmup, args.repeats)))
    rows.append(("flat_layer_contig", *_bench("flat layer matmul contig", flat_layer_matmul_contig, device, args.warmup, args.repeats)))

    baseline = rows[0][1]
    print("summary")
    for name, mean, median in rows:
        speedup = baseline / mean if mean > 0 else float("nan")
        print(f"{name:20s} mean={mean:9.3f}ms median={median:9.3f}ms speedup_vs_einsum={speedup:6.2f}x")


if __name__ == "__main__":
    main()
