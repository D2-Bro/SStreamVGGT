#!/usr/bin/env python3
"""Synthetic tests for analyze_key_layer_stats.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from analyze_key_layer_stats import (
    analyze_layer_caches,
    collect_no_eviction_key_caches,
    exact_qr_leverage_pr,
    resolve_sequence_images,
    uncentered_effective_dim,
    write_results,
)
from streamvggt.layers.confidence_state import pack_kv_cache, unpack_kv_cache


def _assert_close(actual: float, expected: float, atol: float = 1e-5) -> None:
    if abs(actual - expected) > atol:
        raise AssertionError(f"expected {expected}, got {actual}")


def check_effective_dim_known_cases() -> None:
    rank_one = torch.tensor([[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0]])
    _assert_close(uncentered_effective_dim(rank_one), 1.0)
    equal_orthogonal = 3.0 * torch.eye(4, 7)
    _assert_close(uncentered_effective_dim(equal_orthogonal), 4.0)


def check_scale_properties() -> None:
    torch.manual_seed(7)
    keys = torch.randn(9, 5)
    scaled = 3.5 * keys
    _assert_close(uncentered_effective_dim(keys), uncentered_effective_dim(scaled), atol=1e-4)
    _assert_close(
        torch.linalg.vector_norm(scaled, dim=1).mean().item(),
        3.5 * torch.linalg.vector_norm(keys, dim=1).mean().item(),
    )
    leverage_pr, rank = exact_qr_leverage_pr(keys)
    scaled_leverage_pr, scaled_rank = exact_qr_leverage_pr(scaled)
    _assert_close(leverage_pr, scaled_leverage_pr, atol=1e-4)
    if rank != scaled_rank:
        raise AssertionError(f"leverage rank changed under scaling: {rank} vs {scaled_rank}")


def check_leverage_pr_known_cases() -> None:
    uniform_rank_one = torch.ones(8, 1)
    leverage_pr, rank = exact_qr_leverage_pr(uniform_rank_one)
    _assert_close(leverage_pr, 8.0)
    if rank != 1:
        raise AssertionError(f"expected leverage rank 1, got {rank}")

    concentrated = torch.zeros(8, 3)
    concentrated[:3] = torch.eye(3)
    leverage_pr, rank = exact_qr_leverage_pr(concentrated)
    _assert_close(leverage_pr, 3.0)
    if rank != 3:
        raise AssertionError(f"expected leverage rank 3, got {rank}")


def check_sequence_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sequence_dir = Path(tmp)
        for name in ("000.png", "001.jpg", "002.jpeg", "ignored.txt"):
            (sequence_dir / name).touch()
        selected = resolve_sequence_images(sequence_dir, num_frames=2, frame_stride=2)
        if [path.name for path in selected] != ["000.png", "002.jpeg"]:
            raise AssertionError(f"unexpected selected frames: {selected}")


def check_no_eviction_cache_collection() -> None:
    class FakeAggregator:
        depth = 2

        def reset_stream_state(self) -> None:
            self.was_reset = True

        def __call__(self, frame, **kwargs):
            if kwargs["cache_evict_current_frame"] is not False:
                raise AssertionError("cache eviction was not disabled")
            if kwargs["cache_write_current_frame"] is not True:
                raise AssertionError("current-frame cache writing was not enabled")
            frame_idx = kwargs["past_frame_idx"]
            updated = []
            for layer, previous in enumerate(kwargs["past_key_values"]):
                current = torch.full((1, 2, 3, 4), float(frame_idx + layer + 1))
                if previous is not None:
                    old_k, old_v, _, _ = unpack_kv_cache(previous)
                    current = torch.cat([old_k, current], dim=2)
                    value = torch.cat([old_v, current[:, :, -3:]], dim=2)
                else:
                    value = current.clone()
                updated.append(pack_kv_cache(current, value))
            return [], 0, updated

    aggregator = FakeAggregator()
    model = SimpleNamespace(aggregator=aggregator)
    images = torch.randn(3, 3, 8, 8)
    caches = collect_no_eviction_key_caches(model, images, torch.device("cpu"))
    if not getattr(aggregator, "was_reset", False):
        raise AssertionError("stream state was not reset")
    if sorted(caches) != [0, 1] or any(tuple(keys.shape) != (1, 2, 9, 4) for keys in caches.values()):
        raise AssertionError(f"unexpected accumulated cache shapes: {[v.shape for v in caches.values()]}")


def check_analysis_and_outputs() -> None:
    torch.manual_seed(11)
    layer_caches = {
        layer: torch.randn(1, 2, 7, 3) * (layer + 1) for layer in range(4)
    }
    rows, mean_norms, dimensions, leverage_prs = analyze_layer_caches(layer_caches, num_frames=5)
    if (
        len(rows) != 4
        or sorted(mean_norms) != list(range(4))
        or sorted(dimensions) != list(range(4))
        or sorted(leverage_prs) != list(range(4))
    ):
        raise AssertionError("layer cache analysis did not produce one result per layer")

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        written_rows = write_results(layer_caches, num_frames=5, output_dir=output_dir)
        if len(written_rows) != 4:
            raise AssertionError(f"expected 4 rows, got {len(written_rows)}")
        for filename in (
            "layer_key_stats.csv",
            "mean_key_norm_by_layer.png",
            "effective_dim_by_layer.png",
            "leverage_pr_by_layer.png",
        ):
            path = output_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"missing or empty output: {path}")
        with (output_dir / "layer_key_stats.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        if len(csv_rows) != 4:
            raise AssertionError(f"expected 4 CSV rows, got {len(csv_rows)}")


def check_mismatched_layers_rejected() -> None:
    layer_caches = {0: torch.randn(1, 2, 7, 3), 1: torch.randn(1, 2, 6, 3)}
    try:
        analyze_layer_caches(layer_caches, num_frames=2)
    except ValueError as exc:
        if "not comparable" not in str(exc):
            raise
    else:
        raise AssertionError("mismatched layer caches were accepted")


def main() -> None:
    check_effective_dim_known_cases()
    check_scale_properties()
    check_leverage_pr_known_cases()
    check_sequence_selection()
    check_no_eviction_cache_collection()
    check_analysis_and_outputs()
    check_mismatched_layers_rejected()
    print("analyze_key_layer_stats synthetic tests passed")


if __name__ == "__main__":
    main()
