#!/usr/bin/env python3
"""Smoke tests for eviction nearest-retained analysis helpers."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.utils.cache_analysis import (  # noqa: E402
    EvictionNNAnalysisConfig,
    compute_knn_distance_ratio,
    compute_nearest_retained,
    dump_eviction_nn_analysis,
)


def check_known_distances() -> None:
    retained = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    evicted = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    retained_idx = torch.tensor([10, 11])
    nn = compute_nearest_retained(retained, evicted, retained_idx, chunk_size=1)
    dist = 1.0 - nn["nearest_cosine_similarity"]
    if dist[0].abs().item() > 1e-6:
        raise AssertionError(f"identical token distance should be near 0, got {dist[0].item()}")
    if dist[1].item() < 0.99:
        raise AssertionError(f"orthogonal/opposite token distance should be large, got {dist[1].item()}")


def check_chunking_matches() -> None:
    torch.manual_seed(5)
    retained = torch.randn(17, 8)
    evicted = torch.randn(23, 8)
    retained_idx = torch.arange(100, 117)
    one = compute_nearest_retained(retained, evicted, retained_idx, chunk_size=999)
    chunked = compute_nearest_retained(retained, evicted, retained_idx, chunk_size=4)
    if not torch.allclose(one["nearest_cosine_similarity"], chunked["nearest_cosine_similarity"], atol=1e-6):
        raise AssertionError("chunked cosine similarities differ")
    if not torch.equal(one["nearest_retained_cache_idx"], chunked["nearest_retained_cache_idx"]):
        raise AssertionError("chunked nearest indices differ")


def check_knn_ratio_known_cases() -> None:
    evicted = torch.tensor([[1.0, 0.0]])
    retained = torch.tensor([[0.0, 1.0]])
    pre = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ratio = compute_knn_distance_ratio(
        retained,
        pre,
        evicted,
        torch.tensor([0]),
        k=1,
        chunk_size=1,
    )
    if abs(ratio["r_knn"] - 1.0) > 1e-6:
        raise AssertionError(f"expected R_kNN near 1, got {ratio['r_knn']}")

    retained_far = torch.tensor([[0.0, 1.0]])
    pre_close = torch.tensor([[1.0, 0.0], [0.99, 0.1]])
    ratio_far = compute_knn_distance_ratio(
        retained_far,
        pre_close,
        evicted,
        torch.tensor([0]),
        k=1,
        chunk_size=1,
    )
    if ratio_far["r_knn"] is None or ratio_far["r_knn"] <= 10.0:
        raise AssertionError(f"expected far retained neighbor to give large R_kNN, got {ratio_far['r_knn']}")


def check_knn_ratio_chunking_matches() -> None:
    torch.manual_seed(11)
    retained = torch.randn(19, 7)
    pre = torch.randn(31, 7)
    rows = torch.tensor([2, 5, 8, 13, 21, 25])
    evicted = pre.index_select(0, rows) + 0.01 * torch.randn(rows.numel(), 7)
    one = compute_knn_distance_ratio(retained, pre, evicted, rows, k=5, chunk_size=99)
    chunked = compute_knn_distance_ratio(retained, pre, evicted, rows, k=5, chunk_size=2)
    for key in ("r_knn", "retained_knn_distance_sum", "pre_knn_distance_sum"):
        if abs(float(one[key]) - float(chunked[key])) > 1e-5:
            raise AssertionError(f"chunked {key} differs: {one[key]} vs {chunked[key]}")


def check_knn_ratio_large_k() -> None:
    retained = torch.tensor([[1.0, 0.0]])
    pre = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    evicted = torch.tensor([[0.0, 1.0]])
    ratio = compute_knn_distance_ratio(retained, pre, evicted, torch.tensor([0]), k=5, chunk_size=1)
    if ratio["retained_knn_effective_k"] != 1 or ratio["pre_knn_effective_k"] != 1:
        raise AssertionError("large K should use available neighbors")
    if ratio["r_knn"] is None:
        raise AssertionError("large K with available neighbors should still compute R_kNN")


def check_dump_writes_knn_fields() -> None:
    tmp = tempfile.mkdtemp(prefix="eviction_nn_knn_dump_")
    try:
        config = EvictionNNAnalysisConfig(tmp, save_topk_pairs=0, knn_k=2)
        k = torch.tensor([[[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]]])
        scores = torch.zeros(1, 1, 4)
        dump_eviction_nn_analysis(
            config,
            k_before=k,
            v_before=None,
            kept_candidate_indices=torch.tensor([[[0, 1]]]),
            policy_scores=scores,
            leverage_basis=None,
            metadata=None,
            layer_id=0,
            step_idx=2,
            cache_budget=2,
            num_anchor_tokens=0,
            eviction_policy="mean",
            leverage_granularity="head",
            leverage_feature="key",
        )
        summary_path = os.path.join(tmp, "summary.jsonl")
        line = open(summary_path, "r", encoding="utf-8").readline()
        if '"r_knn"' not in line or '"knn_k":2' not in line:
            raise AssertionError("R_kNN fields were not written to summary.jsonl")
    finally:
        shutil.rmtree(tmp)


def check_empty_cases() -> None:
    tmp = tempfile.mkdtemp(prefix="eviction_nn_test_")
    try:
        config = EvictionNNAnalysisConfig(tmp, save_topk_pairs=0)
        k = torch.randn(1, 1, 4, 2)
        scores = torch.zeros(1, 1, 4)
        dump_eviction_nn_analysis(
            config,
            k_before=k,
            v_before=None,
            kept_candidate_indices=torch.empty(1, 1, 0, dtype=torch.long),
            policy_scores=scores,
            leverage_basis=None,
            metadata=None,
            layer_id=0,
            step_idx=0,
            cache_budget=0,
            num_anchor_tokens=0,
            eviction_policy="mean",
            leverage_granularity="head",
            leverage_feature="key",
        )
        dump_eviction_nn_analysis(
            config,
            k_before=k,
            v_before=None,
            kept_candidate_indices=torch.arange(4).view(1, 1, 4),
            policy_scores=scores,
            leverage_basis=None,
            metadata=None,
            layer_id=0,
            step_idx=1,
            cache_budget=4,
            num_anchor_tokens=0,
            eviction_policy="mean",
            leverage_granularity="head",
            leverage_feature="key",
        )
        summary = os.path.join(tmp, "summary.jsonl")
        if not os.path.exists(summary):
            raise AssertionError("summary.jsonl was not written")
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    check_known_distances()
    check_chunking_matches()
    check_knn_ratio_known_cases()
    check_knn_ratio_chunking_matches()
    check_knn_ratio_large_k()
    check_dump_writes_knn_fields()
    check_empty_cases()
    print("eviction NN analysis smoke tests passed")


if __name__ == "__main__":
    main()
