#!/usr/bin/env python3
"""Checks for the strict-budget "redistribute" allocation (eviction._redistribute_anchor_deficit).

Run: python tools/test_strict_redistribute.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamvggt.layers.eviction import (  # noqa: E402
    _allocate_layer_budgets_from_scores,
    _redistribute_anchor_deficit,
)

A, L = 1041, 24


def soft(weights, caps, total):
    budgets, details = _allocate_layer_budgets_from_scores(weights, caps, total, alpha=1.0, return_debug=True)
    return budgets, details["weights"]


def test_total_and_floor() -> None:
    rng = random.Random(0)
    for total in (24984, 30000, 60000, 200000):
        for _ in range(300):
            weights = {l: rng.random() ** 4 * 10 + 1e-3 for l in range(L)}
            caps = {l: rng.choice([A, 2 * A, 3000, 50000, 200000]) for l in range(L)}
            budgets, w = soft(weights, caps, total)
            out = _redistribute_anchor_deficit(budgets, w, caps, A, total)
            assert sum(out.values()) <= total, (total, sum(out.values()))
            assert all(out[l] >= min(A, caps[l]) for l in range(L))
            assert all(out[l] <= caps[l] for l in range(L))


def test_free_layers_keep_kaba_ratios() -> None:
    weights = {l: (10.0 if l in (11, 12, 13, 14, 15) else 0.3) for l in range(L)}
    caps = {l: 10 ** 6 for l in range(L)}
    budgets, w = soft(weights, caps, 30000)
    assert sum(1 for l in range(L) if budgets[l] < A) == 19
    out = _redistribute_anchor_deficit(budgets, w, caps, A, 30000)
    assert all(out[l] == A for l in range(L) if l not in (11, 12, 13, 14, 15))
    free = [out[l] for l in (11, 12, 13, 14, 15)]
    assert sum(out.values()) == 30000 and max(free) - min(free) <= 1, free  # equal weights -> equal shares


def test_no_deficit_equals_soft() -> None:
    rng = random.Random(1)
    weights = {l: 1.0 + rng.random() for l in range(L)}
    caps = {l: 10 ** 6 for l in range(L)}
    budgets, w = soft(weights, caps, 200000)
    assert min(budgets.values()) >= A
    assert _redistribute_anchor_deficit(budgets, w, caps, A, 200000) == budgets


def test_cascade() -> None:
    # Discounting pushes a borderline layer below the anchor: it must be fixed too.
    weights = {l: 0.1 for l in range(L)}
    weights[0], weights[1] = 30.0, 1.25
    caps = {l: 10 ** 6 for l in range(L)}
    budgets, w = soft(weights, caps, 40000)
    out = _redistribute_anchor_deficit(budgets, w, caps, A, 40000)
    assert sum(out.values()) <= 40000 and all(v >= A for v in out.values())


def main() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
