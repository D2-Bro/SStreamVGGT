#!/usr/bin/env python3
"""Checks the statistics of tools/attention_peak_analysis.py against a numpy brute force (CPU)."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_peak_analysis as ap  # noqa: E402


def main():
    g = torch.Generator().manual_seed(0)
    P, T, H, D = 21, 3, 4, 8  # 5 special + 16 patch tokens per frame, step 3 -> 63 keys
    ap.State.tokens_per_frame = P
    ap.State.layer = 7
    q = torch.randn(1, H, P, D, generator=g)
    k = torch.randn(1, H, P * T, D, generator=g) * 2
    v = torch.randn(1, H, P * T, D, generator=g)
    ap.State.verify = True
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    ap._record(q, k, v, out)
    assert ap.State.max_out_diff < 1e-5 and ap.State.max_rowsum_err < 1e-5

    qn, kn = q.double().numpy()[0], k.double().numpy()[0]
    logits = np.einsum("hqd,hkd->hqk", qn, kn) / math.sqrt(D)
    a = np.exp(logits - logits.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    N = P * T
    special = (np.arange(N) % P) < 5

    def check(row, mat):
        s = -np.sort(-mat, axis=-1)
        assert np.isclose(row["rowmax_mean"], s[:, 0].mean(), atol=1e-6)
        assert np.isclose(row["top5_mean"], s[:, :5].mean(), atol=1e-6)
        assert np.isclose(row["top10_mass"], s[:, :10].sum(-1).mean(), atol=1e-6)
        assert np.isclose(row["top1pct_mean"], s[:, :math.ceil(0.01 * N)].mean(), atol=1e-6)
        ent = -(mat * np.log(mat)).sum(-1) / math.log(N)
        assert np.isclose(row["ent_norm_mean"], ent.mean(), atol=1e-6)
        assert np.isclose(row["cross_frame_mass"], mat[:, :N - P].sum(-1).mean(), atol=1e-6)
        assert np.isclose(row["special_key_mass"], mat[:, special].sum(-1).mean(), atol=1e-6)

    rows = {(r["head"], r["query_type"]): r for r in ap.State.rows}
    check(rows[(2, "all")], a[2])
    check(rows[(1, "patch")], a[1, 5:])
    check(rows[(-1, "all")], a.mean(0))
    check(rows[(-1, "special")], a.mean(0)[:5])

    am = a.mean(0)[5:][:, ~special]  # AVGGT: head mean, patch x patch
    top = ap.State.top50[0]
    ref = -np.sort(-am.flatten())[:50]
    assert np.isclose(top["top50_mean"], ref.mean(), atol=1e-6) and np.isclose(top["top50_min"], ref.min(), atol=1e-7)
    # categories: patch index of key == query patch, in the current frame (self) or another frame (same position)
    flat = np.argsort(-am.flatten(), kind="stable")[:50]
    qi, kcol = flat // am.shape[1], flat % am.shape[1]
    kframe, kpatch = kcol // 16, kcol % 16
    assert np.isclose(top["frac_self"], ((kframe == T - 1) & (kpatch == qi)).mean())
    assert np.isclose(top["frac_same_position"], ((kframe != T - 1) & (kpatch == qi)).mean())
    assert np.isclose(top["frac_cross_frame"], (kframe != T - 1).mean())
    print("PASS attention peak statistics")


if __name__ == "__main__":
    main()
