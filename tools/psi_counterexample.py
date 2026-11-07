"""Reproduce the leverage participation-ratio (psi) counterexample and place real caches next to it.

psi = (sum l)^2 / sum l^2 is n both for n identical keys (rank 1, l_i = 1/n) and for n keys on n orthogonal
directions (l_i = 1). Effective rank and token cosine separate the two at once. With --redundancy_csv, the same
statistics of real IDC caches (from launch.py --redundancy_log) are printed next to them.

Usage:
  python tools/psi_counterexample.py [--n 256 --d 1024] [--redundancy_csv <output_dir>/<dataset>/redundancy/*.csv]
"""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from streamvggt.utils.redundancy_log import set_statistics  # noqa: E402

COLUMNS = ["n_tokens", "psi", "psi_over_n", "rank", "erank", "erank_over_n", "mean_cos", "frac_cos_gt_0.9",
           "nn_cos_mean"]


def synthetic_sets(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    unit = lambda x: torch.nn.functional.normalize(x, dim=1)  # noqa: E731
    same = unit(torch.randn(1, d, generator=g, dtype=torch.float64)).repeat(n, 1)
    orthogonal = torch.linalg.qr(torch.randn(d, n, generator=g, dtype=torch.float64))[0].T
    near_same = unit(same + 0.3 / d ** 0.5 * torch.randn(n, d, generator=g, dtype=torch.float64))
    return {
        "identical keys (rank 1)": same,
        "orthogonal keys": orthogonal,
        "near-identical keys (noise norm 0.3)": near_same,
    }


def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--redundancy_csv", nargs="*", default=[], help="CSV files or globs from --redundancy_log")
    args = parser.parse_args()
    if args.n > args.d:
        parser.error("--n must be <= --d for the orthogonal set")

    rows = []
    for name, x in synthetic_sets(args.n, args.d).items():
        rows.append((name, set_statistics(x, (0.5, 0.7, 0.9))))

    paths = sorted(p for pattern in args.redundancy_csv for p in glob.glob(pattern))
    if paths:
        import pandas as pd
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        for set_name in ("candidates", "retained"):
            sub = df[df["set"] == set_name]
            if len(sub):
                rows.append((f"real cache, {set_name} (median of {len(sub)})",
                             {c: float(sub[c].median()) for c in COLUMNS if c in sub}))

    width = max(len(name) for name, _ in rows)
    print(" | ".join(["set".ljust(width)] + COLUMNS))
    for name, stats in rows:
        print(" | ".join([name.ljust(width)] + [fmt(stats.get(c, float("nan"))) for c in COLUMNS]))


if __name__ == "__main__":
    main()
