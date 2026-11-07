"""Compare cache redundancy across eviction conditions (launch.py --redundancy_log outputs).

Each condition is a run output directory; its <dataset>/redundancy/*.csv files are read. Reports, per condition
and set (candidates = before eviction, retained = after):
  - median / p90 of mean token cosine, fraction of pairs with cosine > tau, nearest-neighbour cosine
  - effective rank, effective rank / N, stable rank, psi / N
  - degenerate-case frequency: share of (scene, step, layer) with effective rank < --erank_min or mean cosine
    > --cos_max (the regime where psi alone would be misleading)
and saves plots: per-layer boxplots (erank, mean cosine) and psi/N vs effective rank scatter.

Usage:
  python tools/analyze_cache_redundancy.py --run IDC=<dir> --run random=<dir> --run mean=<dir> --out <dir>
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

METRICS = ["mean_cos", "frac_cos_gt_0.5", "frac_cos_gt_0.7", "frac_cos_gt_0.9", "nn_cos_mean", "erank",
           "erank_over_n", "stable_rank", "psi_over_n", "n_tokens"]


def load(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "*", "redundancy", "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No redundancy CSVs under {run_dir}/*/redundancy/")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["dataset"] = os.path.basename(os.path.dirname(os.path.dirname(path)))
        df["scene"] = os.path.splitext(os.path.basename(path))[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(df, erank_min, cos_max):
    rows = []
    for (cond, set_name), sub in df.groupby(["condition", "set"], sort=False):
        row = {"condition": cond, "set": set_name, "rows": len(sub)}
        for m in METRICS:
            if m in sub:
                row[f"{m}_median"] = sub[m].median()
                row[f"{m}_p90"] = sub[m].quantile(0.9)
        row["degenerate_share"] = float(((sub["erank"] < erank_min) | (sub["mean_cos"] > cos_max)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plots(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(dict.fromkeys(df["condition"]))
    for metric in ("erank", "mean_cos", "nn_cos_mean"):
        fig, ax = plt.subplots(figsize=(12, 4))
        layers = sorted(df["layer"].unique())
        width = 0.8 / max(len(conditions), 1)
        for i, cond in enumerate(conditions):
            sub = df[(df["condition"] == cond) & (df["set"] == "retained")]
            data = [sub[sub["layer"] == layer][metric].dropna().values for layer in layers]
            ax.boxplot(data, positions=np.array(layers) + (i - (len(conditions) - 1) / 2) * width, widths=width,
                       showfliers=False, patch_artist=True, boxprops=dict(facecolor=f"C{i}", alpha=0.6))
        ax.set_xticks(layers)
        ax.set_xlabel("layer")
        ax.set_ylabel(f"{metric} (retained cache)")
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc=f"C{i}", alpha=0.6) for i in range(len(conditions))], conditions)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"box_{metric}_by_layer.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, cond in enumerate(conditions):
        sub = df[(df["condition"] == cond) & (df["set"] == "retained")]
        ax.scatter(sub["erank"], sub["psi_over_n"], s=4, alpha=0.4, color=f"C{i}", label=cond)
    ax.axvspan(0, 2, color="red", alpha=0.08, label="degenerate (erank < 2)")
    ax.set_xscale("log")
    ax.set_xlabel("effective rank")
    ax.set_ylabel("psi / N")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scatter_psi_vs_erank.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="NAME=output_dir (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--erank_min", type=float, default=2.0)
    parser.add_argument("--cos_max", type=float, default=0.9)
    args = parser.parse_args()

    frames = []
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        if not run_dir:
            parser.error(f"--run expects NAME=dir, got {spec}")
        df = load(run_dir)
        df["condition"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    os.makedirs(args.out, exist_ok=True)
    table = summary_table(df, args.erank_min, args.cos_max)
    table.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    per_layer = df.groupby(["condition", "set", "layer"])[METRICS].median().reset_index()
    per_layer.to_csv(os.path.join(args.out, "per_layer_median.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        cols = ["condition", "set", "rows", "mean_cos_median", "frac_cos_gt_0.9_median", "nn_cos_mean_median",
                "erank_median", "erank_over_n_median", "psi_over_n_median", "degenerate_share"]
        print(table[[c for c in cols if c in table]].to_string(index=False))
    plots(df, args.out)
    print(f"Saved: {args.out}/summary.csv, per_layer_median.csv, box_*.png, scatter_psi_vs_erank.png")


if __name__ == "__main__":
    main()
