"""Local-density heterogeneity of the cache (rebuttal: psi counterexample, Figure 1b).

If tokens were spread evenly over independent directions, every token would have about the same number of similar
neighbours N_i (cosine > tau). Clustered keys (dense "floor"/"table" directions, sparse others) give very different
N_i across tokens. This script reports, per run / space (full 1024-d, IDC sketch 256-d) / set (candidates before
eviction, retained after):

  from the CSVs (launch.py --redundancy_log): per (step, layer) std, CV, top-10% / bottom-10% mean ratio, max / min
  ratio (+1 smoothed), Gini of N_i, and the std of the per-token mean cosine; medians and p10/p90 over (step, layer).
  from the per-token arrays (--redundancy_dump_tokens, candidates): tokens binned by density decile within each
  (step, layer); per decile the mean leverage (relative to the (step, layer) mean) and the share IDC kept.
  reference: two synthetic sets with the same psi/N, one balanced over 8 directions and one skewed (200 ... 3).

Usage:
  python tools/analyze_density_heterogeneity.py --run IDC_100f=<run dir> --run IDC_full=<run dir> --out <dir>
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TAUS = tuple(os.environ.get("SSTREAMVGGT_DENSITY_TAUS", "0.7,0.8,0.9").split(","))  # as logged
HET = ("cv", "topbot", "maxmin", "gini", "std", "mean")


def load_csvs(run_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "redundancy", "*.csv"))):
        df = pd.read_csv(path)
        df["dataset"] = path.split(os.sep)[-3]
        df["scene"] = os.path.splitext(os.path.basename(path))[0]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no redundancy CSVs under {run_dir}")
    df = pd.concat(frames, ignore_index=True)
    if "space" not in df:
        df["space"] = "full"
    return df


def csv_summary(df, cond):
    rows = []
    for (space, set_name), sub in df.groupby(["space", "set"]):
        row = {"run": cond, "space": space, "set": set_name, "rows": len(sub)}
        for tau in TAUS:
            for k in HET:
                col = f"dens{tau}_{k}"
                if col in sub and sub[col].notna().any():
                    row[f"{col}_median"] = float(sub[col].median())
                    row[f"{col}_p10"] = float(sub[col].quantile(0.1))
                    row[f"{col}_p90"] = float(sub[col].quantile(0.9))
        for k in ("tokcos_std", "tokcos_top10", "tokcos_bot10"):
            if k in sub and sub[k].notna().any():
                row[f"{k}_median"] = float(sub[k].median())
        rows.append(row)
    return rows


def token_deciles(run_dir, cond):
    """Per density decile (within each step/layer): mean relative leverage and the share of tokens kept."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "redundancy", "*_tokens.npz"))):
        z = np.load(path)
        for space in ("full", "sketch"):
            if f"{space}_offsets" not in z:
                continue
            off = z[f"{space}_offsets"]
            for tau in TAUS:
                dens, lev, kept = z[f"{space}_dens{tau}"], z[f"{space}_leverage"], z[f"{space}_kept"]
                for a, b in zip(off[:-1], off[1:]):
                    d, l, k = dens[a:b].astype(float), lev[a:b].astype(float), kept[a:b]
                    if b - a < 20 or l.mean() <= 0:
                        continue
                    ranks = d.argsort(kind="stable").argsort(kind="stable")  # ties broken by order
                    dec = np.minimum((ranks * 10) // (b - a), 9)
                    for q in range(10):
                        m = dec == q
                        rows.append({"run": cond, "space": space, "tau": tau, "decile": q + 1,
                                     "density": d[m].mean(), "rel_leverage": l[m].mean() / l.mean(),
                                     "kept_share": k[m].mean()})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["run", "space", "tau", "decile"]).mean(numeric_only=True).reset_index()


def synthetic_reference():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import torch
    from streamvggt.utils.redundancy_log import set_statistics

    g = torch.Generator().manual_seed(0)
    d = 64
    unit = lambda x: torch.nn.functional.normalize(x, dim=1)  # noqa: E731
    dirs = torch.linalg.qr(torch.randn(d, 8, generator=g, dtype=torch.float64))[0].T
    noise = lambda n: 0.3 / d ** 0.5 * torch.randn(n, d, generator=g, dtype=torch.float64)  # noqa: E731
    sets = {
        "synthetic: balanced, 8 directions x 40": unit(dirs.repeat_interleave(40, 0) + noise(320)),
        "synthetic: skewed, 200/60/20/15/10/8/4/3": unit(torch.cat(
            [dirs[i].repeat(n, 1) for i, n in enumerate([200, 60, 20, 15, 10, 8, 4, 3])]) + noise(320)),
    }
    rows = []
    for name, x in sets.items():
        st = set_statistics(x, (0.5,) + tuple(float(t) for t in TAUS), chunk=256)
        row = {"run": name, "space": "-", "set": "-", "rows": 1, "psi_over_n": st["psi_over_n"]}
        for tau in TAUS:
            for k in HET:
                row[f"dens{tau}_{k}_median"] = st.get(f"dens{tau}_{k}")
        row["tokcos_std_median"] = st.get("tokcos_std")
        rows.append(row)
    return rows


def plot(deciles, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for tau in TAUS:
        sub = deciles[deciles["tau"] == tau]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for i, ((run, space), s) in enumerate(sub.groupby(["run", "space"])):
            style = "-" if space == "full" else "--"
            axes[0].plot(s["decile"], s["rel_leverage"], style, marker="o", color=f"C{i // 2}", label=f"{run} {space}")
            axes[1].plot(s["decile"], s["kept_share"], style, marker="o", color=f"C{i // 2}", label=f"{run} {space}")
        axes[0].axhline(1.0, color="gray", lw=0.8)
        axes[0].set_ylabel("leverage / mean leverage")
        axes[1].set_ylabel("share kept by IDC")
        for ax in axes:
            ax.set_xlabel(f"local density decile (cosine > {tau}), 1 = sparsest")
            ax.set_xticks(range(1, 11))
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"deciles_tau{tau}.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir (repeatable)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary, decs = synthetic_reference(), []
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        summary += csv_summary(load_csvs(run_dir), name)
        decs.append(token_deciles(run_dir, name))
    table = pd.DataFrame(summary)
    table.to_csv(os.path.join(args.out, "heterogeneity_summary.csv"), index=False)
    deciles = pd.concat([d for d in decs if not d.empty], ignore_index=True) if any(not d.empty for d in decs) \
        else pd.DataFrame()
    if not deciles.empty:
        deciles.to_csv(os.path.join(args.out, "density_deciles.csv"), index=False)
        plot(deciles, args.out)
    cols = ["run", "space", "set"] + [f"dens0.8_{k}_median" for k in ("mean", "cv", "topbot", "maxmin", "gini")] + \
        ["tokcos_std_median"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
        if not deciles.empty:
            print(deciles[(deciles.tau == "0.8")].round(3).to_string(index=False))
    print(f"Saved: {args.out}/heterogeneity_summary.csv, density_deciles.csv, deciles_tau*.png")


if __name__ == "__main__":
    main()
