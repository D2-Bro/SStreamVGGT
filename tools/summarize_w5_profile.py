#!/usr/bin/env python
"""Aggregate W5 profiling runs into tables (T1-T3) and figures (F1-F4).

Expected layout (written by src/eval/*/run_w5_profile.sh):
  <root>/mv_recon/ours_b{B}_{R1|R2|R3}_rep{k}/
  <root>/pose_evaluation/{dataset}/ours_b{B}_{R1|R2|R3}_rep{k}/
each with run.log and (R1) w5_profile/summary_rank*.csv, w5_profile/{dataset}/{seq}_{frames,kv}.csv.

Usage:
  python tools/summarize_w5_profile.py --root eval_results/w5 --out eval_results/w5/summary
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

RUN_RE = re.compile(r"ours_b(?P<budget>\d+)_(?P<mode>R[123])_rep(?P<rep>\d+)$")
LINE_RE = re.compile(r"^\[(?P<tag>PerfTraceSummary|EvictionProfileSummary)\]\s+(?P<label>\S+)\s+(?P<body>.*)$")
KV_RE = re.compile(r"(\S+?)=([-+0-9.eE]+)ms")


def find_runs(root):
    runs = []
    for path in glob.glob(os.path.join(root, "**", "ours_b*_R*_rep*"), recursive=True):
        m = RUN_RE.search(os.path.basename(path))
        if not m or not os.path.isdir(path):
            continue
        rel = os.path.relpath(path, root).split(os.sep)
        runs.append(
            {
                "path": path,
                "bench": rel[0],
                "budget": int(m["budget"]),
                "mode": m["mode"],
                "rep": int(m["rep"]),
            }
        )
    return runs


# ------------------------------------------------------------------ R1 tables
def load_r1(runs):
    frames = []
    for run in runs:
        if run["mode"] != "R1":
            continue
        for csv_path in glob.glob(os.path.join(run["path"], "w5_profile", "summary_rank*.csv")):
            df = pd.read_csv(csv_path)
            for key in ("bench", "budget", "rep", "path"):
                df[key] = run[key]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


T1_COLS = [
    "n_frames", "e2e_time_s", "e2e_fps", "infer_time_s", "infer_fps", "idc_time_s", "kaba_time_s",
    "steady_fps", "lat_p50_ms", "lat_p95_ms", "lat_max_ms", "peak_alloc_GB", "peak_reserved_GB", "baseline_GB",
]


def table_t1(r1):
    # median over repetitions per sequence, then aggregate per dataset
    per_seq = r1.groupby(["bench", "dataset", "budget", "seq"], as_index=False)[T1_COLS].median()
    agg = per_seq.groupby(["bench", "dataset", "budget"]).agg(
        n_seqs=("seq", "count"),
        frames=("n_frames", "sum"),
        e2e_time_s=("e2e_time_s", "sum"),
        infer_time_s=("infer_time_s", "sum"),
        idc_time_s=("idc_time_s", "sum"),
        kaba_time_s=("kaba_time_s", "sum"),
        lat_p50_ms=("lat_p50_ms", "mean"),
        lat_p95_ms=("lat_p95_ms", "mean"),
        peak_alloc_GB_mean=("peak_alloc_GB", "mean"),
        peak_alloc_GB_max=("peak_alloc_GB", "max"),
    ).reset_index()
    # frame-weighted FPS over the whole dataset
    agg["e2e_fps"] = agg["frames"] / agg["e2e_time_s"]
    agg["infer_fps"] = agg["frames"] / agg["infer_time_s"]
    agg["idc_share_%"] = 100 * agg["idc_time_s"] / agg["e2e_time_s"]
    agg["kaba_share_%"] = 100 * agg["kaba_time_s"] / agg["e2e_time_s"]
    return per_seq, agg


def _anchor_overage_csv(row):
    run_dir = row["path"]
    candidates = [
        os.path.join(run_dir, row["dataset"], row["seq"], "anchor_budget_overage.csv"),  # mv_recon
        os.path.join(run_dir, row["seq"], "anchor_budget_overage.csv"),  # pose
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def table_t2(r1):
    rows = []
    first_rep = r1[r1["rep"] == r1.groupby(["bench", "dataset", "budget"])["rep"].transform("min")]
    for _, row in first_rep.iterrows():
        kv_csv = os.path.join(row["path"], "w5_profile", row["dataset"], f"{row['seq']}_kv.csv")
        out = {
            "bench": row["bench"], "dataset": row["dataset"], "budget": row["budget"], "seq": row["seq"],
            "n_frames": row["n_frames"],
            "kv_tokens_max": row["kv_tokens_max"],
            "kv_excess_max": row["kv_excess_max"],
            "kv_excess_mean": row["kv_excess_mean"],
            "n_frames_excess_pos": row["n_frames_excess_pos"],
            "kv_bytes_max_MB": row["kv_bytes_max_MB"],
            "cam_cache_tokens_final(ref)": row["cam_cache_tokens_final"],
        }
        overage_csv = _anchor_overage_csv(row)
        if overage_csv and os.path.exists(kv_csv):
            kv = pd.read_csv(kv_csv)
            ana = pd.read_csv(overage_csv)
            if len(ana):
                out["analytic_overage_max"] = int(ana["global_overage_tokens"].max())
                merged = kv.merge(ana, left_on="frame_idx", right_on="step", how="inner")
                measured = merged["kv_excess"].clip(lower=0)
                out["measured_vs_analytic_max_absdiff"] = int(
                    (measured - merged["global_overage_tokens"]).abs().max()
                ) if len(merged) else None
        rows.append(out)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- R2/R3 table
def parse_summary_lines(log_path):
    totals = defaultdict(float)
    if not os.path.exists(log_path):
        return totals
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            for key, value in KV_RE.findall(m["body"]):
                if key.endswith("_mean") or "cpu_enqueue" in key:
                    continue
                totals[(m["tag"], m["label"], key)] += float(value)
    return totals


def table_t3(runs):
    rows = []
    for run in runs:
        if run["mode"] not in ("R2", "R3"):
            continue
        totals = parse_summary_lines(os.path.join(run["path"], "run.log"))
        for (tag, label, key), ms in totals.items():
            rows.append({
                "bench": run["bench"], "budget": run["budget"], "mode": run["mode"], "rep": run["rep"],
                "dataset_dir": os.path.relpath(run["path"]),
                "tag": tag, "label": label, "component": key, "total_ms": ms,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Percent within each (run, tag, label) group, excluding keys that are sums of other keys
    # (e.g. stream "heads" = camera/depth/point head; inference total/profiled/other).
    # In the perf_trace "eviction" line, manager_* entries break down "select", so they get a
    # separate denominator (select) from the top-level select/index_update split.
    agg_keys = df["component"].str.contains(r"^(?:total|forward_total|profiled|other|heads)_", regex=True)
    sub_keys = (df["tag"] == "PerfTraceSummary") & (df["label"] == "eviction") & df["component"].str.startswith("manager_")
    df["level"] = "top"
    df.loc[sub_keys, "level"] = "sub_of_select"
    df.loc[agg_keys, "level"] = "aggregate"
    df["share_%"] = None
    group_cols = ["dataset_dir", "tag", "label", "level"]
    keep = df["level"] != "aggregate"
    denom = df[keep].groupby(group_cols)["total_ms"].transform("sum")
    df.loc[keep, "share_%"] = 100 * df.loc[keep, "total_ms"] / denom
    return df


# ------------------------------------------------------------------ figures
def make_figures(r1, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures")
        return
    os.makedirs(out_dir, exist_ok=True)
    base = r1[r1["rep"] == r1.groupby(["bench", "dataset", "budget"])["rep"].transform("min")]
    main_budget = 60000 if (base["budget"] == 60000).any() else int(base["budget"].mode()[0])

    for dataset, grp in base[base["budget"] == main_budget].groupby("dataset"):
        fig1, ax1 = plt.subplots(figsize=(7, 3.5))
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        for _, row in grp.iterrows():
            fr = os.path.join(row["path"], "w5_profile", dataset, f"{row['seq']}_frames.csv")
            kv = os.path.join(row["path"], "w5_profile", dataset, f"{row['seq']}_kv.csv")
            if os.path.exists(fr):
                d = pd.read_csv(fr)
                ax1.plot(d["frame_idx"], d["latency_ms"], lw=0.8, alpha=0.7)
            if os.path.exists(kv):
                d = pd.read_csv(kv)
                ax2.plot(d["frame_idx"], d["kv_tokens_total"], lw=0.8, alpha=0.7)
        ax1.set(xlabel="frame index", ylabel="latency (ms)", title=f"F1 {dataset}: per-frame latency (E2E)")
        ax2.axhline(main_budget, color="k", ls="--", lw=1, label=f"budget {main_budget}")
        ax2.set(xlabel="frame index", ylabel="stored KV tokens (24 layers)", title=f"F2 {dataset}: KV storage")
        ax2.legend()
        fig1.tight_layout(); fig1.savefig(os.path.join(out_dir, f"F1_latency_{dataset}.png"), dpi=150)
        fig2.tight_layout(); fig2.savefig(os.path.join(out_dir, f"F2_kv_tokens_{dataset}.png"), dpi=150)
        plt.close(fig1); plt.close(fig2)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for dataset, grp in base[base["budget"] == main_budget].groupby("dataset"):
        ax.scatter(grp["n_frames"], grp["peak_alloc_GB"], s=12, label=dataset)
    ax.set(xlabel="sequence length (frames)", ylabel="peak allocated (GB)", title="F3 peak memory vs length")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F3_peak_memory.png"), dpi=150); plt.close(fig)

    if base["budget"].nunique() > 1:
        sweep = base.groupby(["dataset", "budget"]).agg(
            peak=("peak_alloc_GB", "mean"), frames=("n_frames", "sum"), t=("e2e_time_s", "sum")
        ).reset_index()
        sweep["fps"] = sweep["frames"] / sweep["t"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
        for dataset, grp in sweep.groupby("dataset"):
            a1.plot(grp["budget"], grp["peak"], marker="o", label=dataset)
            a2.plot(grp["budget"], grp["fps"], marker="o", label=dataset)
        a1.set(xlabel="budget", ylabel="peak allocated (GB)")
        a2.set(xlabel="budget", ylabel="E2E FPS")
        a1.legend(fontsize=7)
        fig.suptitle("F4 budget sweep")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "F4_budget_sweep.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_results/w5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "summary")
    os.makedirs(out, exist_ok=True)

    runs = find_runs(args.root)
    print(f"found {len(runs)} runs under {args.root}")
    r1 = load_r1(runs)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    if not r1.empty:
        per_seq, t1 = table_t1(r1)
        per_seq.to_csv(os.path.join(out, "T1_per_sequence.csv"), index=False)
        t1.to_csv(os.path.join(out, "T1_dataset.csv"), index=False)
        print("\n== T1 (per dataset; median over reps)\n", t1.round(3).to_string(index=False))
        t2 = table_t2(r1)
        t2.to_csv(os.path.join(out, "T2_budget_excess.csv"), index=False)
        print("\n== T2 (24-layer KV budget excess; camera cache is reference only)\n", t2.to_string(index=False))
        make_figures(r1, os.path.join(out, "figures"))
    t3 = table_t3(runs)
    if not t3.empty:
        t3.to_csv(os.path.join(out, "T3_components.csv"), index=False)
        print(f"\n== T3 written: {len(t3)} component rows")
    print(f"\noutputs -> {out}")


if __name__ == "__main__":
    main()
