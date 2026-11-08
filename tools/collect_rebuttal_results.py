"""Collect the rebuttal experiment results into one JSON file.

Reads eval_results/mv_recon/rebuttal/ (strict, stride, compactor, pr_analysis, queue) and writes results.json:
  - per run and dataset: per-scene metrics and their means (CD = (Acc + Comp) / 2, F1 at 5/10/25 cm, FPS,
    PeakMemGB, frames), and whether the run finished (logs_all.txt written)
  - strict: actual cache overage over the budget per scene (from anchor_budget_overage.csv)
  - pr_analysis: cache redundancy medians before/after eviction and the share of degenerate cases
  - queue: job status (finished / failed / running / pending)
Safe to run while experiments are still going; unfinished runs are marked "complete": false.

Usage: python tools/collect_rebuttal_results.py [--root eval_results/mv_recon/rebuttal] [--out <root>/results.json]
"""

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime

import numpy as np

LINE = re.compile(
    r"Idx:\s*(?P<scene>[^,]+),\s*Acc:\s*(?P<acc>[^,]+),\s*Comp:\s*(?P<comp>[^,]+),.*?"
    r"Acc_med:\s*(?P<acc_med>[^,]+),\s*Compc_med:\s*(?P<comp_med>[^,]+),.*?"
    r"F1@5cm:\s*(?P<f1_5cm>[^,]+),\s*F1@10cm:\s*(?P<f1_10cm>[^,]+),\s*F1@25cm:\s*(?P<f1_25cm>[^,\s]+)"
    r"(?:,\s*Time:\s*(?P<time>[^,]+),\s*FPS:\s*(?P<fps>[^,]+))?(?:,\s*PeakMemGB:\s*(?P<peak_mem_gb>[^,]+))?"
)
EXPECTED_SCENES = {"7scenes": 18, "NRGBD": 9}
METRICS = ["cd", "acc", "comp", "acc_med", "comp_med", "f1_5cm", "f1_10cm", "f1_25cm", "fps", "time",
           "peak_mem_gb", "frames"]
REDUNDANCY_METRICS = ["n_tokens", "mean_cos", "frac_cos_gt_0.5", "frac_cos_gt_0.7", "frac_cos_gt_0.9",
                      "nn_cos_mean", "erank", "erank_over_n", "stable_rank", "psi_over_n"]


def parse_scenes(dataset_dir):
    scenes = {}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "logs_[0-9]*.txt"))):
        with open(path) as f:
            for line in f:
                m = LINE.search(line)
                if not m:
                    continue
                row = {"scene": m["scene"].strip()}
                for key, value in m.groupdict().items():
                    if key != "scene" and value is not None:
                        row[key] = float(value)
                row["cd"] = (row["acc"] + row["comp"]) / 2
                if "fps" in row and "time" in row:
                    row["frames"] = int(round(row["fps"] * row["time"]))
                scenes[row["scene"]] = row
    return list(scenes.values())


def overage(dataset_dir):
    """Per scene: how far the real cache (assigned budgets + anchor floor) went over the global budget."""
    out = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*", "anchor_budget_overage.csv"))):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        over = np.array([float(r["global_overage_tokens"]) for r in rows])
        budget = float(rows[0]["total_budget"])
        out.append({
            "scene": os.path.basename(os.path.dirname(path)),
            "budget": int(budget),
            "steps": len(rows),
            "steps_over_budget": int((over > 0).sum()),
            "max_overage_tokens": int(over.max()),
            "mean_overage_tokens": float(over.mean()),
            "max_overage_pct": float(100 * over.max() / budget),
        })
    if not out:
        return None
    return {
        "scenes": out,
        "max_overage_tokens": max(s["max_overage_tokens"] for s in out),
        "max_overage_pct": max(s["max_overage_pct"] for s in out),
        "share_of_steps_over_budget": float(sum(s["steps_over_budget"] for s in out) / sum(s["steps"] for s in out)),
    }


def redundancy(dataset_dir, erank_min=2.0, cos_max=0.9):
    paths = sorted(glob.glob(os.path.join(dataset_dir, "redundancy", "*.csv")))
    if not paths:
        return None
    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    out = {"rows": int(len(df)), "degenerate_rule": f"erank < {erank_min} or mean_cos > {cos_max}"}
    for set_name, sub in df.groupby("set"):
        out[set_name] = {
            "median": {m: float(sub[m].median()) for m in REDUNDANCY_METRICS if m in sub},
            "p10": {m: float(sub[m].quantile(0.1)) for m in REDUNDANCY_METRICS if m in sub},
            "p90": {m: float(sub[m].quantile(0.9)) for m in REDUNDANCY_METRICS if m in sub},
            "min_erank": float(sub["erank"].min()),
            "degenerate_share": float(((sub["erank"] < erank_min) | (sub["mean_cos"] > cos_max)).mean()),
        }
    return out


def run_summary(run_dir):
    datasets = {}
    for dataset_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        name = os.path.basename(dataset_dir)
        if not os.path.isdir(dataset_dir) or name.startswith("."):
            continue
        scenes = parse_scenes(dataset_dir)
        entry = {
            "complete": os.path.exists(os.path.join(dataset_dir, "logs_all.txt")),
            "n_scenes": len(scenes),
            "expected_scenes": EXPECTED_SCENES.get(name),
        }
        if scenes:
            entry["mean"] = {m: float(np.mean([s[m] for s in scenes if m in s]))
                             for m in METRICS if any(m in s for s in scenes)}
        entry["scenes"] = scenes
        extra = overage(dataset_dir)
        if extra:
            entry["budget_overage"] = extra
        extra = redundancy(dataset_dir)
        if extra:
            entry["redundancy"] = extra
        if scenes or extra:
            datasets[name] = entry
    return {
        "dir": run_dir,
        "complete": bool(datasets) and all(d["complete"] for d in datasets.values()),
        "datasets": datasets,
    }


def queue_status(root):
    qdir = os.path.join(root, "queue")
    jobs_file = os.path.join(os.path.dirname(__file__), "..", "src", "eval", "mv_recon", "rebuttal", "jobs_all.txt")
    names = []
    if os.path.exists(jobs_file):
        with open(jobs_file) as f:
            names = [l.split("\t")[0] for l in f if l.strip() and not l.lstrip().startswith("#")]
    done = {}
    if os.path.exists(os.path.join(qdir, "status.tsv")):
        with open(os.path.join(qdir, "status.tsv")) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    done[parts[0]] = {"gpu": parts[1], "rc": int(parts[2])}
    started = set()
    if os.path.exists(os.path.join(qdir, "queue.out")):
        with open(os.path.join(qdir, "queue.out")) as f:
            started = {m.group(1) for m in re.finditer(r" start (\S+) ", f.read())}
    jobs = {}
    for name in names:
        if name in done:
            jobs[name] = "finished" if done[name]["rc"] == 0 else f"failed (rc={done[name]['rc']})"
        elif name in started:
            jobs[name] = "running"
        else:
            jobs[name] = "pending"
    counts = {}
    for state in jobs.values():
        key = state.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    return {"counts": counts, "jobs": jobs}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(here, "..", "eval_results", "mv_recon", "rebuttal"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    out_path = args.out or os.path.join(root, "results.json")

    experiments = {}
    for exp in ("strict", "stride", "pr_analysis", "compactor"):
        runs = {}
        for run_dir in sorted(glob.glob(os.path.join(root, exp, "*"))):
            if os.path.isdir(run_dir) and not os.path.basename(run_dir).startswith("summary"):
                runs[os.path.basename(run_dir)] = run_summary(run_dir)
        experiments[exp] = runs
    # Compactor table A "ours" is the strict B60000_soft run (same configuration plus an overage log).
    if "B60000_soft" in experiments.get("strict", {}):
        experiments["compactor"]["A_ours_framesfull"] = {
            "same_as": "strict/B60000_soft", **experiments["strict"]["B60000_soft"]
        }

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": root,
        "notes": {
            "cd": "(Acc + Comp) / 2, metres",
            "fps": "inference only (excludes metric computation); CUDA determinism off",
            "frames": "fps * time, number of input frames of the scene",
            "compactor_A_ours": "reuses strict/B60000_soft",
        },
        "queue": queue_status(root),
        "experiments": experiments,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    counts = result["queue"]["counts"]
    print(f"Wrote {out_path}  (queue: {counts})")


if __name__ == "__main__":
    main()
