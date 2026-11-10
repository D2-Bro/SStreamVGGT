#!/usr/bin/env python3
"""Aggregate tools/attention_peak_analysis.py over scenes, plot the layer profile, and compare it with KABA.

Outputs (in --out):
  attn_peak_by_layer.csv   per layer: AVGGT top-50 (union of steps = full 5-view causal matrix, and step 5), per-row
                           max x N, normalised entropy, top-10 mass, cross-frame and special-key mass (patch queries,
                           step 5, mean over scenes; per-head values averaged over heads, and head-averaged matrix)
  kaba_compare.csv         the same next to KABA's per-layer allocation (share of the budget, fraction of steps at
                           the anchor floor), psi/N and key norm; Spearman correlations printed and saved
  fig_attn_peak.png        x = global layer (0-23), y = AVGGT top-50 attention value (median and IQR over scenes)
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EARLY, MIDDLE = range(0, 9), range(9, 20)  # AVGGT: early = 0..8 (t_early = 9); middle/last split from examples


def load(attn_dir):
    tops, rows, norms, metas = [], [], [], []
    for d in sorted(glob.glob(os.path.join(attn_dir, "[0-9][0-9][0-9][0-9]_*"))):
        meta = json.load(open(os.path.join(d, "meta.json")))
        c = meta["checks"]
        ok = c["calls_ok"] and c["k_shape_equals_step_x_tokens"] and c["first_frame_no_past"] and \
            c.get("eviction_policy_invariant", True) and c.get("patch_invariant", True)
        if not ok:
            raise RuntimeError(f"{d}: checks failed {c}")
        metas.append(meta)
        scene = os.path.basename(d)
        for name, store in (("attn_peak_top50.csv", tops), ("attn_peak_layers.csv", rows), ("key_norm.csv", norms)):
            df = pd.read_csv(os.path.join(d, name))
            df["scene"] = scene
            store.append(df)
    return pd.concat(tops), pd.concat(rows), pd.concat(norms), metas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn", required=True)
    p.add_argument("--kaba_layers", required=True, help="glob of anchor_budget_overage_layers.csv (KABA run)")
    p.add_argument("--density", required=True, help="pr_density_summary/per_layer_median.csv")
    p.add_argument("--density_condition", default="IDC_full")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    top, rows, norms, metas = load(a.attn)
    T = str(max(int(s) for s in top.step.unique() if s != "union"))
    n_scenes = top.scene.nunique()

    def per_layer(df, col):
        g = df.groupby(["scene", "layer"])[col].mean().unstack(0)
        return g.mean(axis=1), g.quantile(0.25, axis=1), g.quantile(0.75, axis=1), g.median(axis=1)

    tu, ts = top[top.step == "union"], top[top.step == T]
    patch = rows[(rows.query_type == "patch") & (rows.step == int(T))]
    heads, avg = patch[patch["head"] >= 0], patch[patch["head"] == -1]
    tab = pd.DataFrame({
        "top50_union": per_layer(tu, "top50_mean")[0],
        "top50_union_median": per_layer(tu, "top50_mean")[3],
        f"top50_step{T}": per_layer(ts, "top50_mean")[0],
        "top50_frac_self": per_layer(tu, "frac_self")[0],
        "top50_frac_same_position": per_layer(tu, "frac_same_position")[0],
        "top50_frac_cross_frame": per_layer(tu, "frac_cross_frame")[0],
        "rowmax_x_n_head": per_layer(heads, "rowmax_x_n")[0],
        "rowmax_x_n_avgattn": per_layer(avg, "rowmax_x_n")[0],
        "ent_norm_head": per_layer(heads, "ent_norm_mean")[0],
        "ent_norm_avgattn": per_layer(avg, "ent_norm_mean")[0],
        "top10_mass_head": per_layer(heads, "top10_mass")[0],
        "cross_frame_mass_head": per_layer(heads, "cross_frame_mass")[0],
        "special_key_mass_head": per_layer(heads, "special_key_mass")[0],
        "key_norm": norms[norms.step == int(T)].groupby("layer").key_norm_all.mean(),
    })
    tab.index.name = "layer"
    tab["group"] = ["early" if l in EARLY else "middle" if l in MIDDLE else "last" for l in tab.index]
    tab.to_csv(os.path.join(a.out, "attn_peak_by_layer.csv"))

    # KABA: share of the total budget per layer and the fraction of steps a layer sits at the anchor floor
    kl = pd.concat([pd.read_csv(f).assign(run=i) for i, f in enumerate(sorted(glob.glob(a.kaba_layers)))])
    kl = kl[kl.capacity > kl.anchor_tokens]  # steps where eviction is needed
    kl["share"] = kl.assigned_budget / kl.groupby(["run", "step"]).assigned_budget.transform("sum")
    kaba = pd.DataFrame({"kaba_share": kl.groupby("layer").share.mean(),
                         "kaba_retained_median": kl.groupby("layer").retained_tokens.median(),
                         "kaba_anchor_floor_frac": kl.assign(f=kl.assigned_budget <= kl.anchor_tokens)
                         .groupby("layer").f.mean()})
    dens = pd.read_csv(a.density)
    dens = dens[(dens.condition == a.density_condition) & (dens.space == "full") & (dens.set == "candidates")]
    kaba["psi_over_n"] = dens.set_index("layer").psi_over_n
    cmp_ = tab.join(kaba)
    cmp_.to_csv(os.path.join(a.out, "kaba_compare.csv"))

    corr = []
    for x in ("top50_union", "rowmax_x_n_head", "ent_norm_head", "cross_frame_mass_head"):
        for y in ("kaba_share", "kaba_retained_median", "kaba_anchor_floor_frac", "psi_over_n", "key_norm"):
            s = cmp_[[x, y]].dropna()
            rho, pv = spearmanr(s[x], s[y])
            corr.append({"attention": x, "kaba": y, "n_layers": len(s), "spearman_rho": rho, "p_value": pv})
    corr = pd.DataFrame(corr)
    corr.to_csv(os.path.join(a.out, "kaba_spearman.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, q1, q3, med = per_layer(tu, "top50_mean")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for rng, c in ((EARLY, "#e8eef7"), (range(20, 24), "#e8eef7")):
        ax.axvspan(rng.start - 0.5, rng.stop - 0.5, color=c, lw=0)
    ax.fill_between(med.index, q1, q3, color="C0", alpha=0.25, lw=0, label="IQR over sequences")
    ax.plot(med.index, med, "o-", color="C0", ms=4, label="median over sequences")
    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("global attention layer (0-indexed)")
    ax.set_ylabel("top-50 attention entry\n(head-avg., patch tokens)")
    ax.set_title(f"StreamVGGT causal global attention, 7-Scenes ({n_scenes} seq., {T} frames, full cache)", fontsize=9)
    ax.set_ylim(top=float(q3.max()) * 1.15)
    ymax = ax.get_ylim()[1]
    for x, name in ((4, "early"), (14, "middle"), (21.5, "last")):
        ax.text(x, ymax * 0.93, name, ha="center", fontsize=8, color="0.35")
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_attn_peak.png"), dpi=200)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["group", "top50_union", "top50_frac_self", "rowmax_x_n_head", "ent_norm_head", "cross_frame_mass_head",
                "kaba_share", "kaba_anchor_floor_frac", "psi_over_n", "key_norm"]
        print(cmp_[cols].round(4).to_string())
        print(tab.groupby("group")[["top50_union", "rowmax_x_n_head", "ent_norm_head", "cross_frame_mass_head"]]
              .mean().round(4).to_string())
        print(corr.round(3).to_string(index=False))
    print(f"scenes={n_scenes}; saved {a.out}")


if __name__ == "__main__":
    main()
