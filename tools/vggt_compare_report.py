#!/usr/bin/env python3
"""Compare the per-layer attention statistics of offline VGGT and StreamVGGT on identical inputs.

Reads tools/vggt_attention_peak.py (--vggt) and tools/attention_peak_analysis.py (--stream) outputs, checks that every
scene's input tensor hash matches, and writes (in --out):
  by_layer.csv     per layer, mean over scenes, for both models:
                     top50_full    VGGT whole matrix (AVGGT's setting; StreamVGGT has no full matrix)
                     top50_last    queries of the last frame over all patch keys (N = 5 x 1041 for both models)
                     top50_causal  key frame <= query frame (StreamVGGT: union of its streaming steps)
                     rowmax / rowmax_x_n / ent_norm   last frame, patch queries, head-averaged attention
  four_layers.csv  the same for layers 1, 5, 12, 22 (AVGGT Fig. 2)
  spearman.csv     rank agreement of the two models over the 24 layers, per statistic
  fig_vggt_vs_stream.png
"""

import argparse
import glob
import json
import os

import pandas as pd
from scipy.stats import spearmanr

FOUR = [1, 5, 12, 22]


def scenes(root):
    return {os.path.basename(d): d for d in glob.glob(os.path.join(root, "[0-9][0-9][0-9][0-9]_*"))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vggt", required=True)
    p.add_argument("--stream", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    V, St = scenes(a.vggt), scenes(a.stream)
    common = sorted(set(V) & set(St))
    recs = []
    for sc in common:
        mv, ms = (json.load(open(os.path.join(d[sc], "meta.json"))) for d in (V, St))
        assert mv["input_sha256"] == ms.get("input_sha256"), f"{sc}: inputs differ"
        assert mv["checks"]["calls_ok"] and mv["checks"].get("patch_invariant", True), sc
        assert ms["checks"]["calls_ok"] and ms["checks"].get("patch_invariant", True), sc
        vt = pd.read_csv(os.path.join(V[sc], "vggt_top50.csv")).pivot(index="layer", columns="scope", values="top50_mean")
        vr = pd.read_csv(os.path.join(V[sc], "vggt_rows.csv"))
        last_f = int(vr.query_frame.max())
        vr = vr[(vr["head"] == -1) & (vr.query_type == "patch") & (vr.query_frame == last_f)].set_index("layer")
        st = pd.read_csv(os.path.join(St[sc], "attn_peak_top50.csv"))
        sr = pd.read_csv(os.path.join(St[sc], "attn_peak_layers.csv"))
        sr = sr[(sr["head"] == -1) & (sr.query_type == "patch") & (sr.step == last_f + 1)].set_index("layer")
        for layer in range(24):
            recs.append({
                "scene": sc, "layer": layer,
                "vggt_top50_full": vt.loc[layer, "full"], "vggt_top50_last": vt.loc[layer, "last"],
                "vggt_top50_causal": vt.loc[layer, "causal"],
                "stream_top50_last": st[(st.layer == layer) & (st.step == str(last_f + 1))].top50_mean.iloc[0],
                "stream_top50_causal": st[(st.layer == layer) & (st.step == "union")].top50_mean.iloc[0],
                "vggt_rowmax": vr.loc[layer, "rowmax_mean"], "stream_rowmax": sr.loc[layer, "rowmax_mean"],
                "vggt_rowmax_x_n": vr.loc[layer, "rowmax_x_n"], "stream_rowmax_x_n": sr.loc[layer, "rowmax_x_n"],
                "vggt_ent_norm": vr.loc[layer, "ent_norm_mean"], "stream_ent_norm": sr.loc[layer, "ent_norm_mean"],
            })
    df = pd.DataFrame(recs)
    by = df.groupby("layer").mean(numeric_only=True)
    by.to_csv(os.path.join(a.out, "by_layer.csv"))
    by.loc[FOUR].T.to_csv(os.path.join(a.out, "four_layers.csv"))
    sp = []
    for v, s in (("vggt_top50_last", "stream_top50_last"), ("vggt_top50_causal", "stream_top50_causal"),
                 ("vggt_top50_full", "stream_top50_causal"), ("vggt_rowmax_x_n", "stream_rowmax_x_n"),
                 ("vggt_ent_norm", "stream_ent_norm")):
        rho, pv = spearmanr(by[v], by[s])
        ratio = (by[s] / by[v]).median()
        sp.append({"vggt": v, "stream": s, "spearman_rho": rho, "p_value": pv, "median_stream_over_vggt": ratio})
    sp = pd.DataFrame(sp)
    sp.to_csv(os.path.join(a.out, "spearman.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axvspan(-0.5, 8.5, color="#e8eef7", lw=0)
    ax.axvspan(19.5, 23.5, color="#e8eef7", lw=0)
    ax.plot(by.index, by.vggt_top50_full, "o-", ms=4, color="C3", label="VGGT, whole matrix (AVGGT setting)")
    ax.plot(by.index, by.vggt_top50_last, "s--", ms=3.5, color="C1", label="VGGT, last-frame queries")
    ax.plot(by.index, by.stream_top50_last, "o-", ms=4, color="C0", label="StreamVGGT, last-frame queries")
    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("global attention layer (0-indexed)")
    ax.set_ylabel("top-50 attention entry\n(head-avg., patch tokens)")
    ax.set_title(f"Same inputs: 7-Scenes, {len(common)} sequences, 5 frames", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_vggt_vs_stream.png"), dpi=200)

    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(by.round(4).to_string())
        print(by.loc[FOUR].T.round(5).to_string())
        print(sp.round(4).to_string(index=False))
    print(f"scenes={len(common)}; saved {a.out}")


if __name__ == "__main__":
    main()
