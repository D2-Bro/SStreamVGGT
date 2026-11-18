#!/usr/bin/env python3
"""EpiCache-style layer sensitivity (Kim et al., ICML 2026, Sec. 3.2) for StreamVGGT, offline.

Compares the keys each global layer produces for every frame in two streaming runs of the same sequence
(launch.py --key_dump_dir): a full-cache run (no eviction) and a run with eviction at a uniform per-layer budget
(IDC + uniform allocation). A key is computed once, when its frame enters, from that layer's input, so the two runs
differ only through the attention of earlier layers over the reduced cache.

  sigma_l = mean over heads, tokens and steps of cos(k_full, k_evict)          (Eq. 9; all tokens of every frame)
  s_l     = 1 - sigma_l
  w_l     = s_l^alpha / sum_j s_j^alpha                                          (Eq. 10)

Steps before the first eviction are identical in both runs (cos = 1); they are excluded by default
(--include_warmup keeps them, as the paper's average over all positions would).

Usage:
  python tools/epicache_profile.py --full <dump dir>/<dataset> --evict <dump dir>/<dataset> \
      --calibration <scene.pt> [--extra <scene.pt> ...] --alphas 1 2 4 --out <dir>
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def layer_profile(full_path, evict_path, include_warmup=False):
    full = torch.load(full_path, map_location="cpu")["keys"]
    evict = torch.load(evict_path, map_location="cpu")["keys"]
    assert len(full) == len(evict), "layer count differs"
    per_step = []  # [layers, steps]
    for kf, ke in zip(full, evict):
        assert kf.shape == ke.shape, f"key shapes differ: {tuple(kf.shape)} vs {tuple(ke.shape)}"
        cos = F.cosine_similarity(kf.float(), ke.float(), dim=-1)  # [steps, heads, tokens]
        per_step.append(cos.mean(dim=(1, 2)).double().numpy())
    per_step = np.stack(per_step)
    differs = (per_step < 1.0 - 1e-6).any(axis=0)
    start = int(np.argmax(differs)) if differs.any() else per_step.shape[1]
    window = per_step if include_warmup else per_step[:, start:]
    return window.mean(axis=1), per_step, start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", required=True, help="dump dir of the full-cache run (one dataset)")
    p.add_argument("--evict", required=True, help="dump dir of the run with eviction (same dataset)")
    p.add_argument("--calibration", required=True, nargs="+", help="scene file name(s) used for the allocation")
    p.add_argument("--extra_full", default=None, help="dump dir of other scenes (consistency check only)")
    p.add_argument("--extra_evict", default=None)
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--include_warmup", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    profiles = {}
    for name in a.calibration:
        sigma, per_step, start = layer_profile(os.path.join(a.full, name), os.path.join(a.evict, name), a.include_warmup)
        profiles[name] = sigma
        np.savetxt(os.path.join(a.out, f"cos_per_step_{os.path.splitext(name)[0]}.csv"), per_step.T, delimiter=",",
                   header=",".join(f"layer{l}" for l in range(per_step.shape[0])), comments="")
        print(f"{name}: {per_step.shape[1]} steps, eviction from step {start}")
    sigma = np.mean([profiles[n] for n in a.calibration], axis=0)
    s = np.clip(1.0 - sigma, 0.0, None)

    extra = {}
    if a.extra_full and a.extra_evict:
        for path in sorted(glob.glob(os.path.join(a.extra_full, "*.pt"))):
            name = os.path.basename(path)
            if not os.path.exists(os.path.join(a.extra_evict, name)):
                continue
            sig, _, _ = layer_profile(path, os.path.join(a.extra_evict, name), a.include_warmup)
            extra[name] = 1.0 - sig

    rows = {"layer": list(range(len(s))), "sigma": sigma.tolist(), "sensitivity": s.tolist()}
    for alpha in a.alphas:
        w = s ** alpha
        w = w / w.sum()
        rows[f"proportion_alpha{alpha:g}"] = w.tolist()
        with open(os.path.join(a.out, f"proportions_alpha{alpha:g}.json"), "w") as f:
            json.dump({"num_layers": len(w), "proportions": w.tolist(), "alpha": alpha,
                       "calibration": a.calibration, "include_warmup": a.include_warmup,
                       "definition": "EpiCache: s_l = 1 - mean cos(k_full, k_evict); w = s^alpha / sum s^alpha"}, f, indent=1)
    for name, sx in extra.items():
        rows[f"sensitivity_{os.path.splitext(name)[0]}"] = sx.tolist()
    import pandas as pd
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(a.out, "profile.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(5).to_string(index=False))
    if extra:
        from scipy.stats import spearmanr
        for name, sx in extra.items():
            print(f"Spearman(calibration, {name}) = {spearmanr(s, sx)[0]:.3f}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
