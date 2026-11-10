#!/usr/bin/env python3
"""Layer-wise "top activated entry" size of StreamVGGT's causal global attention (rebuttal R2-Q3, AVGGT check).

Streams a short 7-Scenes sequence through StreamVGGT with a full KV cache, with eviction and KABA disabled, and
records post-softmax attention statistics of every global layer, without storing attention matrices.

Nothing in src/ is modified. Inside this process only:
  * streamvggt.layers.attention.F is replaced by a proxy whose scaled_dot_product_attention calls the original and
    returns its output unchanged; while a global block's attention is running (forward hooks), it also recomputes
    A = softmax(q k^T / sqrt(D)) in fp32 from the same q, k and accumulates the statistics.
  * Attention.eviction and EvictionManager.select raise if they are ever called (no eviction, no token scoring),
    and the layer budget allocator is checked to run with zero scores and alpha = 0 (uniform; no KABA) and to give
    every layer more budget than its live keys.

AVGGT (arXiv 2512.02541, Sec. 3.1 / Fig. 2): attention averaged over heads, restricted to patch tokens, and the
top-k (k = 50) query-key entries of the whole matrix; global layers indexed 0-23. Reproduced as top50_* below
(per streaming step and over the union of steps = the full 5-view causal matrix), plus per-row statistics.

Usage (CPU):
  CUDA_VISIBLE_DEVICES= python tools/attention_peak_analysis.py --scene_index 0 --out <dir> [--verify]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NUM_SPECIAL = 5  # camera token + 4 register tokens (aggregator.patch_start_idx)
TOPK_ROW = (1, 5, 10)
TOP_AVGGT = 50
HEAD_CHUNK = 4


class State:
    layer = None  # global layer index while its attention runs
    calls = {}  # step -> number of global attention calls
    rows = []  # per (layer, step, head, query type) statistics
    top50 = []  # per (layer, step) AVGGT statistics
    union = {}  # layer -> list of (value, step, q_patch, key_frame, k_patch)
    keynorm = []
    tokens_per_frame = None
    verify = False
    max_out_diff = 0.0
    max_rowsum_err = 0.0
    collect = True
    alloc_calls = 0


def _row_metrics(a: torch.Tensor, n_keys: int, past: int, special_key_mask: torch.Tensor) -> dict:
    """a: [R, N] post-softmax rows. Returns the per-row metrics averaged (and some quantiles) over rows."""
    top, _ = a.topk(max(max(TOPK_ROW), math.ceil(0.01 * n_keys)), dim=-1)
    k1p = math.ceil(0.01 * n_keys)
    rowmax = top[:, 0]
    ent = -(a * torch.log(a.clamp_min(1e-30))).sum(-1) / math.log(n_keys)
    out = {
        "rowmax_mean": rowmax.mean(), "rowmax_median": rowmax.median(), "rowmax_p90": rowmax.quantile(0.9),
        "rowmax_x_n": rowmax.mean() * n_keys,
        "ent_norm_mean": ent.mean(), "ent_norm_median": ent.median(),
        "top1pct_mean": top[:, :k1p].mean(), "top1pct_mass": top[:, :k1p].sum(-1).mean(),
        "special_key_mass": a[:, special_key_mask].sum(-1).mean(),
        "cross_frame_mass": a[:, :past].sum(-1).mean() if past > 0 else torch.tensor(float("nan")),
    }
    for k in TOPK_ROW:
        out[f"top{k}_mean"] = top[:, :k].mean()
        out[f"top{k}_mass"] = top[:, :k].sum(-1).mean()
    return {key: float(val) for key, val in out.items()}


def _record(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
    layer, P = State.layer, State.tokens_per_frame
    B, H, Q, D = q.shape
    N = k.shape[2]
    assert B == 1 and Q == P, f"layer {layer}: expected one frame of queries, got {tuple(q.shape)}"
    assert N % P == 0, f"layer {layer}: {N} keys is not a whole number of frames"
    step = N // P  # 1-based frame index; full cache -> keys = step * P
    State.calls[step] = State.calls.get(step, 0) + 1
    past = N - P
    special_key = (torch.arange(N) % P) < NUM_SPECIAL
    qtypes = {"all": slice(0, Q), "patch": slice(NUM_SPECIAL, Q), "special": slice(0, NUM_SPECIAL)}
    a_mean = torch.zeros(Q, N, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for h0 in range(0, H, HEAD_CHUNK):
        a = torch.softmax(qf[0, h0:h0 + HEAD_CHUNK] @ kf[0, h0:h0 + HEAD_CHUNK].transpose(-1, -2) / math.sqrt(D), -1)
        State.max_rowsum_err = max(State.max_rowsum_err, float((a.sum(-1) - 1).abs().max()))
        if State.verify:
            ref = a @ vf[0, h0:h0 + HEAD_CHUNK]
            State.max_out_diff = max(State.max_out_diff, float((ref - out[0, h0:h0 + HEAD_CHUNK].float()).abs().max()))
        a_mean += a.sum(0)
        for i in range(a.shape[0]):
            for qt, sl in qtypes.items():
                State.rows.append({"layer": layer, "step": step, "n_keys": N, "head": h0 + i, "query_type": qt,
                                   **_row_metrics(a[i, sl], N, past, special_key)})
    a_mean /= H
    for qt, sl in qtypes.items():
        State.rows.append({"layer": layer, "step": step, "n_keys": N, "head": -1, "query_type": qt,
                           **_row_metrics(a_mean[sl], N, past, special_key)})
    # AVGGT: head-averaged, patch queries x patch keys, top-50 entries of the whole matrix (no renormalisation)
    patch_keys = torch.nonzero(~special_key).squeeze(1)
    sub = a_mean[NUM_SPECIAL:, patch_keys]
    vals, flat = sub.flatten().topk(TOP_AVGGT)
    qi = (flat // sub.shape[1]).numpy()
    kj = patch_keys[flat % sub.shape[1]].numpy()
    kframe, kpatch = kj // P, kj % P - NUM_SPECIAL
    cur = step - 1
    self_e = (kframe == cur) & (kpatch == qi)
    same_pos = (kframe != cur) & (kpatch == qi)
    vals_np = vals.numpy()
    State.top50.append({"layer": layer, "step": step, "n_keys": N,
                        "top50_mean": float(vals_np.mean()), "top50_min": float(vals_np.min()),
                        "top50_max": float(vals_np.max()), "top50_mean_x_n": float(vals_np.mean()) * N,
                        "frac_self": float(self_e.mean()), "frac_same_position": float(same_pos.mean()),
                        "frac_cross_frame": float((kframe != cur).mean())})
    State.union.setdefault(layer, []).extend(
        zip(vals_np.tolist(), [step] * TOP_AVGGT, qi.tolist(), kframe.tolist(), kpatch.tolist()))
    knorm = kf[0].norm(dim=-1)  # [H, N]
    State.keynorm.append({"layer": layer, "step": step, "key_norm_all": float(knorm.mean()),
                          "key_norm_current": float(knorm[:, past:].mean())})


def install_patches(model, collect: bool) -> None:
    import streamvggt.layers.attention as attn_mod
    import streamvggt.models.aggregator as agg_mod
    from streamvggt.layers.eviction import EvictionManager

    def forbid(name):
        def _raise(*args, **kwargs):
            raise RuntimeError(f"{name} was called: eviction must not run in this experiment")
        return _raise

    attn_mod.Attention.eviction = forbid("Attention.eviction")
    EvictionManager.select = forbid("EvictionManager.select")

    original_alloc = agg_mod._allocate_layer_budgets_from_scores

    def checked_alloc(score_dict, capacities, total_budget, *args, **kwargs):
        assert all(float(s) == 0.0 for s in score_dict.values()), "layer scores are non-zero: KABA is active"
        assert float(kwargs.get("alpha", 0.0)) == 0.0, "alpha != 0: KABA is active"
        result = original_alloc(score_dict, capacities, total_budget, *args, **kwargs)
        budgets = result[0] if isinstance(result, tuple) else result
        # every layer may keep all of its live keys (the eviction gate is keys > budget)
        assert all(int(budgets[i]) >= int(c) for i, c in capacities.items()), "a layer budget is below its live keys"
        State.alloc_calls += 1
        return result

    agg_mod._allocate_layer_budgets_from_scores = checked_alloc

    if not collect:
        return
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, *args, **kwargs):
        out = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, *args, **kwargs)
        if State.layer is not None:
            assert attn_mask is None and dropout_p == 0.0 and not args and not kwargs, "unexpected SDPA arguments"
            with torch.no_grad():
                _record(q, k, v, out)
        return out

    proxy = types.ModuleType("F_attention_peak_proxy")
    proxy.__dict__.update(torch.nn.functional.__dict__)
    proxy.scaled_dot_product_attention = sdpa
    attn_mod.F = proxy

    for idx, block in enumerate(model.aggregator.global_blocks):
        assert block.attn.fused_attn, f"global block {idx} does not use the SDPA path"
        block.attn.register_forward_pre_hook(lambda m, a, i=idx: setattr(State, "layer", i))
        block.attn.register_forward_hook(lambda m, a, o: setattr(State, "layer", None))


def load_batch(args):
    from torch.utils.data._utils.collate import default_collate
    from eval.mv_recon.data import SevenScenes

    dataset = SevenScenes(split="test", ROOT=args.root, resolution=(518, 392), num_seq=1, full_video=True,
                          kf_every=args.kf_every, max_frames=args.max_frames)
    batch = default_collate([dataset[args.scene_index]])
    for view in batch:
        view["img"] = (view["img"] + 1.0) / 2.0  # as in eval/mv_recon/launch.py
    label = str(batch[0]["label"][0]).rsplit("/", 1)[0]
    frames = [str(view["instance"][0]) for view in batch]
    return batch, label, frames


def run(model, batch, eviction_policy):
    with torch.no_grad():
        results = model.inference(
            batch, total_budget=args.budget, eviction_policy=eviction_policy, stream_chunk_size=1,
            layer_budget_strategy="uniform", layer_budget_score_only=False, leverage_attention_utility=False,
            leverage_conf_gate=False, leverage_projected_key_cache=False)
    return results.ress


def outputs_equal(a, b) -> bool:
    for ra, rb in zip(a, b):
        for key, val in ra.items():
            if isinstance(val, torch.Tensor) and not torch.equal(val, rb[key]):
                print(f"  output mismatch: {key}")
                return False
    return len(a) == len(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(ROOT / "ckpt" / "checkpoints.pth"))
    parser.add_argument("--root", default="/home/dongjae/dataset/7scenes")
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--kf_every", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--budget", type=int, default=1_000_000_000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--verify", action="store_true",
                        help="also check A @ v == SDPA output, and that outputs do not depend on the patch or on "
                             "--eviction_policy (two extra unpatched runs)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump_inputs", default=None,
                        help="only save the preprocessed frames ([S, 3, H, W] in [0, 1]) to this .npz and exit "
                             "(input for tools/vggt_attention_peak.py)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run on CPU"

    sys.path.insert(0, str(Path(args.weights).resolve().parent))
    if args.dump_inputs:
        batch, label, frames = load_batch(args)
        images = torch.cat([view["img"] for view in batch], 0).float().numpy()
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_inputs)), exist_ok=True)
        np.savez(args.dump_inputs, images=images, frames=np.array(frames), label=label,
                 sha=__import__("hashlib").sha256(images.tobytes()).hexdigest())
        print(f"saved {args.dump_inputs}: {label}, images {images.shape}")
        sys.exit(0)
    from streamvggt.models.streamvggt import StreamVGGT

    t0 = time.time()
    model = StreamVGGT(total_budget=args.budget)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=True)
    model.eval()
    batch, label, frames = load_batch(args)
    State.tokens_per_frame = NUM_SPECIAL + (batch[0]["img"].shape[-2] // 14) * (batch[0]["img"].shape[-1] // 14)
    print(f"scene {label}, frames {frames}, tokens/frame {State.tokens_per_frame}, load {time.time() - t0:.0f}s")

    checks = {}
    if args.verify:  # reference runs: eviction guards on, no attention collection
        install_patches(model, collect=False)
        t1 = time.time()
        ref_mean = run(model, batch, "mean")
        ref_lev = run(model, batch, "svd_leverage")
        checks["eviction_policy_invariant"] = outputs_equal(ref_mean, ref_lev)
        print(f"reference runs {time.time() - t1:.0f}s; mean vs svd_leverage identical: "
              f"{checks['eviction_policy_invariant']}")
    install_patches(model, collect=True)
    t1 = time.time()
    State.verify = args.verify
    preds = run(model, batch, "mean")
    print(f"analysis run {time.time() - t1:.0f}s")
    if args.verify:
        checks["patch_invariant"] = outputs_equal(ref_mean, preds)
        checks["max_abs_diff_softmax_v_vs_sdpa"] = State.max_out_diff
    T = len(batch)
    checks.update({
        "global_calls_per_step": {int(s): c for s, c in sorted(State.calls.items())},
        "calls_ok": sorted(State.calls) == list(range(1, T + 1)) and all(c == 24 for c in State.calls.values()),
        "max_abs_rowsum_minus_1": State.max_rowsum_err,
        "allocator_calls_checked": State.alloc_calls,
        "eviction_calls": 0,  # any call would have raised
    })

    import pandas as pd

    out = Path(args.out) / f"{args.scene_index:04d}_{label.replace('/', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(State.rows)
    rows.to_csv(out / "attn_peak_layers.csv", index=False)
    top = pd.DataFrame(State.top50)
    union = []
    for layer, entries in sorted(State.union.items()):
        best = sorted(entries, reverse=True)[:TOP_AVGGT]
        v = np.array([e[0] for e in best])
        qi, kf, kp, st = (np.array([e[i] for e in best]) for i in (2, 3, 4, 1))
        union.append({"layer": layer, "step": "union", "n_keys": T * State.tokens_per_frame,
                      "top50_mean": v.mean(), "top50_min": v.min(), "top50_max": v.max(),
                      "frac_self": float(((kf == st - 1) & (kp == qi)).mean()),
                      "frac_same_position": float(((kf != st - 1) & (kp == qi)).mean()),
                      "frac_cross_frame": float((kf != st - 1).mean())})
    pd.concat([top, pd.DataFrame(union)], ignore_index=True).to_csv(out / "attn_peak_top50.csv", index=False)
    pd.DataFrame(State.keynorm).to_csv(out / "key_norm.csv", index=False)
    checks["k_shape_equals_step_x_tokens"] = bool((rows["n_keys"] == rows["step"] * State.tokens_per_frame).all())
    checks["first_frame_no_past"] = bool(rows[rows.step == 1]["cross_frame_mass"].isna().all())
    imgs = torch.cat([view["img"] for view in batch], 0).float().numpy()
    meta = {"scene": label, "frames": frames, "input_sha256": __import__("hashlib").sha256(imgs.tobytes()).hexdigest(), "tokens_per_frame": State.tokens_per_frame, "kf_every": args.kf_every,
            "max_frames": args.max_frames, "budget": args.budget, "dtype": "float32 (CPU)",
            "eviction": "disabled (never called)", "layer_budget_strategy": "uniform (KABA off)", "checks": checks}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(checks, indent=2))
    print(f"saved {out}")
