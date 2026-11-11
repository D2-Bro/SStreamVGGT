#!/usr/bin/env python3
"""Same attention statistics as tools/attention_peak_analysis.py, on the official offline VGGT (VGGT-1B).

Input: the exact preprocessed frames StreamVGGT saw (attention_peak_analysis.py --dump_inputs), so both models get
bit-identical image tensors ([S, 3, 392, 518] in [0, 1]; both normalise with the same ResNet mean/std inside).
VGGT's global attention is non-causal over all S frames (N = S * 1041 keys for every query).

Statistics per global layer (head-averaged attention unless a head is given):
  top50 scope "full"      AVGGT: patch queries x patch keys of the whole matrix, top-50 entries (VGGT's own setting)
  top50 scope "causal"    the same restricted to key frame <= query frame (the region StreamVGGT can attend to)
  top50 scope "last"      queries of the last frame only, all patch keys (matches StreamVGGT's last step)
  rows                    per query frame f (and all frames): per-row max, top-k, entropy / log N, past-frame mass
                          (key frame < f), other-frame mass (key frame != f), special-key mass; per head and head-avg.

The official code is used as is (cloned, not installed): only vggt.layers.attention.F is replaced in this process by a
proxy whose scaled_dot_product_attention returns the original output unchanged.

Usage (CPU):
  CUDA_VISIBLE_DEVICES= python tools/vggt_attention_peak.py --inputs 'dir/scene*.npz' --out <dir> [--verify]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_peak_analysis as ap  # noqa: E402  (statistics helpers; imports nothing named vggt)

TOP = 50
HEAD_CHUNK = 4


class S:
    layer = None
    calls = 0
    rows, top50, keynorm = [], [], []
    verify = False
    max_out_diff = 0.0
    max_rowsum_err = 0.0
    P = None


def _top_entries(mat, mask_value=None):
    vals, flat = mat.flatten().topk(TOP)
    return vals.numpy(), (flat // mat.shape[1]).numpy(), (flat % mat.shape[1]).numpy()


def _record(q, k, v, out):
    layer, P = S.layer, S.P
    _, H, N, D = q.shape
    F_ = N // P
    assert N % P == 0 and q.shape[2] == N
    S.calls += 1
    key_frame = torch.arange(N) // P
    special_key = (torch.arange(N) % P) < ap.NUM_SPECIAL
    qf, kf, vf = q.float(), k.float(), v.float()
    a_mean = torch.zeros(N, N)

    def frame_rows(a2d, head):
        for f in range(F_):
            blk = a2d[f * P:(f + 1) * P]
            for qt, sl in (("all", slice(0, P)), ("patch", slice(ap.NUM_SPECIAL, P)), ("special", slice(0, ap.NUM_SPECIAL))):
                rows = blk[sl]
                m = ap._row_metrics(rows, N, f * P, special_key)  # cross_frame_mass = keys of earlier frames
                m["past_frame_mass"] = m.pop("cross_frame_mass")
                m["other_frame_mass"] = float(1 - rows[:, f * P:(f + 1) * P].sum(-1).mean())
                S.rows.append({"layer": layer, "query_frame": f, "n_keys": N, "head": head, "query_type": qt, **m})

    for h0 in range(0, H, HEAD_CHUNK):
        a = torch.softmax(qf[0, h0:h0 + HEAD_CHUNK] @ kf[0, h0:h0 + HEAD_CHUNK].transpose(-1, -2) / math.sqrt(D), -1)
        S.max_rowsum_err = max(S.max_rowsum_err, float((a.sum(-1) - 1).abs().max()))
        if S.verify:
            S.max_out_diff = max(S.max_out_diff, float((a @ vf[0, h0:h0 + HEAD_CHUNK] - out[0, h0:h0 + HEAD_CHUNK].float()).abs().max()))
        a_mean += a.sum(0)
        for i in range(a.shape[0]):
            frame_rows(a[i], h0 + i)
        del a
    a_mean /= H
    frame_rows(a_mean, -1)

    pk = torch.nonzero(~special_key).squeeze(1)  # patch keys
    pq = pk  # patch queries (same index set: queries are all tokens of all frames)
    sub = a_mean[pq][:, pk]  # [S*1036, S*1036]
    qframe = (pq // P).numpy()
    kframe_all = (pk // P).numpy()
    qpatch = (pq % P - ap.NUM_SPECIAL).numpy()
    kpatch_all = (pk % P - ap.NUM_SPECIAL).numpy()
    scopes = {
        "full": sub,
        "causal": sub.masked_fill(torch.from_numpy(kframe_all[None, :] > qframe[:, None]), -1.0),
        "last": sub.masked_fill(torch.from_numpy(np.broadcast_to(qframe[:, None] != F_ - 1, sub.shape).copy()), -1.0),
    }
    for scope, mat in scopes.items():
        vals, qi, kj = _top_entries(mat)
        qf_, kf_ = qframe[qi], kframe_all[kj]
        same_patch = qpatch[qi] == kpatch_all[kj]
        S.top50.append({"layer": layer, "scope": scope, "n_keys": N,
                        "top50_mean": float(vals.mean()), "top50_min": float(vals.min()), "top50_max": float(vals.max()),
                        "frac_self": float((same_patch & (qf_ == kf_)).mean()),
                        "frac_same_position": float((same_patch & (qf_ != kf_)).mean()),
                        "frac_cross_frame": float((qf_ != kf_).mean())})
    knorm = kf[0].norm(dim=-1)  # [H, N], per-head (64-d) norm
    tok = kf[0].square().sum(dim=(0, 2)).sqrt()  # [N], heads concatenated (1024-d): KABA's key-norm term
    S.keynorm.append({"layer": layer, "key_norm_all": float(knorm.mean()), "key_token_norm_all": float(tok.mean()),
                      "key_token_norm_last": float(tok[(F_ - 1) * P:].mean()),
                      "key_token_norm_patch": float(tok[~special_key].mean()),
                      "key_token_norm_special": float(tok[special_key].mean()), "key_token_norm_std": float(tok.std())})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vggt_root", default="/home/dongjae/vggt_official")
    p.add_argument("--weights", default=None, help="default: <vggt_root>/ckpt/model.pt (facebook/VGGT-1B)")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--verify", action="store_true", help="also run unpatched and check outputs and A @ v")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run on CPU"

    # the official package has no __init__.py (namespace package), so src/vggt would shadow it: drop src first
    src = str((Path(__file__).resolve().parents[1] / "src").resolve())
    sys.path[:] = [x for x in sys.path if os.path.abspath(x or ".") != src]
    sys.path.insert(0, a.vggt_root)
    for name in [m for m in sys.modules if m == "vggt" or m.startswith("vggt.")]:
        del sys.modules[name]
    import vggt
    import vggt.layers.attention as vattn
    from vggt.models.vggt import VGGT
    assert Path(vattn.__file__).resolve().is_relative_to(Path(a.vggt_root).resolve()), vattn.__file__

    model = VGGT()
    weights = a.weights or os.path.join(a.vggt_root, "ckpt", "model.pt")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    agg = model.aggregator
    commit = os.popen(f"git -C {a.vggt_root} rev-parse HEAD").read().strip()

    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kwargs):
        out = original_sdpa(q, k, v, *args, **kwargs)
        if S.layer is not None:
            assert not args and kwargs.get("attn_mask") is None and kwargs.get("dropout_p", 0.0) == 0.0
            with torch.no_grad():
                _record(q, k, v, out)
        return out

    proxy = types.ModuleType("F_vggt_attention_peak_proxy")
    proxy.__dict__.update(torch.nn.functional.__dict__)
    proxy.scaled_dot_product_attention = sdpa
    for idx, block in enumerate(agg.global_blocks):
        assert block.attn.fused_attn
        block.attn.register_forward_pre_hook(lambda m, x, i=idx: setattr(S, "layer", i))
        block.attn.register_forward_hook(lambda m, x, o: setattr(S, "layer", None))

    import pandas as pd

    for path in sorted(glob.glob(a.inputs)):
        z = np.load(path)
        images = torch.from_numpy(z["images"]).float()
        assert hashlib.sha256(z["images"].tobytes()).hexdigest() == str(z["sha"])
        label = str(z["label"])
        idx = int(Path(path).stem.replace("scene", ""))
        out_dir = Path(a.out) / f"{idx:04d}_{label.replace('/', '_')}"
        S.P = ap.NUM_SPECIAL + (images.shape[-2] // 14) * (images.shape[-1] // 14)
        S.rows, S.top50, S.keynorm, S.calls, S.max_out_diff, S.max_rowsum_err = [], [], [], 0, 0.0, 0.0
        t0 = time.time()
        checks = {}
        with torch.no_grad():
            if a.verify:
                vattn.F = torch.nn.functional
                ref, _ = agg(images[None])
            vattn.F = proxy
            S.verify = a.verify
            got, _ = agg(images[None])
        if a.verify:
            checks["patch_invariant"] = all((r is None and g is None) or torch.equal(r, g) for r, g in zip(ref, got))
            checks["max_abs_diff_softmax_v_vs_sdpa"] = S.max_out_diff
        checks.update({"global_calls": S.calls, "calls_ok": S.calls == len(agg.global_blocks),
                       "max_abs_rowsum_minus_1": S.max_rowsum_err})
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(S.rows).to_csv(out_dir / "vggt_rows.csv", index=False)
        pd.DataFrame(S.top50).to_csv(out_dir / "vggt_top50.csv", index=False)
        pd.DataFrame(S.keynorm).to_csv(out_dir / "key_norm.csv", index=False)
        meta = {"scene": label, "frames": [str(f) for f in z["frames"]], "input_sha256": str(z["sha"]),
                "tokens_per_frame": S.P, "model": "facebook/VGGT-1B", "code": f"facebookresearch/vggt@{commit}",
                "dtype": "float32 (CPU)", "attention": "non-causal over all frames", "checks": checks}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{label}: {time.time() - t0:.0f}s {json.dumps(checks)}", flush=True)


if __name__ == "__main__":
    main()
