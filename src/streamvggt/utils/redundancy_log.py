"""Cache redundancy statistics for the leverage participation-ratio (psi) analysis.

psi = (sum l)^2 / sum l^2 cannot tell n identical keys (rank 1, l_i = 1/n, psi = n) from n keys on n orthogonal
directions (l_i = 1, psi = n). This log measures, at each eviction, how far the actual cache is from that
degenerate case: token-to-token cosine similarity and the (effective) rank of the key matrix, for the candidate set
before eviction and the retained set after it.

Keys are represented as IDC scores them: post-RoPE keys, each head normalized to unit length, heads concatenated
(then divided by sqrt(H) so that a row is a unit vector and a dot product is the head-averaged cosine). Anchor
(first-frame) tokens are excluded. One CSV per scene, one row per (step, layer, space, set), space = full (1024-d) or sketch (IDC's
right-sketched 256-d matrix).

Enabled from a launcher with configure(); attention.eviction calls record() after choosing the kept tokens.
"""

import csv
import math
import os
from typing import Optional, Sequence

import torch

_CONFIG = {"dir": None, "every": 10, "taus": (0.5, 0.7, 0.9), "scene": None, "path": None}

FIELDS_BASE = [
    "step", "layer", "space", "set", "n_tokens", "head_shared", "psi", "psi_over_n", "rank", "erank", "erank_over_n",
    "stable_rank", "mean_cos",
]


def configure(log_dir: Optional[str], every: int = 10, taus: Sequence[float] = (0.5, 0.7, 0.9)):
    _CONFIG.update(dir=log_dir, every=max(int(every), 1), taus=tuple(float(t) for t in taus))


def set_scene(scene_key: Optional[str]):
    """Start a new CSV for this scene (overwrites an earlier file of the same scene)."""
    _CONFIG["scene"] = scene_key
    _CONFIG["path"] = None
    if _CONFIG["dir"] and scene_key:
        os.makedirs(_CONFIG["dir"], exist_ok=True)
        _CONFIG["path"] = os.path.join(_CONFIG["dir"], f"{scene_key}.csv")
        with open(_CONFIG["path"], "w", newline="") as f:
            csv.writer(f).writerow(_fields())


def enabled(step_idx: Optional[int]) -> bool:
    return _CONFIG["path"] is not None and step_idx is not None and int(step_idx) % _CONFIG["every"] == 0


def _fields():
    taus = [f"frac_cos_gt_{t:g}" for t in _CONFIG["taus"]]
    return FIELDS_BASE + taus + ["nn_cos_mean", "nn_cos_p10", "nn_cos_p50", "nn_cos_p90"]


def key_features(k_tokens: torch.Tensor) -> torch.Tensor:
    """[H, N, D] keys -> [N, H*D] float64 unit rows (per-head unit norm, concatenated, / sqrt(H))."""
    H = k_tokens.shape[0]
    x = torch.nn.functional.normalize(k_tokens.double(), p=2, dim=-1)
    return x.permute(1, 0, 2).reshape(k_tokens.shape[1], -1) / math.sqrt(H)


def set_statistics(x: torch.Tensor, taus: Sequence[float], chunk: int = 4096,
                   spectral: Optional[torch.Tensor] = None) -> dict:
    """Statistics of a set of unit rows x [N, d] (float64). Rank, effective rank, stable rank and psi use
    ``spectral`` (same rows, any scale; default x), e.g. the raw sketched matrix IDC computes leverage on."""
    n = x.shape[0]
    out = {"n_tokens": n}
    m = x if spectral is None else spectral
    # spectrum: Gram in feature space (d x d), eigenvalues = squared singular values (one eigendecomposition)
    evals, evecs = torch.linalg.eigh(m.T @ m)
    evals = evals.clamp_min(0.0)
    eig = evals.flip(0)
    total = float(eig.sum())
    tol = float(eig[0]) * max(m.shape) * torch.finfo(torch.float64).eps if total > 0 else 0.0
    rank = int((eig > tol).sum())
    p = eig[eig > tol] / total
    erank = float(torch.exp(-(p * p.log()).sum())) if total > 0 else 0.0
    out.update(rank=rank, erank=erank, erank_over_n=erank / n, stable_rank=total / float(eig[0]) if total > 0 else 0.0)
    # exact leverage on the same rows: l_i = m_i (M^T M)^+ m_i^T over the numerical range
    keep = evals > tol
    proj = m @ evecs[:, keep]
    lev = (proj.square() / evals[keep]).sum(dim=1)
    psi = float(lev.sum() ** 2 / lev.square().sum()) if float(lev.square().sum()) > 0 else 0.0
    out.update(psi=psi, psi_over_n=psi / n)
    # cosine: exact mean over pairs, threshold fractions and nearest-neighbour cosine over chunks
    s = x.sum(dim=0)
    out["mean_cos"] = float((s @ s - n) / (n * (n - 1))) if n > 1 else float("nan")
    counts = torch.zeros(len(taus), dtype=torch.float64, device=x.device)
    nn = torch.empty(n, dtype=torch.float64, device=x.device)
    taus_t = torch.tensor(list(taus), dtype=torch.float64, device=x.device)
    for i in range(0, n, chunk):
        c = x[i:i + chunk] @ x.T
        idx = torch.arange(i, min(i + chunk, n), device=x.device)
        c[idx - i, idx] = -2.0  # drop self-pairs
        counts += (c.unsqueeze(0) > taus_t.view(-1, 1, 1)).sum(dim=(1, 2)).double()
        nn[i:i + chunk] = c.max(dim=1).values
    pairs = n * (n - 1)
    for t, cnt in zip(taus, counts.tolist()):
        out[f"frac_cos_gt_{t:g}"] = cnt / pairs if pairs else float("nan")
    if n > 1:
        q = torch.quantile(nn.float(), torch.tensor([0.1, 0.5, 0.9], device=x.device))
        out.update(nn_cos_mean=float(nn.mean()), nn_cos_p10=float(q[0]), nn_cos_p50=float(q[1]),
                   nn_cos_p90=float(q[2]))
    return out


_WARNED_PER_HEAD = [False]


@torch.no_grad()
def record(k: torch.Tensor, num_anchor_tokens: int, kept_candidate_indices: torch.Tensor,
           layer_id: Optional[int], step_idx: Optional[int], per_head_cache: bool = False,
           sketch_features: Optional[torch.Tensor] = None):
    """k: pre-eviction cache [B, H, N, D]; kept_candidate_indices: [B, H, K] indices into the candidates.

    Skipped for per-head caches (compactor policy): there, the same cache position holds a different token in
    each head, so the head-concatenated representation is not defined.
    """
    if not enabled(step_idx):
        return
    if per_head_cache:
        if not _WARNED_PER_HEAD[0]:
            print("[redundancy_log] per-head cache (compactor policy): redundancy statistics are not logged")
            _WARNED_PER_HEAD[0] = True
        return
    cand = k[0, :, int(num_anchor_tokens):, :]
    kept = kept_candidate_indices[0]
    head_shared = bool((kept == kept[:1]).all())
    x = key_features(cand)
    spaces = [("full", x, None)]
    # IDC's own sketch space: the right-sketched candidate matrix Z [Nc, r] its leverage scores come from.
    if sketch_features is not None and sketch_features.shape[1] == x.shape[0]:
        z = sketch_features[0].double()
        spaces.append(("sketch", torch.nn.functional.normalize(z, p=2, dim=-1), z))
    rows = []
    for space, feats, spectral in spaces:
        for name, idx in (("candidates", None), ("retained", kept[0])):
            f = feats if idx is None else feats[idx]
            sp = None if spectral is None else (spectral if idx is None else spectral[idx])
            stats = set_statistics(f, _CONFIG["taus"], spectral=sp)
            stats.update(step=int(step_idx), layer=int(layer_id) if layer_id is not None else -1, space=space,
                         set=name, head_shared=int(head_shared))
            rows.append(stats)
    with open(_CONFIG["path"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fields(), extrasaction="ignore")
        for row in rows:
            writer.writerow(row)
