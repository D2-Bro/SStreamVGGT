"""GPU version of the legacy ICP + nearest-neighbor distances in cpu_evaluation.evaluate_arrays.

Reproduces, in float64 on CUDA:
  ICP   open3d.pipelines.registration.registration_icp (legacy, point-to-point, identity init, default
        ICPConvergenceCriteria): correspondences = nearest target point within the threshold; update =
        Eigen::umeyama without scaling (Kabsch) on them; stop when |d fitness| < 1e-6 and |d inlier_rmse| < 1e-6,
        at most 30 iterations.
  NN    exact nearest-neighbor distances both ways (eval.mv_recon.utils.accuracy / completion use pykdtree).
Nearest-neighbor searches use open3d.core.nns hybrid (radius) search at growing radii: the nearest point within a
radius is the nearest overall, so only the points with no neighbor within one radius go on to the next. Small radii
first keep the hash-grid cells sparse (one 0.1 m search on dense clouds is ~8x slower). Points with no neighbor
within the ICP threshold (outliers, typically a few %) get their distance from pykdtree on the CPU: larger radii
make the grid cells so dense that a few thousand far points take longer than building the KD-tree. Searches are batched so that GPU memory does
not grow with the cloud size. Same approach as oxford_eval recon/recon_eval.py (_icp_and_metrics), without the voxel
downsampling.
"""

import numpy as np
import open3d.core as o3c
import torch
import torch.utils.dlpack
from pykdtree.kdtree import KDTree

MAX_ITERATION = 30
RELATIVE_FITNESS = 1e-6
RELATIVE_RMSE = 1e-6
ICP_RADIUS_SCALES = (0.25, 1.0)             # multiples of the ICP threshold; the last one is the threshold
NN_RADIUS_SCALES = (0.25, 1.0)             # 0.1 m threshold -> 0.025, 0.1 m, then pykdtree for the rest
BATCH = 2_000_000


def _devices(device):
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"GPU evaluation needs a CUDA device, got {device}")
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    return torch.device("cuda", index), o3c.Device(f"CUDA:{index}")


def _to_o3(x):
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))


def _from_o3(x):
    return torch.utils.dlpack.from_dlpack(x.to_dlpack())


def _transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def _radius_index(ref, radius):
    index = o3c.nns.NearestNeighborSearch(_to_o3(ref))
    if not index.hybrid_index(radius):
        raise RuntimeError(f"open3d hybrid index failed (radius {radius})")
    return index


def _search(index, radius, query):
    """Nearest ref point within radius: (found mask, ref index, squared distance), torch on the query's device."""
    j, d2, cnt = index.hybrid_search(_to_o3(query), radius, 1)
    return _from_o3(cnt).reshape(-1) > 0, _from_o3(j).reshape(-1).long(), _from_o3(d2).reshape(-1)


def _nearest_within(indices, query):
    """indices: [(radius, index)] with growing radii. Returns (ref index or -1, squared distance or 0)."""
    ind = torch.full((len(query),), -1, dtype=torch.int64, device=query.device)
    d2 = torch.zeros(len(query), dtype=query.dtype, device=query.device)
    rest = torch.arange(len(query), device=query.device)
    for radius, index in indices:
        if len(rest) == 0:
            break
        ok, j, dd = _search(index, radius, query[rest])
        ind[rest[ok]], d2[rest[ok]] = j[ok], dd[ok]
        rest = rest[~ok]
    return ind, d2


def _icp(src, tgt, threshold, dev):
    """Legacy point-to-point ICP; src/tgt float64 (N, 3) on dev. Returns the 4x4 transformation (torch, dev)."""
    indices = [(threshold * s, _radius_index(tgt, threshold * s)) for s in ICP_RADIUS_SCALES]
    n_src = len(src)

    def correspond(T):  # nearest target index (-1: none within threshold), fitness, inlier_rmse
        ind = torch.empty(n_src, dtype=torch.int64, device=dev)
        err2 = torch.zeros((), dtype=torch.float64, device=dev)
        for i in range(0, n_src, BATCH):
            j, d2 = _nearest_within(indices, _transform(src[i:i + BATCH], T))
            ind[i:i + BATCH] = j
            err2 += d2.sum()
        n = int((ind >= 0).sum())
        return ind, n / n_src, (float(err2) / n) ** 0.5 if n else 0.0

    def kabsch(T, ind):  # Eigen::umeyama(source, target, with_scaling=false) on the correspondences
        ok = ind >= 0
        n = int(ok.sum())
        if n == 0:
            return torch.eye(4, dtype=torch.float64, device=dev)
        s_sum = torch.zeros(3, dtype=torch.float64, device=dev)
        t_sum = torch.zeros(3, dtype=torch.float64, device=dev)
        for i in range(0, n_src, BATCH):
            k, m = ind[i:i + BATCH], ok[i:i + BATCH]
            s_sum += _transform(src[i:i + BATCH][m], T).sum(0)
            t_sum += tgt[k[m]].sum(0)
        mu_s, mu_t = s_sum / n, t_sum / n
        sigma = torch.zeros(3, 3, dtype=torch.float64, device=dev)
        for i in range(0, n_src, BATCH):
            k, m = ind[i:i + BATCH], ok[i:i + BATCH]
            sigma += (tgt[k[m]] - mu_t).T @ (_transform(src[i:i + BATCH][m], T) - mu_s)
        u, _, vt = np.linalg.svd((sigma / n).cpu().numpy())
        sign = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            sign[2, 2] = -1
        rotation = u @ sign @ vt
        update = np.eye(4)
        update[:3, :3] = rotation
        update[:3, 3] = mu_t.cpu().numpy() - rotation @ mu_s.cpu().numpy()
        return torch.from_numpy(update).to(dev)

    T = torch.eye(4, dtype=torch.float64, device=dev)
    ind, fitness, rmse = correspond(T)
    for _ in range(MAX_ITERATION):
        T = kabsch(T, ind) @ T
        previous = (fitness, rmse)
        ind, fitness, rmse = correspond(T)
        if abs(previous[0] - fitness) < RELATIVE_FITNESS and abs(previous[1] - rmse) < RELATIVE_RMSE:
            break
    return T


def _nn_distances(query, ref, radii):
    """Exact distance from each query point to its nearest ref point; query/ref float64 torch on the GPU."""
    dist = torch.full((len(query),), float("nan"), dtype=torch.float64, device=query.device)
    rest = torch.arange(len(query), device=query.device)
    for radius in radii:  # one index at a time to bound GPU memory
        if len(rest) == 0:
            break
        index = _radius_index(ref, radius)
        found = torch.zeros(len(rest), dtype=torch.bool, device=query.device)
        for i in range(0, len(rest), BATCH):
            sel = rest[i:i + BATCH]
            ok, _, d2 = _search(index, radius, query[sel])
            dist[sel[ok]] = d2[ok].sqrt()
            found[i:i + BATCH] = ok
        rest = rest[~found]
        del index
        o3c.cuda.release_cache()
    dist = dist.cpu().numpy()
    if len(rest):  # nothing within the largest radius: exact KD-tree on the CPU (same as utils.accuracy)
        rest = rest.cpu().numpy()
        dist[rest] = KDTree(ref.cpu().numpy()).query(query[torch.from_numpy(rest).to(query.device)].cpu().numpy(), k=1)[0]
    return dist


def icp_and_distances(pred, gt, threshold, device="cuda"):
    """pred/gt: float64 (N, 3) numpy arrays (cpu_evaluation.prepare_points). Returns (pred->gt, gt->pred) distances
    after ICP, i.e. the distances used for Acc (and precision) and Comp (and recall)."""
    dev, _ = _devices(device)
    with torch.cuda.device(dev):
        try:
            src = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float64)).to(dev)
            tgt = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float64)).to(dev)
            T = _icp(src, tgt, threshold, dev)
            o3c.cuda.release_cache()
            aligned = _transform(src, T).contiguous()
            del src
            radii = tuple(threshold * s for s in NN_RADIUS_SCALES)
            acc_dist = _nn_distances(aligned, tgt, radii)
            comp_dist = _nn_distances(tgt, aligned, radii)
            del aligned, tgt
        finally:
            o3c.cuda.release_cache()
            torch.cuda.empty_cache()
    return acc_dist, comp_dist
