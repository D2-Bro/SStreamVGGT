"""CPU reconstruction metrics shared by inline and deferred evaluation."""

import os
from pathlib import Path

import numpy as np
import open3d as o3d

from eval.mv_recon.utils import accuracy, completion


THRESHOLDS = (0.05, 0.1, 0.25)


def save_eval_inputs(path, pred, gt, masks, *, scene_id, dataset, rank, infer_time, fps, use_proj):
    """Write one complete scene atomically so an interrupted run is detectable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez(
            temporary, pred=pred, gt=gt, masks=masks,
            scene_id=scene_id, dataset=dataset, rank=rank,
            infer_time=infer_time, fps=fps, use_proj=use_proj,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_arrays(pred, gt, masks, dataset, use_proj=False):
    """Run the legacy point-to-point ICP and nearest-neighbor metrics on CPU."""
    if pred.shape != gt.shape or pred.shape[:-1] != masks.shape or pred.shape[-1] != 3:
        raise ValueError(f"Incompatible point/mask shapes: {pred.shape}, {gt.shape}, {masks.shape}")
    selected = masks > 0
    pred = pred[selected].reshape(-1, 3)
    gt = gt[selected].reshape(-1, 3)
    paired_finite = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if use_proj:
        # Projection points require a Sim(3) alignment before ICP.
        src, dst = pred[paired_finite].astype(np.float64), gt[paired_finite].astype(np.float64)
        if len(src) < 3:
            raise ValueError("Too few paired finite points for Sim(3) alignment")
        mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
        src_c, dst_c = src - mu_src, dst - mu_dst
        u, singular, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
        sign = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            sign[-1, -1] = -1
        rotation = u @ sign @ vt
        scale = (singular * sign.diagonal()).sum() / np.mean(np.sum(src_c ** 2, axis=1))
        pred = (scale * (rotation @ pred.T)).T + mu_dst - scale * rotation @ mu_src
    pred = pred[np.isfinite(pred).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    if len(pred) == 0 or len(gt) == 0:
        raise ValueError("No finite points remain after masking")

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pred)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(gt)
    threshold = 100 if "DTU" in dataset else 0.1
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    source.transform(reg.transformation)
    acc, acc_med, nc1, nc1_med, precisions = accuracy(
        target.points, source.points, dist_ths=THRESHOLDS,
    )
    comp, comp_med, nc2, nc2_med, recalls = completion(
        target.points, source.points, dist_ths=THRESHOLDS,
    )
    metrics = dict(acc=acc, comp=comp, nc1=nc1, nc2=nc2,
                   acc_med=acc_med, comp_med=comp_med,
                   nc1_med=nc1_med, nc2_med=nc2_med)
    for threshold in THRESHOLDS:
        precision, recall = precisions[threshold], recalls[threshold]
        denominator = precision + recall
        metrics[f"f1_{round(threshold * 100)}cm"] = (
            2 * precision * recall / denominator if denominator > 0 else 0.0
        )
    return metrics


def format_metric(scene_id, m, infer_time, fps):
    return (
        f"Idx: {scene_id}, "
        f"Acc: {m['acc']}, Comp: {m['comp']}, NC1: {m['nc1']}, NC2: {m['nc2']} - "
        f"Acc_med: {m['acc_med']}, Compc_med: {m['comp_med']}, "
        f"NC1c_med: {m['nc1_med']}, NC2c_med: {m['nc2_med']}, "
        f"F1@5cm: {m['f1_5cm']:.6f}, F1@10cm: {m['f1_10cm']:.6f}, "
        f"F1@25cm: {m['f1_25cm']:.6f}, "
        f"Time: {infer_time}, FPS: {fps}, "
    )
