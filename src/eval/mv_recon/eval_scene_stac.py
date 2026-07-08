import numpy as np
import open3d as o3d
import torch

from dust3r.utils.geometry import geotrf
from eval.mv_recon.criterion import L21, Regr3D_t_ScaleShiftInv
from eval.mv_recon.utils import accuracy, completion


def _to_cpu_view(view):
    return {
        key: value.to("cpu") if isinstance(value, torch.Tensor) else value
        for key, value in view.items()
    }


def _to_cpu_pred(pred):
    return {
        key: value.to("cpu") if isinstance(value, torch.Tensor) else value
        for key, value in pred.items()
    }


def _as_batched_tensor(array_or_tensor):
    if isinstance(array_or_tensor, torch.Tensor):
        tensor = array_or_tensor.detach().cpu()
    else:
        tensor = torch.from_numpy(np.asarray(array_or_tensor)).cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _make_projected_preds(point_map_by_unprojection, depth_conf, num_frames):
    preds = []
    for idx in range(num_frames):
        pred = {
            "pts3d_in_other_view": _as_batched_tensor(point_map_by_unprojection[idx]),
        }
        if depth_conf is not None:
            pred["conf"] = _as_batched_tensor(depth_conf[0, idx])
        preds.append(pred)
    return preds


def _filtered_points(points, gt_points, colors):
    points = points.reshape(-1, 3)
    gt_points = gt_points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(gt_points).all(axis=1)
    return points[finite], gt_points[finite], colors[finite]


def _to_numpy(array_or_tensor):
    if isinstance(array_or_tensor, torch.Tensor):
        return array_or_tensor.detach().cpu().numpy()
    return np.asarray(array_or_tensor)


def eval_scene_stac(
    batch,
    preds,
    dataset_name,
    *,
    eval_frame_stride=1,
    icp_voxel_size=0.0,
    point_map_by_unprojection=None,
    depth_conf=None,
    use_gpu=False,
):
    """STAC-style reconstruction eval for SStreamVGGT prediction payloads.

    This keeps SStreamVGGT's model output format, but applies the STAC eval_scene
    point preparation: ScaleShiftInv alignment, GT z-shift restoration, and
    first-camera coordinate transform before ICP + Acc/Comp/NC metrics.
    """
    if eval_frame_stride < 1:
        raise ValueError(f"eval_frame_stride must be >= 1, got {eval_frame_stride}")

    batch_cpu = [_to_cpu_view(view) for view in batch]
    if point_map_by_unprojection is not None:
        eval_preds = _make_projected_preds(
            point_map_by_unprojection,
            depth_conf,
            len(batch_cpu),
        )
    else:
        eval_preds = [_to_cpu_pred(pred) for pred in preds]

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    gt_pts, pred_pts, _, _, masks, monitoring = criterion.get_all_pts3d_t(
        batch_cpu,
        eval_preds,
    )

    gt_shift_z = monitoring["gt_shift_z"]
    in_camera1 = None
    pts_all = []
    pts_gt_all = []
    images_all = []
    masks_all = []
    eval_frame_count = 0

    for frame_idx, view in enumerate(batch_cpu):
        if frame_idx % eval_frame_stride != 0:
            continue
        eval_frame_count += 1

        if in_camera1 is None:
            in_camera1 = view["camera_pose"][0].cpu()

        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
        valid_mask = view["valid_mask"].cpu().numpy()[0]

        pts = pred_pts[frame_idx].detach().cpu().numpy()[0]
        pts_gt = gt_pts[frame_idx].detach().cpu().numpy()[0]

        # STAC eval_scene restores the GT depth shift to both clouds and moves
        # them into the same first-camera coordinate system before ICP.
        pts[..., -1] += gt_shift_z.cpu().numpy().item()
        pts = _to_numpy(geotrf(in_camera1, pts))
        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
        pts_gt = _to_numpy(geotrf(in_camera1, pts_gt))

        images_all.append((image[None, ...] + 1.0) / 2.0)
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(valid_mask[None, ...])

    if eval_frame_count == 0:
        raise ValueError("No frames selected for STAC-style reconstruction eval")

    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)

    pts_all_masked = pts_all[masks_all > 0]
    pts_gt_all_masked = pts_gt_all[masks_all > 0]
    images_all_masked = images_all[masks_all > 0]
    pts_all_masked, pts_gt_all_masked, images_all_masked = _filtered_points(
        pts_all_masked,
        pts_gt_all_masked,
        images_all_masked,
    )
    if pts_all_masked.shape[0] == 0 or pts_gt_all_masked.shape[0] == 0:
        raise ValueError("No valid finite points available for reconstruction eval")

    threshold = 100 if "DTU" in dataset_name else 0.1
    if use_gpu:
        try:
            import cupoch as cph
        except ImportError as exc:
            raise ImportError(
                "--eval_gpu requires cupoch, but cupoch is not importable in this environment"
            ) from exc

        pcd = cph.geometry.PointCloud()
        pcd.points = cph.utility.Vector3fVector(pts_all_masked.astype(np.float32))
        pcd.colors = cph.utility.Vector3fVector(images_all_masked.astype(np.float32))

        pcd_gt = cph.geometry.PointCloud()
        pcd_gt.points = cph.utility.Vector3fVector(pts_gt_all_masked.astype(np.float32))
        pcd_gt.colors = cph.utility.Vector3fVector(images_all_masked.astype(np.float32))

        trans_init = np.eye(4, dtype=np.float32)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()
        reg_p2p = cph.registration.registration_icp(
            pcd,
            pcd_gt,
            threshold,
            trans_init,
            cph.registration.TransformationEstimationPointToPlane(),
            cph.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        pcd.transform(reg_p2p.transformation)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        gt_points = np.asarray(pcd_gt.points.cpu())
        rec_points = np.asarray(pcd.points.cpu())
        gt_normal = np.asarray(pcd_gt.normals.cpu())
        pred_normal = np.asarray(pcd.normals.cpu())
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
        pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
        pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

        icp_source = pcd
        icp_target = pcd_gt
        if icp_voxel_size > 0:
            icp_source = pcd.voxel_down_sample(icp_voxel_size)
            icp_target = pcd_gt.voxel_down_sample(icp_voxel_size)
            if len(icp_source.points) == 0 or len(icp_target.points) == 0:
                icp_source = pcd
                icp_target = pcd_gt

        reg_p2p = o3d.pipelines.registration.registration_icp(
            icp_source,
            icp_target,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd = pcd.transform(reg_p2p.transformation)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        gt_points = np.asarray(pcd_gt.points)
        rec_points = np.asarray(pcd.points)
        gt_normal = np.asarray(pcd_gt.normals)
        pred_normal = np.asarray(pcd.normals)

    acc, acc_med, nc1, nc1_med = accuracy(
        gt_points,
        rec_points,
        gt_normal,
        pred_normal,
    )
    comp, comp_med, nc2, nc2_med = completion(
        gt_points,
        rec_points,
        gt_normal,
        pred_normal,
    )

    torch.cuda.empty_cache()
    return {
        "acc": float(acc),
        "comp": float(comp),
        "nc1": float(nc1),
        "nc2": float(nc2),
        "acc_med": float(acc_med),
        "comp_med": float(comp_med),
        "nc1_med": float(nc1_med),
        "nc2_med": float(nc2_med),
        "eval_frame_count": eval_frame_count,
        "use_gpu": bool(use_gpu),
    }
