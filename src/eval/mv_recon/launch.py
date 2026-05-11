import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict
from streamvggt.layers.recent_merge import RecentMergeConfig


def resolve_global_attn_idx_ranges(args):
    if args.middle_global_only and args.global_attn_idx_ranges is not None:
        raise ValueError("--middle-global-only cannot be combined with --global-attn-idx-ranges")
    if args.middle_global_only:
        return "9:"
    return args.global_attn_idx_ranges


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--eviction_policy", type=str, default="mean", help="Cache eviction policy")
    parser.add_argument(
        "--leverage_sketch_dim",
        type=int,
        default=16,
        help="Right sketch dimension for svd_leverage eviction; set 0 for exact full-space QR",
    )
    parser.add_argument(
        "--leverage_granularity",
        type=str,
        default="head",
        choices=("head", "layer"),
        help="Granularity for svd_leverage eviction: per-head or one shared layer-wise score vector",
    )
    parser.add_argument(
        "--leverage_feature",
        type=str,
        default="key",
        choices=("key", "key_value"),
        help="Feature tensor for svd_leverage eviction: keys only or concatenated keys and values",
    )
    parser.add_argument(
        "--enable_recent_merge",
        action="store_true",
        help="Enable geometry-validated recent KV cache merging",
    )
    parser.add_argument(
        "--merge_window",
        type=int,
        default=3,
        help="Number of recent frames considered for KV cache merging",
    )
    parser.add_argument(
        "--merge_similarity_threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for recent KV cache merging",
    )
    parser.add_argument(
        "--merge_voxel_size",
        type=float,
        default=0.05,
        help="Voxel size for geometry validation during recent KV cache merging",
    )
    parser.add_argument(
        "--merge_use_depth_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use depth confidence to weight recent KV EMA merges",
    )
    parser.add_argument(
        "--merge_debug",
        action="store_true",
        help="Print per-layer recent merge diagnostics",
    )
    parser.add_argument(
        "--merge_chunk_size",
        type=int,
        default=512,
        help="Current-token chunk size for batched recent merge cosine search",
    )
    parser.add_argument(
        "--merge_disable_geometry_check",
        action="store_true",
        help="Disable voxel validation for ablations; geometry check is enabled by default",
    )
    parser.add_argument(
        "--merge_candidate_mode",
        choices=("full", "spatial", "voxel", "voxel_spatial"),
        default="full",
        help="Candidate search mode for recent merge",
    )
    parser.add_argument(
        "--merge_patch_radius",
        type=int,
        default=1,
        help="Patch-grid radius for local spatial recent merge candidate search",
    )
    parser.add_argument(
        "--merge_voxel_neighbor_radius",
        type=int,
        default=0,
        help="Chebyshev voxel neighbor radius for local voxel recent merge candidates",
    )
    parser.add_argument(
        "--merge_max_candidates_per_token",
        type=int,
        default=64,
        help="Maximum local recent merge candidates retained per current token",
    )
    parser.add_argument(
        "--merge_local_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow local candidate modes to fall back to weaker local candidates",
    )
    parser.add_argument(
        "--merge_profile",
        action="store_true",
        help="Print recent merge profiling timings",
    )
    parser.add_argument(
        "--merge_recall_debug",
        action="store_true",
        help="Compare local recent merge candidates against full-window candidates for diagnostics",
    )
    parser.add_argument(
        "--merge_recall_debug_max_tokens",
        type=int,
        default=1024,
        help="Maximum source tokens sampled per layer/head for recent merge recall diagnostics",
    )
    parser.add_argument(
        "--global_attn_idx_ranges",
        "--global-attn-idx-ranges",
        type=str,
        default=None,
        help="Half-open global attention index ranges to keep global, e.g. '9:', '9:20', '6:10,14:20'",
    )
    parser.add_argument(
        "--middle_global_only",
        "--middle-global-only",
        action="store_true",
        help="Shortcut for --global-attn-idx-ranges 9:",
    )
    parser.add_argument(
        "--global_attn_debug",
        "--global-attn-debug",
        action="store_true",
        help="Print per-block global-to-frame and KV cache decisions",
    )
    parser.add_argument(
        "--budget", type=int, default=200000, help="Total token budget for StreamVGGT (if applicable)"
    )
    return parser


def main(args):
    try:
        global_attn_idx_ranges = resolve_global_attn_idx_ranges(args)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if global_attn_idx_ranges is not None:
        print(f"Global attention index ranges enabled: {global_attn_idx_ranges}")
    if args.eviction_policy == "svd_leverage":
        sketch_label = "exact" if args.leverage_sketch_dim == 0 else str(args.leverage_sketch_dim)
        print(
            "Using SVD leverage eviction: "
            f"sketch_dim={sketch_label}, "
            f"granularity={args.leverage_granularity}, "
            f"feature={args.leverage_feature}"
        )
    if args.merge_window < 1:
        raise SystemExit(f"Error: --merge_window must be >= 1, got {args.merge_window}.")
    if not (0.0 <= args.merge_similarity_threshold <= 1.0):
        raise SystemExit(
            "Error: --merge_similarity_threshold must be in [0, 1], "
            f"got {args.merge_similarity_threshold}."
        )
    if args.merge_voxel_size <= 0:
        raise SystemExit(f"Error: --merge_voxel_size must be > 0, got {args.merge_voxel_size}.")
    if args.merge_chunk_size < 1:
        raise SystemExit(f"Error: --merge_chunk_size must be >= 1, got {args.merge_chunk_size}.")
    if args.merge_patch_radius < 0:
        raise SystemExit(f"Error: --merge_patch_radius must be >= 0, got {args.merge_patch_radius}.")
    if args.merge_voxel_neighbor_radius < 0:
        raise SystemExit(
            "Error: --merge_voxel_neighbor_radius must be >= 0, "
            f"got {args.merge_voxel_neighbor_radius}."
        )
    if args.merge_max_candidates_per_token < 1:
        raise SystemExit(
            "Error: --merge_max_candidates_per_token must be >= 1, "
            f"got {args.merge_max_candidates_per_token}."
        )
    if args.merge_recall_debug_max_tokens < 1:
        raise SystemExit(
            "Error: --merge_recall_debug_max_tokens must be >= 1, "
            f"got {args.merge_recall_debug_max_tokens}."
        )

    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
        # resolution = (518, 336)
    else:
        raise NotImplementedError
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/home/dongjae/data/7scenes_sfm",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=2,
            max_frames=args.max_frames,
        ),  # 20),
        # "NRGBD": NRGBD(
        #     split="test",
        #     ROOT="/home/ma-user/work/dataset/3D_Reconstruction/neural_rgbd_data",
        #     resolution=resolution,
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=500,
        # ),
    }

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name == "StreamVGGT":
        # from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = StreamVGGT(total_budget=args.budget)
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to("cuda")
    elif model_name == "VGGT":
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = VGGT()
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to("cuda")

    else:
        raise NotImplementedError
    del ckpt
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []
                    in_camera1 = None  

                    if model_name == "stream3r" or "VGGT":
                        revisit = args.revisit
                        update = not args.freeze
                        num_input_frames = len(batch)
                        if revisit > 1:
                            # repeat input for 'revisit' times
                            new_views = []
                            for r in range(revisit):
                                for i in range(len(batch)):
                                    new_view = deepcopy(batch[i])
                                    new_view["idx"] = [
                                        (r * len(batch) + i)
                                        for _ in range(len(batch[i]["idx"]))
                                    ]
                                    new_view["instance"] = [
                                        str(r * len(batch) + i)
                                        for _ in range(len(batch[i]["instance"]))
                                    ]
                                    if r > 0:
                                        if not update:
                                            new_view["update"] = torch.zeros_like(
                                                batch[i]["update"]
                                            ).bool()
                                    new_views.append(new_view)
                            batch = new_views
                        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                        with torch.cuda.amp.autocast(dtype=dtype):
                            if isinstance(batch, dict) and "img" in batch:
                                batch["img"] = (batch["img"] + 1.0) / 2.0
                            elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                                for view in batch:
                                    view["img"] = (view["img"] + 1.0) / 2.0

                        with torch.cuda.amp.autocast(dtype=dtype):
                            with torch.no_grad():
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize(device)
                                infer_start = time.perf_counter()
                                recent_merge_config = RecentMergeConfig(
                                    enabled=args.enable_recent_merge,
                                    window=args.merge_window,
                                    similarity_threshold=args.merge_similarity_threshold,
                                    voxel_size=args.merge_voxel_size,
                                    use_depth_confidence=args.merge_use_depth_confidence,
                                    debug=args.merge_debug,
                                    chunk_size=args.merge_chunk_size,
                                    disable_geometry_check=args.merge_disable_geometry_check,
                                    candidate_mode=args.merge_candidate_mode,
                                    patch_radius=args.merge_patch_radius,
                                    voxel_neighbor_radius=args.merge_voxel_neighbor_radius,
                                    max_candidates_per_token=args.merge_max_candidates_per_token,
                                    local_fallback=args.merge_local_fallback,
                                    profile=args.merge_profile,
                                    recall_debug=args.merge_recall_debug,
                                    recall_debug_max_tokens=args.merge_recall_debug_max_tokens,
                                )
                                results = model.inference(
                                    batch,
                                    eviction_policy=args.eviction_policy,
                                    leverage_sketch_dim=args.leverage_sketch_dim,
                                    leverage_granularity=args.leverage_granularity,
                                    leverage_feature=args.leverage_feature,
                                    recent_merge_config=recent_merge_config,
                                    global_attn_idx_ranges=global_attn_idx_ranges,
                                    global_attn_debug=args.global_attn_debug,
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize(device)
                                infer_time = time.perf_counter() - infer_start
                                fps = num_input_frames / infer_time if infer_time > 0 else float("inf")
                                time_all.append(infer_time)
                                fps_all.append(fps)

                            preds, batch = results.ress, results.views 

                            if args.use_proj:
                                pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(len(preds))], dim=1)
                                depth_map = torch.stack([preds[s]["depth"] for s in range(len(preds))], dim=1)
                                depth_conf = torch.stack([preds[s]["depth_conf"] for s in range(len(preds))], dim=1)
                                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
                                                                                    batch[0]["img"].shape[-2:])

                                if "DTU" in name_data:
                                    depth_map = depth_map * 1000.0
                                    extrinsic[..., :3, 3] *= 1000.0

                                point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                                                extrinsic.squeeze(0),
                                                                                                intrinsic.squeeze(0))
                            valid_length = len(preds) // args.revisit
                            if args.revisit > 1:
                                preds = preds[-valid_length:]
                                batch = batch[-valid_length:]

                            timing_scene_id = batch[0]["label"][0].rsplit("/", 1)[0]
                            timing_msg = (
                                f"Timing before eval - Idx: {timing_scene_id}, "
                                f"Time: {infer_time:.6f}, FPS: {fps:.3f}"
                            )
                            print(timing_msg)
                                

                        # Evaluation
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            if args.use_proj:
                                pts = point_map_by_unprojection[j]
                                conf = depth_conf[0, j].cpu().data.numpy()
                            else:
                                pts = pred_pts[j].cpu().numpy()[0]
                                conf = preds[j]["conf"].cpu().data.numpy()[0]

                            # mask = mask & (conf > 1.8)

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                            H, W = image.shape[:2]
                            cx = W // 2
                            cy = H // 2
                            l, t = cx - 112, cy - 112
                            r, b = cx + 112, cy + 112
                            image = image[t:b, l:r]
                            mask = mask[t:b, l:r]
                            pts = pts[t:b, l:r]
                            pts_gt = pts_gt[t:b, l:r]

                            # Align predicted 3D points to the ground truth
                            # pts = geotrf(in_camera1, pts)
                            # pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append(image[None, ...])
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}

                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all

                    # np.save(
                    #     os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                    #     save_params,
                    # )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    mask = np.isfinite(pts_all_masked)  
                    pts_all_masked = pts_all_masked[mask]

                    mask_gt = np.isfinite(pts_gt_all_masked)
                    pts_gt_all_masked = pts_gt_all_masked[mask]

                    if args.use_proj:
                        def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
                            assert src.shape == dst.shape
                            N, dim = src.shape

                            mu_src = src.mean(axis=0)
                            mu_dst = dst.mean(axis=0)
                            src_c = src - mu_src
                            dst_c = dst - mu_dst

                            Sigma = dst_c.T @ src_c / N  # (3,3)

                            U, D, Vt = np.linalg.svd(Sigma) 

                            S = np.eye(dim)
                            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                                S[-1, -1] = -1

                            R = U @ S @ Vt

                            if with_scale:
                                var_src = (src_c ** 2).sum() / N
                                s = (D * S.diagonal()).sum() / var_src
                            else:
                                s = 1.0

                            t = mu_dst - s * R @ mu_src

                            return s, R, t

                        pts_all_masked = pts_all_masked.reshape(-1, 3)
                        pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                        s, R, t = umeyama_alignment(pts_all_masked, pts_gt_all_masked, with_scale=True)
                        pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                        pts_all_masked = pts_all_aligned

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    # o3d.io.write_point_cloud(
                    #     os.path.join(
                    #         save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                    #     ),
                    #     pcd,
                    # )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    # o3d.io.write_point_cloud(
                    #     os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                    #     pcd_gt,
                    # )

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)

                    # o3d.io.write_point_cloud(
                    #     os.path.join(
                    #         save_path, f"{scene_id.replace('/', '_')}-mask_align.ply"
                    #     ),
                    #     pcd,
                    # )

                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, Time: {infer_time}, FPS: {fps}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, Time: {infer_time}, FPS: {fps}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id" and value is not None:
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)



from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
    (?:,\s*Time:\s*(?P<time>[^,]+),\s*FPS:\s*(?P<fps>[^,]+))?
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
