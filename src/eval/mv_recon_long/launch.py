import os
import sys
import argparse
import glob
import time
import copy
from contextlib import nullcontext
import torch
import numpy as np
import open3d as o3d
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from eval.mv_recon.launch import (
    get_args_parser as get_mv_recon_args_parser,
    parse_eviction_policy_layers,
    resolve_global_attn_idx_ranges,
)
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.layers.recent_merge import RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.geometry import FrameDiskCache
from streamvggt.utils.cache_analysis import (
    eviction_nn_config_from_args,
    leverage_score_histogram_config_from_args,
)
from eval.mv_recon.utils import accuracy, completion


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def autocast_context(device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def require_path(parser, args, name):
    value = getattr(args, name)
    if not value:
        parser.error(f"--{name} is required")


def validate_sstream_args(args):
    if args.model_name != "StreamVGGT":
        raise SystemExit("mv_recon_long currently supports --model_name StreamVGGT only.")
    if args.eviction_protect_recent_frames < 0:
        raise SystemExit("Error: --eviction_protect_recent_frames must be >= 0.")
    if args.eviction_protect_special_token_interval < 1:
        raise SystemExit("Error: --eviction_protect_special_token_interval must be >= 1.")
    if args.kf_interval < 1:
        raise SystemExit("Error: --kf_interval must be >= 1.")
    if args.evict_interval < 1:
        raise SystemExit("Error: --evict_interval must be >= 1.")
    if args.leverage_head_mean_dim < 1:
        raise SystemExit("Error: --leverage_head_mean_dim must be >= 1.")
    if args.rls_refresh_interval <= 0:
        raise SystemExit("Error: --rls_refresh_interval must be >= 1.")
    if args.leverage_approx_method == "right_sketch_ridge" and args.leverage_ridge_dim is None:
        raise SystemExit(
            "Error: --leverage_approx_method right_sketch_ridge requires --leverage_ridge_dim."
        )
    if args.leverage_dpp_feature_projection == "random" and args.leverage_ridge_dim is None:
        raise SystemExit(
            "Error: --leverage_dpp_feature_projection random requires --leverage_ridge_dim."
        )
    if args.layer_budget_strategy not in ("uniform", "cosine_precomputed") and (
        args.eviction_policy != "svd_leverage" or args.leverage_granularity != "layer"
    ):
        raise SystemExit(
            "Error: leverage/covariance-based --layer_budget_strategy requires "
            "--eviction_policy svd_leverage and --leverage_granularity layer."
        )
    if args.layer_budget_strategy == "cosine_precomputed" and not args.layer_budget_proportions_path:
        raise SystemExit(
            "Error: --layer_budget_strategy cosine_precomputed requires "
            "--layer_budget_proportions_path."
        )
    try:
        eviction_nn_config_from_args(args, output_dir=args.eviction_nn_analysis_dir)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def has_images(path):
    return any(
        os.path.isfile(os.path.join(path, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        for name in os.listdir(path)
    )


def find_sequence_image_dir(seq_dir):
    for candidate in (
        os.path.join(seq_dir, "images"),
        os.path.join(seq_dir, "images", "images"),
        seq_dir,
    ):
        if os.path.isdir(candidate) and has_images(candidate):
            return candidate
    for root, dirs, _ in os.walk(seq_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if has_images(root):
            return root
    return None


def safe_name(name):
    return name.replace(os.sep, "_").replace(" ", "_")


def resolve_sequence_gt(seq_dir, gt_root):
    candidates = []
    seq_name = os.path.basename(seq_dir)
    if gt_root:
        candidates.extend([
            os.path.join(gt_root, seq_name, "dense_cloud_map.pcd"),
            os.path.join(gt_root, seq_name, "dense_cloud_map.ply"),
            os.path.join(gt_root, f"{seq_name}.pcd"),
            os.path.join(gt_root, f"{seq_name}.ply"),
        ])
    candidates.extend([
        os.path.join(seq_dir, "dense_cloud_map.pcd"),
        os.path.join(seq_dir, "dense_cloud_map.ply"),
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find GT point cloud for sequence {seq_name!r}. Tried: "
        + ", ".join(candidates)
    )


def discover_sequences(args):
    if args.input_root:
        seqs = []
        root = args.input_root
        gt_root = args.gt_root or args.input_root
        for name in sorted(os.listdir(root)):
            if name.startswith("."):
                continue
            seq_dir = os.path.join(root, name)
            if not os.path.isdir(seq_dir):
                continue
            input_dir = find_sequence_image_dir(seq_dir)
            if input_dir is None:
                continue
            seqs.append({
                "name": name,
                "input_dir": input_dir,
                "gt_path": resolve_sequence_gt(seq_dir, gt_root),
                "output_dir": os.path.join(args.output_dir, safe_name(name)),
            })
        if not seqs:
            raise FileNotFoundError(f"No sequence image directories found under {root}")
        return seqs

    if not args.input_dir:
        raise ValueError("Either --input_root or --input_dir is required.")
    if not args.gt_path:
        raise ValueError("--gt_path is required when using --input_dir.")
    return [{
        "name": os.path.basename(os.path.normpath(args.input_dir)) or "sequence",
        "input_dir": args.input_dir,
        "gt_path": args.gt_path,
        "output_dir": args.output_dir,
    }]


class LazyImageFrames:
    def __init__(self, image_names, log_interval=10):
        self.image_names = list(image_names)
        self.log_interval = int(log_interval)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.image_names):
            raise IndexError(idx)
        image_path = self.image_names[idx]
        if self.log_interval > 0 and (idx == 0 or idx + 1 == len(self.image_names) or (idx + 1) % self.log_interval == 0):
            print(f"[Inference] Loading frame {idx + 1}/{len(self.image_names)}: {os.path.basename(image_path)}", flush=True)
        image = load_and_preprocess_images([image_path])
        return {"img": image}

    def __iter__(self):
        for idx in range(len(self.image_names)):
            yield self[idx]


class ProgressFrameWriter:
    def __init__(self, writer, total_frames, log_interval=10):
        self.writer = writer
        self.total_frames = int(total_frames)
        self.log_interval = int(log_interval)

    def __call__(self, frame_idx, frame, result):
        self.writer(frame_idx, frame, result)
        current = int(frame_idx) + 1
        if self.log_interval > 0 and (current == 1 or current == self.total_frames or current % self.log_interval == 0):
            print(f"[Inference] Finished frame {current}/{self.total_frames}", flush=True)


class Inference:
    def __init__(self, args, device):
        self.args = args
        self.device = torch.device(device)
        self.model = None
        self.eviction_policy_layers = None

    def _load_model(self):
        print(f"\n[Inference] Loading model from {self.args.weights}...")
        model = StreamVGGT(total_budget=self.args.budget)
        ckpt = torch.load(self.args.weights, map_location=self.device)
        model.load_state_dict(ckpt, strict=True)
        del ckpt
        model = model.to(self.device)
        model.eval()
        try:
            eviction_policy_layers = parse_eviction_policy_layers(self.args.eviction_policy_layers, model.aggregator.depth)
        except ValueError as exc:
            raise SystemExit(f"Error: --eviction_policy_layers: {exc}") from exc
        if eviction_policy_layers is not None:
            print("Using layer-range eviction override: " f"policy_layers={sorted(eviction_policy_layers)}, fallback_policy=fifo")
        return model, eviction_policy_layers

    def _inference_kwargs(self, eviction_policy_layers):
        global_attn_idx_ranges = resolve_global_attn_idx_ranges(self.args)
        if global_attn_idx_ranges is not None:
            print(f"Global attention index ranges enabled: {global_attn_idx_ranges}")
        recent_merge_config = RecentMergeConfig(enabled=self.args.enable_recent_merge, window=self.args.merge_window, similarity_threshold=self.args.merge_similarity_threshold, voxel_size=self.args.merge_voxel_size, use_depth_confidence=self.args.merge_use_depth_confidence, debug=self.args.merge_debug, chunk_size=self.args.merge_chunk_size, disable_geometry_check=self.args.merge_disable_geometry_check, candidate_mode=self.args.merge_candidate_mode, patch_radius=self.args.merge_patch_radius, voxel_neighbor_radius=self.args.merge_voxel_neighbor_radius, max_candidates_per_token=self.args.merge_max_candidates_per_token, local_fallback=self.args.merge_local_fallback, profile=self.args.merge_profile, recall_debug=self.args.merge_recall_debug, recall_debug_max_tokens=self.args.merge_recall_debug_max_tokens)
        svd_eviction_merge_config = SvdEvictionMergeConfig(enabled=self.args.enable_svd_eviction_merge, mode=self.args.svd_eviction_merge_mode, candidate_axes=self.args.svd_eviction_merge_candidate_axes, reps_per_axis=self.args.svd_eviction_merge_reps_per_axis, similarity_threshold=self.args.svd_eviction_merge_similarity_threshold, use_u_sigma=self.args.svd_eviction_merge_use_u_sigma, geometry_gate=self.args.svd_eviction_merge_geometry_gate, voxel_neighbor_radius=self.args.svd_eviction_merge_voxel_neighbor_radius, allow_missing_geometry=self.args.svd_eviction_merge_allow_missing_geometry, ema_decay=self.args.svd_eviction_merge_ema_decay, use_depth_confidence=self.args.svd_eviction_merge_use_depth_confidence, max_candidates_per_token=self.args.svd_eviction_merge_max_candidates_per_token, chunk_size=self.args.svd_eviction_merge_chunk_size, debug=self.args.svd_eviction_merge_debug, profile=self.args.svd_eviction_merge_profile)
        voxel_covis_config = VoxelCovisConfig(enabled=self.args.use_voxel_covis, voxel_size=self.args.voxel_size, min_shared_voxels=self.args.covis_min_shared_voxels, min_overlap=self.args.covis_min_overlap, max_covis_frames=self.args.max_covis_frames, fallback_recent=self.args.covis_fallback_recent, debug=self.args.covis_debug_log)
        eviction_nn_analysis_config = None
        if self.args.eviction_nn_analysis_dir:
            eviction_nn_analysis_config = eviction_nn_config_from_args(self.args, output_dir=os.path.join(self.args.eviction_nn_analysis_dir, "mv_recon_long"))
        leverage_score_histogram_config = None
        if self.args.leverage_score_histogram_dir:
            leverage_score_histogram_config = leverage_score_histogram_config_from_args(self.args, output_dir=os.path.join(self.args.leverage_score_histogram_dir, "mv_recon_long"))
        return {
            "eviction_policy": self.args.eviction_policy, "eviction_policy_layers": eviction_policy_layers, "leverage_sketch_dim": self.args.leverage_sketch_dim, "leverage_granularity": self.args.leverage_granularity, "leverage_feature": self.args.leverage_feature, "leverage_projection": self.args.leverage_projection, "leverage_head_mean_dim": self.args.leverage_head_mean_dim, "leverage_normalize_rows": self.args.leverage_normalize_rows, "leverage_approx_method": self.args.leverage_approx_method, "leverage_ridge_lambda": self.args.leverage_ridge_lambda, "leverage_ridge_lambda_mode": self.args.leverage_ridge_lambda_mode, "leverage_ridge_score_chunk_size": self.args.leverage_ridge_score_chunk_size, "leverage_ridge_jitter": self.args.leverage_ridge_jitter, "leverage_ridge_dim": self.args.leverage_ridge_dim, "rls_refresh_interval": self.args.rls_refresh_interval, "leverage_random_seed": self.args.leverage_random_seed, "leverage_eviction_selector": self.args.leverage_eviction_selector, "leverage_eviction_risk_mode": self.args.leverage_eviction_risk_mode, "leverage_high_outlier_z": self.args.leverage_high_outlier_z, "leverage_dpp_candidate_multiplier": self.args.leverage_dpp_candidate_multiplier, "leverage_dpp_greedy_block_size": self.args.leverage_dpp_greedy_block_size, "leverage_dpp_quality_beta": self.args.leverage_dpp_quality_beta, "leverage_dpp_diversity_beta": self.args.leverage_dpp_diversity_beta, "leverage_dpp_feature_projection": self.args.leverage_dpp_feature_projection, "leverage_dpp_recency_bonus": self.args.leverage_dpp_recency_bonus, "leverage_dpp_recency_lambda": self.args.leverage_dpp_recency_lambda, "leverage_dpp_recency_window": self.args.leverage_dpp_recency_window, "leverage_dpp_recency_gate_power": self.args.leverage_dpp_recency_gate_power, "leverage_dpp_recency_debug": self.args.leverage_dpp_recency_debug, "layer_budget_strategy": self.args.layer_budget_strategy, "layer_budget_value_gamma": self.args.layer_budget_value_gamma, "layer_budget_value_norm_type": self.args.layer_budget_value_norm_type, "layer_budget_norm_source": self.args.layer_budget_norm_source, "layer_budget_alpha": self.args.layer_budget_alpha, "layer_budget_min_tokens": self.args.layer_budget_min_tokens, "layer_budget_eps": self.args.layer_budget_eps, "layer_budget_depth_mu": self.args.layer_budget_depth_mu, "layer_budget_depth_sigma": self.args.layer_budget_depth_sigma, "layer_budget_depth_floor": self.args.layer_budget_depth_floor, "slots_per_direction": self.args.slots_per_direction, "hybrid_beta": self.args.hybrid_beta, "eviction_protect_recent_frames": self.args.eviction_protect_recent_frames, "eviction_protect_special_tokens": self.args.eviction_protect_special_tokens, "eviction_protect_special_token_interval": self.args.eviction_protect_special_token_interval, "history_anchor_strategy": self.args.history_anchor_strategy, "anchor_interval": self.args.anchor_interval, "min_anchor_interval": self.args.min_anchor_interval, "window_protect_frames": self.args.window_protect_frames, "max_anchors": self.args.max_anchors, "coverage_threshold": self.args.coverage_threshold, "camera_motion_threshold": self.args.camera_motion_threshold, "anchor_keep_ratio": self.args.anchor_keep_ratio, "eviction_debug": self.args.eviction_debug or self.args.profile_eviction, "eviction_nn_analysis_config": eviction_nn_analysis_config, "leverage_score_histogram_config": leverage_score_histogram_config, "recent_merge_config": recent_merge_config, "svd_eviction_merge_config": svd_eviction_merge_config, "voxel_covis_config": voxel_covis_config, "covis_log_fn": print if self.args.covis_debug_log else None, "global_attn_idx_ranges": global_attn_idx_ranges, "global_attn_debug": self.args.global_attn_debug, "kf_interval": self.args.kf_interval, "evict_interval": self.args.evict_interval, "global_cache_history_anchor_special_tokens_only": self.args.global_cache_history_anchor_special_tokens_only, "first_frame_special_tokens_only": self.args.first_frame_special_tokens_only, "camera_cache_history_anchors_only": self.args.camera_cache_history_anchors_only, "camera_cache_keep_dropped_anchors": self.args.camera_cache_keep_dropped_anchors,
        }, leverage_score_histogram_config

    def run(self, input_dir, output_dir):
        if self.model is None:
            self.model, self.eviction_policy_layers = self._load_model()
        cache_dir = os.path.join(output_dir, "frames_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Inference] Loading images from {input_dir}...")
        image_names = [path for path in sorted(glob.glob(os.path.join(input_dir, "*"))) if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS]
        if not image_names:
            raise FileNotFoundError(f"No images found in {input_dir}")
        if self.args.max_frames is not None:
            image_names = image_names[: self.args.max_frames]
        frames = LazyImageFrames(image_names, log_interval=self.args.frame_log_interval)
        frame_writer = ProgressFrameWriter(
            FrameDiskCache(cache_dir),
            total_frames=len(frames),
            log_interval=self.args.frame_log_interval,
        )
        print(f"[Inference] Processing {len(frames)} frames with lazy CPU image loading...")
        inference_kwargs, leverage_score_histogram_config = self._inference_kwargs(self.eviction_policy_layers)
        with torch.no_grad():
            with autocast_context(self.device):
                self.model.inference(frames, frame_writer=frame_writer, cache_results=False, **inference_kwargs)
        if leverage_score_histogram_config is not None:
            leverage_score_histogram_config.flush()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return cache_dir

    def build_ply(self, cache_dir, output_dir):
        ply_path = os.path.join(output_dir, "pred_cloud.ply")
        print(f"[Post-Process] Building PLY to {ply_path}...")

        stride = self.args.stride
        frame_paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[::stride]
        if not frame_paths:
            raise FileNotFoundError(f"No cached frame files found in {cache_dir}")

        raw_pts_list, colors_list, masks_list = [], [], []

        for path in frame_paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            view = payload["view"]
            raw_pts_list.append(payload["pred"]["pts3d_in_other_view"].float())
            colors_list.append(view["img"].float().permute(0, 2, 3, 1))

            mask = view.get("valid_mask")
            if not isinstance(mask, torch.Tensor):
                batch_size, _, height, width = view["img"].shape
                mask = torch.ones((batch_size, height, width), dtype=torch.bool)
            masks_list.append(mask.bool())

        def flatten_valid(data, mask):
            data_flat = data.reshape(data.shape[0], -1, data.shape[-1])
            mask_flat = mask.reshape(mask.shape[0], -1)
            pieces = [
                data_flat[b][mask_flat[b]]
                for b in range(data_flat.shape[0])
                if mask_flat[b].any()
            ]
            if not pieces:
                return torch.empty((0, data.shape[-1]), dtype=data.dtype)
            return torch.cat(pieces, dim=0)

        raw_points = torch.cat(
            [flatten_valid(points, mask) for points, mask in zip(raw_pts_list, masks_list)],
            dim=0,
        )
        raw_colors = torch.cat(
            [flatten_valid(colors, mask) for colors, mask in zip(colors_list, masks_list)],
            dim=0,
        )

        if raw_points.shape[0] == 0:
            raise RuntimeError("Generated point cloud is empty.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points.numpy().astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(raw_colors.numpy(), 0.0, 1.0).astype(np.float64)
        )
        o3d.io.write_point_cloud(ply_path, pcd)
        return ply_path

class Evaluation:
    def __init__(self, gt_path, init_scale_factor=1.0):
        self.gt_path = gt_path
        self.init_scale_factor = float(init_scale_factor)
        self.voxel_size = 0.015
        self.seed = 42
    
    def preprocess(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        return pcd_clean

    def get_scale_and_init_transform(self, source, target):
        center_src = source.get_center()
        center_tgt = target.get_center()
        
        src_centered = np.asarray(source.points) - center_src
        tgt_centered = np.asarray(target.points) - center_tgt
        
        rms_src = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
        rms_tgt = np.sqrt(np.mean(np.sum(tgt_centered**2, axis=1)))
        scale = self.init_scale_factor * (rms_tgt / rms_src)
        print(f"[Eval] RMS init scale factor={self.init_scale_factor:.3f}, scale={scale:.6f}")
        
        T_pre = np.eye(4); T_pre[:3, 3] = -center_src
        T_scale = np.diag([scale, scale, scale, 1.0])
        T_post = np.eye(4); T_post[:3, 3] = center_tgt
        
        return T_post @ T_scale @ T_pre

    def run(self, pred_path, output_metrics_path):
        print(f"\n[Eval] Comparing Pred: {pred_path} \n       vs GT: {self.gt_path}")
        
        np.random.seed(self.seed)
        o3d.utility.random.seed(self.seed)

        pred_pcd = o3d.io.read_point_cloud(pred_path)
        gt_pcd = o3d.io.read_point_cloud(self.gt_path)

        if len(pred_pcd.points) == 0:
            print("[Error] Prediction point cloud is empty.")
            return

        # Initial Alignment (Camera -> LiDAR frame)
        T_cam_wrt_lidar = np.array([
            [-0.012577, -0.999915, -0.003397, -0.03],
            [0.345639, -0.001159, -0.938367, -0.04],
            [0.938283, -0.012976, 0.345625, -0.03],
            [0.0, 0.0, 0.0, 1.0]
        ])
        pred_pcd.transform(np.linalg.inv(T_cam_wrt_lidar))

        pred_clean = self.preprocess(pred_pcd)
        gt_clean = self.preprocess(gt_pcd)

        n_pred, n_gt = len(pred_clean.points), len(gt_clean.points)
        if n_pred == 0 or n_gt == 0:
            print("[Error] Point cloud empty after preprocessing.")
            return

        if n_pred > n_gt:
            pred_eval = pred_clean.random_down_sample(n_gt / n_pred)
            gt_eval = gt_clean
        else:
            gt_eval = gt_clean.random_down_sample(n_pred / n_gt)
            pred_eval = pred_clean

        T_init = self.get_scale_and_init_transform(pred_eval, gt_eval)

        try:
            sim3_reg = o3d.pipelines.registration.registration_icp(
                pred_eval,
                gt_eval,
                0.20,
                T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            T_sim3 = sim3_reg.transformation
            sim3_scale = np.cbrt(abs(np.linalg.det(T_sim3[:3, :3])))
            print(
                f"[Eval] Sim3 ICP scale={sim3_scale:.6f}, "
                f"fitness={sim3_reg.fitness:.6f}, rmse={sim3_reg.inlier_rmse:.6f}"
            )
        except Exception as e:
            print(f"[Warning] Sim3 ICP Failed: {e}. Using RMS init transform.")
            T_sim3 = T_init

        gt_eval.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))

        try:
            reg = o3d.pipelines.registration.registration_icp(
                pred_eval, gt_eval, 0.05, T_sim3,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            final_scale = np.cbrt(abs(np.linalg.det(reg.transformation[:3, :3])))
            print(
                f"[Eval] Point-to-plane ICP final_scale={final_scale:.6f}, "
                f"fitness={reg.fitness:.6f}, rmse={reg.inlier_rmse:.6f}"
            )
            pred_eval.transform(reg.transformation)
        except Exception as e:
            print(f"[Warning] Point-to-plane ICP Failed: {e}. Using Sim3 transform.")
            pred_eval.transform(T_sim3)

        print("[Eval] Calculating Metrics...")
        gt_eval.estimate_normals()
        pred_eval.estimate_normals()
        
        gt_pts, pred_pts = gt_eval.points, pred_eval.points
        gt_norm, pred_norm = np.asarray(gt_eval.normals), np.asarray(pred_eval.normals)

        acc, acc_med, nc1, nc1_med = accuracy(gt_pts, pred_pts, gt_norm, pred_norm)
        comp, comp_med, nc2, nc2_med = completion(gt_pts, pred_pts, gt_norm, pred_norm)

        nc_mean = (nc1 + nc2) / 2
        nc_med = (nc1_med + nc2_med) / 2

        # Print to console
        print("-" * 30)
        print(f"Acc: {acc:.4f} | Comp: {comp:.4f}")
        print(f"NC1: {nc1:.4f} | NC2: {nc2:.4f}")
        print(f"NC_mean: {nc_mean:.4f} | NC_med: {nc_med:.4f}")
        print("-" * 30)

        # Save to file
        with open(output_metrics_path, "w") as f:
            f.write(f"Acc: {acc}\nComp: {comp}\nNC1: {nc1}\nNC2: {nc2}\n")
            f.write(f"Acc_med: {acc_med}\nComp_med: {comp_med}\n")
            f.write(f"NC_mean: {nc_mean}\nNC_med: {nc_med}\n")
        
        print(f"[Success] Metrics saved to {output_metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="SStreamVGGT long-sequence reconstruction evaluation", parents=[get_mv_recon_args_parser()])
    parser.set_defaults(model_name="StreamVGGT")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images for one sequence")
    parser.add_argument("--input_root", type=str, default=None, help="Root containing sequence directories")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to one sequence ground-truth .pcd/.ply")
    parser.add_argument("--gt_root", type=str, default=None, help="Root containing per-sequence ground-truth point clouds")
    parser.add_argument("--stride", type=int, default=10, help="Stride for PLY generation")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference if PLY already exists")
    parser.add_argument("--frame_log_interval", type=int, default=10, help="Print input frame progress every N frames; 0 disables")
    args = parser.parse_args()
    require_path(parser, args, "weights")
    require_path(parser, args, "output_dir")
    validate_sstream_args(args)
    try:
        sequences = discover_sequences(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(f"[Info] Found {len(sequences)} sequence(s).")
    inferencer = None
    for seq in sequences:
        os.makedirs(seq["output_dir"], exist_ok=True)
        print("=" * 40)
        print(f"[Sequence] {seq['name']}")
        print(f"Input:  {seq['input_dir']}")
        print(f"GT:     {seq['gt_path']}")
        print(f"Output: {seq['output_dir']}")
        ply_path = os.path.join(seq["output_dir"], "pred_cloud.ply")
        print(f"ply path: {ply_path}")
        if args.skip_inference and os.path.exists(ply_path):
            print(f"[Info] Skipping inference, using existing: {ply_path}")
        else:
            if inferencer is None:
                inferencer = Inference(args, device=args.device)
            cache_dir = inferencer.run(seq["input_dir"], seq["output_dir"])
            ply_path = inferencer.build_ply(cache_dir, seq["output_dir"])
        metrics_path = os.path.join(seq["output_dir"], "metrics.txt")
        evaluator = Evaluation(gt_path=seq["gt_path"], init_scale_factor=1.0)
        evaluator.run(pred_path=ply_path, output_metrics_path=metrics_path)

if __name__ == "__main__":
    main()