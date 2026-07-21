import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.pose_evaluation.metadata import dataset_metadata
from eval.pose_evaluation.utils import *

from accelerate import PartialState
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.layers.confidence_state import parse_confidence_gate_init
from streamvggt.utils.cache_analysis import (
    add_leverage_score_histogram_args,
    leverage_score_histogram_config_from_args,
)

from tqdm import tqdm
import time

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
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
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="7scenes",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument("--kf_every", type=int, default=1, help="take one frame every N frames")
    parser.add_argument("--pose_eval_stride", type=int, default=1, help="stride for pose evaluation; effective stride is kf_every * pose_eval_stride")
    parser.add_argument(
        "--stream_chunk_size",
        "--stream-chunk-size",
        type=int,
        default=1,
        help="Number of consecutive stream frames to process in one chunk-causal forward",
    )
    parser.add_argument(
        "--empty_cache_interval",
        "--empty-cache-interval",
        type=int,
        default=1,
        help="Call torch.cuda.empty_cache() every N output frames; 0 disables it. Default 1 preserves previous behavior.",
    )
    parser.add_argument("--eviction_policy", type=str, default="svd_leverage", help="Cache eviction policy: mean, baseline_mean, svd_leverage")
    parser.add_argument(
        "--leverage_granularity",
        type=str,
        default="layer",
        help="Granularity for svd_leverage eviction: per-head or one shared layer-wise score vector",
    )
    parser.add_argument(
        "--leverage_feature",
        type=str,
        default="key",
        help="Feature tensor for svd_leverage eviction: keys only, concatenated keys and values, or concatenated low-dimensional key/value sketches",
    )
    parser.add_argument(
        "--leverage_projection",
        type=str,
        default="random",
        choices=("random"),
        help="Projection mode for svd_leverage eviction: random right sketch",
    )
    parser.add_argument(
        "--leverage_normalize_rows",
        "--leverage-normalize-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize token feature rows before svd_leverage QR/leverage scoring",
    )
    parser.add_argument(
        "--leverage_normalize_before_projection",
        "--leverage-normalize-before-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize layer-wise key rows before random leverage projection",
    )
    parser.add_argument(
        "--leverage_normalize_before_projection_headwise",
        "--leverage-normalize-before-projection-headwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When normalizing before projection, normalize each head key row independently",
    )
    parser.add_argument(
        "--leverage_projected_key_cache",
        "--leverage-projected-key-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse cached layer-wise random projected key features for surviving tokens",
    )
    parser.add_argument(
        "--leverage_approx_method",
        "--leverage-approx-method",
        type=str,
        default="right_sketch_ridge",
        choices=("exact_qr", "full_d_ridge", "right_sketch_ridge"),
        help="Leverage approximation: exact QR or ridge-based scoring",
    )
    parser.add_argument(
        "--leverage_ridge_lambda",
        "--leverage-ridge-lambda",
        type=float,
        default=0.0,
        help="Ridge lambda for full_d_ridge and right_sketch_ridge leverage scoring",
    )
    parser.add_argument(
        "--leverage_ridge_lambda_mode",
        "--leverage-ridge-lambda-mode",
        type=str,
        default="absolute",
        choices=("relative", "absolute"),
        help="Use absolute lambda or scale it by trace(X^T X) / D",
    )
    parser.add_argument("--leverage_ridge_score_chunk_size", type=int, default=16384, help="Token chunk size for ridge leverage Cholesky solves")
    parser.add_argument("--leverage_ridge_jitter", type=float, default=1e-6, help="Absolute diagonal jitter added to ridge systems before Cholesky")
    parser.add_argument("--leverage_ridge_dim", type=int, default=256, help="Projection dimension for right_sketch_ridge; required for right_sketch_ridge")
    parser.add_argument("--rls_refresh_interval", type=int, default=8, help="Refresh interval for ridge leverage K^T K and Cholesky factorization; 1 refreshes every frame")
    parser.add_argument("--leverage_random_seed", type=int, default=42, help="Random seed for leverage sketches")
    parser.add_argument("--leverage_eviction_selector", type=str, default="topk", help="Eviction selector for svd_leverage scores: topk")
    parser.add_argument("--leverage_conf_gate", action="store_true", help="Apply normalized depth/world-point confidence gate to SVD leverage keep scores")
    parser.add_argument("--leverage_conf_gate_floor", type=float, default=0.2, help="Minimum multiplicative confidence gate value")
    parser.add_argument("--leverage_conf_gate_depth_alpha", type=float, default=1.0, help="Exponent applied to transformed depth confidence in the confidence gate")
    parser.add_argument("--leverage_conf_gate_point_beta", type=float, default=0.0, help="Exponent applied to transformed world-point confidence in the confidence gate")
    parser.add_argument("--leverage_conf_gate_k", type=float, default=1.0, help="Legacy positive normalization parameter retained for compatibility")
    parser.add_argument("--leverage_conf_gate_transform", type=str, default="sigmoid", choices=("ratio", "sigmoid"), help="Confidence gate transform: ratio uses (c - 1) / c; sigmoid uses torch.sigmoid(c)")
    parser.add_argument("--leverage_conf_gate_init", type=str, default="mean", help="Initial current-frame confidence gate before head confidence update: mean or finite non-negative float")
    parser.add_argument("--leverage_attention_utility", action="store_true", help="Use frozen early-attention utility with STAC CUDA post-attention eviction")
    parser.add_argument("--leverage_attention_beta", type=float, default=0.3, help="Weight of normalized frozen attention utility in the keep score")
    parser.add_argument("--leverage_attention_ema_decay", type=float, default=0.9, help="EMA decay used during each token's finite attention observation horizon")
    parser.add_argument("--leverage_attention_freeze_updates", type=int, default=5, help="Number of attention observations accumulated before freezing token utility")
    parser.add_argument("--leverage_attention_colsum_subsample_ratio", type=float, default=1.0, help="Fraction of query rows used by the STAC CUDA column-sum kernel")
    parser.add_argument("--leverage_conf_gate_special_mode", type=str, default="mean", choices=("mean", "one"), help="Gate mode for special/prefix tokens: mean uses the patch gate mean; one sets them to 1.0")
    parser.add_argument("--layer_budget_strategy", type=str, default="value_weighted_leverage_pr", choices=("uniform", "leverage_pr", "key_norm", "value_weighted_leverage_pr"), help="Layer-wise KV budget allocation strategy")
    parser.add_argument("--layer_budget_alpha", type=float, default=0.7)
    parser.add_argument("--layer_budget_min_tokens", type=int, default=0)
    parser.add_argument("--layer_budget_eps", type=float, default=0)
    parser.add_argument("--layer_budget_value_gamma", type=float, default=0.7)
    parser.add_argument("--layer_budget_value_norm_type", type=str, default="mean", choices=("mean", "rms"), help="Layer value-norm prior type for value_weighted_leverage_pr budget allocation")
    parser.add_argument("--layer_budget_norm_source", type=str, default="key", help="Tensor source for value_weighted_leverage_pr norm prior: value cache or key cache")
    parser.add_argument("--layer_budget_log_scores", action="store_true", help="Write per-step layer budget scores to layer_budget_scores.csv under each scene output directory")
    parser.add_argument("--layer_budget_log_path", type=str, default=None, help="Optional explicit CSV path for layer budget score logs")
    parser.add_argument("--eviction_debug", action="store_true", help="Print verbose eviction summaries without enabling latency profiling")
    parser.add_argument("--profile_eviction", action="store_true", help="Print per-eviction svd_leverage timing/profile fields without changing eviction behavior")
    add_leverage_score_histogram_args(parser)
    parser.add_argument("--budget", type=int, default=200000, help="Total token budget for StreamVGGT (if applicable)")
    parser.add_argument("--budget_frame_multiplier", type=float, default=None, help="Set StreamVGGT total budget to ceil(multiplier * tokens_per_frame) * num_global_layers; overrides --budget")
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = None
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                if not os.path.isdir(dir_path):
                    raise FileNotFoundError(
                        f"Missing sequence directory for eval_dataset={args.eval_dataset}: {dir_path}"
                    )
                filelist_func = metadata.get("filelist_func", None)
                if filelist_func is not None:
                    filelist = filelist_func(dir_path)
                else:
                    filelist = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                    filelist.sort()
                effective_stride = args.kf_every * args.pose_eval_stride
                filelist = filelist[::effective_stride]
                if args.max_frames is not None:
                    filelist = filelist[: args.max_frames]
                if not filelist:
                    raise FileNotFoundError(
                        f"No input images found for eval_dataset={args.eval_dataset}: {dir_path}"
                    )

                images = load_and_preprocess_images(filelist)
                frames = []
                for i in range(images.shape[0]):
                    frame = {
                        "img": images[i]
                    }
                    frames.append(frame)

                predictions = {}
                with torch.no_grad():
                    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()

                        leverage_score_histogram_config = None
                        if args.leverage_score_histogram_dir:
                            safe_seq = str(seq).replace("/", "_").replace(os.sep, "_").replace(" ", "_")
                            hist_dir = os.path.join(
                                args.leverage_score_histogram_dir,
                                args.eval_dataset,
                                f"rank_{distributed_state.process_index}",
                                safe_seq,
                            )
                            leverage_score_histogram_config = leverage_score_histogram_config_from_args(args, output_dir=hist_dir)
                        output = model.inference(
                            frames,
                            total_budget=args.budget,
                            stream_chunk_size=args.stream_chunk_size,
                            budget_frame_multiplier=args.budget_frame_multiplier,
                            eviction_policy=args.eviction_policy,
                            leverage_granularity=args.leverage_granularity,
                            leverage_feature=args.leverage_feature,
                            leverage_projection=args.leverage_projection,
                            leverage_normalize_rows=args.leverage_normalize_rows,
                            leverage_normalize_before_projection=args.leverage_normalize_before_projection,
                            leverage_normalize_before_projection_headwise=args.leverage_normalize_before_projection_headwise,
                            leverage_projected_key_cache=args.leverage_projected_key_cache,
                            leverage_approx_method=args.leverage_approx_method,
                            leverage_ridge_lambda=args.leverage_ridge_lambda,
                            leverage_ridge_lambda_mode=args.leverage_ridge_lambda_mode,
                            leverage_ridge_score_chunk_size=args.leverage_ridge_score_chunk_size,
                            leverage_ridge_jitter=args.leverage_ridge_jitter,
                            leverage_ridge_dim=args.leverage_ridge_dim,
                            rls_refresh_interval=args.rls_refresh_interval,
                            leverage_random_seed=args.leverage_random_seed,
                            leverage_eviction_selector=args.leverage_eviction_selector,
                            leverage_conf_gate=args.leverage_conf_gate,
                            leverage_conf_gate_floor=args.leverage_conf_gate_floor,
                            leverage_conf_gate_depth_alpha=args.leverage_conf_gate_depth_alpha,
                            leverage_conf_gate_point_beta=args.leverage_conf_gate_point_beta,
                            leverage_conf_gate_k=args.leverage_conf_gate_k,
                            leverage_conf_gate_transform=args.leverage_conf_gate_transform,
                            leverage_conf_gate_init=args.leverage_conf_gate_init,
                            leverage_attention_utility=args.leverage_attention_utility,
                            leverage_attention_beta=args.leverage_attention_beta,
                            leverage_attention_ema_decay=args.leverage_attention_ema_decay,
                            leverage_attention_freeze_updates=args.leverage_attention_freeze_updates,
                            leverage_attention_colsum_subsample_ratio=args.leverage_attention_colsum_subsample_ratio,
                            leverage_conf_gate_special_mode=args.leverage_conf_gate_special_mode,
                            layer_budget_strategy=args.layer_budget_strategy,
                            layer_budget_value_gamma=args.layer_budget_value_gamma,
                            layer_budget_value_norm_type=args.layer_budget_value_norm_type,
                            layer_budget_norm_source=args.layer_budget_norm_source,
                            layer_budget_alpha=args.layer_budget_alpha,
                            layer_budget_min_tokens=args.layer_budget_min_tokens,
                            layer_budget_eps=args.layer_budget_eps,
                            profile_eviction=args.profile_eviction,
                            empty_cache_interval=args.empty_cache_interval,
                            eviction_debug=args.eviction_debug,
                            leverage_score_histogram_config=leverage_score_histogram_config,
                        )
                        if leverage_score_histogram_config is not None:
                            leverage_score_histogram_config.flush()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device)
                        infer_time = time.perf_counter() - start
                fps = len(filelist) / infer_time if infer_time > 0 else float("inf")
                print(
                    f"Finished pose estimation for {args.eval_dataset} {seq: <16}, "
                    f"Inference time: {infer_time:.6f}s, FPS: {fps:.2f}"
                )

                all_camera_pose = []
                for res in output.ress:
                    all_camera_pose.append(res['camera_pose'].squeeze(0))
                    
                predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0) # (S, 9)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], 
                        images.shape[-2:]
                    )
                predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
                predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None # (S, 3, 3)

                # Convert predicted world-to-camera extrinsics to cam-to-world poses.
                # TUM/evo trajectories store camera centers in world coordinates.
                add_row = torch.tensor(
                    [0, 0, 0, 1],
                    device=predictions["extrinsic"].device,
                    dtype=predictions["extrinsic"].dtype,
                ).expand(predictions["extrinsic"].size(0), 1, 4)
                pred_w2c_poses = torch.cat((predictions["extrinsic"], add_row), dim=1)
                pred_c2w_poses = torch.linalg.inv(pred_w2c_poses)

                # Extract focal length and principal point from intrinsics for saving
                if predictions["intrinsic"] is not None:
                    focals_x = predictions["intrinsic"][:, 0, 0]
                    focals_y = predictions["intrinsic"][:, 1, 1]
                    focal = (focals_x + focals_y) / 2.0 # Average focal length
                    pp = predictions["intrinsic"][:, :2, 2] # Principal points (S, 2)
                    cam_dict = {
                        "focal": focal.cpu().numpy(),
                        "pp": pp.cpu().numpy(),
                    }
                else: # Fallback if no intrinsics are predicted
                    H, W = images.shape[-2:]
                    cam_dict = {
                        "focal": np.full(len(images), max(H, W)), # A common heuristic
                        "pp": np.tile([W/2, H/2], (len(images), 1)),
                    }

                pred_traj = get_tum_poses(pred_c2w_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pred_c2w_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                pose_save_path = os.path.join(save_dir, f"{seq}_poses.npz")
                np.savez(
                    pose_save_path,
                    pose_enc=predictions["pose_enc"].cpu().numpy(),
                    extrinsic=predictions["extrinsic"].cpu().numpy()
                )

                print(f"Pose encoding and extrinsics saved to: {pose_save_path}")


                gt_traj_loader = metadata.get("gt_traj_loader", None)
                if gt_traj_loader is not None:
                    gt_traj = gt_traj_loader(img_path, anno_path, seq, filelist)
                else:
                    gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                    traj_format = metadata.get("traj_format", None)

                    if gt_traj_file is None:
                        gt_traj = None
                    elif args.eval_dataset == "sintel":
                        gt_traj = load_traj(
                            gt_traj_file=gt_traj_file,
                            stride=effective_stride,
                            num_frames=len(filelist),
                        )
                    elif traj_format is not None:
                        gt_traj = load_traj(
                            gt_traj_file=gt_traj_file,
                            traj_format=traj_format,
                            stride=effective_stride,
                            num_frames=len(filelist),
                        )
                    else:
                        gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    from streamvggt.utils.load_fn import load_and_preprocess_images 
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    from dust3r.utils.image import load_images_for_eval as load_images

    args.full_seq = False
    args.no_crop = False

    print("Loading StreamVGGT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name or "StreamVGGT"
    if model_name != "StreamVGGT":
        raise NotImplementedError(f"Unsupported model_name for pose evaluation: {model_name}")
    model = StreamVGGT(total_budget=args.budget)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    if args.layer_budget_strategy == "cosine_precomputed":
        model.set_layer_budget_proportions(args.layer_budget_proportions_path)
        print(f"Loaded precomputed layer budget proportions: {args.layer_budget_proportions_path}")
    model.eval()
    print("Model loaded successfully.")

    eval_pose_estimation(args, model, save_dir=args.output_dir)
