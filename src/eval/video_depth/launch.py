import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import save_depth_maps
from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r
import time
import subprocess
from pathlib import Path
from tqdm import tqdm
from streamvggt.utils.cache_analysis import (
    add_eviction_nn_analysis_args,
    add_leverage_score_histogram_args,
    add_token_overlay_dump_args,
    eviction_nn_config_from_args,
    leverage_score_histogram_config_from_args,
    token_overlay_dump_config_from_args,
)

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
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
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument(
        "--eviction_policy",
        type=str,
        default="mean",
        help="Cache eviction policy: mean, baseline_mean, svd_leverage, or dpp",
    )
    parser.add_argument(
        "--leverage_sketch_dim",
        type=int,
        default=0,
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
        "--leverage_projection",
        type=str,
        default="random",
        choices=("random", "head_mean"),
        help="Projection mode for svd_leverage eviction: random right sketch or deterministic per-head means",
    )
    parser.add_argument(
        "--leverage_head_mean_dim",
        type=int,
        default=1,
        help="Number of mean-pooled channel groups per head for leverage_projection='head_mean'",
    )
    parser.add_argument(
        "--leverage_normalize_rows",
        "--leverage-normalize-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize token feature rows before svd_leverage QR/leverage scoring",
    )
    parser.add_argument(
        "--leverage_approx_method",
        "--leverage-approx-method",
        type=str,
        default="right_sketch",
        choices=("exact_qr", "right_sketch", "full_d_ridge", "right_sketch_ridge"),
        help="Leverage approximation: exact QR, right-sketched/Compactor-style, or ridge-based scoring",
    )
    parser.add_argument(
        "--leverage_ridge_lambda",
        "--leverage-ridge-lambda",
        type=float,
        default=1e-3,
        help="Ridge lambda for full_d_ridge and right_sketch_ridge leverage scoring",
    )
    parser.add_argument(
        "--leverage_ridge_lambda_mode",
        "--leverage-ridge-lambda-mode",
        type=str,
        default="relative",
        choices=("relative", "absolute"),
        help="Use absolute lambda or scale it by trace(X^T X) / D",
    )
    parser.add_argument(
        "--leverage_ridge_score_chunk_size",
        "--leverage-ridge-score-chunk-size",
        type=int,
        default=4096,
        help="Token chunk size for ridge leverage Cholesky solves",
    )
    parser.add_argument(
        "--leverage_ridge_jitter",
        "--leverage-ridge-jitter",
        type=float,
        default=1e-6,
        help="Relative diagonal jitter added to ridge systems before Cholesky",
    )
    parser.add_argument(
        "--leverage_ridge_dim",
        "--leverage-ridge-dim",
        type=int,
        default=None,
        help="Projection dimension for right_sketch_ridge; required for right_sketch_ridge",
    )
    parser.add_argument(
        "--leverage_diag",
        "--leverage-diag",
        action="store_true",
        help="Print ridge leverage diagnostic statistics at selected eviction steps",
    )
    parser.add_argument(
        "--leverage_diag_interval",
        "--leverage-diag-interval",
        type=int,
        default=0,
        help="Diagnostic interval; 0 prints only the first eviction step, positive values print every N steps",
    )
    parser.add_argument("--leverage_random_seed", "--leverage-random-seed", type=int, default=0)
    parser.add_argument(
        "--leverage_eviction_selector",
        "--leverage-eviction-selector",
        type=str,
        default="topk",
        choices=("topk", "fast_dpp", "layer_head_fast_dpp", "similarity_topk"),
        help="Eviction selector for svd_leverage scores: topk, shared Fast DPP, similarity top-k, or GPU head-wise Fast DPP with layer scores",
    )
    parser.add_argument(
        "--leverage_similarity_granularity",
        "--leverage-similarity-granularity",
        type=str,
        default="layer",
        choices=("layer", "head"),
        help="Similarity feature granularity for similarity_topk: layer preserves shared layer-wise eviction, head uses head-wise cosine and head-specific eviction",
    )
    parser.add_argument(
        "--leverage_similarity_feature_projection",
        "--leverage-similarity-feature-projection",
        type=str,
        default="raw",
        choices=("raw", "random"),
        help="Feature source for similarity_topk cosine: raw key features or random projected leverage features reused from score computation",
    )
    parser.add_argument(
        "--leverage_similarity_leverage_gamma",
        "--leverage-similarity-leverage-gamma",
        type=float,
        default=1.0,
        help="Exponent gamma in similarity_topk eviction score: max_cosine / leverage**gamma",
    )
    parser.add_argument(
        "--leverage_eviction_risk_mode",
        "--leverage-eviction-risk-mode",
        type=str,
        default="low_leverage",
        choices=("low_leverage", "outlier_then_low"),
        help="SVD leverage eviction risk mode: existing low-leverage eviction or direct high-outlier eviction before low-score selection",
    )
    parser.add_argument(
        "--leverage_high_outlier_z",
        "--leverage-high-outlier-z",
        type=float,
        default=3.0,
        help="Robust z-score threshold for direct high-leverage outlier eviction when leverage_eviction_risk_mode=outlier_then_low",
    )
    parser.add_argument(
        "--leverage_dpp_candidate_multiplier",
        "--leverage-dpp-candidate-multiplier",
        type=int,
        default=2,
        help="Fast DPP low-score candidate pool multiplier relative to eviction count",
    )
    parser.add_argument(
        "--leverage_dpp_greedy_block_size",
        "--leverage-dpp-greedy-block-size",
        type=int,
        default=32,
        help="Fast DPP greedy selection block size; 1 is quality-oriented, larger values favor speed",
    )
    parser.add_argument(
        "--leverage_dpp_quality_beta",
        "--leverage-dpp-quality-beta",
        type=float,
        default=1.0,
        help="Fast DPP quality log-term weight; 0 removes quality from DPP selection",
    )
    parser.add_argument(
        "--leverage_dpp_diversity_beta",
        "--leverage-dpp-diversity-beta",
        type=float,
        default=1.0,
        help="Fast DPP diversity log-term weight; 1.0 preserves the default DPP balance",
    )
    parser.add_argument(
        "--leverage_dpp_feature_projection",
        "--leverage-dpp-feature-projection",
        type=str,
        default="raw",
        choices=("raw", "random"),
        help="Feature projection for Fast DPP diversity similarity; random reuses leverage_ridge_dim",
    )
    parser.add_argument("--leverage_dpp_recency_bonus", "--leverage-dpp-recency-bonus", action="store_true", help="Add a soft recency bonus to SVD leverage eviction scores")
    parser.add_argument("--leverage_dpp_recency_lambda", "--leverage-dpp-recency-lambda", type=float, default=0.2, help="Strength of the eviction-score recency bonus")
    parser.add_argument("--leverage_dpp_recency_window", "--leverage-dpp-recency-window", type=int, default=10, help="Frame window for linear eviction-score freshness")
    parser.add_argument("--leverage_dpp_recency_gate_power", "--leverage-dpp-recency-gate-power", type=float, default=1.0, help="Power applied to the low-score gate for eviction-score recency")
    parser.add_argument("--leverage_dpp_recency_debug", "--leverage-dpp-recency-debug", action="store_true", help="Print eviction-score recency bonus summary statistics")
    parser.add_argument(
        "--layer_budget_strategy",
        "--layer-budget-strategy",
        type=str,
        default="uniform",
        choices=("uniform", "cosine_precomputed", "leverage_pr", "covariance_pr", "hybrid_cap", "hybrid_geom", "leverage_entropy", "value_weighted_leverage_pr", "value_weighted_covariance_pr", "value_weighted_hybrid_cap", "value_weighted_hybrid_geom"),
        help="Layer-wise KV budget allocation strategy",
    )
    parser.add_argument(
        "--layer_budget_proportions_path",
        "--layer-budget-proportions-path",
        type=str,
        default=None,
        help="JSON file containing fixed proportions for cosine_precomputed layer budgets",
    )
    parser.add_argument("--layer_budget_alpha", "--layer-budget-alpha", type=float, default=0.5)
    parser.add_argument("--layer_budget_min_tokens", "--layer-budget-min-tokens", type=int, default=0)
    parser.add_argument("--layer_budget_eps", "--layer-budget-eps", type=float, default=1e-12)
    parser.add_argument("--layer_budget_value_gamma", "--layer-budget-value-gamma", type=float, default=0.5)
    parser.add_argument("--slots_per_direction", "--slots-per-direction", type=float, default=4.0)
    parser.add_argument("--hybrid_beta", "--hybrid-beta", type=float, default=0.5)
    parser.add_argument(
        "--layer_budget_value_norm_type",
        "--layer-budget-value-norm-type",
        type=str,
        default="rms",
        choices=("mean", "rms"),
        help="Layer value-norm prior type for value_weighted_leverage_pr budget allocation",
    )
    parser.add_argument(
        "--layer_budget_norm_source",
        "--layer-budget-norm-source",
        type=str,
        default="value",
        choices=("value", "key"),
        help="Tensor source for value_weighted_leverage_pr norm prior: value cache or key cache",
    )
    parser.add_argument(
        "--layer_budget_log_scores",
        "--layer-budget-log-scores",
        action="store_true",
        help="Write per-step layer budget scores to layer_budget_scores.csv under each sequence output directory",
    )
    parser.add_argument(
        "--layer_budget_log_path",
        "--layer-budget-log-path",
        type=str,
        default=None,
        help="Optional explicit CSV path for layer budget score logs",
    )
    parser.add_argument(
        "--eviction_protect_recent_frames",
        "--eviction-protect-recent-frames",
        type=int,
        default=0,
        help=(
            "Protect tokens from the most recent N processed frames from eviction while still "
            "including them in SVD leverage computation."
        ),
    )
    parser.add_argument(
        "--eviction_protect_special_tokens",
        "--eviction-protect-special-tokens",
        action="store_true",
        help="Protect cached camera/CLS and register tokens from eviction.",
    )
    parser.add_argument(
        "--eviction_protect_special_token_interval",
        "--eviction-protect-special-token-interval",
        type=int,
        default=1,
        help="Protect special tokens only for processed frame IDs divisible by N; 1 protects every cached frame.",
    )
    parser.add_argument(
        "--kf_interval",
        "--kf-interval",
        type=int,
        default=1,
        help="Cache KV only for every Nth keyframe while still reading cached keyframe KV on intermediate frames",
    )
    parser.add_argument(
        "--evict_interval",
        "--evict-interval",
        type=int,
        default=1,
        help="Run cache eviction only every N cached keyframes; 1 preserves current behavior",
    )
    parser.add_argument(
        "--global_cache_history_anchor_special_tokens_only",
        "--global-cache-history-anchor-special-tokens-only",
        action="store_true",
        help="Store global KV special tokens only for frame 0 and active history anchor frames.",
    )
    parser.add_argument(
        "--first_frame_special_tokens_only",
        "--first-frame-special-tokens-only",
        action="store_true",
        help="Protect only first-frame special tokens as the global anchor; first-frame patch tokens can be evicted.",
    )
    parser.add_argument(
        "--camera_cache_history_anchors_only",
        "--camera-cache-history-anchors-only",
        action="store_true",
        help="Keep camera-head KV cache only for frame 0 and active history anchor frames.",
    )
    parser.add_argument(
        "--camera_cache_keep_dropped_anchors",
        "--camera-cache-keep-dropped-anchors",
        action="store_true",
        help="With camera anchor-only cache, keep camera KV for anchors even after FIFO demotion.",
    )
    parser.add_argument(
        "--history_anchor_strategy",
        "--history-anchor-strategy",
        type=str,
        default="none",
        choices=("none", "fixed_interval", "coverage", "camera_motion"),
        help="History Anchor selection strategy",
    )
    parser.add_argument("--anchor_interval", "--anchor-interval", type=int, default=250)
    parser.add_argument("--min_anchor_interval", "--min-anchor-interval", type=int, default=100)
    parser.add_argument("--window_protect_frames", "--window-protect-frames", type=int, default=0)
    parser.add_argument("--max_anchors", "--max-anchors", type=int, default=0)
    parser.add_argument("--coverage_threshold", "--coverage-threshold", type=float, default=0.2)
    parser.add_argument("--camera_motion_threshold", "--camera-motion-threshold", type=float, default=0.2)
    parser.add_argument("--anchor_keep_ratio", "--anchor-keep-ratio", type=float, default=0.05)
    parser.add_argument(
        "--history_anchor_patch_topk_per_frame",
        "--history-anchor-patch-topk-per-frame",
        type=int,
        default=0,
        help="Protect special tokens plus leverage top-k patch tokens for selected history anchor frames; 0 disables.",
    )
    parser.add_argument(
        "--profile_eviction",
        "--profile-eviction",
        action="store_true",
        help="Print per-eviction svd_leverage timing/profile fields without changing eviction behavior",
    )
    add_eviction_nn_analysis_args(parser)
    add_leverage_score_histogram_args(parser)
    add_token_overlay_dump_args(parser)
    parser.add_argument(
        "--budget",
        type=int,
        default=200000,
        help="Total token budget for StreamVGGT inference",
    )

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument(
        "--first_seq_only",
        "--first-seq-only",
        action="store_true",
        help="Run only the first sorted sequence from the resolved sequence list.",
    )
    parser.add_argument(
        "--stream_depth_save",
        "--stream-depth-save",
        action="store_true",
        help="Write each predicted depth .npy during inference instead of accumulating a full sequence before saving.",
    )
    return parser


def summarize_layer_budget_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return
    script_path = Path(__file__).resolve().parents[3] / "tools" / "summarize_layer_budget_scores.py"
    if not script_path.exists():
        print(f"[LayerBudget] summary script not found: {script_path}")
        return
    try:
        subprocess.run(
            [sys.executable, str(script_path), str(log_path)],
            check=True,
        )
    except Exception as exc:
        print(f"[LayerBudget] failed to summarize {log_path}: {exc}")



def save_stream_depth_frame(seq_save_dir, frame_idx, result):
    os.makedirs(seq_save_dir, exist_ok=True)
    depth = result["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu()
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    np.save(os.path.join(seq_save_dir, f"frame_{frame_idx:04d}.npy"), depth.astype(np.float32, copy=False))


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

    seq_list = args.seq_list
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
    if args.first_seq_only:
        if not seq_list:
            raise RuntimeError(
                f"No sequences found for eval_dataset={args.eval_dataset}, img_path={img_path}"
            )
        seq_list = seq_list[11:12]
        print(f"[video_depth] first_seq_only: running sequence {seq_list[0]}")

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
        assert load_img_size == 518
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

                filelist_func = metadata.get("filelist_func", None)
                if filelist_func is not None:
                    filelist = filelist_func(dir_path)
                else:
                    filelist = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                    filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]
                if args.max_frames is not None:
                    filelist = filelist[: args.max_frames]

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                )
                for view in views:
                    view["img"] = (view["img"] + 1.0) / 2.0
                start = time.time()
                safe_seq = str(seq).replace("/", "_").replace(os.sep, "_").replace(" ", "_")
                rank_label = f"rank_{distributed_state.process_index}"
                eviction_nn_analysis_config = None
                if args.eviction_nn_analysis_dir:
                    nn_dir = os.path.join(
                        args.eviction_nn_analysis_dir,
                        args.eval_dataset,
                        rank_label,
                        safe_seq,
                    )
                    eviction_nn_analysis_config = eviction_nn_config_from_args(args, output_dir=nn_dir)
                leverage_score_histogram_config = None
                if args.leverage_score_histogram_dir:
                    hist_dir = os.path.join(
                        args.leverage_score_histogram_dir,
                        args.eval_dataset,
                        rank_label,
                        safe_seq,
                    )
                    leverage_score_histogram_config = leverage_score_histogram_config_from_args(args, output_dir=hist_dir)
                token_overlay_dump_config = None
                if args.token_overlay_dump_dir:
                    overlay_dump_dir = os.path.join(
                        args.token_overlay_dump_dir,
                        args.eval_dataset,
                        rank_label,
                        safe_seq,
                    )
                    token_overlay_dump_config = token_overlay_dump_config_from_args(args, output_dir=overlay_dump_dir)
                layer_budget_log_path = None
                if args.layer_budget_log_path:
                    layer_budget_log_path = args.layer_budget_log_path
                elif args.layer_budget_log_scores:
                    layer_budget_log_path = os.path.join(save_dir, safe_seq, "layer_budget_scores.csv")
                seq_save_dir = f"{save_dir}/{seq}"
                frame_writer = None
                if args.stream_depth_save:
                    frame_writer = lambda frame_idx, frame, result, seq_save_dir=seq_save_dir: save_stream_depth_frame(
                        seq_save_dir, frame_idx, result
                    )

                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.cuda.amp.autocast(dtype=dtype):
                    output = model.inference(
                        views,
                        frame_writer=frame_writer,
                        cache_results=not args.stream_depth_save,
                        eviction_policy=args.eviction_policy,
                        leverage_sketch_dim=args.leverage_sketch_dim,
                        leverage_granularity=args.leverage_granularity,
                        leverage_feature=args.leverage_feature,
                        leverage_projection=args.leverage_projection,
                        leverage_head_mean_dim=args.leverage_head_mean_dim,
                        leverage_normalize_rows=args.leverage_normalize_rows,
                        leverage_approx_method=args.leverage_approx_method,
                                    leverage_ridge_lambda=args.leverage_ridge_lambda,
                        leverage_ridge_lambda_mode=args.leverage_ridge_lambda_mode,
                        leverage_ridge_score_chunk_size=args.leverage_ridge_score_chunk_size,
                        leverage_ridge_jitter=args.leverage_ridge_jitter,
                        leverage_ridge_dim=args.leverage_ridge_dim,
                        leverage_diag=args.leverage_diag,
                        leverage_diag_interval=args.leverage_diag_interval,
                        leverage_random_seed=args.leverage_random_seed,
                        leverage_eviction_selector=args.leverage_eviction_selector,
                        leverage_similarity_granularity=args.leverage_similarity_granularity,
                        leverage_similarity_feature_projection=args.leverage_similarity_feature_projection,
                        leverage_similarity_leverage_gamma=args.leverage_similarity_leverage_gamma,
                        leverage_eviction_risk_mode=args.leverage_eviction_risk_mode,
                        leverage_high_outlier_z=args.leverage_high_outlier_z,
                        leverage_dpp_candidate_multiplier=args.leverage_dpp_candidate_multiplier,
                        leverage_dpp_greedy_block_size=args.leverage_dpp_greedy_block_size,
                        leverage_dpp_quality_beta=args.leverage_dpp_quality_beta,
                        leverage_dpp_diversity_beta=args.leverage_dpp_diversity_beta,
                        leverage_dpp_feature_projection=args.leverage_dpp_feature_projection,
                        leverage_dpp_recency_bonus=args.leverage_dpp_recency_bonus,
                        leverage_dpp_recency_lambda=args.leverage_dpp_recency_lambda,
                        leverage_dpp_recency_window=args.leverage_dpp_recency_window,
                        leverage_dpp_recency_gate_power=args.leverage_dpp_recency_gate_power,
                        leverage_dpp_recency_debug=args.leverage_dpp_recency_debug,
                        layer_budget_strategy=args.layer_budget_strategy,
                        layer_budget_value_gamma=args.layer_budget_value_gamma,
                        layer_budget_value_norm_type=args.layer_budget_value_norm_type,
                        layer_budget_norm_source=args.layer_budget_norm_source,
                        layer_budget_alpha=args.layer_budget_alpha,
                        layer_budget_min_tokens=args.layer_budget_min_tokens,
                        layer_budget_eps=args.layer_budget_eps,
                        slots_per_direction=args.slots_per_direction,
                        hybrid_beta=args.hybrid_beta,
                        layer_budget_log_path=layer_budget_log_path,
                        eviction_protect_recent_frames=args.eviction_protect_recent_frames,
                        eviction_protect_special_tokens=args.eviction_protect_special_tokens,
                        eviction_protect_special_token_interval=args.eviction_protect_special_token_interval,
                        history_anchor_strategy=args.history_anchor_strategy,
                        anchor_interval=args.anchor_interval,
                        min_anchor_interval=args.min_anchor_interval,
                        window_protect_frames=args.window_protect_frames,
                        max_anchors=args.max_anchors,
                        coverage_threshold=args.coverage_threshold,
                        camera_motion_threshold=args.camera_motion_threshold,
                        anchor_keep_ratio=args.anchor_keep_ratio,
                        history_anchor_patch_topk_per_frame=args.history_anchor_patch_topk_per_frame,
                        eviction_debug=args.profile_eviction,
                        eviction_nn_analysis_config=eviction_nn_analysis_config,
                        leverage_score_histogram_config=leverage_score_histogram_config,
                        token_overlay_dump_config=token_overlay_dump_config,
                        kf_interval=args.kf_interval,
                        evict_interval=args.evict_interval,
                        global_cache_history_anchor_special_tokens_only=args.global_cache_history_anchor_special_tokens_only,

                        first_frame_special_tokens_only=args.first_frame_special_tokens_only,
                        camera_cache_history_anchors_only=args.camera_cache_history_anchors_only,
                        camera_cache_keep_dropped_anchors=args.camera_cache_keep_dropped_anchors,
                    )
                    outputs = dict(views=output.views, pred=output.ress)
                if leverage_score_histogram_config is not None:
                    leverage_score_histogram_config.flush()
                end = time.time()
                # fps = len(filelist) / (end - start)
                if args.stream_depth_save:
                    summarize_layer_budget_log(layer_budget_log_path)
                else:
                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        (
                            pts3ds_self,
                            conf_self,
                        ) = prepare_output(outputs)

                        os.makedirs(seq_save_dir, exist_ok=True)
                        save_depth_maps(pts3ds_self, seq_save_dir, conf_self=conf_self)
                        summarize_layer_budget_log(layer_budget_log_path)

                del views
                if "output" in locals():
                    del output
                if "outputs" in locals():
                    del outputs
                if "pts3ds_self" in locals():
                    del pts3ds_self
                if "conf_self" in locals():
                    del conf_self
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence: {str(e)}\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping: {e}")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    return None, None, None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit(f"Error: --max_frames must be >= 1, got {args.max_frames}.")
    if args.leverage_head_mean_dim < 1:
        raise SystemExit(
            "Error: --leverage_head_mean_dim must be >= 1, "
            f"got {args.leverage_head_mean_dim}."
        )
    if args.leverage_high_outlier_z < 0:
        raise SystemExit(
            "Error: --leverage_high_outlier_z must be >= 0, "
            f"got {args.leverage_high_outlier_z}."
        )
    if args.leverage_score_histogram_bins < 1:
        raise SystemExit(
            "Error: --leverage_score_histogram_bins must be >= 1, "
            f"got {args.leverage_score_histogram_bins}."
        )
    if args.leverage_score_histogram_max <= args.leverage_score_histogram_min:
        raise SystemExit(
            "Error: --leverage_score_histogram_max must be greater than --leverage_score_histogram_min, "
            f"got min={args.leverage_score_histogram_min}, max={args.leverage_score_histogram_max}."
        )

    if args.leverage_dpp_candidate_multiplier < 1:
        raise SystemExit(
            "Error: --leverage_dpp_candidate_multiplier must be >= 1, "
            f"got {args.leverage_dpp_candidate_multiplier}."
        )
    if args.leverage_dpp_greedy_block_size < 1:
        raise SystemExit(
            "Error: --leverage_dpp_greedy_block_size must be >= 1, "
            f"got {args.leverage_dpp_greedy_block_size}."
        )
    if args.leverage_dpp_quality_beta < 0:
        raise SystemExit(
            "Error: --leverage_dpp_quality_beta must be >= 0, "
            f"got {args.leverage_dpp_quality_beta}."
        )
    if args.leverage_dpp_diversity_beta < 0:
        raise SystemExit(
            "Error: --leverage_dpp_diversity_beta must be >= 0, "
            f"got {args.leverage_dpp_diversity_beta}."
        )
    if args.leverage_dpp_feature_projection == "random" and args.leverage_ridge_dim is None:
        raise SystemExit(
            "Error: --leverage_dpp_feature_projection random requires "
            "--leverage_ridge_dim >= 1."
        )
    if args.leverage_dpp_recency_lambda < 0:
        raise SystemExit(
            "Error: --leverage_dpp_recency_lambda must be >= 0, "
            f"got {args.leverage_dpp_recency_lambda}."
        )
    if args.leverage_dpp_recency_window < 1:
        raise SystemExit(
            "Error: --leverage_dpp_recency_window must be >= 1, "
            f"got {args.leverage_dpp_recency_window}."
        )
    if args.leverage_dpp_recency_gate_power < 0:
        raise SystemExit(
            "Error: --leverage_dpp_recency_gate_power must be >= 0, "
            f"got {args.leverage_dpp_recency_gate_power}."
        )
    if args.layer_budget_alpha < 0:
        raise SystemExit(
            "Error: --layer_budget_alpha must be >= 0, "
            f"got {args.layer_budget_alpha}."
        )
    if args.layer_budget_min_tokens < 0:
        raise SystemExit(
            "Error: --layer_budget_min_tokens must be >= 0, "
            f"got {args.layer_budget_min_tokens}."
        )
    if args.layer_budget_eps <= 0:
        raise SystemExit(
            "Error: --layer_budget_eps must be > 0, "
            f"got {args.layer_budget_eps}."
        )
    if args.layer_budget_value_gamma < 0:
        raise SystemExit(
            "Error: --layer_budget_value_gamma must be >= 0, "
            f"got {args.layer_budget_value_gamma}."
        )
    if args.leverage_ridge_lambda < 0:
        raise SystemExit(
            "Error: --leverage_ridge_lambda must be >= 0, "
            f"got {args.leverage_ridge_lambda}."
        )
    if args.leverage_diag_interval < 0:
        raise SystemExit(
            "Error: --leverage_diag_interval must be >= 0, "
            f"got {args.leverage_diag_interval}."
        )
    if args.leverage_ridge_jitter <= 0:
        raise SystemExit(
            "Error: --leverage_ridge_jitter must be > 0, "
            f"got {args.leverage_ridge_jitter}."
        )
    if args.leverage_ridge_score_chunk_size < 1:
        raise SystemExit(
            "Error: --leverage_ridge_score_chunk_size must be >= 1, "
            f"got {args.leverage_ridge_score_chunk_size}."
        )
    if args.leverage_ridge_dim is not None and args.leverage_ridge_dim < 1:
        raise SystemExit(
            "Error: --leverage_ridge_dim must be >= 1 when provided, "
            f"got {args.leverage_ridge_dim}."
        )
    if args.leverage_approx_method == "right_sketch_ridge" and args.leverage_ridge_dim is None:
        raise SystemExit(
            "Error: --leverage_approx_method right_sketch_ridge requires "
            "--leverage_ridge_dim >= 1."
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
    if args.eviction_protect_recent_frames < 0:
        raise SystemExit(
            "Error: --eviction_protect_recent_frames must be >= 0, "
            f"got {args.eviction_protect_recent_frames}."
        )
    if args.eviction_protect_special_token_interval < 1:
        raise SystemExit(
            "Error: --eviction_protect_special_token_interval must be >= 1, "
            f"got {args.eviction_protect_special_token_interval}."
        )
    if args.kf_interval < 1:
        raise SystemExit(f"Error: --kf_interval must be >= 1, got {args.kf_interval}.")
    if args.evict_interval < 1:
        raise SystemExit(f"Error: --evict_interval must be >= 1, got {args.evict_interval}.")
    if args.anchor_interval < 1:
        raise SystemExit(f"Error: --anchor_interval must be >= 1, got {args.anchor_interval}.")
    if args.min_anchor_interval is not None and args.min_anchor_interval < 0:
        raise SystemExit(
            "Error: --min_anchor_interval must be >= 0, "
            f"got {args.min_anchor_interval}."
        )
    if args.window_protect_frames < 0:
        raise SystemExit(
            "Error: --window_protect_frames must be >= 0, "
            f"got {args.window_protect_frames}."
        )
    if args.max_anchors < 0:
        raise SystemExit(f"Error: --max_anchors must be >= 0, got {args.max_anchors}.")
    if not (0.0 <= args.coverage_threshold <= 1.0):
        raise SystemExit(
            "Error: --coverage_threshold must be in [0, 1], "
            f"got {args.coverage_threshold}."
        )
    if args.camera_motion_threshold < 0.0:
        raise SystemExit(
            "Error: --camera_motion_threshold must be >= 0, "
            f"got {args.camera_motion_threshold}."
        )
    if not (0.0 <= args.anchor_keep_ratio <= 1.0):
        raise SystemExit(
            "Error: --anchor_keep_ratio must be in [0, 1], "
            f"got {args.anchor_keep_ratio}."
        )
    if args.history_anchor_patch_topk_per_frame < 0:
        raise SystemExit(
            "Error: --history_anchor_patch_topk_per_frame must be >= 0, "
            f"got {args.history_anchor_patch_topk_per_frame}."
        )
    try:
        eviction_nn_config_from_args(args, output_dir=args.eviction_nn_analysis_dir)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if args.eviction_policy == "svd_leverage":
        sketch_label = "exact" if args.leverage_sketch_dim == 0 else str(args.leverage_sketch_dim)
        print(
            "Using SVD leverage eviction: "
            f"sketch_dim={sketch_label}, "
            f"granularity={args.leverage_granularity}, "
            f"feature={args.leverage_feature}, "
            f"approx={args.leverage_approx_method}, "
            f"ridge_dim={args.leverage_ridge_dim}, ridge_lambda={args.leverage_ridge_lambda}, "
            f"ridge_lambda_mode={args.leverage_ridge_lambda_mode}, "
            f"ridge_chunk={args.leverage_ridge_score_chunk_size}, ridge_jitter={args.leverage_ridge_jitter}, "
            f"projection={args.leverage_projection}, "
            f"head_mean_dim={args.leverage_head_mean_dim}, "
            f"normalize_rows={args.leverage_normalize_rows}, "
            f"selector={args.leverage_eviction_selector}, "
            f"risk_mode={args.leverage_eviction_risk_mode}, "
            f"high_outlier_z={args.leverage_high_outlier_z}, "
            f"dpp_candidate_multiplier={args.leverage_dpp_candidate_multiplier}, "
            f"dpp_greedy_block_size={args.leverage_dpp_greedy_block_size}, "
            f"dpp_quality_beta={args.leverage_dpp_quality_beta}, "
            f"dpp_diversity_beta={args.leverage_dpp_diversity_beta}, "
            f"dpp_feature_projection={args.leverage_dpp_feature_projection}, "
            f"layer_budget_strategy={args.layer_budget_strategy}, "
            f"layer_budget_alpha={args.layer_budget_alpha}, "
            f"layer_budget_min_tokens={args.layer_budget_min_tokens}, "
            f"layer_budget_value_gamma={args.layer_budget_value_gamma}, "
            f"layer_budget_value_norm_type={args.layer_budget_value_norm_type}, "
            f"layer_budget_norm_source={args.layer_budget_norm_source}, "
            f"protect_recent_frames={args.eviction_protect_recent_frames}, "
            f"kf_interval={args.kf_interval}, "
            f"evict_interval={args.evict_interval}"
        )
    elif args.eviction_policy == "dpp":
        print(
            "Using DPP-only eviction: "
            f"granularity={args.leverage_granularity}, "
            f"feature={args.leverage_feature}, "
            f"projection={args.leverage_projection}, "
            f"head_mean_dim={args.leverage_head_mean_dim}, "
            f"dpp_greedy_block_size={args.leverage_dpp_greedy_block_size}, "
            f"dpp_quality_beta={args.leverage_dpp_quality_beta}, "
            f"dpp_diversity_beta={args.leverage_dpp_diversity_beta}, "
            f"dpp_feature_projection={args.leverage_dpp_feature_projection}, "
            f"protect_recent_frames={args.eviction_protect_recent_frames}"
        )
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    from streamvggt.utils.geometry import unproject_depth_map_to_point_map
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"].to(device='cuda'),
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ).to(device='cuda'),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]).to(device='cuda'),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0).to(device='cuda'),
                    "img_mask": torch.tensor(True).unsqueeze(0).to(device='cuda'),
                    "ray_mask": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                    "update": torch.tensor(True).unsqueeze(0).to(device='cuda'),
                    "reset": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"].to(device='cuda')
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan).to(device='cuda')
                    ),
                    "ray_map": (
                        raymaps[k].to(device='cuda')
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan).to(device='cuda')
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"]).to(device='cuda')
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]])).to(device='cuda')
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0).to(device='cuda'),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0).to(device='cuda'),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0).to(device='cuda'),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0).to(device='cuda'),
                    "reset": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        pts3ds_self = [output["depth"].cpu() for output in outputs["pred"]]
        conf_self = [output["depth_conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self, 0)
        return (
            pts3ds_self,
            conf_self,
        )

    model = StreamVGGT(total_budget=args.budget)
    ckpt = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(ckpt, strict=True)
    if args.layer_budget_strategy == "cosine_precomputed":
        model.set_layer_budget_proportions(args.layer_budget_proportions_path)
        print(f"Loaded precomputed layer budget proportions: {args.layer_budget_proportions_path}")
    model.eval()
    model = model.to("cuda")
    del ckpt
    with torch.no_grad():
        eval_pose_estimation(args, model, save_dir=args.output_dir)
