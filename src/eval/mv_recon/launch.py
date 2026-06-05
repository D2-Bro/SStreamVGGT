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
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict
from streamvggt.layers.recent_merge import RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig
from streamvggt.utils.cache_analysis import add_eviction_nn_analysis_args, eviction_nn_config_from_args


def wait_for_rank_logs(
    save_path, dataset_name, num_processes, min_mtime, timeout_s=6000, poll_s=2
):
    done_dir = osp.join(save_path, ".rank_done")
    os.makedirs(done_dir, exist_ok=True)

    deadline = time.time() + timeout_s
    pending = set(range(num_processes))
    while pending and time.time() < deadline:
        for rank in list(pending):
            marker = osp.join(done_dir, f"rank_{rank}.json")
            if not osp.exists(marker):
                continue
            try:
                if osp.getmtime(marker) < min_mtime:
                    continue
                with open(marker, "r") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("dataset") == dataset_name and data.get("rank") == rank:
                pending.remove(rank)
        if pending:
            time.sleep(poll_s)

    if pending:
        raise TimeoutError(
            f"Timed out waiting for rank log completion markers in {save_path}: "
            f"missing ranks {sorted(pending)}"
        )


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
    parser.add_argument("--eviction_policy", type=str, default="mean", help="Cache eviction policy: mean, baseline_mean, svd_leverage, or dpp")
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
        default="exact_qr",
        choices=("exact_qr", "right_sketch", "drineas_srht", "full_d_ridge", "right_sketch_ridge"),
        help="Leverage approximation: exact QR, right-sketched/Compactor-style, or Drineas left SRHT",
    )
    parser.add_argument(
        "--leverage_left_sketch_dim",
        "--leverage-left-sketch-dim",
        type=int,
        default=2048,
        help="Left SRHT row count r1 for leverage_approx_method=drineas_srht",
    )
    parser.add_argument(
        "--leverage_right_jl_dim",
        "--leverage-right-jl-dim",
        type=int,
        default=64,
        help="Right JL dimension r2 for leverage_approx_method=drineas_srht; <=0 uses no right JL",
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
        help="Projection dimension for right_sketch_ridge; defaults to --leverage_right_jl_dim when omitted",
    )
    parser.add_argument(
        "--leverage_random_seed",
        "--leverage-random-seed",
        type=int,
        default=0,
        help="Random seed for leverage sketches",
    )
    parser.add_argument(
        "--leverage_eviction_selector",
        "--leverage-eviction-selector",
        type=str,
        default="topk",
        choices=("topk", "fast_dpp"),
        help="Eviction selector for svd_leverage scores: topk or low-score Fast DPP",
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
        "--leverage_dpp_diversity_beta",
        "--leverage-dpp-diversity-beta",
        type=float,
        default=1.0,
        help="Fast DPP diversity log-term weight; 1.0 preserves the default DPP balance",
    )
    parser.add_argument(
        "--layer_budget_strategy",
        "--layer-budget-strategy",
        type=str,
        default="uniform",
        choices=("uniform", "cosine_precomputed", "leverage_pr", "leverage_entropy", "value_weighted_leverage_pr"),
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
        "--layer_budget_debug",
        "--layer-budget-debug",
        action="store_true",
        help="Print layer-wise budget allocation details",
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
        "--eviction_debug",
        "--eviction-debug",
        action="store_true",
        help="Print eviction summaries and svd_leverage timing/profile fields",
    )
    parser.add_argument(
        "--history_anchor_strategy",
        "--history-anchor-strategy",
        type=str,
        default="none",
        choices=("none", "fixed_interval", "coverage"),
        help="History Anchor selection strategy",
    )
    parser.add_argument("--anchor_interval", "--anchor-interval", type=int, default=250)
    parser.add_argument("--min_anchor_interval", "--min-anchor-interval", type=int, default=100)
    parser.add_argument("--window_protect_frames", "--window-protect-frames", type=int, default=0)
    parser.add_argument("--max_anchors", "--max-anchors", type=int, default=0)
    parser.add_argument("--coverage_threshold", "--coverage-threshold", type=float, default=0.2)
    parser.add_argument("--anchor_keep_ratio", "--anchor-keep-ratio", type=float, default=0.05)
    parser.add_argument(
        "--profile_eviction",
        "--profile-eviction",
        action="store_true",
        help="Print per-eviction svd_leverage timing/profile fields without changing eviction behavior",
    )
    add_eviction_nn_analysis_args(parser)
    parser.add_argument(
        "--enable_svd_eviction_merge",
        "--enable-svd-eviction-merge",
        action="store_true",
        help="Enable feature-first SVD-guided merge for tokens selected by svd_leverage eviction",
    )
    parser.add_argument(
        "--svd_eviction_merge_mode",
        "--svd-eviction-merge-mode",
        choices=("head", "layer_candidates", "layer"),
        default="head",
    )
    parser.add_argument("--svd_eviction_merge_candidate_axes", "--svd-eviction-merge-candidate-axes", type=int, default=2)
    parser.add_argument("--svd_eviction_merge_reps_per_axis", "--svd-eviction-merge-reps-per-axis", type=int, default=8)
    parser.add_argument("--svd_eviction_merge_similarity_threshold", "--svd-eviction-merge-similarity-threshold", type=float, default=0.9)
    parser.add_argument(
        "--svd_eviction_merge_use_u_sigma",
        "--svd-eviction-merge-use-u-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--svd_eviction_merge_geometry_gate",
        "--svd-eviction-merge-geometry-gate",
        choices=("none", "voxel_neighbor"),
        default="voxel_neighbor",
    )
    parser.add_argument("--svd_eviction_merge_voxel_neighbor_radius", "--svd-eviction-merge-voxel-neighbor-radius", type=int, default=1)
    parser.add_argument(
        "--svd_eviction_merge_allow_missing_geometry",
        "--svd-eviction-merge-allow-missing-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--svd_eviction_merge_ema_decay", "--svd-eviction-merge-ema-decay", type=float, default=0.5)
    parser.add_argument(
        "--svd_eviction_merge_use_depth_confidence",
        "--svd-eviction-merge-use-depth-confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--svd_eviction_merge_max_candidates_per_token", "--svd-eviction-merge-max-candidates-per-token", type=int, default=32)
    parser.add_argument("--svd_eviction_merge_chunk_size", "--svd-eviction-merge-chunk-size", type=int, default=512)
    parser.add_argument("--svd_eviction_merge_debug", "--svd-eviction-merge-debug", action="store_true")
    parser.add_argument("--svd_eviction_merge_profile", "--svd-eviction-merge-profile", action="store_true")
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
        "--use_voxel_covis",
        action="store_true",
        help="Enable read-only voxel covisibility filtering for streaming KV cache reads",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size in world units for covisibility frame selection",
    )
    parser.add_argument(
        "--covis_min_shared_voxels",
        type=int,
        default=20,
        help="Minimum shared voxels required for a covisible frame",
    )
    parser.add_argument(
        "--covis_min_overlap",
        type=float,
        default=0.05,
        help="Minimum shared/min voxel overlap required for a covisible frame",
    )
    parser.add_argument(
        "--max_covis_frames",
        type=int,
        default=8,
        help="Maximum number of covisible previous frames to read from KV cache; <=0 disables the cap",
    )
    parser.add_argument(
        "--covis_fallback_recent",
        type=int,
        default=1,
        help="Fallback recent frames when covisibility selection is empty",
    )
    parser.add_argument(
        "--covis_debug_log",
        action="store_true",
        help="Print per-frame voxel covisibility KV filtering diagnostics",
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
    if not (0.0 <= args.anchor_keep_ratio <= 1.0):
        raise SystemExit(
            "Error: --anchor_keep_ratio must be in [0, 1], "
            f"got {args.anchor_keep_ratio}."
        )
    if args.leverage_head_mean_dim < 1:
        raise SystemExit(
            "Error: --leverage_head_mean_dim must be >= 1, "
            f"got {args.leverage_head_mean_dim}."
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
    if args.leverage_dpp_diversity_beta < 0:
        raise SystemExit(
            "Error: --leverage_dpp_diversity_beta must be >= 0, "
            f"got {args.leverage_dpp_diversity_beta}."
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
    if args.leverage_approx_method == "right_sketch_ridge":
        resolved_ridge_dim = args.leverage_ridge_dim if args.leverage_ridge_dim is not None else args.leverage_right_jl_dim
        if resolved_ridge_dim is None or int(resolved_ridge_dim) < 1:
            raise SystemExit(
                "Error: --leverage_approx_method right_sketch_ridge requires "
                "--leverage_ridge_dim >= 1 or --leverage_right_jl_dim >= 1."
            )
    if args.layer_budget_strategy in ("leverage_pr", "leverage_entropy", "value_weighted_leverage_pr") and (
        args.eviction_policy != "svd_leverage" or args.leverage_granularity != "layer"
    ):
        raise SystemExit(
            "Error: leverage-based --layer_budget_strategy requires "
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
    if args.eviction_policy == "svd_leverage":
        sketch_label = "exact" if args.leverage_sketch_dim == 0 else str(args.leverage_sketch_dim)
        print(
            "Using SVD leverage eviction: "
            f"sketch_dim={sketch_label}, "
            f"granularity={args.leverage_granularity}, "
            f"feature={args.leverage_feature}, "
            f"approx={args.leverage_approx_method}, "
            f"r1={args.leverage_left_sketch_dim}, r2={args.leverage_right_jl_dim}, "
            f"ridge_dim={args.leverage_ridge_dim}, ridge_lambda={args.leverage_ridge_lambda}, "
            f"ridge_lambda_mode={args.leverage_ridge_lambda_mode}, "
            f"ridge_chunk={args.leverage_ridge_score_chunk_size}, ridge_jitter={args.leverage_ridge_jitter}, "
            f"projection={args.leverage_projection}, "
            f"head_mean_dim={args.leverage_head_mean_dim}, "
            f"normalize_rows={args.leverage_normalize_rows}, "
            f"selector={args.leverage_eviction_selector}, "
            f"dpp_candidate_multiplier={args.leverage_dpp_candidate_multiplier}, "
            f"dpp_greedy_block_size={args.leverage_dpp_greedy_block_size}, "
            f"dpp_diversity_beta={args.leverage_dpp_diversity_beta}, "
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
            f"dpp_diversity_beta={args.leverage_dpp_diversity_beta}, "
            f"protect_recent_frames={args.eviction_protect_recent_frames}"
        )
    if args.svd_eviction_merge_candidate_axes < 1:
        raise SystemExit("Error: --svd_eviction_merge_candidate_axes must be >= 1.")
    if args.svd_eviction_merge_reps_per_axis < 1:
        raise SystemExit("Error: --svd_eviction_merge_reps_per_axis must be >= 1.")
    if not (0.0 <= args.svd_eviction_merge_similarity_threshold <= 1.0):
        raise SystemExit("Error: --svd_eviction_merge_similarity_threshold must be in [0, 1].")
    if args.svd_eviction_merge_voxel_neighbor_radius < 0:
        raise SystemExit("Error: --svd_eviction_merge_voxel_neighbor_radius must be >= 0.")
    if not (0.0 <= args.svd_eviction_merge_ema_decay <= 1.0):
        raise SystemExit("Error: --svd_eviction_merge_ema_decay must be in [0, 1].")
    if args.svd_eviction_merge_max_candidates_per_token < 1:
        raise SystemExit("Error: --svd_eviction_merge_max_candidates_per_token must be >= 1.")
    if args.svd_eviction_merge_chunk_size < 1:
        raise SystemExit("Error: --svd_eviction_merge_chunk_size must be >= 1.")
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
    if args.voxel_size <= 0:
        raise SystemExit(f"Error: --voxel_size must be > 0, got {args.voxel_size}.")
    if args.covis_min_shared_voxels < 0:
        raise SystemExit(
            "Error: --covis_min_shared_voxels must be >= 0, "
            f"got {args.covis_min_shared_voxels}."
        )
    if not (0.0 <= args.covis_min_overlap <= 1.0):
        raise SystemExit(
            "Error: --covis_min_overlap must be in [0, 1], "
            f"got {args.covis_min_overlap}."
        )
    if args.covis_fallback_recent < 0:
        raise SystemExit(
            "Error: --covis_fallback_recent must be >= 0, "
            f"got {args.covis_fallback_recent}."
        )

    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD, ETH3D
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
            # ROOT="/data2/dongjae/datasets/7scenes_sfm",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=2,
            max_frames=args.max_frames,
        ),
        # "ETH3D": ETH3D
            # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/home/dongjae/data/neural_rgbd_data",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=2,
            max_frames=args.max_frames,
        ),
    }

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=6000))]
    )
    run_started_at = time.time()
    device = accelerator.device
    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = accelerator.local_process_index
            if device_index >= torch.cuda.device_count():
                device_index = 0
            device = torch.device("cuda", device_index)

        torch.cuda.set_device(device_index)
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
        if args.layer_budget_strategy == "cosine_precomputed":
            model.set_layer_budget_proportions(args.layer_budget_proportions_path)
            print(f"Loaded precomputed layer budget proportions: {args.layer_budget_proportions_path}")
        model.eval()
        model = model.to(device)
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
        model = model.to(device)

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
                                svd_eviction_merge_config = SvdEvictionMergeConfig(
                                    enabled=args.enable_svd_eviction_merge,
                                    mode=args.svd_eviction_merge_mode,
                                    candidate_axes=args.svd_eviction_merge_candidate_axes,
                                    reps_per_axis=args.svd_eviction_merge_reps_per_axis,
                                    similarity_threshold=args.svd_eviction_merge_similarity_threshold,
                                    use_u_sigma=args.svd_eviction_merge_use_u_sigma,
                                    geometry_gate=args.svd_eviction_merge_geometry_gate,
                                    voxel_neighbor_radius=args.svd_eviction_merge_voxel_neighbor_radius,
                                    allow_missing_geometry=args.svd_eviction_merge_allow_missing_geometry,
                                    ema_decay=args.svd_eviction_merge_ema_decay,
                                    use_depth_confidence=args.svd_eviction_merge_use_depth_confidence,
                                    max_candidates_per_token=args.svd_eviction_merge_max_candidates_per_token,
                                    chunk_size=args.svd_eviction_merge_chunk_size,
                                    debug=args.svd_eviction_merge_debug,
                                    profile=args.svd_eviction_merge_profile,
                                )
                                voxel_covis_config = VoxelCovisConfig(
                                    enabled=args.use_voxel_covis,
                                    voxel_size=args.voxel_size,
                                    min_shared_voxels=args.covis_min_shared_voxels,
                                    min_overlap=args.covis_min_overlap,
                                    max_covis_frames=args.max_covis_frames,
                                    fallback_recent=args.covis_fallback_recent,
                                    debug=args.covis_debug_log,
                                )
                                covis_log_fn = None
                                if args.covis_debug_log:
                                    def covis_log_fn(msg):
                                        print(msg)
                                eviction_nn_analysis_config = None
                                if args.eviction_nn_analysis_dir:
                                    scene_label = str(batch[0]["label"][0]).rsplit("/", 1)[0]
                                    safe_scene = scene_label.replace("/", "_").replace(os.sep, "_").replace(" ", "_")
                                    nn_dir = osp.join(
                                        args.eviction_nn_analysis_dir,
                                        name_data,
                                        f"rank_{accelerator.process_index}",
                                        f"{int(data_idx):04d}_{safe_scene}",
                                    )
                                    eviction_nn_analysis_config = eviction_nn_config_from_args(args, output_dir=nn_dir)

                                results = model.inference(
                                    batch,
                                    eviction_policy=args.eviction_policy,
                                    leverage_sketch_dim=args.leverage_sketch_dim,
                                    leverage_granularity=args.leverage_granularity,
                                    leverage_feature=args.leverage_feature,
                                    leverage_projection=args.leverage_projection,
                                    leverage_head_mean_dim=args.leverage_head_mean_dim,
                                    leverage_normalize_rows=args.leverage_normalize_rows,
                                    leverage_approx_method=args.leverage_approx_method,
                                    leverage_left_sketch_dim=args.leverage_left_sketch_dim,
                                    leverage_right_jl_dim=args.leverage_right_jl_dim,
                                    leverage_random_seed=args.leverage_random_seed,
                                    leverage_eviction_selector=args.leverage_eviction_selector,
                                    leverage_dpp_candidate_multiplier=args.leverage_dpp_candidate_multiplier,
                                    leverage_dpp_greedy_block_size=args.leverage_dpp_greedy_block_size,
                                    leverage_dpp_diversity_beta=args.leverage_dpp_diversity_beta,
                                    layer_budget_strategy=args.layer_budget_strategy,
                                    layer_budget_value_gamma=args.layer_budget_value_gamma,
                                    layer_budget_value_norm_type=args.layer_budget_value_norm_type,
                                    layer_budget_norm_source=args.layer_budget_norm_source,
                                    layer_budget_alpha=args.layer_budget_alpha,
                                    layer_budget_min_tokens=args.layer_budget_min_tokens,
                                    layer_budget_eps=args.layer_budget_eps,
                                    layer_budget_debug=args.layer_budget_debug,
                                    eviction_protect_recent_frames=args.eviction_protect_recent_frames,
                                    eviction_protect_special_tokens=args.eviction_protect_special_tokens,
                                    eviction_protect_special_token_interval=args.eviction_protect_special_token_interval,
                                    history_anchor_strategy=args.history_anchor_strategy,
                                    anchor_interval=args.anchor_interval,
                                    min_anchor_interval=args.min_anchor_interval,
                                    window_protect_frames=args.window_protect_frames,
                                    max_anchors=args.max_anchors,
                                    coverage_threshold=args.coverage_threshold,
                                    anchor_keep_ratio=args.anchor_keep_ratio,
                                    eviction_debug=args.eviction_debug or args.profile_eviction,
                                    eviction_nn_analysis_config=eviction_nn_analysis_config,
                                    recent_merge_config=recent_merge_config,
                                    svd_eviction_merge_config=svd_eviction_merge_config,
                                    voxel_covis_config=voxel_covis_config,
                                    covis_log_fn=covis_log_fn,
                                    global_attn_idx_ranges=global_attn_idx_ranges,
                                    global_attn_debug=args.global_attn_debug,
                                    kf_interval=args.kf_interval,
                                    evict_interval=args.evict_interval,
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

            done_dir = osp.join(save_path, ".rank_done")
            os.makedirs(done_dir, exist_ok=True)
            with open(
                osp.join(done_dir, f"rank_{accelerator.process_index}.json"), "w"
            ) as f_done:
                json.dump(
                    {"dataset": name_data, "rank": accelerator.process_index}, f_done
                )

            # The eval loop only needs synchronization before log aggregation. Avoid
            # a late NCCL barrier here because this script does not use DDP collectives,
            # and initializing NCCL at shutdown can timeout if one rank exits early.
            if accelerator.is_main_process:
                wait_for_rank_logs(
                    save_path,
                    name_data,
                    accelerator.num_processes,
                    min_mtime=run_started_at,
                    timeout_s=6000,
                )
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
