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
from streamvggt.layers.recent_merge import RecentMergeConfig
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig

from tqdm import tqdm
import time


def resolve_global_attn_idx_ranges(args):
    if args.middle_global_only and args.global_attn_idx_ranges is not None:
        raise ValueError("--middle-global-only cannot be combined with --global-attn-idx-ranges")
    if args.middle_global_only:
        return "9:"
    return args.global_attn_idx_ranges


def validate_streamvggt_args(args):
    try:
        global_attn_idx_ranges = resolve_global_attn_idx_ranges(args)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if global_attn_idx_ranges is not None:
        print(f"Global attention index ranges enabled: {global_attn_idx_ranges}")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit(f"Error: --max_frames must be >= 1, got {args.max_frames}.")
    if args.kf_every < 1:
        raise SystemExit(f"Error: --kf_every must be >= 1, got {args.kf_every}.")
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
    return global_attn_idx_ranges


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
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--eviction_policy", type=str, default="mean", help="Cache eviction policy: mean, baseline_mean, svd_leverage, or dpp")
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
        "--leverage_approx_method",
        "--leverage-approx-method",
        type=str,
        default="right_sketch",
        choices=("exact_qr", "right_sketch", "drineas_srht", "full_d_ridge", "right_sketch_ridge"),
        help="Leverage approximation: exact QR, right-sketched/Compactor-style, or Drineas left SRHT",
    )
    parser.add_argument(
        "--leverage_left_sketch_dim",
        "--leverage-left-sketch-dim",
        type=int,
        default=2048,
        help="Left SRHT row count r1 for leverage_approx_method drineas_srht",
    )
    parser.add_argument(
        "--leverage_right_jl_dim",
        "--leverage-right-jl-dim",
        type=int,
        default=64,
        help="Right JL dimension r2 for leverage_approx_method drineas_srht; <=0 uses no right JL",
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
    parser.add_argument("--max_anchors", "--max-anchors", type=int, default=3)
    parser.add_argument("--coverage_threshold", "--coverage-threshold", type=float, default=0.2)
    parser.add_argument("--anchor_keep_ratio", "--anchor-keep-ratio", type=float, default=0.05)
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

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
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

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
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
    global_attn_idx_ranges = validate_streamvggt_args(args)

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

                images = load_and_preprocess_images(filelist).to(device)
                frames = []
                for i in range(images.shape[0]):
                    image = images[i].unsqueeze(0) 
                    frame = {
                        "img": image
                    }
                    frames.append(frame)

                start = time.time()
                predictions = {}
                with torch.no_grad():
                    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
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
                        output = model.inference(
                            frames,
                            eviction_policy=args.eviction_policy,
                            leverage_sketch_dim=args.leverage_sketch_dim,
                            leverage_granularity=args.leverage_granularity,
                            leverage_feature=args.leverage_feature,
                            leverage_projection=args.leverage_projection,
                            leverage_head_mean_dim=args.leverage_head_mean_dim,
                            leverage_approx_method=args.leverage_approx_method,
                            leverage_left_sketch_dim=args.leverage_left_sketch_dim,
                            leverage_right_jl_dim=args.leverage_right_jl_dim,
                            leverage_ridge_lambda=args.leverage_ridge_lambda,
                            leverage_ridge_lambda_mode=args.leverage_ridge_lambda_mode,
                            leverage_ridge_score_chunk_size=args.leverage_ridge_score_chunk_size,
                            leverage_ridge_jitter=args.leverage_ridge_jitter,
                            leverage_ridge_dim=args.leverage_ridge_dim,
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
                            recent_merge_config=recent_merge_config,
                            svd_eviction_merge_config=svd_eviction_merge_config,
                            voxel_covis_config=voxel_covis_config,
                            global_attn_idx_ranges=global_attn_idx_ranges,
                            global_attn_debug=args.global_attn_debug,
                            kf_interval=args.kf_interval,
                            evict_interval=args.evict_interval,
                        )
                end = time.time()
                fps = len(filelist) / (end - start)
                print(f"Finished pose estimation for {args.eval_dataset} {seq: <16}, FPS: {fps:.2f}")

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
