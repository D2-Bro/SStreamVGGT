import os

# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
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
import subprocess
from collections import defaultdict
from pathlib import Path
from streamvggt.layers.confidence_state import parse_confidence_gate_init
from streamvggt.utils.cache_analysis import (
    add_eviction_nn_analysis_args,
    add_leverage_score_histogram_args,
    add_projected_norm_histogram_args,
    add_token_overlay_dump_args,
    eviction_nn_config_from_args,
    leverage_score_histogram_config_from_args,
    projected_norm_histogram_config_from_args,
    token_overlay_dump_config_from_args,
)

import hashlib
from pathlib import Path
import attn_cuda._ext as attn_ext

def seed_everything(seed: int) -> int:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed % (2**32))
    # cv2.setRNGSeed(seed % (2**31 - 1))
    o3d.utility.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    print(
        f"[Seed] rank={os.environ.get('RANK', '0')}, seed={seed}",
        flush=True,
    )

    return seed


def parse_max_frames(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none", "all", "full", "full_seq"):
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"max_frames must be a positive integer or full_seq, got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"max_frames must be >= 1, got {parsed}")
    return parsed


_attn_ext_path = Path(attn_ext.__file__).resolve()
print(
    f"[attn_cuda] path={_attn_ext_path} "
    f"sha256={hashlib.sha256(_attn_ext_path.read_bytes()).hexdigest()}",
    flush=True,
)

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


def parse_eviction_policy_layers(spec, num_layers):
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    selected = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"empty layer selector in {spec!r}")
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise ValueError(f"invalid layer range {part!r}; expected start:end")
            try:
                start = int(pieces[0])
                end = int(pieces[1])
            except ValueError as exc:
                raise ValueError(f"invalid layer range {part!r}; bounds must be integers") from exc
            if start < 0 or end < 0:
                raise ValueError(f"invalid layer range {part!r}; bounds must be non-negative")
            if start >= end:
                raise ValueError(f"invalid layer range {part!r}; start must be < end")
            if end > num_layers:
                raise ValueError(
                    f"layer range {part!r} ends at {end}, but only {num_layers} global layers exist"
                )
            selected.update(range(start, end))
        else:
            try:
                layer = int(part)
            except ValueError as exc:
                raise ValueError(f"invalid layer selector {part!r}; expected integer or start:end") from exc
            if layer < 0:
                raise ValueError(f"invalid layer selector {part!r}; layer must be non-negative")
            if layer >= num_layers:
                raise ValueError(
                    f"layer selector {part!r} is out of range; only {num_layers} global layers exist"
                )
            selected.add(layer)
    return selected


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
    parser.add_argument(
        "--max_frames",
        type=parse_max_frames,
        default=None,
        help="Maximum sampled frames per sequence; use full_seq for no limit",
    )
    parser.add_argument("--stream_chunk_size", type=int, default=1, help="Frames per streaming chunk; chunks attend causally to past chunks while frames inside a chunk attend bidirectionally")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--eviction_policy", type=str, default="svd_leverage", help="Cache eviction policy: mean, baseline_mean, svd_leverage")
    parser.add_argument("--eviction_policy_layers", type=str, default="svd_leverage", help="Comma-separated zero-based global layer selectors that use --eviction_policy; other layers use FIFO. Example: 4:12,18,20:24",)
    parser.add_argument("--leverage_granularity", type=str, default="layer", choices=("head", "layer"), help="Granularity for svd_leverage eviction: per-head or one shared layer-wise score vector")
    parser.add_argument("--leverage_feature", type=str, default="key", help="Feature tensor for svd_leverage eviction: keys only, concatenated keys and values, or concatenated low-dimensional key/value sketches")
    parser.add_argument("--leverage_projection", type=str, default="random", choices=("random"),        help="Projection mode for svd_leverage eviction: random right sketch or deterministic per-head means")
    parser.add_argument("--leverage_normalize_rows", action=argparse.BooleanOptionalAction, default=False, help="L2-normalize token feature rows before svd_leverage QR/leverage scoring")
    parser.add_argument("--leverage_normalize_before_projection", action=argparse.BooleanOptionalAction, default=False, help="L2-normalize layer-wise key rows before random leverage projection")
    parser.add_argument("--leverage_normalize_before_projection_headwise", action=argparse.BooleanOptionalAction, default=False, help="When normalizing before projection, normalize each head key row independently")
    parser.add_argument("--leverage_projected_key_cache", action=argparse.BooleanOptionalAction, default=False, help="Reuse cached layer-wise random projected key features for surviving tokens")
    parser.add_argument("--leverage_approx_method", type=str, default="right_sketch_ridge", choices=("exact_qr", "full_d_ridge", "right_sketch_ridge"), help="Leverage approximation: exact QR, right-sketched/Compactor-style, or ridge-based scoring")
    parser.add_argument("--leverage_ridge_lambda", "--leverage-ridge-lambda", type=float, default=0, help="Ridge lambda for full_d_ridge and right_sketch_ridge leverage scoring")
    parser.add_argument("--leverage_ridge_lambda_mode", type=str, default="absolute", choices=("relative", "absolute"), help="Use absolute lambda or scale it by trace(X^T X) / D")
    parser.add_argument("--leverage_ridge_score_chunk_size", type=int, default=16384, help="Token chunk size for ridge leverage Cholesky solves")
    parser.add_argument("--leverage_ridge_jitter", type=float, default=1e-6, help="Absolute diagonal jitter added to ridge systems before Cholesky")
    parser.add_argument("--leverage_ridge_dim", type=int, default=256, help="Projection dimension for right_sketch_ridge; required for right_sketch_ridge")
    parser.add_argument("--rls_refresh_interval", type=int, default=8, help="Refresh interval for ridge leverage K^T K and Cholesky factorization; 1 refreshes every frame")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for leverage sketches")
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
    parser.add_argument("--leverage_attention_freeze_updates", type=int, default=5, help="Number of chunk-level attention observations accumulated before freezing token utility")
    parser.add_argument("--leverage_attention_colsum_subsample_ratio", type=float, default=1.0, help="Fraction of query rows used by the STAC CUDA column-sum kernel")
    parser.add_argument("--leverage_conf_gate_special_mode", type=str, default="mean", choices=("mean", "one"), help="Gate mode for special/prefix tokens: mean uses the patch gate mean; one sets them to 1.0")
    parser.add_argument("--layer_budget_strategy", type=str, default="value_weighted_leverage_pr", choices=("uniform", "leverage_pr", "key_norm", "value_weighted_leverage_pr"), help="Layer-wise KV budget allocation strategy")
    parser.add_argument(
        "--layer_budget_score_only",
        "--layer-budget-score-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute value_weighted_leverage_pr every frame without removing KV cache tokens",
    )
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
    parser.add_argument(
        "--perf_trace",
        action="store_true",
        help="Collect non-synchronizing CUDA-event performance totals and print one summary per inference",
    )
    add_eviction_nn_analysis_args(parser)
    add_leverage_score_histogram_args(parser)
    add_projected_norm_histogram_args(parser)
    add_token_overlay_dump_args(parser)
    parser.add_argument("--eval_frame_stride", type=int, default=1, help="Use every Nth frame only when building eval point clouds; inference still uses all frames")
    parser.add_argument(
        "--recon_eval_mode",
        type=str,
        default="legacy",
        choices=("legacy", "voxel_icp"),
        help="Reconstruction evaluator: existing path or voxelized ICP followed by full-point metrics",
    )
    parser.add_argument(
        "--eval_voxel_size",
        type=float,
        default=0.02,
        help="Metric voxel size for --recon_eval_mode voxel_icp",
    )
    parser.add_argument(
        "--eval_metrics_on_voxels",
        action="store_true",
        help="Compute reconstruction metrics on voxel centroids instead of full-resolution points",
    )
    parser.add_argument("--budget", type=int, default=60000, help="Total token budget for StreamVGGT (if applicable)")
    parser.add_argument("--budget_frame_multiplier", type=float, default=None, help="Set StreamVGGT total budget to ceil(multiplier * tokens_per_frame) * num_global_layers; overrides --budget")
    parser.add_argument("--kf_every", type=int, default=2)
    return parser


def main(args):
    if args.recon_eval_mode == "voxel_icp":
        if args.use_proj:
            raise SystemExit(
                "Error: --recon_eval_mode voxel_icp uses direct predicted 3D points "
                "and cannot be combined with --use_proj."
            )
        if not np.isfinite(args.eval_voxel_size) or args.eval_voxel_size <= 0:
            raise SystemExit(
                "Error: --eval_voxel_size must be finite and > 0 when "
                "--recon_eval_mode voxel_icp is selected."
            )
        print(
            "Using reconstruction evaluator: "
            f"mode=voxel_icp, voxel_size={args.eval_voxel_size}, "
            "alignment=umeyama_sim3_then_icp, icp_threshold=0.1, "
            "points=direct_predicted_3d, "
            f"metrics={'voxel_centroids' if args.eval_metrics_on_voxels else 'full_points'}"
        )
    if args.eviction_policy == "svd_leverage":
        print(
            "Using SVD leverage eviction: "
            f"granularity={args.leverage_granularity}, "
            f"feature={args.leverage_feature}, "
            f"approx={args.leverage_approx_method}, "
            f"ridge_dim={args.leverage_ridge_dim}, ridge_lambda={args.leverage_ridge_lambda}, "
            f"ridge_lambda_mode={args.leverage_ridge_lambda_mode}, "
            f"ridge_chunk={args.leverage_ridge_score_chunk_size}, ridge_jitter={args.leverage_ridge_jitter}, "
            f"rls_refresh_interval={args.rls_refresh_interval}, "
            f"projection={args.leverage_projection}, "
            f"normalize_rows={args.leverage_normalize_rows}, "
            f"normalize_before_projection={args.leverage_normalize_before_projection}, "
            f"normalize_before_projection_headwise={args.leverage_normalize_before_projection_headwise}, "
            f"projected_key_cache={args.leverage_projected_key_cache}, "
            f"selector={args.leverage_eviction_selector}, "
            f"conf_gate={args.leverage_conf_gate}, "
            f"conf_gate_floor={args.leverage_conf_gate_floor}, "
            f"conf_gate_depth_alpha={args.leverage_conf_gate_depth_alpha}, "
            f"conf_gate_point_beta={args.leverage_conf_gate_point_beta}, "
            f"conf_gate_k={args.leverage_conf_gate_k}, "
            f"conf_gate_transform={args.leverage_conf_gate_transform}, "
            f"conf_gate_init={args.leverage_conf_gate_init}, "
            f"conf_gate_special_mode={args.leverage_conf_gate_special_mode}, "
            f"attention_utility={args.leverage_attention_utility}, "
            f"attention_beta={args.leverage_attention_beta}, "
            f"attention_ema_decay={args.leverage_attention_ema_decay}, "
            f"attention_freeze_updates={args.leverage_attention_freeze_updates}, "
            f"attention_colsum_subsample_ratio={args.leverage_attention_colsum_subsample_ratio}, "
            f"layer_budget_strategy={args.layer_budget_strategy}, "
            f"layer_budget_alpha={args.layer_budget_alpha}, "
            f"layer_budget_min_tokens={args.layer_budget_min_tokens}, "
            f"layer_budget_value_gamma={args.layer_budget_value_gamma}, "
            f"layer_budget_value_norm_type={args.layer_budget_value_norm_type}, "
            f"layer_budget_norm_source={args.layer_budget_norm_source}, "
        )
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD, ETH3D, ETH3D_undistort
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
            ROOT="/home/dongjae/data/7scenes",
            # ROOT="/data2/dongjae/datasets/7scenes_sfm",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
        ),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/home/dongjae/data/neural_rgbd_data",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
            # test_id=[ "whiteroom"]
        ),
        # "ETH3D": ETH3D(
        #     ROOT="/home/dongjae/data/eth3d",
        #     full_video=True,
        #     resolution=resolution,
        #     kf_every=1,
        # ),
        # "ETH3D_undistort": ETH3D_undistort(
        #     ROOT="/home/dongjae/data/eth3d",
        #     full_video=True,
        #     resolution=resolution,
        #     kf_every=1,
        # ),
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
    eviction_policy_layers = None
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
        try:
            eviction_policy_layers = parse_eviction_policy_layers(
                None,
                model.aggregator.depth,
            )
        except ValueError as exc:
            raise SystemExit(f"Error: --eviction_policy_layers: {exc}") from exc
        if eviction_policy_layers is not None:
            print(
                "Using layer-range eviction override: "
                f"policy_layers={sorted(eviction_policy_layers)}, "
                "fallback_policy=fifo"
            )
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
            # Each rank owns its log file. Start a fresh file for every run so
            # metrics left by an earlier run cannot be included in this run's mean.
            with open(log_file, "w"):
                pass

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
                    in_camera1 = None

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
                            # if torch.cuda.is_available():
                            #     torch.cuda.synchronize(device)
                            # infer_start = time.perf_counter()

                            scene_label = str(batch[0]["label"][0]).rsplit("/", 1)[0]
                            safe_scene = scene_label.replace("/", "_").replace(os.sep, "_").replace(" ", "_")
                            scene_key = f"{int(data_idx):04d}_{safe_scene}"
                            rank_label = f"rank_{accelerator.process_index}"
                            eviction_nn_analysis_config = None
                            leverage_score_histogram_config = None
                            projected_norm_histogram_config = None
                            token_overlay_dump_config = None
                            layer_budget_log_path = None
                            if args.eviction_nn_analysis_dir:
                                nn_dir = osp.join(
                                    args.eviction_nn_analysis_dir,
                                    name_data,
                                    rank_label,
                                    scene_key,
                                )
                                eviction_nn_analysis_config = eviction_nn_config_from_args(args, output_dir=nn_dir)
                            if args.leverage_score_histogram_dir:
                                hist_dir = osp.join(
                                    args.leverage_score_histogram_dir,
                                    name_data,
                                    rank_label,
                                    scene_key,
                                )
                                leverage_score_histogram_config = leverage_score_histogram_config_from_args(args, output_dir=hist_dir)
                            if args.projected_norm_histogram_dir:
                                norm_hist_dir = osp.join(
                                    args.projected_norm_histogram_dir,
                                    name_data,
                                    rank_label,
                                    scene_key,
                                )
                                projected_norm_histogram_config = projected_norm_histogram_config_from_args(args, output_dir=norm_hist_dir)
                            if args.token_overlay_dump_dir:
                                overlay_dump_dir = osp.join(
                                    args.token_overlay_dump_dir,
                                    name_data,
                                    rank_label,
                                    scene_key,
                                )
                                token_overlay_dump_config = token_overlay_dump_config_from_args(args, output_dir=overlay_dump_dir)
                            if args.layer_budget_log_path:
                                layer_budget_log_path = args.layer_budget_log_path
                            elif args.layer_budget_log_scores:
                                layer_budget_log_path = osp.join(
                                    args.output_dir,
                                    name_data,
                                    scene_key,
                                    "layer_budget_scores.csv",
                                )

                            if torch.cuda.is_available():
                                torch.cuda.synchronize(device)
                            infer_start = time.perf_counter()
                            results = model.inference(
                                batch,
                                eviction_policy=args.eviction_policy,
                                stream_chunk_size=args.stream_chunk_size,
                                budget_frame_multiplier=args.budget_frame_multiplier,
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
                                leverage_random_seed=args.random_seed,
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
                                layer_budget_score_only=args.layer_budget_score_only,
                                layer_budget_value_gamma=args.layer_budget_value_gamma,
                                layer_budget_value_norm_type=args.layer_budget_value_norm_type,
                                layer_budget_norm_source=args.layer_budget_norm_source,
                                layer_budget_alpha=args.layer_budget_alpha,
                                layer_budget_min_tokens=args.layer_budget_min_tokens,
                                layer_budget_eps=args.layer_budget_eps,
                                layer_budget_log_path=layer_budget_log_path,
                                profile_eviction=args.profile_eviction,
                                perf_trace=args.perf_trace,
                                eviction_debug=args.eviction_debug,
                                eviction_nn_analysis_config=eviction_nn_analysis_config,
                                leverage_score_histogram_config=leverage_score_histogram_config,
                                projected_norm_histogram_config=projected_norm_histogram_config,
                                token_overlay_dump_config=token_overlay_dump_config,
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

                    summarize_layer_budget_log(layer_budget_log_path)

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
                    eval_frame_count = 0

                    for j, view in enumerate(batch):
                        if j % args.eval_frame_stride != 0:
                            continue
                        eval_frame_count += 1

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

                    scene_id = batch[0]["label"][0].rsplit("/", 1)[0]
                    if args.eval_frame_stride > 1:
                        eval_stride_msg = (
                            f"Eval frame stride: stride={args.eval_frame_stride}, "
                            f"using {eval_frame_count}/{len(batch)} frames for {scene_id}"
                        )
                        print(eval_stride_msg)
                        print(eval_stride_msg, file=open(log_file, "a"))

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

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

                    # pred_finite_mask = np.isfinite(pts_all).all(axis=-1)
                    # gt_finite_mask = np.isfinite(pts_gt_all).all(axis=-1)
                    # finite_mask = pred_finite_mask & gt_finite_mask

                    # pts_all = pts_all[finite_mask]
                    # pts_gt_all = pts_gt_all[finite_mask]
                    # images_all = images_all[finite_mask]

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

                        s, R, t = umeyama_alignment(
                            pts_all, pts_gt_all, with_scale=True
                        )
                        pts_all = (s * (R @ pts_all.T)).T + t  # (N,3)

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
                    # continue

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    # pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all)
                    # pcd_gt.colors = o3d.utility.Vector3dVector(images_all)
                    # o3d.io.write_point_cloud(
                    #     os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                    #     pcd_gt,
                    # )

                    trans_init = np.eye(4)

                    icp_source = pcd
                    icp_target = pcd_gt

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        icp_source,
                        icp_target,
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

                    # pcd.estimate_normals()
                    # pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    thresholds = (0.05, 0.1, 0.25)

                    acc, acc_med, nc1, nc1_med, precisions = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal, dist_ths=thresholds
                    )
                    comp, comp_med, nc2, nc2_med, recalls = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal, dist_ths=thresholds
                    )
                    f1_scores = {}
                    for th in thresholds:
                        precision = precisions[th]
                        recall = recalls[th]
                        denom = precision + recall

                        f1_scores[th] = (
                            2.0 * precision * recall / denom
                            if denom > 0
                            else 0.0
                        )
                    f1_str = ", ".join(
                        f"F1@{round(th * 100)}cm: {f1_scores[th]:.6f}"
                        for th in thresholds
                    )

                    metric_msg = (
                        f"Idx: {scene_id}, "
                        f"Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                        f"Acc_med: {acc_med}, Compc_med: {comp_med}, "
                        f"NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, "
                        f"{f1_str}, "
                        f"Time: {infer_time}, FPS: {fps}, "
                    )

                    print(metric_msg)
                    with open(log_file, "a") as f:
                        print(metric_msg, file=f)
                    # print(
                    #     f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, Time: {infer_time}, FPS: {fps}"
                    # )
                    # print(
                    #     f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, Time: {infer_time}, FPS: {fps}",
                    #     file=open(log_file, "a"),
                    # )

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
                for i in range(accelerator.num_processes):
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
    NC2c_med:\s*(?P<nc2_med>[^,]+),\s*
    F1@5cm:\s*(?P<f1_5cm>[^,]+),\s*
    F1@10cm:\s*(?P<f1_10cm>[^,]+),\s*
    F1@25cm:\s*(?P<f1_25cm>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*
    (?:,?\s*Time:\s*(?P<time>[^,]+),\s*FPS:\s*(?P<fps>[^,]+))?
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    seed_everything(args.random_seed)

    main(args)
