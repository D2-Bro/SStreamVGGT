#!/usr/bin/env python3
"""W&B sweep wrapper for mv_recon evaluation.

This keeps the evaluation code unchanged: W&B chooses parameters, this wrapper
launches src/eval/mv_recon/launch.py, then logs the final mean metrics.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import wandb


MEAN_LINE_RE = re.compile(r"^mean\s*:\s*(?P<body>.*)$")


def parse_mean_metrics(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        raise FileNotFoundError(f"missing mv_recon aggregate log: {log_path}")

    mean_line = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("mean"):
            mean_line = line

    if mean_line is None:
        raise RuntimeError(f"no final mean metric line found in {log_path}")

    match = MEAN_LINE_RE.match(mean_line)
    if match is None:
        raise RuntimeError(f"could not parse mean metric line: {mean_line}")

    metrics: dict[str, float] = {}
    for field in match.group("body").split("|"):
        field = field.strip()
        if not field or ":" not in field:
            continue
        name, value = field.split(":", 1)
        metrics[f"mean_{name.strip()}"] = float(value.strip())

    if not metrics:
        raise RuntimeError(f"mean metric line had no parseable values: {mean_line}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mv_recon evaluation from a W&B sweep config."
    )

    parser.add_argument("--num_processes", type=int, default=3)
    parser.add_argument("--main_process_port", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--budget_frame_multiplier", type=str, default="8")
    parser.add_argument("--total_budget", type=int, default=200000)
    parser.add_argument("--eval_frame_stride", type=int, default=5)

    parser.add_argument("--layer_budget_alpha", type=float, default=0.7)
    parser.add_argument("--layer_budget_value_gamma", type=float, default=0.7)
    parser.add_argument("--layer_budget_value_norm_type", type=str, default="mean")
    parser.add_argument("--layer_budget_norm_source", type=str, default="key")

    parser.add_argument("--leverage_ridge_dim", type=int, default=64)
    parser.add_argument("--leverage_random_seed", type=int, default=42)
    parser.add_argument("--leverage_conf_gate_special_mode", type=str, default="mean")
    parser.add_argument("--leverage_conf_gate_floor", type=float, default=0.0)
    parser.add_argument("--leverage_conf_gate_depth_alpha", type=float, default=1.0)
    parser.add_argument("--leverage_conf_gate_point_beta", type=float, default=0.0)
    parser.add_argument("--leverage_conf_gate_k", type=float, default=1.0)
    parser.add_argument("--leverage_conf_gate_init", type=str, default="mean")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser


def cfg_value(config: wandb.Config, args: argparse.Namespace, name: str):
    return config.get(name, getattr(args, name))


def main() -> None:
    args = build_parser().parse_args()

    repo = Path(__file__).resolve().parents[1]
    src_dir = repo / "src"

    init_kwargs = {}
    if args.wandb_project:
        init_kwargs["project"] = args.wandb_project
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    with wandb.init(**init_kwargs) as run:
        cfg = {
            name: cfg_value(wandb.config, args, name)
            for name in vars(args)
            if name not in {"wandb_project", "wandb_entity"}
        }

        if int(cfg["main_process_port"]) <= 0:
            port_offset = int(run.id[:4], 36) % 1000
            cfg["main_process_port"] = 29000 + port_offset

        budget_suffix = (
            f"budgetFrameMult{cfg['budget_frame_multiplier']}"
            if str(cfg["budget_frame_multiplier"])
            else f"budget{cfg['total_budget']}"
        )
        run_name = (
            f"a{cfg['layer_budget_alpha']}"
            f"_g{cfg['layer_budget_value_gamma']}"
            f"_dim{cfg['leverage_ridge_dim']}"
            f"_confGateFloor{cfg['leverage_conf_gate_floor']}"
            f"_confGateDepthAlpha{cfg['leverage_conf_gate_depth_alpha']}"
            f"_confGatePointBeta{cfg['leverage_conf_gate_point_beta']}"
            f"_confGateK{cfg['leverage_conf_gate_k']}"
        )
        output_dir = (
            repo
            / "eval_results"
            / "mv_recon"
            / (
                f"wandb_{run.id}_headNorm_{run_name}_evalStride{cfg['eval_frame_stride']}"
                f"_{budget_suffix}"
            )
        )

        wandb.config.update(
            {
                **cfg,
                "output_dir": str(output_dir),
                "weights": "../ckpt/checkpoints.pth",
                "base_script": "src/eval/mv_recon/launch.py",
                "reference_script": "src/eval/mv_recon/run_termProject_topK_conf_headNorm_evalStride_best.sh",
            },
            allow_val_change=True,
        )

        budget_args = (
            ["--budget_frame_multiplier", str(cfg["budget_frame_multiplier"])]
            if str(cfg["budget_frame_multiplier"])
            else ["--budget", str(cfg["total_budget"])]
        )

        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(cfg["num_processes"]),
            "--main_process_port",
            str(cfg["main_process_port"]),
            "./eval/mv_recon/launch.py",
            "--weights",
            "../ckpt/checkpoints.pth",
            "--output_dir",
            str(output_dir),
            "--model_name",
            "StreamVGGT",
            "--max_frames",
            str(cfg["max_frames"]),
            "--eviction_policy",
            "svd_leverage",
            "--leverage_granularity",
            "layer",
            "--leverage_feature",
            "key",
            "--leverage_projection",
            "random",
            "--leverage_eviction_selector",
            "topk",
            "--leverage_eviction_risk_mode",
            "low_leverage",
            "--layer_budget_strategy",
            "value_weighted_leverage_pr",
            "--layer_budget_alpha",
            str(cfg["layer_budget_alpha"]),
            "--leverage_approx_method",
            "right_sketch_ridge",
            "--leverage_ridge_lambda",
            "0",
            "--leverage_ridge_lambda_mode",
            "absolute",
            "--leverage_ridge_score_chunk_size",
            "4096",
            "--leverage_ridge_jitter",
            "1e-6",
            "--leverage_ridge_dim",
            str(cfg["leverage_ridge_dim"]),
            "--leverage_random_seed",
            str(cfg["leverage_random_seed"]),
            "--layer_budget_value_gamma",
            str(cfg["layer_budget_value_gamma"]),
            "--layer_budget_value_norm_type",
            str(cfg["layer_budget_value_norm_type"]),
            "--layer_budget_norm_source",
            str(cfg["layer_budget_norm_source"]),
            *budget_args,
            "--icp_voxel_size",
            "0",
            "--leverage_conf_gate",
            "--leverage_conf_gate_floor",
            str(cfg["leverage_conf_gate_floor"]),
            "--leverage_conf_gate_depth_alpha",
            str(cfg["leverage_conf_gate_depth_alpha"]),
            "--leverage_conf_gate_point_beta",
            str(cfg["leverage_conf_gate_point_beta"]),
            "--leverage_conf_gate_k",
            str(cfg["leverage_conf_gate_k"]),
            "--leverage_conf_gate_special_mode",
            str(cfg["leverage_conf_gate_special_mode"]),
            "--leverage_conf_gate_init",
            str(cfg["leverage_conf_gate_init"]),
            "--eval_frame_stride",
            str(cfg["eval_frame_stride"]),
            "--leverage_normalize_before_projection",
            "--leverage_normalize_before_projection_headwise",
        ]

        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "16",
                "OPENBLAS_NUM_THREADS": "16",
                "MKL_NUM_THREADS": "16",
                "NUMEXPR_NUM_THREADS": "16",
            }
        )

        print("Running mv_recon sweep command:")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=src_dir, env=env, check=True)

        log_path = output_dir / "7scenes" / "logs_all.txt"
        metrics = parse_mean_metrics(log_path)
        metrics["output_dir_exists"] = float(output_dir.exists())
        wandb.log(metrics)
        print(f"Logged W&B metrics from {log_path}: {metrics}")


if __name__ == "__main__":
    main()
