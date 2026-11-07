"""Evaluate mv_recon inference outputs on a CPU-only machine.

Run after launch.py --eval_stage infer has completed. No model or CUDA extension is loaded.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.mv_recon.cpu_evaluation import evaluate_arrays, format_metric


def evaluate_file(path):
    with np.load(path, allow_pickle=False) as data:
        scene_id = str(data["scene_id"].item())
        dataset = str(data["dataset"].item())
        rank = int(data["rank"].item())
        infer_time = float(data["infer_time"].item())
        fps = float(data["fps"].item())
        metrics = evaluate_arrays(
            data["pred"], data["gt"], data["masks"],
            dataset, bool(data["use_proj"].item()),
        )
    return rank, format_metric(scene_id, metrics, infer_time, fps), metrics, infer_time, fps


def evaluate_dataset(dataset_dir, workers):
    manifest_path = dataset_dir / "inference_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Inference completion manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["recon_eval_mode"] != "legacy":
        raise ValueError("Deferred evaluation currently supports --recon_eval_mode legacy only")
    paths = [dataset_dir / "eval_inputs" / f"{i:06d}.npz" for i in range(manifest["scene_count"])]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} scene input(s), first: {missing[0]}")
    if workers == 1:
        results = [evaluate_file(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate_file, paths))

    per_rank = defaultdict(list)
    values = defaultdict(list)
    for rank, message, metrics, infer_time, fps in results:
        if not 0 <= rank < manifest["num_processes"]:
            raise ValueError(f"Invalid saved rank: {rank}")
        print(message)
        per_rank[rank].append(message)
        for name, value in metrics.items():
            values[name].append(float(value))
        values["nc"].append((metrics["nc1"] + metrics["nc2"]) / 2)
        values["nc_med"].append((metrics["nc1_med"] + metrics["nc2_med"]) / 2)
        values["time"].append(infer_time)
        values["fps"].append(fps)

    for rank in range(manifest["num_processes"]):
        (dataset_dir / f"logs_{rank}.txt").write_text(
            "\n".join(per_rank[rank]) + ("\n" if per_rank[rank] else "")
        )
    combined = "".join((dataset_dir / f"logs_{rank}.txt").read_text()
                       for rank in range(manifest["num_processes"]))
    summary = "mean".ljust(20) + ": " + "".join(
        f"{name}: {np.mean(series):.3f} | " for name, series in values.items()
    )
    (dataset_dir / "logs_all.txt").write_text(combined + summary + "\n")
    print(f"Saved metrics: {dataset_dir / 'logs_all.txt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="CPU scene workers (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    datasets = sorted(p.parent for p in args.output_dir.glob("*/inference_manifest.json"))
    if not datasets:
        parser.error(f"No completed inference manifests found in {args.output_dir}")
    for dataset_dir in datasets:
        evaluate_dataset(dataset_dir, args.workers)


if __name__ == "__main__":
    main()
