import argparse
import csv
import glob
import json
import os
import sys
from typing import Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.tools import depth_evaluation, group_by_directory
from eval.video_depth.utils import save_depth_error_visuals


METRIC_ORDER = [
    "Abs Rel",
    "Sq Rel",
    "RMSE",
    "Log RMSE",
    "δ < 1.",
    "δ < 1.25",
    "δ < 1.25^2",
    "δ < 1.25^3",
]


def read_sintel_depth(filename):
    tag_float = 202021.25
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if check != tag_float:
            raise ValueError(f"Invalid Sintel depth tag in {filename}: {check}")
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        return np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))


def read_bonn_depth(filename):
    depth_png = np.asarray(Image.open(filename))
    if np.max(depth_png) <= 255:
        raise ValueError(f"Expected 16-bit Bonn depth map, got {filename}")
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def read_kitti_depth(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    if np.max(depth_png) <= 255:
        raise ValueError(f"Expected 16-bit KITTI depth map, got {filename}")
    depth = depth_png.astype(float) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def replica_depth_scale():
    cam_params_path = os.path.join(dataset_metadata["replica"]["img_path"], "cam_params.json")
    try:
        with open(cam_params_path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("camera", {}).get("scale", 6553.5))
    except (OSError, ValueError, TypeError):
        return 6553.5


def read_replica_depth(filename):
    depth_png = np.asarray(Image.open(filename))
    depth = depth_png.astype(np.float32) / replica_depth_scale()
    depth[depth_png == 0] = -1.0
    return depth


def collect_sintel(args):
    pred_paths = sorted(glob.glob(f"{args.output_dir}/*/frame_*.npy"))
    if len(pred_paths) > 643:
        depth_paths = sorted(glob.glob("/home/ma-user/work/datasets/sintel/training/depth/*/*.dpt"))
    else:
        seq_list = dataset_metadata["sintel"]["seq_list"]
        depth_paths = []
        for seq in seq_list:
            depth_paths.extend(glob.glob(f"/home/dongjae/data/datasets/sintel/training/depth/{seq}/*.dpt"))
        depth_paths = sorted(depth_paths)
    return pred_paths, depth_paths, read_sintel_depth, 70, -1, lambda key: key.replace("_pred_depth", "")


def collect_bonn(args):
    bonn_number = args.eval_dataset.split("_")[-1] if "_" in args.eval_dataset else "110"
    bonn_root = dataset_metadata[args.eval_dataset]["img_path"]
    seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
    depth_paths = []
    for seq in seq_list:
        depth_paths.extend(glob.glob(f"{bonn_root}/rgbd_bonn_{seq}/depth_{bonn_number}/*.png"))
    pred_paths = sorted(glob.glob(f"{args.output_dir}/*/frame*.npy"))
    return sorted(pred_paths), sorted(depth_paths), read_bonn_depth, 70, -2, lambda key: key[10:]


def collect_kitti(args):
    if args.eval_dataset.startswith("kitti_s1_"):
        count = args.eval_dataset.rsplit("_", 1)[-1]
        gt_root = f"/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/groundtruth_depth_gathered_{count}"
    else:
        gt_root = "/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/groundtruth_depth_gathered"
    depth_paths = sorted(glob.glob(f"{gt_root}/*/*.png"))
    pred_paths = sorted(glob.glob(f"{args.output_dir}/*/frame_*.npy"))
    return pred_paths, depth_paths, read_kitti_depth, None, -1, lambda key: key


def collect_replica(args):
    metadata = dataset_metadata["replica"]
    root = metadata["img_path"]
    seq_list = metadata.get("seq_list") or []
    depth_paths = []
    for seq in seq_list:
        depth_paths.extend(glob.glob(os.path.join(root, seq, "results", "depth*.png")))
    pred_paths = sorted(glob.glob(f"{args.output_dir}/*/frame*.npy"))
    return sorted(pred_paths), sorted(depth_paths), read_replica_depth, 10.0, -2, lambda key: key


def collect_paths(args):
    if args.eval_dataset == "sintel":
        return collect_sintel(args)
    if args.eval_dataset.startswith("bonn"):
        return collect_bonn(args)
    if args.eval_dataset.startswith("kitti"):
        return collect_kitti(args)
    if args.eval_dataset == "replica":
        return collect_replica(args)
    raise ValueError(f"Per-sequence video depth eval is not implemented for {args.eval_dataset!r}")




def collect_eval_image_paths(args):
    metadata = dataset_metadata[args.eval_dataset]
    img_root = metadata.get("img_path")
    if args.eval_dataset == "sintel":
        seqs = metadata.get("seq_list", [])
        paths = []
        for seq in seqs:
            paths.extend(glob.glob(os.path.join(img_root, seq, "*.png")))
        return sorted(paths)
    if args.eval_dataset.startswith("bonn"):
        bonn_number = args.eval_dataset.split("_")[-1] if "_" in args.eval_dataset else "110"
        seqs = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
        paths = []
        for seq in seqs:
            paths.extend(glob.glob(os.path.join(img_root, f"rgbd_bonn_{seq}", f"rgb_{bonn_number}", "*.png")))
        return sorted(paths)
    if args.eval_dataset == "replica":
        seqs = metadata.get("seq_list", []) or []
        paths = []
        for seq in seqs:
            paths.extend(glob.glob(os.path.join(img_root, seq, "results", "frame*.jpg")))
        return sorted(paths)
    if args.eval_dataset.startswith("kitti"):
        return sorted(glob.glob(os.path.join(img_root, "*", "*.png")))
    return []


def grouped_image_paths(args, image_paths):
    if args.eval_dataset.startswith("bonn") or args.eval_dataset == "replica":
        return group_by_directory(image_paths, idx=-2)
    return group_by_directory(image_paths)


def image_paths_for_sequence(args, grouped_images, key, count):
    paths = grouped_images.get(key)
    if paths is None and args.eval_dataset.startswith("bonn"):
        paths = grouped_images.get(f"rgbd_bonn_{key}")
    if paths is None:
        return None
    if len(paths) > count:
        print(f"WARN: RGB {key}: image={len(paths)} depth={count} using first {count} frames")
        paths = paths[:count]
    elif len(paths) < count:
        print(f"WARN: RGB {key}: image={len(paths)} depth={count}; cannot make overlay")
        return None
    return paths


def evaluate_depth(pr_depth, gt_depth, *, align, max_depth, use_gpu, return_maps=False):
    kwargs = {"max_depth": max_depth, "use_gpu": use_gpu}
    if align == "scale&shift":
        kwargs["align_with_lad2"] = True
        if max_depth is not None:
            kwargs["post_clip_max"] = max_depth
    elif align == "scale":
        kwargs["align_with_scale"] = True
        if max_depth is not None and max_depth == 70:
            kwargs["post_clip_max"] = max_depth
    elif align == "metric":
        kwargs["metric_scale"] = True
        if max_depth is not None and max_depth == 70:
            kwargs["post_clip_max"] = max_depth
    else:
        raise ValueError(f"Unknown alignment mode: {align}")
    metrics, error_map, depth_predict, depth_gt = depth_evaluation(pr_depth, gt_depth, **kwargs)
    metrics = {key: float(value) for key, value in metrics.items()}
    if return_maps:
        return metrics, error_map, depth_predict, depth_gt
    return metrics


def evaluate_sequence(
    pred_paths,
    gt_paths,
    depth_reader: Callable,
    *,
    align,
    max_depth,
    use_gpu,
    strict_lengths,
    save_error_visuals=False,
    save_error_overlays=False,
    error_visual_dir=None,
    sequence=None,
    image_paths=None,
    error_visual_fps=10,
    error_visual_percentile=99.0,
    error_overlay_alpha=0.55,
):
    if len(pred_paths) != len(gt_paths):
        n = min(len(pred_paths), len(gt_paths))
        message = f"pred={len(pred_paths)} gt={len(gt_paths)} using first {n} frames"
        if strict_lengths:
            raise ValueError(message)
        print(f"WARN: {message}", file=sys.stderr)
        pred_paths = pred_paths[:n]
        gt_paths = gt_paths[:n]
    if not pred_paths:
        raise ValueError("No prediction frames to evaluate")

    gt_depth = np.stack([depth_reader(path) for path in gt_paths], axis=0)
    pr_depth = np.stack(
        [
            cv2.resize(
                np.load(path),
                (gt_depth.shape[2], gt_depth.shape[1]),
                interpolation=cv2.INTER_CUBIC,
            )
            for path in pred_paths
        ],
        axis=0,
    )
    result = evaluate_depth(
        pr_depth,
        gt_depth,
        align=align,
        max_depth=max_depth,
        use_gpu=use_gpu,
        return_maps=save_error_visuals or save_error_overlays,
    )
    if not (save_error_visuals or save_error_overlays):
        return result
    if save_error_overlays and image_paths is None:
        raise FileNotFoundError(f"No RGB frames found for overlay sequence {sequence!r}")

    metrics, error_map, depth_predict, depth_gt = result
    save_depth_error_visuals(
        depth_predict,
        depth_gt,
        error_map,
        os.path.join(error_visual_dir, sequence),
        sequence=sequence,
        sequence_shape=gt_depth.shape,
        fps=error_visual_fps,
        percentile=error_visual_percentile,
        image_paths=image_paths if save_error_overlays else None,
        overlay_alpha=error_overlay_alpha,
    )
    return metrics


def weighted_average(rows, metric_keys):
    if not rows:
        return {}
    weights = [row["metrics"]["valid_pixels"] for row in rows]
    return {
        key: float(np.average([row["metrics"][key] for row in rows], weights=weights))
        for key in metric_keys
    }


def print_table(rows, metric_keys):
    header = ["sequence", "frames", *metric_keys]
    print("\t".join(header))
    for row in rows:
        values = [row["sequence"], str(row["frames"])]
        metric_values = row["metrics"]
        values.extend(f"{metric_values[key]:.6g}" for key in metric_keys)
        print("\t".join(values))


def save_json(path, rows, average, metric_keys):
    payload = {
        "average": average,
        "sequences": [
            {
                "sequence": row["sequence"],
                "frames": row["frames"],
                "metrics": {key: row["metrics"][key] for key in metric_keys},
                "valid_pixels": row["metrics"].get("valid_pixels"),
            }
            for row in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_csv(path, rows, metric_keys):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "frames", *metric_keys, "valid_pixels"])
        for row in rows:
            writer.writerow(
                [
                    row["sequence"],
                    row["frames"],
                    *[row["metrics"][key] for key in metric_keys],
                    row["metrics"].get("valid_pixels"),
                ]
            )


def get_args_parser():
    parser = argparse.ArgumentParser(description="Print video depth metrics for each sequence.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_dataset", required=True, choices=list(dataset_metadata.keys()))
    parser.add_argument("--align", default="scale", choices=("scale&shift", "scale", "metric"))
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDA inside depth_evaluation; CPU is safer for long sequences")
    parser.add_argument("--strict_lengths", action="store_true", help="Fail when prediction/GT frame counts differ")
    parser.add_argument("--save_json", default=None, help="Optional path to save per-sequence metrics as JSON")
    parser.add_argument("--save_csv", default=None, help="Optional path to save per-sequence metrics as CSV")
    parser.add_argument("--save_error_visuals", "--save-error-visuals", action="store_true", help="Save per-frame absolute and relative depth error heatmaps")
    parser.add_argument("--error_visual_dir", "--error-visual-dir", default=None, help="Output root for error heatmaps; defaults to output_dir/error_visuals_<align>")
    parser.add_argument("--error_visual_fps", "--error-visual-fps", type=int, default=10, help="FPS for per-sequence error heatmap mp4 files")
    parser.add_argument("--error_visual_percentile", "--error-visual-percentile", type=float, default=99.0, help="Percentile used as the heatmap color range maximum")
    parser.add_argument("--save_error_overlays", "--save-error-overlays", action="store_true", help="Also save error heatmaps alpha-blended over RGB frames")
    parser.add_argument("--error_overlay_alpha", "--error-overlay-alpha", type=float, default=0.55, help="Heatmap alpha for RGB overlay outputs")
    return parser


def main(args):
    pred_paths, depth_paths, depth_reader, max_depth, gt_group_idx, pred_to_gt_key = collect_paths(args)
    if not pred_paths:
        raise FileNotFoundError(f"No prediction .npy files found under {args.output_dir}")
    if not depth_paths:
        raise FileNotFoundError(f"No ground-truth depth files found for {args.eval_dataset}")

    pred_by_seq = group_by_directory(pred_paths)
    gt_by_seq = group_by_directory(depth_paths, idx=gt_group_idx)
    grouped_images = grouped_image_paths(args, collect_eval_image_paths(args))
    error_visual_dir = args.error_visual_dir or os.path.join(args.output_dir, f"error_visuals_{args.align}")
    rows = []
    for pred_key in tqdm(sorted(pred_by_seq), desc="sequences"):
        gt_key = pred_to_gt_key(pred_key)
        if gt_key not in gt_by_seq:
            print(f"WARN: skipping {pred_key}; missing GT key {gt_key}", file=sys.stderr)
            continue
        metrics = evaluate_sequence(
            pred_by_seq[pred_key],
            gt_by_seq[gt_key],
            depth_reader,
            align=args.align,
            max_depth=max_depth,
            use_gpu=args.use_gpu,
            strict_lengths=args.strict_lengths,
            save_error_visuals=args.save_error_visuals,
            save_error_overlays=args.save_error_overlays,
            error_visual_dir=error_visual_dir,
            sequence=pred_key,
            image_paths=image_paths_for_sequence(args, grouped_images, pred_key, len(pred_by_seq[pred_key])),
            error_visual_fps=args.error_visual_fps,
            error_visual_percentile=args.error_visual_percentile,
            error_overlay_alpha=args.error_overlay_alpha,
        )
        rows.append({"sequence": pred_key, "frames": len(pred_by_seq[pred_key]), "metrics": metrics})

    if not rows:
        raise RuntimeError("No sequence metrics were computed")

    metric_keys = [key for key in METRIC_ORDER if key in rows[0]["metrics"]]
    average = weighted_average(rows, metric_keys)
    print(f"OUTPUT_DIR {args.output_dir}")
    print(f"SEQ_COUNT {len(rows)}")
    print("AVERAGE " + json.dumps(average, sort_keys=True))
    print_table(rows, metric_keys)

    if args.save_json:
        save_json(args.save_json, rows, average, metric_keys)
    if args.save_csv:
        save_csv(args.save_csv, rows, metric_keys)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
