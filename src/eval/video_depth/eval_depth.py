import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.video_depth.tools import depth_evaluation, group_by_directory
from eval.video_depth.utils import save_depth_error_visuals
import numpy as np
import cv2
from tqdm import tqdm
import glob
from PIL import Image
import argparse
import json
from eval.video_depth.metadata import dataset_metadata


def average_depth_metrics(gathered_depth_metrics, *, output_dir, eval_dataset, align):
    if not gathered_depth_metrics:
        raise RuntimeError(
            "No depth metrics were computed. "
            f"eval_dataset={eval_dataset}, align={align}, output_dir={output_dir}. "
            "Check that prediction .npy files and ground-truth depth files were found."
        )
    return {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    parser.add_argument(
        "--align",
        type=str,
        default="scale&shift",
        choices=["scale&shift", "scale", "metric"],
    )
    parser.add_argument(
        "--first_seq_only",
        "--first-seq-only",
        action="store_true",
        help="Evaluate only the first sorted prediction sequence.",
    )
    parser.add_argument(
        "--eval_stride",
        "--eval-stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame only. This does not affect inference outputs.",
    )
    parser.add_argument(
        "--save_error_visuals",
        "--save-error-visuals",
        action="store_true",
        help="Save per-frame absolute and relative depth error heatmaps.",
    )
    parser.add_argument(
        "--error_visual_dir",
        "--error-visual-dir",
        default=None,
        help="Output root for error heatmaps; defaults to output_dir/error_visuals_<align>.",
    )
    parser.add_argument(
        "--error_visual_fps",
        "--error-visual-fps",
        type=int,
        default=10,
        help="FPS for per-sequence error heatmap mp4 files.",
    )
    parser.add_argument(
        "--error_visual_percentile",
        "--error-visual-percentile",
        type=float,
        default=99.0,
        help="Percentile used as the heatmap color range maximum.",
    )
    parser.add_argument(
        "--save_error_overlays",
        "--save-error-overlays",
        action="store_true",
        help="Also save error heatmaps alpha-blended over RGB frames.",
    )
    parser.add_argument(
        "--error_overlay_alpha",
        "--error-overlay-alpha",
        type=float,
        default=0.55,
        help="Heatmap alpha for RGB overlay outputs.",
    )
    return parser



def selected_sequence_keys(grouped_pred_depth, args):
    keys = sorted(grouped_pred_depth.keys())
    if args.first_seq_only:
        if keys:
            keys = keys[:1]
            print(f"[video_depth] first_seq_only: evaluating sequence {keys[0]}")
        else:
            print("[video_depth] first_seq_only: no prediction sequences found")
    return keys


def error_visual_root(args):
    if args.error_visual_dir:
        return args.error_visual_dir
    return os.path.join(args.output_dir, f"error_visuals_{args.align}")


def maybe_save_error_visuals(
    args,
    sequence,
    gt_depth,
    error_map,
    depth_predict,
    depth_gt,
    image_paths=None,
):
    if not (args.save_error_visuals or args.save_error_overlays):
        return
    if args.save_error_overlays and image_paths is None:
        raise FileNotFoundError(f"No RGB frames found for overlay sequence {sequence!r}")
    save_depth_error_visuals(
        depth_predict,
        depth_gt,
        error_map,
        os.path.join(error_visual_root(args), sequence),
        sequence=sequence,
        sequence_shape=gt_depth.shape,
        fps=args.error_visual_fps,
        percentile=args.error_visual_percentile,
        image_paths=image_paths if args.save_error_overlays else None,
        overlay_alpha=args.error_overlay_alpha,
    )


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




def collect_eval_image_paths(args, seq_list=None, full=False):
    metadata = dataset_metadata[args.eval_dataset]
    img_root = metadata.get("img_path")
    if args.eval_dataset == "sintel":
        if full:
            return sorted(glob.glob(os.path.join(img_root, "*", "*.png")))
        seqs = seq_list or metadata.get("seq_list", [])
        paths = []
        for seq in seqs:
            paths.extend(glob.glob(os.path.join(img_root, seq, "*.png")))
        return sorted(paths)
    if args.eval_dataset.startswith("bonn"):
        bonn_number = args.eval_dataset.split("_")[-1] if "_" in args.eval_dataset else "110"
        seqs = seq_list or ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
        paths = []
        for seq in seqs:
            paths.extend(glob.glob(os.path.join(img_root, f"rgbd_bonn_{seq}", f"rgb_{bonn_number}", "*.png")))
        return sorted(paths)
    if args.eval_dataset == "replica":
        seqs = seq_list or metadata.get("seq_list", [])
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


def timestamp_from_path(path):
    return float(os.path.splitext(os.path.basename(path))[0])


def align_bonn_paths_by_timestamp(pred_paths, gt_paths, image_paths, sequence, max_time_diff=0.02):
    if image_paths is None:
        return pred_paths, gt_paths, image_paths

    count = min(len(pred_paths), len(image_paths))
    pred_paths = pred_paths[:count]
    image_paths = image_paths[:count]
    gt_times = [timestamp_from_path(path) for path in gt_paths]
    matched_pred_paths = []
    matched_gt_paths = []
    matched_image_paths = []
    last_gt_idx = -1
    gt_idx = 0

    for pred_path, image_path in zip(pred_paths, image_paths):
        image_time = timestamp_from_path(image_path)
        while (
            gt_idx + 1 < len(gt_times)
            and abs(gt_times[gt_idx + 1] - image_time) < abs(gt_times[gt_idx] - image_time)
        ):
            gt_idx += 1
        if gt_idx <= last_gt_idx:
            continue
        if abs(gt_times[gt_idx] - image_time) > max_time_diff:
            continue
        matched_pred_paths.append(pred_path)
        matched_gt_paths.append(gt_paths[gt_idx])
        matched_image_paths.append(image_path)
        last_gt_idx = gt_idx

    if len(matched_pred_paths) != min(len(pred_paths), len(gt_paths)):
        print(
            f"WARN: Bonn {sequence}: timestamp-aligned "
            f"pred={len(pred_paths)} gt={len(gt_paths)} rgb={len(image_paths)} "
            f"-> {len(matched_pred_paths)} frames"
        )
    if not matched_pred_paths:
        raise RuntimeError(f"No timestamp-aligned Bonn frames found for sequence {sequence!r}")
    return matched_pred_paths, matched_gt_paths, matched_image_paths


def apply_eval_stride(args, pred_paths, gt_paths, image_paths=None, sequence=None):
    if args.eval_stride < 1:
        raise ValueError(f"--eval_stride must be >= 1, got {args.eval_stride}")
    if args.eval_stride == 1:
        return pred_paths, gt_paths, image_paths

    frame_count = min(len(pred_paths), len(gt_paths))
    pred_paths = pred_paths[:: args.eval_stride]
    gt_paths = gt_paths[:: args.eval_stride]
    if image_paths is not None:
        image_paths = image_paths[:: args.eval_stride]
    print(
        f"[video_depth] eval_stride={args.eval_stride}: "
        f"{sequence or 'sequence'} {frame_count} -> {len(pred_paths)} frames"
    )
    if not pred_paths or not gt_paths:
        raise ValueError(
            f"--eval_stride={args.eval_stride} leaves no frames for sequence {sequence!r}"
        )
    return pred_paths, gt_paths, image_paths


def main(args):
    if args.eval_dataset == "sintel":
        TAG_FLOAT = 202021.25

        def depth_read(filename):
            """Read depth data from file, return as numpy array."""
            f = open(filename, "rb")
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert (
                check == TAG_FLOAT
            ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
                TAG_FLOAT, check
            )
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            size = width * height
            assert (
                width > 0 and height > 0 and size > 1 and size < 100000000
            ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
                width, height
            )
            depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
            return depth

        pred_pathes = glob.glob(
            f"{args.output_dir}/*/frame_*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)

        seq_list = None
        if len(pred_pathes) > 643:
            full = True
        else:
            full = False

        if full:
            depth_pathes = glob.glob(f"/home/ma-user/work/datasets/sintel/training/depth/*/*.dpt")
            depth_pathes = sorted(depth_pathes)
        else:
            seq_list = [
                "alley_2",
                "ambush_4",
                "ambush_5",
                "ambush_6",
                "cave_2",
                "cave_4",
                "market_2",
                "market_5",
                "market_6",
                "shaman_3",
                "sleeping_1",
                "sleeping_2",
                "temple_2",
                "temple_3",
            ]
            depth_pathes_folder = [
                f"/home/dongjae/data/datasets/sintel/training/depth/{seq}" for seq in seq_list
            ]
            depth_pathes = []
            for depth_pathes_folder_i in depth_pathes_folder:
                depth_pathes += glob.glob(depth_pathes_folder_i + "/*.dpt")
            depth_pathes = sorted(depth_pathes)
        image_pathes = collect_eval_image_paths(args, seq_list=seq_list, full=full)

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)

            grouped_gt_depth = group_by_directory(depth_pathes)
            grouped_images = grouped_image_paths(args, img_pathes if args.eval_dataset.startswith("bonn") else image_pathes)
            gathered_depth_metrics = []

            for key in tqdm(selected_sequence_keys(grouped_pred_depth, args)):
                pd_pathes = grouped_pred_depth[key]
                gt_pathes = grouped_gt_depth[key.replace("_pred_depth", "")]
                image_paths = image_paths_for_sequence(
                    args,
                    grouped_images,
                    key.replace("_pred_depth", ""),
                    len(gt_pathes),
                )
                pd_pathes, gt_pathes, image_paths = apply_eval_stride(
                    args, pd_pathes, gt_pathes, image_paths, sequence=key
                )

                gt_depth = np.stack(
                    [depth_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )
                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_lad2=True,
                            use_gpu=True,
                            post_clip_max=70,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_scale=True,
                            use_gpu=True,
                            post_clip_max=70,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            metric_scale=True,
                            use_gpu=True,
                            post_clip_max=70,
                        )
                    )
                gathered_depth_metrics.append(depth_results)
                maybe_save_error_visuals(
                    args,
                    pred_key if args.eval_dataset.startswith("bonn") else key,
                    gt_depth,
                    error_map,
                    depth_predict,
                    depth_gt,
                    image_paths,
                )

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = average_depth_metrics(
                gathered_depth_metrics,
                output_dir=args.output_dir,
                eval_dataset=args.eval_dataset,
                align=args.align,
            )
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset.startswith("bonn"):
        print(f"{args.eval_dataset}")
        def depth_read(filename):
            # loads depth map D from png file
            # and returns it as a numpy array
            depth_png = np.asarray(Image.open(filename))
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert np.max(depth_png) > 255
            depth = depth_png.astype(np.float64) / 5000.0
            depth[depth_png == 0] = -1.0
            return depth

        seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

        if "_" in args.eval_dataset:
            bonn_number = args.eval_dataset.split("_")[-1]
        else:
            bonn_number = "110"  # default value

        bonn_root = dataset_metadata[args.eval_dataset]["img_path"]
        img_pathes_folder = [
            f"{bonn_root}/rgbd_bonn_{seq}/rgb_{bonn_number}/*.png"
            for seq in seq_list
        ]
        img_pathes = []
        for img_pathes_folder_i in img_pathes_folder:
            img_pathes += glob.glob(img_pathes_folder_i)
        img_pathes = sorted(img_pathes)
        depth_pathes_folder = [
            f"{bonn_root}/rgbd_bonn_{seq}/depth_{bonn_number}/*.png"
            for seq in seq_list
        ]
        depth_pathes = []
        for depth_pathes_folder_i in depth_pathes_folder:
            depth_pathes += glob.glob(depth_pathes_folder_i)
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.output_dir}/*/frame*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
        if not depth_pathes:
            raise FileNotFoundError(
                "No Bonn ground-truth depth files found. Searched:\n"
                + "\n".join(depth_pathes_folder)
            )
        if not pred_pathes:
            raise FileNotFoundError(
                f"No predicted depth files found under {args.output_dir}/*/frame*.npy"
            )

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)
            grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
            grouped_images = grouped_image_paths(args, img_pathes if args.eval_dataset.startswith("bonn") else image_pathes)
            gathered_depth_metrics = []
            for key in tqdm(grouped_gt_depth.keys()):
                pred_key = key[10:]
                if pred_key not in grouped_pred_depth:
                    raise FileNotFoundError(
                        f"No predictions found for Bonn sequence {pred_key!r} "
                        f"under {args.output_dir}"
                    )
                pd_pathes = grouped_pred_depth[pred_key]
                gt_pathes = grouped_gt_depth[key]
                image_paths = image_paths_for_sequence(
                    args,
                    grouped_images,
                    pred_key,
                    len(pd_pathes),
                )
                pd_pathes, gt_pathes, image_paths = align_bonn_paths_by_timestamp(
                    pd_pathes, gt_pathes, image_paths, pred_key
                )
                pd_pathes, gt_pathes, image_paths = apply_eval_stride(
                    args, pd_pathes, gt_pathes, image_paths, sequence=pred_key
                )
                gt_depth = np.stack(
                    [depth_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )
                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_lad2=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_scale=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            metric_scale=True,
                            use_gpu=True,
                        )
                    )
                gathered_depth_metrics.append(depth_results)
                maybe_save_error_visuals(
                    args,
                    pred_key if args.eval_dataset.startswith("bonn") else key,
                    gt_depth,
                    error_map,
                    depth_predict,
                    depth_gt,
                    image_paths,
                )

                # seq_len = gt_depth.shape[0]
                # error_map = error_map.reshape(seq_len, -1, error_map.shape[-1]).cpu()
                # error_map_colored = colorize(error_map, range=(error_map.min(), error_map.max()), append_cbar=True)
                # ImageSequenceClip([x for x in (error_map_colored.numpy()*255).astype(np.uint8)], fps=10).write_videofile(f'{args.output_dir}/errormap_{key}_{args.align}.mp4', fps=10)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = average_depth_metrics(
                gathered_depth_metrics,
                output_dir=args.output_dir,
                eval_dataset=args.eval_dataset,
                align=args.align,
            )
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset == "replica":
        metadata = dataset_metadata["replica"]
        replica_root = metadata["img_path"]
        seq_list = metadata.get("seq_list") or []
        depth_pathes = []
        for seq in seq_list:
            depth_pathes.extend(glob.glob(os.path.join(replica_root, seq, "results", "depth*.png")))
        depth_pathes = sorted(depth_pathes)
        pred_pathes = sorted(glob.glob(f"{args.output_dir}/*/frame*.npy"))
        image_pathes = collect_eval_image_paths(args, seq_list=seq_list)
        if not depth_pathes:
            raise FileNotFoundError(f"No Replica ground-truth depth files found under {replica_root}")
        if not pred_pathes:
            raise FileNotFoundError(f"No predicted depth files found under {args.output_dir}/*/frame*.npy")

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)
            grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
            grouped_images = grouped_image_paths(args, img_pathes if args.eval_dataset.startswith("bonn") else image_pathes)
            gathered_depth_metrics = []
            for key in tqdm(selected_sequence_keys(grouped_pred_depth, args)):
                if key not in grouped_gt_depth:
                    raise FileNotFoundError(f"No Replica GT depth found for sequence {key!r}")
                pd_pathes = grouped_pred_depth[key]
                gt_pathes = grouped_gt_depth[key]
                if len(pd_pathes) != len(gt_pathes):
                    n = min(len(pd_pathes), len(gt_pathes))
                    print(f"WARN: Replica {key}: pred={len(pd_pathes)} gt={len(gt_pathes)} using first {n} frames")
                    pd_pathes = pd_pathes[:n]
                    gt_pathes = gt_pathes[:n]
                image_paths = image_paths_for_sequence(
                    args,
                    grouped_images,
                    key,
                    len(gt_pathes),
                )
                pd_pathes, gt_pathes, image_paths = apply_eval_stride(
                    args, pd_pathes, gt_pathes, image_paths, sequence=key
                )
                gt_depth = np.stack([read_replica_depth(gt_path) for gt_path in gt_pathes], axis=0)
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                        pr_depth,
                        gt_depth,
                        max_depth=10.0,
                        align_with_lad2=True,
                        use_gpu=True,
                        post_clip_max=10.0,
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                        pr_depth,
                        gt_depth,
                        max_depth=10.0,
                        align_with_scale=True,
                        use_gpu=True,
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                        pr_depth,
                        gt_depth,
                        max_depth=10.0,
                        metric_scale=True,
                        use_gpu=True,
                    )
                gathered_depth_metrics.append(depth_results)
                maybe_save_error_visuals(
                    args,
                    pred_key if args.eval_dataset.startswith("bonn") else key,
                    gt_depth,
                    error_map,
                    depth_predict,
                    depth_gt,
                    image_paths,
                )

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = average_depth_metrics(
                gathered_depth_metrics,
                output_dir=args.output_dir,
                eval_dataset=args.eval_dataset,
                align=args.align,
            )
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset == "kitti_s1_500":

        def depth_read(filename):
            # loads depth map D from png file
            # and returns it as a numpy array,
            # for details see readme.txt
            img_pil = Image.open(filename)
            depth_png = np.array(img_pil, dtype=int)
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert np.max(depth_png) > 255

            depth = depth_png.astype(float) / 256.0
            depth[depth_png == 0] = -1.0
            return depth

        depth_pathes = glob.glob(
            "/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/groundtruth_depth_gathered_500/*/*.png"
        )
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.output_dir}/*/frame_*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
        image_pathes = collect_eval_image_paths(args)

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)
            grouped_gt_depth = group_by_directory(depth_pathes)
            grouped_images = grouped_image_paths(args, img_pathes if args.eval_dataset.startswith("bonn") else image_pathes)
            gathered_depth_metrics = []
            for key in tqdm(selected_sequence_keys(grouped_pred_depth, args)):
                pd_pathes = grouped_pred_depth[key]
                gt_pathes = grouped_gt_depth[key]
                image_paths = image_paths_for_sequence(
                    args,
                    grouped_images,
                    key,
                    len(gt_pathes),
                )
                pd_pathes, gt_pathes, image_paths = apply_eval_stride(
                    args, pd_pathes, gt_pathes, image_paths, sequence=key
                )
                gt_depth = np.stack(
                    [depth_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )

                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            align_with_lad2=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            align_with_scale=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            metric_scale=True,
                            use_gpu=True,
                        )
                    )
                gathered_depth_metrics.append(depth_results)
                maybe_save_error_visuals(
                    args,
                    pred_key if args.eval_dataset.startswith("bonn") else key,
                    gt_depth,
                    error_map,
                    depth_predict,
                    depth_gt,
                    image_paths,
                )

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            average_metrics = average_depth_metrics(
                gathered_depth_metrics,
                output_dir=args.output_dir,
                eval_dataset=args.eval_dataset,
                align=args.align,
            )
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
