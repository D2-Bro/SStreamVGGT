import os
import glob
from tqdm import tqdm


def _seven_scenes_root():
    return os.environ.get("SSTREAMVGGT_7SCENES_ROOT", "/home/dongjae/data/7scenes_sfm")


def _seven_scenes_seq_from_split_line(line):
    num_part = "".join(filter(str.isdigit, line))
    return f"seq-{num_part.zfill(2)}"


def _seven_scenes_test_sequences(root=None):
    root = root or _seven_scenes_root()
    if not os.path.isdir(root):
        return []

    seq_list = []
    for scene in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, scene)
        split_file = os.path.join(scene_dir, "TestSplit.txt")
        if not os.path.isdir(scene_dir) or not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    seq_list.append(f"{scene}/{_seven_scenes_seq_from_split_line(line)}")
    return seq_list


def _seven_scenes_color_files(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "frame-*.color.png")))


def _kitti_odometry_root():
    return os.environ.get("SSTREAMVGGT_KITTI_ODOMETRY_ROOT", "/home/dongjae/data/kitti/dataset")


def _kitti_image_files(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "*.png")))


def _vbr_root():
    return os.environ.get("SSTREAMVGGT_VBR_ROOT", "/home/dongjae/data/vbr")


def _vbr_image_files(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "*.png")))


def _replica_root():
    return os.environ.get("SSTREAMVGGT_REPLICA_ROOT", "/home/dongjae/data/replica/Replica")


def _replica_sequences(root=None):
    root = root or _replica_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        seq
        for seq in os.listdir(root)
        if os.path.isdir(os.path.join(root, seq))
        and os.path.exists(os.path.join(root, seq, "traj.txt"))
    )


def _replica_image_files(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "frame*.jpg")))


def _replica_repeat_image_files(dir_path):
    filelist = _replica_image_files(dir_path)
    return filelist + list(reversed(filelist))


def _replica_frame_index(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if not stem.startswith("frame") or not stem[5:].isdigit():
        raise ValueError(f"Unsupported Replica frame filename: {image_path}")
    return int(stem[5:])


def _replica_repeat_gt_traj_from_filelist(img_path, anno_path, seq, filelist):
    import numpy as np
    from eval.pose_evaluation.evo_utils import replica_pose_rows_to_traj_tum

    if not filelist:
        raise ValueError(f"No selected frames for Replica repeat sequence {seq}")

    traj_path = os.path.join(anno_path, seq, "traj.txt")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Missing Replica traj.txt for sequence {seq}: {traj_path}")

    traj_rows = np.loadtxt(traj_path, dtype=np.float64)
    if traj_rows.ndim == 1:
        traj_rows = traj_rows[None, :]
    if traj_rows.ndim != 2 or traj_rows.shape[1] not in (12, 16):
        raise ValueError(
            f"Unsupported Replica trajectory shape {traj_rows.shape} in {traj_path}"
        )

    selected_rows = []
    for image_path in filelist:
        frame_idx = _replica_frame_index(image_path)
        if frame_idx >= traj_rows.shape[0]:
            raise IndexError(
                f"Frame index {frame_idx} from {image_path} is outside trajectory length {traj_rows.shape[0]}"
            )
        selected_rows.append(traj_rows[frame_idx])

    return replica_pose_rows_to_traj_tum(np.stack(selected_rows, axis=0))


def _replica_multiagent_root():
    return os.environ.get(
        "SSTREAMVGGT_REPLICA_MULTIAGENT_ROOT",
        "/home/dongjae/data/ReplicaMultiagent",
    )


def _replica_multiagent_part_dirs(scene_dir):
    if not os.path.isdir(scene_dir):
        return []
    return sorted(
        part_dir
        for part_dir in glob.glob(os.path.join(scene_dir, "*"))
        if os.path.isdir(part_dir)
        and os.path.isdir(os.path.join(part_dir, "results"))
        and os.path.exists(os.path.join(part_dir, "traj.txt"))
    )


def _replica_multiagent_sequences(root=None):
    root = root or _replica_multiagent_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        scene
        for scene in os.listdir(root)
        if len(_replica_multiagent_part_dirs(os.path.join(root, scene))) >= 2
    )


def _replica_multiagent_image_files(scene_dir):
    part_dirs = _replica_multiagent_part_dirs(scene_dir)
    if len(part_dirs) < 2:
        raise FileNotFoundError(
            f"ReplicaMultiagent scene needs at least two part folders with results/ and traj.txt: {scene_dir}"
        )

    filelist = []
    for part_dir in part_dirs:
        part_files = _replica_image_files(os.path.join(part_dir, "results"))
        if not part_files:
            raise FileNotFoundError(f"No ReplicaMultiagent RGB frames found in {part_dir}/results")
        filelist.extend(part_files)
    return filelist


def _replica_multiagent_frame_index(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if not stem.startswith("frame") or not stem[5:].isdigit():
        raise ValueError(f"Unsupported ReplicaMultiagent frame filename: {image_path}")
    return int(stem[5:])


def _replica_multiagent_gt_traj_from_filelist(img_path, anno_path, seq, filelist):
    import numpy as np
    from eval.pose_evaluation.evo_utils import replica_pose_rows_to_traj_tum

    if not filelist:
        raise ValueError(f"No selected frames for ReplicaMultiagent sequence {seq}")

    traj_cache = {}
    selected_rows = []
    for image_path in filelist:
        results_dir = os.path.dirname(image_path)
        part_dir = os.path.dirname(results_dir)
        traj_path = os.path.join(part_dir, "traj.txt")
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Missing ReplicaMultiagent traj.txt for frame {image_path}")

        if traj_path not in traj_cache:
            traj_rows = np.loadtxt(traj_path, dtype=np.float64)
            if traj_rows.ndim == 1:
                traj_rows = traj_rows[None, :]
            if traj_rows.ndim != 2 or traj_rows.shape[1] not in (12, 16):
                raise ValueError(
                    f"Unsupported ReplicaMultiagent trajectory shape {traj_rows.shape} in {traj_path}"
                )
            traj_cache[traj_path] = traj_rows

        frame_idx = _replica_multiagent_frame_index(image_path)
        traj_rows = traj_cache[traj_path]
        if frame_idx >= traj_rows.shape[0]:
            raise IndexError(
                f"Frame index {frame_idx} from {image_path} is outside trajectory length {traj_rows.shape[0]}"
            )
        selected_rows.append(traj_rows[frame_idx])

    return replica_pose_rows_to_traj_tum(np.stack(selected_rows, axis=0))


def _nrgbd_root():
    return os.environ.get("SSTREAMVGGT_NRGBD_ROOT", "/home/dongjae/data/neural_rgbd_data")


def _nrgbd_sequences(root=None):
    root = root or _nrgbd_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        seq
        for seq in os.listdir(root)
        if os.path.isdir(os.path.join(root, seq, "images"))
        and os.path.exists(os.path.join(root, seq, "poses.txt"))
    )


def _nrgbd_image_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("img"):
        suffix = stem[3:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, stem)


def _nrgbd_image_files(dir_path):
    return sorted(
        glob.glob(os.path.join(dir_path, "img*.png")),
        key=_nrgbd_image_sort_key,
    )


KITTI_ODOMETRY_GT_SEQS = [f"{i:02d}" for i in range(11)]
KITTI_ODOMETRY_TEST_SEQS = [f"{i:02d}" for i in range(11, 22)]
VBR_SEQS = [
    "campus_train1",
    "diag_train0",
    "colosseo_train0",
    "pincio_train0",
    "ciampino_train1",
    "spagna_train0",
    "campus_train0",
]


# Define the merged dataset metadata dictionary
dataset_metadata = {
    "7scenes": {
        "img_path": _seven_scenes_root(),
        "anno_path": _seven_scenes_root(),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "filelist_func": _seven_scenes_color_files,
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": "7scenes",
        "seq_list": _seven_scenes_test_sequences(),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "kitti": {
        "img_path": "data/kitti/depth_selection/val_selection_cropped/image_gathered",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    },
    "kitti_trajectory": {
        "img_path": os.path.join(_kitti_odometry_root(), "sequences"),
        "anno_path": os.path.join(_kitti_odometry_root(), "poses"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "image_2"),
        "filelist_func": _kitti_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, f"{seq}.txt"),
        "traj_format": "kitti",
        "seq_list": KITTI_ODOMETRY_GT_SEQS,
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "kitti_trajectory_test": {
        "img_path": os.path.join(_kitti_odometry_root(), "sequences"),
        "anno_path": None,
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "image_2"),
        "filelist_func": _kitti_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": KITTI_ODOMETRY_TEST_SEQS,
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "vbr": {
        "img_path": _vbr_root(),
        "anno_path": os.path.join(_vbr_root(), "processed_gt"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"{seq}_processed_aligned", "rgb"
        ),
        "filelist_func": _vbr_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            anno_path, f"{seq}_gt.txt"
        ),
        "traj_format": "tum",
        "seq_list": VBR_SEQS,
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "replica": {
        "img_path": _replica_root(),
        "anno_path": _replica_root(),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "results"),
        "filelist_func": _replica_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq, "traj.txt"),
        "traj_format": "replica",
        "seq_list": _replica_sequences(),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "replica_repeat": {
        "img_path": _replica_root(),
        "anno_path": _replica_root(),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "results"),
        "filelist_func": _replica_repeat_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "gt_traj_loader": _replica_repeat_gt_traj_from_filelist,
        "traj_format": "replica_repeat",
        "seq_list": _replica_sequences(),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "replica_multiagent": {
        "img_path": _replica_multiagent_root(),
        "anno_path": _replica_multiagent_root(),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "filelist_func": _replica_multiagent_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "gt_traj_loader": _replica_multiagent_gt_traj_from_filelist,
        "traj_format": "replica_multiagent",
        "seq_list": _replica_multiagent_sequences(),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "NRGBD": {
        "img_path": _nrgbd_root(),
        "anno_path": _nrgbd_root(),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "images"),
        "filelist_func": _nrgbd_image_files,
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq, "poses.txt"),
        "traj_format": "nrgbd",
        "seq_list": _nrgbd_sequences(),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": lambda args, img_path: process_nyu(args, img_path),
    },
    "scannet": {
        "img_path": "data/scannetv2",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-257": {
        "img_path": "data/scannetv2_3_257",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-129": {
        "img_path": "data/scannetv2_3_129",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-65": {
        "img_path": "data/scannetv2_3_65",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-33": {
        "img_path": "data/scannetv2_3_33",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "/home/ma-user/work/dataset/3D_Reconstruction/sintel/training/final",
        "anno_path": "/home/ma-user/work/dataset/3D_Reconstruction/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
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
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
    },
}

scannet_numbers = [50, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
scannet_configs = {
    f"scannet_s3_{num}": {
        "img_path": "data/long_scannet_s3",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(img_path, seq, f"color_{num}"),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(
            img_path, seq, f"pose_{num}.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    }
    for num in scannet_numbers
}
# then update dataset_metadata
dataset_metadata.update(scannet_configs)

tum_numbers = [50, 90, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
tum_configs = {
    f"tum_{num}": {
        "img_path": "/home/ma-user/work/dataset/3D_Reconstruction/long_tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(img_path, seq, f"rgb_{num}"),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(
            img_path, seq, f"groundtruth_{num}.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
        }
    for num in tum_numbers
}
dataset_metadata.update(tum_configs)


# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/rgb/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = (
            ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
            if args.seq_list is None
            else args.seq_list
        )
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_110/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir


def process_nyu(args, img_path):
    filelist = sorted(glob.glob(f"{img_path}/*.png"))
    save_dir = f"{args.output_dir}"
    yield filelist, save_dir


def process_scannet(args, img_path):
    seq_list = sorted(glob.glob(f"{img_path}/*"))
    for seq in tqdm(seq_list):
        filelist = sorted(glob.glob(f"{seq}/color_90/*.jpg"))
        save_dir = f"{args.output_dir}/{os.path.basename(seq)}"
        yield filelist, save_dir


def process_sintel(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
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
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/{seq}/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir
