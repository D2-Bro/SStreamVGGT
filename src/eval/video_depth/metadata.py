import os
import glob
from tqdm import tqdm


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

# Define the merged dataset metadata dictionary
dataset_metadata = {
    "davis": {
        "img_path": "../data/eval/davis/DAVIS/JPEGImages/480p",
        "mask_path": "../data/eval/davis/DAVIS/masked_images/480p",
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
        "img_path": "/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/image_gathered",  # Default path
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
    "bonn": {
        "img_path": "../data/eval/bonn/rgbd_bonn_dataset",
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
        "img_path": "..、data/eval/nyu_v2/val/nyu_images",
        "mask_path": None,
        "process_func": lambda args, img_path: process_nyu(args, img_path),
    },
    "scannet": {
        "img_path": "../data/eval/scannetv2",
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
        "process_func": lambda args, img_path: process_replica(args, img_path),
    },
    "tum": {
        "img_path": "../data/eval/tum",
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
        "img_path": "/home/dongjae/data/datasets/sintel/training/final",
        "anno_path": "../data/eval/sintel/training/camdata_left",
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

kitti_numbers = [50, 100, 110, 150, 200, 250, 300, 350, 400, 450, 500]
kitti_configs = {
    f"kitti_s1_{num}": {
        "img_path": f"/home/dongjae/data/kitti_depth/depth_selection/val_selection_cropped/image_gathered_{num}",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    }
    for num in kitti_numbers
}
dataset_metadata.update(kitti_configs)


bonn_numbers = [50, 100, 110, 150, 200, 250, 300, 350, 400, 450, 500]
bonn_configs = {
    f"bonn_{num}": {
        "img_path": "/home/dongjae/data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(
            img_path, f"rgbd_bonn_{seq}", f"rgb_{num}"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(
            img_path, f"rgbd_bonn_{seq}", f"groundtruth_{num}.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    }
    for num in bonn_numbers
}
dataset_metadata.update(bonn_configs)

bonn_config = {
    f"bonn_stride2": {
        "img_path": "/home/dongjae/data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", f"rgb_stride2"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq : os.path.join(
            img_path, f"rgbd_bonn_{seq}", f"groundtruth_stride2.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    }
}
dataset_metadata.update(bonn_config)

# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    bonn_number = (
        args.eval_dataset.split("_")[-1]
        if "_" in getattr(args, "eval_dataset", "")
        else "110"
    )
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
            filelist = sorted(
                glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_{bonn_number}/*.png")
            )
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


def process_replica(args, img_path):
    seq_list = args.seq_list if args.seq_list is not None else _replica_sequences(img_path)
    for seq in tqdm(seq_list):
        filelist = _replica_image_files(os.path.join(img_path, seq, "results"))
        save_dir = f"{args.output_dir}/{seq}"
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
