"""Check that everything the W5 experiments need is in place. Exit 1 on failure.

Reads the environment exported by w5/config.sh (dataset roots, GPU, matrix).
"""

import glob
import importlib
import os
import sys

ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"  [{'OK' if passed else 'MISSING'}] {name}" + (f": {detail}" if detail else ""))


def check_dir(name, root, pattern, min_count=1):
    if not os.path.isdir(root):
        report(name, False, f"{root} does not exist")
        return
    hits = glob.glob(os.path.join(root, pattern))
    report(name, len(hits) >= min_count, f"{root} ({len(hits)} match '{pattern}')")


print("[preflight] python environment")
print(f"  python: {sys.executable}")
for module in ("torch", "accelerate", "open3d", "evo", "cv2", "pykdtree", "roma", "einops", "transformers"):
    try:
        importlib.import_module(module)
        report(module, True)
    except Exception as exc:  # noqa: BLE001
        report(module, False, str(exc).splitlines()[0])

try:
    import torch

    gpu = os.environ.get("GPU", "0")
    report("CUDA", torch.cuda.is_available(), f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', gpu)}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__} cuda {torch.version.cuda}")
        free, total = torch.cuda.mem_get_info(0)
        used_gb = (total - free) / 2**30
        print(f"  GPU memory in use by other processes: {used_gb:.1f} / {total / 2**30:.1f} GB")
        if used_gb > 1.0:
            print("  [WARN] GPU is shared with other jobs: FPS/latency/peak-memory numbers will be unreliable"
                  " (and long sequences may OOM). Pick a free GPU via GPU=<idx>.")
except Exception:  # noqa: BLE001
    pass

import subprocess

check = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_attn_cuda.py")],
    capture_output=True, text=True,
)
report("attn-cuda kernel", check.returncode == 0, check.stdout.strip() + ("" if check.returncode == 0 else " -> run: bash w5/setup_env.sh"))

repo = os.environ.get("W5_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ckpt = os.environ.get("W5_CKPT", os.path.join(repo, "ckpt", "checkpoints.pth"))
size = os.path.getsize(ckpt) if os.path.exists(ckpt) else 0
report("checkpoint", size > 4_000_000_000, f"{ckpt} ({size / 2**30:.2f} GB)")

print("[preflight] datasets")
benches = os.environ.get("BENCHES", "mv_recon pose").split()
if "mv_recon" in benches:
    for name in [d.strip() for d in os.environ.get("MV_DATASETS", "7scenes,NRGBD,ETH3D").split(",") if d.strip()]:
        if name == "7scenes":
            check_dir("mv_recon 7scenes", os.environ.get("SSTREAMVGGT_7SCENES_ROOT", ""), "*/seq-*")
        elif name == "NRGBD":
            check_dir("mv_recon NRGBD", os.environ.get("SSTREAMVGGT_NRGBD_ROOT", ""), "*/images")
        elif name.startswith("ETH3D"):
            check_dir(f"mv_recon {name}", os.environ.get("SSTREAMVGGT_ETH3D_ROOT", ""), "*")
        else:
            report(f"mv_recon {name}", False, "unknown dataset")
if "pose" in benches:
    for name in os.environ.get("POSE_DATASETS", "tum scannet").split():
        if name == "tum":
            check_dir("pose tum", os.environ.get("SSTREAMVGGT_TUM_ROOT", ""), "*/rgb_90")
        elif name == "scannet":
            check_dir("pose scannet", os.environ.get("SSTREAMVGGT_SCANNET_ROOT", ""), "*/color_90")
        else:
            report(f"pose {name}", False, "unknown dataset")

print(f"[preflight] {'ALL OK' if ok else 'FAILED (see MISSING above; edit w5/config.sh or run w5/setup_env.sh)'}")
sys.exit(0 if ok else 1)
