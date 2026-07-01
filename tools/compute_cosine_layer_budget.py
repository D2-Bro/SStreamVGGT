#!/usr/bin/env python3
"""Compute one fixed layer-budget profile from multiple calibration sequences.

Manifest format::

    {"sequences": [{"name": "kitti_00", "image_dir": "/data/kitti/00", "image_glob": "*.png"}]}
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def compute_cosine_similarity(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    """Return mean row-wise cosine similarity for matching block tensors."""
    if x_in.shape != x_out.shape:
        raise ValueError(f"Block input/output shapes differ: {tuple(x_in.shape)} vs {tuple(x_out.shape)}")
    if x_in.numel() == 0:
        raise ValueError("Cannot compute cosine similarity for an empty tensor")
    x_in = x_in.detach().float().reshape(-1, x_in.shape[-1])
    x_out = x_out.detach().float().reshape(-1, x_out.shape[-1])
    return F.cosine_similarity(x_in, x_out, dim=-1).mean()


def cosine_to_proportions(cosine_similarity, temperature: float = 0.5, eps: float = 1e-6):
    """Convert layer cosine similarity to GHOST-style fixed proportions."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    cosine = torch.as_tensor(cosine_similarity, dtype=torch.float32).reshape(-1)
    if cosine.numel() == 0 or not torch.isfinite(cosine).all():
        raise ValueError("cosine_similarity must be non-empty and finite")
    importance = (1.0 - cosine).clamp(min=eps, max=1.0)
    proportions = torch.softmax(importance / float(temperature), dim=0)
    return importance, proportions


def load_sequence_manifest(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration manifest not found: {path}. "
            "Create it as JSON with a sequences list containing name, image_dir, and image_glob."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("sequences") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a non-empty list or contain a non-empty sequences list")

    resolved = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest sequence {index} must be an object")
        name = str(entry.get("name", f"sequence_{index}"))
        image_dir = entry.get("image_dir")
        image_glob = entry.get("image_glob", entry.get("glob"))
        if not image_dir or not image_glob:
            raise ValueError(f"Manifest sequence {name!r} requires image_dir and image_glob")
        image_dir = Path(image_dir).expanduser()
        if not image_dir.is_absolute():
            image_dir = (path.parent / image_dir).resolve()
        resolved.append({"name": name, "image_dir": str(image_dir), "image_glob": str(image_glob)})
    return path, resolved


def resolve_sequence_images(entry, frame_stride: int, max_frames: int):
    pattern = os.path.join(entry["image_dir"], entry["image_glob"])
    image_paths = sorted(path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path))
    image_paths = image_paths[::frame_stride]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        name = entry["name"]
        raise ValueError(f"No images matched calibration sequence {name!r}: {pattern}")
    return image_paths


class LayerCosineCollector:
    def __init__(self, blocks):
        self.sums = [0.0] * len(blocks)
        self.counts = [0] * len(blocks)
        self.inputs = {}
        self.handles = []
        for layer_idx, block in enumerate(blocks):
            self.handles.append(block.register_forward_pre_hook(self._make_pre_hook(layer_idx)))
            self.handles.append(block.register_forward_hook(self._make_post_hook(layer_idx)))

    def _make_pre_hook(self, layer_idx):
        def hook(module, inputs):
            self.inputs[layer_idx] = inputs[0].detach()
        return hook

    def _make_post_hook(self, layer_idx):
        def hook(module, inputs, output):
            x_in = self.inputs.pop(layer_idx)
            x_out = output[0] if isinstance(output, tuple) else output
            value = compute_cosine_similarity(x_in, x_out).item()
            self.sums[layer_idx] += float(value)
            self.counts[layer_idx] += 1
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.inputs.clear()

    def means(self):
        missing = [idx for idx, count in enumerate(self.counts) if count == 0]
        if missing:
            raise RuntimeError(f"No cosine samples collected for layers: {missing}")
        return [total / count for total, count in zip(self.sums, self.counts)]


def reset_layer_budget_state(model):
    names = ("last_scores", "last_layer_budget_scores", "last_layer_budget_base_scores", "last_layer_budget_value_norms")
    for name in names:
        tensor = getattr(model.aggregator, name, None)
        if tensor is not None:
            tensor.zero_()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="StreamVGGT checkpoint path")
    parser.add_argument("--manifest", required=True, help="JSON sequence manifest")
    parser.add_argument("--output_path", required=True, help="Output proportions JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total_budget", type=int, default=1200000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--image_mode", choices=("crop", "pad"), default="crop")
    return parser


def main():
    args = build_parser().parse_args()
    if args.frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {args.frame_stride}")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {args.max_frames}")

    manifest_path, sequences = load_sequence_manifest(args.manifest)

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = StreamVGGT(total_budget=args.total_budget)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)

    collector = LayerCosineCollector(model.aggregator.global_blocks)
    processed = []
    try:
        with torch.no_grad():
            for entry in sequences:
                image_paths = resolve_sequence_images(entry, args.frame_stride, args.max_frames)
                sequence_name = entry["name"]
                print(f"[cosine-budget] {sequence_name}: {len(image_paths)} frames")
                images = load_and_preprocess_images(image_paths, mode=args.image_mode).to(device)
                frames = [{"img": image.unsqueeze(0)} for image in images]
                reset_layer_budget_state(model)
                model.inference(
                    frames,
                    cache_results=False,
                    eviction_policy="mean",
                    layer_budget_strategy="uniform",
                )
                processed.append({"name": sequence_name, "num_frames": len(frames)})
                del frames, images
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        collector.close()

    cosine = collector.means()
    importance, proportions = cosine_to_proportions(cosine, temperature=args.temperature, eps=args.eps)
    result = {
        "strategy": "cosine_precomputed",
        "num_layers": model.aggregator.depth,
        "num_heads": model.aggregator.global_blocks[0].attn.num_heads,
        "cos_sim_per_layer": cosine,
        "importance_per_layer": importance.tolist(),
        "proportions": proportions.tolist(),
        "temperature": args.temperature,
        "eps": args.eps,
        "calibration_manifest": str(manifest_path),
        "calibration_sequences": processed,
        "settings": {
            "total_budget": args.total_budget,
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "image_mode": args.image_mode,
            "eviction_policy": "mean",
            "layer_budget_strategy": "uniform",
        },
    }
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved cosine layer-budget proportions to {output_path}")


if __name__ == "__main__":
    main()
