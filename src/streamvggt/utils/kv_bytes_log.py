"""Per-step KV cache storage in bytes (actual tensor nbytes, after each step's eviction).

Columns per step: tokens (sum over the global layers), K / V bytes and dtypes, IDC sidecar state (KVConfidenceState
tensors), projected key cache (right-sketched rows kept for the leverage computation), other eviction auxiliaries
(RLS factorisations), camera-head cache, and the total. Enabled by a launcher with configure() / set_scene();
StreamVGGT.inference calls record() once per streamed chunk.
"""

import csv
import os
from typing import Optional

import torch

_CONFIG = {"dir": None, "path": None}
FIELDS = ["step", "tokens", "k_bytes", "v_bytes", "kv_bytes", "state_bytes", "projected_bytes", "aux_bytes",
          "camera_bytes", "total_bytes", "k_dtype", "v_dtype"]


def configure(log_dir: Optional[str]):
    _CONFIG.update(dir=log_dir, path=None)


def set_scene(scene_key: Optional[str]):
    _CONFIG["path"] = None
    if _CONFIG["dir"] and scene_key:
        os.makedirs(_CONFIG["dir"], exist_ok=True)
        _CONFIG["path"] = os.path.join(_CONFIG["dir"], f"{scene_key}.csv")
        with open(_CONFIG["path"], "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)


def enabled() -> bool:
    return _CONFIG["path"] is not None


def _nbytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(o) for o in obj)
    if hasattr(obj, "__dataclass_fields__"):
        return sum(_nbytes(getattr(obj, name)) for name in obj.__dataclass_fields__)
    return 0


@torch.no_grad()
def record(step: int, past_key_values, past_key_values_camera, aggregator) -> None:
    if not enabled():
        return
    row = {k: 0 for k in FIELDS if k.endswith("bytes") or k == "tokens"}
    k_dtype = v_dtype = ""
    for kv in past_key_values or []:
        if kv is None:
            continue
        k, v = kv[0], kv[1]
        row["tokens"] += int(k.shape[2])
        row["k_bytes"] += _nbytes(k)
        row["v_bytes"] += _nbytes(v)
        k_dtype, v_dtype = str(k.dtype).replace("torch.", ""), str(v.dtype).replace("torch.", "")
        for extra in kv[2:]:  # KVConfidenceState sidecar
            row["state_bytes"] += _nbytes(extra)
    for block in getattr(aggregator, "global_blocks", []):
        for manager in getattr(block.attn, "_eviction_managers", {}).values():
            row["projected_bytes"] += _nbytes(getattr(manager, "_projected_key_cache", None))
            row["projected_bytes"] += _nbytes(getattr(manager, "_projected_key_pre_norm_cache", None))
            for name in ("cached_ktk", "cached_rls_chol", "cached_rls_inv", "cached_rls_lam"):
                row["aux_bytes"] += _nbytes(getattr(manager, name, None))
    for kv in past_key_values_camera or []:
        if kv is not None:
            row["camera_bytes"] += _nbytes(kv[:2])
    row["kv_bytes"] = row["k_bytes"] + row["v_bytes"]
    row["total_bytes"] = (row["kv_bytes"] + row["state_bytes"] + row["projected_bytes"] + row["aux_bytes"]
                          + row["camera_bytes"])
    row.update(step=int(step), k_dtype=k_dtype, v_dtype=v_dtype)
    with open(_CONFIG["path"], "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
