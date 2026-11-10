"""Lightweight profiling for the W5 rebuttal analysis.

Collects, per streamed sequence:
  * per-frame wall-clock latency (the per-frame ``.cpu()`` transfer already
    synchronizes the stream, so ``perf_counter`` deltas reflect GPU completion),
  * IDC (token eviction) and KABA (layer budget allocation) GPU time via
    non-synchronizing CUDA events, resolved once at the end of the sequence,
  * actual stored KV tokens per global layer after eviction (shape reads only),
  * camera-head trunk cache size (kept separate from the global-layer budget),
  * bytes held by KV tensors, confidence state and eviction-manager side state.

The profiler is a module-level singleton so that the attention/aggregator hot
paths only pay a single boolean check when it is disabled.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

IDC = "idc"
KABA = "kaba"


class W5Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self._events: Dict[str, List] = {IDC: [], KABA: []}
        self._frame_start: Optional[float] = None
        self.frame_records: List[Dict[str, float]] = []
        self.kv_records: List[Dict[str, float]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._frame_idx = 0

    def configure(self, enabled: bool) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.reset()

    # ----------------------------------------------------------- CUDA events
    def start(self):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(self, category: str, start_event) -> None:
        if start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._events[category].append((self._frame_idx, start_event, end_event))

    # ------------------------------------------------------------ frame loop
    def begin_sequence(self) -> None:
        if not self.enabled:
            return
        self.reset()
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def begin_chunk(self, frame_idx: int) -> None:
        if not self.enabled:
            return
        self._frame_idx = int(frame_idx)
        self._frame_start = time.perf_counter()

    def end_chunk(self, frame_ids: List[int]) -> None:
        if not self.enabled or self._frame_start is None:
            return
        elapsed_ms = (time.perf_counter() - self._frame_start) * 1000.0
        per_frame = elapsed_ms / max(len(frame_ids), 1)
        for fid in frame_ids:
            self.frame_records.append({"frame_idx": int(fid), "latency_ms": per_frame})
        self._frame_start = None

    def record_cache(
        self,
        frame_idx: int,
        past_key_values,
        past_key_values_camera,
        global_blocks,
        budget: Optional[int],
    ) -> None:
        """Record actual stored cache sizes right after eviction (no sync)."""
        if not self.enabled:
            return
        kv_bytes = 0
        conf_bytes = 0
        layer_tokens = []
        for layer_kv in past_key_values:
            if layer_kv is None:
                layer_tokens.append(0)
                continue
            k, v = layer_kv[0], layer_kv[1]
            layer_tokens.append(int(k.shape[2]))
            kv_bytes += _tensor_bytes(k) + _tensor_bytes(v)
            if len(layer_kv) >= 3 and layer_kv[-1] is not None:
                conf_bytes += _object_tensor_bytes(layer_kv[-1])
        side_bytes = 0
        for block in global_blocks:
            attn = getattr(block, "attn", None)
            managers = getattr(attn, "_eviction_managers", {}) or {}
            for manager in managers.values():
                side_bytes += _object_tensor_bytes(manager)
        cam_tokens = 0
        cam_bytes = 0
        for layer_kv in past_key_values_camera or []:
            if layer_kv is None:
                continue
            cam_tokens = max(cam_tokens, int(layer_kv[0].shape[2]))
            cam_bytes += _tensor_bytes(layer_kv[0]) + _tensor_bytes(layer_kv[1])
        total_tokens = int(sum(layer_tokens))
        record = {
            "frame_idx": int(frame_idx),
            "budget": int(budget) if budget is not None else -1,
            "kv_tokens_total": total_tokens,
            "kv_excess": (total_tokens - int(budget)) if budget is not None else 0,
            "kv_tokens_min_layer": int(min(layer_tokens)) if layer_tokens else 0,
            "kv_tokens_max_layer": int(max(layer_tokens)) if layer_tokens else 0,
            "kv_bytes": kv_bytes,
            "conf_state_bytes": conf_bytes,
            "eviction_side_bytes": side_bytes,
            "cam_cache_tokens": cam_tokens,
            "cam_cache_bytes": cam_bytes,
        }
        for layer_id, tokens in enumerate(layer_tokens):
            record[f"L{layer_id:02d}"] = tokens
        self.kv_records.append(record)

    def end_sequence(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        per_frame = {IDC: {}, KABA: {}}
        totals = {IDC: 0.0, KABA: 0.0}
        for category, events in self._events.items():
            for frame_idx, start_event, end_event in events:
                ms = float(start_event.elapsed_time(end_event))
                totals[category] += ms
                per_frame[category][frame_idx] = per_frame[category].get(frame_idx, 0.0) + ms
        for rec in self.frame_records:
            rec["idc_ms"] = per_frame[IDC].get(rec["frame_idx"], 0.0)
            rec["kaba_ms"] = per_frame[KABA].get(rec["frame_idx"], 0.0)
            rec["pure_ms"] = rec["latency_ms"] - rec["idc_ms"] - rec["kaba_ms"]
        self._events = {IDC: [], KABA: []}
        return self.summary(idc_ms=totals[IDC], kaba_ms=totals[KABA])

    # ---------------------------------------------------------------- output
    def summary(self, idc_ms: float, kaba_ms: float) -> Dict[str, float]:
        n_frames = len(self.frame_records)
        e2e_s = (self.end_time - self.start_time) if self.start_time is not None else 0.0
        infer_s = e2e_s - (idc_ms + kaba_ms) / 1000.0
        lat = np.array([r["latency_ms"] for r in self.frame_records], dtype=np.float64)
        steady = lat[10:] if lat.size > 10 else lat
        excess = np.array([r["kv_excess"] for r in self.kv_records], dtype=np.float64)
        tokens = np.array([r["kv_tokens_total"] for r in self.kv_records], dtype=np.float64)
        kv_total_bytes = np.array(
            [r["kv_bytes"] + r["conf_state_bytes"] + r["eviction_side_bytes"] for r in self.kv_records],
            dtype=np.float64,
        )
        last = self.kv_records[-1] if self.kv_records else {}
        return {
            "n_frames": n_frames,
            "e2e_time_s": e2e_s,
            "e2e_fps": n_frames / e2e_s if e2e_s > 0 else 0.0,
            "infer_time_s": infer_s,
            "infer_fps": n_frames / infer_s if infer_s > 0 else 0.0,
            "idc_time_s": idc_ms / 1000.0,
            "kaba_time_s": kaba_ms / 1000.0,
            "steady_fps": 1000.0 / float(steady.mean()) if steady.size else 0.0,
            "lat_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
            "lat_p95_ms": float(np.percentile(lat, 95)) if lat.size else 0.0,
            "lat_max_ms": float(lat.max()) if lat.size else 0.0,
            "kv_tokens_max": int(tokens.max()) if tokens.size else 0,
            "kv_tokens_final": int(tokens[-1]) if tokens.size else 0,
            "kv_excess_max": int(excess.max()) if excess.size else 0,
            "kv_excess_mean": float(np.clip(excess, 0, None).mean()) if excess.size else 0.0,
            "n_frames_excess_pos": int((excess > 0).sum()) if excess.size else 0,
            "kv_bytes_max_MB": float(kv_total_bytes.max()) / 2**20 if kv_total_bytes.size else 0.0,
            "kv_only_bytes_final_MB": float(last.get("kv_bytes", 0)) / 2**20,
            "cam_cache_tokens_final": int(last.get("cam_cache_tokens", 0)),
            "cam_cache_bytes_final_MB": float(last.get("cam_cache_bytes", 0)) / 2**20,
        }

    def write_sequence_csvs(self, out_dir: str, seq_name: str) -> None:
        if not self.enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        safe = str(seq_name).replace("/", "__")
        _write_rows(os.path.join(out_dir, f"{safe}_frames.csv"), self.frame_records)
        _write_rows(os.path.join(out_dir, f"{safe}_kv.csv"), self.kv_records)


def _write_rows(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_summary_row(path: str, row: Dict) -> None:
    """Append one row to a summary CSV, writing a header if the file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def gpu_memory_begin() -> float:
    """Reset peak stats and return currently allocated GB (model + inputs)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated() / 2**30


def gpu_memory_end() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_alloc_GB": 0.0, "peak_reserved_GB": 0.0}
    torch.cuda.synchronize()
    return {
        "peak_alloc_GB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_GB": torch.cuda.max_memory_reserved() / 2**30,
    }


def gpu_info() -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {"gpu": "cpu"}
    return {"gpu": torch.cuda.get_device_name(torch.cuda.current_device())}


def _tensor_bytes(t) -> int:
    if isinstance(t, torch.Tensor):
        return int(t.numel() * t.element_size())
    return 0


def _object_tensor_bytes(obj, _depth: int = 0) -> int:
    """Sum bytes of tensors held directly by ``obj`` (one level into containers)."""
    if isinstance(obj, torch.Tensor):
        return _tensor_bytes(obj)
    if _depth > 2:
        return 0
    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple)):
        values = obj
    elif hasattr(obj, "__dict__"):
        values = vars(obj).values()
    else:
        return 0
    total = 0
    for value in values:
        if isinstance(value, torch.Tensor):
            total += _tensor_bytes(value)
        elif isinstance(value, (dict, list, tuple)) or (
            hasattr(value, "__dict__") and not isinstance(value, (torch.nn.Module, type))
        ):
            total += _object_tensor_bytes(value, _depth + 1)
    return total


W5 = W5Profiler()
