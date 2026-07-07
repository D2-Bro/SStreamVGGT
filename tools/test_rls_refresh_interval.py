#!/usr/bin/env python3
"""Smoke tests for ridge leverage score refresh interval caching."""

from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.eviction import EvictionManager  # noqa: E402


def _set_frame(manager: EvictionManager, frame_idx: int, *, granularity: str = "layer") -> None:
    manager._set_leverage_diag_context(
        layer_id=0,
        step_idx=frame_idx,
        current_frame_idx=frame_idx,
        granularity=granularity,
        batch_size=1,
        num_heads=1,
    )


def _manager(method: str, *, interval: int, profile: bool = False, device: torch.device | str = "cpu") -> EvictionManager:
    kwargs = {
        "policy": "svd_leverage",
        "profile": profile,
        "leverage_approx_method": method,
        "leverage_sketch_dim": 0,
        "leverage_ridge_lambda": 1e-3,
        "leverage_ridge_lambda_mode": "relative",
        "leverage_ridge_score_chunk_size": 3,
        "leverage_ridge_jitter": 1e-6,
        "rls_refresh_interval": interval,
        "leverage_random_seed": 123,
    }
    if method == "right_sketch_ridge":
        kwargs["leverage_ridge_dim"] = 4
    mgr = EvictionManager(**kwargs)
    # Touch the target device through the first input rather than manager state.
    return mgr


def check_interval_one_matches_fresh() -> None:
    torch.manual_seed(7)
    frames = [torch.randn(9, 6) for _ in range(4)]
    for method in ("full_d_ridge", "right_sketch_ridge"):
        cached = _manager(method, interval=1)
        for frame_idx, mat in enumerate(frames):
            _set_frame(cached, frame_idx)
            score_cached = cached.compute_svd_leverage_scores(mat)
            fresh = _manager(method, interval=1)
            _set_frame(fresh, frame_idx)
            score_fresh = fresh.compute_svd_leverage_scores(mat)
            if not torch.allclose(score_cached, score_fresh, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"interval=1 differs from fresh baseline for {method} at frame {frame_idx}")
        if cached.rls_cache_hit_count != 0 or cached.rls_refresh_count != len(frames):
            raise AssertionError(
                f"interval=1 should refresh every frame for {method}, "
                f"got refresh={cached.rls_refresh_count} hits={cached.rls_cache_hit_count}"
            )


def check_interval_three_refresh_counts() -> None:
    torch.manual_seed(11)
    manager = _manager("full_d_ridge", interval=3, profile=True)
    expected_refresh_counts = [1, 1, 1, 2, 2, 2, 3]
    expected_hit_counts = [0, 1, 2, 2, 3, 4, 4]
    expected_refreshed = [1, 0, 0, 1, 0, 0, 1]
    for frame_idx in range(7):
        mat = torch.randn(10, 5)
        _set_frame(manager, frame_idx)
        manager.compute_svd_leverage_scores(mat)
        profile = manager._last_leverage_profile
        if manager.rls_refresh_count != expected_refresh_counts[frame_idx]:
            raise AssertionError(f"bad refresh count at frame {frame_idx}: {manager.rls_refresh_count}")
        if manager.rls_cache_hit_count != expected_hit_counts[frame_idx]:
            raise AssertionError(f"bad cache hit count at frame {frame_idx}: {manager.rls_cache_hit_count}")
        if int(profile.get("rls_cache_refreshed", -1)) != expected_refreshed[frame_idx]:
            raise AssertionError(f"bad refreshed profile flag at frame {frame_idx}: {profile}")
    if manager.last_rls_refresh_frame != 6:
        raise AssertionError(f"expected last refresh frame 6, got {manager.last_rls_refresh_frame}")


def check_offset_interval_refresh_counts() -> None:
    torch.manual_seed(17)
    manager = _manager("full_d_ridge", interval=4, profile=True)
    frames = [3, 5, 7, 11]
    expected_refresh_counts = [1, 1, 2, 3]
    expected_hit_counts = [0, 1, 1, 1]
    expected_refreshed = [1, 0, 1, 1]
    expected_last_refresh = [3, 3, 7, 11]
    for idx, frame_idx in enumerate(frames):
        mat = torch.randn(10, 5)
        _set_frame(manager, frame_idx)
        manager.compute_svd_leverage_scores(mat)
        profile = manager._last_leverage_profile
        if manager.rls_refresh_count != expected_refresh_counts[idx]:
            raise AssertionError(f"bad offset refresh count at frame {frame_idx}: {manager.rls_refresh_count}")
        if manager.rls_cache_hit_count != expected_hit_counts[idx]:
            raise AssertionError(f"bad offset cache hit count at frame {frame_idx}: {manager.rls_cache_hit_count}")
        if int(profile.get("rls_cache_refreshed", -1)) != expected_refreshed[idx]:
            raise AssertionError(f"bad offset refreshed profile flag at frame {frame_idx}: {profile}")
        if manager.last_rls_refresh_frame != expected_last_refresh[idx]:
            raise AssertionError(
                f"bad last refresh frame at frame {frame_idx}: {manager.last_rls_refresh_frame}"
            )


def check_reset_forces_refresh() -> None:
    manager = _manager("full_d_ridge", interval=5)
    _set_frame(manager, 0)
    manager.compute_svd_leverage_scores(torch.randn(8, 4))
    _set_frame(manager, 1)
    manager.compute_svd_leverage_scores(torch.randn(8, 4))
    if manager.rls_cache_hit_count != 1:
        raise AssertionError("frame 1 should reuse cache before reset")
    manager.reset_rls_cache()
    _set_frame(manager, 2)
    manager.compute_svd_leverage_scores(torch.randn(8, 4))
    if manager.rls_refresh_count != 1 or manager.rls_cache_hit_count != 0 or manager.last_rls_refresh_frame != 2:
        raise AssertionError("reset_rls_cache did not force a refresh on the next frame")


def check_device_dtype(device: torch.device) -> None:
    dtypes = [torch.float32, torch.float64]
    if device.type == "cuda":
        dtypes.append(torch.float16)
    manager = _manager("full_d_ridge", interval=2, device=device)
    for frame_idx, dtype in enumerate(dtypes):
        mat = torch.randn(7, 3, device=device, dtype=dtype)
        _set_frame(manager, frame_idx)
        scores = manager.compute_svd_leverage_scores(mat)
        if scores.device != mat.device:
            raise AssertionError(f"score device mismatch: {scores.device} vs {mat.device}")
        if scores.dtype != torch.float32:
            raise AssertionError(f"scores should be float32, got {scores.dtype}")
        if not torch.isfinite(scores).all():
            raise AssertionError("scores contain non-finite values")


def check_backward_reuse_path() -> None:
    manager = _manager("full_d_ridge", interval=4)
    _set_frame(manager, 0)
    manager.compute_svd_leverage_scores(torch.randn(8, 4))
    mat = torch.randn(8, 4, requires_grad=True)
    _set_frame(manager, 1)
    scores = manager.compute_svd_leverage_scores(mat)
    loss = scores.sum()
    loss.backward()
    if mat.grad is None or not torch.isfinite(mat.grad).all() or mat.grad.abs().sum().item() <= 0.0:
        raise AssertionError("backward through cached RLS scoring path did not produce a valid gradient")


def check_invalid_interval() -> None:
    try:
        _manager("full_d_ridge", interval=0)
    except ValueError as exc:
        if "rls_refresh_interval" not in str(exc):
            raise AssertionError(f"unexpected ValueError: {exc}")
    else:
        raise AssertionError("rls_refresh_interval=0 should raise ValueError")


def main() -> None:
    check_interval_one_matches_fresh()
    check_interval_three_refresh_counts()
    check_offset_interval_refresh_counts()
    check_reset_forces_refresh()
    check_device_dtype(torch.device("cpu"))
    if torch.cuda.is_available():
        check_device_dtype(torch.device("cuda"))
    check_backward_reuse_path()
    check_invalid_interval()
    print("RLS refresh interval smoke tests passed")


if __name__ == "__main__":
    main()
