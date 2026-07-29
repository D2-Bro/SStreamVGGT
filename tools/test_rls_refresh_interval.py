#!/usr/bin/env python3
"""Smoke tests for ridge leverage score refresh interval caching."""

from __future__ import annotations

import os
import sys
from unittest import mock

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.layers.eviction import EvictionManager  # noqa: E402


def _set_frame(manager: EvictionManager, frame_idx: int, *, granularity: str = "layer") -> None:
    manager._set_rls_context(step_idx=frame_idx, current_frame_idx=frame_idx)


def _manager(
    method: str,
    *,
    interval: int,
    profile: bool = False,
    device: torch.device | str = "cpu",
    jitter: float = 1e-6,
) -> EvictionManager:
    kwargs = {
        "policy": "svd_leverage",
        "profile": profile,
        "leverage_approx_method": method,
        "leverage_ridge_lambda": 1e-3,
        "leverage_ridge_lambda_mode": "relative",
        "leverage_ridge_score_chunk_size": 3,
        "leverage_ridge_jitter": jitter,
        "rls_refresh_interval": interval,
        "leverage_random_seed": 123,
    }
    if method == "right_sketch_ridge":
        kwargs["leverage_ridge_dim"] = 4
    mgr = EvictionManager(**kwargs)
    # Touch the target device through the first input rather than manager state.
    return mgr


def _scores(manager: EvictionManager, matrix: torch.Tensor) -> torch.Tensor:
    return manager._layer_svd_leverage_scores(matrix.unsqueeze(0).unsqueeze(0)).squeeze(0)


def check_interval_one_matches_fresh() -> None:
    torch.manual_seed(7)
    frames = [torch.randn(9, 6) for _ in range(4)]
    for method in ("right_sketch_ridge",):
        cached = _manager(method, interval=1)
        for frame_idx, mat in enumerate(frames):
            _set_frame(cached, frame_idx)
            score_cached = _scores(cached, mat)
            fresh = _manager(method, interval=1)
            _set_frame(fresh, frame_idx)
            score_fresh = _scores(fresh, mat)
            if not torch.allclose(score_cached, score_fresh, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"interval=1 differs from fresh baseline for {method} at frame {frame_idx}")
        if cached.rls_cache_hit_count != 0 or cached.rls_refresh_count != len(frames):
            raise AssertionError(
                f"interval=1 should refresh every frame for {method}, "
                f"got refresh={cached.rls_refresh_count} hits={cached.rls_cache_hit_count}"
            )


def check_interval_three_refresh_counts() -> None:
    torch.manual_seed(11)
    manager = _manager("right_sketch_ridge", interval=3, profile=True)
    expected_refresh_counts = [1, 1, 1, 2, 2, 2, 3]
    expected_hit_counts = [0, 1, 2, 2, 3, 4, 4]
    expected_refreshed = [1, 0, 0, 1, 0, 0, 1]
    for frame_idx in range(7):
        mat = torch.randn(10, 5)
        _set_frame(manager, frame_idx)
        _scores(manager, mat)
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
    manager = _manager("right_sketch_ridge", interval=4, profile=True)
    frames = [3, 5, 7, 11]
    expected_refresh_counts = [1, 1, 2, 3]
    expected_hit_counts = [0, 1, 1, 1]
    expected_refreshed = [1, 0, 1, 1]
    expected_last_refresh = [3, 3, 7, 11]
    for idx, frame_idx in enumerate(frames):
        mat = torch.randn(10, 5)
        _set_frame(manager, frame_idx)
        _scores(manager, mat)
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
    manager = _manager("right_sketch_ridge", interval=5)
    _set_frame(manager, 0)
    _scores(manager, torch.randn(8, 4))
    _set_frame(manager, 1)
    _scores(manager, torch.randn(8, 4))
    if manager.rls_cache_hit_count != 1:
        raise AssertionError("frame 1 should reuse cache before reset")
    manager.reset_rls_cache()
    _set_frame(manager, 2)
    _scores(manager, torch.randn(8, 4))
    if manager.rls_refresh_count != 1 or manager.rls_cache_hit_count != 0 or manager.last_rls_refresh_frame != 2:
        raise AssertionError("reset_rls_cache did not force a refresh on the next frame")


def check_device_dtype(device: torch.device) -> None:
    dtypes = [torch.float32, torch.float64]
    if device.type == "cuda":
        dtypes.append(torch.float16)
    manager = _manager("right_sketch_ridge", interval=2, device=device)
    for frame_idx, dtype in enumerate(dtypes):
        mat = torch.randn(7, 3, device=device, dtype=dtype)
        _set_frame(manager, frame_idx)
        scores = _scores(manager, mat)
        if scores.device != mat.device:
            raise AssertionError(f"score device mismatch: {scores.device} vs {mat.device}")
        if scores.dtype != torch.float32:
            raise AssertionError(f"scores should be float32, got {scores.dtype}")
        if not torch.isfinite(scores).all():
            raise AssertionError("scores contain non-finite values")


def check_backward_reuse_path() -> None:
    manager = _manager("right_sketch_ridge", interval=4)
    _set_frame(manager, 0)
    _scores(manager, torch.randn(8, 4))
    mat = torch.randn(8, 4, requires_grad=True)
    _set_frame(manager, 1)
    scores = _scores(manager, mat)
    loss = scores.sum()
    loss.backward()
    if mat.grad is None or not torch.isfinite(mat.grad).all() or mat.grad.abs().sum().item() <= 0.0:
        raise AssertionError("backward through cached RLS scoring path did not produce a valid gradient")


def check_inverse_backend_matches_cholesky_solve() -> None:
    torch.manual_seed(23)
    manager = _manager("right_sketch_ridge", interval=4, profile=True)
    for frame_idx in (0, 1):
        mat = torch.randn(11, 5)
        _set_frame(manager, frame_idx)
        scores = _scores(manager, mat)
        profile = manager._last_leverage_profile
        if manager.cached_rls_inv is None or manager.cached_rls_chol is None:
            raise AssertionError("expected both cached inverse and cached Cholesky factor")
        if profile.get("score_backend") != "inverse":
            raise AssertionError(f"expected inverse scoring backend, got {profile}")
        candidate_k = mat.unsqueeze(0).unsqueeze(0)
        sketch_dim = manager._resolve_ridge_sketch_dim(mat.shape[-1], mat.shape[-2])
        omega = manager._get_leverage_right_sketch(
            mat.shape[-1], sketch_dim, device=mat.device, seed=manager.leverage_random_seed
        )
        work, _ = manager._project_key_with_omega(
            candidate_k, omega.view(1, mat.shape[-1], sketch_dim)
        )
        rhs = work.transpose(-2, -1).contiguous()
        solved = torch.cholesky_solve(rhs, manager.cached_rls_chol)
        expected = torch.nan_to_num((rhs * solved).sum(dim=-2), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if not torch.allclose(scores, expected, atol=1e-4, rtol=1e-4):
            raise AssertionError("inverse-cache scores differ from Cholesky-solve reference")
    if manager.rls_refresh_count != 1 or manager.rls_cache_hit_count != 1:
        raise AssertionError(
            f"expected one refresh and one cache hit, got refresh={manager.rls_refresh_count} "
            f"hits={manager.rls_cache_hit_count}"
        )
    if float(manager._last_leverage_profile.get("inverse_build", -1.0)) != 0.0:
        raise AssertionError(f"cache-hit profile should not rebuild inverse: {manager._last_leverage_profile}")


def check_invalid_interval() -> None:
    try:
        _manager("right_sketch_ridge", interval=0)
    except ValueError as exc:
        if "rls_refresh_interval" not in str(exc):
            raise AssertionError(f"unexpected ValueError: {exc}")
    else:
        raise AssertionError("rls_refresh_interval=0 should raise ValueError")


def check_zero_jitter_cholesky_failure_falls_back_after_one_attempt() -> None:
    manager = _manager("right_sketch_ridge", interval=1, jitter=0.0)
    _set_frame(manager, 0)

    def fail_cholesky(matrix: torch.Tensor):
        info = torch.ones(matrix.shape[:-2], device=matrix.device, dtype=torch.int32)
        return torch.zeros_like(matrix), info

    with mock.patch.object(torch.linalg, "cholesky_ex", side_effect=fail_cholesky) as cholesky_ex:
        scores = _scores(manager, torch.randn(8, 4))

    if cholesky_ex.call_count != 1:
        raise AssertionError(f"jitter=0 should attempt Cholesky once, got {cholesky_ex.call_count}")
    if manager.cached_rls_chol is not None or manager.cached_rls_inv is None:
        raise AssertionError("jitter=0 Cholesky failure did not use the pinv fallback")
    if not torch.isfinite(scores).all():
        raise AssertionError("pinv fallback returned non-finite scores")


def main() -> None:
    check_interval_one_matches_fresh()
    check_interval_three_refresh_counts()
    check_offset_interval_refresh_counts()
    check_reset_forces_refresh()
    check_device_dtype(torch.device("cpu"))
    if torch.cuda.is_available():
        check_device_dtype(torch.device("cuda"))
    check_backward_reuse_path()
    check_inverse_backend_matches_cholesky_solve()
    check_invalid_interval()
    check_zero_jitter_cholesky_failure_falls_back_after_one_attempt()
    print("RLS refresh interval smoke tests passed")


if __name__ == "__main__":
    main()
