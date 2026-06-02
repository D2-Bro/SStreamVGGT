# SVD / Leverage Eviction Latency Audit

Date: 2026-05-28

## Conclusion

The current code has two different approximate leverage paths, and they should not be treated as equivalent.

- `right_sketch` is a dense random right-projection approximation. It avoids QR/SVD on `[N, D]`, but it still pays a dense projection cost `O(N D r)` before QR on `[N, r]`. For `N ~= 9000`, `D = 1024`, and large `r`, this can easily become the bottleneck.
- `drineas_srht` is closer to a true two-stage Drineas-style approximate leverage pipeline: left SRHT to `r1`, QR on `[r1, D]`, triangular solve into a right JL dimension `r2`, then `mat @ c` to estimate row norms. It avoids full QR/SVD on `[N, D]`, but the final dense GEMM is still `O(N D r2)`, so latency depends strongly on `r2`.

For your target settings, `drineas_srht` is the path that is intended to reduce wall-clock latency. However, for a layer feature dimension `D = 1024`, the requested `r1 = 384/512/768` is smaller than `D`. The current implementation requires the QR `R` from the left sketch to be square, so those settings trigger the SVD fallback instead of the fast QR/triangular-solve branch. `right_sketch` is not a fast two-stage leverage-score method; it is closer to dense feature-dimension reduction by `mat @ omega`.

## Files and Functions Inspected

- `src/streamvggt/layers/eviction.py`
  - `EvictionManager.select`
  - `_qr_leverage_scores_from_matrix`
  - `_drineas_srht_leverage_scores_from_matrix`
  - `_drineas_svd_fallback`
  - `compute_svd_leverage_scores`
  - `_svd_leverage_scores`
  - `_layer_svd_leverage_scores`
  - `_layer_svd_leverage_scores_head_mean`
  - `_layer_svd_leverage_scores_sketched`
  - `_get_leverage_right_sketch`
  - `_get_left_srht_factors`
- `src/streamvggt/layers/attention.py`
  - `Attention.eviction`
  - `Attention.forward`
- `src/streamvggt/layers/block.py`
- `src/streamvggt/models/aggregator.py`
- `src/streamvggt/models/streamvggt.py`
- `src/eval/mv_recon/launch.py`
- `src/eval/video_depth/launch.py`
- `run_inference.py`
- `tools/validate_leverage_approx.py`

## Current Code Paths

| Path | Classification | Main Operations | Dominant Complexity | Latency Risk |
|---|---:|---|---|---|
| `exact_qr` in `_qr_leverage_scores_from_matrix` | A/E exact QR leverage, not SVD | QR on `mat` with shape `[N, D]`; score from `Q` row norms | `O(N D^2)` for tall-skinny QR when `N > D` | Avoids full SVD but still expensive at `9000 x 1024` |
| `right_sketch` in `_qr_leverage_scores_from_matrix` | C dense random projection approximation | cached `omega` `[D, r]`; `mat @ omega`; QR on `[N, r]`; score from `Q` | `O(N D r) + O(N r^2)` | The dense projection can dominate for large `r`; not a two-stage fast leverage method |
| layer `right_sketch` in `_layer_svd_leverage_scores_sketched` | C dense random projection approximation | cached `omega` `[H*D, r]`; `einsum` equivalent to `[N, H*D] @ omega`; QR on `[N, r]` | `O(N H D r) + O(N r^2)` | Avoids materializing `[N, H*D]`, but not the dense projection cost |
| `drineas_srht` fast branch in `_drineas_srht_leverage_scores_from_matrix` | D two-stage approximate leverage method | left SRHT to `[r1, D]`; QR on `[r1, D]`; solve `R C = Pi2`; final `mat @ C`; row norm scores | `O(N D log N)` FWHT-ish work + `O(r1 D^2)` QR when `r1 >= D` + `O(D^2 r2)` solve + `O(N D r2)` final GEMM | Only used when QR produces square non-singular `R`; for `D=1024`, `r1 < 1024` does not use this branch |
| `drineas_srht` SVD fallback | B truncated-ish SVD on sketched matrix | SVD on `b_mat` `[r1, D]`; `mat @ V`; optional JL | `O(min(r1,D)^2 max(r1,D)) + O(N D min(r1,D)) + O(N min(r1,D) r2)` | This is what the proposed `r1=384/512/768, D=1024` settings currently hit; it can replace exact QR/SVD with another large dense projection |
| `head_mean` projection | E deterministic low-dimensional feature QR | mean-pool head channels to `[N, H*m]`; QR; score from `Q` | `O(N D)` feature build + `O(N (H m)^2)` QR | Fast if `H*m` is small, but it changes the feature space substantially |

## Expensive Operations Found

- Full SVD on `[N, D]`: not in the eviction path. Full SVD exists in offline analysis tools and the new benchmark baseline.
- QR on `[N, D]`: yes, `exact_qr` uses `torch.linalg.qr(mat, mode="reduced")`.
- Dense `N x D` by `D x r`: yes, `right_sketch` uses `leverage_matrix = mat @ omega` in `eviction.py`; layer mode uses `torch.einsum("bhnd,hds->bns", ...)`, the same dense projection without materializing `[N, H*D]`.
- Dense `r1 x N` by `N x D`: not explicitly constructed in `drineas_srht`; left SRHT is applied by signs + FWHT + sampled rows.
- Pseudoinverse of a large matrix: not found in the active eviction path; `torch.linalg.solve_triangular` is used instead.
- Pairwise `N x N` similarity: not in leverage scoring; pairwise-like matching exists in merge/analysis code, not this scoring path.
- Repeated random sketch allocation: right sketches and SRHT factors are cached by `(device, dimensions, seed)`. They are reused by each `EvictionManager` instance. Managers are cached in `Attention._eviction_managers` by policy/config, so normal repeated steps should reuse them.
- CPU-GPU synchronization: profiling/debug paths use `torch.cuda.synchronize()` and `.item()` checks. These are opt-in except some protection/debug/control-flow `.item()` calls. `drineas_srht` has `.item()` in the ill-conditioning check and fallback active-rank check.
- Printing inside loop: only when eviction debug/profile is enabled.

## Sketch Caching Notes

- `_get_leverage_right_sketch` caches by stringified device, `embed_dim`, `sketch_dim`, and seed. It creates CPU `float32` random matrices, then transfers to the target device.
- `_get_left_srht_factors` caches signs and sampled rows by device, token count, padded token count, `r1`, seed, and label.
- Dtype policy is deliberate but fixed: leverage scoring converts inputs to `float32`, and sketches are `float32`.
- Multi-layer/multi-head safety is mostly OK because each `Attention` module has its own `EvictionManager` cache, and sketch keys include the feature dimensions. Compatible layers with identical dimensions may reuse equivalent sketches inside a module/config; incompatible shapes produce distinct cache keys.

## What Changed

- Added `--profile_eviction` / `--profile-eviction` to `src/eval/mv_recon/launch.py`, `src/eval/video_depth/launch.py`, and `run_inference.py`. It reuses the existing eviction debug/profile plumbing and is off by default.
- Split profile timing for the dense right-projection path into:
  - `sketch_matrix_retrieval`
  - `projection_matmul`
  - `qr`
  - `scoring`
  - `candidate_matrix_preparation`
  - `total`
- Added high-level per-eviction output from `Attention.eviction` for `manager_select`, `metadata_index_update`, and `total_eviction`.
- Added `tools/benchmark_leverage_latency.py` for synthetic latency comparison on `N=9000, D=1024`.

## Benchmark Command

```bash
cd /home/dongjae/SStreamVGGT
PYTHONPATH=src python tools/benchmark_leverage_latency.py --n 9000 --d 1024 --device cuda --dtype float32 --warmup 2 --repeats 5 --profile
```

This reports mean latency, median latency, peak CUDA memory, speedup over the full exact SVD baseline, and whether each approximate path is actually faster.

## Practical Recommendation

Use `drineas_srht` for the `r1/r2` sweep you care about, but interpret the current results carefully: with layer feature dimension `D=1024`, the current code falls back for all proposed `r1 < D` settings. The benchmark will show whether that fallback is still faster, but it is not the intended fast QR/triangular-solve branch.

- aggressive: `--leverage_approx_method drineas_srht --leverage_left_sketch_dim 384 --leverage_right_jl_dim 64`
- balanced: `--leverage_approx_method drineas_srht --leverage_left_sketch_dim 512 --leverage_right_jl_dim 128`
- stable: `--leverage_approx_method drineas_srht --leverage_left_sketch_dim 768 --leverage_right_jl_dim 128` or `256`

Do not interpret `right_sketch` with large `--leverage_sketch_dim` as the same latency story. It is explicitly the dense `mat @ omega` route, and its main bottleneck is `O(N D r)` projection plus QR on `[N, r]`.
