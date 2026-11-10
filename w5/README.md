# W5 profiling experiments

Ours only, `run_final_early_attention.sh` setting. Measures per sequence: 24-layer KV budget excess,
peak GPU memory, per-frame latency / FPS, component latency, End-to-End vs Inference-only (without IDC/KABA) time.

## On a new machine
1. Copy the whole repository folder (including `ckpt/checkpoints.pth` if you have it; otherwise it is downloaded).
2. Edit **`w5/config.sh`** only: `W5_DATA_ROOT` (and per-dataset folder names if they differ), `GPU`, `W5_CONDA_ENV`.
3. Run:
   ```bash
   bash w5/run_all.sh
   ```
   It runs `w5/setup_env.sh` automatically when needed (creates the conda env from `w5/requirements.txt`,
   rebuilds attn-cuda for the local GPU — needs `nvcc` from CUDA 12.x — and downloads the checkpoint),
   then `w5/preflight.py`, then the full matrix, then `tools/summarize_w5_profile.py`.

Check only (no experiments): `bash -c 'source w5/config.sh && w5_activate_env && python w5/preflight.py'`

## Expected dataset layout under `W5_DATA_ROOT`
| dataset | folder | must contain |
|---|---|---|
| mv_recon 7scenes | `7scenes/` | `<scene>/seq-XX/...` |
| mv_recon NRGBD | `neural_rgbd_data/` | `<scene>/images/` |
| mv_recon ETH3D | `eth3d/` | ETH3D scenes |
| pose TUM | `tum/` | `<seq>/rgb_90/`, `<seq>/groundtruth_90.txt` |
| pose ScanNet | `scannetv2/` | `<seq>/color_90/`, `<seq>/pose_90.txt` |

Run a subset, e.g. only what exists: `BENCHES=mv_recon MV_DATASETS=7scenes,NRGBD bash w5/run_all.sh`.

## Matrix (`src/eval/run_w5_all.sh`)
| run | purpose | evaluation |
|---|---|---|
| R1 rep0 | clean timing/memory/KV (`--w5_profile`) | yes |
| R1 rep1, rep2 | repeats (median reported) | skipped |
| R2 | `--perf_trace` component ms | skipped |
| R3 | `--profile_eviction` IDC breakdown (%) | skipped |
| sweep (`SWEEP=true`) | budgets 30k / 120k, R1 | skipped |

Outputs: `eval_results/w5/{mv_recon,pose_evaluation/<ds>}/ours_b<budget>_<R>_rep<k>/`, summary in `eval_results/w5/summary/`.
Single runs: `RUN_MODE=R1 REP=0 bash src/eval/mv_recon/run_w5_profile.sh` (see header of each script for env vars).

Notes: use a GPU no other job is using (preflight warns otherwise). `torch.use_deterministic_algorithms`
is disabled for profiling runs (it costs ~33% FPS); inputs stay in host RAM and are streamed per frame.
