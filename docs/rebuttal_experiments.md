# Rebuttal experiments (DiaStream)

Plan and run book for the rebuttal experiments. All runs use the mv_recon evaluation (7-Scenes + NRGBD,
CD / F1) and start from the baseline setting below. Results go under `eval_results/mv_recon/rebuttal/`.

## Baseline (ours)

`src/eval/mv_recon/run_final_early_attention.sh`, which runs `run_final_ablation.sh`:

| Item | Value |
|---|---|
| Token retention (IDC) | svd_leverage, heads concatenated (`--leverage_granularity layer`), head-wise key normalization, sketch dim 256, ridge 0 |
| Attention importance | `--leverage_attention_utility`, beta 0.5, decay 0.9, freeze after T=5 updates |
| Layer allocation (KABA) | `value_weighted_leverage_pr`, alpha = gamma = 0.7 |
| Budget | 60000 tokens over 24 global layers |
| Other | confidence gate on, projected key cache on, chunk 1, kf_every 2, full sequence, seed 42 |

With beta 0.5 the blend `(1-b)·lev/mean + b·util/mean` gives observed tokens `0.5·(lev+util)` and tokens with no
attention yet (the current frame) `lev = 0.5·(2·lev)`, which is the paper's "2x leverage for new tokens" rule.

`run_final_ablation.sh` reads these environment variables (defaults = the baseline values):

| Variable | Default | Meaning |
|---|---|---|
| `TOTAL_BUDGET` | 60000 | `--budget` |
| `BUDGET_FRAME_MULTIPLIER` | empty | `--budget_frame_multiplier` instead of `--budget` |
| `KF_EVERY` | 2 | keyframe stride |
| `MAX_FRAMES` | full | `--max_frames` (`full` = whole sequence, or an integer) |
| `OUTPUT_DIR` | `../eval_results/mv_recon/Rebuttal_beta0.5_wo_none_60000_FullSeq` | relative to `src/` |
| `EVAL_DEVICE` | cpu | `cuda` computes the metrics on the GPU (same values as cpu, about 2x faster) |
| `NUM_PROCESSES`, `MAIN_PROCESS_PORT` | 1, 29202 | accelerate |
| `EXTRA_ARGS` | empty | extra `launch.py` flags, appended last (so they override earlier ones) |

Every metric line now also reports `PeakMemGB` (peak allocated GPU memory during inference of that scene).

Wrappers live in `src/eval/mv_recon/rebuttal/`. Pick the GPU with `CUDA_VISIBLE_DEVICES`. If `ckpt/` is not
present in this repo, pass `EXTRA_ARGS="--weights /home/dongjae/InfiniteVGGT/ckpt/checkpoints.pth"`.

---

## Running everything and collecting results

```bash
cd src/eval/mv_recon/rebuttal
GPUS="0 2 3 5" CUDA_DETERMINISTIC=false bash run_queue.sh jobs_all.txt   # 31 runs, one per GPU at a time
python tools/collect_rebuttal_results.py      # from the repo root -> eval_results/mv_recon/rebuttal/results.json
```
`results.json` holds, per experiment / run / dataset, the per-scene metrics and means (CD = (Acc + Comp) / 2,
F1 at 5/10/25 cm, FPS, PeakMemGB, frames), whether the run finished, the strict-budget overage, the redundancy
medians (pr_analysis), and the queue status. It can be regenerated at any time; unfinished runs are marked
`"complete": false`.

Metrics use the CPU evaluation (`EVAL_DEVICE=cpu`, the default of the wrappers). The GPU evaluation is
exact but uses hash-grid radius search on the full point clouds; on full sequences (~25M points, the same room seen
500 times) the cells get so dense that it is far slower than the CPU KD-tree. oxford_eval avoids this by voxelizing
first (0.05 m), which our metric protocol does not do. CUDA determinism is off for these runs (`--no_cuda_deterministic`, 2.8 → 4.1 FPS on a 3090).

## 7-Scenes trajectory evaluation

Every experiment that reports reconstruction metrics is also evaluated on 7-Scenes camera poses with the same
method configuration (`eval/pose_evaluation/launch.py`, 518 px, frame stride = `KF_EVERY`, ATE / RPE with evo).
Trajectory runs always use the full sequence; `--pose_eval_prefix_frames 100` also reports the metrics of the first
100 input frames of the same run (streaming is causal, so this is the 100-frame result).

- `eval/pose_evaluation/run_pose_ablation.sh` / `run_pose_early_attention.sh`: same env variables as the mv_recon scripts.
- `rebuttal/pose_strict.sh`, `pose_stride.sh`, `pose_compactor.sh`: same configurations as `strict.sh`,
  `stride.sh`, `compactor.sh`; outputs in `eval_results/pose_evaluation/rebuttal/`.
- `rebuttal/jobs_pose.txt` (26 runs) with `run_queue_wait.sh` (each worker waits until its GPU is empty).
  Not run separately because they coincide at full sequence: compactor A ours = compB C7_ours = strict B60000_soft;
  compB C0_compactor = compactor_kaba.
- `results.json` → `trajectory_7scenes` (per-sequence ATE / RPE, means, `first_k_frames["100"]`, strict overage).

## KV cache storage in bytes

`--kv_bytes_log` writes per step the actual `nbytes` of K, V (with dtypes), the IDC state, the projected key cache,
other eviction auxiliaries and the camera-head cache to `<output>/<dataset>/kv_bytes/<scene>.csv`.

## Experiment 2: strict budget

**Issue.** The first-frame anchor (1041 tokens per layer) is never evicted. When KABA gives a layer less than
that, the anchor floor keeps all 1041, so the total cache exceeds the global budget.

**Change.** `--strict_budget`: reserve the anchor in every layer first and split only `B − 24·1041` by the KABA
weights (implemented with the allocator's existing `min_tokens` floor, `aggregator.py` before
`_calculate_dynamic_budgets`). The total is then at most B. B below 24·1041 = 24,984 is rejected.

Second strict variant, `--strict_budget_mode redistribute`: allocate the full B by KABA weights as usual, lift every
layer below the anchor to the anchor, and take the resulting overage from the other layers in proportion to their
KABA weights (repeated while that pushes more layers below the anchor; `eviction._redistribute_anchor_deficit`).
Low-weight layers keep only the anchor, the KABA shares of the others are scaled down to fit B. Default mode is
`reserve` (the anchor first, the rest by weights). Run: `bash strict_redistribute.sh 30000 60000 200000` and
`pose_strict_redistribute.sh`; jobs in `jobs_strict_redistribute.txt`. Test: `tools/test_strict_redistribute.py`.

**Run.**
```bash
cd src/eval/mv_recon/rebuttal && bash strict.sh            # B = 30000, 60000 × {soft, strict}
```
soft runs log the overage to `<output>/<dataset>/<scene>/anchor_budget_overage.csv`
(`global_overage_tokens` per step).

**Report.** CD / F1, actual total cache (mean / max, % of B), share of steps over budget, PeakMemGB.

Per-layer breakdown: `--log-anchor-budget-overage` also writes `anchor_budget_overage_layers.csv` (step, layer,
capacity, assigned budget, anchor tokens, anchor deficit, retained tokens). The soft runs were repeated with it,
inference only (`--eval_stage none`), in `strict_layers/B{60000,30000}_soft`; `results.json` summarises per layer
how often the budget falls below the anchor and how often a layer keeps only the anchor.

Smoke test (30 frames, B = 30K, soft): peak overage about 4.1K tokens (13.6% over the budget).

---

## Experiment 4: large stride

```bash
cd src/eval/mv_recon/rebuttal && bash stride.sh            # stride 20, 50, 100 × {60K, 30K, full cache}
```
Full cache = budget 1e9, so nothing is ever evicted (the attention-utility kernel still runs; it does not change
the attention output).

Frames per sequence:

| stride | 7-Scenes (stairs) | NRGBD |
|---|---|---|
| 20 | 50 (25) | 20–84 |
| 50 | 20 (10) | 8–34 |
| 100 | 10 (5) | 4–17 |

At stride 100 a 60K budget holds about 2.4 frames per layer, i.e. a large share of a 5–17 frame sequence, so
report the cache-to-sequence ratio as well.

### Other methods at the same strides (to run later)

Use the same frames (same `kf_every`, full sequence) and the same metric code.

- **InfiniteVGGT** (`/home/dongjae/InfiniteVGGT`): `src/eval/mv_recon/launch.py` hardcodes the datasets. In
  `datasets_all`, set `kf_every=<stride>`, set `ROOT` to `/home/dongjae/dataset/7scenes` and
  `/home/dongjae/dataset/neural_rgbd_data`, enable NRGBD, and leave `max_frames` unset. Run
  `src/eval/mv_recon/run.sh` from `src/` with `max_frames` removed. Budget: its default (1.2M); give it 60K too
  for a matched comparison (`total_budget` in `streamvggt/models/aggregator.py`).
- **OVGGT / STAC**: not on this server. Clone, then make the same three dataset edits (stride, roots, full
  sequence) in their mv_recon launcher; use their default budgets (OVGGT 200K, STAC 225K) and, if possible, 60K.
- Check that every method sees the same frame list (print the number of frames per scene).

---

## Experiment 5: leverage participation ratio (psi) counterexample

**Reviewer concern.** n identical keys have leverage 1/n each, so psi = n; n keys on n orthogonal directions also
give psi = n. psi alone cannot tell the two apart.

**Answer.** Agree that it is possible in theory (psi only measures how evenly leverage is spread; the two cases
differ in rank and in token similarity). Then show that the caches IDC actually keeps are far from the degenerate
case: tokens are not similar to each other and the key matrix is far from rank 1.

**(a) Synthetic counterexample.**
```bash
python tools/psi_counterexample.py --n 256 --d 1024
```
Example (n = 128, d = 512):

| set | psi/N | rank | effective rank | mean cosine |
|---|---|---|---|---|
| identical keys | 1 | 1 | 1 | 1 |
| orthogonal keys | 1 | 128 | 128 | 0 |
| near-identical keys (noise norm 0.3) | 1 | 128 | 1.95 | 0.92 |

**(b) Real caches.** `launch.py --redundancy_log [--redundancy_log_every 10]` writes
`<output>/<dataset>/redundancy/<scene>.csv`, one row per (step, layer, set), for the candidate set before eviction
and the retained set after it. Keys are represented as IDC scores them (post-RoPE, unit norm per head, heads
concatenated); anchor tokens are excluded.

Columns: `psi` (exact leverage on these features), `rank`, `erank` (exp of the entropy of the normalized squared
singular values), `stable_rank`, `mean_cos` (exact mean over pairs), `frac_cos_gt_{0.5,0.7,0.9}`, nearest-neighbour
cosine mean / p10 / p50 / p90, `n_tokens`, `head_shared`.

Conditions:

| Condition | Flags |
|---|---|
| IDC (baseline) | — |
| random | `--eviction_policy random --layer_budget_strategy uniform` (new policy: random subset shared across heads, seeded) |
| centroid (InfiniteVGGT-like) | `--eviction_policy mean --layer_budget_strategy uniform` |

Note: the cache stores one token set per layer (`_materialize_kv_by_keep_indices(head_shared=True)` uses head 0's
indices), so the `mean` policy's per-head choice is in practice head 0's choice; the log records that set.

```bash
cd src/eval/mv_recon/rebuttal && bash pr_analysis.sh       # IDC, random, mean
cd ../../../.. && python tools/analyze_cache_redundancy.py \
    --run IDC=eval_results/mv_recon/rebuttal/pr_analysis/IDC \
    --run random=eval_results/mv_recon/rebuttal/pr_analysis/random \
    --run mean=eval_results/mv_recon/rebuttal/pr_analysis/mean \
    --out eval_results/mv_recon/rebuttal/pr_analysis/summary
python tools/psi_counterexample.py --redundancy_csv 'eval_results/mv_recon/rebuttal/pr_analysis/IDC/*/redundancy/*.csv'
```
Sketch space: each row also has `space` = `full` (1024-d head-normalized concatenated keys) or `sketch` (the
right-sketched 256-d matrix Z IDC computes leverage and psi on; cosine on its normalized rows, ranks and psi on Z).
IDC only, at 100 frames (every 10 steps, `pr_sketch_frames100/`) and full sequence (every 25 steps,
`pr_sketch_framesfull/`); comparison in `pr_sketch_summary/`.

Local density: for tau in 0.7 / 0.8 / 0.9, each token's number of other tokens (same set) with cosine > tau;
per step/layer/set the mean, median, p90, max and the share of tokens with >= 1 and >= 10 such neighbours
(`dens{tau}_*`). On candidate rows also the density (within the candidate set) of the kept vs the evicted tokens
and the Spearman correlation between a token's leverage and its density (`spearman_lev_dens{tau}`). Runs:
`pr_density_frames100/`, `pr_density_framesfull/` (IDC; queue_extra), summary in `pr_density_summary/`.

Density heterogeneity (does N_i differ across tokens, as for clustered keys, or is it uniform, as for keys spread
evenly over independent directions?): per step/layer/set the std, CV, top-10% / bottom-10% mean ratio, max / min
ratio (+1 smoothed) and Gini of N_i (`dens{tau}_{std,cv,topbot,maxmin,gini}`), and the spread of each token's mean
cosine (`tokcos_*`), in both spaces. `--redundancy_dump_tokens` saves the per-token arrays (`<scene>_tokens.npz`).
`tools/analyze_density_heterogeneity.py` summarises them, bins tokens by density decile (mean relative leverage and
share kept per decile) and adds a balanced vs skewed synthetic reference with the same psi/N. Runs:
`pr_het_frames100/`, `pr_het_framesfull/` (jobs_het.txt), summary in `pr_het_summary/`.

Outputs: `summary.csv` (medians, p90, share of degenerate cases: erank < 2 or mean cosine > 0.9),
`per_layer_median.csv`, per-layer boxplots, and a psi/N vs effective rank scatter.

---

## Experiment 3: Compactor

Compactor (Chari & Van Durme 2025; official code github.com/vnchari/compactor-vllm) ported as
`--eviction_policy compactor` (`EvictionManager._compactor_scores`, `src/streamvggt/layers/eviction.py`):

| Part | Official code | Port here |
|---|---|---|
| attention | non-causal softmax column sum inside 128-token blocks, raw q·k | current-frame queries over all live keys (anchor + candidates, before eviction), softmax with 1/sqrt(D) |
| leverage | per KV head, pre-RoPE keys mean-centred, sketch 48, ridge 5e-3 | per head, post-RoPE keys mean-centred, exact (head dim 64 < sketch 256), ridge 5e-3 (`--compactor_ridge`) |
| combination | z(attn) + 0.5·z(lev), population std pooled over heads and tokens | same (`--leverage_compactor_lambda 0.5`) |
| selection | top-k over (token, head) pairs; protected first 16 / last 64 tokens | top-k per head, same count per head; the first-frame anchor is protected |
| cache | per-head token sets | per-head token sets (`_materialize_kv_by_keep_indices(head_shared=False)`) |

The compactor policy needs no per-token state, so it runs without the confidence gate and attention utility
(`launch.py` rejects the combination). With KABA, the layer statistic is the per-head leverage participation
ratio averaged over heads.

The six differences, as switches (defaults = DiaStream):

| # | Difference | DiaStream | Compactor | Switch |
|---|---|---|---|---|
| 1 | causal attention history | utility accumulated from later frames | current-frame queries, recomputed each eviction | `--leverage_attention_source causal_history / noncausal_current` |
| 2 | head-wise normalization | unit norm per head | raw keys | `HEADWISE_NORM=true/false` (IDC path), `--compactor_headwise_norm` (compactor) |
| 3 | concatenation | heads concatenated, one token set per layer | per-head leverage and token sets | `--eviction_policy svd_leverage` (layer) / `compactor` |
| 4 | freezing importance | frozen after T=5 | never frozen | `LEVERAGE_ATTENTION_FREEZE_UPDATES=5 / 0` (0 = never) |
| 5 | mean-normalization | lev/mean + util/mean, equal weights | z(util) + 0.5·z(lev) | `--leverage_score_norm mean / zscore`, `--leverage_compactor_lambda` |
| 6 | 2x leverage for new tokens | 2x | 1x | `--leverage_unobserved_lev_multiplier 2 / 1` (unset = legacy = 2x at beta 0.5) |

Process-wide switches live in `src/streamvggt/utils/ablation_config.py`; unit tests in
`tools/test_compactor_ablation.py` (compactor scores vs a numpy reference, per-head selection and cache
materialisation, blend modes, freeze 0).

```bash
cd src/eval/mv_recon/rebuttal
bash compactor.sh A        # full sequence: compactor_uniform, compactor_kaba, ours
bash compactor.sh B        # first 100 frames: C0..C7 and ours_wo_*
DRY_RUN=1 bash compactor.sh B   # print the commands only
```

Table B, cumulative (KABA throughout):

| Step | Adds | Config |
|---|---|---|
| C0 | Compactor | `C0_compactor` |
| C1 | (2) head-wise normalization | `C1_headnorm` |
| C2 | (3) concatenation (also switches to IDC's leverage: layer sketch 256, ridge 0, no centring) | `C2_concat` |
| C3 | (1) causal attention history (never frozen, 1x new tokens) | `C3_history` |
| C4 | (4) freezing T=5 | `C4_freeze` |
| C5 | (5) mean-normalization, equal weights | `C5_meannorm` |
| C6 | (6) 2x leverage for new tokens | `C6_double` |
| C7 | confidence gate (= ours) | `C7_ours` |

Table B is also run at full sequence (`MAX_FRAMES=full bash compactor.sh <config>`, outputs
`B_<config>_framesfull`); there C0 equals table A `compactor_kaba` and C7 equals strict `B60000_soft`, so 11 runs.
The full-sequence table B and the trajectory runs are queued together in `jobs_phase2.txt`.

Frame-block attention (closer to the official Compactor, whose score is a non-causal column sum inside its own
block): `--compactor_attention insert` (compactor policy, per head) and `--leverage_attention_source
noncausal_insert` (IDC path, head-averaged). When a frame enters, its queries attend to its own keys without a causal
mask; the column sum is stored with each token (kept aligned through eviction) and fixed afterwards. Runs:
`compactor_insert.sh` / `jobs_compactor_insert.txt` (A uniform/KABA, C1, C2; recon full + 100 frames, 7-Scenes
trajectory).

Leave-one-out from ours: `ours_wo_history`, `ours_wo_headnorm`, `ours_wo_freeze`, `ours_wo_meannorm`
(z-score with lambda 1, so only the normalisation changes), `ours_wo_double`. Removing (3) alone is not possible:
per-head token sets would need per-head confidence/attention state; C1 → C2 measures it instead.

## Attention peak per global layer (R2-Q3, AVGGT check)

AVGGT (arXiv 2512.02541, Sec. 3.1 / Fig. 2) shows qualitatively that early global layers of VGGT have weaker top
activated entries (head-averaged attention, patch tokens, top-50 entries of the whole matrix; layers 0-indexed 0-23).
`tools/attention_peak_analysis.py` measures this in StreamVGGT's causal global attention, on CPU, without touching
`src/`: the SDPA call of the global blocks is wrapped in-process (output returned unchanged) and A = softmax(qk^T/sqrt(D))
is recomputed from the same q, k; only statistics are kept. Eviction and KABA are disabled and guarded:
`Attention.eviction` / `EvictionManager.select` raise if called, the layer allocator must see zero scores and alpha 0
(uniform), and each step must have keys = step x 1041.

Setting: 7-Scenes, kf_every 20, first 5 frames, full cache (budget 1e9), fp32 CPU.

```bash
python tools/test_attention_peak.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=32 python tools/attention_peak_analysis.py --scene_index 0 --verify \
  --out eval_results/mv_recon/rebuttal/attn_peak          # ~2.5 min per sequence with --verify
R=eval_results/mv_recon/rebuttal
python tools/attention_peak_report.py --attn $R/attn_peak \
  --kaba_layers "$R/strict_layers/B60000_soft/7scenes/*/anchor_budget_overage_layers.csv" \
  --density $R/pr_density_summary/per_layer_median.csv --out $R/attn_peak/report
```

`--verify` adds two unpatched reference runs (eviction_policy mean and svd_leverage) and checks that all model
outputs are bit-identical to the patched run and that softmax(qk^T)v matches the SDPA output.
Outputs per sequence: `attn_peak_layers.csv` (layer x step x head x query type), `attn_peak_top50.csv` (AVGGT
statistic per step and over the union of steps = the full causal 5-view matrix), `key_norm.csv`, `meta.json`
(checks). Report: `attn_peak_by_layer.csv`, `kaba_compare.csv`, `kaba_spearman.csv`, `fig_attn_peak.png`.
