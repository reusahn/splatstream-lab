# 3DGS Training Efficiency Study

This study extends SplatStream Lab from **post-training compression** into the second major systems question in Netflix JR40251: **how much training time can be removed before held-out novel-view quality degrades materially?**

## Research question

> How aggressively can a 3D Gaussian Splatting training schedule be shortened or simplified while preserving a useful held-out quality operating point?

The first experiment is deliberately conservative. It does **not** claim a new training algorithm. It establishes a controlled baseline for training-budget and densification-schedule trade-offs using the same scene family, metric stack, and pinned gsplat revision as the verified SplatStream compression experiment.

## Controlled variables

- Dataset: Mip-NeRF 360 / Bonsai
- Renderer/trainer: pinned `gsplat` revision `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`
- Held-out split: gsplat's existing Bonsai validation split
- Metrics: PSNR, SSIM, LPIPS
- Hardware: use one GPU model/runtime for the complete sweep
- Initialization: same SfM-based initialization
- Data factor: 2

## Sweep

| Preset | Max steps | Densification stop | Refine cadence | SH interval | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_7k` | 7000 | 7000 | 100 | 1000 | Verified-budget reference |
| `early_stop_5k` | 5000 | 5000 | 100 | 1000 | Pure training-budget reduction |
| `early_stop_3p5k` | 3500 | 3500 | 100 | 1000 | Aggressive early stop |
| `densify_stop_3k_5k` | 5000 | 3000 | 100 | 1000 | Stop scene growth before optimization ends |
| `sparse_refine_5k` | 5000 | 4000 | 200 | 1000 | Fewer densification/pruning events |
| `densify_stop_3k_shfast_5k` | 5000 | 3000 | 100 | 500 | Earlier full SH capacity under shorter budget |

The default gsplat strategy periodically duplicates/splits high-gradient Gaussians and prunes low-opacity Gaussians. The pinned implementation exposes `refine_stop_iter` and `refine_every`, so these presets alter the **schedule**, not the underlying rasterizer or loss.

## Measurements

For every preset record:

- external wall-clock training/evaluation time
- trainer-reported loop time when available
- Gaussian count
- checkpoint bytes
- peak GPU memory when available
- held-out PSNR
- held-out SSIM
- held-out LPIPS

The first decision plot should be:

- x-axis: training wall time
- y-axis: PSNR
- marker size or annotation: Gaussian count

Then repeat for SSIM and LPIPS.

## Decision rule

A shorter configuration is interesting only if the reduction in training time is accompanied by a tolerable quality penalty. Do not report a single percentage as a success without the paired quality metrics.

A useful portfolio statement would eventually have the form:

> Reduced 3DGS training wall time by **X%** on Bonsai while retaining **Y dB PSNR / Z SSIM / W LPIPS**, with **N** final Gaussians.

All placeholders must be replaced by measured results.

## Run

From the SplatStream Lab repository on a CUDA environment where the pinned gsplat checkout and Bonsai data already exist:

```bash
python tools/run_training_efficiency_sweep.py \
  --gsplat-dir /content/gsplat \
  --data-dir /content/gsplat/examples/data/360_v2/bonsai
```

Inspect commands without executing:

```bash
python tools/run_training_efficiency_sweep.py --dry-run
```

Run a smaller subset first:

```bash
python tools/run_training_efficiency_sweep.py \
  --only baseline_7k early_stop_5k densify_stop_3k_5k
```

## Next phase

After the single-scene study is stable:

1. repeat the best 2-3 schedules with multiple seeds/runs to avoid single-timing claims;
2. validate on at least one high-frequency outdoor scene such as Garden;
3. compare the selected training-efficient checkpoint against the existing SplatStream compression pipeline;
4. characterize the combined system: **training time -> model size -> rendered quality -> delivery rate**.

That final combined experiment is the strongest direct mapping to the Netflix internship because it joins the two open problems named in the posting: training/encoding efficiency and compression efficiency.
