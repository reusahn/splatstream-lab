# Training Efficiency Pilot Results

Dataset: Mip-NeRF 360 / Bonsai  
GPU: NVIDIA A100-SXM4-40GB  
gsplat commit: `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`

## Status

This is a **single-run pilot**. The numbers below are measured, but the apparent quality improvement of the 5k / densification-stop schedule must be repeated before it is treated as a stable result.

The external wall time of the first 7k run was 5045.9 s because it included one-time setup/JIT overhead. For schedule comparison, this document uses `train_loop_seconds`.

## Measured pilot

| Preset | Train loop | Gaussians | Checkpoint | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline_7k | 421.8 s | 944,291 | 212.53 MiB | 29.6619 | 0.924574 | 0.152724 |
| early_stop_5k | 299.7 s | 920,498 | 207.18 MiB | 29.5314 | 0.925716 | 0.156233 |
| densify_stop_3k_5k | 299.9 s | 839,074 | 188.85 MiB | 30.1942 | 0.930217 | 0.150745 |

## Preliminary interpretation

Relative to the 7k baseline, `early_stop_5k` reduced the measured training loop by **28.9%** while changing PSNR by **-0.13 dB**. This is a useful low-risk early-stop point, but LPIPS was slightly worse.

The strongest pilot point was `densify_stop_3k_5k`. Relative to the 7k baseline it used:

- **28.9% less training-loop time**
- **11.1% fewer Gaussians**
- **11.1% smaller checkpoint**

In this one run it also measured **+0.53 dB PSNR**, **+0.0056 SSIM**, and **-0.0020 LPIPS** relative to baseline.

That quality increase is encouraging but **not yet a publishable speed/quality claim**. It could be affected by stochastic training variation. Repeat runs with controlled seeds are required.

## Important negative / cautionary measurement

`densify_stop_3k_5k` did **not** render faster in this pilot. The recorded seconds-per-image increased from 0.003041 s to 0.003718 s. Do not claim render-speed improvement from the lower Gaussian count without repeated timing analysis.

The field called `peak_memory_bytes` in the JSON contains values around 1.3-1.6 and appears to be a GiB-scale trainer statistic rather than literal bytes. Do not publish a memory reduction claim until the source field/unit is verified.

## Next experiment

1. Repeat `baseline_7k` and `densify_stop_3k_5k` at least two more times with controlled/reported seeds.
2. Run the remaining schedule presets only after the repeat path is stable.
3. Add a second scene, preferably `garden`, to test whether the early densification stop generalizes to high-frequency outdoor content.
4. Use repeated mean/std for training time and quality in the final portfolio.
