# Cross-Scene Training Efficiency Results

This document compares the selected 3D Gaussian Splatting training schedules on two Mip-NeRF 360 scenes: Bonsai and Garden. All experiments used the same pinned gsplat revision, data factor 2, held-out evaluation procedure, and NVIDIA A100-SXM4-40GB runtime class.

## Main finding

The Bonsai sweep suggested that stopping densification at 3K steps, training to 5K steps, and progressing SH degree every 500 steps could improve the training-efficiency operating point. On Garden, the same schedule still reduced training cost and representation size substantially, but it no longer improved held-out quality.

The result is therefore scene dependent. The schedule generalizes as a systems-efficiency trade-off, not as a universal quality improvement.

## Garden validation

| Preset | Train loop | Peak CUDA mem | Gaussians | Checkpoint | PSNR | SSIM | LPIPS | Render s/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_7k | 999.3 s | 4.053 GiB | 2,628,056 | 591.49 MiB | 25.0484 | 0.728419 | 0.288914 | 0.007572 |
| densify_stop_3k_shfast_5k | 699.5 s | 2.554 GiB | 1,325,117 | 298.24 MiB | 24.5905 | 0.687846 | 0.353648 | 0.006471 |
| sparse_refine_5k | 692.4 s | 2.102 GiB | 935,554 | 210.57 MiB | 24.3232 | 0.675055 | 0.374987 | 0.004999 |

Relative to the Garden 7K baseline, `densify_stop_3k_shfast_5k` produced:

- 30.0% lower measured training-loop time
- 49.6% fewer Gaussians
- 49.6% smaller checkpoint
- 37.0% lower peak CUDA memory
- 14.5% lower measured seconds per rendered image
- 0.458 dB lower PSNR
- 0.0406 lower SSIM
- 0.0647 higher LPIPS

Relative to the Garden 7K baseline, `sparse_refine_5k` produced:

- 30.7% lower measured training-loop time
- 64.4% fewer Gaussians
- 64.4% smaller checkpoint
- 48.1% lower peak CUDA memory
- 34.0% lower measured seconds per rendered image
- 0.725 dB lower PSNR
- 0.0534 lower SSIM
- 0.0861 higher LPIPS

## Cross-scene interpretation

### Bonsai

The strongest Bonsai schedule was `densify_stop_3k_shfast_5k`. Compared with the 7K baseline, the single run reduced training-loop time by about 29%, reduced representation size by about 12%, and measured slightly better PSNR, SSIM, and LPIPS.

### Garden

The same schedule reduced training-loop time by 30%, cut the final representation almost in half, lowered peak CUDA memory, and reduced measured render time per image. However, the quality metrics degraded, especially SSIM and LPIPS.

### What can be claimed

The supported conclusion is not that the shortened schedule universally improves quality. A more defensible conclusion is:

> A shorter 5K schedule with early densification stop substantially reduced training cost and representation size on both Bonsai and Garden. The quality effect was scene dependent: it improved the single Bonsai run but incurred a measurable quality penalty on Garden.

This is useful for a streaming-oriented research portfolio because it exposes an explicit rate, compute, and quality trade-off rather than presenting one configuration as universally superior.

## Two useful operating points

`densify_stop_3k_shfast_5k` is the current balanced candidate. It preserves more quality than the sparse schedule while delivering meaningful reductions in training time, memory, and representation size.

`sparse_refine_5k` is the compact candidate. It reaches a much smaller representation and faster Garden rendering, but with a larger perceptual-quality penalty.

## Cautions

These are single runs per configuration. The first baseline external wall time in each fresh runtime includes one-time setup or CUDA JIT overhead, so comparisons use trainer-reported loop time rather than raw external wall time.

The pinned trainer uses a fixed nominal seed for single-GPU runs. Repeat measurements are still useful for timing variance and CUDA nondeterminism, but a third scene may provide more portfolio value than repeated identical runs if compute time is limited.

## Recommended next step

For the job portfolio, the most useful next experiment is to connect the selected training-efficient checkpoint to the existing SplatStream compression pipeline. That creates an end-to-end systems story:

`training schedule -> Gaussian count -> checkpoint / payload size -> rendered quality -> delivery-rate estimate`

A third-scene validation can then be added if time permits. The project should avoid claiming a universal quality improvement until broader validation supports it.
