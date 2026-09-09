# Bonsai Full-Scene CUDA Rate-Distortion Results

Measured on 2026-09-09 with the reproducible Colab pipeline in this repository.

## Environment

- Dataset: Mip-NeRF 360 / Bonsai
- Trainer/evaluator: pinned `gsplat` `simple_trainer.py`
- Training: 7,000 steps
- gsplat commit: `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Baseline scene: 947,033 Gaussians

## Full held-out CUDA sweep

Every compressed variant was evaluated on the same held-out Bonsai cameras with the full `gsplat` CUDA rasterizer and the same PSNR, SSIM, and LPIPS metric path. Quantized tensors were dequantized for rasterization. Payload byte counts were measured from the corresponding stored quantized representation, with zlib used as a generic entropy backend.

| Preset | Gaussians | zlib payload | Compression | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_f32 | 947,033 | 185.43 MiB | 1.00x | 29.7705 dB | 0.927644 | 0.150789 |
| prune25_q16 | 710,275 | 70.04 MiB | 2.65x | 27.6234 dB | 0.900674 | 0.165871 |
| prune50_q16 | 473,516 | 46.78 MiB | 3.96x | 23.4229 dB | 0.817833 | 0.236364 |
| prune50_q8 | 473,516 | 20.24 MiB | 9.16x | 17.2307 dB | 0.565262 | 0.556337 |
| prune75_q8 | 236,758 | 10.31 MiB | 17.98x | 17.1499 dB | 0.561705 | 0.563792 |

## Rate-distortion interpretation

The conservative `prune25_q16` point reduces the zlib-backed payload from 185.43 MiB to 70.04 MiB, a **2.65x reduction**, while retaining **27.62 dB PSNR, 0.9007 SSIM, and 0.1659 LPIPS** on the held-out CUDA evaluation.

Increasing compression to `prune50_q16` reaches **3.96x** reduction but costs 6.35 dB of PSNR relative to the baseline. The 8-bit operating points show a clear quality cliff: `prune50_q8` reaches **9.16x** reduction but falls to 17.23 dB PSNR and 0.5653 SSIM. Further pruning to `prune75_q8` increases reduction to **17.98x** with similarly low quality.

This means the useful operating region for this simple view-independent importance heuristic lies on the conservative side of the sweep. The experiment also demonstrates why compression ratio should never be reported without a matched rendered-quality measurement.

## Timing note

The artifact records per-image `ellipse_time` values from the pinned evaluator. The `prune25_q16` point measured slower than the baseline while more aggressive variants measured faster, so these single-run timings are not used here to claim rendering speedup. The primary result is the measured payload-versus-quality tradeoff.

## Reproducibility

Machine-readable copies of the measured table and environment are stored under [`results/bonsai_a100_full_rd/`](../results/bonsai_a100_full_rd/). The public Colab notebook and `tools/full_cuda_rate_distortion_v2.py` reproduce the evaluation path.
