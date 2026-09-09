# Bonsai A100 CUDA Results

Measured on 2026-09-08 with the reproducible Colab pipeline in this repository.

## Environment

- Dataset: Mip-NeRF 360 / Bonsai
- Trainer: `gsplat` `simple_trainer.py`, 7,000 steps
- gsplat commit: `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`
- GPU: NVIDIA A100-SXM4-40GB
- Compute capability: 8.0
- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8

## Full gsplat validation

The 7,000-step run produced 941,481 Gaussians and a 222,190,993-byte exported PLY (211.90 MiB).

| Metric | Measured value |
| --- | ---: |
| PSNR | 29.7369 dB |
| SSIM | 0.927614 |
| LPIPS | 0.151824 |
| Validation render time | 0.00303 s/image |
| Gaussian count | 941,481 |
| Peak CUDA memory recorded by trainer | 1.5814 GiB |
| Training-loop elapsed time recorded by trainer | 419.39 s |
| Exported PLY | 222,190,993 bytes / 211.90 MiB |

`ellipse_time` is the field name used by the pinned gsplat trainer for `time.time() - global_tic` at the final checkpoint. Because CUDA kernels are JIT compiled on first use in this workflow, that number should be interpreted as the measured training-loop elapsed time for this run, not as a hardware-independent benchmark.

## Portable compression sanity sweep

The compression harness was then run on a 10,000-Gaussian subset of the real exported PLY. These results validate parsing, pruning, quantization, payload encoding, and the portable CPU reference path.

| Preset | Gaussians | Payload | Reduction vs fp32 | CPU-reference PSNR | CPU-reference SSIM | Encode time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_fp32 | 10,000 | 2,042,613 B | 1.00x | inf | 1.000000 | 112.37 ms |
| prune25_q16 | 7,500 | 765,995 B | 2.67x | 57.21 dB | 0.999842 | 41.85 ms |
| prune50_q16 | 5,000 | 512,716 B | 3.98x | 47.93 dB | 0.998584 | 27.98 ms |
| prune50_q8 | 5,000 | 241,627 B | 8.45x | 42.48 dB | 0.995892 | 15.50 ms |

## Interpretation

The real-scene CUDA run verifies that the pipeline can train and evaluate a standard 3DGS scene end to end, export a conventional SH/opacity/scale/rotation PLY, and feed that representation into the compression harness.

The 8.45x figure above is **not** a full-scene CUDA rate-distortion result. It is the measured payload reduction on the 10,000-Gaussian portable sanity subset, and its PSNR/SSIM values come from the intentionally simplified CPU DC-only reference renderer. The next experiment should apply the same compressed variants to the full 941,481-Gaussian scene and evaluate them on the held-out gsplat CUDA views.
