# NeuralScene Bench Results

## Scope

Controlled comparison of Splatfacto and Nerfacto on two Mip-NeRF 360 scenes using the same pinned Nerfstudio environment, COLMAP parser route, held-out split, 2x image downscale, and `ns-eval` evaluation pipeline.

The nominal training budget is 5,000 iterations for both methods. This matches optimizer step count only. It does not imply matched compute, convergence, model capacity, memory footprint, or wall-clock cost.

Nerfacto uses the tiny-cuda-nn implementation with camera optimization explicitly disabled. Splatfacto uses gsplat.

## Held-out results

| Scene | Method | Backend | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ns-eval FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bonsai | Splatfacto | gsplat | **27.93796** | **0.89582** | **0.17753** | 0.76850 |
| Bonsai | Nerfacto | tiny-cuda-nn | 25.79558 | 0.80945 | 0.24965 | **0.89980** |
| Garden | Splatfacto | gsplat | **24.00065** | **0.67932** | **0.32143** | **0.26987** |
| Garden | Nerfacto | tiny-cuda-nn | 22.33591 | 0.50766 | 0.52352 | 0.08501 |

## Observations

Splatfacto achieved higher held-out reconstruction quality on both scenes under this matched nominal 5K-iteration protocol.

On Bonsai, Splatfacto improved PSNR by approximately 2.14 dB relative to Nerfacto and reduced LPIPS by approximately 28.9%.

On Garden, Splatfacto improved PSNR by approximately 1.66 dB and reduced LPIPS by approximately 38.6%.

Observed evaluator throughput was scene dependent. Nerfacto had slightly higher `ns-eval` FPS on Bonsai, while Splatfacto had substantially higher `ns-eval` FPS on Garden.

## Interpretation boundary

These results should not be described as evidence that Gaussian Splatting is universally better or faster than NeRF.

`ns-eval` FPS is evaluator and implementation specific. It is not a pure renderer or rasterizer benchmark.

A matched iteration count is not a matched compute or convergence comparison. Nerfacto is commonly trained with longer schedules, so this experiment is best interpreted as a fixed nominal optimization-budget comparison.

Checkpoint directories contain training state and should not be interpreted as deployable streaming payload sizes.

The current study covers two scenes and uses one completed run per method and scene. It is descriptive rather than statistical.

## Reproducibility notes

- Dataset: Mip-NeRF 360 Bonsai and Garden
- Image scale: explicit 2x downscale through the Nerfstudio COLMAP parser
- Split: interval mode, evaluation interval 8
- Camera optimizer: off for Nerfacto to match Splatfacto camera treatment
- Nerfacto backend: tiny-cuda-nn
- Splatfacto backend: gsplat
- GPU used in the benchmark runtime: NVIDIA A100-SXM4-40GB
- Python environment: isolated Python 3.11 environment
- PyTorch: 2.7.1 with CUDA 12.8 build

Full Colab artifacts, final checkpoints, held-out renders, configs, logs, and JSON metrics were archived separately for both Bonsai and Garden.