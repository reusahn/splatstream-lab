# NeuralScene Bench

## Goal

Compare two neural scene representation families under one evaluation framework: Nerfstudio `splatfacto` for 3D Gaussian Splatting and `nerfacto` for a hash-grid NeRF pipeline.

The first pilot uses Mip-NeRF 360 / Bonsai. Garden is the planned second scene only after the Bonsai workflow is stable.

## Why this design

The benchmark uses one framework, one COLMAP parser, one train/eval split policy, and one metric implementation so the representation comparison is less confounded by incompatible evaluation code.

This is a representation benchmark, not a claim that one method is universally superior.

## Controlled dataset protocol

- Dataset: Mip-NeRF 360 / Bonsai
- Input layout: original Mip-NeRF 360 COLMAP data
- Image scale: 2x downsampled images
- COLMAP path: `sparse/0`
- Eval mode: interval
- Eval interval: 8
- Metrics: PSNR, SSIM, LPIPS from `ns-eval`
- GPU: one consistent NVIDIA GPU runtime for both methods

Nerfstudio's COLMAP parser explicitly supports Mip-NeRF 360 style datasets and interval evaluation. The same parser configuration must be used for both methods.

## Methods

### Splatfacto

Nerfstudio Gaussian Splatting implementation built on gsplat.

### Nerfacto

Nerfstudio's real-world NeRF pipeline. The pilot may use the PyTorch hash-grid implementation if tiny-cuda-nn is unavailable in the Colab Python runtime. If so, training-time results must be labeled as implementation-specific rather than an absolute representation-speed comparison.

## Pilot decision rule

Start with Bonsai only. Verify that both methods train and that `ns-eval` writes PSNR, SSIM, and LPIPS. Do not launch Garden until the full Bonsai artifact set is saved outside the ephemeral Colab runtime.

Record for each method:

- exact Nerfstudio commit/version
- Python / PyTorch / CUDA / GPU
- training wall time
- max training iterations
- checkpoint/model directory bytes
- peak GPU memory when available
- PSNR
- SSIM
- LPIPS

## Interpretation rules

1. A matched iteration count is not the same as matched compute or matched convergence.
2. If methods use different computational backends, training-time differences are implementation-specific.
3. Quality comparisons are meaningful only when dataset parsing, split, image scale, and evaluation metrics are held constant.
4. Do not compare checkpoint bytes as if they were production streaming payloads.
5. Report negative results. A slower or lower-quality method can still reveal an important quality, memory, or representation trade-off.

## Planned phases

Phase 1: environment and parser validation on Bonsai.

Phase 2: matched-protocol Splatfacto and Nerfacto pilot.

Phase 3: evaluate and package results.

Phase 4: repeat the selected operating point on Garden if compute and runtime stability allow.
