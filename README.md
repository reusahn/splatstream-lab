[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reusahn/splatstream-lab/blob/main/notebooks/SplatStream_Lab_CUDA_Validation.ipynb)

# SplatStream Lab

**Compression, quality, and progressive-streaming experiments for 3D Gaussian Splatting**

SplatStream Lab is a research harness for studying how pruning, quantization, compact payload encoding, and progressive transmission affect the size and rendered quality of 3D Gaussian Splatting scenes.

The project separates portable systems validation from full GPU rendering so that compression logic, payload accounting, and experiment orchestration remain reproducible on CPU while real-scene evaluation is performed with CUDA.

## Verified full-scene CUDA result

A reproducible 7,000-step `gsplat` run on **Mip-NeRF 360 / Bonsai** was completed on an **NVIDIA A100-SXM4-40GB**, producing **947,033 Gaussians**. The baseline held-out result was **29.7705 dB PSNR, 0.927644 SSIM, and 0.150789 LPIPS**.

The trained checkpoint was then reused for a full-scene held-out CUDA rate-distortion sweep. Every compressed operating point was evaluated with the same Bonsai cameras and the full `gsplat` rasterizer.

| Preset | Gaussians | zlib payload | Reduction | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_f32 | 947,033 | 185.43 MiB | 1.00x | 29.7705 dB | 0.927644 | 0.150789 |
| prune25_q16 | 710,275 | 70.04 MiB | **2.65x** | **27.6234 dB** | **0.900674** | **0.165871** |
| prune50_q16 | 473,516 | 46.78 MiB | 3.96x | 23.4229 dB | 0.817833 | 0.236364 |
| prune50_q8 | 473,516 | 20.24 MiB | 9.16x | 17.2307 dB | 0.565262 | 0.556337 |
| prune75_q8 | 236,758 | 10.31 MiB | 17.98x | 17.1499 dB | 0.561705 | 0.563792 |

The conservative `prune25_q16` point reduces the entropy-coded experimental payload from **185.43 MiB to 70.04 MiB** while retaining **27.62 dB PSNR / 0.9007 SSIM / 0.1659 LPIPS**. More aggressive compression shows a clear quality cliff, especially at 8-bit quantization.

See [`docs/BONSAI_FULL_CUDA_RD.md`](docs/BONSAI_FULL_CUDA_RD.md) for methodology and interpretation. Machine-readable results are under [`results/bonsai_a100_full_rd/`](results/bonsai_a100_full_rd/). The earlier baseline-only validation is preserved in [`docs/BONSAI_A100_RESULTS.md`](docs/BONSAI_A100_RESULTS.md).

## Research questions

1. How much model size can be removed through view-independent importance pruning?
2. How much additional reduction comes from 16-bit or 8-bit parameter quantization?
3. How does compression affect rendered quality measured with PSNR, SSIM, and LPIPS?
4. Where does aggressive compression create a visible quality cliff?
5. How does payload size translate into transfer time across illustrative network conditions?
6. Can importance ordering support useful progressive refinement?
7. How do encoding cost and payload size scale with scene complexity?

## System overview

```text
3DGS PLY / trained checkpoint
          │
          ▼
  GaussianScene parameters
          │
          ├────────────── baseline held-out render
          │
          ▼
  importance scoring
          │
          ▼
       pruning
          │
          ▼
 parameter quantization
          │
          ▼
 compact payload encoding
          │
          ├────────────── payload bytes
          ├────────────── transfer-time estimate
          │
          ▼
 dequantized scene for evaluation
          │
          ▼
 full gsplat CUDA rendering
          │
          ▼
 PSNR / SSIM / LPIPS
          │
          ▼
 size-quality tradeoff
```

## Core components

- `splatstream/model.py` — Gaussian scene representation
- `splatstream/ply_io.py` — GraphDeco-style 3DGS PLY reader
- `splatstream/compress.py` — pruning, quantization, payload encoding, progressive ordering
- `splatstream/streaming.py` — transfer-time estimation
- `splatstream/cpu_renderer.py` — deterministic CPU reference renderer for portable validation
- `splatstream/metrics.py` — PSNR, SSIM, MAE
- `splatstream/real_ply_experiment.py` — real-PLY compression sanity-check pipeline
- `tools/full_cuda_rate_distortion_v2.py` — full-scene held-out CUDA rate-distortion evaluator
- `notebooks/SplatStream_Lab_CUDA_Validation.ipynb` — reproducible CUDA evaluation on Mip-NeRF 360 / Bonsai

## Quick start

```bash
python -m splatstream.experiment \
  --output results/demo \
  --gaussians 1200 \
  --width 192 \
  --height 192
```

Run tests:

```bash
pytest -q
```

## Real 3DGS PLY evaluation

Inspect a scene:

```bash
python -m splatstream.inspect_ply /path/to/point_cloud.ply
```

Run the portable compression sweep:

```bash
python -m splatstream.real_ply_experiment \
  /path/to/point_cloud.ply \
  --output results/real_scene \
  --max-gaussians 50000
```

The CPU renderer is intentionally a reference implementation for compression-system validation. It does not replace a full anisotropic, view-dependent 3DGS rasterizer.

## Reproducible CUDA evaluation

Open the Colab notebook from the badge above and select a GPU runtime. The notebook uses a pinned `gsplat` revision, downloads Mip-NeRF 360 / Bonsai, trains a 7,000-step baseline, exports the checkpoint to PLY, runs the portable sanity check, and then evaluates the full-scene compressed variants on held-out CUDA views.

A complete run produces two evidence bundles:

```text
splatstream_bonsai_evidence.zip
splatstream_full_cuda_rd_v2.zip
```

The full CUDA bundle includes machine-readable rate-distortion tables, environment metadata, plots, and held-out validation renders.

## Compression presets

The full CUDA sweep evaluates:

- baseline float32
- 25% importance pruning + 16-bit quantization
- 50% pruning + 16-bit quantization
- 50% pruning + 8-bit quantization
- 75% pruning + 8-bit quantization

Positions and log-scales are quantized directly. Quaternion components are quantized and renormalized. Opacity is quantized in probability space and converted back to logits for rasterization. SH coefficients remain in coefficient space. For 8-bit storage, per-channel min/max side information is included in payload accounting.

## Evaluation discipline

For real-scene results, record:

- dataset and source
- training/test split
- Gaussian count
- baseline and compressed payload size
- compression ratio
- GPU model and CUDA/PyTorch versions
- PSNR / SSIM / LPIPS
- transfer-time estimates under explicitly stated illustrative bandwidths

Compression ratio is treated as a rate-distortion result rather than a standalone number: model-size reduction is only meaningful together with rendered-quality impact. Render-time values from a single sweep are retained as measurements but are not used to claim speedup without repeated timing runs.
