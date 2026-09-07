[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reusahn/splatstream-lab/blob/main/notebooks/SplatStream_Lab_CUDA_Validation.ipynb)

# SplatStream Lab

**Compression, quality, and progressive-streaming experiments for 3D Gaussian Splatting**

SplatStream Lab is a research harness for studying how pruning, quantization, compact payload encoding, and progressive transmission affect the size and rendered quality of 3D Gaussian Splatting scenes.

The project separates portable systems validation from full GPU rendering so that compression logic, payload accounting, and experiment orchestration remain reproducible on CPU while real-scene evaluation is performed with CUDA.

## Research questions

1. How much model size can be removed through view-independent importance pruning?
2. How much additional reduction comes from 16-bit or 8-bit parameter quantization?
3. How does compression affect rendered quality measured with PSNR and SSIM?
4. Where does aggressive compression create a visible quality cliff?
5. How does payload size translate into transfer time across illustrative network conditions?
6. Can importance ordering support useful progressive refinement?
7. How do encoding cost and payload size scale with scene complexity?

## System overview

```text
3DGS PLY / synthetic scene
          │
          ▼
  GaussianScene model
          │
          ├────────────── baseline render
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
          ├────────────── payload bytes / encode time
          ├────────────── progressive prefixes
          │
          ▼
 reconstructed scene
          │
          ▼
       rendering
          │
          ▼
 PSNR / SSIM / MAE
          │
          ▼
 size-quality-streaming tradeoff
```

## Core components

- `splatstream/model.py` — Gaussian scene representation
- `splatstream/ply_io.py` — GraphDeco-style 3DGS PLY reader
- `splatstream/compress.py` — pruning, quantization, payload encoding, progressive ordering
- `splatstream/streaming.py` — transfer-time estimation
- `splatstream/cpu_renderer.py` — deterministic CPU reference renderer for portable validation
- `splatstream/metrics.py` — PSNR, SSIM, MAE
- `splatstream/real_ply_experiment.py` — real-PLY compression sanity-check pipeline
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

Open the Colab notebook from the badge above and select a GPU runtime. The notebook uses a pinned `gsplat` revision, downloads Mip-NeRF 360 / Bonsai, trains a 7,000-step baseline, exports the checkpoint to PLY, and packages the measured artifacts required to reproduce the run.

A successful run produces:

```text
splatstream_bonsai_evidence.zip
├── environment.json
├── splatstream_bonsai_7k.log
├── checkpoint_inventory.txt
├── ply_inventory.txt
├── gsplat_results/
└── portable_sanity/
```

## Compression presets

The reference sweep evaluates:

- baseline float32
- 25% importance pruning + 16-bit quantization
- 50% pruning + 16-bit quantization
- 50% pruning + 8-bit quantization
- 75% pruning + 8-bit quantization

The exact operating points depend on scene content and renderer. Reported measurements should always include the corresponding quality metrics and evaluation environment.

## Evaluation discipline

For real-scene results, record:

- dataset and source
- training/test split
- Gaussian count
- baseline PLY size
- compressed payload size
- compression ratio
- encoding time
- GPU model and CUDA/PyTorch versions
- PSNR / SSIM and, when available, LPIPS
- transfer-time estimates under explicitly stated illustrative bandwidths

Compression ratio is treated as a rate-distortion result rather than a standalone number: model-size reduction is only meaningful together with rendered-quality impact.
