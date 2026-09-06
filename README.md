[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reusahn/splatstream-lab/blob/main/notebooks/SplatStream_Lab_CUDA_Validation.ipynb)

# SplatStream Lab

**Compression, quality, and progressive-streaming experiments for 3D Gaussian Splatting**

This project was designed around a practical systems question:

> What combination of pruning, quantization, representation choices, and progressive transmission can reduce a Gaussian Splatting scene enough for streaming while preserving rendered quality?

![Portable compression preview](docs/portable_rate_distortion_preview.jpg)

The project deliberately separates **portable compression research** from **GPU rendering evaluation**:

- `cpu_reference`: deterministic reference renderer used to validate compression logic and run reproducible tests on any machine.
- `gsplat` adapter: optional CUDA path for full 3DGS rendering on real pretrained scenes.
- `streaming`: converts compressed model size into transfer-time / startup-time estimates under user-provided bandwidth assumptions.
- `ply_io`: reads standard GraphDeco-style Gaussian Splatting PLY files without external PLY dependencies.

No Netflix-internal target is assumed. Any bitrate used by this repository is explicitly user-configured and illustrative.

## Research questions

1. How much model size can be removed by view-independent importance pruning?
2. How much additional reduction comes from 16-bit or 8-bit parameter quantization?
3. What is the rendered-quality cost measured by PSNR / SSIM?
4. At what point does aggressive compression create a visible quality cliff?
5. How much does each configuration reduce transfer time at a chosen network bitrate?
6. How can a scene be ordered progressively so that the most important Gaussians arrive first?
7. How do encoding time and payload size change as scene size grows?

## Architecture

```text
3DGS PLY / Synthetic Scene
          │
          ▼
  GaussianScene model
          │
          ├─────────────── Baseline render
          │
          ▼
 Importance scoring
          │
          ▼
       Pruning
          │
          ▼
 Parameter quantization
          │
          ▼
 Binary payload encoding
          │
          ├─────────────── model bytes / encode time
          │
          ├─────────────── progressive chunks
          │
          ▼
   Reconstructed scene
          │
          ▼
       Rendering
          │
          ▼
 PSNR / SSIM / MAE
          │
          ▼
 size-quality-streaming tradeoff
```

## Quick start

```bash
python -m splatstream.experiment \
  --output results/demo \
  --gaussians 1200 \
  --width 192 \
  --height 192
```

Outputs:

```text
results/demo/
├── benchmark.csv
├── summary.json
├── baseline_view_00.png
├── ...
└── compressed_*.png
```

Run tests:

```bash
pytest -q
```

## Real GraphDeco PLY

Inspect:

```bash
python -m splatstream.inspect_ply /path/to/point_cloud.ply
```

Run a portable compression sweep using the DC color component as a CPU reference:

```bash
python -m splatstream.real_ply_experiment \
  /path/to/point_cloud.ply \
  --output results/real_scene \
  --max-gaussians 50000
```

This CPU path is not presented as a replacement for the official 3DGS rasterizer. It exists to validate model parsing, compression behavior, size accounting, and experiment orchestration without requiring CUDA.

## CUDA / gsplat evaluation

On an NVIDIA CUDA machine:

```bash
pip install -e ".[gpu]"
```

Use `splatstream/gsplat_adapter.py` as the integration point for real full-SH, anisotropic Gaussian rendering. The final portfolio report should use the same compression presets with full rendered views on an open dataset.

## Compression presets

The included sweep evaluates:

- baseline float32
- 25% importance pruning + float16-like reconstruction
- 50% pruning + 16-bit quantization
- 50% pruning + 8-bit quantization
- 75% pruning + 8-bit quantization

The exact results depend on the scene and renderer. The repository never hard-codes a claimed quality improvement.

## Why this project is useful

The work is not "I opened a Gaussian viewer." It demonstrates:

- reading a production 3DGS representation
- building an experiment harness
- defining a reference baseline
- implementing compression transforms
- measuring payload size and encode time
- evaluating rendered quality
- studying rate-distortion tradeoffs
- designing progressive delivery
- maintaining deterministic tests and CI

## Open-data plan for final evaluation

Use open scenes from the original 3DGS ecosystem or another explicitly redistributable dataset. Record in the report:

- dataset name and license
- number of training/test views
- number of Gaussians
- baseline PLY size
- compressed payload size
- encoding time
- rendering hardware
- PSNR / SSIM
- transfer-time estimate at several illustrative bandwidths

## Important claims policy

Do **not** describe the CPU reference renderer as Netflix-quality rendering or as the official 3DGS rasterizer.

Do **not** claim a compression ratio, quality score, or training-time reduction until the corresponding experiment has actually been run.

The portable benchmark included in the repository is a systems-validation benchmark. The final application version should add at least one real open 3DGS scene rendered with a full Gaussian rasterizer.
