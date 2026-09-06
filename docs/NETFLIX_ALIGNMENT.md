# Alignment with Netflix JR40251

This file maps the project to the public job description without claiming private knowledge.

## "Explore GS model compression strategies using open datasets"

Project evidence:
- standard GraphDeco PLY reader
- importance pruning
- float16 / 8-bit parameter quantization
- entropy-coded experimental payload
- repeatable preset sweeps
- open-dataset evaluation plan

## "Characterize trade-offs among GS model size, training time, and rendered quality"

Project evidence:
- payload byte count
- encode time
- PSNR / SSIM
- training-run record schema
- progressive delivery quality curve
- bandwidth-dependent transfer-time calculation

Training time is recorded only from an actual GPU training run. The CPU-only benchmark does not invent a training-time claim.

## "Quantify the gap relative to streaming-rate targets"

Project evidence:
- model-to-transfer-time conversion for user-provided bandwidth assumptions
- progressive-prefix experiment
- no claim that a chosen bitrate is Netflix's internal target

## "Identify strategies to reduce training/encoding time and/or improve compression"

Current work:
- post-training pruning
- parameter quantization
- compact payload encoding
- progressive importance ordering

Future GPU extension:
- early-stop / densification schedule experiments
- SH-degree reduction
- learned/vector quantization
- scene-adaptive pruning
- rate-distortion optimized pruning
- SPZ comparison
- parallel encoding

## "Design and implement a PoC that showcases GS-based rendering"

Current PoC:
- deterministic CPU Gaussian reference renderer for systems validation
- generated before/after views and rate-distortion benchmark

Final application PoC:
- full anisotropic + SH rendering using gsplat on an NVIDIA GPU
- held-out camera evaluation on open real scenes

## Candidate skill mapping

Public requirement | Evidence target
---|---
Python | this repository
software engineering practices | Git, tests, CI, reproducible configs
3D reconstruction / GS / neural graphics | real-scene 3DGS experiment + prior graphics portfolio
ML training/evaluation | actual CUDA 3DGS training/evaluation run + PyTorch background
real-time rendering | existing Unity/Unreal work + Ray-Scene Acceleration project
GPU programming | optional gsplat/CUDA run; do not overclaim custom CUDA unless written
video compression | rate-distortion framing + codec reading; do not claim HEVC/AV1 implementation without doing it
open-source graphics | public GitHub repository and documented upstream references
