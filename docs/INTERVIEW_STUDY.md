# Gaussian Splatting interview study checklist

Be able to explain these without memorized buzzwords.

## Representation
A 3D Gaussian generally carries:
- 3D mean
- anisotropic covariance, commonly parameterized by scale + rotation
- opacity
- color represented with spherical harmonics

## Rendering
Explain:
- camera projection
- covariance projection into screen space
- visibility / depth ordering
- alpha compositing
- why tile-based rasterization matters for performance

## Training
Explain:
- SfM / COLMAP initialization
- differentiable rasterization
- photometric loss
- densification and pruning
- why Gaussian count can grow substantially
- relation between training time, quality, and model size

## Compression questions
Be ready to reason about:
- pruning low-contribution Gaussians
- quantizing positions / scales / rotations / SH
- dropping higher-order SH coefficients
- entropy coding
- vector quantization
- spatial clustering
- progressive / level-of-detail transmission
- temporal redundancy for 4DGS

## Quality
Know:
- PSNR
- SSIM
- LPIPS
- why image quality should be evaluated on held-out views
- why a higher compression ratio is meaningless without rate-distortion quality

## Streaming
Be able to distinguish:
- model payload size
- startup latency
- sustained bitrate
- progressive refinement
- decoder / renderer memory
- constrained consumer hardware

## Important honesty boundary
Do not claim:
- custom CUDA work unless you actually write it
- Netflix streaming targets
- a 3DGS publication
- a full 3DGS training result before actually running it
