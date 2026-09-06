# Portfolio copy draft

## SplatStream Lab
### Rate-distortion experiments for 3D Gaussian Splatting

I built an experimental compression and progressive-delivery harness for 3D Gaussian Splatting to study a practical question: how aggressively can a scene be reduced before rendered quality falls off?

The system reads standard 3DGS representations, ranks Gaussians with an explicit view-independent importance heuristic, applies pruning and parameter quantization, serializes compact payloads, and measures size, encoding time, and image quality. A progressive mode orders high-contribution Gaussians first and estimates transfer time under user-defined bandwidth conditions.

A deterministic CPU reference renderer keeps the compression pipeline testable on any machine. Full application results are evaluated separately with a CUDA Gaussian rasterizer on real open scenes so that portable validation results are not confused with production 3DGS rendering.

**Python · 3D Gaussian Splatting · Compression · PSNR/SSIM · Progressive Streaming · GPU evaluation**
