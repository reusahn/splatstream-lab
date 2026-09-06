# Resume bullets: use only after the corresponding measurements exist

**SplatStream Lab — 3D Gaussian Splatting Compression Research** · Python, 3DGS, gsplat · 2026

Safe now:
- Built a reproducible 3D Gaussian Splatting compression harness that reads standard PLY scene representations, applies importance pruning and parameter quantization, and evaluates payload size, encoding time, and rendered-quality tradeoffs.
- Implemented exact experiment bookkeeping, PSNR/SSIM evaluation, progressive importance ordering, transfer-time simulation, tests, and CI, with a separate CUDA integration path for full real-scene Gaussian rendering.

Replace only after real GPU/open-scene results:
- Evaluated [N] open 3DGS scenes and reduced average model payload by [X]× at [Y] dB PSNR / [Z] SSIM, while measuring encoding cost and progressive startup quality.
- Compared [compression strategy A/B/C] across [dataset names], identifying [specific result] as the best rate-distortion operating point on [hardware].
