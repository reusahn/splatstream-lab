# Final Netflix application checklist

## Must finish before calling the GS project "complete"

- [ ] Run `notebooks/SplatStream_Lab_CUDA_Validation.ipynb` on an NVIDIA GPU.
- [ ] Train at least one open real scene with gsplat.
- [ ] Save actual training wall time.
- [ ] Save held-out baseline PSNR / SSIM / LPIPS.
- [ ] Export or identify a standard 3DGS PLY.
- [ ] Run the SplatStream compression sweep.
- [ ] Re-render compressed variants with a full CUDA Gaussian rasterizer.
- [ ] Produce a rate-distortion table.
- [ ] Add one visual comparison figure: baseline vs moderate vs aggressive.
- [ ] Add one progressive-delivery figure or video.
- [ ] Record hardware, software versions, dataset source, and license.
- [ ] Update resume bullets only with measured numbers.
- [ ] Link the public GitHub repository from the portfolio.
- [ ] Prepare a 60-second verbal explanation of the project.

## Stronger version

- [ ] Repeat on 3 scenes with different content characteristics.
- [ ] Add SH-degree truncation.
- [ ] Add SPZ or another established compression baseline.
- [ ] Compare view-independent heuristic pruning with a view-aware importance metric.
- [ ] Measure render FPS and GPU memory.
- [ ] Test progressive startup at multiple illustrative network rates.
- [ ] Write a 1–2 page research note with method, experiments, results, failure cases, and next steps.
