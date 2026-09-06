"""
CUDA integration point for final real-scene evaluation.

The portable project can be developed and tested without CUDA. For the final
Netflix-facing experiment, use a CUDA machine and plug the compressed/rebuilt
parameters into gsplat's rasterization API, then evaluate held-out views with
PSNR/SSIM/LPIPS.

Why this is separated:
- CI should remain deterministic and inexpensive.
- The application should distinguish portable systems tests from full GPU 3DGS evaluation.
- Full SH coefficients and anisotropic covariance should be evaluated with a real Gaussian rasterizer.

Reference:
https://github.com/nerfstudio-project/gsplat

Suggested final pipeline:
1. Train or download an open 3DGS scene.
2. Export standard GraphDeco PLY.
3. Run the same prune/quantization presets.
4. Reconstruct tensors.
5. Render held-out cameras with gsplat.
6. Compute PSNR, SSIM, LPIPS.
7. Record GPU model, CUDA version, renderer version, and render FPS.
"""

def require_cuda_stack():
    try:
        import torch
        import gsplat
    except Exception as exc:
        raise RuntimeError(
            "Install the GPU extras on an NVIDIA CUDA machine: pip install -e '.[gpu]'"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")
    return torch, gsplat
