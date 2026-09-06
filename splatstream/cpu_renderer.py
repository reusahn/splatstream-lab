from __future__ import annotations
import math
import numpy as np
from .model import GaussianScene
from .camera import Camera, camera_basis

def render(
    scene: GaussianScene,
    camera: Camera,
    width: int = 192,
    height: int = 192,
    background=(0.015, 0.018, 0.025),
) -> np.ndarray:
    """
    Portable reference renderer.

    This is intentionally a compact CPU approximation for compression-system
    validation. It is NOT the official 3DGS tile rasterizer and does not model
    full anisotropic covariance or view-dependent spherical harmonics.

    It projects each Gaussian to a 2D Gaussian footprint and composites
    front-to-back.
    """
    eye, right, up, forward = camera_basis(camera)
    rel = scene.means - eye[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward

    visible = z > 0.15
    if not np.any(visible):
        return np.broadcast_to(np.array(background, np.float32), (height,width,3)).copy()

    x, y, z = x[visible], y[visible], z[visible]
    scales = scene.scales[visible]
    colors = scene.colors[visible]
    opacity = scene.opacity[visible]

    fy = 0.5 * height / math.tan(math.radians(camera.fov_y_deg) * 0.5)
    fx = fy
    u = width * 0.5 + fx * (x / z)
    v = height * 0.5 - fy * (y / z)

    # Approximate projected footprint from x/y scales.
    sigma_world = np.sqrt(np.maximum(scales[:,0] * scales[:,1], 1e-8))
    sigma = np.clip(fx * sigma_world / z, 0.55, 7.5)

    # Front-to-back
    order = np.argsort(z)
    img = np.zeros((height,width,3), np.float32)
    img[:] = np.array(background, np.float32)
    trans = np.ones((height,width), np.float32)

    for i in order:
        if u[i] < -20 or u[i] > width+20 or v[i] < -20 or v[i] > height+20:
            continue
        s = float(sigma[i])
        r = max(1, int(math.ceil(3.0*s)))
        x0 = max(0, int(math.floor(u[i])) - r)
        x1 = min(width, int(math.floor(u[i])) + r + 1)
        y0 = max(0, int(math.floor(v[i])) - r)
        y1 = min(height, int(math.floor(v[i])) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - u[i])**2 + (yy - v[i])**2) / (s*s + 1e-8)
        a = np.clip(float(opacity[i]) * np.exp(-0.5*d2), 0.0, 0.985).astype(np.float32)
        t = trans[y0:y1, x0:x1]
        w = t * a
        patch = img[y0:y1, x0:x1]
        patch += w[...,None] * (colors[i][None,None,:] - patch)
        trans[y0:y1, x0:x1] = t * (1.0 - a)

    return np.clip(img, 0, 1)
