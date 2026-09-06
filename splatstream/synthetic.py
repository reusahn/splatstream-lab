from __future__ import annotations
import numpy as np
from .model import GaussianScene

def make_scene(n: int = 1200, seed: int = 7) -> GaussianScene:
    """
    Deterministic synthetic scene for portable compression experiments.

    It combines a floor-like sheet, a vertical cluster, and a curved object so
    that pruning/quantization alter both silhouette and appearance.
    """
    rng = np.random.default_rng(seed)

    n_floor = n // 3
    n_wall = n // 3
    n_obj = n - n_floor - n_wall

    # Floor
    x = rng.uniform(-2.2, 2.2, n_floor)
    z = rng.uniform(2.0, 6.0, n_floor)
    y = rng.normal(-1.0, 0.03, n_floor)
    floor = np.c_[x, y, z]
    floor_color = np.c_[
        0.18 + 0.10 * (z - 2) / 4,
        0.28 + 0.10 * (x + 2.2) / 4.4,
        0.42 + 0.08 * rng.random(n_floor)
    ]

    # Vertical colorful cluster
    wall = np.c_[
        rng.normal(-1.1, 0.35, n_wall),
        rng.uniform(-0.8, 1.4, n_wall),
        rng.normal(4.6, 0.12, n_wall),
    ]
    wall_color = np.c_[
        0.55 + 0.35 * rng.random(n_wall),
        0.12 + 0.35 * rng.random(n_wall),
        0.12 + 0.18 * rng.random(n_wall)
    ]

    # Curved / human-scale object
    phi = rng.uniform(0, 2*np.pi, n_obj)
    costheta = rng.uniform(-1, 1, n_obj)
    theta = np.arccos(costheta)
    radius = 0.65 + 0.12 * rng.normal(size=n_obj)
    obj = np.c_[
        0.85 + radius*np.sin(theta)*np.cos(phi),
        0.15 + 1.25*radius*np.cos(theta),
        3.7 + radius*np.sin(theta)*np.sin(phi),
    ]
    obj_color = np.c_[
        0.10 + 0.18 * rng.random(n_obj),
        0.48 + 0.35 * rng.random(n_obj),
        0.55 + 0.38 * rng.random(n_obj)
    ]

    means = np.vstack([floor, wall, obj]).astype(np.float32)
    colors = np.clip(np.vstack([floor_color, wall_color, obj_color]), 0, 1).astype(np.float32)

    base_scale = rng.lognormal(mean=-2.8, sigma=0.35, size=(n,3)).astype(np.float32)
    base_scale[:, 2] *= 0.65
    opacity = np.clip(rng.beta(5.0, 1.7, n), 0.08, 0.995).astype(np.float32)

    # A small population of low-importance splats for meaningful pruning.
    low = rng.choice(n, size=max(1, n//10), replace=False)
    opacity[low] *= 0.18

    rotation = np.zeros((n,4), dtype=np.float32)
    rotation[:,0] = 1.0

    return GaussianScene(means, base_scale, colors, opacity, rotation=rotation)
