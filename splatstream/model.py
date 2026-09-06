from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class GaussianScene:
    means: np.ndarray       # [N, 3], world-space
    scales: np.ndarray      # [N, 3], positive standard-deviation-like scale
    colors: np.ndarray      # [N, 3], linear-ish RGB in [0,1]
    opacity: np.ndarray     # [N], [0,1]
    rotation: np.ndarray | None = None  # [N,4], optional quaternion
    sh_rest: np.ndarray | None = None   # optional higher-order SH coefficients

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.colors = np.asarray(self.colors, dtype=np.float32)
        self.opacity = np.asarray(self.opacity, dtype=np.float32).reshape(-1)
        n = self.means.shape[0]
        if self.means.shape != (n, 3):
            raise ValueError("means must be [N,3]")
        if self.scales.shape != (n, 3):
            raise ValueError("scales must be [N,3]")
        if self.colors.shape != (n, 3):
            raise ValueError("colors must be [N,3]")
        if self.opacity.shape != (n,):
            raise ValueError("opacity must be [N]")
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=np.float32)
            if self.rotation.shape != (n,4):
                raise ValueError("rotation must be [N,4]")
        if self.sh_rest is not None:
            self.sh_rest = np.asarray(self.sh_rest, dtype=np.float32)
            if self.sh_rest.shape[0] != n:
                raise ValueError("sh_rest first dimension must match N")

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def subset(self, indices) -> "GaussianScene":
        idx = np.asarray(indices)
        return GaussianScene(
            means=self.means[idx],
            scales=self.scales[idx],
            colors=self.colors[idx],
            opacity=self.opacity[idx],
            rotation=None if self.rotation is None else self.rotation[idx],
            sh_rest=None if self.sh_rest is None else self.sh_rest[idx],
        )

    def copy(self) -> "GaussianScene":
        return GaussianScene(
            self.means.copy(),
            self.scales.copy(),
            self.colors.copy(),
            self.opacity.copy(),
            None if self.rotation is None else self.rotation.copy(),
            None if self.sh_rest is None else self.sh_rest.copy(),
        )
