from __future__ import annotations
import math
import numpy as np

def mae(reference: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean(np.abs(reference.astype(np.float64) - test.astype(np.float64))))

def mse(reference: np.ndarray, test: np.ndarray) -> float:
    d = reference.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(d*d))

def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    err = mse(reference, test)
    if err <= 1e-15:
        return float("inf")
    return 10.0 * math.log10((data_range*data_range) / err)

def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(reference, test, channel_axis=2, data_range=data_range))
    except Exception:
        # Fallback: global SSIM-like score, explicitly only a fallback.
        x = reference.astype(np.float64)
        y = test.astype(np.float64)
        mu_x, mu_y = x.mean(), y.mean()
        var_x, var_y = x.var(), y.var()
        cov = ((x-mu_x)*(y-mu_y)).mean()
        c1 = (0.01*data_range)**2
        c2 = (0.03*data_range)**2
        return float(((2*mu_x*mu_y+c1)*(2*cov+c2))/((mu_x**2+mu_y**2+c1)*(var_x+var_y+c2)))
