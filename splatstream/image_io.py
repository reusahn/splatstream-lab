from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

def save_image(path, image: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
