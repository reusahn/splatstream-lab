from __future__ import annotations
from dataclasses import dataclass
import io, json, struct, time, zlib
import numpy as np
from .model import GaussianScene

@dataclass(frozen=True)
class CompressionConfig:
    name: str
    prune_fraction: float = 0.0
    quant_bits: int = 32

def importance(scene: GaussianScene) -> np.ndarray:
    """
    Simple view-independent heuristic.

    opacity * geometric mean spatial scale.

    It is deliberately labeled a heuristic, not a published importance metric.
    """
    volume_proxy = np.cbrt(np.maximum(np.prod(scene.scales, axis=1), 1e-12))
    return scene.opacity * volume_proxy

def prune(scene: GaussianScene, fraction: float) -> GaussianScene:
    fraction = float(np.clip(fraction, 0.0, 0.95))
    if fraction <= 0:
        return scene.copy()
    keep = max(1, int(round(scene.n * (1.0 - fraction))))
    scores = importance(scene)
    idx = np.argpartition(scores, -keep)[-keep:]
    # Stable importance order is helpful for progressive transmission.
    idx = idx[np.argsort(scores[idx])[::-1]]
    return scene.subset(idx)

def quantize_array(x: np.ndarray, bits: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if bits >= 32:
        return x.copy()
    if bits == 16:
        return x.astype(np.float16).astype(np.float32)
    if bits != 8:
        raise ValueError("Portable implementation supports 8, 16, or 32 bits.")

    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    span = np.maximum(hi-lo, 1e-12)
    q = np.round((x-lo)/span*255.0).astype(np.uint8)
    return (q.astype(np.float32)/255.0*span + lo).astype(np.float32)

def quantize_scene(scene: GaussianScene, bits: int) -> GaussianScene:
    if bits >= 32:
        return scene.copy()

    # Scale is more stable in log domain.
    log_scale = np.log(np.maximum(scene.scales, 1e-8))
    q_scale = np.exp(quantize_array(log_scale, bits))

    rot = None
    if scene.rotation is not None:
        rot = quantize_array(scene.rotation, bits)
        norm = np.linalg.norm(rot, axis=1, keepdims=True)
        rot = rot / np.clip(norm, 1e-8, None)

    sh_rest = None if scene.sh_rest is None else quantize_array(scene.sh_rest, bits)

    return GaussianScene(
        means=quantize_array(scene.means, bits),
        scales=q_scale,
        colors=np.clip(quantize_array(scene.colors, bits), 0, 1),
        opacity=np.clip(quantize_array(scene.opacity[:,None], bits).reshape(-1), 0, 1),
        rotation=rot,
        sh_rest=sh_rest,
    )

def compress(scene: GaussianScene, cfg: CompressionConfig) -> GaussianScene:
    return quantize_scene(prune(scene, cfg.prune_fraction), cfg.quant_bits)

def _encode_array(buf: io.BytesIO, name: str, arr: np.ndarray, bits: int):
    arr = np.asarray(arr, dtype=np.float32)
    name_b = name.encode("utf-8")
    buf.write(struct.pack("<H", len(name_b)))
    buf.write(name_b)
    buf.write(struct.pack("<B", arr.ndim))
    for d in arr.shape:
        buf.write(struct.pack("<I", int(d)))
    buf.write(struct.pack("<B", bits))

    if bits >= 32:
        payload = arr.astype("<f4").tobytes()
        meta = b""
    elif bits == 16:
        payload = arr.astype("<f2").tobytes()
        meta = b""
    elif bits == 8:
        lo = arr.min(axis=0, keepdims=True).astype("<f4")
        hi = arr.max(axis=0, keepdims=True).astype("<f4")
        span = np.maximum(hi-lo, 1e-12)
        q = np.round((arr-lo)/span*255.0).astype(np.uint8)
        meta = lo.tobytes() + hi.tobytes()
        payload = q.tobytes()
    else:
        raise ValueError("bits must be 8,16,32")

    buf.write(struct.pack("<I", len(meta)))
    buf.write(meta)
    buf.write(struct.pack("<Q", len(payload)))
    buf.write(payload)

def encode_payload(scene: GaussianScene, bits: int = 32, level: int = 6) -> tuple[bytes, float]:
    """
    A compact experimental payload used only for relative systems benchmarking.

    zlib is used as a generic entropy coder. This is not a proposed Netflix codec.
    """
    start = time.perf_counter()
    raw = io.BytesIO()
    raw.write(b"SPLATLAB1")
    raw.write(struct.pack("<I", scene.n))
    _encode_array(raw, "means", scene.means, bits)
    _encode_array(raw, "scales", scene.scales, bits)
    _encode_array(raw, "colors", scene.colors, bits)
    _encode_array(raw, "opacity", scene.opacity[:,None], bits)
    if scene.rotation is not None:
        _encode_array(raw, "rotation", scene.rotation, bits)
    if scene.sh_rest is not None:
        _encode_array(raw, "sh_rest", scene.sh_rest, bits)
    payload = zlib.compress(raw.getvalue(), level)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return payload, elapsed_ms

def progressive_order(scene: GaussianScene) -> np.ndarray:
    return np.argsort(importance(scene))[::-1]

def progressive_prefix(scene: GaussianScene, fraction: float) -> GaussianScene:
    order = progressive_order(scene)
    count = max(1, int(round(scene.n * float(np.clip(fraction, 0.01, 1.0)))))
    return scene.subset(order[:count])
