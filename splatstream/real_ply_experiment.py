from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np

from .ply_io import graphdeco_to_scene, inspect
from .compress import CompressionConfig, compress, encode_payload
from .camera import Camera
from .cpu_renderer import render
from .metrics import psnr, ssim
from .image_io import save_image

PRESETS = [
    CompressionConfig("baseline_fp32", 0.00, 32),
    CompressionConfig("prune25_q16", 0.25, 16),
    CompressionConfig("prune50_q16", 0.50, 16),
    CompressionConfig("prune50_q8", 0.50, 8),
]

def auto_camera(scene):
    center = scene.means.mean(axis=0)
    ext = np.linalg.norm(scene.means.max(axis=0)-scene.means.min(axis=0))
    eye = center + np.array([0, 0, max(ext*1.2, 1.0)], np.float32)
    return Camera(eye=eye, target=center, up=np.array([0,1,0],np.float32), fov_y_deg=55)

def run(ply_path, output, max_gaussians=50000, width=256, height=256):
    output.mkdir(parents=True, exist_ok=True)
    metadata = inspect(ply_path)
    scene = graphdeco_to_scene(ply_path, max_gaussians=max_gaussians)
    cam = auto_camera(scene)
    ref = render(scene, cam, width, height)
    save_image(output/"reference_cpu_dc_only.png", ref)

    rows = []
    baseline_size = None
    for cfg in PRESETS:
        s = time.perf_counter()
        c = compress(scene, cfg)
        transform_ms = (time.perf_counter()-s)*1000
        payload, encode_ms = encode_payload(c, cfg.quant_bits)
        baseline_size = baseline_size or len(payload)
        img = render(c, cam, width, height)
        save_image(output/f"{cfg.name}.png", img)
        rows.append({
            "preset": cfg.name,
            "gaussians": c.n,
            "payload_bytes": len(payload),
            "ratio": baseline_size/len(payload),
            "transform_ms": transform_ms,
            "encode_ms": encode_ms,
            "psnr_cpu_dc_only": psnr(ref,img),
            "ssim_cpu_dc_only": ssim(ref,img),
        })

    with (output/"benchmark.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (output/"source_inspection.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return rows

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("--output", type=Path, default=Path("results/real"))
    p.add_argument("--max-gaussians", type=int, default=50000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    a=p.parse_args()
    for row in run(a.ply,a.output,a.max_gaussians,a.width,a.height):
        print(row)
