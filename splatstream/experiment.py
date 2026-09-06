from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np

from .synthetic import make_scene
from .camera import default_cameras
from .cpu_renderer import render
from .compress import CompressionConfig, compress, encode_payload, progressive_prefix
from .metrics import psnr, ssim, mae
from .streaming import report as streaming_report
from .image_io import save_image

PRESETS = [
    CompressionConfig("baseline_fp32", 0.00, 32),
    CompressionConfig("prune25_q16", 0.25, 16),
    CompressionConfig("prune50_q16", 0.50, 16),
    CompressionConfig("prune50_q8",  0.50, 8),
    CompressionConfig("prune75_q8",  0.75, 8),
]

def mean_finite(values):
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("inf")

def run(output: Path, n: int, width: int, height: int, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    scene = make_scene(n, seed)
    cameras = default_cameras()

    reference = []
    for i, cam in enumerate(cameras):
        img = render(scene, cam, width, height)
        reference.append(img)
        save_image(output / f"baseline_view_{i:02d}.png", img)

    rows = []
    baseline_payload_size = None

    for cfg in PRESETS:
        start = time.perf_counter()
        compressed = compress(scene, cfg)
        transform_ms = (time.perf_counter() - start) * 1000.0
        payload, encode_ms = encode_payload(compressed, cfg.quant_bits)
        if baseline_payload_size is None:
            baseline_payload_size = len(payload)

        psnrs, ssims, maes = [], [], []
        render_ms = []
        for i, (cam, ref) in enumerate(zip(cameras, reference)):
            s = time.perf_counter()
            img = render(compressed, cam, width, height)
            render_ms.append((time.perf_counter()-s)*1000.0)
            psnrs.append(psnr(ref, img))
            ssims.append(ssim(ref, img))
            maes.append(mae(ref, img))
            save_image(output / f"{cfg.name}_view_{i:02d}.png", img)

        row = {
            "preset": cfg.name,
            "gaussians": compressed.n,
            "prune_fraction": cfg.prune_fraction,
            "quant_bits": cfg.quant_bits,
            "payload_bytes": len(payload),
            "payload_kib": len(payload)/1024.0,
            "compression_ratio_vs_baseline": baseline_payload_size/len(payload),
            "transform_ms": transform_ms,
            "payload_encode_ms": encode_ms,
            "mean_render_ms_cpu_reference": float(np.mean(render_ms)),
            "mean_psnr_db": mean_finite(psnrs),
            "mean_ssim": float(np.mean(ssims)),
            "mean_mae": float(np.mean(maes)),
            **streaming_report(len(payload)),
        }
        rows.append(row)

    # Progressive-delivery experiment from baseline scene.
    progressive = []
    for fraction in [0.10,0.25,0.50,0.75,1.0]:
        prefix = progressive_prefix(scene, fraction)
        payload, enc_ms = encode_payload(prefix, 16)
        q = []
        for cam, ref in zip(cameras, reference):
            img = render(prefix, cam, width, height)
            q.append(psnr(ref, img))
        progressive.append({
            "fraction": fraction,
            "gaussians": prefix.n,
            "payload_bytes_q16": len(payload),
            "mean_psnr_db": mean_finite(q),
            "encode_ms": enc_ms,
            **streaming_report(len(payload)),
        })

    with (output/"benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (output/"progressive.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=progressive[0].keys())
        writer.writeheader()
        writer.writerows(progressive)

    summary = {
        "benchmark_kind": "portable synthetic CPU reference",
        "scene_gaussians": scene.n,
        "resolution": [width,height],
        "views": len(cameras),
        "seed": seed,
        "warning": "CPU reference renderer is not the official 3DGS rasterizer.",
        "presets": rows,
        "progressive": progressive,
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, progressive

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/demo"))
    p.add_argument("--gaussians", type=int, default=1200)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rows, _ = run(args.output, args.gaussians, args.width, args.height, args.seed)
    for row in rows:
        print(
            f"{row['preset']:16s} "
            f"N={row['gaussians']:5d} "
            f"size={row['payload_kib']:8.2f} KiB "
            f"ratio={row['compression_ratio_vs_baseline']:5.2f}x "
            f"PSNR={row['mean_psnr_db']:6.2f} "
            f"SSIM={row['mean_ssim']:.4f}"
        )
