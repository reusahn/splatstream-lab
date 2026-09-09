from __future__ import annotations

import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

GSPLAT_DIR = Path("/content/gsplat")
EXAMPLES = GSPLAT_DIR / "examples"
DATA_DIR = EXAMPLES / "data/360_v2/bonsai"
BASE_RESULT = EXAMPLES / "results/splatstream_bonsai_7k"
OUT = Path("/content/splatstream_full_cuda_rd")
ARCHIVE_BASE = Path("/content/splatstream_full_cuda_rd")

PRESETS = [
    ("baseline_f32", 0.00, 32),
    ("prune25_q16", 0.25, 16),
    ("prune50_q16", 0.50, 16),
    ("prune50_q8", 0.50, 8),
    ("prune75_q8", 0.75, 8),
]

SUPPORTED = ("means", "scales", "quats", "opacities", "sh0", "shN")


def newest_checkpoint() -> Path:
    ckpts = sorted(BASE_RESULT.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f"No trained checkpoint was found under {BASE_RESULT}. "
            "Run the Bonsai CUDA baseline in the same Colab runtime first."
        )
    return ckpts[-1]


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return None


def qdq(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, bytes]:
    """Quantize/dequantize along the Gaussian axis and return exact stored bytes.

    16-bit uses IEEE fp16. 8-bit uses per-channel min/max scalar quantization,
    where all dimensions after axis 0 are treated as channels. Side information
    (min/max arrays) is included in the returned payload bytes.
    """
    x = x.detach().float().cpu().contiguous()

    if bits >= 32:
        arr = x.numpy().astype("<f4", copy=False)
        return x.clone(), arr.tobytes(order="C")

    if bits == 16:
        q = x.to(torch.float16)
        payload = q.numpy().astype("<f2", copy=False).tobytes(order="C")
        return q.float(), payload

    if bits != 8:
        raise ValueError(f"Unsupported quantization depth: {bits}")

    lo = x.amin(dim=0, keepdim=True)
    hi = x.amax(dim=0, keepdim=True)
    span = torch.clamp(hi - lo, min=1e-12)
    q = torch.round((x - lo) / span * 255.0).clamp_(0, 255).to(torch.uint8)
    dq = q.float() / 255.0 * span + lo

    lo_b = lo.numpy().astype("<f4", copy=False).tobytes(order="C")
    hi_b = hi.numpy().astype("<f4", copy=False).tobytes(order="C")
    payload = lo_b + hi_b + q.numpy().tobytes(order="C")
    return dq.float(), payload


def add_array_blob(buf: bytearray, name: str, shape, bits: int, payload: bytes) -> None:
    name_b = name.encode("utf-8")
    buf.extend(struct.pack("<H", len(name_b)))
    buf.extend(name_b)
    buf.extend(struct.pack("<B", len(shape)))
    for d in shape:
        buf.extend(struct.pack("<I", int(d)))
    buf.extend(struct.pack("<B", int(bits)))
    buf.extend(struct.pack("<Q", len(payload)))
    buf.extend(payload)


def importance_order(splats: dict[str, torch.Tensor]) -> torch.Tensor:
    # View-independent heuristic shared with the portable experiment:
    # opacity_probability * geometric_mean(spatial_scale).
    opacity = torch.sigmoid(splats["opacities"].detach().float().cpu()).reshape(-1)
    log_scales = splats["scales"].detach().float().cpu()
    scale_gmean = torch.exp(log_scales.mean(dim=-1))
    score = opacity * scale_gmean
    return torch.argsort(score, descending=True)


def build_variant(
    base_splats: dict[str, torch.Tensor],
    order: torch.Tensor,
    prune_fraction: float,
    bits: int,
) -> tuple[dict[str, torch.Tensor], bytes, int]:
    n = int(base_splats["means"].shape[0])
    keep = max(1, int(round(n * (1.0 - prune_fraction))))
    if prune_fraction <= 0:
        idx = torch.arange(n)
    else:
        idx = order[:keep]

    selected = {k: v.detach().cpu()[idx] for k, v in base_splats.items()}
    out: dict[str, torch.Tensor] = {}
    blob = bytearray(b"SPLATRD1")

    # Positions are stored directly in world coordinates.
    out["means"], packed = qdq(selected["means"], bits)
    add_array_blob(blob, "means", out["means"].shape, bits, packed)

    # gsplat checkpoints store log-scales, so quantization here is in log domain.
    out["scales"], packed = qdq(selected["scales"], bits)
    add_array_blob(blob, "scales_log", out["scales"].shape, bits, packed)

    # Quantize quaternion components, then renormalize the dequantized values.
    quats_dq, packed = qdq(selected["quats"], bits)
    quats_dq = quats_dq / torch.clamp(torch.linalg.vector_norm(quats_dq, dim=-1, keepdim=True), min=1e-8)
    out["quats"] = quats_dq
    add_array_blob(blob, "quats", selected["quats"].shape, bits, packed)

    # Quantize opacity in probability space, then convert back to logits for gsplat.
    opacity_prob = torch.sigmoid(selected["opacities"].float())
    opacity_dq, packed = qdq(opacity_prob, bits)
    opacity_dq = opacity_dq.clamp(1e-6, 1.0 - 1e-6)
    out["opacities"] = torch.logit(opacity_dq)
    add_array_blob(blob, "opacity_probability", opacity_prob.shape, bits, packed)

    # SH coefficients remain in coefficient space. This preserves the actual
    # view-dependent representation used by the CUDA renderer.
    out["sh0"], packed = qdq(selected["sh0"], bits)
    add_array_blob(blob, "sh0", out["sh0"].shape, bits, packed)
    out["shN"], packed = qdq(selected["shN"], bits)
    add_array_blob(blob, "shN", out["shN"].shape, bits, packed)

    # Preserve unexpected auxiliary tensors losslessly if the checkpoint has any.
    for key, tensor in selected.items():
        if key in SUPPORTED:
            continue
        out[key] = tensor.detach().float().cpu().contiguous()
        raw = out[key].numpy().astype("<f4", copy=False).tobytes(order="C")
        add_array_blob(blob, key, out[key].shape, 32, raw)

    return out, bytes(blob), keep


def run_live(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    print("\n$", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "2")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def find_val_stats(result_dir: Path) -> dict | None:
    files = sorted((result_dir / "stats").glob("val_step*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def evaluate_checkpoint(name: str, ckpt: Path) -> tuple[dict, Path]:
    eval_dir = OUT / "evals" / name
    stats = find_val_stats(eval_dir)
    if stats is not None:
        print(f"Reusing existing held-out metrics for {name}: {stats}")
        return stats, eval_dir

    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "simple_trainer.py",
        "default",
        "--disable_viewer",
        "--disable_video",
        "--data_factor",
        "2",
        "--data_dir",
        "data/360_v2/bonsai",
        "--result_dir",
        str(eval_dir),
        "--ckpt",
        str(ckpt),
    ]
    run_live(cmd, cwd=EXAMPLES, log_path=log_path)
    stats = find_val_stats(eval_dir)
    if stats is None:
        raise RuntimeError(f"Evaluation finished but no validation stats were produced for {name}")
    return stats, eval_dir


def copy_sample_renders(name: str, source_dir: Path, limit: int = 3) -> None:
    renders = sorted((source_dir / "renders").glob("val_step*.png"))
    if not renders:
        return
    dst = OUT / "sample_renders" / name
    dst.mkdir(parents=True, exist_ok=True)
    if len(renders) <= limit:
        chosen = renders
    else:
        positions = np.linspace(0, len(renders) - 1, limit, dtype=int)
        chosen = [renders[i] for i in positions]
    for p in chosen:
        shutil.copy2(p, dst / p.name)


def save_plots(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    ratios = [r["zlib_compression_ratio"] for r in rows]
    labels = [r["preset"] for r in rows]

    for key, ylabel, filename in [
        ("psnr", "PSNR (dB)", "rate_distortion_psnr.png"),
        ("ssim", "SSIM", "rate_distortion_ssim.png"),
        ("lpips", "LPIPS (lower is better)", "rate_distortion_lpips.png"),
    ]:
        values = [r[key] for r in rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(ratios, values, marker="o")
        ax.set_xlabel("Experimental payload compression ratio (zlib-backed)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        for x, y, label in zip(ratios, values, labels):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=160)
        plt.close(fig)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full held-out rate-distortion sweep.")
    if not GSPLAT_DIR.exists() or not DATA_DIR.exists():
        raise FileNotFoundError(
            "The gsplat source and Bonsai dataset must already exist in this Colab runtime. "
            "Run the baseline CUDA notebook first."
        )

    base_ckpt_path = newest_checkpoint()
    print("Base checkpoint:", base_ckpt_path)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)

    if OUT.exists():
        # Keep completed eval subdirectories if rerunning after a partial failure.
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
    else:
        OUT.mkdir(parents=True)
        (OUT / "logs").mkdir(parents=True)

    base = torch.load(base_ckpt_path, map_location="cpu", weights_only=True)
    if "splats" not in base:
        raise KeyError("The base checkpoint does not contain a 'splats' state dict.")
    base_splats = base["splats"]
    missing = [k for k in SUPPORTED if k not in base_splats]
    if missing:
        raise KeyError(f"Base checkpoint is missing required splat tensors: {missing}")

    n0 = int(base_splats["means"].shape[0])
    print("Base Gaussians:", n0)
    print("Computing view-independent importance order once...")
    order = importance_order(base_splats)

    rows: list[dict] = []
    payload_records: dict[str, dict] = {}
    baseline_zlib = None
    baseline_raw = None

    variants_dir = OUT / "variant_checkpoints"
    variants_dir.mkdir(parents=True, exist_ok=True)

    for name, prune_fraction, bits in PRESETS:
        print("\n" + "=" * 72)
        print(f"Preset: {name} | prune={prune_fraction:.0%} | quant={bits}-bit")
        print("=" * 72)

        splats_dq, raw_blob, keep = build_variant(base_splats, order, prune_fraction, bits)
        zblob = zlib.compress(raw_blob, level=6)
        raw_bytes = len(raw_blob)
        zlib_bytes = len(zblob)

        if baseline_zlib is None:
            baseline_zlib = zlib_bytes
            baseline_raw = raw_bytes

        payload_records[name] = {
            "gaussians": keep,
            "packed_uncompressed_bytes": raw_bytes,
            "zlib_payload_bytes": zlib_bytes,
        }

        if name == "baseline_f32":
            ckpt_path = base_ckpt_path
            # Reuse metrics/renders from the completed baseline run when present.
            stats = find_val_stats(BASE_RESULT)
            if stats is None:
                stats, eval_dir = evaluate_checkpoint(name, ckpt_path)
                copy_sample_renders(name, eval_dir)
            else:
                eval_dir = BASE_RESULT
                print("Reusing baseline held-out stats:", stats)
                copy_sample_renders(name, eval_dir)
        else:
            variant_ckpt = dict(base)
            variant_ckpt["splats"] = splats_dq
            variant_ckpt["compression_metadata"] = {
                "preset": name,
                "prune_fraction": prune_fraction,
                "quant_bits": bits,
                "importance": "sigmoid(opacity_logit) * exp(mean(log_scale))",
            }
            ckpt_path = variants_dir / f"{name}.pt"
            torch.save(variant_ckpt, ckpt_path)
            stats, eval_dir = evaluate_checkpoint(name, ckpt_path)
            copy_sample_renders(name, eval_dir)
            # The dequantized float32 checkpoint is an evaluation carrier, not the
            # compressed payload. Delete it after evaluation to save Colab disk.
            ckpt_path.unlink(missing_ok=True)

        ratio_z = float(baseline_zlib / zlib_bytes)
        ratio_raw = float(baseline_raw / raw_bytes)
        row = {
            "preset": name,
            "prune_fraction": prune_fraction,
            "quant_bits": bits,
            "num_gaussians": int(stats.get("num_GS", keep)),
            "packed_uncompressed_bytes": raw_bytes,
            "zlib_payload_bytes": zlib_bytes,
            "raw_compression_ratio": ratio_raw,
            "zlib_compression_ratio": ratio_z,
            "psnr": float(stats["psnr"]),
            "ssim": float(stats["ssim"]),
            "lpips": float(stats["lpips"]),
            "render_seconds_per_image": float(stats["ellipse_time"]),
            "transfer_seconds_25mbps": zlib_bytes * 8.0 / 25_000_000.0,
            "transfer_seconds_50mbps": zlib_bytes * 8.0 / 50_000_000.0,
            "transfer_seconds_100mbps": zlib_bytes * 8.0 / 100_000_000.0,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    base_psnr = rows[0]["psnr"]
    base_ssim = rows[0]["ssim"]
    base_lpips = rows[0]["lpips"]
    for row in rows:
        row["psnr_delta_db"] = row["psnr"] - base_psnr
        row["ssim_delta"] = row["ssim"] - base_ssim
        row["lpips_delta"] = row["lpips"] - base_lpips

    csv_path = OUT / "full_cuda_rate_distortion.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "full_cuda_rate_distortion.json").write_text(json.dumps(rows, indent=2))

    env = {
        "dataset": "Mip-NeRF 360 / Bonsai",
        "data_factor": 2,
        "test_every": 8,
        "base_checkpoint": str(base_ckpt_path),
        "base_step": int(base.get("step", -1)),
        "base_gaussians": n0,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gsplat_commit": git_head(GSPLAT_DIR),
        "splatstream_commit": git_head(Path("/content/splatstream-lab")),
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))

    method = """# Full CUDA rate-distortion method\n\n- Dataset: Mip-NeRF 360 / Bonsai, held-out split from gsplat's standard COLMAP loader (`test_every=8`).\n- Renderer and evaluator: pinned gsplat `simple_trainer.py` held-out evaluation path.\n- Importance heuristic: `sigmoid(opacity_logit) * exp(mean(log_scale))`.\n- Pruning: retain the highest-scoring Gaussians globally.\n- 16-bit quantization: IEEE fp16 followed by fp32 dequantization for rendering.\n- 8-bit quantization: per-channel min/max uniform quantization with min/max side information included in payload accounting.\n- Scale parameters are quantized in checkpoint log-scale space.\n- Opacity is quantized in probability space and converted back to logits before rendering.\n- Quaternion components are quantized and the dequantized quaternions are normalized before rendering.\n- SH coefficients (`sh0`, `shN`) are quantized in coefficient space, preserving view-dependent rendering.\n- Payload sizes use the exact experimental binary representation above plus zlib level 6 as a generic entropy backend. zlib is not presented as a proposed production codec.\n- The saved `.pt` variants are dequantized float32 evaluation carriers and are not used as compressed-size measurements.\n- PSNR, SSIM, LPIPS, and render time are measured on the same held-out cameras for every operating point.\n"""
    (OUT / "METHOD.md").write_text(method)

    save_plots(rows)

    # Do not include all held-out renders or temporary evaluation checkpoints.
    # Keep metrics, plots, logs, method notes, and three representative renders per preset.
    package = Path("/content/splatstream_full_cuda_rd_package")
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    for name in [
        "full_cuda_rate_distortion.csv",
        "full_cuda_rate_distortion.json",
        "environment.json",
        "METHOD.md",
        "rate_distortion_psnr.png",
        "rate_distortion_ssim.png",
        "rate_distortion_lpips.png",
    ]:
        shutil.copy2(OUT / name, package / name)
    if (OUT / "logs").exists():
        shutil.copytree(OUT / "logs", package / "logs")
    if (OUT / "sample_renders").exists():
        shutil.copytree(OUT / "sample_renders", package / "sample_renders")

    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=package)
    print("\n" + "=" * 72)
    print("FULL CUDA RATE-DISTORTION SWEEP COMPLETE")
    print("CSV:", csv_path)
    print("Artifact ZIP:", archive)
    print("=" * 72)

    try:
        from google.colab import files
        files.download(archive)
    except Exception:
        print("Download manually from:", archive)


if __name__ == "__main__":
    main()
