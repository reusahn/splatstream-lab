from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

GSPLAT_COMMIT = "28e794ca44a4c25ffc39175370c5ee7b38bfcc36"
GSPLAT_DIR = Path("/content/gsplat")
EXAMPLES = GSPLAT_DIR / "examples"
BONSAI = EXAMPLES / "data/360_v2/bonsai"
RESULT = EXAMPLES / "results/splatstream_bonsai_7k"
LOG = Path("/content/splatstream_bonsai_7k.log")
SPLATSTREAM = Path("/content/splatstream-lab")
PORTABLE_OUT = SPLATSTREAM / "results/bonsai_real_ply_sanity"


def run(cmd, *, cwd=None, env=None):
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    subprocess.run(
        cmd,
        cwd=None if cwd is None else str(cwd),
        env=merged,
        check=True,
    )


def run_live(cmd, *, cwd=None, env=None, log_path=None):
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})

    log_file = open(log_path, "w") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=None if cwd is None else str(cwd),
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            if log_file:
                log_file.write(line)
                log_file.flush()
        rc = proc.wait()
    finally:
        if log_file:
            log_file.close()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def version_tuple(v: str):
    core = v.split("+")[0].split(".")
    return tuple(int(x) for x in core[:2])


def check_environment():
    print("=" * 72)
    print("SPLATSTREAM LAB / NETFLIX GS COLAB PIPELINE")
    print("=" * 72)
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Colab choose Runtime > Change runtime type > GPU, "
            "reconnect, then run the notebook again."
        )
    print("GPU:", torch.cuda.get_device_name(0))

    if version_tuple(torch.__version__) < (2, 7):
        raise RuntimeError(
            f"PyTorch {torch.__version__} is too old for the pinned gsplat source. "
            "Open a fresh current Colab GPU runtime instead of patching this one."
        )


def setup_gsplat():
    print("\n[1/6] Setting up pinned gsplat source")

    if not GSPLAT_DIR.exists():
        run(["git", "clone", "https://github.com/nerfstudio-project/gsplat.git", GSPLAT_DIR])
    run(["git", "fetch", "--all", "--tags"], cwd=GSPLAT_DIR)
    run(["git", "reset", "--hard", GSPLAT_COMMIT], cwd=GSPLAT_DIR)

    run([
        sys.executable, "-m", "pip", "install", "-U",
        "pip", "setuptools", "wheel", "ninja", "rich"
    ])

    # Deliberately keep Colab's existing CUDA-enabled torch/torchvision pair.
    # These are the dependencies needed by the standard COLMAP trainer path.
    deps = [
        "numpy>=2.0,<3.0",
        "Pillow",
        "tqdm",
        "tyro>=0.8.8,!=1.0.9,!=1.0.10",
        "imageio[ffmpeg]",
        "scipy",
        "scikit-learn",
        "torchmetrics==1.8.2",
        "opencv-python-headless",
        "pyyaml",
        "matplotlib",
        "splines",
        "tensorboard",
        "tensorly",
        "piexif",
        "viser",
        "pycolmap>=3.10.0",
    ]
    run([sys.executable, "-m", "pip", "install", *deps])

    run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787",
    ])

    # Upstream-recommended development/JIT path. Avoid AOT CUDA wheel build.
    run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"],
        cwd=GSPLAT_DIR,
        env={"BUILD_NO_CUDA": "1"},
    )

    run([
        sys.executable,
        "-c",
        "import gsplat, torch; "
        "print('gsplat import OK'); "
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
        "print('gpu', torch.cuda.get_device_name(0))",
    ])


def prepare_dataset():
    print("\n[2/6] Preparing Mip-NeRF 360 / Bonsai")
    if BONSAI.exists():
        print("Bonsai already exists:", BONSAI)
        return

    # The upstream downloader defaults to mipnerf360 and examples/data.
    run([sys.executable, "datasets/download_dataset.py"], cwd=EXAMPLES)
    if not BONSAI.exists():
        raise RuntimeError(f"Download finished but Bonsai was not found at {BONSAI}")
    print("Bonsai ready:", BONSAI)


def train():
    print("\n[3/6] Training Bonsai for 7,000 steps")
    ckpts = sorted(RESULT.rglob("*.pt")) if RESULT.exists() else []
    if ckpts:
        print("Existing checkpoint found. Skipping retraining:")
        for p in ckpts:
            print(" ", p, f"{p.stat().st_size / 1024**2:.2f} MiB")
        return

    cmd = [
        sys.executable,
        "simple_trainer.py",
        "default",
        "--eval_steps", "7000",
        "--save_steps", "7000",
        "--disable_viewer",
        "--disable_video",
        "--data_factor", "2",
        "--data_dir", "data/360_v2/bonsai",
        "--result_dir", "results/splatstream_bonsai_7k",
        "--max_steps", "7000",
    ]

    print(
        "The first CUDA call may spend several minutes JIT-compiling gsplat kernels. "
        "That is expected."
    )
    start = time.time()
    run_live(
        cmd,
        cwd=EXAMPLES,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        log_path=LOG,
    )
    print(f"Training/evaluation wall time: {time.time() - start:.1f} s")

    ckpts = sorted(RESULT.rglob("*.pt"), key=lambda p: p.stat().st_mtime) if RESULT.exists() else []
    if not ckpts:
        raise RuntimeError(
            f"Training returned but no checkpoint exists under {RESULT}. "
            f"Full log: {LOG}"
        )


def setup_splatstream():
    print("\n[4/6] Cloning SplatStream and exporting checkpoint to PLY")
    if not SPLATSTREAM.exists():
        run(["git", "clone", "https://github.com/reusahn/splatstream-lab.git", SPLATSTREAM])
    else:
        run(["git", "pull", "--ff-only"], cwd=SPLATSTREAM)

    exporter = SPLATSTREAM / "tools/export_ply_from_checkpoint.py"
    run([sys.executable, exporter, RESULT])

    plys = sorted(RESULT.rglob("*.ply"), key=lambda p: p.stat().st_size)
    if not plys:
        raise RuntimeError("Checkpoint export completed but no PLY was found.")

    ply = plys[-1]
    print("PLY:", ply)
    print("PLY size:", f"{ply.stat().st_size / 1024**2:.2f} MiB")
    return ply


def portable_sanity(ply: Path):
    print("\n[5/6] Running real-PLY portable compression sanity check")
    run([sys.executable, "-m", "pip", "install", "-e", SPLATSTREAM])
    run([sys.executable, "-m", "splatstream.inspect_ply", ply], cwd=SPLATSTREAM)
    run([
        sys.executable,
        "-m",
        "splatstream.real_ply_experiment",
        ply,
        "--output", PORTABLE_OUT,
        "--max-gaussians", "10000",
        "--width", "192",
        "--height", "192",
    ], cwd=SPLATSTREAM)

    bench = PORTABLE_OUT / "benchmark.csv"
    if bench.exists():
        print("\nPortable benchmark:\n")
        print(bench.read_text()[:5000])


def package_evidence(ply: Path):
    print("\n[6/6] Packaging measured evidence")
    bundle = Path("/content/netflix_gs_evidence")
    zipbase = Path("/content/netflix_gs_evidence")

    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    manifest = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "gsplat_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=GSPLAT_DIR, text=True
        ).strip(),
        "splatstream_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SPLATSTREAM, text=True
        ).strip(),
        "ply_path": str(ply),
        "ply_bytes": ply.stat().st_size,
    }
    (bundle / "environment.json").write_text(json.dumps(manifest, indent=2))

    if LOG.exists():
        shutil.copy2(LOG, bundle / LOG.name)

    for p in RESULT.rglob("*.json"):
        dst = bundle / "gsplat_results" / p.relative_to(RESULT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    ckpts = sorted(RESULT.rglob("*.pt"))
    (bundle / "checkpoint_inventory.txt").write_text(
        "\n".join(f"{p}\t{p.stat().st_size}" for p in ckpts)
    )
    (bundle / "ply_inventory.txt").write_text(
        f"{ply}\t{ply.stat().st_size}\n"
    )

    # Do not duplicate a potentially huge real-scene PLY into the upload bundle.
    if PORTABLE_OUT.exists():
        shutil.copytree(PORTABLE_OUT, bundle / "portable_sanity", dirs_exist_ok=True)

    archive = shutil.make_archive(str(zipbase), "zip", root_dir=bundle)
    print("\nSUCCESS")
    print("Evidence bundle:", archive)
    print("Evidence ZIP size:", f"{Path(archive).stat().st_size / 1024**2:.2f} MiB")
    print("Real PLY remains in Colab at:", ply)

    try:
        from google.colab import files
        files.download(archive)
    except Exception:
        print("Automatic download did not start. Download manually from:", archive)


def main():
    check_environment()
    setup_gsplat()
    prepare_dataset()
    train()
    ply = setup_splatstream()
    portable_sanity(ply)
    package_evidence(ply)


if __name__ == "__main__":
    main()
