from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch


def run(cmd, *, cwd=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=None if cwd is None else str(cwd), check=True)


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Colab choose Runtime > Change runtime type > GPU.")
    print("GPU:", torch.cuda.get_device_name(0))

    gsplat = Path("/content/gsplat")
    examples = gsplat / "examples"
    bonsai = examples / "data/360_v2/bonsai"
    result = examples / "results/splatstream_bonsai_7k"

    if not gsplat.exists():
        run(["git", "clone", "https://github.com/nerfstudio-project/gsplat.git", gsplat])
    else:
        run(["git", "-C", gsplat, "pull", "--ff-only"])

    run([sys.executable, "-m", "pip", "install", "-e", gsplat])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        examples / "requirements.txt",
        "--no-build-isolation",
    ])

    if not bonsai.exists():
        run([sys.executable, examples / "datasets/download_dataset.py"], cwd=examples)
    if not bonsai.exists():
        raise RuntimeError(f"Dataset download completed but Bonsai was not found at {bonsai}")

    print("\nBonsai dataset ready:", bonsai)

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
        "results/splatstream_bonsai_7k",
        "--max_steps",
        "7000",
        "--eval_steps",
        "7000",
        "--save_steps",
        "7000",
        "--save_ply",
        "--ply_steps",
        "7000",
    ]

    print("\nStarting 7k Bonsai training. This is the long step.")
    start = time.time()
    run(cmd, cwd=examples)
    elapsed = time.time() - start
    print(f"\nTraining/evaluation command completed in {elapsed:.1f} seconds")

    if not result.exists():
        raise RuntimeError(f"Training returned success but result directory does not exist: {result}")

    ckpts = sorted(result.rglob("*.pt"))
    plys = sorted(result.rglob("*.ply"), key=lambda p: p.stat().st_size)
    jsons = sorted(result.rglob("*.json"))

    print("\nCHECKPOINTS")
    for p in ckpts:
        print(" ", p, f"{p.stat().st_size / 1024**2:.2f} MiB")

    print("\nPLY FILES")
    for p in plys:
        print(" ", p, f"{p.stat().st_size / 1024**2:.2f} MiB")

    print("\nJSON / METRIC FILES")
    for p in jsons:
        print(" ", p)

    if not plys:
        raise RuntimeError(
            "The result directory was created but no PLY was exported. "
            "Send the full training output to ChatGPT."
        )

    print("\nSUCCESS")
    print("Result directory:", result)
    print("Largest PLY:", plys[-1])
    print("Largest PLY size (MiB):", plys[-1].stat().st_size / 1024**2)


if __name__ == "__main__":
    main()
