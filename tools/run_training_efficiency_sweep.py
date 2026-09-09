from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

GSPLAT_COMMIT = "28e794ca44a4c25ffc39175370c5ee7b38bfcc36"


def run_live(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> float:
    print("\n$", " ".join(cmd), flush=True)
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
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
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return time.time() - started


def merged_env() -> dict[str, str]:
    env = os.environ.copy()
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MAX_JOBS": "2",
                "CMAKE_BUILD_PARALLEL_LEVEL": "2",
                "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}",
                "BUILD_EXPERIMENTAL": "0",
            }
        )
    return env


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def find_metric(obj: Any, names: set[str]) -> float | int | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in names and isinstance(value, (int, float)):
                return value
        for value in obj.values():
            found = find_metric(value, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_metric(value, names)
            if found is not None:
                return found
    return None


def newest(path: Path, pattern: str) -> Path | None:
    items = list(path.rglob(pattern)) if path.exists() else []
    if not items:
        return None
    return max(items, key=lambda p: p.stat().st_mtime)


def summarize_result(result_dir: Path, wall_seconds: float) -> dict[str, Any]:
    val_json = newest(result_dir / "stats", "val_step*.json")
    train_json = newest(result_dir / "stats", "train_step*_rank0.json")
    ckpt = newest(result_dir, "*.pt")

    val = load_json(val_json) if val_json else {}
    train = load_json(train_json) if train_json else {}

    summary: dict[str, Any] = {
        "wall_seconds_external": wall_seconds,
        "validation_stats_file": str(val_json) if val_json else None,
        "training_stats_file": str(train_json) if train_json else None,
        "checkpoint": str(ckpt) if ckpt else None,
        "checkpoint_bytes": ckpt.stat().st_size if ckpt else None,
        "psnr": find_metric(val, {"psnr", "psnr_db"}),
        "ssim": find_metric(val, {"ssim"}),
        "lpips": find_metric(val, {"lpips"}),
        "seconds_per_image": find_metric(val, {"ellipse_time", "seconds_per_image", "render_time"}),
        "train_loop_seconds": find_metric(train, {"ellipse_time", "elapsed", "elapsed_seconds", "train_time"}),
        "peak_memory_bytes": find_metric(train, {"mem", "memory", "max_memory_allocated", "peak_memory"}),
    }

    if ckpt:
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location="cpu")
        splats = data.get("splats", {}) if isinstance(data, dict) else {}
        means = splats.get("means") if isinstance(splats, dict) else None
        if means is not None:
            summary["gaussians"] = int(means.shape[0])
    return summary


def build_command(preset: dict[str, Any], *, examples: Path, data_dir: Path, result_dir: Path) -> list[str]:
    steps = int(preset["max_steps"])
    cmd = [
        sys.executable,
        "simple_trainer.py",
        "default",
        "--disable_viewer",
        "--disable_video",
        "--data_factor",
        "2",
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(result_dir),
        "--max_steps",
        str(steps),
        "--eval_steps",
        str(steps),
        "--save_steps",
        str(steps),
        "--strategy.refine-stop-iter",
        str(int(preset["refine_stop_iter"])),
        "--strategy.refine-every",
        str(int(preset["refine_every"])),
        "--sh_degree_interval",
        str(int(preset["sh_degree_interval"])),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled gsplat training-time / quality sweep on one scene."
    )
    parser.add_argument("--gsplat-dir", type=Path, default=Path("/content/gsplat"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/gsplat/examples/data/360_v2/bonsai"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_efficiency_sweep.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/gsplat/examples/results/splatstream_training_efficiency"),
    )
    parser.add_argument("--only", nargs="*", default=None, help="Optional preset names to run")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA is required for the real sweep. Use --dry-run only for command inspection.")

    gsplat_dir = args.gsplat_dir.resolve()
    examples = gsplat_dir / "examples"
    trainer = examples / "simple_trainer.py"
    if not trainer.exists():
        raise FileNotFoundError(f"gsplat trainer not found: {trainer}")

    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=gsplat_dir, text=True).strip()
    if actual_commit != GSPLAT_COMMIT:
        raise RuntimeError(
            f"Expected pinned gsplat commit {GSPLAT_COMMIT}, found {actual_commit}. "
            "Use the same revision as the verified SplatStream experiment."
        )

    config = load_json(args.config)
    presets = config["presets"]
    if args.only:
        wanted = set(args.only)
        presets = [p for p in presets if p["name"] in wanted]
        missing = wanted - {p["name"] for p in presets}
        if missing:
            raise ValueError(f"Unknown preset(s): {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "study": config["study"],
        "dataset": config["dataset"],
        "gsplat_commit": actual_commit,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "presets": [],
    }

    env = merged_env()
    for preset in presets:
        name = preset["name"]
        result_dir = args.output_dir / name
        log_path = args.output_dir / f"{name}.log"
        cmd = build_command(preset, examples=examples, data_dir=args.data_dir, result_dir=result_dir)

        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)
        print("Command:", " ".join(cmd))

        if args.dry_run:
            manifest["presets"].append({"preset": preset, "command": cmd, "dry_run": True})
            continue

        if result_dir.exists() and args.overwrite:
            shutil.rmtree(result_dir)
        if result_dir.exists() and newest(result_dir, "*.pt") is not None:
            print("Existing checkpoint found; summarizing without rerunning.")
            wall_seconds = 0.0
        else:
            wall_seconds = run_live(cmd, cwd=examples, env=env, log_path=log_path)

        summary = summarize_result(result_dir, wall_seconds)
        summary["preset"] = preset
        summary["command"] = cmd
        manifest["presets"].append(summary)
        (args.output_dir / "training_efficiency_results.json").write_text(
            json.dumps(manifest, indent=2)
        )

    out = args.output_dir / "training_efficiency_results.json"
    out.write_text(json.dumps(manifest, indent=2))
    print("\nWrote:", out)
    print("Important: do not claim a speed/quality improvement until repeated runs are complete.")


if __name__ == "__main__":
    main()
