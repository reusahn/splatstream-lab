from __future__ import annotations
import json
from pathlib import Path

def write_training_record(
    path,
    scene: str,
    iterations: int,
    wall_seconds: float,
    final_gaussians: int,
    gpu: str,
    framework: str,
    notes: str = "",
):
    """
    Store measured training/encoding information from an actual CUDA training run.
    This intentionally does not fabricate training-time results on a CPU-only machine.
    """
    record = {
        "scene": scene,
        "iterations": int(iterations),
        "wall_seconds": float(wall_seconds),
        "final_gaussians": int(final_gaussians),
        "gpu": gpu,
        "framework": framework,
        "notes": notes,
    }
    Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
