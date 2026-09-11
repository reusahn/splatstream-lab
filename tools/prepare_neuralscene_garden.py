from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import fsspec

SOURCE_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
DATA_ROOT = Path("/content/data/mipnerf360/garden")
TEMP_ZIP = Path("/content/mipnerf360_garden_source.zip")
SCENE_PREFIX = "garden/"


def main() -> None:
    print("=== NeuralScene Bench: prepare Mip-NeRF 360 Garden ===")
    print("Source:", SOURCE_URL)
    print("Target:", DATA_ROOT)

    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if TEMP_ZIP.exists():
        TEMP_ZIP.unlink()

    fs = fsspec.filesystem("http")
    with fs.open(SOURCE_URL, "rb") as src, open(TEMP_ZIP, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

    wanted_prefixes = (
        f"{SCENE_PREFIX}images_2/",
        f"{SCENE_PREFIX}sparse/",
    )

    with zipfile.ZipFile(TEMP_ZIP) as zf:
        members = [m for m in zf.namelist() if m.startswith(wanted_prefixes)]
        if not members:
            raise RuntimeError("Garden images_2/sparse members were not found in source archive.")

        for member in members:
            rel = Path(member).relative_to(SCENE_PREFIX)
            target = DATA_ROOT / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    images = sorted((DATA_ROOT / "images_2").glob("*"))
    sparse = DATA_ROOT / "sparse" / "0"
    colmap_files = [p.name for p in sparse.glob("*") if p.is_file()]

    print("Garden root:", DATA_ROOT)
    print("images_2 exists:", (DATA_ROOT / "images_2").exists())
    print("Image count:", len(images))
    print("sparse/0 exists:", sparse.exists())
    print("COLMAP files:", colmap_files)

    required = {"cameras.bin", "images.bin", "points3D.bin"}
    if not (DATA_ROOT / "images_2").exists() or not images:
        raise RuntimeError("Garden images_2 extraction failed.")
    if not sparse.exists() or not required.issubset(set(colmap_files)):
        raise RuntimeError("Garden COLMAP sparse model is incomplete.")

    print("GARDEN DATA READY")


if __name__ == "__main__":
    main()
