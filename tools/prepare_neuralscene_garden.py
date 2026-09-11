from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import fsspec

SOURCE_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
DATA_ROOT = Path("/content/data/mipnerf360/garden")
SCENE_PREFIX = "garden/"


def main() -> None:
    print("=== NeuralScene Bench: prepare Mip-NeRF 360 Garden ===")
    print("Source:", SOURCE_URL)
    print("Target:", DATA_ROOT)
    print("Mode: HTTP range extraction, only garden/images_2 and garden/sparse")

    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    wanted_prefixes = (
        f"{SCENE_PREFIX}images_2/",
        f"{SCENE_PREFIX}sparse/",
    )

    fs = fsspec.filesystem("http")
    with fs.open(SOURCE_URL, "rb", block_size=8 * 1024 * 1024) as remote:
        with zipfile.ZipFile(remote) as zf:
            members = [m for m in zf.namelist() if m.startswith(wanted_prefixes)]
            if not members:
                raise RuntimeError("Garden images_2/sparse members were not found in source archive.")

            print("Selected archive members:", len(members))
            for idx, member in enumerate(members, start=1):
                rel = Path(member).relative_to(SCENE_PREFIX)
                target = DATA_ROOT / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                if idx % 50 == 0:
                    print(f"Extracted {idx}/{len(members)} members")

    images = sorted(p for p in (DATA_ROOT / "images_2").glob("*") if p.is_file())
    sparse = DATA_ROOT / "sparse" / "0"
    colmap_files = sorted(p.name for p in sparse.glob("*") if p.is_file())

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
