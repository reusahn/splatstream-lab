from __future__ import annotations

import argparse
from pathlib import Path
import torch
from gsplat import export_splats


def newest_checkpoint(result_dir: Path) -> Path:
    ckpts = sorted(result_dir.rglob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f'No checkpoint was found under {result_dir}. '
            'The training command likely failed before checkpoint export.'
        )
    return ckpts[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    ckpt_path = newest_checkpoint(result_dir)
    print(f'Using checkpoint: {ckpt_path}')

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'splats' not in ckpt:
        raise KeyError(f"Checkpoint does not contain 'splats'. Keys: {list(ckpt.keys())}")

    splats = ckpt['splats']
    required = ['means', 'scales', 'quats', 'opacities', 'sh0', 'shN']
    missing = [k for k in required if k not in splats]
    if missing:
        raise KeyError(f'Missing splat parameters: {missing}. Available: {list(splats.keys())}')

    out = args.output
    if out is None:
        step = ckpt.get('step', 'recovered')
        out = result_dir / 'ply' / f'point_cloud_{step}.ply'
    out.parent.mkdir(parents=True, exist_ok=True)

    export_splats(
        means=splats['means'],
        scales=splats['scales'],
        quats=splats['quats'],
        opacities=splats['opacities'],
        sh0=splats['sh0'],
        shN=splats['shN'],
        format='ply',
        save_to=str(out),
    )

    print(f'PLY written to: {out}')
    print(f'PLY size: {out.stat().st_size / (1024**2):.2f} MiB')


if __name__ == '__main__':
    main()
