from __future__ import annotations

import os
import subprocess
from pathlib import Path

VENV = Path('/content/neuralscene-env')
NS_TRAIN = VENV / 'bin/ns-train'
PY = VENV / 'bin/python'
DATA = Path('/content/data/mipnerf360/bonsai')
OUT = Path('/content/outputs/neuralscene_bench_smoke_debug')


def run_capture(cmd, env=None):
    print('\n$', ' '.join(map(str, cmd)), flush=True)
    p = subprocess.run(
        list(map(str, cmd)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(p.stdout)
    print('RETURN CODE:', p.returncode)
    return p.returncode


def main():
    env = os.environ.copy()
    env['PATH'] = f"{VENV / 'bin'}:" + env.get('PATH', '')

    print('=== Environment ===')
    run_capture([
        PY,
        '-c',
        "import torch, gsplat, nerfstudio; "
        "from PIL import __version__ as pillow_version; "
        "print('torch', torch.__version__); "
        "print('torch cuda', torch.version.cuda); "
        "print('cuda available', torch.cuda.is_available()); "
        "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
        "print('gsplat', getattr(gsplat, '__version__', 'unknown')); "
        "print('Pillow', pillow_version); "
        "print('nerfstudio', getattr(nerfstudio, '__version__', 'source-checkout'))"
    ], env=env)

    print('\n=== CUDA compiler ===')
    run_capture(['bash', '-lc', 'which nvcc || true; nvcc --version || true'], env=env)

    print('\n=== Splatfacto CLI smoke test ===')
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        NS_TRAIN,
        'splatfacto',
        '--output-dir', OUT,
        '--experiment-name', 'neuralscene-bench-debug',
        '--max-num-iterations', '2',
        '--vis', 'tensorboard',
        'colmap',
        '--data', DATA,
        '--images-path', 'images',
        '--colmap-path', 'sparse/0',
        '--downscale-factor', '2',
        '--downscale-rounding-mode', 'ceil',
        '--eval-mode', 'interval',
        '--eval-interval', '8',
    ]
    rc = run_capture(cmd, env=env)
    print('\nDIAGNOSTIC COMPLETE')
    print('Smoke-test return code:', rc)


if __name__ == '__main__':
    main()
