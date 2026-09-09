from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

VERSION = "v3-fresh-process-2026-09-09"
GSPLAT_ROOT = Path('/content/gsplat')
EXAMPLES_ROOT = GSPLAT_ROOT / 'examples'
BASE_RESULT = EXAMPLES_ROOT / 'results' / 'splatstream_bonsai_7k'
DATA_DIR = EXAMPLES_ROOT / 'data' / '360_v2' / 'bonsai'
RD_RUNNER = Path('/content/full_cuda_rate_distortion_v2_pinned.py')
RD_ZIP = Path('/content/splatstream_full_cuda_rd_v2.zip')
RD_COMMIT = '65063a51a5792ea56cb4c8bb51e72a1f49a098b5'
RD_URL = f'https://raw.githubusercontent.com/reusahn/splatstream-lab/{RD_COMMIT}/tools/full_cuda_rate_distortion_v2.py'


def newest_checkpoint() -> Path:
    ckpts = sorted(BASE_RESULT.rglob('*.pt'), key=lambda p: p.stat().st_mtime) if BASE_RESULT.exists() else []
    if not ckpts:
        raise FileNotFoundError(f'No checkpoint under {BASE_RESULT}')
    return ckpts[-1]


def verify() -> Path:
    print('PIPELINE VERSION:', VERSION, flush=True)
    package = GSPLAT_ROOT / 'gsplat' / '__init__.py'
    color = GSPLAT_ROOT / 'gsplat' / 'color_correct.py'
    sparse = DATA_DIR / 'sparse'
    for p in (package, color, sparse):
        if not p.exists():
            raise FileNotFoundError(p)
    ckpt = newest_checkpoint()
    print('Verified source:', package, flush=True)
    print('Verified color_correct:', color, flush=True)
    print('Verified checkpoint:', ckpt, flush=True)
    print('Verified Bonsai COLMAP:', sparse, flush=True)
    return ckpt


def main() -> None:
    verify()

    # Download an exact commit, not main, to avoid stale CDN/notebook copies.
    urllib.request.urlretrieve(RD_URL, RD_RUNNER)
    print('Pinned RD runner commit:', RD_COMMIT, flush=True)
    print('Pinned RD runner:', RD_RUNNER, flush=True)

    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    source_path = os.pathsep.join([str(GSPLAT_ROOT), str(EXAMPLES_ROOT)])
    env['PYTHONPATH'] = source_path + (os.pathsep + existing if existing else '')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env.setdefault('MAX_JOBS', '2')
    env['PYTHONUNBUFFERED'] = '1'

    print('\nLaunching ONE brand-new Python interpreter for the complete RD sweep.', flush=True)
    print('The notebook process will not import or reload gsplat.', flush=True)
    print('Command:', sys.executable, '-u', RD_RUNNER, flush=True)

    subprocess.run(
        [sys.executable, '-u', str(RD_RUNNER)],
        cwd=str(EXAMPLES_ROOT),
        env=env,
        check=True,
    )

    if not RD_ZIP.exists():
        raise RuntimeError(f'RD process exited successfully but ZIP is missing: {RD_ZIP}')

    print('\nFINAL ZIP:', RD_ZIP, flush=True)
    print('ZIP size:', f'{RD_ZIP.stat().st_size / 1024**2:.2f} MiB', flush=True)
    try:
        from google.colab import files
        files.download(str(RD_ZIP))
    except Exception:
        print('Download manually:', RD_ZIP, flush=True)


if __name__ == '__main__':
    main()
