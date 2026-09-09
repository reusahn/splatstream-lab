from __future__ import annotations

import os
import runpy
import subprocess
import sys
import urllib.request
from pathlib import Path

GSPLAT_ROOT = Path('/content/gsplat')
GSPLAT_PACKAGE = GSPLAT_ROOT / 'gsplat' / '__init__.py'
EXAMPLES_ROOT = GSPLAT_ROOT / 'examples'
BASE_RESULT = EXAMPLES_ROOT / 'results' / 'splatstream_bonsai_7k'
BASELINE_RUNNER_URL = 'https://raw.githubusercontent.com/reusahn/splatstream-lab/main/tools/run_bonsai_cuda.py'
RD_RUNNER_URL = 'https://raw.githubusercontent.com/reusahn/splatstream-lab/main/tools/full_cuda_rate_distortion_v2.py'
RD_ZIP = Path('/content/splatstream_full_cuda_rd_v2.zip')


def checkpoints() -> list[Path]:
    if not BASE_RESULT.exists():
        return []
    return sorted(BASE_RESULT.rglob('*.pt'), key=lambda p: p.stat().st_mtime)


def ensure_baseline() -> None:
    have_source = GSPLAT_PACKAGE.exists()
    have_ckpt = bool(checkpoints())
    print('Pinned gsplat source present:', have_source)
    print('7K checkpoint present:', have_ckpt)

    if have_source and have_ckpt:
        print('Reusing the existing baseline runtime state.')
        return

    print('\nBaseline runtime state is incomplete. Rebuilding it now.')
    print('Colab /content is ephemeral, so a reconnected runtime may require this step again.')
    runner = Path('/content/run_bonsai_cuda.py')
    urllib.request.urlretrieve(BASELINE_RUNNER_URL, runner)
    runpy.run_path(str(runner), run_name='__main__')

    if not GSPLAT_PACKAGE.exists():
        raise RuntimeError(f'Baseline setup finished but source package is missing: {GSPLAT_PACKAGE}')
    if not checkpoints():
        raise RuntimeError(f'Baseline setup finished but no checkpoint exists under: {BASE_RESULT}')


def verify_source_files() -> None:
    color_correct = GSPLAT_ROOT / 'gsplat' / 'color_correct.py'
    bonsai_sparse = EXAMPLES_ROOT / 'data' / '360_v2' / 'bonsai' / 'sparse'
    if not GSPLAT_PACKAGE.exists():
        raise FileNotFoundError(GSPLAT_PACKAGE)
    if not color_correct.exists():
        raise FileNotFoundError(color_correct)
    if not bonsai_sparse.exists():
        raise FileNotFoundError(bonsai_sparse)

    print('\nPinned source files verified:')
    print('  gsplat:', GSPLAT_PACKAGE)
    print('  color_correct:', color_correct)
    print('  checkpoint:', checkpoints()[-1])
    print('  Bonsai COLMAP:', bonsai_sparse)


def run_rate_distortion_fresh_process() -> None:
    runner = Path('/content/full_cuda_rate_distortion_v2.py')
    urllib.request.urlretrieve(RD_RUNNER_URL, runner)

    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    source_path = os.pathsep.join([str(GSPLAT_ROOT), str(EXAMPLES_ROOT)])
    env['PYTHONPATH'] = source_path + (os.pathsep + existing_pythonpath if existing_pythonpath else '')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env.setdefault('MAX_JOBS', '2')

    print('\nLaunching rate-distortion sweep in a fresh Python interpreter.')
    print('This avoids re-registering gsplat torch.library kernels in the notebook process.')
    print('Rate-distortion runner:', runner)
    print('PYTHONPATH head:', source_path)

    subprocess.run(
        [sys.executable, str(runner)],
        cwd=str(EXAMPLES_ROOT),
        env=env,
        check=True,
    )

    if not RD_ZIP.exists():
        raise RuntimeError(f'Rate-distortion process finished but ZIP is missing: {RD_ZIP}')

    print('\nFull CUDA rate-distortion bundle ready:', RD_ZIP)
    print('ZIP size:', f'{RD_ZIP.stat().st_size / 1024**2:.2f} MiB')
    try:
        from google.colab import files
        files.download(str(RD_ZIP))
    except Exception:
        print('Download manually from:', RD_ZIP)


def main() -> None:
    ensure_baseline()
    verify_source_files()
    run_rate_distortion_fresh_process()


if __name__ == '__main__':
    main()
