from __future__ import annotations

import importlib
import runpy
import sys
import urllib.request
from pathlib import Path

GSPLAT_ROOT = Path('/content/gsplat')
GSPLAT_PACKAGE = GSPLAT_ROOT / 'gsplat' / '__init__.py'
EXAMPLES_ROOT = GSPLAT_ROOT / 'examples'
BASE_RESULT = EXAMPLES_ROOT / 'results' / 'splatstream_bonsai_7k'
BASELINE_RUNNER_URL = 'https://raw.githubusercontent.com/reusahn/splatstream-lab/main/tools/run_bonsai_cuda.py'
RD_RUNNER_URL = 'https://raw.githubusercontent.com/reusahn/splatstream-lab/main/tools/full_cuda_rate_distortion_v2.py'


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


def bind_pinned_source() -> None:
    if not GSPLAT_PACKAGE.exists():
        raise FileNotFoundError(f'Pinned gsplat source is missing: {GSPLAT_PACKAGE}')

    # Prefer the exact source checkout used for the baseline over any package
    # that may already be installed in the notebook kernel.
    for p in (str(EXAMPLES_ROOT), str(GSPLAT_ROOT)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(GSPLAT_ROOT))
    sys.path.insert(1, str(EXAMPLES_ROOT))

    # Clear stale imports, including the common Hugging Face `datasets` name
    # collision with gsplat/examples/datasets.
    for prefix in ('gsplat', 'datasets'):
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                del sys.modules[name]
    importlib.invalidate_caches()

    import gsplat
    import gsplat.color_correct

    package_file = Path(gsplat.__file__).resolve()
    color_file = Path(gsplat.color_correct.__file__).resolve()
    if GSPLAT_ROOT not in package_file.parents:
        raise RuntimeError(f'Wrong gsplat package resolved: {package_file}')

    print('\nPinned source binding verified:')
    print('  gsplat:', package_file)
    print('  color_correct:', color_file)
    print('  checkpoint:', checkpoints()[-1])


def run_rate_distortion() -> None:
    runner = Path('/content/full_cuda_rate_distortion_v2.py')
    urllib.request.urlretrieve(RD_RUNNER_URL, runner)
    print('\nRate-distortion runner:', runner)
    runpy.run_path(str(runner), run_name='__main__')


def main() -> None:
    ensure_baseline()
    bind_pinned_source()
    run_rate_distortion()


if __name__ == '__main__':
    main()
