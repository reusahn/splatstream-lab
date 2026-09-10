from __future__ import annotations

import os
import subprocess
from pathlib import Path

VENV = Path('/content/neuralscene-env')
PY = VENV / 'bin/python'
TCNN_COMMIT = '749dd70c5afc5a9dadb85e5652ed65d55e0ba187'
TCNN_SPEC = (
    'git+https://github.com/NVlabs/tiny-cuda-nn.git@'
    + TCNN_COMMIT
    + '#subdirectory=bindings/torch'
)
SETUPTOOLS_VERSION = '81.0.0'


def run(cmd, env):
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    p = subprocess.Popen(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end='', flush=True)
    return p.wait()


def main() -> int:
    if not PY.exists():
        raise FileNotFoundError(f'Missing benchmark Python: {PY}')

    env = os.environ.copy()
    env['PATH'] = f"{VENV / 'bin'}:" + env.get('PATH', '')
    env['TCNN_CUDA_ARCHITECTURES'] = '80'
    env['TORCH_CUDA_ARCH_LIST'] = '8.0'
    env.setdefault('MAX_JOBS', '4')

    print('=== NeuralScene Bench: install tiny-cuda-nn ===', flush=True)
    print('Pinned tiny-cuda-nn commit:', TCNN_COMMIT, flush=True)
    print('Target GPU architecture: A100 / sm_80', flush=True)
    print('Build isolation: disabled because tiny-cuda-nn setup.py imports torch during build metadata generation', flush=True)
    print('Setuptools pinned to:', SETUPTOOLS_VERSION, 'because tiny-cuda-nn setup.py imports pkg_resources', flush=True)

    # tiny-cuda-nn setup.py imports `pkg_resources.parse_version`.
    # Setuptools 82+ removed pkg_resources, so use the last compatible major line
    # in this isolated benchmark venv for building the extension.
    rc = run([
        PY,
        '-m',
        'pip',
        'install',
        '--upgrade',
        'ninja',
        f'setuptools=={SETUPTOOLS_VERSION}',
        'wheel',
        'packaging',
    ], env)
    if rc != 0:
        print('Dependency setup failed.')
        return rc

    rc = run([
        PY,
        '-c',
        "import setuptools, pkg_resources, torch; "
        "print('setuptools', setuptools.__version__); "
        "print('pkg_resources import OK'); "
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)",
    ], env)
    if rc != 0:
        print('Build preflight failed.')
        return rc

    rc = run([
        PY,
        '-m',
        'pip',
        'install',
        '--no-build-isolation',
        '--no-cache-dir',
        TCNN_SPEC,
    ], env)
    if rc != 0:
        print('TCNN build/install failed.')
        return rc

    rc = run([
        PY,
        '-c',
        "import torch, tinycudann as tcnn; "
        "print('tinycudann import OK'); "
        "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
        "print('gpu', torch.cuda.get_device_name(0)); "
        "print('tcnn module', tcnn)",
    ], env)

    print('\nTCNN INSTALL RETURN CODE:', rc, flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
