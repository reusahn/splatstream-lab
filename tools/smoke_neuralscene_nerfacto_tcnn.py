from __future__ import annotations

import os
import subprocess
from pathlib import Path

VENV = Path('/content/neuralscene-env')
PY = VENV / 'bin/python'
NS_TRAIN = VENV / 'bin/ns-train'
DATA = Path('/content/data/mipnerf360/bonsai')
OUT = Path('/content/outputs/neuralscene_bench_smoke_nerfacto')


def run(cmd, env):
    print('\n$', ' '.join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(proc.stdout)
    print('RETURN CODE:', proc.returncode)
    return proc.returncode


def main() -> int:
    env = os.environ.copy()
    env['PATH'] = f"{VENV / 'bin'}:" + env.get('PATH', '')
    env.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    env.setdefault('TCNN_CUDA_ARCHITECTURES', '80')

    print('=== NeuralScene Bench: Nerfacto / TCNN smoke test ===')
    print('Camera optimizer: off')
    print('Dataset protocol: Bonsai, images_2 via downscale-factor 2, interval-8 eval split')

    if not PY.exists() or not NS_TRAIN.exists():
        raise FileNotFoundError('NeuralScene venv or ns-train is missing.')

    print('\n=== tiny-cuda-nn availability ===')
    tcnn_rc = run([
        PY,
        '-c',
        "import tinycudann as tcnn; print('tinycudann import OK'); print(tcnn)",
    ], env)

    if tcnn_rc != 0:
        print('\nTCNN_NOT_AVAILABLE')
        print('Do not run the 5K benchmark yet. Install/repair tiny-cuda-nn first, or explicitly switch to the torch backend and label timing as backend-specific.')
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        NS_TRAIN,
        'nerfacto',
        '--output-dir', OUT,
        '--experiment-name', 'neuralscene-bench-debug',
        '--max-num-iterations', '2',
        '--vis', 'tensorboard',
        '--pipeline.model.camera-optimizer.mode', 'off',
        '--pipeline.model.implementation', 'tcnn',
        'colmap',
        '--data', DATA,
        '--images-path', 'images',
        '--colmap-path', 'sparse/0',
        '--downscale-factor', '2',
        '--downscale-rounding-mode', 'ceil',
        '--eval-mode', 'interval',
        '--eval-interval', '8',
    ]

    print('\n=== Nerfacto 2-step training ===')
    rc = run(cmd, env)
    print('\nNERFACTO SMOKE RETURN CODE:', rc)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
