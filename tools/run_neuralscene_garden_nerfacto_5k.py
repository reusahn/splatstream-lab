from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

VENV = Path('/content/neuralscene-env')
NS_TRAIN = VENV / 'bin/ns-train'
DATA = Path('/content/data/mipnerf360/garden')
OUT = Path('/content/outputs/neuralscene_bench_garden_nerfacto')
LOG = OUT / 'nerfacto_5k_console.log'


def main() -> int:
    if not NS_TRAIN.exists():
        raise FileNotFoundError(f'Missing ns-train: {NS_TRAIN}')
    if not (DATA / 'images_2').is_dir():
        raise FileNotFoundError(f'Missing Garden images_2: {DATA / "images_2"}')
    if not (DATA / 'sparse/0').is_dir():
        raise FileNotFoundError(f'Missing Garden COLMAP model: {DATA / "sparse/0"}')

    OUT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PATH'] = f"{VENV / 'bin'}:" + env.get('PATH', '')
    env['TCNN_CUDA_ARCHITECTURES'] = '80'
    env['TORCH_CUDA_ARCH_LIST'] = '8.0'

    # Match the Bonsai Nerfacto benchmark protocol and the Garden Splatfacto
    # parser/eval conditions. Camera optimization is explicitly disabled so the
    # two representations use the same camera treatment.
    cmd = [
        str(NS_TRAIN),
        'nerfacto',
        '--output-dir', str(OUT),
        '--experiment-name', 'garden-bench',
        '--timestamp', 'nerfacto-5k',
        '--max-num-iterations', '5000',
        '--steps-per-save', '1000',
        '--vis', 'tensorboard',
        '--pipeline.model.camera-optimizer.mode', 'off',
        '--pipeline.model.implementation', 'tcnn',
        'colmap',
        '--data', str(DATA),
        '--images-path', 'images',
        '--colmap-path', 'sparse/0',
        '--downscale-factor', '2',
        '--downscale-rounding-mode', 'ceil',
        '--eval-mode', 'interval',
        '--eval-interval', '8',
    ]

    print('=== NeuralScene Bench: Garden / Nerfacto TCNN / 5K ===', flush=True)
    print('$ ' + ' '.join(cmd), flush=True)
    print(f'Console log: {LOG}', flush=True)

    start = time.perf_counter()
    with LOG.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()

    elapsed = time.perf_counter() - start
    print(f'\nBENCHMARK RETURN CODE: {rc}', flush=True)
    print(f'END-TO-END WALL SECONDS: {elapsed:.3f}', flush=True)
    print(f'LOG: {LOG}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
