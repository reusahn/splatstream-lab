from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

VENV = Path('/content/neuralscene-env')
NS_EVAL = VENV / 'bin/ns-eval'
ROOT = Path('/content/outputs/neuralscene_bench_garden/bonsai-bench/splatfacto')
OUT = Path('/content/outputs/neuralscene_bench_garden/splatfacto_5k_eval.json')
RENDERS = Path('/content/outputs/neuralscene_bench_garden/splatfacto_5k_eval_renders')


def main() -> int:
    if not NS_EVAL.exists():
        raise FileNotFoundError(f'Missing ns-eval: {NS_EVAL}')
    if not ROOT.exists():
        raise FileNotFoundError(f'Missing Garden Splatfacto output root: {ROOT}')

    candidates = []
    for cfg in ROOT.glob('*/config.yml'):
        model_dir = cfg.parent / 'nerfstudio_models'
        ckpts = sorted(model_dir.glob('*.ckpt')) if model_dir.exists() else []
        if ckpts:
            candidates.append((max(p.stat().st_mtime for p in ckpts), cfg, ckpts[-1]))

    if not candidates:
        raise RuntimeError('No completed Garden Splatfacto run with a checkpoint was found.')

    _, config, checkpoint = sorted(candidates, key=lambda x: x[0])[-1]
    print('Using config:', config)
    print('Using checkpoint:', checkpoint)
    print('Writing metrics to:', OUT)
    print('Writing held-out renders to:', RENDERS)

    env = os.environ.copy()
    env['PATH'] = f"{VENV / 'bin'}:" + env.get('PATH', '')
    env.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')

    cmd = [
        str(NS_EVAL),
        '--load-config', str(config),
        '--output-path', str(OUT),
        '--render-output-path', str(RENDERS),
    ]

    proc = subprocess.run(cmd, env=env)
    print('\nEVAL RETURN CODE:', proc.returncode)
    if proc.returncode != 0:
        return proc.returncode

    data = json.loads(OUT.read_text('utf-8'))
    print('\n=== GARDEN SPLATFACTO 5K HELD-OUT RESULTS ===')
    print(json.dumps(data.get('results', {}), indent=2))
    print('\nMETRICS JSON:', OUT)
    print('RENDERS:', RENDERS)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
