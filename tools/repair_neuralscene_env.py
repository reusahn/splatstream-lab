from __future__ import annotations

import subprocess
from pathlib import Path

VENV = Path('/content/neuralscene-env')
PY = VENV / 'bin/python'

if not PY.exists():
    raise RuntimeError(f'Expected environment not found: {PY}')

print('Repairing NeuralScene Bench Python environment...')
subprocess.run([
    'uv', 'pip', 'install',
    '--python', str(PY),
    'pip', 'setuptools', 'wheel', 'ninja'
], check=True)

print('\nVerifying pip inside the isolated environment...')
subprocess.run([str(PY), '-m', 'pip', '--version'], check=True)

print('\nNEURALSCENE ENV REPAIR COMPLETE')
