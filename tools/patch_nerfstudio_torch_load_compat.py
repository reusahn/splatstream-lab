from __future__ import annotations

from pathlib import Path

TARGET = Path('/content/nerfstudio/nerfstudio/utils/eval_utils.py')
OLD = 'loaded_state = torch.load(load_path, map_location="cpu")'
NEW = 'loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)'


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f'Missing Nerfstudio eval utils: {TARGET}')

    text = TARGET.read_text(encoding='utf-8')

    if NEW in text:
        print('TORCH LOAD COMPAT PATCH ALREADY APPLIED')
        return

    if OLD not in text:
        raise RuntimeError(
            'Expected pinned Nerfstudio torch.load call was not found. '
            'Refusing to patch an unexpected source revision.'
        )

    TARGET.write_text(text.replace(OLD, NEW, 1), encoding='utf-8')

    verify = TARGET.read_text(encoding='utf-8')
    if NEW not in verify:
        raise RuntimeError('Patch verification failed.')

    print('Patched:', TARGET)
    print('Reason: PyTorch >=2.6 defaults torch.load(weights_only=True), while this pinned Nerfstudio checkpoint contains trusted training-state objects that require the legacy full loader.')
    print('TORCH LOAD COMPAT PATCH COMPLETE')


if __name__ == '__main__':
    main()
