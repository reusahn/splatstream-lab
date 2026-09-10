from pathlib import Path
import subprocess

DATA_UTILS = Path('/content/nerfstudio/nerfstudio/data/utils/data_utils.py')
PY = Path('/content/neuralscene-env/bin/python')

old = '    e.setimage(im.im)\n'
new = '    e.setimage(im.im, (0, 0, *im.size))\n'

text = DATA_UTILS.read_text()
if new in text:
    print('Pillow compatibility patch already applied.')
elif old in text:
    DATA_UTILS.write_text(text.replace(old, new, 1))
    print('Applied Pillow compatibility patch.')
else:
    raise RuntimeError('Expected Nerfstudio pil_to_numpy line was not found.')

# Verify conversion on a real Bonsai image.
check = (
    "from pathlib import Path; "
    "from PIL import Image, __version__ as pillow_version; "
    "from nerfstudio.data.utils.data_utils import pil_to_numpy; "
    "p=sorted(Path('/content/data/mipnerf360/bonsai/images_2').iterdir())[0]; "
    "im=Image.open(p); arr=pil_to_numpy(im); "
    "print('Pillow:', pillow_version); "
    "print('Image:', p.name, im.size, im.mode); "
    "print('Array:', arr.shape, arr.dtype); "
    "print('PIL_TO_NUMPY READY')"
)
subprocess.run([str(PY), '-c', check], check=True)
