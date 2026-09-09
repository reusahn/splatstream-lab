from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

GSPLAT_DIR = Path('/content/gsplat')
EXAMPLES = GSPLAT_DIR / 'examples'
DATA_DIR = EXAMPLES / 'data/360_v2/bonsai'
BASE_RESULT = EXAMPLES / 'results/splatstream_bonsai_7k'
REPO = Path('/content/splatstream-lab')
OUT = Path('/content/splatstream_full_cuda_rd_v2')
ARCHIVE_BASE = Path('/content/splatstream_full_cuda_rd_v2')

PRESETS = [
    ('baseline_f32', 0.00, 32),
    ('prune25_q16', 0.25, 16),
    ('prune50_q16', 0.50, 16),
    ('prune50_q8', 0.50, 8),
    ('prune75_q8', 0.75, 8),
]
SUPPORTED = ('means', 'scales', 'quats', 'opacities', 'sh0', 'shN')


def newest_checkpoint() -> Path:
    ckpts = sorted(BASE_RESULT.rglob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f'No checkpoint under {BASE_RESULT}')
    return ckpts[-1]


def qdq(x: torch.Tensor, bits: int):
    x = x.detach().float().cpu().contiguous()
    if bits >= 32:
        arr = x.numpy().astype('<f4', copy=False)
        return x.clone(), arr.tobytes(order='C')
    if bits == 16:
        q = x.to(torch.float16)
        return q.float(), q.numpy().astype('<f2', copy=False).tobytes(order='C')
    if bits != 8:
        raise ValueError(bits)
    lo = x.amin(dim=0, keepdim=True)
    hi = x.amax(dim=0, keepdim=True)
    span = torch.clamp(hi - lo, min=1e-12)
    q = torch.round((x - lo) / span * 255.0).clamp_(0, 255).to(torch.uint8)
    dq = q.float() / 255.0 * span + lo
    payload = (
        lo.numpy().astype('<f4', copy=False).tobytes(order='C')
        + hi.numpy().astype('<f4', copy=False).tobytes(order='C')
        + q.numpy().tobytes(order='C')
    )
    return dq.float(), payload


def add_blob(buf: bytearray, name: str, shape, bits: int, payload: bytes):
    b = name.encode('utf-8')
    buf.extend(struct.pack('<H', len(b)))
    buf.extend(b)
    buf.extend(struct.pack('<B', len(shape)))
    for d in shape:
        buf.extend(struct.pack('<I', int(d)))
    buf.extend(struct.pack('<B', bits))
    buf.extend(struct.pack('<Q', len(payload)))
    buf.extend(payload)


def importance_order(splats):
    opacity = torch.sigmoid(splats['opacities'].detach().float().cpu()).reshape(-1)
    log_scales = splats['scales'].detach().float().cpu()
    score = opacity * torch.exp(log_scales.mean(dim=-1))
    return torch.argsort(score, descending=True)


def build_variant(base_splats, order, prune_fraction, bits):
    n = int(base_splats['means'].shape[0])
    keep = max(1, int(round(n * (1.0 - prune_fraction))))
    idx = torch.arange(n) if prune_fraction <= 0 else order[:keep]
    selected = {k: v.detach().cpu()[idx] for k, v in base_splats.items()}
    out = {}
    blob = bytearray(b'SPLATRD2')

    out['means'], packed = qdq(selected['means'], bits)
    add_blob(blob, 'means', out['means'].shape, bits, packed)
    out['scales'], packed = qdq(selected['scales'], bits)
    add_blob(blob, 'scales_log', out['scales'].shape, bits, packed)

    q, packed = qdq(selected['quats'], bits)
    out['quats'] = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-8)
    add_blob(blob, 'quats', selected['quats'].shape, bits, packed)

    prob = torch.sigmoid(selected['opacities'].float())
    q, packed = qdq(prob, bits)
    out['opacities'] = torch.logit(q.clamp(1e-6, 1 - 1e-6))
    add_blob(blob, 'opacity_probability', prob.shape, bits, packed)

    out['sh0'], packed = qdq(selected['sh0'], bits)
    add_blob(blob, 'sh0', out['sh0'].shape, bits, packed)
    out['shN'], packed = qdq(selected['shN'], bits)
    add_blob(blob, 'shN', out['shN'].shape, bits, packed)

    for key, tensor in selected.items():
        if key in SUPPORTED:
            continue
        out[key] = tensor.detach().float().cpu().contiguous()
        raw = out[key].numpy().astype('<f4', copy=False).tobytes(order='C')
        add_blob(blob, key, out[key].shape, 32, raw)
    return out, bytes(blob), keep


def load_simple_trainer():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    module_name = 'splatstream_simple_trainer'
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES / 'simple_trainer.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_splats(runner, trainer, splats_cpu):
    params = {}
    for k, v in splats_cpu.items():
        params[k] = torch.nn.Parameter(v.to(runner.device, non_blocking=True), requires_grad=False)
    runner.splats = torch.nn.ParameterDict(params)
    runner.scene = trainer.GaussianScene.from_splats(runner.splats, id='scene')
    runner.splats = runner.scene.splats
    runner.stage = trainer.Stage()
    runner.stage.add_scene(runner.scene, runner.rasterize_splats)
    torch.cuda.synchronize()


def read_baseline_stats():
    files = sorted((BASE_RESULT / 'stats').glob('val_step*.json'))
    if not files:
        raise FileNotFoundError('Baseline validation stats are missing.')
    return json.loads(files[-1].read_text())


def eval_variant(runner, trainer, name, splats_cpu, step):
    stat_file = Path(runner.stats_dir) / f'{name}_step{step:04d}.json'
    if stat_file.exists():
        print('Reusing:', stat_file)
        return json.loads(stat_file.read_text())
    print(f'Installing {name} tensors on GPU...', flush=True)
    install_splats(runner, trainer, splats_cpu)
    print(f'Evaluating {name} on held-out Bonsai views...', flush=True)
    tic = time.time()
    runner.eval(step=step, stage=name)
    print(f'{name} evaluation wall time: {time.time() - tic:.1f}s', flush=True)
    if not stat_file.exists():
        raise RuntimeError(f'Expected stats not written: {stat_file}')
    return json.loads(stat_file.read_text())


def save_outputs(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'full_cuda_rate_distortion.json').write_text(json.dumps(rows, indent=2))
    with (OUT / 'full_cuda_rate_distortion.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import matplotlib.pyplot as plt
    ratios = [r['zlib_compression_ratio'] for r in rows]
    labels = [r['preset'] for r in rows]
    for key, ylabel, filename in [
        ('psnr', 'PSNR (dB)', 'rate_distortion_psnr.png'),
        ('ssim', 'SSIM', 'rate_distortion_ssim.png'),
        ('lpips', 'LPIPS (lower is better)', 'rate_distortion_lpips.png'),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        vals = [r[key] for r in rows]
        ax.plot(ratios, vals, marker='o')
        ax.set_xlabel('Experimental payload compression ratio (zlib-backed)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        for x, y, label in zip(ratios, vals, labels):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    base_path = newest_checkpoint()
    base = torch.load(base_path, map_location='cpu', weights_only=True)
    base_splats = base['splats']
    step = int(base.get('step', 6999))

    print('Base checkpoint:', base_path)
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
    print('Base Gaussians:', int(base_splats['means'].shape[0]))

    OUT.mkdir(parents=True, exist_ok=True)
    eval_root = OUT / 'inprocess_eval'
    eval_root.mkdir(parents=True, exist_ok=True)

    print('Loading pinned gsplat trainer once...', flush=True)
    trainer = load_simple_trainer()
    cfg = trainer.Config(
        disable_viewer=True,
        disable_video=True,
        data_factor=2,
        data_dir='data/360_v2/bonsai',
        result_dir=str(eval_root),
    )
    print('Initializing held-out evaluator once...', flush=True)
    runner = trainer.Runner(0, 0, 1, cfg)
    print('Evaluator ready.', flush=True)

    order = importance_order(base_splats)
    baseline_stats = read_baseline_stats()
    rows = []
    baseline_z = baseline_raw = None

    for name, prune_fraction, bits in PRESETS:
        print('\n' + '=' * 72)
        print(f'{name}: prune={prune_fraction:.0%}, quant={bits}-bit', flush=True)
        splats_dq, raw_blob, keep = build_variant(base_splats, order, prune_fraction, bits)
        zblob = zlib.compress(raw_blob, 6)
        raw_bytes, zbytes = len(raw_blob), len(zblob)
        if baseline_z is None:
            baseline_z, baseline_raw = zbytes, raw_bytes

        if name == 'baseline_f32':
            stats = baseline_stats
            print('Reusing verified baseline stats:', stats)
        else:
            stats = eval_variant(runner, trainer, name, splats_dq, step)

        row = {
            'preset': name,
            'prune_fraction': prune_fraction,
            'quant_bits': bits,
            'num_gaussians': int(stats.get('num_GS', keep)),
            'packed_uncompressed_bytes': raw_bytes,
            'zlib_payload_bytes': zbytes,
            'raw_compression_ratio': float(baseline_raw / raw_bytes),
            'zlib_compression_ratio': float(baseline_z / zbytes),
            'psnr': float(stats['psnr']),
            'ssim': float(stats['ssim']),
            'lpips': float(stats['lpips']),
            'render_seconds_per_image': float(stats['ellipse_time']),
            'transfer_seconds_25mbps': zbytes * 8 / 25_000_000,
            'transfer_seconds_50mbps': zbytes * 8 / 50_000_000,
            'transfer_seconds_100mbps': zbytes * 8 / 100_000_000,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        save_outputs(rows)
        del splats_dq, raw_blob, zblob
        torch.cuda.empty_cache()

    env = {
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'gsplat_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=GSPLAT_DIR, text=True).strip(),
        'base_checkpoint': str(base_path),
        'method': 'single-process held-out evaluation; same Runner/dataset/metric networks reused across presets',
    }
    (OUT / 'environment.json').write_text(json.dumps(env, indent=2))
    (OUT / 'METHOD.md').write_text(
        '# Full CUDA rate-distortion evaluation\n\n'
        'All compressed variants are evaluated in one Python process using the same pinned gsplat Runner, '
        'Bonsai held-out split, camera configuration, CUDA rasterizer, and PSNR/SSIM/LPIPS metric instances. '
        'The baseline metrics are reused from the verified 7,000-step run. Quantized tensors are dequantized '
        'for rasterization; payload byte counts are computed from the corresponding stored quantized representation.\n'
    )
    archive = shutil.make_archive(str(ARCHIVE_BASE), 'zip', root_dir=OUT)
    print('\nSUCCESS:', archive)
    try:
        from google.colab import files
        files.download(archive)
    except Exception:
        print('Download manually:', archive)


if __name__ == '__main__':
    main()
