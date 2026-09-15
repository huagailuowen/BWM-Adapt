#!/usr/bin/env python3
"""Add matched-frame Soft LPIPS without changing existing inference or scores."""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for site in (ROOT / '.venv-real97-eval-20260909/lib').glob('python*/site-packages'):
    sys.path.append(str(site))

import cv2
import numpy as np
import torch
import lpips

SOURCE = ROOT / 'outputs/eval_real97_soft_family_mean_comparison_20260914_v1/initialization_comparison_per_query.json'
OURS = ROOT / 'outputs/infer_real97_soft_balanced9_family_mean_static_20260914_v1'
STANDARD = ROOT / 'outputs/infer_real97_soft_standard_matched_balanced9_static_20260914_v1'
TABLE = ROOT / 'results/real97_all_methods_main_table_v1'
OUT = TABLE / 'soft_lpips'


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(content)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot decode ' + str(path))
    selected = []
    wanted = set(indices)
    try:
        for index in range(max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Missing frame {index} in {path}')
            if index in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected.append(cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_LINEAR))
    finally:
        cap.release()
    array = np.stack(selected).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(array).to(device='cuda', dtype=torch.float32) / 127.5 - 1.0


def aggregate(rows):
    if not rows:
        raise RuntimeError('Empty LPIPS cohort')
    return {
        'queries': len(rows),
        'environments': sorted({r['environment'] for r in rows}),
        'standard': float(np.mean([r['standard']['mean'] for r in rows])),
        'ours': float(np.mean([r['ours']['mean'] for r in rows])),
    }


def publish(summary):
    csv_path = TABLE / 'real97_all_methods_main_table.csv'
    reader = csv.DictReader(io.StringIO(csv_path.read_text()))
    fields = reader.fieldnames
    rows = list(reader)
    for row in rows:
        if row['method'] in ('Standard', 'Ours'):
            row['soft_lpips'] = format(summary['test'][row['method'].lower()], '.5f')
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    readme_path = TABLE / 'README.md'
    readme = readme_path.read_text()
    updates = [
        ('| Standard | 33.10 | 0.9644 | pending | 30.28 | 48.75 |', 'test', 'standard'),
        ('| Ours | 32.47 | 0.9608 | pending | 22.69 | 40.35 |', 'test', 'ours'),
        ('| Standard | 32.90 | 0.9642 | pending | 32.78 | 39.41 |', 'test_shared_training_environments', 'standard'),
        ('| Ours | 32.61 | 0.9615 | pending | 18.10 | 22.38 |', 'test_shared_training_environments', 'ours'),
    ]
    for old, cohort, method in updates:
        new = old.replace('pending', format(summary[cohort][method], '.5f'))
        if old in readme:
            readme = readme.replace(old, new, 1)
        elif new not in readme:
            raise RuntimeError('Table changed; refusing to replace an unexpected row')
    readme = readme.replace('LPIPS is being added using AlexNet', 'LPIPS was computed using AlexNet')
    readme = readme.replace(
        'Pending\nentries are filled automatically after the compute-node scoring job completes.',
        'The entries were filled after the compute-node scoring job completed.',
    )
    atomic_text(csv_path, output.getvalue())
    atomic_text(readme_path, readme)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run on an allocated GPU compute node, not a login node')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    source_bytes = SOURCE.read_bytes()
    records = json.loads(source_bytes)
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval()
    model.requires_grad_(False)
    results = []
    for row in records:
        key = row['key']
        paths = {
            'gt': OURS / 'raw/gt' / (key + '.mp4'),
            'standard': STANDARD / 'raw/standard' / (key + '.mp4'),
            'ours': OURS / 'raw/stage2' / (key + '.mp4'),
        }
        gt = frames(paths['gt'], list(range(1, 29)))
        result = {
            'key': key, 'environment': row['environment'], 'split': row['split'],
            'cohort': row['cohort'],
            'paths': {k: str(v) for k, v in paths.items()},
            'native_future_frames': list(range(3, 85, 3)),
        }
        for method, indices in [('standard', list(range(5, 33))), ('ours', list(range(1, 29)))]:
            prediction = frames(paths[method], indices)
            values = []
            with torch.inference_mode():
                for start in range(0, 28, 8):
                    values.extend(model(gt[start:start + 8], prediction[start:start + 8]).flatten().cpu().tolist())
            result[method] = {'mean': float(np.mean(values)), 'per_frame': values, 'video_indices': indices}
            del prediction
        del gt
        results.append(result)
        write_json(OUT / 'per_query.json', results)
        print(json.dumps({'query': key, 'standard': result['standard']['mean'], 'ours': result['ours']['mean']}), flush=True)
    test = [r for r in results if r['split'] == 'test']
    shared = [r for r in test if r['cohort']['standard_environment_seen'] and r['cohort']['ours_environment_seen']]
    summary = {
        'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'],
        'source': str(SOURCE), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'metric': 'LPIPS', 'backbone': 'alex', 'version': '0.1',
        'rgb_size': [512, 256], 'resize': 'opencv_INTER_LINEAR',
        'input_range': [-1, 1], 'native_future_frames': list(range(3, 85, 3)),
        'aggregation': 'mean over 28 future frames, then mean over queries',
        'tracking_mask_applied': False, 'condition_frames_scored': False,
        'test': aggregate(test),
        'test_shared_training_environments': aggregate(shared),
        'train': aggregate([r for r in results if r['split'] == 'train']),
    }
    write_json(OUT / 'summary.json', summary)
    publish(summary)
    write_json(OUT / 'table_update_complete.json', {'job_id': os.environ['SLURM_JOB_ID'], 'status': 'complete'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
