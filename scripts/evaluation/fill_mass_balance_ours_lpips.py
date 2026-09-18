"""Fill LPIPS from the current formal query manifest without rerunning tracking."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wan_video_action.evaluation.io import read_video_frames, write_json_atomic, write_jsonl_atomic
from wan_video_action.metrics.aggregation import aggregate_query_metrics
from wan_video_action.metrics.global_video import LPIPSEvaluator, _align_frames


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'yejin-lo':
        raise RuntimeError('Run on a low-priority compute node only.')
    os.chdir(ROOT)
    run = ROOT / 'results/mass_balance/fixed_pose_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/ours/step_3900/seed_20260723'
    metrics = run / 'video_metrics'
    manifest_bytes = (metrics / 'manifest.jsonl').read_bytes()
    manifest = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    global_path = metrics / 'global/global_per_query.jsonl'
    original_global = global_path.read_bytes()
    rows = [json.loads(line) for line in original_global.splitlines() if line.strip()]
    def key(row):
        return int(row['source_index']), int(row['target_index'])
    by_key = {key(row): row for row in rows}
    manifest_keys = [key(record['metadata']) for record in manifest]
    if len(rows) != 140 or len(by_key) != 140 or len(set(manifest_keys)) != 140 or set(manifest_keys) != set(by_key):
        raise RuntimeError('Expected exactly the same 140 formal disjoint queries.')
    evaluator = LPIPSEvaluator(net='alex', device='cuda')
    for index, record in enumerate(manifest, 1):
        frames = []
        for prefix in ('gt', 'pred'):
            video = np.asarray(read_video_frames(record[f'{prefix}_video_path']))
            start = int(record[f'{prefix}_start_frame'])
            stride = int(record[f'{prefix}_frame_stride'])
            count = int(record['num_frames'])
            selected = video[start:start + stride * count:stride, :, :224]
            if len(selected) != count:
                raise RuntimeError(f'Short video: {record[f"{prefix}_video_path"]}')
            frames.append(selected)
        gt, pred = _align_frames(*frames)
        value = float(np.mean(evaluator(gt, pred, batch_size=8)))
        if not np.isfinite(value):
            raise RuntimeError('Non-finite LPIPS.')
        by_key[key(record['metadata'])]['lpips'] = value
        print(f'[lpips] {index}/{len(manifest)} value={value:.8f}', flush=True)
    summary = aggregate_query_metrics(rows, ['psnr', 'ssim', 'lpips'])
    envs = summary['environment']
    if len(envs) != 10 or sum(row['domain'] == 'id' for row in envs) != 5:
        raise RuntimeError('Expected five ID and five OOD environments.')
    value = float(np.mean([row['lpips'] for row in envs]))
    table_path = ROOT / 'configs/evaluation/sim_all_methods_main_table_v1.yaml'
    table_text = table_path.read_text()
    table = yaml.safe_load(table_text)
    # Locate by task identity, regardless of the top-level collection key.
    tasks = next(items for items in table.values() if isinstance(items, list)
                 and any(isinstance(item, dict) and item.get('id') == 'mass_balance' for item in items))
    task = next(item for item in tasks if item.get('id') == 'mass_balance')
    ours = task['values']['ours']
    if not str(run.relative_to(ROOT)).startswith(task['source']['ours'] + '/'):
        raise RuntimeError('The table now points at a different Ours experiment; refusing to overwrite.')
    for metric in ('psnr', 'ssim'):
        measured = float(np.mean([row[metric] for row in envs]))
        if not np.isclose(measured, ours[metric], rtol=0, atol=1e-7):
            raise RuntimeError(f'Table/query mismatch for {metric}.')
    start = table_text.index('  - id: mass_balance\n')
    end = table_text.find('\n  - id:', start + 1)
    if end < 0:
        end = len(table_text)
    section = table_text[start:end]
    import re
    section, replacements = re.subn(r'(      ours: \{[^\n]*?lpips: )[^,}]+',
                                    lambda match: match[1] + repr(value), section)
    if replacements != 1:
        raise RuntimeError('Could not uniquely update the Ours LPIPS cell.')
    if global_path.read_bytes() != original_global:
        raise RuntimeError('Global metrics changed concurrently; refusing to overwrite.')
    write_jsonl_atomic(global_path, rows)
    write_json_atomic(metrics / 'global/global_summary.json', summary)
    write_json_atomic(metrics / 'global/lpips_fill_protocol.json', {
        'network': 'official lpips alex', 'device': 'cuda', 'batch_size': 8,
        'query_count': len(rows), 'environment_count': len(envs),
        'lpips': value, 'aggregation': 'mean frames, then queries per environment, then environments',
        'view': 'main only, first 224 pixels', 'temporal_alignment': 'existing manifest start/stride/count',
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'job_id': os.environ['SLURM_JOB_ID'], 'videos_regenerated': False,
        'other_metrics_preserved': True, 'cross_dataset_marker_preserved': True,
    })
    if table_path.read_text() != table_text:
        raise RuntimeError('Table config changed concurrently; metrics saved, table update deferred.')
    temporary = table_path.with_suffix('.yaml.lpips.tmp')
    temporary.write_text(table_text[:start] + section + table_text[end:])
    temporary.replace(table_path)
    subprocess.run([sys.executable, 'scripts/evaluation/build_sim_all_methods_table.py',
                    '--config', str(table_path)], check=True)
    print(f'[done] Mass balance Ours LPIPS={value:.10f}; main table updated', flush=True)


if __name__ == '__main__':
    main()
