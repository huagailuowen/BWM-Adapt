#!/usr/bin/env python3
"""Train-only support ablation with frozen queries, weights and initialization."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from real916_stick_final_common import ROOT, read_rows, write_json


def read(path):
    return json.loads(Path(path).read_text())


def prepare(config, config_path):
    old = ROOT / config['reference_prepared']
    dest = ROOT / config['prepared']
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = dest.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (dest / 'prepared_complete.json').exists():
        if read(dest / 'prepared_complete.json')['config_sha256'] != digest:
            raise RuntimeError('Prepared config changed')
        return
    if dest.exists():
        raise RuntimeError('Refusing incomplete published destination')
    plan = read(old / 'plan.json')
    queries = read_rows(old / 'query.jsonl')
    supports = read_rows(old / 'support.jsonl')
    annotations = {(r['environment'], r['episode_index']): r for r in
                   read_rows(Path(plan['source_dataset']) / 'episode_split_manifest.jsonl')
                   if r['split'] == 'train'}
    windows = defaultdict(list)
    forbidden = {(r['environment'], r['episode_index']) for r in queries}
    for row in read_rows(ROOT / config['manifest'] / 'train.jsonl'):
        key = row['environment'], row['episode_index']
        if row['dataset_split'] != 'train':
            raise RuntimeError('Split leakage')
        if row['sampling_kind'] == 'lift' and key not in forbidden:
            windows[key].append(row)
    audit = []
    changed = []
    fallback = []
    for env in plan['environments']:
        name = env['environment']
        candidates = [r for key, r in annotations.items() if key[0] == name and key in windows
                      and r.get('vision_ok') and r.get('support_fraction') is not None
                      and r.get('angle_change_deg') is not None]
        balanced = [r for r in candidates if r['outcome'] == 'balanced']
        if not balanced:
            raise RuntimeError('No training balance reference: ' + name)
        positions = sorted(float(r['support_fraction']) for r in balanced)
        middle = (positions[(len(positions)-1)//2] + positions[len(positions)//2]) / 2
        left = [r for r in candidates if r['outcome'] == 'left_down']
        right = [r for r in candidates if r['outcome'] == 'right_down']
        before = {k: copy.deepcopy(env[k]) for k in
                  ('support_episode_indices', 'support_outcomes', 'support_fraction', 'support_tilt_change_deg')}
        if left and right:
            # Bracket the training-derived balance position with explicitly
            # opposite outcomes, not tiny sign changes inside balanced samples.
            pairs = [(a, b) for a in right for b in left
                     if float(a['support_fraction']) < middle < float(b['support_fraction'])]
            if not pairs:
                raise RuntimeError('Labels/position ordering inconsistent: ' + name)
            chosen = min(pairs, key=lambda p: (
                float(p[1]['support_fraction']) - float(p[0]['support_fraction']),
                max(abs(float(r['support_fraction'])-middle) for r in p),
                sum(abs(float(r['angle_change_deg'])) for r in p),
                p[0]['episode_index'], p[1]['episode_index']))
            for index, item in zip(env['support_indices'], chosen):
                rows = sorted(windows[(name, item['episode_index'])], key=lambda r: r['start_frame'])
                supports[index] = dict(rows[(len(rows)-1)//2])
            env.update(support_episode_indices=[r['episode_index'] for r in chosen],
                       support_outcomes=[r['outcome'] for r in chosen],
                       support_fraction=[r['support_fraction'] for r in chosen],
                       support_tilt_change_deg=[r['angle_change_deg'] for r in chosen],
                       train_only_balance_fraction=middle,
                       support_selection_policy='explicit_opposite_tilts_nearest_training_balance')
            changed.append(name)
        else:
            if name not in config['expected_one_sided_environments']:
                raise RuntimeError('Unexpected missing tilt direction: ' + name)
            fallback.append(name)
        entry = {'environment': name, 'before': before,
                 'after': {k: env[k] for k in before},
                 'changed': name in changed,
                 'training_balanced_position_range': [positions[0], positions[-1]],
                 'training_balanced_position_median': middle,
                 'candidate_counts': {'left_down': len(left), 'right_down': len(right), 'balanced': len(balanced)}}
        audit.append(entry)
        print('[support_selection] ' + json.dumps(entry), flush=True)
    if set(changed) != set(config['expected_changed_environments']) or set(fallback) != set(config['expected_one_sided_environments']):
        raise RuntimeError('Unexpected ablation cohort')
    for row in supports:
        if row['dataset_split'] != 'train' or (row['environment'], row['episode_index']) in forbidden:
            raise RuntimeError('Support/query contamination')
    tmp = Path(tempfile.mkdtemp(prefix=dest.name+'.partial-', dir=dest.parent))
    for filename in ('query.jsonl', 'action_stats.json'):
        shutil.copy2(old / filename, tmp / filename)
    for method in ('ours', 'standard'):
        shutil.copytree(old / method, tmp / method, copy_function=os.link)
    (tmp / 'support.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in supports))
    staged = {p for r in queries+supports for p in r['video'] +
              [r['action'], r['action'].split('/')[0]+'/meta/info.json']}
    (tmp / 'stage_files.txt').write_text('\n'.join(sorted(staged))+'\n')
    shutil.copy2(config_path, tmp / 'evaluation_config.json')
    plan.update(selection=config['support_policy'], config_sha256=digest,
                query_selection_unchanged=True, support_ablation_changed_environments=changed)
    write_json(tmp / 'plan.json', plan)
    write_json(tmp / 'support_selection.json', audit)
    # Decode review evidence only on the allocated CPU node.
    import cv2
    import numpy as np
    from real97_trial_common import native_frame
    cv2.setNumThreads(1)
    for env in plan['environments']:
        panels = []
        for j, index in enumerate(env['support_indices']):
            row = supports[index]
            frames = []
            end_k = row['valid_video_frames'] - 1
            for k in (0, end_k//2, end_k):
                f = min(row['start_frame']+3*k, row['total_frames']-1)
                frame = native_frame(Path(plan['source_dataset']) / row['video'][0], f)
                frame = cv2.copyMakeBorder(frame, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245,245,245))
                cv2.putText(frame, f"ep{row['episode_index']} f{f} {env['support_outcomes'][j]}",
                            (6,20), 0, .5, (0,0,0), 1)
                frames.append(frame)
            panels.append(np.hstack(frames))
        path = tmp / 'support_review' / (env['environment']+'.jpg')
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), np.vstack(panels)):
            raise RuntimeError('Support evidence write failed')
    write_json(tmp / 'prepared_complete.json', {'config_sha256': digest, 'queries': len(queries),
               'supports': len(supports), 'changed_environments': changed, 'fallback_unchanged': fallback})
    tmp.rename(dest)


def action_scores(rows, method):
    selected = [r for r in rows if r['action'][method]['5']['state'] == 'positive']
    tp = sum(r['dataset_outcome_for_audit'] == 'balanced' for r in selected)
    actual = sum(r['dataset_outcome_for_audit'] == 'balanced' for r in rows)
    return {'queries': len(rows), 'selected': len(selected), 'true_balanced': tp,
            'false_balanced': len(selected)-tp,
            'precision': tp/len(selected) if selected else None,
            'balanced_recall': tp/actual if actual else None,
            'unknown_predictions': sum(r['action'][method]['5']['state'] == 'unknown' for r in rows)}


def score(config):
    from score_real916_stick_final import setup, process, summarize, export_table
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.cpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'protocol.json').exists() and read(output / 'protocol.json') != config:
        raise RuntimeError('Changed metric protocol')
    write_json(output / 'protocol.json', config)
    tasks = setup(config, output)
    for task in tasks:
        hi = task['wanted'][-1]
        lo = hi-2
        if lo < 1 or not all(i in task['wanted'] for i in range(lo,hi+1)):
            raise RuntimeError('Insufficient visible terminal window')
        task['phase']['decision'] = {'terminal_window': {'frame_interval': [lo,hi]},
                                      'horizon_status': 'visible_real_video_tail'}
        task['action_eligible'] = True
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        records = list(pool.map(process, tasks))
    summary = summarize(records)
    summary['notes'] = ['Action uses final real 0.3 seconds, not commanded lift-end eligibility.',
                        'Diagnostic classifier bounds are separate from annotation-GT comparison.json.',
                        'Support-only ablation; complete fixed test45 primary; seed0927 subset secondary.',
                        'LPIPS not recomputed; visual audit pending.']
    write_json(output / 'per_query.json', records)
    write_json(output / 'summary.json', summary)
    export_table(summary, output)
    previous = read(ROOT / config['reference_metrics'] / 'per_query.json')
    selected = read(ROOT / config['selected_cohort'])['selected_run']
    keys = {key for env in selected['selection'] for key in env['keys']}
    old_by_key = {r['key']: r for r in previous}
    groups = {}
    populations = {'test45': [r for r in records if r['split']=='test'],
                   'fixed_seed20260927_test36': [r for r in records if r['key'] in keys],
                   'changed_environments_test25': [r for r in records if r['split']=='test'
                       and r['environment'] in config['expected_changed_environments']]}
    if [len(populations[k]) for k in populations] != [45,36,25]:
        raise RuntimeError('Comparison population changed')
    for name, rows in populations.items():
        old = [old_by_key[r['key']] for r in rows]
        if any(r['dataset_outcome_for_audit'] not in ('balanced','left_down','right_down') for r in rows):
            raise RuntimeError('Ambiguous annotation GT')
        groups[name] = {'standard': action_scores(rows, 'standard'),
                        'stage1': action_scores(rows, 'ours_stage1'),
                        'stage2_old_support': action_scores(old, 'ours_stage2'),
                        'stage2_new_support': action_scores(rows, 'ours_stage2')}
        for label, data, method in [('standard',rows,'standard'),('stage1',rows,'ours_stage1'),
                                    ('stage2_old_support',old,'ours_stage2'),('stage2_new_support',rows,'ours_stage2')]:
            metrics = {}
            for section, fields in [('object',('ade_px','fde_px','angle_mae_deg')),('image',('psnr','ssim'))]:
                for field in fields:
                    values = [r[section][method][field] for r in data if r[section][method][field] is not None]
                    metrics[field] = sum(values)/len(values) if values else None
                    metrics[field+'_count'] = len(values)
            groups[name][label].update(metrics)
    write_json(output / 'comparison.json', {'groups': groups, 'gt_source': 'dataset outcome labels',
        'primary': 'test45', 'post_hoc_seed_subset_secondary': True, 'manual_review_pending': True,
        'note': 'Object means include reported valid-frame masks; FDE counts may differ. No automatic main-table overwrite.'})
    write_json(output / 'cpu_complete.json', {'queries': len(records)})
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare','score'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    os.chdir(ROOT)
    config = read(args.config)
    if args.mode == 'prepare':
        prepare(config, args.config)
    else:
        score(config)


if __name__ == '__main__':
    main()
