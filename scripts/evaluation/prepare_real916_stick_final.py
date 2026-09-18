#!/usr/bin/env python3
"""Freeze train-only supports, shared factual queries, and completed checkpoints."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from real916_stick_final_common import ROOT, plot_pca, read_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run preparation on a compute node')
    config = json.loads(args.config.read_text())
    destination = ROOT / config['prepared']
    destination.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(args.config.read_bytes()).hexdigest()
    with destination.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / 'prepared_complete.json').exists():
            if json.loads((destination / 'prepared_complete.json').read_text())['config_sha256'] != fingerprint:
                raise RuntimeError('Prepared cohort belongs to a different config')
            return
        if destination.exists():
            raise RuntimeError('Incomplete published destination; refusing to overwrite')
        temp = Path(tempfile.mkdtemp(prefix=destination.name + '.partial-', dir=destination.parent))
        data = ROOT / config['manifest']
        summary = json.loads((data / 'manifest_summary.json').read_text())
        source = Path(summary['dataset_base_path'])
        by_ep = defaultdict(list)
        for subset in ('train', 'test'):
            for row in read_rows(data / (subset + '.jsonl')):
                if row['dataset_split'] != subset:
                    raise ValueError('Split mismatch')
                if row['sampling_kind'] == 'lift':
                    by_ep[(row['environment'], int(row['episode_index']), subset)].append(row)
        # Read training outcomes only for support selection. Test outcome labels
        # never enter the selection or adaptation routine.
        annotations = {(r['environment'], int(r['episode_index'])): r
                       for r in read_rows(source / 'episode_split_manifest.jsonl') if r['split'] == 'train'}
        def window(key):
            candidates = sorted(by_ep[key], key=lambda r: int(r['start_frame']))
            return dict(candidates[(len(candidates)-1)//2])
        rng = random.Random(config['seed'])
        supports, queries, envs = [], [], []
        for name, env_id in sorted(summary['environment_ids'].items(), key=lambda p: p[1]):
            train = sorted(k for k in by_ep if k[0] == name and k[2] == 'train')
            candidates = []
            for key in train:
                r = annotations.get(key[:2], {})
                u, angle = r.get('support_fraction'), r.get('angle_change_deg')
                if r.get('vision_ok') and u is not None and angle is not None and abs(angle) <= 15:
                    candidates.append((float(u), float(angle), key))
            candidates.sort()
            outcome = lambda item: annotations[item[2][:2]].get('outcome')
            available_directions = {outcome(item) for item in candidates} & {'left_down', 'right_down'}
            fallback = config.get('support_fallback', {})
            use_one_sided = (name in fallback.get('environments', [])
                             and len(available_directions) == 1)
            selection_notes = {}
            if use_one_sided:
                balanced = [item for item in candidates if outcome(item) == 'balanced']
                nonbalanced = [item for item in candidates
                               if outcome(item) in {'left_down', 'right_down'}
                               and abs(item[1]) >= .5]
                if not balanced or not nonbalanced:
                    raise RuntimeError(f'No labeled balanced/nonbalanced train pair for {name}')
                # Position proximity, not episode index, chooses the pair.
                # Ambiguous labels and all test episodes are excluded.
                stable, tilted = min(
                    ((a, b) for a in balanced for b in nonbalanced),
                    key=lambda pair: (abs(pair[0][0]-pair[1][0]),
                                      abs(pair[0][1]), abs(pair[1][1]),
                                      pair[0][2], pair[1][2]))
                chosen = [stable, tilted]
                balance = None  # Do not invent a zero-crossing estimate.
                selection_notes = {
                    'support_selection_policy': 'balanced_plus_nearest_nonbalanced_train_pair',
                    'one_sided_training_outcomes': sorted(available_directions),
                    'balanced_reference_fraction': stable[0],
                    'support_position_gap': abs(stable[0]-tilted[0]),
                }
            else:
                crossings = [(a, b) for a, b in zip(candidates, candidates[1:]) if a[1]*b[1] <= 0 and a[0] < b[0]]
                if not crossings:
                    raise RuntimeError(f'No train-only balance bracket for {name}; do not silently omit environment')
                a, b = min(crossings, key=lambda pair: pair[1][0]-pair[0][0])
                balance = a[0] + (b[0]-a[0])*abs(a[1])/max(abs(a[1])+abs(b[1]), 1e-9)
                left = [r for r in candidates if r[0] < balance and abs(r[1]) >= .5]
                right = [r for r in candidates if r[0] > balance and abs(r[1]) >= .5]
                if not left or not right:
                    raise RuntimeError(f'No informative train support on both sides for {name}')
                chosen = [max(left, key=lambda r:r[0]), min(right, key=lambda r:r[0])]
                selection_notes['support_selection_policy'] = 'train_only_balance_bracket'
            support_indices = list(range(len(supports), len(supports)+2))
            supports.extend(window(item[2]) for item in chosen)
            used = {item[2][1] for item in chosen}
            rest = [key for key in train if key[1] not in used]
            rng.shuffle(rest)
            tests = sorted(k for k in by_ep if k[0] == name and k[2] == 'test')
            if len(tests) != 5 or len(rest) < 2:
                raise ValueError(f'Unexpected cohort for {name}')
            query_indices = []
            for key in rest[:2] + tests:
                row = window(key)
                row['sample_index'] = len(queries)
                row['native_frame_indices'] = [min(row['start_frame']+3*k, row['total_frames']-1) for k in range(33)]
                row['evaluation_frame_indices'] = [k for k in range(1,33) if row['start_frame']+3*k < row['total_frames']]
                query_indices.append(len(queries))
                queries.append(row)
            envs.append({'environment':name, 'environment_index':env_id, 'support_indices':support_indices,
                         'query_indices':query_indices, 'train_only_balance_fraction':balance,
                         'support_fraction':[r[0] for r in chosen], 'support_tilt_change_deg':[r[1] for r in chosen],
                         'support_outcomes':[outcome(r) for r in chosen],
                         'support_episode_indices':[r[2][1] for r in chosen], **selection_notes})
            print('[support_selected] '+json.dumps(envs[-1]),flush=True)
        if len(envs) != 9 or len(queries) != 63 or len(supports) != 18:
            raise RuntimeError('Incomplete nine-environment cohort')
        for name, rows in [('query',queries),('support',supports)]:
            (temp / (name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        staged = {p for r in queries+supports for p in r['video']+[r['action'], r['action'].split('/')[0]+'/meta/info.json']}
        (temp/'stage_files.txt').write_text('\n'.join(sorted(staged))+'\n')
        shutil.copy2(data/'action_stats.json',temp/'action_stats.json')
        shutil.copy2(args.config,temp/'evaluation_config.json')
        table = None
        for method in ('ours','standard'):
            setting=config['methods'][method];run=ROOT/setting['training_run'];checkpoint=run/setting['checkpoint']
            if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
                raise RuntimeError(f'Missing final checkpoint: {checkpoint}')
            reference=temp/method;reference.mkdir()
            if method=='ours':
                if not (checkpoint.parent/('.'+checkpoint.name+'.complete')).is_file():
                    raise RuntimeError('Ours checkpoint is not completely published')
                table_path=run/setting['table']
                table=json.loads(table_path.read_text())
                if set(int(r['friction_mu']) for r in table['records']) != set(summary['environment_ids'].values()):
                    raise RuntimeError('Final table/environment mismatch')
                shutil.copy2(table_path,reference/'context_table.json')
            # Pin the exact weight file without another multi-GB copy.
            os.link(checkpoint,reference/'model.safetensors')
            shutil.copy2(run/'submitted_config.yaml',reference/'training_config.yaml')
            if (run/'input_manifest/action_stats.json').read_bytes() != (data/'action_stats.json').read_bytes():
                raise RuntimeError('Training and evaluation action statistics differ')
            write_json(reference/'checkpoint_selection.json',{'source':str(checkpoint),'step':setting['step'],
                       'size':checkpoint.stat().st_size,'mtime_ns':checkpoint.stat().st_mtime_ns,
                       'protected_by_hardlink':True,'training_job':setting['training_job']})
        plan={'task':'stick916','source_dataset':str(source),'environments':envs,'query_count':63,'support_count':18,
              'query_splits':{'train':18,'test':45},'unavailable_environments':[],
              'selection':'training-only balance bracket; configured one-sided environments use balanced plus nearest nonbalanced pair; midpoint of legal Lift starts; all held-out episodes',
              'no_query_GT_in_adaptation':True,'config_sha256':fingerprint}
        write_json(temp/'plan.json',plan)
        plot_pca(table,envs,temp/'ours',config['methods']['ours']['step'])
        # Small contact sheets are generated only on this CPU allocation.
        from real97_trial_common import native_frame
        import cv2
        cv2.setNumThreads(1)
        review=temp/'support_review';review.mkdir()
        for env in envs:
            panels=[]
            for i in env['support_indices']:
                row=supports[i];frames=[]
                for k in (0,16,32):
                    frame=native_frame(source/row['video'][0],min(row['start_frame']+3*k,row['total_frames']-1))
                    cv2.putText(frame,f"ep{row['episode_index']} f{min(row['start_frame']+3*k,row['total_frames']-1)}",(8,22),0,.6,(0,0,255),1)
                    frames.append(frame)
                panels.append(np.hstack(frames))
            cv2.imwrite(str(review/(env['environment']+'.jpg')),np.vstack(panels))
        write_json(temp/'prepared_complete.json',{'config_sha256':fingerprint,'environments':9,'queries':63,'supports':18})
        temp.rename(destination)
        print(json.dumps(plan),flush=True)


if __name__=='__main__':
    main()
