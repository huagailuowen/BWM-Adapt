#!/usr/bin/env python3
"""Freeze ten train-only support episodes per environment without changing queries."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from real97_trial_common import ROOT, read_jsonl, write_json, write_jsonl


def prepare(config_path, task):
    config = json.loads(config_path.read_text())
    setting = config['tasks'][task]
    source = ROOT / setting['support10_source_prepared']
    dest = ROOT / setting['prepared']
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite an existing preparation: {dest}')
    plan = json.loads((source / 'plan.json').read_text())
    queries = read_jsonl(source / 'query.jsonl')
    old_supports = read_jsonl(source / 'support.jsonl')
    templates = read_jsonl(source / 'extension_action_templates.jsonl')
    manifest = ROOT / setting['training_run'] / 'input_manifest'
    windows = read_jsonl(manifest / 'train.jsonl')
    splits = {(r['environment'].removesuffix('_lerobot'), int(r['episode_index'])): r['split']
              for r in read_jsonl(manifest / 'episode_split.jsonl')}
    events = {(r['environment'].removesuffix('_lerobot'), int(r['episode_index'])): r
              for r in read_jsonl(manifest / 'chunk_events.jsonl')}
    selected = []
    audit = []
    for env in plan['environments']:
        name = env['environment']
        env_id = int(env['environment_index'])
        meta = read_jsonl(Path(plan['source_dataset']) / (name + '_lerobot') / 'meta/episodes.jsonl')
        levels = {int(r['episode_index']): int(r.get('skill_gear', r.get('level', -1))) for r in meta}
        excluded = {int(r['episode_index']) for r in queries if r['environment'] == name}
        previous = [old_supports[i] for i in env['support_indices']]
        env_windows = {}
        for row in windows:
            physical = row.get('physical_environment', row['environment'].split('__z')[0].removesuffix('_lerobot'))
            ep = int(row['episode_index'])
            if physical != name or row.get('sampling_kind') != 'precise' or ep in excluded:
                continue
            if row.get('dataset_split') != 'train' or splits.get((name, ep)) != 'train':
                continue
            env_windows.setdefault(ep, {})[int(row['start_frame'])] = row
        env_indices = []
        for level in range(1, 11):
            def eligible(row):
                ep = int(row['episode_index'])
                return (row['environment'] == name and row.get('dataset_split') == 'train'
                        and ep not in excluded and splits.get((name, ep)) == 'train'
                        and levels.get(ep) == level and int(row.get('action_level', level)) == level)
            choices = [r for r in previous if eligible(r)]
            origin = 'existing_support'
            if not choices:
                choices = [r for r in templates if eligible(r)]
                origin = 'existing_train_action_template'
            if choices:
                row = copy.deepcopy(choices[0])
            else:
                candidates = sorted(ep for ep in env_windows if levels.get(ep) == level)
                if not candidates:
                    raise RuntimeError(f'{name} level {level}: no train episode disjoint from ALL fixed queries')
                # Deterministic selection independent of generated quality or test labels.
                seed = f"{plan['protocol']['seed']}:{name}:{level}"
                ep = candidates[int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(candidates)]
                starts = sorted(env_windows[ep])
                row = copy.deepcopy(env_windows[ep][starts[(len(starts) - 1) // 2]])
                row['sample_id'] = row.get('source_sample_id', row['sample_id'].split('__z')[0])
                row.update(environment=name, environment_index=env_id, friction_mu=float(env_id))
                for key in ['virtual_environment_id', 'latent_replica', 'template_index']:
                    row.pop(key, None)
                ann = events[(name, ep)]
                if ann['split'] != 'train' or int(ann['num_frames']) != int(row['total_frames']):
                    raise RuntimeError(f'Frozen event annotation mismatch: {name}/{ep}')
                start, end, stride = int(row['start_frame']), int(row['end_frame']), int(row['frame_stride'])
                native = [min(start + k * stride, end) for k in range(int(row['length']))]
                row.update(native_frame_indices=native,
                           evaluation_frame_indices=[k for k in range(1, len(native)) if native[k] > native[k - 1]],
                           source_event_annotation=ann['events'])
                origin = 'frozen_train_manifest_precise_window'
            row.update(environment=name, environment_index=env_id, friction_mu=float(env_id),
                       action_level=level, support_selection_origin=origin)
            if not eligible(row):
                raise RuntimeError(f'Invalid support selection: {name}/{level}')
            env_indices.append(len(selected))
            selected.append(row)
            audit.append(dict(environment=name, level=level, episode_index=row['episode_index'],
                              start_frame=row['start_frame'], end_frame=row['end_frame'], origin=origin))
        env.update(support_indices=env_indices, requested_support_levels=list(range(1, 11)),
                   actual_support_levels=list(range(1, 11)), missing_support_levels=[],
                   support_exception='ten_train_levels_ablation', extension_template_indices=[])
        env.pop('support_measurement', None)
        env.pop('preferred_support_interval_satisfied', None)
        env['support_selection_status'] = 'train_only_all_levels_query_episode_disjoint'
    dest.mkdir(parents=True)
    shutil.copytree(source / 'reference', dest / 'reference', copy_function=os.link)
    for filename in ['query.jsonl', 'extension_action_templates.jsonl']:
        shutil.copy2(source / filename, dest / filename)
    write_jsonl(dest / 'support.jsonl', selected)
    files = set((source / 'stage_files.txt').read_text().splitlines())
    for row in selected:
        videos = row['video'] if isinstance(row['video'], list) else [row['video']]
        files.update(videos)
        files.add(row['action'])
    (dest / 'stage_files.txt').write_text('\n'.join(sorted(files)) + '\n')
    plan.update(prepared=str(dest), support_count=len(selected), active_support_count=len(selected),
                support_count_per_environment=10, support_manifest_retains_unused_original_rows=False,
                support_reconstruction=True, support_reconstruction_is_heldout=False,
                support_selection_audit=audit, old_outputs_modified=False)
    plan['protocol'][task].update(support_count=10,
        explicit_support_levels_by_environment={e['environment']: list(range(1, 11)) for e in plan['environments']})
    plan['protocol']['purpose'] = 'ROI10 support-count ablation; ten training levels; fixed queries'
    plan['experiment']['extension']['enabled'] = False
    plan['historical_support_selection_provenance'] = {'source_prepared': str(source), 'superseded': True}
    plan['source_manifest_sha256'] = {
        name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
        for name in ['query.jsonl', 'support.jsonl', 'extension_action_templates.jsonl', 'stage_files.txt']}
    plan['query_manifest_unchanged'] = True
    write_json(dest / 'plan.json', plan)
    write_json(dest / 'experiment.json', config)
    write_json(dest / 'support_selection.json', audit)
    write_json(dest / 'prepared_complete.json', dict(task=task, supports=len(selected), queries=len(queries),
        environments=len(plan['environments']), support_query_episode_disjoint=True,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
    print(json.dumps(dict(task=task, prepared=str(dest), supports=len(selected), queries=len(queries))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['ball', 'door'], required=True)
    args = parser.parse_args()
    prepare(args.config, args.task)
