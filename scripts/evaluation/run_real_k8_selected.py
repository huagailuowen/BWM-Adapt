#!/usr/bin/env python3
"""Extend the selected real Door/Stick K=4 protocols to eight train supports."""

import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess

import run_real_support_number as common
import run_real_globalmean_support_number as globalmean
import run_real_globalmean_sum_support as sumloss


ROOT = common.ROOT
DEST = common.DEST
DOOR_SOURCE = DEST / 'globalmean_sumloss_original_support_20260922/door/k4'
STICK_K4_PROTOCOL = DEST / 'stick/k4/support_protocol.json'


def add_door_supports(source, destination):
    prepared = destination / 'prepared'
    if prepared.exists():
        plan = common.read(prepared / 'plan.json')
        if not all(len(env['support_indices']) == 8 for env in plan['environments']):
            raise RuntimeError('Existing Door K=8 prepared state is incomplete')
        return prepared

    temporary = destination / f'.prepared.{os.getpid()}.partial'
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source / 'prepared', temporary, symlinks=True)
    plan = common.read(temporary / 'plan.json')
    supports = common.rows(temporary / 'support.jsonl')
    queries = common.rows(temporary / 'query.jsonl')
    templates = common.rows(temporary / 'extension_action_templates.jsonl')
    forbidden = {(row['environment'], row['episode_index']) for row in queries}
    selection = []

    for env in plan['environments']:
        name = env['environment']
        original = [supports[index] for index in env['support_indices']]
        if len(original) != 4 or len({row['episode_index'] for row in original}) != 4:
            raise RuntimeError(f'Expected four distinct Door K=4 supports: {name}')
        used_episodes = {row['episode_index'] for row in original}
        used_levels = {int(row['action_level']) for row in original}
        center = sum(used_levels) / len(used_levels)
        candidates = [row for row in templates if row['environment'] == name
                      and row['dataset_split'] == 'train'
                      and (name, row['episode_index']) not in forbidden]
        candidates.sort(key=lambda row: (abs(int(row['action_level']) - center),
                                         int(row['action_level']), int(row['episode_index'])))
        for candidate in candidates:
            if len(env['support_indices']) == 8:
                break
            level = int(candidate['action_level'])
            episode = candidate['episode_index']
            if episode in used_episodes or level in used_levels:
                continue
            row = copy.deepcopy(candidate)
            row.pop('template_index', None)
            env['support_indices'].append(len(supports))
            supports.append(row)
            used_episodes.add(episode)
            used_levels.add(level)
        if len(env['support_indices']) != 8:
            raise RuntimeError(f'Fewer than eight legal Door train supports: {name}')
        chosen = [supports[index] for index in env['support_indices']]
        if chosen[:4] != original:
            raise RuntimeError(f'Door K=4 prefix changed: {name}')
        if any((name, row['episode_index']) in forbidden for row in chosen):
            raise RuntimeError(f'Door support/query leakage: {name}')
        levels = [int(row['action_level']) for row in chosen]
        env.update(requested_support_levels=levels, actual_support_levels=levels,
                   missing_support_levels=[], extension_template_indices=[])
        selection.append({'environment': name, 'support_indices': env['support_indices'],
                          'episodes': [row['episode_index'] for row in chosen],
                          'levels': levels, 'test_query_indices': [index for index in env['query_indices']
                          if queries[index]['dataset_split'] == 'test']})

    plan['prepared'] = str(prepared)
    plan['active_support_count'] = sum(len(env['support_indices']) for env in plan['environments'])
    plan['protocol']['ttt']['ttt_support_loss_reduction'] = 'sum'
    common.write_rows(temporary / 'support.jsonl', supports)
    common.write(temporary / 'plan.json', plan)
    staged = set((temporary / 'stage_files.txt').read_text().splitlines())
    for row in supports:
        staged.update(row['video'])
        staged.add(row['action'])
    (temporary / 'stage_files.txt').write_text('\n'.join(sorted(staged)) + '\n')
    temporary.rename(prepared)
    common.write(destination / 'support_protocol.json', {
        'K': 8, 'source': str(source), 'initialization': 'global mean',
        'loss_reduction': 'sum', 'query_GT_for_adaptation': False,
        'support_policy': 'retain selected K4, add four distinct train episodes and action levels',
        'query_cohort_unchanged': True, 'selection': selection,
    })
    return prepared


def door(destination):
    source = DOOR_SOURCE
    prepared = add_door_supports(source, destination)
    config = common.read(source / 'config.json')
    config['output'] = str(destination)
    setting = config['tasks']['door']
    setting.update(source_prepared=str(source / 'prepared'), source_inference=str(source),
                   prepared=str(prepared), support_levels={},
                   partial_stage2_replacement=False, require_exact_original_support=False)
    config['support_count_experiment'] = {
        'K': 8, 'source': str(source), 'initialization': 'global mean',
        'support_selection': 'selected K4 plus four train-only levels per environment',
        'support_loss_reduction': 'sum',
    }
    config_path = destination / 'config.json'
    if config_path.exists() and common.read(config_path) != config:
        previous = common.read(config_path)
        previous['tasks']['door']['require_exact_original_support'] = False
        if previous != config:
            raise RuntimeError('Existing Door K=8 configuration differs beyond the corrected support check')
    common.write(config_path, config)

    stage1 = source / 'raw/stage1'
    if not stage1.is_dir():
        raise RuntimeError('Door K=4 Stage1 source is missing')
    target_stage1 = destination / 'raw/stage1'
    target_stage1.parent.mkdir(parents=True, exist_ok=True)
    if not target_stage1.exists():
        target_stage1.symlink_to(stage1.resolve(), target_is_directory=True)
    if target_stage1.resolve() != stage1.resolve():
        raise RuntimeError('Door Stage1 reuse points to a different source')
    markers = sorted((source / 'completed_variants').glob('*_stage1.json'))
    for marker in markers:
        link = destination / 'completed_variants' / marker.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(marker.resolve())

    if not (destination / 'unbounded_inference_complete.json').is_file():
        sumloss.reuse_compute_cache('door', source / 'prepared', prepared)
        subprocess.run([
            str(ROOT / '.venv/bin/python'),
            str(ROOT / 'scripts/evaluation/infer_real97_unbounded_context.py'),
            '--config', str(config_path), '--task', 'door',
            '--prepared', str(prepared), '--output', str(destination),
        ], cwd=ROOT, check=True)
    if not (destination / 'globalmean_metric_pipeline_complete.json').is_file():
        globalmean.score('door', destination)
    sumloss.summarize('door', 8, destination)


def stick(destination):
    original_audit = common.audit
    k4 = {entry['environment']: entry['episodes']
          for entry in common.read(STICK_K4_PROTOCOL)['selection']}

    def audit_with_k4_prefix(plan, supports, queries, k, output):
        for env in plan['environments']:
            observed = [supports[index]['episode_index'] for index in env['support_indices'][:4]]
            if observed != k4[env['environment']]:
                raise RuntimeError(f'Stick K=4 support prefix changed: {env["environment"]}')
        return original_audit(plan, supports, queries, k, output)

    common.audit = audit_with_k4_prefix
    try:
        if not (destination / 'support_number_inference_complete.json').is_file():
            common.stick(8, destination)
            common.write(destination / 'support_number_inference_complete.json',
                         {'task': 'stick', 'K': 8, 'source_K4': str(STICK_K4_PROTOCOL)})
    finally:
        common.audit = original_audit
    if not (destination / 'metric_pipeline_complete.json').is_file():
        common.score('stick', destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('door', 'stick'), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    os.chdir(ROOT)
    output = DEST / args.task / 'k8'
    output.mkdir(parents=True, exist_ok=True)
    (door if args.task == 'door' else stick)(output)


if __name__ == '__main__':
    main()
