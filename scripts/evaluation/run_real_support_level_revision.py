#!/usr/bin/env python3
"""Explicit, post-hoc support-level experiments; preserve the original results."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys

import run_real_support_number as common

ROOT = common.ROOT
SPECS = [
    ('ball', 2, {'ball-2': [2, 4], 'ball-9': [7, 8]}),
    ('ball', 4, {'ball-2': [2, 3, 4, 3], 'ball-9': [7, 7, 8, 8]}),
    ('door', 4, {'door-0': [2, 5, 1, 4], 'door-6u-7d': [6, 9, 5, 7],
                 'door-6u-8d': [7, 10, 8, 9]}),
]


def prepare(task, k, requested, out):
    import prepare_real97_support_override as override
    old = common.DEST / task / f'k{k}'
    source = old / 'prepared'
    config = common.read(old / 'config.json')
    plan = common.read(source / 'plan.json')
    support = common.rows(source / 'support.jsonl')
    queries = common.rows(source / 'query.jsonl')
    templates = common.rows(source / 'extension_action_templates.jsonl')
    forbidden = {(r['environment'], r['episode_index']) for r in queries}
    groups = {e['environment']: e for e in plan['environments']}
    manifest = ROOT / plan['setting']['training_run'] / 'input_manifest'
    windows = {}
    for row in common.rows(manifest / 'train.jsonl'):
        env = row.get('physical_environment', row['environment'].split('__z')[0].removesuffix('_lerobot'))
        key = env, row['episode_index']
        if (env in requested and row['dataset_split'] == 'train'
                and row['sampling_kind'] == 'precise' and key not in forbidden
                and row.get('latent_replica', 0) == 0):
            windows.setdefault(key, []).append(row)
    events = {(r['environment'].removesuffix('_lerobot'), r['episode_index']): r
              for r in common.rows(manifest / 'chunk_events.jsonl')}
    extras = []
    for env in requested:
        metadata = common.rows(Path(plan['source_dataset']) / (env + '_lerobot') / 'meta/episodes.jsonl')
        levels = {r['episode_index']: int(r['skill_gear'] if task == 'ball' else r['level'])
                  for r in metadata}
        for (name, episode), candidates in sorted(windows.items()):
            if name != env or levels[episode] not in requested[env]:
                continue
            candidates.sort(key=lambda r: r['start_frame'])
            row = copy.deepcopy(candidates[(len(candidates) - 1) // 2])
            event = events[(env, episode)]
            if event['split'] != 'train' or int(event['num_frames']) != int(row['total_frames']):
                raise RuntimeError('Training window/event mismatch')
            for key in ['latent_replica', 'virtual_environment_id']:
                row.pop(key, None)
            row.update(environment=env, environment_index=groups[env]['environment_index'],
                       friction_mu=float(groups[env]['environment_index']), action_level=levels[episode],
                       sample_id=row.get('source_sample_id', row['sample_id'].split('__z')[0]),
                       source_event_annotation=event['events'],
                       support_candidate_source='frozen_train_precise_window_distinct_episode')
            native = [min(row['start_frame'] + i * row['frame_stride'], row['end_frame'])
                      for i in range(row['length'])]
            row.update(native_frame_indices=native,
                       evaluation_frame_indices=[i for i in range(1, len(native)) if native[i] > native[i-1]])
            extras.append(row)
    selected = {}
    for env, desired in requested.items():
        pool = [support[i] for i in groups[env]['support_indices']] + templates + extras
        chosen, used = [], set()
        for level in desired:
            candidate = next((r for r in pool if r['environment'] == env
                              and int(r['action_level']) == level and r['dataset_split'] == 'train'
                              and (env, r['episode_index']) not in forbidden
                              and r['episode_index'] not in used), None)
            if candidate is None:
                raise RuntimeError(f'No distinct query-disjoint train episode: {env}, Level {level}')
            chosen.append(copy.deepcopy(candidate))
            used.add(candidate['episode_index'])
        selected[env] = chosen
    # A private source adds legal template candidates; old preparation remains untouched.
    augmented = out / 'source_prepared'
    augmented.mkdir(parents=True, exist_ok=True)
    for name in ['plan.json', 'experiment.json', 'query.jsonl', 'support.jsonl', 'stage_files.txt']:
        common.frozen_file(source / name, augmented / name)
    if not (augmented / 'reference').exists():
        (augmented / 'reference').symlink_to(source / 'reference', target_is_directory=True)
    common.write_rows(augmented / 'extension_action_templates.jsonl', templates + extras)
    prepared = out / 'prepared'
    config['output'] = str(out)
    config['support_level_revision'] = {
        'requested': requested, 'distinct_episodes_for_repeated_levels': True,
        'selection_status': 'post-hoc exploratory revision after inspecting previous failures',
        'source_output': str(old), 'support_loss_reduction': 'mean'}
    setting = config['tasks'][task]
    setting.update(source_prepared=str(augmented), source_inference=str(old), prepared=str(prepared),
                   support_levels=requested, partial_stage2_replacement=True,
                   require_exact_original_support=True)
    path = out / 'config.json'
    common.write(path, config)
    override.locations = lambda task, job_id: (prepared, out)
    override.prepare(path, task, 'explicit_levels')
    # Legacy preparation selects a single candidate per level. Resolve repeated
    # levels explicitly to different trajectories before any adaptation runs.
    newplan = common.read(prepared / 'plan.json')
    newrows = common.rows(prepared / 'support.jsonl')
    for env in newplan['environments']:
        name = env['environment']
        env['extension_template_indices'] = []
        if name in selected:
            indices = []
            for row in selected[name]:
                row.pop('template_index', None)
                if row in newrows:
                    index = newrows.index(row)
                else:
                    index = len(newrows)
                    newrows.append(row)
                indices.append(index)
            env['support_indices'] = indices
        else:
            common.frozen_file(old / 'initializations' / (name + '.json'),
                               out / 'initializations' / (name + '.json'))
    newplan['protocol']['ttt']['ttt_support_loss_reduction'] = 'mean'
    newplan['active_support_count'] = sum(len(e['support_indices']) for e in newplan['environments'])
    common.write(prepared / 'plan.json', newplan)
    common.write_rows(prepared / 'support.jsonl', newrows)
    files = set((prepared / 'stage_files.txt').read_text().splitlines())
    for row in newrows:
        files.update(row['video'])
        files.add(row['action'])
    (prepared / 'stage_files.txt').write_text('\n'.join(sorted(files)) + '\n')
    common.audit(newplan, newrows, queries, k, out)
    common.write(out / 'selected_supports.json', selected)
    return path, prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    os.chdir(ROOT)
    task, k, requested = SPECS[args.index]
    out = common.DEST / 'support_level_revision_20260922' / task / f'k{k}'
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'support_number_inference_complete.json').exists():
        config, prepared = prepare(task, k, requested, out)
        from infer_real97_standard_reference import copy_cached
        cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache/eval_checkpoints'
        checkpoint = cache / f'real_support_{task}_dual6500.safetensors'
        copy_cached(prepared / 'reference/model.safetensors', checkpoint)
        alias = cache / (hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16] + '.safetensors')
        if not alias.exists():
            alias.symlink_to(checkpoint)
        Path(str(alias) + '.copy_complete').write_text('complete\n')
        import infer_real97_unbounded_context as runner
        sys.argv = ['infer_real97_unbounded_context', '--config', str(config), '--task', task,
                    '--prepared', str(prepared), '--output', str(out)]
        runner.main()
        common.write(out / 'support_number_inference_complete.json', {'task': task, 'K': k})
    if not (out / 'metric_pipeline_complete.json').exists():
        common.score(task, out)
    report = common.read(out / 'metrics/summary_provisional.json')
    action = next(x for x in report['action'] if x['method'] == 'ours_stage2')
    if task == 'door':
        protocol = common.read(ROOT / 'results/real97_all_methods_main_table_v1/metrics/door_first_close_level_tolerance1_20260919.json')
        indexed = {d['environment']: d for d in action['decisions']}
        decisions = []
        for env, truth in protocol['ground_truth_lowest_level'].items():
            states = indexed[env]['closed_by_level']
            if {int(x) for x in states} != set(range(1, 11)):
                raise RuntimeError('Incomplete action candidates')
            predicted = next((i for i in range(1, 11) if states[str(i)] is True), None)
            error = abs(predicted - truth) if predicted is not None else None
            decisions.append(dict(environment=env, gt_lowest_level=truth, predicted_lowest_level=predicted,
                                  score=1.0 if error == 0 else .5 if error == 1 else 0.0))
        action = dict(metric=protocol['metric'], decisions=decisions,
                      macro_score=sum(x['score'] for x in decisions) / len(decisions))
    common.write(out / 'formal_action.json', action)
    print('[formal_action]', task, k, action['macro_score'], flush=True)


if __name__ == '__main__':
    main()
