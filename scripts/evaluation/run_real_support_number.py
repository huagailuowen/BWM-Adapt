#!/usr/bin/env python3
"""Isolated support-count experiments using the published real-task runners."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/support_number_analysis/real'
RUNS = [('ball', 1), ('ball', 2), ('ball', 4), ('door', 1), ('door', 4),
        ('stick', 1), ('stick', 4), ('soft', 1), ('soft', 2)]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def write_rows(path, values):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in values))


def frozen_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def audit(plan, supports, queries, k, destination):
    forbidden = {(r['environment'], r['episode_index']) for r in queries}
    records = []
    for env in plan['environments']:
        selected = [supports[i] for i in env['support_indices']]
        if len(selected) != k or len({r['episode_index'] for r in selected}) != k:
            raise RuntimeError('Incorrect support count or duplicate trajectory: ' + env['environment'])
        for row in selected:
            if (row['dataset_split'] != 'train' or row['environment'] != env['environment']
                    or (row['environment'], row['episode_index']) in forbidden):
                raise RuntimeError('Support/query leakage')
        records.append({'environment': env['environment'], 'support_indices': env['support_indices'],
                        'episodes': [r['episode_index'] for r in selected],
                        'levels': [r.get('action_level') for r in selected],
                        'test_query_indices': [i for i in env['query_indices']
                                               if queries[i]['dataset_split'] == 'test']})
    write(destination / 'support_protocol.json', {
        'K': k, 'loss_reduction': 'mean', 'query_GT_for_adaptation': False,
        'primary_split': 'test', 'action_candidates': 'model predictions only',
        'query_windows_and_seeds': 'unchanged formal cohort; original row indices retained',
        'selection': records})


def ball_door(task, k, out):
    import prepare_real97_support_override as prep
    base = ROOT / f'outputs/evaluation_real97_{task}_dual6500_simlr_cluster_20260918_v1'
    cfg = read(base / 'resolved_config.json')
    if task == 'ball':
        previous = read(ROOT / 'outputs/evaluation_real97_ball_dual6500_simlr_cluster_supportL7_ball7_ball9_20260919_v1/request_config_job120635.json')
        source_prepared = ROOT / previous['tasks']['ball']['prepared']
        source_output = ROOT / previous['output']
    else:
        source_prepared = base / 'door_prep'
        source_output = ROOT / 'outputs/infer_real97_door_dual6500_simlr_cluster_20260918_v1'
    plan = read(source_prepared / 'plan.json')
    support = rows(source_prepared / 'support.jsonl')
    queries = rows(source_prepared / 'query.jsonl')
    templates = rows(source_prepared / 'extension_action_templates.jsonl')
    forbidden = {(r['environment'], r['episode_index']) for r in queries}
    requested = {}
    for env in plan['environments']:
        name = env['environment']
        old = [support[i] for i in env['support_indices']]
        if task == 'ball' and name in ('ball-0', 'ball-1'):
            selected = [next(r for r in old if int(r['action_level']) == 2)] if k == 1 else list(old)
        else:
            selected = list(old[:k])
        pool = [r for r in templates if r['environment'] == name and r['dataset_split'] == 'train'
                and (name, r['episode_index']) not in forbidden]
        anchor = sum(float(r['action_level']) for r in old) / len(old)
        pool.sort(key=lambda r: (abs(float(r['action_level']) - anchor), int(r['action_level']), r['episode_index']))
        for row in pool:
            if len(selected) >= k:
                break
            if (row['episode_index'] not in {x['episode_index'] for x in selected}
                    and row['action_level'] not in {x['action_level'] for x in selected}):
                selected.append(row)
        if len(selected) != k:
            raise RuntimeError('Insufficient distinct legal train supports: ' + name)
        levels = [int(r['action_level']) for r in selected]
        if levels != [int(r['action_level']) for r in old]:
            requested[name] = levels
    prepared = out / 'prepared'
    setting = cfg['tasks'][task]
    setting.update(source_prepared=str(source_prepared), source_inference=str(source_output),
                   prepared=str(prepared), support_levels=requested,
                   initial_context='explicit_training_cluster_mean',
                   partial_stage2_replacement=True, require_exact_original_support=True)
    cfg['output'] = str(out)
    cfg['support_number_experiment'] = {'K': k, 'loss_reduction': 'mean'}
    config = out / 'config.json'
    write(config, cfg)
    # Reuse original preparation and cached predictions without its old output naming.
    prep.locations = lambda task, job_id: (prepared, out)
    prep.prepare(config, task, 'support_number')
    plan = read(prepared / 'plan.json')
    for env in plan['environments']:
        env['extension_template_indices'] = []  # No unpaired counterfactual sweeps.
    plan['protocol']['ttt']['ttt_support_loss_reduction'] = 'mean'
    write(prepared / 'plan.json', plan)
    audit(plan, rows(prepared / 'support.jsonl'), queries, k, out)
    for env in plan['environments']:
        name = env['environment']
        if name not in requested:
            initialization = source_output / 'initializations' / (name + '.json')
            if not initialization.is_file():
                initialization = (ROOT / f'outputs/infer_real97_{task}_dual6500_simlr_cluster_20260918_v1'
                                  / 'initializations' / (name + '.json'))
            frozen_file(initialization, out / 'initializations' / (name + '.json'))
    # All K variants share the immutable checkpoint cache, even though manifests differ.
    from infer_real97_standard_reference import copy_cached
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache/eval_checkpoints'
    common = cache / f'real_support_{task}_dual6500.safetensors'
    copy_cached(prepared / 'reference/model.safetensors', common)
    alias = cache / (hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16] + '.safetensors')
    if not alias.exists():
        alias.symlink_to(common)
    Path(str(alias) + '.copy_complete').write_text('complete\n')
    import infer_real97_unbounded_context as runner
    sys.argv = ['infer_real97_unbounded_context', '--config', str(config), '--task', task,
                '--prepared', str(prepared), '--output', str(out)]
    runner.main()


def soft(k, out):
    import infer_real97_soft_family_mean_static as runner
    cfg = read(ROOT / 'configs/evaluation/real97_soft_ours_simlr_mean_20260918_v1.json')
    cfg['support_number_experiment'] = {'K': k, 'loss_reduction': 'mean'}
    cfg['output'] = str(out)
    config = out / 'config.json'
    write(config, cfg)
    original_prepare = runner.prepare

    def prepare(config, output):
        plan, runtime, checkpoint, queries, supports = original_prepare(config, output)
        for env in plan['environments']:
            indices = list(env['support_indices'])
            first = indices[0]
            opposite = next((i for i in indices[1:]
                             if supports[i].get('support_direction') != supports[first].get('support_direction')), None)
            if opposite is None:
                raise RuntimeError('Expected opposite-direction historical Soft support')
            env['support_indices'] = [first] if k == 1 else [first, opposite]
        audit(plan, supports, queries, k, output)
        plan['active_support_count'] = k * len(plan['environments'])
        write(output / 'plan.json', plan)
        import yaml
        parsed = yaml.safe_load(runtime.read_text())
        parsed['inference']['ttt_support_loss_reduction'] = 'mean'
        runtime.write_text(yaml.safe_dump(parsed, sort_keys=False))
        return plan, runtime, checkpoint, queries, supports

    runner.prepare = prepare
    sys.argv = ['infer_real97_soft_family_mean_static', '--config', str(config), '--output', str(out)]
    runner.main()


def stick(k, out):
    cfg = read(ROOT / 'configs/evaluation/real97_stick_ours_simlr_mean_20260918_v1.json')
    old = ROOT / cfg['prepared']
    source_output = ROOT / cfg['methods']['ours']['output']
    prepared = out / 'prepared'
    prepared.mkdir(parents=True, exist_ok=True)
    plan = read(old / 'plan.json')
    queries, support = rows(old / 'query.jsonl'), rows(old / 'support.jsonl')
    original_count = len(support)
    forbidden = {(r['environment'], r['episode_index']) for r in queries}
    windows = {}
    for row in rows(ROOT / cfg['manifest'] / 'train.jsonl'):
        key = row['environment'], row['episode_index']
        if row['dataset_split'] == 'train' and row['sampling_kind'] == 'lift' and key not in forbidden:
            windows.setdefault(key, []).append(row)
    annotations = {(r['environment'], r['episode_index']): r for r in
                   rows(Path(plan['source_dataset']) / 'episode_split_manifest.jsonl') if r['split'] == 'train'}
    for env in plan['environments']:
        name = env['environment']
        env['support_indices'] = list(env['support_indices'][:k])
        chosen = {support[i]['episode_index'] for i in env['support_indices']}
        center = env['train_only_balance_fraction']
        if center is None:
            # One-sided formal cohorts retained a balanced training support but
            # never recorded a two-sided bracket center. Use that train-only
            # reference, not a query outcome or an invented midpoint.
            balanced_positions = [float(position) for position, outcome in
                                  zip(env['support_fraction'], env['support_outcomes'])
                                  if position is not None and outcome == 'balanced']
            if not balanced_positions:
                raise RuntimeError('Missing train-only balance reference: ' + name)
            center = sum(balanced_positions) / len(balanced_positions)
            env['support_number_balance_reference'] = center
            env['support_number_balance_reference_source'] = 'historical_balanced_training_support'
        candidates = [r for key, r in annotations.items() if key in windows and key[0] == name
                      and r.get('vision_ok') and r.get('support_fraction') is not None
                      and r['episode_index'] not in chosen]
        candidates.sort(key=lambda r: (abs(float(r['support_fraction']) - center), r['episode_index']))
        for item in candidates:
            if len(env['support_indices']) >= k:
                break
            pool = sorted(windows[(name, item['episode_index'])], key=lambda r: r['start_frame'])
            row = copy.deepcopy(pool[(len(pool)-1)//2])
            env['support_indices'].append(len(support))
            support.append(row)
        selected = [annotations[(name, support[i]['episode_index'])] for i in env['support_indices']]
        env.update(support_episode_indices=[r['episode_index'] for r in selected],
                   support_outcomes=[r['outcome'] for r in selected],
                   support_fraction=[r['support_fraction'] for r in selected],
                   support_tilt_change_deg=[r.get('angle_change_deg') for r in selected])
    audit(plan, support, queries, k, out)
    for name in ['query.jsonl', 'action_stats.json']:
        frozen_file(old / name, prepared / name)
    for name in ['ours', 'standard']:
        if not (prepared / name).exists():
            (prepared / name).symlink_to(old / name, target_is_directory=True)
    write_rows(prepared / 'support.jsonl', support)
    write(prepared / 'plan.json', plan)
    files = set((old / 'stage_files.txt').read_text().splitlines())
    for row in support:
        files.update(row['video'])
        files.add(row['action'])
    (prepared / 'stage_files.txt').write_text('\n'.join(sorted(files)) + '\n')
    cfg['prepared'] = str(prepared)
    cfg['reuse_query_reference_from'] = str(source_output)
    cfg.pop('reuse_contexts_from', None)
    cfg['methods']['ours']['output'] = str(out)
    cfg['support_number_experiment'] = {'K': k, 'loss_reduction': 'mean'}
    # Keep the original stage2 config byte-equivalent for reference-reuse checks.
    config = out / 'config.json'
    write(config, cfg)
    write(prepared / 'evaluation_config.json', cfg)
    write(prepared / 'prepared_complete.json', {'K': k, 'support_count': len(support)})
    import infer_real916_stick_simlr_globalmean as runner
    original_read = runner.read_rows

    def read_with_reference_padding(path):
        result = original_read(path)
        if Path(path).resolve() == (source_output / 'input_manifest/support.jsonl').resolve():
            # Newly appended indices have no old counterpart; never reuse their Z.
            result += [None] * (len(support) - original_count)
            old_environments = {e['environment']: e for e in read(old / 'plan.json')['environments']}
            for env in plan['environments']:
                previous_indices = old_environments[env['environment']]['support_indices']
                current_indices = env['support_indices']
                unchanged = (len(previous_indices) == len(current_indices)
                             and all(result[before] == support[after]
                                     for before, after in zip(previous_indices, current_indices)))
                if not unchanged:
                    # The legacy runner compares only selected rows, so a strict
                    # subset otherwise falsely matches its old multi-support Z.
                    # Invalidate only this process-local reuse view, not files.
                    for index in current_indices:
                        result[index] = None
        return result

    runner.read_rows = read_with_reference_padding
    sys.argv = ['infer_real916_stick_simlr_globalmean', '--config', str(config), '--method', 'ours']
    runner.main()


def score(task, out):
    cpu = ROOT / '.venv-real97-eval-20260909/bin/python'

    def run(script, *args, gpu=False):
        python = ROOT / '.venv/bin/python' if gpu else cpu
        subprocess.run([str(python), str(ROOT / 'scripts/evaluation' / script), *map(str, args)], check=True)

    if task in ('ball', 'door'):
        name = ('real97_ball_dual6500_simlr_supportL7_ball7_ball9_scores_20260920_v1.json'
                if task == 'ball' else 'real97_door_simlr_scores_20260918_v1.json')
        cfg = read(ROOT / 'configs/evaluation' / name)
        cfg['tasks'][task]['ours'] = str(out)
        cfg['output'] = str(out / 'metrics')
        path = out / 'metrics_config.json'
        write(path, cfg)
        run(f'score_real97_{task}_simlr6500.py', '--config', path, '--mode', 'cpu', '--workers', '8')
        run(f'score_real97_{task}_simlr6500.py', '--config', path, '--mode', 'lpips', gpu=True)
    elif task == 'soft':
        cfg = read(ROOT / 'configs/evaluation/real97_soft_simlr_scores_20260918_v1.json')
        cfg.update(source_inference=str(out), output=str(out / 'metrics'), shards=1)
        path = out / 'metrics_config.json'
        write(path, cfg)
        run('score_real97_soft_simlr.py', '--config', path, '--shard', '0')
        run('score_real97_soft_simlr.py', '--config', path, '--aggregate-only')
    else:
        cfg = read(ROOT / 'configs/evaluation/real97_stick_simlr_scores_20260918_v1.json')
        cfg.update(ours=str(out), prepared=str(out / 'prepared'), output=str(out / 'metrics'))
        path = out / 'metrics_config.json'
        write(path, cfg)
        run('score_real916_stick_final.py', '--config', path, '--mode', 'cpu', '--workers', '8')
        run('score_real916_stick_final.py', '--config', path, '--mode', 'lpips', gpu=True)
        tail = read(ROOT / 'configs/evaluation/real97_stick_simlr_tail_20260918_v1.json')
        tail.update(source_metrics=str(out / 'metrics'), prepared=str(out / 'prepared'),
                    output=str(out / 'metrics_visible_tail'))
        path = out / 'tail_metrics_config.json'
        write(path, tail)
        run('score_real916_stick_visible_tail.py', '--config', path)
    write(out / 'metric_pipeline_complete.json', {
        'task': task, 'formal_action_cohort_aggregation_pending': task in ('soft', 'stick'),
        'note': 'Retain the formal shared6/test12 Soft and seeded36 Stick cohorts in the final table.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, choices=range(len(RUNS)))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required; no login-node inference or metrics')
    os.chdir(ROOT)
    task, k = RUNS[args.index]
    out = DEST / task / f'k{k}'
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'support_number_inference_complete.json').exists():
        if task in ('ball', 'door'):
            ball_door(task, k, out)
        elif task == 'soft':
            soft(k, out)
        else:
            stick(k, out)
        write(out / 'support_number_inference_complete.json', {'task': task, 'K': k})
    if not (out / 'metric_pipeline_complete.json').exists():
        score(task, out)


if __name__ == '__main__':
    main()
