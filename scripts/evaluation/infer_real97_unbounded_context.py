#!/usr/bin/env python3
"""Opt-in initialization/bounds policy around the unchanged reference evaluator."""

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_cluster_shared_noise(config, shape):
    """One seeded coordinate-wise noise vector, independent of environment ID."""
    import torch
    low, high = float(config['noise_min']), float(config['noise_max'])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError('Invalid shared initialization noise range')
    generator = torch.Generator(device='cpu').manual_seed(int(config['noise_seed']))
    return torch.rand(shape, generator=generator, dtype=torch.float32) * (high - low) + low


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['door', 'ball'], required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--finalize-only', action='store_true')
    cli = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Inference must run on a compute allocation')
    config = read_json(cli.config)
    setting = config['tasks'][cli.task]
    prepared, output = cli.prepared.resolve(), cli.output.resolve()
    plan = read_json(prepared / 'plan.json')
    if setting.get('supplement_existing_dataset_cache', False) and (output / 'runtime.yaml').exists():
        import fcntl
        import subprocess
        import yaml
        runtime = yaml.safe_load((output / 'runtime.yaml').read_text())
        flat = {key: value for section in runtime.values() if isinstance(section, dict)
                for key, value in section.items()}
        cached_dataset = Path(flat['dataset_base_path'])
        allowed_cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache/eval_datasets'
        if cached_dataset.is_dir() and cached_dataset.resolve().is_relative_to(allowed_cache.resolve()):
            with (cached_dataset.parent / (cached_dataset.name + '.support_supplement.lock')).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                subprocess.run(['rsync', '-a', '--ignore-existing',
                                '--files-from=' + str(prepared / 'stage_files.txt'),
                                str(Path(plan['source_dataset'])) + '/', str(cached_dataset) + '/'], check=True)
            print('[support_cache] added missing recorded files without replacing existing cache contents', flush=True)
    if setting.get('require_exact_original_support'):
        source = ROOT / setting['source_prepared']
        old_plan = read_json(source / 'plan.json')
        old_support = read_jsonl(source / 'support.jsonl')
        new_support = read_jsonl(prepared / 'support.jsonl')
        old_groups = {row['environment']: row for row in old_plan['environments']}
        replacement_levels = setting.get('support_levels', {})
        replacement_environments = set(replacement_levels)
        if replacement_environments and not setting.get('partial_stage2_replacement', False):
            raise ValueError('Support replacement requires partial_stage2_replacement=true')
        plan_environments = {row['environment'] for row in plan['environments']}
        missing_replacements = replacement_environments - plan_environments
        if missing_replacements:
            raise ValueError(f'Support replacement environments missing from plan: {sorted(missing_replacements)}')
        for row in plan['environments']:
            environment = row['environment']
            before = [old_support[i] for i in old_groups[environment]['support_indices']]
            after = [new_support[i] for i in row['support_indices']]
            if environment not in replacement_environments:
                if before != after:
                    raise ValueError(f'Original Support changed unexpectedly: {environment}')
                continue
            allowed_levels = {int(level) for level in replacement_levels[environment]}
            if not after:
                raise ValueError(f'No replacement Support selected: {environment}')
            for support in after:
                if support.get('environment') != environment:
                    raise ValueError(f'Replacement Support environment mismatch: {environment}')
                if support.get('dataset_split') != 'train':
                    raise ValueError(f'Replacement Support is not train-only: {environment}')
                if int(support.get('action_level', -1)) not in allowed_levels:
                    raise ValueError(f'Replacement Support level is not explicitly allowed: {environment}')
            print('[support_replacement]', json.dumps({
                'environment': environment,
                'allowed_levels': sorted(allowed_levels),
                'episodes': [support['episode_index'] for support in after],
            }), flush=True)

    import torch
    from scripts import infer_stage2_ttt as core
    # Both supported import spellings refer to the same adapted-state factory.
    sys.modules.setdefault('infer_stage2_ttt', core)
    table = core._load_grouped_context_table(prepared / 'reference/context_table.json')
    targets = {float(mu): torch.tensor(value, dtype=torch.float32)
               for mu, value in zip(table['friction_values'], table['contexts'])}
    environments = {row['environment']: row for row in plan['environments']}
    initializations = output / 'initializations'
    original_adapt = core._adapt_ttt_state
    shared_noise = None
    if setting['initial_context'] == 'explicit_training_cluster_mean_plus_shared_noise':
        shared_noise = sample_cluster_shared_noise(config, next(iter(targets.values())).shape)
        write_json(output / 'shared_initialization_noise.json', {
            'seed': int(config['noise_seed']), 'range_per_coordinate': [config['noise_min'], config['noise_max']],
            'noise': shared_noise.tolist(), 'noise_l2': float(shared_noise.norm()),
            'scope': 'one identical vector for every cluster and environment in this task',
            'resampled_per_environment': False, 'hard_bounds': None,
        })

    @functools.wraps(original_adapt)
    def adapt_with_policy(*positional, **keyword):
        args = positional[2] if len(positional) > 2 else keyword['args']
        pipe = positional[0] if positional else keyword['pipe']
        meta = dict(keyword.get('trajectory_meta') or {})
        environment = meta.get('environment')
        if environment not in environments:
            raise ValueError(f'Missing exact environment identity for initialization: {meta}')
        target = targets[float(environments[environment]['environment_index'])].clone()
        policy = setting['initial_context']
        seed = None
        noise = torch.zeros_like(target)
        if policy == 'corresponding_environment_table_plus_uniform_noise':
            stable_id = int(hashlib.sha256(environment.encode()).hexdigest()[:8], 16)
            seed = (int(config['noise_seed']) + stable_id) % (2**63 - 1)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            noise = torch.rand(target.shape, generator=generator, dtype=torch.float32)
            noise = noise * (float(config['noise_max']) - float(config['noise_min'])) + float(config['noise_min'])
            initial = target + noise
        elif policy == 'mean_training_table':
            initial = torch.tensor(table['mean_context'], dtype=torch.float32)
        elif policy == 'global_table_box_random_shared':
            trial = int(config['global_random_trial'])
            seed = int(config['global_random_seed'])
            initial = torch.tensor(config['global_random_starts'][trial], dtype=torch.float32).reshape_as(target)
            meta.update(global_random_trial=trial, known_family_prior=False,
                        query_GT_used_for_initialization=False,
                        initialization_scope='same global start for every environment')
        elif policy in ('explicit_training_cluster_mean', 'explicit_training_cluster_mean_plus_shared_noise'):
            group = setting['environment_initialization_group'][environment]
            member_ids = setting['initialization_groups'][group]
            source_table = read_json(prepared / 'reference/context_table.json')
            member_rows = [row for row in source_table['records']
                           if int(row['source_virtual_group_id']) in member_ids]
            if len(member_rows) != len(set(member_ids)) or len(member_ids) != len(set(member_ids)):
                raise ValueError('Missing or duplicate cluster members')
            initial = torch.stack([torch.tensor(row['context'], dtype=torch.float32)
                                   for row in member_rows]).mean(0)
            if shared_noise is not None:
                seed = int(config['noise_seed'])
                noise = shared_noise.clone()
                initial = initial + noise
                meta.update(shared_noise=True, shared_noise_seed=seed,
                            noise_scope='task_global_same_vector', noise_l2=float(noise.norm()))
            meta.update(initialization_group=group, cluster_source_virtual_group_ids=member_ids,
                        known_family_prior=True, query_GT_used_for_initialization=False)
        else:
            raise ValueError(f'Unsupported explicit context initialization: {policy}')
        if not config.get('disable_context_clamp'):
            raise ValueError('This entrypoint is exclusively for unbounded-context experiments')
        # The legacy update uses clamp(min, max). Infinite endpoints preserve
        # every finite value and impose no projection or norm restriction.
        args.stage2_context_clamp_min = -math.inf
        args.stage2_context_clamp_max = math.inf
        if float(args.stage2_context_reg_weight) != 0.0:
            raise ValueError('This experiment requires Support loss only, without context regularization')
        meta.update(initial_context_policy=policy, initial_context_seed=seed,
                    context_hard_bounds=None,
                    known_environment_initialization=(policy == 'corresponding_environment_table_plus_uniform_noise'))
        keyword.update(initial_context=initial.to(device=pipe.device, dtype=torch.float32),
                       target_context=target.to(device=pipe.device, dtype=torch.float32),
                       trajectory_meta=meta)
        record = {
            'environment': environment, 'policy': policy, 'seed': seed,
            'training_context': target.tolist(), 'noise': noise.tolist(),
            'initial_context': initial.tolist(), 'initial_clamp_applied': False,
            'effective_context_bounds': None, 'noise_l2': float(noise.norm()),
            'initial_distance_to_training_context': float((initial-target).norm()),
            'initial_coordinates_outside_previous_bounds': int(((initial < -1) | (initial > 1)).sum()),
        }
        if policy in ('explicit_training_cluster_mean', 'explicit_training_cluster_mean_plus_shared_noise'):
            record.update(initialization_group=group, cluster_source_virtual_group_ids=member_ids,
                          known_family_prior=True)
            if shared_noise is not None:
                record.update(cluster_center=(initial-noise).tolist(), shared_noise=True,
                              noise_scope='task_global_same_vector')
        write_json(initializations / (environment + '.json'), record)
        print('[unbounded_initialization]', json.dumps(record), flush=True)
        result = original_adapt(*positional, **keyword)
        if not torch.isfinite(result[0]).all():
            raise FloatingPointError(f'Nonfinite adapted context: {environment}')
        observed_initial = torch.tensor(result[4][0]['context_flat'], dtype=torch.float32).reshape_as(initial)
        if not torch.allclose(observed_initial, initial, atol=1e-6, rtol=0):
            raise ValueError('The optimized initial context differs from the requested unprojected value')
        record['final_distance_to_training_context'] = float((result[0].detach().float().cpu()-target).norm())
        record['final_distance_to_initial_context'] = float((result[0].detach().float().cpu()-initial).norm())
        record['final_context'] = result[0].detach().float().cpu().tolist()
        write_json(initializations / (environment + '.json'), record)
        return result

    def synchronize_contexts():
        for path in initializations.glob('*.json'):
            if setting.get('rerun_environments') and path.stem not in setting['rerun_environments']:
                continue
            context_path = output / 'contexts' / path.name
            if not context_path.exists():
                continue
            initialization = read_json(path)
            context = read_json(context_path)
            context['initial'] = initialization['initial_context']
            context['initialization_policy'] = initialization['policy']
            context['initialization_seed'] = initialization['seed']
            context['effective_context_bounds'] = None
            write_json(context_path, context)

    core._adapt_ttt_state = adapt_with_policy
    from scripts.evaluation import infer_real97_ball_door_reference as reference
    synchronize_contexts()
    if not cli.finalize_only:
        sys.argv = [str(ROOT / 'scripts/evaluation/infer_real97_ball_door_reference.py'),
                    '--prepared', str(prepared), '--output', str(output)]
        if config.get('stage2_only', False):
            sys.argv.append('--stage2-only')
        reference.main()
    synchronize_contexts()
    records, trajectory = [], []
    finalized_environments = list(environments)
    if setting.get('partial_stage2_replacement', False):
        finalized_environments = sorted(setting.get('support_levels', {}))
        if not finalized_environments:
            raise ValueError('Partial Stage2 replacement has no declared Support environments')
    for environment in finalized_environments:
        record = read_json(initializations / (environment + '.json'))
        if 'final_context' not in record:
            raise ValueError(f'Incomplete context adaptation: {environment}')
        records.append(record)
        trajectory.extend(read_json(output / 'contexts' / (environment + '.json'))['trajectory'])
    write_json(output / 'initialization_distances.json', records)
    core._write_context_pca_plot(output / 'initialization_context_trajectory.svg', table, trajectory)
    provenance = read_json(output / 'provenance.json')
    provenance['effective_context_bounds'] = None
    provenance['initialization_policy'] = setting['initial_context']
    provenance['legacy_finite_parser_bounds_overridden'] = True
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'unbounded_inference_complete.json', {
        'task': cli.task, 'environments': len(environments),
        'finalized_initialization_environments': finalized_environments,
        'context_bounds': None,
        'initialization': setting['initial_context'],
        'support_and_query_unchanged': bool(
            setting.get('require_exact_original_support', False)
            and not setting.get('partial_stage2_replacement', False)
        ),
        'support_replacement_environments': sorted(setting.get('support_levels', {})),
        'query_windows_unchanged': True,
        'rerun_environments': setting.get('rerun_environments', list(environments)),
        'actual_slurm_job_id': os.environ['SLURM_JOB_ID'],
        'finalize_only': cli.finalize_only,
        'formal_metric_approved': False,
    })


if __name__ == '__main__':
    main()
