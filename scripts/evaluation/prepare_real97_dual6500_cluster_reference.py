#!/usr/bin/env python3
"""Resolve final-table cluster centers after a successful dual continuation."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.evaluation.prepare_real97_dual_latent_reference import prepare


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['door', 'ball'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Preparation requires a compute allocation')
    config = json.loads(args.config.read_text())
    setting = config['tasks'][args.task]
    run = Path(setting['training_run'])
    step = setting['model_step']
    if step != 6500 or setting['table_step'] != step:
        raise ValueError('Only the matching final step-6500 model/table is allowed')
    for name in (f'step-{step}.safetensors', f'.step-{step}.safetensors.complete',
                 f'step-{step}.context_table.json'):
        path = run / name
        if not path.is_file() or (not name.startswith('.') and not path.stat().st_size):
            raise RuntimeError(f'Missing completed final bundle: {path}')
    table_bytes = (run / f'step-{step}.context_table.json').read_bytes()
    records = json.loads(table_bytes)['records']
    rows = {int(row['friction_mu']): row['context'] for row in records}
    if len(rows) != setting['expected_latents'] or len(records) != len(rows):
        raise ValueError('Unexpected or duplicate final table entries')
    summary = json.loads((run / 'input_manifest/manifest_summary.json').read_text())
    physical = summary['physical_environment_ids']
    aliases = summary['environment_ids']
    selected = {name.removesuffix('_lerobot'): int(aliases[name + '__z0'])
                for name in physical}
    if args.task == 'ball':
        ids = sorted(rows)
        values = np.asarray([rows[i] for i in ids], dtype=np.float64).reshape(len(ids), -1)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite context values')
        distances = ((values[:, None] - values[None, :]) ** 2).sum(-1)
        first, second = np.unravel_index(np.argmax(distances), distances.shape)
        if distances[first, second] <= 0:
            raise ValueError('Final table cannot form two distinct clusters')
        centers = values[[first, second]].copy()
        previous = None
        for _ in range(100):
            labels = ((values[:, None] - centers[None, :]) ** 2).sum(-1).argmin(1)
            if len(set(labels.tolist())) != 2:
                raise ValueError('Empty final-table cluster')
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            centers = np.stack([values[labels == c].mean(0) for c in (0, 1)])
        else:
            raise RuntimeError('Final-table clustering did not converge')
        groups = {f'cluster_{c}': [ids[i] for i in range(len(ids)) if labels[i] == c]
                  for c in (0, 1)}
        membership = {i: name for name, members in groups.items() for i in members}
        assignments = {name: membership[gid] for name, gid in selected.items()}
        # Same-cluster replicas select their shared center; split replicas select
        # z0's cluster deterministically, without inspecting query outcomes.
        setting['initialization_groups'] = groups
        setting['environment_initialization_group'] = assignments
        config['cluster_resolution'] = 'full_32d_two_means_farthest_pair_seed_on_final_table'
    else:
        pair_name = 'door-12u-Half-11d-Half_lerobot'
        pair = [int(aliases[pair_name + f'__z{i}']) for i in (0, 1)]
        excluded = int(aliases['door-6u-8d_lerobot__z1'])
        core = sorted(set(rows) - set(pair) - {excluded})
        if len(pair) != 2 or len(core) != 17 or excluded not in rows:
            raise ValueError('Door semantic cluster membership changed unexpectedly')
        setting['initialization_groups'] = {'12u11d_pair': pair, 'core17': core}
        setting['environment_initialization_group'] = {
            name: '12u11d_pair' if name == pair_name.removesuffix('_lerobot') else 'core17'
            for name in selected}
        setting['excluded_from_group_means'] = [excluded]
        config['cluster_resolution'] = 'previous_door_membership_centers_recomputed_on_final_table'
    for name, members in setting['initialization_groups'].items():
        if not members or any(i not in rows for i in members):
            raise ValueError(f'Invalid members for {name}')
    config['resolved_cluster_centers'] = {
        name: np.asarray([rows[i] for i in members], dtype=float).mean(0).tolist()
        for name, members in setting['initialization_groups'].items()}
    if not all(np.isfinite(np.asarray(v)).all() for v in config['resolved_cluster_centers'].values()):
        raise ValueError('Nonfinite cluster center')
    config['source_table_sha256'] = hashlib.sha256(table_bytes).hexdigest()
    config['request_config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    prepared = Path(setting['prepared'])
    folder = prepared.parent
    folder.mkdir(parents=True, exist_ok=True)
    resolved = folder / 'resolved_config.json'
    with (folder / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if resolved.exists():
            if json.loads(resolved.read_text()) != config:
                raise RuntimeError('Refusing to replace a different resolved experiment')
        else:
            atomic_json(resolved, config)
        if prepared.exists() and not (prepared / 'prepared_complete.json').exists():
            # Preserve incomplete preparation rather than overwriting its files.
            prepared.rename(prepared.with_name(prepared.name + f'.incomplete-{time.time_ns()}'))
        prepare(resolved, args.task)
        plan_path = prepared / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan.update(stage2_initial=setting['initial_context'], known_family_prior=True,
                    initialization_groups=copy.deepcopy(setting['initialization_groups']),
                    environment_initialization_group=copy.deepcopy(setting['environment_initialization_group']),
                    source_table_sha256=config['source_table_sha256'])
        atomic_json(plan_path, plan)
    print(json.dumps({'resolved_config': str(resolved), 'model_step': step,
                      'groups': setting['initialization_groups'],
                      'assignments': setting['environment_initialization_group']}), flush=True)


if __name__ == '__main__':
    main()
