#!/usr/bin/env python3
"""Five global, environment-independent starts; retain every Stage2 result."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.prepare_real97_dual_latent_reference import prepare
from scripts.evaluation.infer_real97_unbounded_context import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['door', 'ball'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    config = json.loads(args.config.read_text())
    setting = config['tasks'][args.task]
    run = ROOT / setting['training_run']
    for name in ('step-6500.safetensors', '.step-6500.safetensors.complete',
                 'step-6500.context_table.json'):
        if not (run / name).is_file():
            raise RuntimeError('Exact completed step6500 required: ' + str(run / name))
    prepared = ROOT / setting['prepared']
    if prepared.exists() and not (prepared / 'prepared_complete.json').is_file():
        prepared.rename(prepared.with_name(prepared.name + '.incomplete-' + str(time.time_ns())))
    prepare(args.config, args.task)
    plan_path = prepared / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan.update(stage2_initial='global_table_box_random_shared', stage2_only=True,
                known_family_prior=False, query_GT_used_for_initialization=False)
    write_json(plan_path, plan)
    table_path = prepared / 'reference/full_training_context_table.json'
    table_bytes = table_path.read_bytes()
    import torch
    table = torch.stack([torch.tensor(row['context'], dtype=torch.float32).reshape(-1)
                         for row in json.loads(table_bytes)['records']])
    if table.shape != (setting['expected_latents'], 32) or not torch.isfinite(table).all():
        raise RuntimeError('Expected a finite full C32 table')
    lower, upper = table.amin(0), table.amax(0)
    generator = torch.Generator(device='cpu').manual_seed(config['global_random_seed'])
    starts = lower + torch.rand((5, 32), generator=generator) * (upper - lower)
    output = ROOT / config['output']
    metadata = {
        'seed': config['global_random_seed'], 'distribution': 'independent coordinate-wise uniform',
        'lower': lower.tolist(), 'upper': upper.tolist(), 'starts': starts.tolist(),
        'scope': 'each of five starts shared across all environments within this task',
        'source_table_sha256': hashlib.sha256(table_bytes).hexdigest(),
        'known_environment_prior': False, 'known_family_prior': False,
        'global_training_table_bounds_used': True, 'query_GT_used_for_initialization': False,
        'query_GT_used_for_trial_selection': False, 'retain_all_trials': True,
        'hard_bounds_during_adaptation': None, 'stage2_only': True,
    }
    record = output / 'global_random_initializations.json'
    if record.exists() and json.loads(record.read_text()) != metadata:
        raise RuntimeError('Refusing to change starts inside an existing experiment')
    write_json(record, metadata)
    for trial in range(5):
        trial_output = output / f'trial_{trial:02d}'
        if (trial_output / 'unbounded_inference_complete.json').is_file():
            continue
        trial_config = copy.deepcopy(config)
        trial_config.update(global_random_starts=starts.tolist(), global_random_trial=trial)
        trial_path = output / f'trial_{trial:02d}.json'
        write_json(trial_path, trial_config)
        subprocess.run([
            sys.executable, str(ROOT / 'scripts/evaluation/infer_real97_unbounded_context.py'),
            '--config', str(trial_path), '--task', args.task,
            '--prepared', str(prepared), '--output', str(trial_output),
        ], check=True, cwd=ROOT)
    write_json(output / 'five_trials_complete.json', {
        'task': args.task, 'trials': 5, 'stage2_only': True,
        'selection_by_query_GT': False, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
    })


if __name__ == '__main__':
    main()
