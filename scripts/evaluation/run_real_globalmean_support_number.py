#!/usr/bin/env python3
"""Fill missing real global-mean support counts with frozen prior support episodes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
import run_real_support_number as common

ROOT = common.ROOT
DEST = common.DEST / 'globalmean_20260922'
TASKS = [('ball', 2), ('ball', 4), ('door', 1), ('door', 4), ('soft', 1), ('soft', 2)]
GLOBAL = {
    'ball': ROOT / 'outputs/infer_real97_ball_dual6500_simlr_globalmean_supportL7_20260922_v1',
    'door': ROOT / 'outputs/infer_real97_door_dual6500_simlr_globalmean_20260922_v1',
    'soft': ROOT / 'outputs/infer_real97_soft_5500_simlr_globalmean_20260922_v1',
}

def cluster_source(task, k):
    if task == 'door' and k == 4:
        return common.DEST / 'support_level_revision_20260922/door/k4'
    return common.DEST / task / f'k{k}'

def ball_door(task, k, out):
    import prepare_real97_support_override as override
    source_output = GLOBAL[task]
    source_prepared = Path(common.read(source_output / 'provenance.json')['prepared'])
    plan = common.read(source_prepared / 'plan.json')
    old_support = common.rows(source_prepared / 'support.jsonl')
    reference = cluster_source(task, k)
    frozen = common.read(reference / 'support_protocol.json')['selection']
    reference_rows = common.rows(reference / 'prepared/support.jsonl')
    goals = {}
    for entry in frozen:
        name = entry['environment']
        goals[name] = [next(copy.deepcopy(row) for row in reference_rows
                            if row['environment'] == name and row['episode_index'] == ep
                            and int(row['action_level']) == int(level))
                       for level, ep in zip(entry['levels'], entry['episodes'])]
    if len(goals) != len(plan['environments']):
        raise RuntimeError('Incomplete fixed environment set')
    queries = common.rows(source_prepared / 'query.jsonl')
    forbidden = {(row['environment'], row['episode_index']) for row in queries}
    if any(row['dataset_split'] != 'train' or (row['environment'], row['episode_index']) in forbidden
           for group in goals.values() for row in group):
        raise RuntimeError('Train/test support leakage')
    augmented = out / 'source_prepared'
    augmented.mkdir(parents=True, exist_ok=True)
    for name in ['plan.json', 'experiment.json', 'support.jsonl', 'query.jsonl', 'stage_files.txt']:
        common.frozen_file(source_prepared / name, augmented / name)
    if not (augmented / 'reference').exists():
        (augmented / 'reference').symlink_to(source_prepared / 'reference', target_is_directory=True)
    templates = common.rows(source_prepared / 'extension_action_templates.jsonl')
    common.write_rows(augmented / 'extension_action_templates.jsonl',
                      templates + [row for group in goals.values() for row in group])
    changed = {}
    for env in plan['environments']:
        name = env['environment']
        before = [old_support[i] for i in env['support_indices']]
        after = goals[name]
        if len(before) != len(after) or any(a['episode_index'] != b['episode_index']
                                             for a, b in zip(before, after)):
            changed[name] = [int(row['action_level']) for row in after]
    if not changed:
        raise RuntimeError('K already complete; use original global-mean result')
    config_file = (f'configs/evaluation/real97_ball_ours_simlr_globalmean_supportL7_20260922_v1.json'
                   if task == 'ball' else 'configs/evaluation/real97_door_ours_simlr_globalmean_20260922_v1.json')
    cfg = common.read(ROOT / config_file)
    prepared = out / 'prepared'
    cfg['output'] = str(out)
    cfg['support_count_experiment'] = {'K': k, 'frozen_support_source': str(reference),
                                       'initialization': 'full training table mean',
                                       'support_loss_reduction': 'mean'}
    cfg['tasks'][task].update(source_prepared=str(augmented), source_inference=str(source_output),
                              prepared=str(prepared), support_levels=changed,
                              initial_context='mean_training_table', partial_stage2_replacement=True,
                              require_exact_original_support=True)
    config = out / 'config.json'
    common.write(config, cfg)
    override.locations = lambda task, job_id: (prepared, out)
    override.prepare(config, task, 'globalmean_support_number')
    newplan = common.read(prepared / 'plan.json')
    support = common.rows(prepared / 'support.jsonl')
    for env in newplan['environments']:
        name = env['environment']
        env['extension_template_indices'] = []
        if name in changed:
            indices = []
            for row in goals[name]:
                row.pop('template_index', None)
                if row in support:
                    index = support.index(row)
                else:
                    index = len(support)
                    support.append(row)
                indices.append(index)
            env['support_indices'] = indices
        else:
            common.frozen_file(source_output / 'initializations' / (name + '.json'),
                               out / 'initializations' / (name + '.json'))
    newplan['protocol']['ttt']['ttt_support_loss_reduction'] = 'mean'
    newplan['active_support_count'] = sum(len(e['support_indices']) for e in newplan['environments'])
    common.write(prepared / 'plan.json', newplan)
    common.write_rows(prepared / 'support.jsonl', support)
    staged = set((prepared / 'stage_files.txt').read_text().splitlines())
    for row in support:
        staged.update(row['video'])
        staged.add(row['action'])
    (prepared / 'stage_files.txt').write_text('\n'.join(sorted(staged)) + '\n')
    common.audit(newplan, support, queries, k, out)
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

def soft(k, out):
    import infer_real97_soft_balanced9_dual_static1 as runner
    source = GLOBAL['soft']
    cfg = common.read(ROOT / 'configs/evaluation/real97_soft_ours_simlr_globalmean_20260922_v1.json')
    cfg['support_count_experiment'] = {'K': k, 'initialization': 'all 18 training latents mean',
                                       'support_loss_reduction': 'mean'}
    config = out / 'config.json'
    common.write(config, cfg)
    for name in ['query.jsonl', 'support.jsonl', 'selection_plan.json', 'selection_complete.json',
                 'input_manifest/resolved_checkpoint.json']:
        common.frozen_file(source / name, out / name)
    original_prepare = runner.prepare
    wanted = {e['environment']: e['episodes']
              for e in common.read(cluster_source('soft', k) / 'support_protocol.json')['selection']}
    def prepare(data, output):
        plan, runtime, checkpoint, queries, supports = original_prepare(data, output)
        for env in plan['environments']:
            env['support_indices'] = [next(i for i in env['support_indices']
                                           if supports[i]['episode_index'] == episode)
                                      for episode in wanted[env['environment']]]
        common.audit(plan, supports, queries, k, output)
        common.write(output / 'plan.json', plan)
        parsed = yaml.safe_load(runtime.read_text())
        parsed['inference']['ttt_support_loss_reduction'] = 'mean'
        runtime.write_text(yaml.safe_dump(parsed, sort_keys=False))
        return plan, runtime, checkpoint, queries, supports
    runner.prepare = prepare
    for index, row in enumerate(common.rows(source / 'query.jsonl')):
        key = f"q{index:04d}_{row['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"
        for relative in [f'raw/stage1/{key}.mp4', f'completed_variants/{key}_stage1.json']:
            if (source / relative).is_file():
                common.frozen_file(source / relative, out / relative)
    sys.argv = ['infer_real97_soft_balanced9_dual_static1', '--config', str(config), '--output', str(out)]
    runner.main()

def score(task, out):
    cpu = ROOT / '.venv-real97-eval-20260909/bin/python'
    gpu = ROOT / '.venv/bin/python'
    if task in ('ball', 'door'):
        cfg = common.read(ROOT / f'configs/evaluation/real97_{task}_simlr_globalmean_scores_20260922_v1.json')
        cfg['tasks'][task]['ours'] = str(out)
        cfg['output'] = str(out / 'metrics')
        config = out / 'metrics_config.json'
        common.write(config, cfg)
        script = ROOT / f'scripts/evaluation/score_real97_{task}_simlr6500.py'
        subprocess.run([str(cpu), str(script), '--config', str(config), '--mode', 'cpu', '--workers', '8'], check=True)
        subprocess.run([str(gpu), str(script), '--config', str(config), '--mode', 'lpips'], check=True)
    else:
        cfg = common.read(ROOT / 'configs/evaluation/real97_soft_simlr_globalmean_scores_20260922_v1.json')
        cfg.update(source_inference=str(out), output=str(out / 'metrics'), shards=1)
        config = out / 'metrics_config.json'
        common.write(config, cfg)
        script = ROOT / 'scripts/evaluation/score_real97_soft_simlr.py'
        subprocess.run([str(cpu), str(script), '--config', str(config), '--shard', '0'], check=True)
        subprocess.run([str(cpu), str(script), '--config', str(config), '--aggregate-only'], check=True)
    common.write(out / 'globalmean_metric_pipeline_complete.json', {'task': task})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, choices=range(len(TASKS)), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    os.chdir(ROOT)
    task, k = TASKS[args.index]
    out = DEST / task / f'k{k}'
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'globalmean_inference_complete.json').exists():
        (soft if task == 'soft' else ball_door)(k, out) if task == 'soft' else ball_door(task, k, out)
        common.write(out / 'globalmean_inference_complete.json', {'task': task, 'K': k})
    if not (out / 'globalmean_metric_pipeline_complete.json').exists():
        score(task, out)

if __name__ == '__main__':
    main()
