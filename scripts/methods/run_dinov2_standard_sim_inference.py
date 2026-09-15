#!/usr/bin/env python3
"""Resumable standard DINO sim inference, grids, and metrics on a low-priority GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.venv/bin/python'


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def jsonl(path):
    with resolve(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(*args):
    command = [str(value) for value in args]
    print('[run] ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def complete_checkpoint(path):
    try:
        size = path.stat().st_size
        with path.open('rb') as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return False
            length = struct.unpack('<Q', raw)[0]
            if not 0 < length < min(size, 64 * 1024 * 1024):
                return False
            header = json.loads(handle.read(length))
        offsets = [value['data_offsets'][1] for key, value in header.items()
                   if key != '__metadata__']
        return bool(offsets) and 8 + length + max(offsets) == size
    except (OSError, ValueError, KeyError, TypeError):
        return False


def select_checkpoint(config, base):
    record = base / 'checkpoint_selection.json'
    if record.is_file():
        selected = json.loads(record.read_text())
        if not complete_checkpoint(Path(selected['checkpoint'])):
            raise RuntimeError('Pinned checkpoint is unavailable or incomplete; refusing to change weights during resume.')
        return selected
    candidates = []
    required_step = config.get('required_checkpoint_step')
    for path in resolve(config['checkpoint_dir']).glob('step-*.safetensors'):
        match = re.fullmatch(r'step-(\d+)\.safetensors', path.name)
        if match and (required_step is None or int(match[1]) == int(required_step)):
            candidates.append((int(match[1]), path))
    for step, path in sorted(candidates, reverse=True):
        if complete_checkpoint(path):
            selected = {'train_job_id': config['train_job_id'], 'checkpoint': str(path),
                        'step': step, 'size_bytes': path.stat().st_size}
            write_json(record, selected)
            return selected
    raise RuntimeError(f"No complete checkpoint available in {config['checkpoint_dir']}")


def cached_directory(source, destination, lock_path):
    destination.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run('rsync', '-aL', '--partial', str(source) + '/', str(destination) + '/')
    return destination


def stage_weights(config, train, selected):
    user = os.environ.get('USER', 'cyzhou05')
    modern = Path('/tmp') / user / 'bwm_shared'
    legacy = Path('/tmp') / user / 'bwm_shared_cache'
    modern.mkdir(parents=True, exist_ok=True)
    wan_source = resolve(train['model']['model_paths'])
    required = ['diffusion_pytorch_model.safetensors.index.json', 'Wan2.2_VAE.pth'] + [
        f'diffusion_pytorch_model-{index:05d}-of-00003.safetensors' for index in range(1, 4)]
    wan = None
    for cache in (modern, legacy):
        candidate = cache / 'Wan2.2-TI2V-5B'
        if all((candidate / name).is_file() and (candidate / name).stat().st_size ==
               (wan_source / name).stat().st_size for name in required):
            wan = candidate
            print(f'[cache] reuse Wan: {wan}', flush=True)
            break
    if wan is None:
        wan = cached_directory(wan_source, modern / 'Wan2.2-TI2V-5B', modern / 'staging.lock')
    dino_source = resolve(train['dinov2_amortized_context']['dinov2_model_path'])
    dino = cached_directory(dino_source, modern / 'dinov2-base', modern / 'staging.lock')
    key = f"{config['task']}-dino-train{config['train_job_id']}-step{selected['step']}"
    checkpoint = modern / 'trained_ckpts' / (key + '.safetensors')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (checkpoint.parent / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not complete_checkpoint(checkpoint) or checkpoint.stat().st_size != selected['size_bytes']:
            partial = checkpoint.with_suffix('.partial')
            run('rsync', '-aL', '--partial', selected['checkpoint'], partial)
            if not complete_checkpoint(partial):
                raise RuntimeError('Incomplete staged checkpoint.')
            os.replace(partial, checkpoint)
    extracted = modern / 'extracted' / (key + '-wan.safetensors')
    # The existing extraction code publishes atomically. Remove no completed cache.
    if extracted.exists() and not complete_checkpoint(extracted):
        os.replace(extracted, extracted.with_suffix(f'.incomplete-{int(time.time())}'))
    return wan, dino, checkpoint, extracted


def prepare_manifest(config, train, metric, output):
    reference = json.loads(resolve(config['reference_plan']).read_text())
    metadata = jsonl(metric['metadata_jsonl'])
    training_rows = jsonl(train['dataset']['dataset_metadata_path'])
    manifest = yaml.safe_load(resolve(train['dinov2_amortized_context']['dinov2_active_environment_manifest']).read_text())
    selection = manifest['selection']
    env_key = selection['environment_key']
    value_key = selection['physical_value_key']
    active_ids = set(selection['active_environment_ids'])
    active_values = {float(row[value_key]) for row in training_rows if row[env_key] in active_ids}
    if len(active_values) != int(selection['active_environment_count']):
        raise RuntimeError('Training physical-value membership is incomplete.')
    rows = []
    all_queries = []
    seen_environments = set()
    for original in reference:
        source = int(original['source_index'])
        queries = [int(index) for index in original['target_indices']]
        if source in queries or len(queries) != len(set(queries)):
            raise ValueError('Support/query overlap or duplicate queries.')
        physical = float(metadata[source][value_key])
        if physical in seen_environments:
            raise ValueError('Repeated environment in the reference plan.')
        seen_environments.add(physical)
        if any(not math.isclose(float(metadata[index][value_key]), physical, rel_tol=1e-6, abs_tol=1e-9)
               for index in queries):
            raise ValueError('A reference query belongs to a different physical environment.')
        domain = 'id' if any(math.isclose(physical, value, rel_tol=1e-6, abs_tol=1e-9)
                             for value in active_values) else 'ood'
        rows.append({**original, 'support_indices': [source], 'query_indices': queries,
                     'domain': domain, 'environment_id': metadata[source][env_key],
                     'physical_value': physical})
        all_queries.extend(queries)
    if len(rows) != config['expected_environments'] or len(all_queries) != config['expected_queries']:
        raise ValueError('Reference environment/query count differs from the standard protocol.')
    if Counter(row['domain'] for row in rows) != {'id': 5, 'ood': 5}:
        raise ValueError('Expected five ID and five OOD environments, classified by physical values.')
    path = output / 'input_support_query_manifest.json'
    if path.exists() and json.loads(path.read_text()) != rows:
        raise RuntimeError('Support/query protocol changed during resume.')
    write_json(path, rows)
    print(f'[protocol] envs={len(rows)} queries={len(all_queries)} K=1 ID=5 OOD=5', flush=True)
    return path, rows


def quarantine_incomplete_videos(flat, expected_frames):
    for path in flat.glob('*.mp4'):
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                '-count_frames', '-show_entries', 'stream=nb_read_frames',
                                '-of', 'json', str(path)], capture_output=True, text=True)
        try:
            count = int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
        except (ValueError, KeyError, IndexError):
            count = 0
        if probe.returncode or count < expected_frames:
            quarantine = flat / 'incomplete'
            quarantine.mkdir(exist_ok=True)
            os.replace(path, quarantine / f'{path.stem}-{time.time_ns()}.mp4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_PARTITION') != 'yejin-lo':
        raise RuntimeError('Standard inference and LPIPS must run in a yejin-lo compute allocation.')
    config = yaml.safe_load(resolve(args.config).read_text())
    train = yaml.safe_load(resolve(config['train_config']).read_text())
    metric = yaml.safe_load(resolve(config['metric_template']).read_text())
    base = resolve(config['output_base'])
    base.mkdir(parents=True, exist_ok=True)
    selected = select_checkpoint(config, base)
    output = base / f"step_{selected['step']}" / f"seed_{config['seed']}"
    flat = output / 'flat'
    flat.mkdir(parents=True, exist_ok=True)
    manifest, rows = prepare_manifest(config, train, metric, output)
    write_json(output / 'launch_protocol.json', {'launch_config': config, 'checkpoint': selected,
               'training_config': train, 'inference_support_size': 1,
               'domain_rule': 'match physical values from the training active set, not evaluation indices'})
    marker = output / 'rollouts.complete'
    if not marker.exists():
        quarantine_incomplete_videos(flat, int(train['video']['num_frames']))
        wan, dino, checkpoint, extracted = stage_weights(config, train, selected)
        run(PYTHON, ROOT / 'scripts/methods/infer_dinov2_event80.py',
            '--config', resolve(config['train_config']),
            '--dataset_metadata_path', resolve(metric['metadata_jsonl']),
            '--dataset_base_path', resolve(metric['dataset_root']),
            '--model_paths', wan, '--dinov2_model_path', dino,
            '--dinov2_checkpoint_path', checkpoint, '--wan_checkpoint_output', extracted,
            '--support_indices', ','.join(str(row['source_index']) for row in rows),
            '--sample_indices', ','.join(str(index) for row in rows for index in row['target_indices']),
            '--expected_supports_per_environment', 1, '--output_path', flat,
            '--num_inference_steps', config['num_inference_steps'], '--cfg_scale', config['cfg_scale'],
            '--fps', config['fps'], '--quality', config['quality'], '--seed', config['seed'], '--skip_existing')
        marker.touch()
    run(PYTHON, ROOT / 'scripts/evaluation/organize_grouped_transfer_outputs.py',
        '--manifest', manifest, '--flat-root', flat, '--output-root', output, '--method-slug', config['method'])
    transfer = output / 'transfer'
    transfer_rows = json.loads((transfer / 'transfer_plan.json').read_text())
    domain_by_source = {int(row['source_index']): row['domain'] for row in rows}
    for domain in ('id', 'ood'):
        write_json(transfer / f'transfer_plan_{domain}.json',
                   [row for row in transfer_rows if domain_by_source[int(row['source_index'])] == domain])
    grids_marker = output / 'grids.complete'
    if not grids_marker.exists():
        run(PYTHON, ROOT / 'scripts/evaluation/compose_context_transfer_support_grids.py',
            '--metadata-path', resolve(metric['metadata_jsonl']), '--dataset-root', resolve(metric['dataset_root']),
            '--transfer-plan', transfer / 'transfer_plan.json', '--prediction-root', transfer / 'raw',
            '--output-dir', output / 'grids_support_plus_queries', '--width', 224, '--height', 224,
            '--fps', config['fps'], '--quality', config['quality'], '--columns', 5,
            '--support-size', 1, '--prediction-label', 'DINO MLP query')
        grids_marker.touch()
    metric['method_name'] = config['method']
    metric['seed'] = config['seed']
    metric['support_query_manifest'] = str(manifest)
    metric['output_dir'] = str(output / 'action_evaluation')
    metric['transfer_plans'] = [{'path': str(transfer / f'transfer_plan_{domain}.json'),
                                 'domain': domain} for domain in ('id', 'ood')]
    metric_path = output / 'evaluation_config.yaml'
    metric_path.write_text(yaml.safe_dump(metric, sort_keys=False))
    if not (output / 'metrics.complete').exists():
        run(PYTHON, ROOT / 'scripts/evaluation/evaluate_sim_action_selection.py', '--config', metric_path)
        run(PYTHON, ROOT / 'scripts/evaluation/evaluate_sim_transfer_metrics.py', '--config', metric_path,
            '--lpips', '--lpips-net', 'alex', '--lpips-device', 'cuda', '--lpips-batch-size', 8)
        (output / 'metrics.complete').touch()
    (output / 'source_checkpoint.txt').write_text(selected['checkpoint'] + '\n')
    (output / 'inference.complete').touch()
    print(f'[done] output={output}', flush=True)


if __name__ == '__main__':
    main()
