#!/usr/bin/env python3
"""Matched static Soft queries, four observed supports, training-faithful TTT replay.

Only complete final/wallclock protected checkpoints qualify. A step2300 checkpoint
alone is not completion. Freeze the chosen checkpoint identity across requeues.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
import yaml

from scripts.evaluation.infer_real97_soft_dino_static_reference import (
    check_sample, episode_id, query_key, resolve,
)
from scripts.evaluation.infer_real97_ball_door_baselines import (
    frame_array, json_write, read_video, rows, snapshot, titled, video_write,
)
from scripts.evaluation.infer_real97_standard_reference import copy_cached
from scripts.evaluation.infer_real97_ttt_reference import MultiSupportReplay
from scripts.methods.infer_ttt_kvb_event80 import build_training_faithful_model
from scripts.methods.infer_ttt_kvb_oneminute_event80 import (
    SameTimestepSupportQueryModelFn, _prepare_support_model_inputs,
)
from scripts.methods.train_ttt_kvb_event80 import add_ttt_kvb_config
from scripts.infer import build_infer_dataset, prepare_sample_for_rollout, _run_autoregressive
from wan_video_action.data.history_prefix import CanonicalTargetEEF
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


def select_checkpoint(training, output):
    selection = output / 'checkpoint_selection.json'
    if selection.is_file():
        selected = json.loads(selection.read_text())
        path = Path(selected['checkpoint'])
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] != selected['identity']:
            raise RuntimeError('Previously selected checkpoint changed; refusing mixed outputs')
        return path, selected['step']
    eligible = []
    for metadata in (training / 'protected').glob('step-*.metadata.json'):
        record = json.loads(metadata.read_text())
        if record.get('reason') not in ('final', 'wallclock_23h30'):
            continue
        step = int(record['step'])
        checkpoint = metadata.parent / f'step-{step}.safetensors'
        marker = checkpoint.parent / ('.' + checkpoint.name + '.complete')
        if checkpoint.is_file() and marker.is_file():
            if int(json.loads(marker.read_text())['step']) != step:
                raise RuntimeError('Checkpoint completion marker mismatch')
            eligible.append((step, checkpoint, metadata, record))
    if not eligible:
        raise RuntimeError('No complete final/wallclock checkpoint; refusing an early fallback')
    step, checkpoint, metadata, record = max(eligible, key=lambda item: item[0])
    snapshot(metadata, output / 'input_manifest/checkpoint_metadata.json')
    stat = checkpoint.stat()
    json_write(selection, {'checkpoint': str(checkpoint), 'step': step,
                          'reason': record['reason'], 'identity': [stat.st_size, stat.st_mtime_ns]})
    return checkpoint, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cli = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('A single GPU compute allocation is required')
    os.chdir(ROOT)
    protocol_path = resolve(cli.config)
    protocol = json.loads(protocol_path.read_text())
    output, training, reference = [resolve(protocol[k]) for k in ('output', 'training_run', 'reference_cohort')]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint, step = select_checkpoint(training, output)
    frozen = output / 'input_manifest'
    for name in ('plan.json', 'query.jsonl', 'support.jsonl'):
        snapshot(reference / name, frozen / name)
    snapshot(reference / 'input_manifest/stage_files.txt', frozen / 'stage_files.txt')
    snapshot(training / 'submitted_config.yaml', frozen / 'training_config.yaml')
    snapshot(training / 'input_manifest/action_stats.json', frozen / 'action_stats.json')
    snapshot(training / 'input_manifest/train.jsonl', frozen / 'training_rows.jsonl')
    snapshot(protocol_path, frozen / 'evaluation_config.json')
    for name in ('query.jsonl', 'support.jsonl'):
        snapshot(frozen / name, output / name)
    plan = json.loads((frozen / 'plan.json').read_text())
    queries, supports, training_rows = [rows(frozen / name) for name in ('query.jsonl', 'support.jsonl', 'training_rows.jsonl')]
    train_episodes = {episode_id(row) for row in training_rows}
    train_starts = {(*episode_id(row), int(row['start_frame'])) for row in training_rows}
    config = yaml.safe_load((frozen / 'training_config.yaml').read_text())['training']
    contract = dict(num_frames=33, num_history_frames=1, frame_stride=3, height=160,
                    width=320, action_type='eef_target', action_dim=14,
                    ttt_protocol='oneminute_write_then_predict')
    if any(config[k] != value for k, value in contract.items()):
        raise RuntimeError('Training/inference contract mismatch')
    if len(queries) != protocol['expected_queries'] or sum(r['dataset_split'] == 'test' for r in queries) != protocol['expected_test_queries']:
        raise RuntimeError('Matched query count changed')
    for row in queries + supports:
        start = int(row['start_frame'])
        if row['action_semantics'] != 'eef_target' or row['native_frame_indices'] != [start + 3*k for k in range(33)]:
            raise RuntimeError('Video/action alignment mismatch')
    for row in queries:
        if int(row['start_frame']) != 0 or ((row['dataset_split'] == 'train') != (episode_id(row) in train_episodes)):
            raise RuntimeError('Static query start or train/test membership changed')
    for env in plan['environments']:
        selected = [supports[i] for i in env['support_indices']]
        if len(selected) != 4 or len({episode_id(r) for r in selected}) != 4:
            raise RuntimeError('Four distinct support episodes required')
        for row in selected:
            if row['environment'] != env['environment'] or row['dataset_split'] != 'train' or (*episode_id(row), int(row['start_frame'])) not in train_starts:
                raise RuntimeError('Support is not an eligible training chunk')
        if {episode_id(r) for r in selected} & {episode_id(queries[i]) for i in env['query_indices']}:
            raise RuntimeError('Support/query episode overlap')

    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    cache.mkdir(parents=True, exist_ok=True)
    wan, base = cache / 'Wan2.2-TI2V-5B', cache / 'ckpt/BLM/step-12000.safetensors'
    copy_cached(ROOT / 'models/Wan2.2-TI2V-5B', wan, directory=True)
    copy_cached(ROOT / 'ckpt/BLM/step-12000.safetensors', base)
    source_data = resolve(config['dataset_base_path'])
    files = (frozen / 'stage_files.txt').read_text().splitlines()
    if any(Path(p).is_absolute() or '..' in Path(p).parts for p in files):
        raise RuntimeError('Unsafe data staging paths')
    required = {p for row in queries + supports for p in [*row['video'], row['action']]}
    if not required.issubset(set(files)):
        raise RuntimeError('Data staging list is incomplete')
    identity = str(source_data.resolve()) + '\n' + '\n'.join(files)
    local_data = cache / 'eval_datasets' / ('soft_dual9_' + hashlib.sha256(identity.encode()).hexdigest()[:16])
    copy_cached(source_data, local_data, directory=True, files=frozen / 'stage_files.txt')
    stat = checkpoint.stat()
    identity = f'{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'
    local_ckpt = cache / 'real97_ttt_eval' / hashlib.sha256(identity.encode()).hexdigest()[:16] / 'model.safetensors'
    copy_cached(checkpoint, local_ckpt)
    config.update(dataset_base_path=str(local_data), dataset_metadata_path=str(output / 'query.jsonl'),
                  action_stat_path=str(frozen / 'action_stats.json'), model_paths=str(wan), ckpt_path=str(base),
                  output_path=str(output / 'raw/ttt'), num_inference_steps=protocol['num_inference_steps'],
                  seed=protocol['seed'], cfg_scale=1.0, fps=protocol['fps'], quality=8,
                  video_light_augmentation_enabled=False, spatial_loss_mode='none',
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference': config}, sort_keys=False))
    inference_parser = add_ttt_kvb_config(add_general_config(argparse.ArgumentParser()))
    if '--frame_stride' not in inference_parser._option_string_actions:
        inference_parser.add_argument('--frame_stride', type=int, default=3)
    args = inference_parser.parse_args(['--config', str(runtime)])
    args = merge_yaml_and_args(str(runtime), inference_parser, args)
    args.ttt_checkpoint_path = str(local_ckpt)
    stats = json.loads((frozen / 'action_stats.json').read_text())['eef_target']

    def dataset_for(path):
        local_args = copy.copy(args)
        local_args.dataset_metadata_path = str(path)
        dataset = build_infer_dataset(local_args)
        dataset.special_operator_map['action'] = CanonicalTargetEEF(local_args, stats)
        return dataset

    dataset, support_dataset = dataset_for(output / 'query.jsonl'), dataset_for(output / 'support.jsonl')
    set_global_seed(int(protocol['seed']))
    model, installation = build_training_faithful_model(args)
    model.requires_grad_(False)
    pipe, controller = model.pipe, installation.controller
    original = pipe.model_fn
    json_write(output / 'plan.json', {**plan, 'method': 'ttt', 'model_step': step, 'support_k': 4})
    json_write(output / 'provenance.json', {
        'protocol': protocol, 'checkpoint': str(checkpoint), 'checkpoint_step': step,
        'query_manifest_sha256': hashlib.sha256((frozen / 'query.jsonl').read_bytes()).hexdigest(),
        'support_manifest_sha256': hashlib.sha256((frozen / 'support.jsonl').read_bytes()).hexdigest(),
        'action': 'training CanonicalTargetEEF; 8 values padded to 14; stride 3',
        'observed_frames': 1, 'predicted_frames': 32, 'query_future_gt_used': False,
        'memory_protocol': 'reset per diffusion timestep; support1..4 then query write/predict',
        'inference_augmentation': False, 'metric_status': 'not_scored',
    })
    completed = []
    for env_index, env in enumerate(plan['environments']):
        name = env['environment']
        replays = []
        for position, index in enumerate(env['support_indices']):
            support = support_dataset[index]
            check_sample(support)
            noise_seed = int(protocol['seed']) + int(protocol['support_noise_seed_offset']) + env_index * 1009 + position
            inputs, latents, noise = _prepare_support_model_inputs(model, support, noise_seed=noise_seed)
            replays.append(SameTimestepSupportQueryModelFn(
                pipe=pipe, controller=controller, original_model_fn=original,
                support_model_inputs=inputs, support_input_latents=latents, support_noise=noise))
        json_write(output / 'support_memory' / (name + '.json'), {
            'environment': name, 'support_indices': env['support_indices'],
            'supports': [supports[i] for i in env['support_indices']],
            'rebuild_at_each_query_timestep': True, 'query_branches_independent': True,
        })
        for index in env['query_indices']:
            row = queries[index]
            key = query_key(index, row)
            gt_path, pred_path = [output / 'raw' / method / (key + '.mp4') for method in ('gt', 'ttt')]
            marker = output / 'completed' / (key + '.json')
            if not (marker.is_file() and gt_path.is_file() and pred_path.is_file()):
                sample = dataset[index]
                check_sample(sample)
                gt = frame_array(sample['video'])
                args.seed = int(protocol['seed']) + index
                set_global_seed(args.seed)
                rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                rollout['output_path'] = str(pred_path)
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                wrapper = MultiSupportReplay(pipe, controller, original, replays)
                pipe.model_fn = wrapper
                try:
                    with torch.no_grad():
                        _run_autoregressive(pipe=pipe, sample=rollout, args=args)
                finally:
                    pipe.model_fn = original
                    controller.clear()
                prediction = read_video(pred_path)
                if prediction.shape != gt.shape or len(prediction) != 33:
                    raise RuntimeError('GT/prediction shape mismatch: ' + key)
                video_write(gt_path, gt, protocol['fps'])
                video_write(pred_path, prediction, protocol['fps'])
                json_write(output / 'memory_trace' / (key + '.json'), {'trace': wrapper.trace, 'support_k': 4})
                json_write(marker, {'query': row, 'checkpoint_step': step, 'seed': args.seed, 'support_indices': env['support_indices']})
                del sample, rollout, prediction, gt
            video_write(output / 'comparisons' / (key + '.mp4'), np.concatenate([
                titled(read_video(gt_path), f"GT {row['dataset_split']} ep{row['episode_index']}"),
                titled(read_video(pred_path), 'TTT K=4'),
            ], axis=1), protocol['fps'])
            completed.append(key)
            json_write(output / 'progress.json', {'completed_queries': len(completed), 'total_queries': len(queries)})
            print(f'[query_done] {key} {len(completed)}/{len(queries)}', flush=True)
        for split in ('train', 'test'):
            indices = [i for i in env['query_indices'] if queries[i]['dataset_split'] == split]
            columns = [read_video(output / 'comparisons' / (query_key(i, queries[i]) + '.mp4')) for i in indices]
            if columns:
                video_write(output / 'grids' / split / (name + '_gt_ttt.mp4'), np.concatenate(columns, axis=2), protocol['fps'])
        del replays
        controller.clear()
        torch.cuda.empty_cache()
    json_write(output / 'inference_complete.json', {
        'method': 'ttt', 'checkpoint': str(checkpoint), 'checkpoint_step': step,
        'queries': len(completed), 'test_queries': 18, 'train_queries': 18,
        'environments': len(plan['environments']), 'support_k': 4,
        'metric_status': 'not_scored', 'main_metric_environments': protocol['main_metric_environments'],
        'common_metric_native_frames': list(range(3, 85, 3)),
    })


if __name__ == '__main__':
    main()
