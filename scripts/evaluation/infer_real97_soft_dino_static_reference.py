#!/usr/bin/env python3
"""Matched Soft DINO inference with four train supports and canonical target EEF."""
from __future__ import annotations

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
from PIL import Image
import torch
import yaml

from scripts.evaluation.infer_real97_ball_door_baselines import (
    frame_array, json_write, read_video, rows, snapshot, titled, video_write,
)
from scripts.evaluation.infer_real97_standard_reference import copy_cached
from scripts.infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
from scripts.methods.infer_dinov2_event80 import materialize_wan_checkpoint, load_support_encoder
from scripts.methods.train_dinov2_event80 import add_dinov2_config
from wan_video_action.data.history_prefix import CanonicalTargetEEF
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def query_key(index, row):
    return f"q{index:04d}_{row['environment']}_{row['dataset_split']}_ep{int(row['episode_index']):06d}"


def episode_id(row):
    return row['environment'], int(row['episode_index'])


def check_sample(sample):
    if tuple(sample['video'].shape) != (1, 3, 33, 160, 320):
        raise RuntimeError('Expected one observed and 32 future video frames at 320x160')
    if tuple(sample['action'].shape) != (1, 33, 14):
        raise RuntimeError('Expected 33 aligned canonical target-EEF actions')


def display_size(frames):
    if frames.shape[1:3] == (160, 320):
        return frames
    if frames.shape[2] != 2 * frames.shape[1]:
        raise RuntimeError('Reference video has an unexpected aspect ratio')
    return np.stack([np.asarray(Image.fromarray(frame).resize((320, 160), Image.Resampling.BILINEAR)) for frame in frames])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    own = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Use a single-GPU compute allocation, not a login node')
    protocol_path = resolve(own.config)
    protocol = json.loads(protocol_path.read_text())
    output = resolve(protocol['output'])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    training = resolve(protocol['training_run'])
    reference = resolve(protocol['reference_cohort'])
    frozen = output / 'input_manifest'
    step = int(protocol['checkpoint_step'])
    source_ckpt = training / 'protected' / f'step-{step}.safetensors'
    source_marker = source_ckpt.parent / ('.' + source_ckpt.name + '.complete')
    if not source_marker.is_file() or not source_ckpt.is_file():
        raise RuntimeError('Final complete checkpoint is required; do not fall back to an earlier step')
    checkpoint_meta = source_ckpt.with_name(f'step-{step}.metadata.json')
    metadata = json.loads(checkpoint_meta.read_text())
    if int(metadata['step']) != step or metadata.get('reason') != 'final':
        raise RuntimeError('Expected the completed final DINO checkpoint')
    for name in ('plan.json', 'query.jsonl', 'support.jsonl'):
        snapshot(reference / name, frozen / name)
    snapshot(reference / 'input_manifest/stage_files.txt', frozen / 'stage_files.txt')
    snapshot(training / 'submitted_config.yaml', frozen / 'training_config.yaml')
    snapshot(training / 'input_manifest/action_stats.json', frozen / 'action_stats.json')
    snapshot(training / 'input_manifest/train.jsonl', frozen / 'training_rows.jsonl')
    snapshot(checkpoint_meta, frozen / 'checkpoint_metadata.json')
    snapshot(protocol_path, frozen / 'evaluation_config.json')
    for name in ('query.jsonl', 'support.jsonl'):
        snapshot(frozen / name, output / name)
    plan = json.loads((frozen / 'plan.json').read_text())
    queries, supports = rows(frozen / 'query.jsonl'), rows(frozen / 'support.jsonl')
    train_rows = rows(frozen / 'training_rows.jsonl')
    train_episodes = {episode_id(row) for row in train_rows}
    train_starts = {(row['environment'], int(row['episode_index']), int(row['start_frame'])) for row in train_rows}
    if {env['environment'] for env in plan['environments']} != set(protocol['environments']):
        raise RuntimeError('Reference environment set differs from the requested nine environments')
    if len(queries) != protocol['expected_query_count'] or sum(row['dataset_split'] == 'test' for row in queries) != protocol['expected_test_query_count']:
        raise RuntimeError('Reference query cohort must retain all 18 train and 18 test queries')
    config = yaml.safe_load((frozen / 'training_config.yaml').read_text())['training']
    for key in ('num_frames', 'num_history_frames', 'frame_stride', 'height', 'width', 'action_type', 'action_dim'):
        if config[key] != protocol[key]:
            raise RuntimeError(f'Training/inference contract mismatch: {key}')
    if config['dinov2_aggregation_mode'] != 'concat_mlp' or str(config['dinov2_support_k_choices']) != '1,2,3,4':
        raise RuntimeError('Expected the new K=1..4 concat-MLP training run')
    for row in queries + supports:
        start = int(row['start_frame'])
        if row['action_semantics'] != 'eef_target' or row['native_frame_indices'] != [start + 3*k for k in range(33)]:
            raise RuntimeError('Video and action must use the same stride-three native frames')
    for row in queries:
        if int(row['start_frame']) != 0:
            raise RuntimeError('The frozen query protocol uses episode-start static inputs')
        if (row['dataset_split'] == 'train') != (episode_id(row) in train_episodes):
            raise RuntimeError('Query split disagrees with the actual frozen training manifest')
    for env in plan['environments']:
        selected = [supports[i] for i in env['support_indices']]
        if len(selected) != 4 or len({episode_id(row) for row in selected}) != 4:
            raise RuntimeError('Exactly four distinct Support episodes are required')
        for row in selected:
            if row['environment'] != env['environment'] or row['dataset_split'] != 'train':
                raise RuntimeError('Support must be from the same environment training split')
            if (*episode_id(row), int(row['start_frame'])) not in train_starts:
                raise RuntimeError('Support start was not eligible in the actual training manifest')
        if {episode_id(row) for row in selected} & {episode_id(queries[i]) for i in env['query_indices']}:
            raise RuntimeError('Evaluation supports and queries must be episode-disjoint')

    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    cache.mkdir(parents=True, exist_ok=True)
    wan = cache / 'Wan2.2-TI2V-5B'
    copy_cached(ROOT / 'models/Wan2.2-TI2V-5B', wan, directory=True)
    source_data = resolve(config['dataset_base_path'])
    files = (frozen / 'stage_files.txt').read_text().splitlines()
    if any(Path(path).is_absolute() or '..' in Path(path).parts for path in files):
        raise RuntimeError('Unsafe dataset-relative file list')
    required = {path for row in queries + supports for path in [*row['video'], row['action']]}
    if not required.issubset(set(files)):
        raise RuntimeError('The frozen file list does not cover all requested supports and queries')
    identity = str(source_data.resolve()) + '\n' + '\n'.join(files)
    local_data = cache / 'eval_datasets' / ('soft_dual9_' + hashlib.sha256(identity.encode()).hexdigest()[:16])
    copy_cached(source_data, local_data, directory=True, files=frozen / 'stage_files.txt')
    dino_local = cache / 'dinov2-base'
    copy_cached(resolve(config['dinov2_model_path']), dino_local, directory=True)
    stat = source_ckpt.stat()
    identity = f'{source_ckpt.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'
    local_ckpt = cache / 'real97_baseline_eval' / hashlib.sha256(identity.encode()).hexdigest()[:16] / 'model.safetensors'
    copy_cached(source_ckpt, local_ckpt)
    wan_ckpt = local_ckpt.with_name('wan.safetensors')
    with local_ckpt.with_name('wan.lock').open('a') as checkpoint_lock:
        fcntl.flock(checkpoint_lock, fcntl.LOCK_EX)
        if not wan_ckpt.is_file():
            temporary = wan_ckpt.with_name(f'wan.{os.getpid()}.partial.safetensors')
            materialize_wan_checkpoint(str(local_ckpt), str(temporary))
            os.replace(temporary, wan_ckpt)
    config.update(
        dataset_base_path=str(local_data), dataset_metadata_path=str(output / 'query.jsonl'),
        action_stat_path=str(frozen / 'action_stats.json'), model_paths=str(wan), ckpt_path=str(wan_ckpt),
        dinov2_model_path=str(dino_local), dinov2_support_k=4,
        output_path=str(output / 'raw/dino'), num_inference_steps=protocol['num_inference_steps'],
        seed=protocol['seed'], cfg_scale=1.0, fps=protocol['fps'], quality=8, max_samples=0,
        video_light_augmentation_enabled=False, dinov2_support_light_augmentation_enabled=False,
        spatial_loss_mode='none', use_gradient_checkpointing_offload=False,
    )
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference': config}, sort_keys=False))
    inference_parser = add_dinov2_config(add_general_config(argparse.ArgumentParser()))
    if '--frame_stride' not in inference_parser._option_string_actions:
        inference_parser.add_argument('--frame_stride', type=int, default=3)
    args = inference_parser.parse_args(['--config', str(runtime)])
    args = merge_yaml_and_args(str(runtime), inference_parser, args)
    args.dinov2_checkpoint_path = str(local_ckpt)
    stats = json.loads((frozen / 'action_stats.json').read_text())['eef_target']

    def dataset_for(path):
        local_args = copy.copy(args)
        local_args.dataset_metadata_path = str(path)
        dataset = build_infer_dataset(local_args)
        dataset.special_operator_map['action'] = CanonicalTargetEEF(local_args, stats)
        return dataset

    dataset = dataset_for(output / 'query.jsonl')
    support_dataset = dataset_for(output / 'support.jsonl')
    json_write(output / 'plan.json', {
        'task': 'soft', 'method': 'dinov2_concat_mlp', 'environments': plan['environments'],
        'query_count': len(queries), 'support_count': len(supports), 'support_k': 4,
        'model_step': step, 'observed_frames': 1, 'predicted_frames': 32,
        'source_cohort': str(reference), 'query_GT_used_for_context': False,
    })
    json_write(output / 'provenance.json', {
        'protocol': protocol, 'job_id': os.environ['SLURM_JOB_ID'],
        'source_checkpoint': str(source_ckpt), 'local_checkpoint': str(local_ckpt),
        'query_manifest_sha256': hashlib.sha256((frozen / 'query.jsonl').read_bytes()).hexdigest(),
        'support_manifest_sha256': hashlib.sha256((frozen / 'support.jsonl').read_bytes()).hexdigest(),
        'action': 'training CanonicalTargetEEF with frozen statistics: 8 coordinates plus 6 zeros',
        'context': 'one amortized DINO code from four train supports; no gradient adaptation',
        'inference_augmentation': False, 'original_outputs_modified': False,
        'four_way_native_frames': list(range(0, 85, 3)),
        'four_way_standard_video_indices': list(range(4, 33)),
        'four_way_ours_dino_video_indices': list(range(29)),
        'standard_unseen_training_environments': ['soft-4l', 'soft-7r', 'soft-6m'],
    })
    set_global_seed(int(protocol['seed']))
    pipe = build_pipeline(args)
    encoder = load_support_encoder(args, pipe.device).eval()
    contexts = {}
    with torch.no_grad():
        for env in plan['environments']:
            name = env['environment']
            context_path = output / 'contexts' / (name + '.json')
            if context_path.is_file():
                state = json.loads(context_path.read_text())
                if state['checkpoint'] != str(source_ckpt) or state['support_indices'] != env['support_indices']:
                    raise RuntimeError('Cannot reuse a context from a different checkpoint/support selection')
                contexts[name] = torch.tensor(state['context'], dtype=torch.float32).reshape(1, 32)
                continue
            data = [support_dataset[i] for i in env['support_indices']]
            for sample in data:
                check_sample(sample)
            visual = [encoder.extract_visual_features(sample['video']) for sample in data]
            context = encoder.project_supports(
                visual_features=tuple(item[0] for item in visual),
                actions=tuple(sample['action'] for sample in data),
                frame_indices=tuple(item[1] for item in visual),
                frame_counts=tuple(item[2] for item in visual),
            )[0].detach().float().cpu().reshape(1, 32)
            contexts[name] = context
            json_write(context_path, {
                'environment': name, 'context': context.tolist(), 'checkpoint': str(source_ckpt),
                'support_indices': env['support_indices'], 'support_k': 4,
                'support_sample_ids': [supports[i]['sample_id'] for i in env['support_indices']],
                'aggregation': 'mean_support_summary_before_output_head', 'uses_query_future': False,
            })
            print('[context] ' + name + ' K=4', flush=True)
            del data, visual
    del encoder
    torch.cuda.empty_cache()
    completed = []
    grid_order = {}
    for env in plan['environments']:
        name = env['environment']
        for index in env['query_indices']:
            row = queries[index]
            key = query_key(index, row)
            gt_path = output / 'raw/gt' / (key + '.mp4')
            pred_path = output / 'raw/dino' / (key + '.mp4')
            marker = output / 'completed' / (key + '.json')
            if not marker.is_file():
                sample = dataset[index]
                check_sample(sample)
                gt = frame_array(sample['video'])
                args.seed = int(protocol['seed']) + index
                set_global_seed(args.seed)
                rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                rollout.update(physical_context=contexts[name].to(device=pipe.device, dtype=pipe.torch_dtype),
                               output_path=str(pred_path))
                with torch.no_grad():
                    _run_autoregressive(pipe=pipe, sample=rollout, args=args)
                pred = read_video(pred_path)
                if pred.shape != gt.shape or len(pred) != 33:
                    raise RuntimeError(f'Prediction/GT contract mismatch: {key}')
                video_write(gt_path, gt, protocol['fps'])
                video_write(pred_path, pred, protocol['fps'])
                json_write(marker, {'query': row, 'checkpoint_step': step, 'seed': args.seed,
                                    'support_k': 4, 'observed_frames': 1, 'paired_gt': True})
                del sample, rollout
            gt, pred = read_video(gt_path), read_video(pred_path)
            video_write(output / 'comparisons' / (key + '.mp4'), np.concatenate([
                titled(gt, f"GT {row['dataset_split']} ep{row['episode_index']}"),
                titled(pred, 'DINO K=4'),
            ], axis=1), protocol['fps'])
            standard = read_video(resolve(protocol['standard_reference']) / 'raw/standard' / (key + '.mp4'))
            ours = read_video(resolve(protocol['ours_reference']) / 'raw/stage2' / (key + '.mp4'))
            if any(len(video) != 33 for video in (gt, pred, standard, ours)):
                raise RuntimeError('Four-way comparison requires the recorded 33-frame sequences')
            four_way = np.concatenate([
                titled(display_size(gt[:29]), f"GT {row['dataset_split']} ep{row['episode_index']}"),
                titled(display_size(standard[4:33]), 'Standard'),
                titled(display_size(ours[:29]), 'Ours family-mean'),
                titled(display_size(pred[:29]), 'DINO K=4'),
            ], axis=1)
            video_write(output / 'comparisons_4way' / (key + '.mp4'), four_way, protocol['fps'])
            completed.append(key)
            json_write(output / 'progress.json', {'completed_queries': len(completed), 'keys': completed})
            print(f'[query_done] {key} ({len(completed)}/{len(queries)})', flush=True)
        grid_order[name] = {}
        for split in ('train', 'test'):
            indices = [i for i in env['query_indices'] if queries[i]['dataset_split'] == split]
            grid_order[name][split] = indices
            if not indices:
                continue
            columns = [read_video(output / 'comparisons' / (query_key(i, queries[i]) + '.mp4')) for i in indices]
            video_write(output / 'grids' / split / (name + '_gt_dino.mp4'), np.concatenate(columns, axis=2), protocol['fps'])
            columns = [read_video(output / 'comparisons_4way' / (query_key(i, queries[i]) + '.mp4')) for i in indices]
            video_write(output / 'grids_4way' / split / (name + '_gt_standard_ours_dino.mp4'), np.concatenate(columns, axis=2), protocol['fps'])
    json_write(output / 'grid_order.json', grid_order)
    json_write(output / 'inference_complete.json', {
        'task': 'soft', 'method': 'dinov2_concat_mlp', 'checkpoint': str(source_ckpt),
        'checkpoint_step': step, 'queries': len(completed), 'test_queries': 18, 'train_queries': 18,
        'environments': len(contexts), 'support_k': 4, 'inference_augmentation': False,
        'observed_frames': 1, 'predicted_frames': 32, 'frame_stride': 3,
        'metric_status': 'not_scored', 'action_metric': 'not_applicable',
    })


if __name__ == '__main__':
    main()
