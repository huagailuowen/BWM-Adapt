#!/usr/bin/env python3
"""Materialize chunk manifests/configs, not videos, for September real tasks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

from annotate_real_97_chunk_events import event, rotation_about_negative_x, sustained_start

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real')
TASKS = {
    'ball': ('ball_friction_9_7_crop', 61, 1, 320, 160, 8, 4, 3, 8, 'joint_target', .5),
    'door': ('door_close_9_7', 49, 1, 256, 192, 10, 5, 3, 8, 'joint_target_export', .5),
    'stick': ('stick_balance_9_7', 41, 3, 256, 192, 8, 4, 3, 8, 'eef_target', .7),
    'soft': ('soft_pull_9_7_crop_512x256', 33, 3, 320, 160, 15, 5, 5, 6, 'joint_target', 0.),
}
HELDOUT = {'soft-3l', 'soft-6m', 'soft-4l', 'soft-7r'}


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_map(root, task):
    result = {}
    if task == 'door':
        return result  # This export carries its frozen split on each episode.
    manifest = json.loads((root / 'TRAIN_TEST_SPLIT.json').read_text())
    if task == 'stick':
        for env in manifest['environments']:
            for split in ('train', 'test'):
                for episode in env[f'{split}_episode_indices']:
                    result[(env['dataset'], int(episode))] = split
    elif task == 'soft':
        for env, entry in manifest['environment_splits'].items():
            for episode in entry['train_episode_indices']:
                result[(env + '_lerobot', int(episode))] = 'train'
            for episode in entry['test']:
                result[(env + '_lerobot', int(episode['episode_index']))] = 'test'
    else:
        for split in ('train', 'test'):
            for ep in manifest[f'{split}_episodes']:
                result[(ep['lerobot_dataset'], int(ep['lerobot_episode_index']))] = split
    return result


def refreshed_door_events(annotation, target_table, fps):
    """Use the new recorded target, not the legacy observed-motion sidecar."""
    signal = rotation_about_negative_x(target_table['target_action'].to_pylist())
    left = sustained_start(signal, -1)
    right = sustained_start(signal, 1)
    if left is None:
        raise ValueError('No sustained left strike in recorded Door target')
    if right is not None and right >= left:
        right = None
    trough = left + int(np.argmin(signal[left:]))
    crossings = np.flatnonzero(signal[left:trough + 1] <= 0)
    if not len(crossings):
        raise ValueError('Recorded Door target never crosses the initial angle')
    crossing = left + int(crossings[0])
    anchors = {
        'right_preparation_start': right,
        'right_extreme': int(np.argmax(signal[:left + 1])) if right is not None else None,
        'left_strike_start': left, 'before_zero_crossing': max(left, crossing - 1),
        'zero_crossing': crossing, 'left_extreme': trough,
        'rebound_start': sustained_start(signal, 1, after=trough + 1),
    }
    return {
        'environment': annotation['environment'], 'episode_index': annotation['episode_index'],
        'num_frames': len(signal), 'source': 'target_action', 'status': 'target_derived',
        'signal_unit': 'degree_about_base_negative_x_relative_to_first_frame',
        'events': {name: event(frame, fps, signal) for name, frame in anchors.items()},
    }


def prepare(task, version, allow_fast_right_start=False, root_override=None, ball_pre_return_margin=2):
    name, frames, stride, width, height, total_envs, initial, batch_envs, batch_eps, action_type, preferred = TASKS[task]
    root = Path(root_override) if root_override else DATA / name
    dest = ROOT / 'data' / f'real97_{task}_target_{version}'
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite an existing manifest: {dest}')
    splits = split_map(root, task)
    annotations = {} if task == 'soft' else {
        (row['lerobot_dataset'], row['episode_index']): row
        for row in jsonl(root / 'chunk_events_v1.jsonl')
    }
    info_paths = sorted(root.glob('*_lerobot/meta/info.json'))
    train_envs = [p.parent.parent.name for p in info_paths
                  if p.parent.parent.name.removesuffix('_lerobot') not in HELDOUT]
    if len(train_envs) != total_envs:
        raise ValueError(f'{task}: expected {total_envs} train environments, found {len(train_envs)}')
    env_ids = {env: i for i, env in enumerate(train_envs)}
    rows = {'train': [], 'test': [], 'ood': []}
    inventory, commands, stage_files, counts = [], [], set(), Counter()
    span = (frames - 1) * stride + 1
    fast_fallbacks = []
    recorded_events = []
    for info_path in info_paths:
        env = info_path.parent.parent.name
        info = json.loads(info_path.read_text())
        if float(info['fps']) != 20:
            raise ValueError(f'Unexpected fps: {info_path}')
        features = info['features']
        if action_type == 'joint_target':
            required = ['raw.commanded_target.joint_rad', 'raw.commanded_target.arm_sent']
        elif action_type == 'joint_target_export':
            required = ['target_joint_action', 'target_action']
            names = features.get('target_joint_action', {}).get('names', [])
            if names[:7] != [f'target_joint_{i}_rad' for i in range(1, 8)]:
                raise ValueError(f'Not an explicitly joint-target export: {info_path}')
        else:
            names = features['action'].get('names', [])
            if names[:7] != ['target_x_m', 'target_y_m', 'target_z_m', 'target_qw', 'target_qx', 'target_qy', 'target_qz']:
                raise ValueError(f'Not an explicitly target-EEF action export: {info_path}')
            required = ['action']
        missing = [key for key in required if key not in features]
        if missing:
            raise ValueError(f'{task}/{env}: missing recorded targets {missing}; no measured fallback')
        views = [key for key, value in features.items() if value.get('dtype') == 'video']
        if len(views) != 1:
            raise ValueError(f'Expected one AgentView video: {info_path}: {views}')
        for ep in jsonl(info_path.parent / 'episodes.jsonl'):
            index, length = int(ep['episode_index']), int(ep['length'])
            key = (env, index)
            split = splits.get(key, ep.get('split'))
            if split not in ('train', 'test'):
                raise ValueError(f'Unspecified frozen episode split: {key}')
            if ep.get('split') is not None and ep['split'] != split:
                raise ValueError(f'Conflicting frozen splits: {key}')
            output_split = 'ood' if env not in env_ids else split
            fmt = dict(episode_index=index, episode_chunk=index // int(info.get('chunks_size', 1000)), video_key=views[0])
            parquet = f"{env}/{info['data_path'].format(**fmt)}"
            video = f"{env}/{info['video_path'].format(**fmt)}"
            annotation = annotations.get(key)
            if task != 'soft' and (annotation is None or annotation['num_frames'] != length):
                raise ValueError(f'Missing or stale chunk annotation: {key}')
            target_table = None
            if task == 'door':
                target_table = pq.read_table(root / parquet, columns=required)
                annotation = refreshed_door_events(annotation, target_table, float(info['fps']))
            elif task == 'ball':
                annotation = dict(annotation)
                annotation['events'] = dict(annotation['events'])
                middle = int(annotation['events']['return_to_initial']['frame'])
                boundary = middle - int(ball_pre_return_margin)
                annotation['events']['pre_return'] = {
                    'frame': boundary, 'time_s': boundary / float(info['fps']),
                }
                annotation['pre_return_margin_frames'] = int(ball_pre_return_margin)
                annotation['status'] = 'target_derived'
            if task == 'soft':
                ranges = {'general': (0, max(0, length - span))}
            elif task == 'stick':
                lift = annotation['lift_interval']
                ranges = {
                    'general': (0, max(0, length - span)),
                    'full_lift': (max(0, lift['end_exclusive'] - span),
                                  min(lift['start_inclusive'], max(0, length - span))),
                }
            else:
                events = annotation['events']
                start = events['right_start' if task == 'ball' else 'left_strike_start']['frame']
                end = events['pre_return' if task == 'ball' else 'before_zero_crossing']['frame']
                if start > end:
                    if task != 'ball' or not allow_fast_right_start:
                        raise ValueError(f'{key}: precise interval [{start}, {end}] is empty; explicit sampling approval required')
                    fast_fallbacks.append({'environment': env, 'episode_index': index, 'R': start, 'B': end})
                    precise_end = start
                else:
                    precise_end = end
                ranges = {'precise': (start, precise_end), 'wide': (0, end)}
            if task in ('ball', 'door'):
                recorded_events.append({
                    'environment': env, 'episode_index': index, 'split': output_split,
                    'num_frames': length, 'source': annotation['source'],
                    'events': annotation['events'], 'start_ranges': ranges,
                })
            for kind, (low, high) in ranges.items():
                if not 0 <= low <= high < length:
                    raise ValueError(f'Empty/invalid chunk range, never prune: {key}, {kind}, {low}, {high}, N={length}')
                for start in range(low, high + 1):
                    rows[output_split].append({
                        'sample_id': f'{env}/ep{index:06d}/{kind}/{start:04d}',
                        'video': [video], 'action': parquet, 'start_frame': start,
                        'end_frame': min(length - 1, start + span - 1), 'frame_stride': stride,
                        'length': frames, 'total_frames': length, 'sampling_kind': kind,
                        'environment': env.removesuffix('_lerobot'), 'environment_index': env_ids.get(env, -1),
                        'friction_mu': float(env_ids.get(env, -1)), 'action_id': index,
                        'episode_index': index, 'source_episode_index': ep.get('source_episode_index', index),
                        'dataset_split': output_split, 'action_semantics': action_type,
                        'prompt': 'Predict the video conditioned on the recorded robot targets.',
                    })
            inventory.append({'environment': env, 'episode_index': index, 'split': output_split,
                              'source_split': split, 'length': length, 'action': parquet, 'video': video,
                              'start_ranges': ranges})
            if output_split == 'train':
                counts[env] += 1
                stage_files.update([parquet, video, f'{env}/meta/info.json'])
                table = (target_table if target_table is not None else pq.read_table(root / parquet, columns=required)).to_pydict()
                if action_type == 'joint_target':
                    if not np.asarray(table[required[1]], dtype=bool).all():
                        raise ValueError(f'Unsent training target: {parquet}')
                    values = np.asarray(table[required[0]], dtype=np.float32)
                elif action_type == 'joint_target_export':
                    exported = np.asarray(table['target_joint_action'], dtype=np.float32)
                    if exported.shape != (length, 8):
                        raise ValueError(f'Invalid target_joint_action shape: {parquet}')
                    values = exported[:, :7]
                else:
                    values = np.asarray(table['action'], dtype=np.float32)
                expected_width = 7 if action_type in ('joint_target', 'joint_target_export') else 8
                if values.shape != (length, expected_width) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid training target shape/values: {parquet}, {values.shape}')
                commands.append(values)
    if any(counts[env] < batch_eps for env in train_envs):
        raise ValueError(f'Insufficient distinct train episodes, not pruning: {dict(counts)}')
    values = np.concatenate(commands)
    stats = {action_type: {'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
                          'source': 'all native frames of training episodes only',
                          'constant_channels': 'zero', 'packing': 'target channels first, zero padding to 14'}}
    dest.mkdir(parents=True)
    for split, records in rows.items():
        with (dest / f'{split}.jsonl').open('w') as stream:
            for row in records:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    with (dest / 'episode_split.jsonl').open('w') as stream:
        for row in inventory:
            stream.write(json.dumps(row) + '\n')
    if recorded_events:
        with (dest / 'chunk_events.jsonl').open('w') as stream:
            for row in recorded_events:
                stream.write(json.dumps(row) + '\n')
    (dest / 'stage_files.txt').write_text('\n'.join(sorted(stage_files)) + '\n')
    (dest / 'action_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    fingerprint = hashlib.sha256((dest / 'train.jsonl').read_bytes()).hexdigest()
    summary = {'task': task, 'dataset_base_path': str(root), 'version': version,
               'train_manifest_sha256': fingerprint, 'environment_ids': env_ids,
               'train_episodes_per_environment': dict(counts),
               'episode_counts': dict(Counter(row['split'] for row in inventory)),
               'chunk_counts': {key: len(value) for key, value in rows.items()},
               'action_type': action_type, 'action_source_fields': required,
               'ball_pre_return_margin_frames': int(ball_pre_return_margin) if task == 'ball' else None,
               'door_event_source': 'target_action' if task == 'door' else None,
               'model_size_wh': [width, height], 'model_frames': frames, 'frame_stride': stride,
               'padding': 'min(start + stride*k, episode_length-1) for both video and target',
               'batch_per_gpu': [batch_envs, batch_eps], 'curriculum_groups': [initial] * (total_envs // initial),
               'fast_ball_precise_branch_fallbacks': fast_fallbacks,
               'heldout_environments': sorted(HELDOUT) if task == 'soft' else []}
    (dest / 'manifest_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    template = ROOT / 'configs/train/train_real_ball_friction_align_medium_7env_60f_curriculum_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500.yaml'
    config = yaml.safe_load(template.read_text())
    # Flatten existing supported settings, then emit one unambiguous section per family.
    base = {key: value for section in config.values() if isinstance(section, dict) for key, value in section.items()}
    overrides = {
        'dataset_base_path': str(root), 'dataset_metadata_path': str(dest / 'train.jsonl'),
        'action_stat_path': str(dest / 'action_stats.json'), 'action_type': action_type, 'action_dim': 14,
        'height': height, 'width': width, 'num_frames': frames, 'frame_stride': stride,
        'resize_mode': 'letterbox', 'pad_short_chunks': True, 'dataset_repeat': 1, 'dataset_num_workers': 0,
        'video_light_augmentation_enabled': True, 'video_light_augmentation_probability': .7,
        'video_light_augmentation_gradient_abs_max': .08, 'video_light_augmentation_noise_std': .004,
        'spatial_loss_mode': 'none', 'learning_rate': 1e-5, 'weight_decay': .01,
        'grouped_context_init_mode': 'uniform', 'grouped_context_init_min': -1., 'grouped_context_init_max': 1.,
        'grouped_context_clamp_min': -1., 'grouped_context_clamp_max': 1.,
        'grouped_context_lr': .03, 'grouped_context_new_context_lr': .15,
        'grouped_context_model_lr_warmup_steps': 100, 'grouped_context_context_lr_warmup_steps': 0,
        'grouped_context_model_phase_warmup_steps': 0, 'grouped_context_reset_model_optimizer_state_on_phase_start': False,
        'grouped_context_phase_max_displacement': 0., 'grouped_context_post_curriculum_lr': None,
        'grouped_context_weight_decay': 0., 'grouped_context_structured_updates': 5500,
        'grouped_context_friction_groups_per_update': batch_envs, 'grouped_context_actions_per_update': batch_eps,
        'grouped_context_microbatches_per_update': batch_envs * batch_eps,
        'grouped_context_sampling_mode': 'uniform_episode_then_window', 'grouped_context_episode_key': 'episode_index',
        'grouped_context_preferred_window_kind': 'full_lift' if task == 'stick' else 'precise',
        'grouped_context_preferred_window_probability': preferred,
        'grouped_context_curriculum_initial_groups': initial, 'grouped_context_curriculum_add_groups': initial,
        'grouped_context_curriculum_total_groups': total_envs, 'grouped_context_curriculum_initial_model_steps': 300,
        'grouped_context_curriculum_new_context_steps': 200, 'grouped_context_curriculum_all_context_steps': 200,
        'grouped_context_curriculum_model_steps': 200, 'grouped_context_post_curriculum_cycle_steps': 200,
        'grouped_context_curriculum_variant': 'default', 'grouped_context_validation_interval': 100,
        'grouped_context_protected_checkpoint_steps': '2300', 'grouped_context_model_phase_checkpoint_policy': 'all',
        'grouped_context_stage_checkpoints_locally': True, 'grouped_context_resume_step': 0,
        'grouped_context_resume_context_table': None,
        'physical_context_dim': 32, 'physical_context_tokens': 1,
        'output_path': str(ROOT / 'outputs' / f'real97_{task}_target_c32_{version}'),
        'ckpt_path': 'ckpt/BLM/step-12000.safetensors', 'resume_from': None,
        'checkpoint_keep_last': 2, 'log_steps': 2, 'save_steps': 100000,
        'checkpoint_save_minutes': 100000, 'batch_size': 1, 'gradient_accumulation_steps': 1,
        'seed': 20260909,
    }
    base.update(overrides)
    for key in list(base):
        if key.startswith('spatial_loss_') and key != 'spatial_loss_mode':
            del base[key]
    config_path = ROOT / 'configs/train' / f'train_real97_{task}_target_c32_2gpu_{version}.yaml'
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(yaml.safe_dump({'training': base}, sort_keys=False))
    print(json.dumps({'task': task, 'config': str(config_path), 'metadata': str(dest),
                      'train_episodes': sum(counts.values()), 'chunks': len(rows['train']),
                      'batch_per_gpu': [batch_envs, batch_eps]}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--allow-fast-ball-right-start', action='store_true')
    parser.add_argument('--ball-pre-return-margin', type=int, default=2)
    parser.add_argument('--dataset-root', help='Only supported when preparing a single task')
    args = parser.parse_args()
    if args.dataset_root and len(args.tasks) != 1:
        parser.error('--dataset-root requires exactly one task')
    for task in args.tasks:
        prepare(task, args.version, args.allow_fast_ball_right_start, args.dataset_root, args.ball_pre_return_margin)
