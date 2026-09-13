#!/usr/bin/env python3
"""Publish legal-start manifests only; never rewrite videos or recorded actions."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(contract_path):
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    digest = hashlib.sha256(contract_bytes).hexdigest()
    source = Path(contract['dataset_root'])
    destination = ROOT / contract['output_manifest_directory']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / (destination.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            summary = read_json(destination / 'manifest_summary.json')
            if summary.get('data_contract_sha256') != digest:
                raise ValueError('Existing frozen manifest has a different contract; refusing overwrite')
            print(json.dumps(summary, indent=2), flush=True)
            return
        split_path = source / contract['split_manifest']
        split_bytes = split_path.read_bytes()
        splits = json.loads(split_bytes)[contract['split_mapping_key']]
        source_hash = hashlib.sha256(split_bytes)
        infos = sorted(source.glob('soft-*_lerobot/meta/info.json'))
        heldout = set(contract['heldout_environments'])
        names = [p.parent.parent.name.removesuffix('_lerobot') for p in infos]
        train_names = [name for name in names if name not in heldout]
        if len(names) != 19 or len(train_names) != 15 or not heldout.issubset(names):
            raise ValueError('Unexpected dataset environment inventory')
        environment_ids = {name: index for index, name in enumerate(train_names)}
        records = {'train': [], 'test': [], 'ood': []}
        inventory, excluded = [], []
        stage_files = set()
        episode_counts = Counter()
        train_counts = Counter()
        total_starts = 0
        source_train_count = 0
        sampling = contract['sampling']
        for info_path in infos:
            info_bytes = info_path.read_bytes()
            source_hash.update(info_bytes)
            info = json.loads(info_bytes)
            directory = info_path.parent.parent.name
            name = directory.removesuffix('_lerobot')
            if float(info['fps']) != 20:
                raise ValueError(f'Expected native 20 Hz: {name}')
            action_names = info['features']['action'].get('names', [])
            if action_names != ['target_x_m', 'target_y_m', 'target_z_m', 'target_qw',
                                'target_qx', 'target_qy', 'target_qz', 'target_gripper_width_m']:
                raise ValueError(f'Expected recorded eight-channel target EEF: {name}')
            views = [key for key, value in info['features'].items() if value.get('dtype') == 'video']
            if len(views) != 1:
                raise ValueError(f'Expected one camera: {name}')
            train_ids = set(splits[name]['train_episode_indices'])
            test_ids = {int(row['episode_index']) for row in splits[name]['test']}
            if train_ids & test_ids:
                raise ValueError(f'Overlapping train/test episode IDs: {name}')
            episode_bytes = (info_path.parent / 'episodes.jsonl').read_bytes()
            source_hash.update(episode_bytes)
            episodes = [json.loads(line) for line in episode_bytes.splitlines() if line.strip()]
            if {int(row['episode_index']) for row in episodes} != train_ids | test_ids:
                raise ValueError(f'Incomplete frozen split: {name}')
            for episode in episodes:
                index, length = int(episode['episode_index']), int(episode['length'])
                source_split = 'train' if index in train_ids else 'test'
                split = 'ood' if name in heldout else source_split
                episode_counts[split] += 1
                source_train_count += int(split == 'train')
                annotation = episode.get('stationary_start_annotation', {})
                if (annotation.get('status') != sampling['required_annotation_status'] or
                        annotation.get('version') != sampling['required_annotation_version'] or
                        annotation.get('model_frames') != 33 or annotation.get('stride') != 3):
                    raise ValueError(f'Missing or incompatible start annotation: {name}/{index}')
                if sampling['start_field'] not in episode:
                    raise ValueError(f'Missing explicit legal-start list: {name}/{index}')
                starts = episode[sampling['start_field']]
                if (not isinstance(starts, list) or len(starts) != len(set(starts)) or
                        any(type(s) is not int or s < 0 or s + 96 >= length for s in starts)):
                    raise ValueError(f'Invalid complete 33x3 starts: {name}/{index}')
                total_starts += len(starts)
                fmt = dict(episode_index=index, episode_chunk=index // int(info.get('chunks_size', 1000)),
                           video_key=views[0])
                parquet = directory + '/' + info['data_path'].format(**fmt)
                video = directory + '/' + info['video_path'].format(**fmt)
                inventory.append({'environment': name, 'episode_index': index, 'split': split,
                                  'source_split': source_split, 'length': length, 'action': parquet,
                                  'video': video, 'valid_33x3_starts': starts})
                if not starts:
                    excluded.append({'environment': name, 'episode_index': index, 'split': split,
                                     'length': length, 'reason': 'empty_approved_valid_33x3_starts'})
                    continue
                environment_index = environment_ids.get(name, -1)
                if split == 'train':
                    train_counts[name] += 1
                    stage_files.update([parquet, video, directory + '/meta/info.json'])
                for start in starts:
                    records[split].append({
                        'sample_id': f'{directory}/ep{index:06d}/stationary_valid/{start:04d}',
                        'video': [video], 'action': parquet, 'start_frame': start,
                        'end_frame': start + 96, 'frame_stride': 3, 'length': 33,
                        'total_frames': length, 'native_frame_indices': [start + 3*k for k in range(33)],
                        'num_history_frames': 1, 'sampling_kind': 'stationary_valid',
                        'stationary_start_annotation_version': annotation['version'],
                        'environment': name, 'environment_index': environment_index,
                        'friction_mu': float(environment_index), 'action_id': index,
                        'episode_index': index,
                        'source_episode_index': episode.get('source_episode_index', index),
                        'dataset_split': split, 'source_dataset_split': source_split,
                        'action_semantics': 'eef_target',
                        'prompt': 'Predict the video conditioned on the recorded robot targets.',
                    })
        if any(train_counts[name] < 6 for name in train_names):
            raise ValueError(f'Insufficient eligible train episodes; never prune an environment: {train_counts}')
        expected = contract['expected_metadata_snapshot']
        actual = {'dataset_episodes': len(inventory), 'dataset_complete_valid_starts': total_starts,
                  'source_train_episodes': source_train_count, 'eligible_train_episodes': sum(train_counts.values()),
                  'eligible_train_starts': len(records['train'])}
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError(f'Dataset changed relative to the approved contract: {actual}')
        temporary = destination.with_name(destination.name + '.partial-' + str(os.getpid()))
        temporary.mkdir()
        hashes = {}
        for split, rows in records.items():
            hasher = hashlib.sha256()
            with (temporary / (split + '.jsonl')).open('wb') as handle:
                for row in rows:
                    encoded = (json.dumps(row, separators=(',', ':')) + '\n').encode()
                    handle.write(encoded)
                    hasher.update(encoded)
            hashes[split] = hasher.hexdigest()
        with (temporary / 'episode_split.jsonl').open('w') as handle:
            for row in inventory:
                handle.write(json.dumps(row) + '\n')
        stats_path = ROOT / contract['action_contract']['frozen_stats_path']
        stats = read_json(stats_path)
        if 'eef_target' not in stats:
            raise ValueError('Missing frozen EEF statistics')
        shutil.copy2(stats_path, temporary / 'action_stats.json')
        shutil.copy2(contract_path, temporary / 'data_contract.json')
        (temporary / 'TRAIN_TEST_SPLIT.json').write_bytes(split_bytes)
        (temporary / 'stage_files.txt').write_text('\n'.join(sorted(stage_files)) + '\n')
        write_json(temporary / 'excluded_episodes.json', excluded)
        summary = {
            'task': 'soft', 'version': 'eef_static_1frame_20260911_v1',
            'dataset_base_path': str(source), 'train_manifest_sha256': hashes['train'],
            'manifest_hashes': hashes, 'data_contract_sha256': digest,
            'source_metadata_sha256': source_hash.hexdigest(),
            'environment_ids': {name + '_lerobot': index for name, index in environment_ids.items()},
            'train_episodes_per_environment': dict(train_counts), 'episode_counts': dict(episode_counts),
            'chunk_counts': {split: len(rows) for split, rows in records.items()},
            'heldout_environments': sorted(heldout), 'excluded_episodes': excluded,
            'action_type': 'eef_target', 'action_stats_source': str(stats_path),
            'model_size_wh': [320, 160], 'model_frames': 33, 'frame_stride': 3,
            'num_history_frames': 1, 'batch_per_gpu': [5, 6], 'curriculum_groups': [5, 5, 5],
            'sampling': 'uniform_active_environment_then_eligible_episode_then_annotated_start',
            'padding_creates_candidates': False,
        }
        write_json(temporary / 'manifest_summary.json', summary)
        os.replace(temporary, destination)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.contract.resolve())
