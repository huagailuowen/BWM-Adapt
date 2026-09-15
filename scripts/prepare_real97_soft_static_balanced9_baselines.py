#!/usr/bin/env python3
"""Publish a physical-nine train-only manifest shared by the Soft baselines."""
from __future__ import annotations
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/real97_soft_static_balanced9_dual_20260913_v1'
DESTINATION = ROOT / 'data/real97_soft_static_balanced9_baselines_20260914_v1'


def read(path):
    return json.loads(path.read_text())


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Dataset preparation must run on a compute allocation')
    summary = read(SOURCE / 'manifest_summary.json')
    dataset = Path(summary['dataset_base_path'])
    selection = read(dataset / 'TRAINING_ENVIRONMENT_SELECTION.json')
    env_split = read(dataset / 'ENVIRONMENT_SPLIT.json')
    names = summary['selected_physical_environments']
    if (selection['selected_environments'] != names or env_split['train_environments'] != names
            or env_split['status'] != 'approved'
            or selection['episode_exclusions'] != {'soft-6m': [10]}):
        raise RuntimeError('Live dataset selection differs from the frozen nine-environment contract')
    payload = (SOURCE / 'physical_train.jsonl').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != summary['manifest_hashes']['physical_train']:
        raise RuntimeError('Physical train manifest hash mismatch')
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    test_payload = (SOURCE / 'test.jsonl').read_bytes()
    if hashlib.sha256(test_payload).hexdigest() != summary['manifest_hashes']['test']:
        raise RuntimeError('Test manifest hash mismatch')
    tests = [json.loads(line) for line in test_payload.splitlines() if line.strip()]
    approved_train = {
        item['environment']: set(item['eligible_train_episode_indices'])
        for item in selection['environments']
    }
    approved_test = {
        item['environment']: set(item['eligible_test_episode_indices'])
        for item in selection['environments']
    }
    live_episodes = {}
    for name in names:
        path = dataset / (name + '_lerobot') / 'meta/episodes.jsonl'
        live_episodes[name] = {
            int(row['episode_index']): row
            for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        }
    train_episodes = defaultdict(set)
    train_ids, test_ids = set(), set()
    for row in rows:
        name, ep = row['environment'], int(row['episode_index'])
        if name not in names or int(row['environment_index']) != names.index(name):
            raise RuntimeError('Unexpected or duplicated virtual environment in baseline manifest')
        if row['dataset_split'] != 'train' or ep not in approved_train[name]:
            raise RuntimeError('Non-training or excluded episode in baseline manifest')
        if 'latent_replica' in row or row['sampling_kind'] != 'stationary_valid':
            raise RuntimeError('Baseline must use physical episodes and approved static starts')
        start = int(row['start_frame'])
        annotation = live_episodes[name][ep]
        if start not in annotation.get('valid_33x3_starts', []):
            raise RuntimeError(f'Start is not currently approved: {name}/{ep}/{start}')
        if (row['native_frame_indices'] != [start + 3 * i for i in range(33)]
                or int(row['end_frame']) != start + 96
                or start + 96 >= int(annotation['length'])
                or row['action_semantics'] != 'eef_target'):
            raise RuntimeError('Static video/action alignment differs from the Ours contract')
        train_episodes[name].add(ep)
        train_ids.add((name, ep))
    for row in tests:
        name, ep = row['environment'], int(row['episode_index'])
        if row['dataset_split'] != 'test' or name not in names or ep not in approved_test[name]:
            raise RuntimeError('Unexpected test episode')
        test_ids.add((name, ep))
    if (len(rows), len(train_ids), len(test_ids), len(tests)) != (2680, 146, 18, 277):
        raise RuntimeError('Counts differ from the approved training/test split')
    if train_ids & test_ids or any(len(train_episodes[name]) < 6 for name in names):
        raise RuntimeError('Train/test overlap or insufficient episodes; never prune an environment')
    if ('soft-6m', 10) in train_ids | test_ids:
        raise RuntimeError('Disabled soft-6m episode 10 leaked into a manifest')
    data_contract = {
        'version': 'soft_static_balanced9_baselines_20260914_v1',
        'source_manifest': str(SOURCE), 'source_train_manifest_sha256': digest,
        'environments': names, 'latent_replicas': False,
        'training_only': True, 'eligible_train_episodes': 146, 'heldout_test_episodes': 18,
        'sampling': 'each_rank_uniform_5_env_then_6_distinct_episodes_then_legal_window',
        'num_history_frames': 1, 'num_frames': 33, 'frame_stride': 3,
        'action': 'CanonicalTargetEEF: eight normalized target coordinates plus six zeros',
        'normalization_reused_from_ours': True,
        'batch_per_gpu': [5, 6], 'gpus': 2, 'global_sampled_chunks_per_update': 60,
        'dino_support_k_choices': [1, 2], 'dino_queries': 'remaining four or five episodes',
        'ttt': 'one shuffled six-chunk write-then-predict stream per environment',
        'curriculum': None, 'ROI': False, 'lighting_probability': 0.7,
        'allocation_checkpoint_seconds': 84600, 'checkpoint_keep_last': 2,
        'protected_checkpoint_step': 2300,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with (DESTINATION.parent / (DESTINATION.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if DESTINATION.exists():
            if read(DESTINATION / 'data_contract.json') != data_contract:
                raise RuntimeError('Refusing to overwrite a different frozen baseline manifest')
            print('[soft_baselines_manifest] reuse', DESTINATION, flush=True)
            return
        temporary = DESTINATION.with_name(DESTINATION.name + '.partial-' + str(os.getpid()))
        temporary.mkdir()
        (temporary / 'train.jsonl').write_bytes(payload)
        (temporary / 'physical_train.jsonl').write_bytes(payload)
        (temporary / 'test.jsonl').write_bytes(test_payload)
        for name in ('action_stats.json', 'stage_files.txt', 'episode_split.jsonl', 'excluded_episodes.json'):
            shutil.copy2(SOURCE / name, temporary / name)
        shutil.copytree(SOURCE / 'source_metadata', temporary / 'source_metadata')
        for name in ('TRAINING_ENVIRONMENT_SELECTION.json', 'ENVIRONMENT_SPLIT.json'):
            shutil.copy2(dataset / name, temporary / 'source_metadata' / name)
        result = {
            'task': 'soft', 'version': data_contract['version'], 'dataset_base_path': str(dataset),
            'train_manifest_sha256': digest,
            'manifest_hashes': {'train': digest, 'physical_train': digest, 'test': summary['manifest_hashes']['test']},
            'environment_ids': {name: i for i, name in enumerate(names)},
            'selected_physical_environments': names, 'physical_environment_count': 9,
            'train_episodes_per_environment': {name: len(train_episodes[name]) for name in names},
            'episode_counts': {'train': 146, 'test': 18},
            'chunk_counts': {'train': len(rows), 'test': len(tests)},
            'source_dataset_cache_identity': summary['source_dataset_cache_identity'],
            'action_type': 'eef_target', 'batch_per_gpu': [5, 6], 'num_history_frames': 1,
            'num_frames': 33, 'frame_stride': 3, 'model_size_wh': [320, 160],
        }
        for name, value in (('manifest_summary.json', result), ('data_contract.json', data_contract)):
            (temporary / name).write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temporary, DESTINATION)
        print('[soft_baselines_manifest]', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
