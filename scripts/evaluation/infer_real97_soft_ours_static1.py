#!/usr/bin/env python3
"""Single observed frame, 32 predicted frames, canonical training target EEF.

Reuse historical evaluation episode identities, never their five-frame prefix.
Training legal-start restrictions do not constrain this episode-start evaluation.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml
from real97_trial_common import ROOT, read_jsonl, require_compute, write_json, write_jsonl
from infer_real97_reference_trial import labeled, pca_plot, write_video
from infer_real97_soft_ours_history5_static import cached_copy, frozen_copy, resolve, rgb_video


def episode_start(row, role):
    result = copy.deepcopy(row)
    last = min(96, int(row['total_frames']) - 1)
    if last < 0:
        raise ValueError('Empty evaluation episode')
    native = [min(3 * index, last) for index in range(33)]
    for key in ('support_direction', 'support_dx_px', 'support_excursion_px', 'support_measurement_frames'):
        if key in result:
            result['original_window_' + key] = result.pop(key)
    result.update(
        source_sample_id=row['sample_id'], source_start_frame=row['start_frame'],
        source_end_frame=row['end_frame'],
        sample_id=f"{row['environment']}/ep{row['episode_index']:06d}/static1_{role}",
        start_frame=0, end_frame=last, frame_stride=3, length=33,
        sampling_kind='evaluation_episode_start', action_semantics='eef_target',
        num_history_frames=1, history_anchor_frame=0, history_padding_frames=0,
        native_frame_indices=native, observed_native_frame_indices=[0],
        prediction_native_frame_indices=native[1:],
        evaluation_frame_indices=[index for index in range(1, 33) if 3 * index <= last],
        comparison_evaluation_frame_indices=[index for index in range(1, 33) if 3 * index <= last],
        prediction_padding_frames=sum(3 * index > last for index in range(1, 33)),
        initial_stationarity='episode-start property supplied by user; no detector selection',
    )
    return result


def freeze_rows(path, rows):
    if path.exists():
        if read_jsonl(path) != rows:
            raise RuntimeError(f'Frozen evaluation selection changed: {path}')
    else:
        write_jsonl(path, rows)


def protect_checkpoint(config):
    run = resolve(config['training_run'])
    snapshot = resolve(config['snapshot'])
    snapshot.mkdir(parents=True, exist_ok=True)
    source = run / f"step-{config['model_step']}.safetensors"
    table = run / f"step-{config['table_step']}.context_table.json"
    if config['model_step'] != config['table_step']:
        raise ValueError('Model/table must come from the same step')
    if not (run / f".step-{config['model_step']}.safetensors.complete").is_file():
        raise RuntimeError('Training checkpoint has no completed-publication marker')
    with (snapshot / '.snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen_copy(run / 'submitted_config.yaml', snapshot / 'training_config.yaml')
        frozen_copy(table, snapshot / 'context_table.json')
        source_cfg = yaml.safe_load((snapshot / 'training_config.yaml').read_text())['training']
        stats = run / 'input_manifest/action_stats.json'
        if not stats.is_file():
            stats = resolve(source_cfg['action_stat_path'])
        frozen_copy(stats, snapshot / 'action_stats.json')
        destination = snapshot / 'model.safetensors'
        identity = {'source': str(source.resolve()), 'bytes': source.stat().st_size,
                    'mtime_ns': source.stat().st_mtime_ns, 'step': config['model_step']}
        if destination.exists():
            if (destination.stat().st_size != identity['bytes']
                    or json.loads((snapshot / 'model_identity.json').read_text()) != identity):
                raise RuntimeError('Protected model identity changed')
        else:
            temporary = snapshot / 'model.safetensors.partial'
            temporary.unlink(missing_ok=True)
            try:
                os.link(source, temporary)
            except OSError:
                subprocess.run(['rsync', '-a', str(source), str(temporary)], check=True)
            os.replace(temporary, destination)
            write_json(snapshot / 'model_identity.json', identity)
    return snapshot


def prepare(config, output):
    snapshot = protect_checkpoint(config)
    original = resolve(config['original_prepared'])
    for name in ('training_config.yaml', 'context_table.json', 'action_stats.json'):
        frozen_copy(snapshot / name, output / 'input_manifest' / name)
    frozen_copy(original / 'plan.json', output / 'input_manifest/original_plan.json')
    frozen_copy(original / 'support.jsonl', output / 'input_manifest/original_support.jsonl')
    frozen_copy(resolve(config['query_episode_reference']), output / 'input_manifest/original_query.jsonl')
    plan = json.loads((output / 'input_manifest/original_plan.json').read_text())
    supports = [episode_start(row, 'support') for row in read_jsonl(output / 'input_manifest/original_support.jsonl')]
    queries = [episode_start(row, 'query') for row in read_jsonl(output / 'input_manifest/original_query.jsonl')]
    if len(queries) != config['query_count'] or len(plan['environments']) != config['environment_count']:
        raise RuntimeError('Unexpected reference episode count')
    training = yaml.safe_load((output / 'input_manifest/training_config.yaml').read_text())
    flat = {key: value for section in training.values() if isinstance(section, dict) for key, value in section.items()}
    trained_episodes = {
        (str(row['environment']).removesuffix('_lerobot'), int(row['episode_index']))
        for row in read_jsonl(resolve(config['training_run']) / 'input_manifest/train.jsonl')
    }
    for row in supports + queries:
        row['episode_seen_in_training_manifest'] = (row['environment'], int(row['episode_index'])) in trained_episodes
    for env in plan['environments']:
        selected = [supports[index] for index in env['support_indices']]
        if len(selected) != config['support_count_per_environment']:
            raise RuntimeError('Unexpected support count')
        if any(row['dataset_split'] != 'train' or row['environment'] != env['environment'] for row in selected):
            raise RuntimeError('Support split/environment mismatch')
        support_ids = {row['episode_index'] for row in selected}
        if any(queries[index]['environment'] != env['environment'] or
               queries[index]['episode_index'] in support_ids for index in env['query_indices']):
            raise RuntimeError('Query identity mismatch or support/query episode overlap')
        env['original_window_direction_counts'] = env.pop('direction_counts', None)
    freeze_rows(output / 'support.jsonl', supports)
    freeze_rows(output / 'query.jsonl', queries)
    source_dataset = Path(flat['dataset_base_path'])
    files = sorted({path for row in supports + queries for path in [*row['video'], row['action']]})
    if any(Path(path).is_absolute() or '..' in Path(path).parts for path in files):
        raise RuntimeError('Expected dataset-relative video/parquet paths')
    filelist = output / 'input_manifest/stage_files.txt'
    filelist.write_text('\n'.join(files) + '\n')
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    wan = cache / 'Wan2.2-TI2V-5B'
    cached_copy(ROOT / 'models/Wan2.2-TI2V-5B', wan, directory=True)
    data_identity = str(source_dataset.resolve()) + '\n' + '\n'.join(files)
    local_data = cache / 'eval_datasets' / ('soft_static1_' + hashlib.sha256(data_identity.encode()).hexdigest()[:16])
    cached_copy(source_dataset, local_data, directory=True, files=filelist)
    model_source = snapshot / 'model.safetensors'
    identity = f'{model_source.resolve()}:{model_source.stat().st_size}:{model_source.stat().st_mtime_ns}'
    checkpoint = cache / 'eval_checkpoints' / ('soft_static1_' + hashlib.sha256(identity.encode()).hexdigest()[:20] + '.safetensors')
    cached_copy(model_source, checkpoint)
    flat.update(plan['protocol']['ttt'])
    flat.update(
        dataset_base_path=str(local_data), dataset_metadata_path=str(output / 'query.jsonl'),
        action_stat_path=str(output / 'input_manifest/action_stats.json'), ckpt_path=str(checkpoint),
        model_paths=str(wan), output_path=str(output / 'raw/stage2'), seed=config['seed'],
        max_samples=0, fps=7, quality=8, num_frames=33, num_history_frames=1, frame_stride=3,
        video_light_augmentation_enabled=False, spatial_loss_mode='none',
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        stage2_group_keys='environment', stage2_inner_lr_schedule=config['stage2_inner_lr_schedule'],
        stage2_inner_steps=config['stage2_inner_steps'],
        stage2_context_clamp_min=config['stage2_context_clamp_min'],
        stage2_context_clamp_max=config['stage2_context_clamp_max'],
        num_inference_steps=config['num_inference_steps'],
    )
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference': flat}, sort_keys=False))
    plan.update(source_dataset=str(source_dataset), model_step=config['model_step'], table_step=config['table_step'],
                window_policy='same_episode_ids_at_start_0_stride3_until_96_with_end_padding',
                observed_frames=1, predicted_frames=32, direction_labels_recertified=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'provenance.json', {
        **config, 'protected_snapshot': str(snapshot), 'local_checkpoint': str(checkpoint),
        'source_dataset': str(source_dataset), 'old_gt_videos_reused': False,
        'action': 'same CanonicalTargetEEF as training; eight normalized channels plus six zeros',
        'support_loss': 'legacy single-frame flow matching; clean first latent excluded from target loss',
        'stage2_trainable': 'Z only', 'query_GT_used_for_adaptation': False,
        'episode_start_evaluation_may_differ_from_training_legal_start_list': True,
        'original_window_direction_labels_not_recertified': True,
    })
    return plan, runtime, checkpoint, queries, supports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    own = parser.parse_args()
    require_compute()
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one allocated usable GPU is required')
    output = own.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen_copy(own.config, output / 'input_manifest/experiment.json')
    config = json.loads(own.config.read_text())
    plan, runtime, checkpoint, queries, supports = prepare(config, output)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from wan_video_action.data.history_prefix import CanonicalTargetEEF
    from wan_video_action.utils import set_global_seed
    sys.argv = ['infer_stage2_ttt', '--config', str(runtime), '--stage2_ckpt_path', str(checkpoint),
                '--support_metadata_path', str(output / 'support.jsonl')]
    args = ttt.parse_args()
    if (args.num_frames, args.num_history_frames, args.frame_stride, args.action_type, args.action_dim,
            args.height, args.width, args.physical_context_dim) != (33, 1, 3, 'eef_target', 14, 160, 320, 32):
        raise RuntimeError('Single-frame training/inference contract mismatch')
    stats = json.loads((output / 'input_manifest/action_stats.json').read_text())['eef_target']
    def dataset_for(path):
        dataset_args = copy.copy(args)
        dataset_args.dataset_metadata_path = str(path)
        dataset = build_infer_dataset(dataset_args)
        dataset.special_operator_map['action'] = CanonicalTargetEEF(dataset_args, stats)
        return dataset
    query_dataset = dataset_for(output / 'query.jsonl')
    support_dataset = dataset_for(output / 'support.jsonl')
    set_global_seed(int(args.seed))
    pipe = build_pipeline(args)
    ttt._freeze_pipe(pipe)
    table = json.loads((output / 'input_manifest/context_table.json').read_text())
    lookup = {float(row['friction_mu']): torch.as_tensor(row['context'], device=pipe.device,
              dtype=torch.float32).reshape(1, 32) for row in table['records']}
    if set(lookup) != {float(env['environment_index']) for env in plan['environments']}:
        raise RuntimeError('Training-table environment identities differ from evaluation')
    initial = torch.stack(list(lookup.values())).mean(0)
    contexts, results = {}, []
    for env in plan['environments']:
        name = env['environment']
        context_path = output / 'contexts' / (name + '.json')
        if context_path.exists():
            state = json.loads(context_path.read_text())
            if (state['model_step'], state['table_step'], state['observed_frames']) != (config['model_step'], config['table_step'], 1):
                raise RuntimeError('Cannot reuse a different model/table/history context')
            adapted = torch.as_tensor(state['context'], device=pipe.device, dtype=torch.float32)
        else:
            support = [support_dataset[index] for index in env['support_indices']]
            set_global_seed(int(config['seed']) + int(env['environment_index']) * 1009)
            adapted, losses, _, metrics, trajectory = ttt._adapt_ttt_state(
                pipe, support, args, adapter_named_params=[], base_adapter_state={}, initial_context=initial,
                target_context=None, trajectory_meta={'environment': name, 'sample_index': env['query_indices'][0],
                                                       'friction_mu': float(env['environment_index'])})
            state = {'context': adapted.detach().cpu().tolist(), 'initial': initial.cpu().tolist(),
                     'losses': losses, 'metrics': metrics, 'trajectory': trajectory,
                     'support_indices': env['support_indices'], 'model_step': config['model_step'],
                     'table_step': config['table_step'], 'observed_frames': 1}
            write_json(context_path, state)
            del support
        contexts[name] = state
        columns = []
        for index in env['query_indices']:
            row = queries[index]
            key = f"q{index:04d}_{name}_{row['dataset_split']}_ep{row['episode_index']:06d}"
            paths = {kind: output / 'raw' / kind / (key + '.mp4') for kind in ('gt', 'stage1', 'stage2')}
            marker = output / 'completed' / (key + '.json')
            if not marker.exists():
                sample = query_dataset[index]
                if tuple(sample['video'].shape) != (1, 3, 33, 160, 320) or tuple(sample['action'].shape) != (1, 33, 14):
                    raise RuntimeError('Expected 33 synchronized video/target-EEF frames')
                write_video(paths['gt'], rgb_video(sample['video']), config['fps'])
                for kind, context in (('stage1', lookup[float(env['environment_index'])]), ('stage2', adapted)):
                    variant = output / 'completed_variants' / (key + '_' + kind + '.json')
                    if variant.exists():
                        continue
                    args.seed = int(config['seed']) + index
                    set_global_seed(args.seed)
                    rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                    rollout.update(physical_context=context, output_path=str(paths[kind]))
                    _run_autoregressive(pipe, rollout, args)
                    frames = imageio.mimread(str(paths[kind]))
                    if len(frames) != 33:
                        raise RuntimeError('Rollout must contain one observed and 32 predicted frames')
                    write_video(paths[kind], frames, config['fps'])
                    write_json(variant, {'query': key, 'method': kind, 'seed': args.seed, 'observed_frames': 1})
                write_json(marker, {'index': index, 'row': row, 'paths': {kind: str(path) for kind, path in paths.items()},
                                    'observed_frames': 1, 'paired_gt': True})
                del sample
                torch.cuda.empty_cache()
            videos = {kind: imageio.mimread(str(path)) for kind, path in paths.items()}
            if any(len(video) != 33 for video in videos.values()):
                raise RuntimeError('GT/Stage1/Stage2 frame counts differ')
            column = [np.vstack([labeled(np.asarray(videos[kind][frame]),
                       f"{kind} | {row['dataset_split']} ep{row['episode_index']}") for kind in ('gt', 'stage1', 'stage2')])
                      for frame in range(33)]
            write_video(output / 'comparisons' / (key + '.mp4'), column, config['fps'])
            columns.append(column)
            results.append({'index': index, 'environment': name, 'split': row['dataset_split'],
                            'paths': {kind: str(path) for kind, path in paths.items()}})
            write_json(output / 'progress.json', {'completed_environments': list(contexts), 'query_results': results})
            print('[query_done] ' + key, flush=True)
        write_video(output / 'grids' / (name + '_gt_stage1_stage2_train_test.mp4'),
                    (np.hstack([column[frame] for column in columns]) for frame in range(33)), config['fps'])
        pca_plot(output / 'training_inference_Z_pca.svg', table, contexts,
                 [item for item in plan['environments'] if item['environment'] in contexts])
    write_json(output / 'inference_complete.json', {
        'method': 'ours_static_single_frame', 'model_step': config['model_step'], 'table_step': config['table_step'],
        'queries': len(results), 'environments': len(contexts), 'observed_frames': 1, 'prediction_frames': 32,
        'raw_frames': 33, 'metric_status': 'CPU_scoring_pending', 'formal_metric_approved': False,
    })
    print('[inference_complete] ' + str(output), flush=True)


if __name__ == '__main__':
    main()
