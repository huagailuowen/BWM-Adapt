#!/usr/bin/env python3
"""Frozen Stick916 cohort, final DINO/TTT checkpoints and two train supports."""
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

from scripts.evaluation.infer_real97_ball_door_baselines import (
    frame_array, json_write, read_video, rows, snapshot, video_write,
)
from scripts.evaluation.infer_real97_reference_trial import labeled
from scripts.evaluation.infer_real97_standard_reference import copy_cached
from scripts.evaluation.infer_real97_soft_ttt_static_reference import select_checkpoint
from scripts.evaluation.infer_real97_ttt_reference import MultiSupportReplay
from scripts.infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
from scripts.methods.infer_dinov2_event80 import materialize_wan_checkpoint, load_support_encoder
from scripts.methods.train_dinov2_event80 import add_dinov2_config
from scripts.methods.infer_ttt_kvb_event80 import build_training_faithful_model
from scripts.methods.infer_ttt_kvb_oneminute_event80 import (
    SameTimestepSupportQueryModelFn, _prepare_support_model_inputs,
)
from scripts.methods.train_ttt_kvb_event80 import add_ttt_kvb_config
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


def query_key(index, row):
    return f"q{index:04d}_{row['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"


def episode(row):
    return row['environment'], int(row['episode_index'])


def check_sample(sample):
    if tuple(sample['video'].shape) != (1, 3, 33, 192, 256):
        raise RuntimeError('Stick video shape differs from training')
    if tuple(sample['action'].shape) != (1, 33, 14):
        raise RuntimeError('Stick target EEF action shape differs from training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    own = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU in a compute allocation is required')
    protocol = json.loads(own.config.read_text())
    method = protocol['method']
    if method not in ('dino', 'ttt'):
        raise ValueError(method)
    training, prepared, output = [ROOT / protocol[k] for k in ('training_run', 'prepared', 'inference_output')]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint, step = select_checkpoint(training, output)
    if not int(protocol['minimum_checkpoint_step']) <= step <= 5500:
        raise RuntimeError(f'Unexpected final checkpoint step: {step}')
    pin = output / 'checkpoint_snapshot' / checkpoint.name
    pin.parent.mkdir(parents=True, exist_ok=True)
    if not pin.exists():
        os.link(checkpoint, pin)
    elif not os.path.samefile(checkpoint, pin):
        raise RuntimeError('Pinned checkpoint changed')
    frozen = output / 'input_manifest'
    for name in ('plan.json', 'query.jsonl', 'support.jsonl', 'stage_files.txt'):
        snapshot(prepared / name, frozen / name)
    snapshot(own.config, frozen / 'evaluation_config.json')
    snapshot(training / 'submitted_config.yaml', frozen / 'training_config.yaml')
    snapshot(training / 'input_manifest/action_stats.json', frozen / 'action_stats.json')
    snapshot(training / 'input_manifest/train.jsonl', frozen / 'training_rows.jsonl')
    if (frozen / 'action_stats.json').read_bytes() != (prepared / 'action_stats.json').read_bytes():
        raise RuntimeError('Action normalization differs from the reference comparison')
    plan = json.loads((frozen / 'plan.json').read_text())
    queries, supports, train = [rows(frozen / name) for name in ('query.jsonl', 'support.jsonl', 'training_rows.jsonl')]
    if len(queries) != 63 or sum(r['dataset_split'] == 'test' for r in queries) != 45 or len(plan['environments']) != 9:
        raise RuntimeError('Frozen Stick cohort changed')
    train_episodes = {episode(r) for r in train}
    train_chunks = {(*episode(r), int(r['start_frame'])) for r in train}
    for row in queries:
        if (row['dataset_split'] == 'train') != (episode(row) in train_episodes):
            raise RuntimeError('Query split differs from actual training membership')
    for env in plan['environments']:
        selected = [supports[i] for i in env['support_indices']]
        if len(selected) != 2 or len({episode(r) for r in selected}) != 2:
            raise RuntimeError('Exactly two distinct train Support episodes are required')
        for row in selected:
            if (row['environment'] != env['environment'] or row['dataset_split'] != 'train'
                    or (*episode(row), int(row['start_frame'])) not in train_chunks):
                raise RuntimeError('Support is not a matching eligible training chunk')
        if {episode(r) for r in selected} & {episode(queries[i]) for i in env['query_indices']}:
            raise RuntimeError('Support/query episode leakage')
    cfg = yaml.safe_load((frozen / 'training_config.yaml').read_text())['training']
    contract = dict(num_frames=33, frame_stride=3, num_history_frames=1, height=192,
                    width=256, action_dim=14, action_type='eef_target', resize_mode='letterbox')
    if any(cfg[k] != v for k, v in contract.items()):
        raise RuntimeError('Training/inference contract mismatch')
    if method == 'dino' and (cfg['dinov2_aggregation_mode'] != 'concat_mlp' or str(cfg['dinov2_support_k_choices']) != '1,2'):
        raise RuntimeError('Expected K=1/2 concat-MLP training')
    if method == 'ttt' and cfg['ttt_protocol'] != 'oneminute_write_then_predict':
        raise RuntimeError('Expected write-then-predict TTT training')
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    wan = cache / 'Wan2.2-TI2V-5B'
    copy_cached(ROOT / 'models/Wan2.2-TI2V-5B', wan, directory=True)
    tag = hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
    local_data = cache / 'eval_datasets' / ('stick916_' + tag)
    copy_cached(Path(plan['source_dataset']), local_data, directory=True, files=frozen / 'stage_files.txt')
    stat = checkpoint.stat()
    tag = hashlib.sha256(f'{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:16]
    local_ckpt = cache / 'stick916_baseline_eval' / tag / 'model.safetensors'
    copy_cached(pin, local_ckpt)
    if method == 'dino':
        dino_local = cache / 'dinov2-base'
        copy_cached(Path(cfg['dinov2_model_path']), dino_local, directory=True)
        cfg.update(dinov2_model_path=str(dino_local), dinov2_support_k=2)
        wan_checkpoint = local_ckpt.with_name('wan.safetensors')
        with local_ckpt.with_name('wan.lock').open('a') as ckpt_lock:
            fcntl.flock(ckpt_lock, fcntl.LOCK_EX)
            materialize_wan_checkpoint(str(local_ckpt), str(wan_checkpoint))
    else:
        wan_checkpoint = cache / 'ckpt/BLM/step-12000.safetensors'
        copy_cached(ROOT / 'ckpt/BLM/step-12000.safetensors', wan_checkpoint)
    cfg.update(dataset_base_path=str(local_data), dataset_metadata_path=str(frozen / 'query.jsonl'),
               action_stat_path=str(frozen / 'action_stats.json'), model_paths=str(wan), ckpt_path=str(wan_checkpoint),
               output_path=str(output / 'raw' / method), num_inference_steps=50, cfg_scale=1.0,
               seed=protocol['seed'], fps=protocol['fps'], quality=8, max_samples=0,
               video_light_augmentation_enabled=False, dinov2_support_light_augmentation_enabled=False,
               spatial_loss_mode='none', use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference': cfg}, sort_keys=False))
    add_method = add_dinov2_config if method == 'dino' else add_ttt_kvb_config
    inference_parser = add_method(add_general_config(argparse.ArgumentParser()))
    if '--frame_stride' not in inference_parser._option_string_actions:
        inference_parser.add_argument('--frame_stride', type=int, default=3)
    args = inference_parser.parse_args(['--config', str(runtime)])
    args = merge_yaml_and_args(str(runtime), inference_parser, args)
    args.dinov2_checkpoint_path = str(local_ckpt)
    args.ttt_checkpoint_path = str(local_ckpt)
    dataset = build_infer_dataset(args)
    support_args = copy.copy(args)
    support_args.dataset_metadata_path = str(frozen / 'support.jsonl')
    support_dataset = build_infer_dataset(support_args)
    set_global_seed(protocol['seed'])
    contexts = {}
    if method == 'dino':
        pipe = build_pipeline(args)
        encoder = load_support_encoder(args, pipe.device)
        with torch.no_grad():
            for env in plan['environments']:
                data = [support_dataset[i] for i in env['support_indices']]
                for sample in data:
                    check_sample(sample)
                features = [encoder.extract_visual_features(sample['video']) for sample in data]
                code = encoder.project_supports(visual_features=tuple(f[0] for f in features),
                    actions=tuple(sample['action'] for sample in data), frame_indices=tuple(f[1] for f in features),
                    frame_counts=tuple(f[2] for f in features))[0].detach().float().cpu()
                contexts[env['environment']] = code
                json_write(output / 'contexts' / (env['environment'] + '.json'),
                           dict(context=code.tolist(), support_indices=env['support_indices'], support_k=2,
                                aggregation='training_concat_mlp', query_future_used=False))
                del data, features
        del encoder
        torch.cuda.empty_cache()
    else:
        model, installation = build_training_faithful_model(args)
        model.requires_grad_(False)
        pipe, controller = model.pipe, installation.controller
        original = pipe.model_fn
    json_write(output / 'provenance.json', dict(protocol=protocol, model_step=step, checkpoint=str(checkpoint),
        query_sha256=hashlib.sha256((frozen / 'query.jsonl').read_bytes()).hexdigest(),
        support_sha256=hashlib.sha256((frozen / 'support.jsonl').read_bytes()).hexdigest(),
        support_k=2, query_future_gt_used=False, action_loader='unchanged training-compatible target EEF loader',
        video_action_stride=3, observed_frames=1, prediction_frames=32, inference_light_augmentation=False))
    completed = []
    for env in plan['environments']:
        name = env['environment']
        if method == 'ttt':
            replays = []
            for position, index in enumerate(env['support_indices']):
                sample = support_dataset[index]
                check_sample(sample)
                inputs, latents, noise = _prepare_support_model_inputs(model, sample,
                    noise_seed=protocol['seed'] + 100000 + int(env['environment_index']) * 1009 + position)
                replays.append(SameTimestepSupportQueryModelFn(pipe=pipe, controller=controller,
                    original_model_fn=original, support_model_inputs=inputs, support_input_latents=latents, support_noise=noise))
            json_write(output / 'support_memory' / (name + '.json'), dict(support_indices=env['support_indices'],
                order='support1, support2, query', reset_per_denoising_timestep=True, independent_query_branches=True))
        for index in env['query_indices']:
            row = queries[index]
            key = query_key(index, row)
            paths = {kind: output / 'raw' / kind / (key + '.mp4') for kind in ('gt', method)}
            comparison = output / 'comparisons' / (key + '.mp4')
            marker = output / 'completed' / (key + '.json')
            if not (marker.is_file() and comparison.is_file() and all(p.is_file() for p in paths.values())):
                sample = dataset[index]
                check_sample(sample)
                gt = frame_array(sample['video'])
                args.seed = int(protocol['seed']) + index
                set_global_seed(args.seed)
                rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                rollout['output_path'] = str(paths[method])
                paths[method].parent.mkdir(parents=True, exist_ok=True)
                if method == 'dino':
                    rollout['physical_context'] = contexts[name].to(device=pipe.device, dtype=pipe.torch_dtype)
                else:
                    replay = MultiSupportReplay(pipe, controller, original, replays)
                    pipe.model_fn = replay
                try:
                    with torch.no_grad():
                        _run_autoregressive(pipe, rollout, args)
                finally:
                    if method == 'ttt':
                        pipe.model_fn = original
                        controller.clear()
                predicted = read_video(paths[method])
                if predicted.shape != gt.shape:
                    raise RuntimeError('Prediction/GT geometry mismatch: ' + key)
                video_write(paths['gt'], gt, protocol['fps'])
                video_write(paths[method], predicted, protocol['fps'])
                video_write(comparison, [np.vstack([labeled(gt[f], f'GT | {row["dataset_split"]}'),
                    labeled(predicted[f], f'{method} | K=2')]) for f in range(33)], protocol['fps'])
                if method == 'ttt':
                    json_write(output / 'memory_trace' / (key + '.json'), dict(trace=replay.trace, support_k=2))
                json_write(marker, dict(index=index, row=row, paths={k: str(v) for k, v in paths.items()},
                    model_step=step, seed=args.seed, support_indices=env['support_indices']))
                del sample, rollout, gt, predicted
            completed.append(key)
            json_write(output / 'progress.json', dict(completed_queries=len(completed), expected_queries=len(queries)))
            print(f'[query_done] {key} {len(completed)}/{len(queries)}', flush=True)
        for split in ('train', 'test'):
            indices = [i for i in env['query_indices'] if queries[i]['dataset_split'] == split]
            columns = [read_video(output / 'comparisons' / (query_key(i, queries[i]) + '.mp4')) for i in indices]
            if columns:
                video_write(output / 'grids' / split / (name + '.mp4'), np.concatenate(columns, axis=2), protocol['fps'])
            del columns
        if method == 'ttt':
            del replays
            controller.clear()
        torch.cuda.empty_cache()
    json_write(output / 'inference_complete.json', dict(method=method, model_step=step, queries=len(completed),
        test_queries=45, train_queries=18, environments=9, support_k=2, metric_status='not_scored'))


if __name__ == '__main__':
    main()
