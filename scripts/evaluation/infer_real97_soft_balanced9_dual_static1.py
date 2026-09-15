#!/usr/bin/env python3
"""Opt-in balanced-nine dual-latent Soft static inference after training completes."""
from __future__ import annotations
import argparse
import colorsys
import copy
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml
from real97_trial_common import ROOT, SilverDetector, native_frame, read_jsonl, require_compute, write_json, write_jsonl
from infer_real97_reference_trial import labeled, write_video
from infer_real97_soft_ours_static1 import episode_start, freeze_rows, protect_checkpoint
from infer_real97_soft_ours_history5_static import cached_copy, frozen_copy, resolve, rgb_video


def paired_config(config, output):
    run = resolve(config['training_run'])
    record = output / 'input_manifest/resolved_checkpoint.json'
    if record.exists():
        resolved = json.loads(record.read_text())
    else:
        steps = []
        for path in run.glob('step-*.safetensors'):
            step = int(path.name.split('-')[1].split('.')[0])
            if (step >= config['minimum_checkpoint_step']
                    and (run / f'.step-{step}.safetensors.complete').is_file()
                    and (run / f'step-{step}.context_table.json').is_file()):
                steps.append(step)
        if not steps:
            raise RuntimeError('No recent fully published model/C-table pair')
        step = max(steps)
        resolved = dict(model_step=step, table_step=step,
                        snapshot=str(resolve(config['snapshot_root']) / f'step{step}'),
                        preferred_final_step=config['preferred_final_step'],
                        preferred_final_step_available=(step == config['preferred_final_step']))
        write_json(record, resolved)
    config.update(resolved)
    print('[checkpoint_pair]', json.dumps(resolved), flush=True)


def static_window(row, role, start=0):
    result = episode_start(row, role)
    last = int(row['total_frames']) - 1
    indices = [min(start + 3 * i, last) for i in range(33)]
    result.update(start_frame=start, end_frame=indices[-1], native_frame_indices=indices,
                  history_anchor_frame=start, observed_native_frame_indices=[start],
                  prediction_native_frame_indices=indices[1:],
                  evaluation_frame_indices=[i for i in range(1, 33) if start+3*i <= last],
                  comparison_evaluation_frame_indices=[i for i in range(1, 33) if start+3*i <= last],
                  prediction_padding_frames=sum(start+3*i > last for i in range(1, 33)),
                  sample_id=f"{row['environment']}/ep{row['episode_index']:06d}/{role}_start{start:04d}",
                  episode_seen_in_training_manifest=(row['dataset_split'] == 'train'))
    if role == 'support':
        result.update(sampling_kind='evaluation_annotated_stationary_support',
                      initial_stationarity='start appears in frozen training legal-start manifest')
    return result


def select_rows(config, output, local_data, physical, tests):
    if (output / 'selection_complete.json').exists():
        return (json.loads((output / 'selection_plan.json').read_text()),
                read_jsonl(output / 'query.jsonl'), read_jsonl(output / 'support.jsonl'))
    candidates, test_episodes = {}, {}
    for row in physical:
        key = (row['environment'], int(row['episode_index']))
        candidates.setdefault(key, []).append(row)
    for row in tests:
        test_episodes.setdefault((row['environment'], int(row['episode_index'])), row)
    if set(candidates) & set(test_episodes):
        raise RuntimeError('Training and test episodes overlap')
    excluded = {(env, int(ep)) for env, episodes in config['quality_exclusions'].items() for ep in episodes}
    if excluded & (set(candidates) | set(test_episodes)):
        raise RuntimeError('Disabled episode leaked into input manifests')
    historical_query = read_jsonl(resolve(config['historical_queries']))
    historical_support = read_jsonl(resolve(config['historical_supports']))
    detector = None
    measurement_cache = {}

    def measure(row, frame):
        nonlocal detector
        env, ep = row['environment'], int(row['episode_index'])
        key = (env, ep)
        if key not in measurement_cache:
            path = resolve(config['tracking_cache']) / f'{env}_ep{ep:06d}' / 'tracks.jsonl'
            measurement_cache[key] = {int(item['frame']): item for item in read_jsonl(path)} if path.is_file() else {}
        item = measurement_cache[key].get(frame)
        if item is None or not item.get('measurement_valid') or item.get('center') is None:
            if detector is None:
                detector = SilverDetector()
            item = detector(native_frame(local_data / row['video'][0], frame))
            item.update(frame=frame, measurement_source='fresh_cpu_detector')
            measurement_cache[key][frame] = item
        return item

    queries, supports, environments = [], [], []
    for env_index, name in enumerate(config['environments']):
        keys = [key for key in candidates if key[0] == name]
        keys.sort(key=lambda key: hashlib.sha256(f"{config['seed']}:{key[0]}:{key[1]}".encode()).hexdigest())
        old_queries = [(name, int(row['episode_index'])) for row in historical_query
                       if row['environment'] == name and row['dataset_split'] == 'train']
        reserved = []
        for key in old_queries + keys:
            if key in candidates and key not in reserved:
                reserved.append(key)
            if len(reserved) == config['query_train_episodes_per_environment']:
                break
        if len(reserved) != 2:
            raise RuntimeError(f'Insufficient disjoint training queries: {name}')
        old_keys = [(name, int(row['episode_index'])) for row in historical_support if row['environment'] == name]
        support_keys = [key for key in dict.fromkeys(old_keys + keys) if key in candidates and key not in reserved]
        buckets, used = {'left': [], 'right': []}, set()
        for phase in ('episode_start', 'other_legal_starts'):
            for key in support_keys:
                if key in used:
                    continue
                options = sorted(candidates[key], key=lambda row: row['start_frame'])
                if phase == 'episode_start':
                    options = [row for row in options if int(row['start_frame']) == 0][:1]
                else:
                    options = [options[i] for i in sorted({0, len(options)//2, len(options)-1})]
                for row in options:
                    candidate = static_window(row, 'support', int(row['start_frame']))
                    frames = sorted({candidate['native_frame_indices'][i] for i in (0, 8, 16, 24, 32)})
                    points = [measure(candidate, frame) for frame in frames]
                    write_json(output / 'support_measurements' /
                               f"{name}_ep{key[1]:06d}_start{candidate['start_frame']:04d}.json",
                               {'row': candidate, 'points': points})
                    if len(points) < 2 or not all(p.get('measurement_valid') and p.get('center') is not None for p in points):
                        continue
                    x = np.asarray([p['center'][0] for p in points], dtype=float)
                    dx = float(x[-1] - x[0])
                    direction = 'right' if dx > 0 else 'left'
                    excursion = float(np.max(x-x[0]) if dx > 0 else np.max(x[0]-x))
                    if abs(dx) < config['support_minimum_net_dx_px'] or excursion < config['support_minimum_excursion_px']:
                        continue
                    candidate.update(support_direction=direction, support_dx_px=dx,
                                     support_excursion_px=excursion, support_measurement_frames=frames)
                    buckets[direction].append(candidate)
                    used.add(key)
                    break
                if all(len(bucket) >= 2 for bucket in buckets.values()):
                    break
            if all(len(bucket) >= 2 for bucket in buckets.values()):
                break
        if not all(buckets.values()):
            raise RuntimeError(f'{name}: no valid two-direction support set; do not drop this environment')
        chosen = buckets['left'][:2] + buckets['right'][:2]
        chosen_ids = {row['episode_index'] for row in chosen}
        chosen += [row for bucket in buckets.values() for row in bucket
                   if row['episode_index'] not in chosen_ids][:4-len(chosen)]
        if len(chosen) != 4:
            raise RuntimeError(f'{name}: fewer than four informative disjoint support episodes')
        si = list(range(len(supports), len(supports)+4))
        supports.extend(chosen)
        selected_test = sorted([key for key in test_episodes if key[0] == name], key=lambda key: key[1])
        if not selected_test:
            raise RuntimeError(f'No held-out episodes for {name}')
        qi = []
        for key in reserved + selected_test:
            base = candidates[key][0] if key in candidates else test_episodes[key]
            query = static_window(base, 'query')
            query.update(environment_index=env_index, friction_mu=float(env_index))
            qi.append(len(queries))
            queries.append(query)
        for row in chosen:
            row.update(environment_index=env_index, friction_mu=float(env_index))
        environments.append(dict(environment=name, environment_index=env_index,
                                 support_indices=si, query_indices=qi,
                                 direction_counts={d: sum(row['support_direction'] == d for row in chosen) for d in buckets}))
        print('[selection]', name, 'supports', [(r['episode_index'], r['start_frame'], r['support_direction']) for r in chosen],
              'queries', [(queries[i]['episode_index'], queries[i]['dataset_split']) for i in qi], flush=True)
    if len(queries) != config['expected_query_count'] or len(environments) != len(config['environments']):
        raise RuntimeError('Expected all nine environments and all eighteen held-out episodes')
    plan = dict(task='soft', environments=environments, unavailable_environments=[],
                expected_environments=len(environments), query_count=len(queries), support_count=len(supports),
                no_query_GT_in_adaptation=True, stage1_reference='corresponding_z0',
                stage2_initial='mean_of_all_18_training_latents',
                query_policy='two_train_and_all_test_episodes_at_start_zero',
                support_policy='four_train_episodes_static_legal_starts_with_both_motion_directions',
                source_outputs_modified=False)
    freeze_rows(output / 'query.jsonl', queries)
    freeze_rows(output / 'support.jsonl', supports)
    write_json(output / 'selection_plan.json', plan)
    write_json(output / 'selection_complete.json', {'query_count': len(queries), 'support_count': len(supports)})
    return plan, queries, supports


def prepare(config, output):
    paired_config(config, output)
    snapshot = protect_checkpoint(config)
    frozen = output / 'input_manifest'
    run = resolve(config['training_run'])
    for name in ('training_config.yaml', 'action_stats.json'):
        frozen_copy(snapshot / name, frozen / name)
    frozen_copy(snapshot / 'context_table.json', frozen / 'full_training_context_table.json')
    for name in ('physical_train.jsonl', 'test.jsonl', 'latent_aliases.json', 'episode_split.jsonl', 'data_contract.json'):
        frozen_copy(run / 'input_manifest' / name, frozen / name)
    physical, tests = read_jsonl(frozen / 'physical_train.jsonl'), read_jsonl(frozen / 'test.jsonl')
    aliases = json.loads((frozen / 'latent_aliases.json').read_text())['records']
    original_table = json.loads((frozen / 'full_training_context_table.json').read_text())
    original = {int(row['friction_mu']): row for row in original_table['records']}
    if len(original) != 18:
        raise RuntimeError('This evaluation requires the complete 18-row dual table')
    remapped = []
    for alias in aliases:
        name = alias['physical_environment']
        physical_id = config['environments'].index(name)
        replica = int(alias['latent_replica'])
        row = copy.deepcopy(original[int(alias['virtual_group_id'])])
        row.update(friction_mu=physical_id + (9 if replica else 0), environment=name,
                   physical_environment_index=physical_id, replica=replica, stage1_selected=(replica == 0),
                   source_virtual_group_id=int(alias['virtual_group_id']))
        remapped.append(row)
    if len(remapped) != 18 or len({row['friction_mu'] for row in remapped}) != 18:
        raise RuntimeError('Duplicate or missing context-table aliases')
    table = {**original_table, 'records': sorted(remapped, key=lambda row: row['friction_mu'])}
    if (frozen / 'context_table.json').exists():
        if json.loads((frozen / 'context_table.json').read_text()) != table:
            raise RuntimeError('Frozen paired context table changed')
    else:
        write_json(frozen / 'context_table.json', table)
    source_dataset = resolve(config['source_dataset'])
    files = sorted({path for row in physical+tests for path in [*row['video'], row['action']]})
    if any(Path(path).is_absolute() or '..' in Path(path).parts for path in files):
        raise RuntimeError('Unsafe dataset-relative path')
    filelist = frozen / 'stage_files.txt'
    filelist.write_text('\n'.join(files) + '\n')
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    identity = str(source_dataset.resolve()) + '\n' + '\n'.join(files)
    local_data = cache / 'eval_datasets' / ('soft_dual9_' + hashlib.sha256(identity.encode()).hexdigest()[:16])
    cached_copy(source_dataset, local_data, directory=True, files=filelist)
    plan, queries, supports = select_rows(config, output, local_data, physical, tests)
    training = yaml.safe_load((frozen / 'training_config.yaml').read_text())
    flat = {key: value for section in training.values() if isinstance(section, dict) for key, value in section.items()}
    wan = cache / 'Wan2.2-TI2V-5B'
    cached_copy(ROOT / 'models/Wan2.2-TI2V-5B', wan, directory=True)
    source = snapshot / 'model.safetensors'
    identity = f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}'
    checkpoint = cache / 'eval_checkpoints' / ('soft_dual9_' + hashlib.sha256(identity.encode()).hexdigest()[:20] + '.safetensors')
    cached_copy(source, checkpoint)
    flat.update(dataset_base_path=str(local_data), dataset_metadata_path=str(output / 'query.jsonl'),
                action_stat_path=str(frozen / 'action_stats.json'), ckpt_path=str(checkpoint),
                model_paths=str(wan), output_path=str(output / 'raw/stage2'), seed=config['seed'],
                max_samples=0, fps=7, quality=8, num_frames=33, num_history_frames=1, frame_stride=3,
                action_type='eef_target', action_dim=14, height=160, width=320,
                video_light_augmentation_enabled=False, spatial_loss_mode='none',
                use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
                stage2_group_keys='environment', stage2_inner_lr_schedule=config['stage2_inner_lr_schedule'],
                stage2_inner_steps=config['stage2_inner_steps'], stage2_inner_grad_clip=0.0,
                stage2_context_reg_weight=0.0, stage2_context_clamp_min=-float('inf'),
                stage2_context_clamp_max=float('inf'), ttt_context_fp32=True,
                ttt_support_gradient_accumulation=True, ttt_adapt_scope='context',
                num_inference_steps=config['num_inference_steps'], cfg_scale=1.0)
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference': flat}, sort_keys=False))
    plan.update(source_dataset=str(source_dataset), model_step=config['model_step'],
                table_step=config['table_step'], observed_frames=1, predicted_frames=32)
    write_json(output / 'plan.json', plan)
    write_json(output / 'provenance.json', {
        **config, 'protected_snapshot': str(snapshot), 'local_checkpoint': str(checkpoint),
        'training_latent_count': 18, 'stage1_replica': 0, 'context_hard_bounds': None,
        'action': 'training CanonicalTargetEEF: eight normalized coordinates plus six zeros',
        'support_loss': 'single-frame flow matching, no ROI, no context regularization',
        'stage2_trainable': 'Z only', 'query_GT_used_for_adaptation': False,
        'original_outputs_modified': False, 'formal_metric_approved': False})
    return plan, runtime, checkpoint, queries, supports


def pca_plot(path, table, contexts, environments):
    records = table['records']
    values = np.asarray([row['context'] for row in records], dtype=float).reshape(len(records), -1)
    center = values.mean(0)
    _, singular, axes = np.linalg.svd(values-center, full_matrices=False)
    training = (values-center) @ axes[:2].T
    endpoints = {env['environment']: (np.asarray(contexts[env['environment']]['context']).reshape(-1)-center) @ axes[:2].T
                 for env in environments}
    points = np.vstack([training, np.zeros((1, 2))] + [xy.reshape(1, 2) for xy in endpoints.values()])
    low = points.min(0)
    span = np.maximum(points.max(0)-low, 1e-6)
    low -= .06*span
    span *= 1.12
    def xy(point):
        return 60+690*(point[0]-low[0])/span[0], 535-450*(point[1]-low[1])/span[1]
    names = [row['environment'] for row in records if row['stage1_selected']]
    colors = {name: '#' + ''.join(f'{int(v*255):02x}' for v in colorsys.hsv_to_rgb(i/len(names), .7, .8))
              for i, name in enumerate(names)}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="630"><rect width="1120" height="630" fill="white"/>',
             '<text x="40" y="30" font-family="sans-serif" font-size="18">Soft: training-time dual Z and inference-time Z</text>']
    for row, point in zip(records, training):
        x, y = xy(point)
        radius = 6 if row['stage1_selected'] else 4
        title = html.escape(f"{row['environment']} z{row['replica']}")
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{colors[row["environment"]]}" stroke="black"><title>{title}</title></circle>')
    for name, point in endpoints.items():
        x, y = xy(point)
        parts.append(f'<polygon points="{x},{y-7} {x-6},{y+5} {x+6},{y+5}" fill="{colors[name]}" stroke="black"><title>{html.escape(name)} inference time</title></polygon>')
    x, y = xy(np.zeros(2))
    parts.append(f'<path d="M{x-5},{y}H{x+5} M{x},{y-5}V{y+5}" stroke="black" stroke-width="2"><title>Initial Z: mean of all 18 training latents</title></path>')
    for i, name in enumerate(names):
        parts.append(f'<text x="785" y="{65+i*27}" fill="{colors[name]}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
    total = max(float(np.sum(singular**2)), 1e-12)
    parts.append(f'<text x="55" y="587" font-family="sans-serif" font-size="13">PC1 {100*singular[0]**2/total:.1f}%; PC2 {100*singular[1]**2/total:.1f}%; fitted on all 18 training latents.</text>')
    parts.append('<text x="55" y="609" font-family="sans-serif" font-size="12">Circles: training time (large = z0). Black-bordered triangles: inference time. Black cross: shared initialization.</text></svg>')
    path.write_text('\n'.join(parts))


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
    expected = {float(env['environment_index']) for env in plan['environments']}
    if not expected.issubset(lookup) or len(lookup) != 2 * len(expected):
        raise RuntimeError('Dual table must retain both replicas for every physical environment')
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
        'method': 'ours_static_single_frame_balanced9_dual', 'stage1_replica': 0,
        'stage2_initialization': 'mean_of_all_18_latents', 'context_hard_bounds': None, 'model_step': config['model_step'], 'table_step': config['table_step'],
        'queries': len(results), 'environments': len(contexts), 'observed_frames': 1, 'prediction_frames': 32,
        'raw_frames': 33, 'metric_status': 'CPU_scoring_pending', 'formal_metric_approved': False,
    })
    for split in ('train', 'test'):
        for env in plan['environments']:
            indices = [i for i in env['query_indices'] if queries[i]['dataset_split'] == split]
            if not indices:
                continue
            videos = []
            for index in indices:
                row = queries[index]
                key = f"q{index:04d}_{env['environment']}_{split}_ep{row['episode_index']:06d}"
                videos.append(imageio.mimread(str(output / 'comparisons' / (key + '.mp4'))))
            destination = output / 'grids' / split / (env['environment'] + '_gt_stage1_stage2.mp4')
            write_video(destination, (np.hstack([video[f] for video in videos]) for f in range(33)), config['fps'])
    ttt._write_context_pca_plot(output / 'initialization_context_trajectory.svg',
        ttt._load_grouped_context_table(output / 'input_manifest/context_table.json'),
        [point for state in contexts.values() for point in state['trajectory']])
    print('[inference_complete] ' + str(output), flush=True)


if __name__ == '__main__':
    main()
