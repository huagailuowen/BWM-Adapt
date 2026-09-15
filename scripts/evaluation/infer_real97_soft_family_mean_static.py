#!/usr/bin/env python3
"""Known-family mean initialization ablation; leaves legacy inference untouched."""
from __future__ import annotations
import argparse
import colorsys
import copy
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import infer_real97_soft_balanced9_dual_static1 as baseline
from infer_real97_soft_balanced9_dual_static1 import (
    ROOT, require_compute, write_json, frozen_copy, resolve, write_video, labeled, rgb_video,
)


def prepare(config, output):
    source = resolve(config['source_inference'])
    if output.resolve() == source.resolve():
        raise RuntimeError('Family initialization must have a separate output directory')
    if not (source / 'inference_complete.json').is_file():
        raise RuntimeError('Reference inference must be complete before freezing its queries')
    for relative in (
        'query.jsonl', 'support.jsonl', 'selection_plan.json', 'selection_complete.json',
        'input_manifest/resolved_checkpoint.json',
    ):
        frozen_copy(source / relative, output / relative)
    plan, runtime, checkpoint, queries, supports = baseline.prepare(config, output)
    table = json.loads((output / 'input_manifest/context_table.json').read_text())
    if table != json.loads((source / 'input_manifest/context_table.json').read_text()):
        raise RuntimeError('Reference and ablation must use the identical context table')
    groups = config['initialization_groups']
    members = [name for names in groups.values() for name in names]
    if len(members) != len(set(members)) or set(members) != set(config['environments']):
        raise RuntimeError('Families must partition all physical environments exactly once')
    entries, membership = {}, {}
    for group, names in groups.items():
        rows = [row for row in table['records'] if row['environment'] in names]
        for name in names:
            replicas = [row['replica'] for row in rows if row['environment'] == name]
            if sorted(replicas) != [0, 1]:
                raise RuntimeError(f'Missing or duplicate replicas for {name}')
            membership[name] = group
        values = np.asarray([row['context'] for row in rows], dtype=np.float32).reshape(len(rows), 32)
        entries[group] = {
            'environments': names, 'row_count': len(rows),
            'source_virtual_group_ids': [row['source_virtual_group_id'] for row in rows],
            'context': values.mean(axis=0).reshape(1, 32).tolist(),
        }
    record = {
        'policy': config['stage2_initial'], 'known_family_prior': True,
        'assignment_policy': config['group_assignment_policy'],
        'groups': entries, 'environment_to_group': membership,
        'model_step': config['model_step'], 'table_step': config['table_step'],
        'query_GT_used_for_initialization': False, 'random_initialization_noise': None,
    }
    destination = output / 'group_initialization.json'
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise RuntimeError('Cannot resume with a changed family mean')
    else:
        write_json(destination, record)
    # Stage1 is identical. Never hardlink GT: it is rewritten for new Stage2 queries.
    for index, row in enumerate(queries):
        key = f"q{index:04d}_{row['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"
        src = source / 'raw/stage1' / (key + '.mp4')
        marker = source / 'completed_variants' / (key + '_stage1.json')
        if not src.is_file() or not src.stat().st_size or not marker.is_file():
            raise RuntimeError(f'Missing complete Stage1 reference: {key}')
        dst = output / 'raw/stage1' / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            temporary = dst.with_suffix('.mp4.partial')
            temporary.unlink(missing_ok=True)
            try:
                os.link(src, temporary)
            except OSError:
                shutil.copy2(src, temporary)
            os.replace(temporary, dst)
        if dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(f'Incomplete Stage1 reference copy: {key}')
        frozen_copy(marker, output / 'completed_variants' / marker.name)
    plan.update(stage2_initial=config['stage2_initial'], initialization_groups=groups,
                known_family_prior=True, source_inference=str(source), stage1_reused=True)
    write_json(output / 'plan.json', plan)
    print('[family_initialization]', json.dumps(
        {key: {'environments': item['environments'], 'row_count': item['row_count']}
         for key, item in entries.items()}), flush=True)
    return plan, runtime, checkpoint, queries, supports


def pca_plot(path, table, contexts, environments):
    rows = table['records']
    values = np.asarray([row['context'] for row in rows], dtype=float).reshape(len(rows), -1)
    center = values.mean(0)
    _, singular, axes = np.linalg.svd(values - center, full_matrices=False)
    training = (values - center) @ axes[:2].T
    endpoints = {
        env['environment']: (np.asarray(contexts[env['environment']]['context']).reshape(-1) - center) @ axes[:2].T
        for env in environments
    }
    starts = {
        state['initialization_group']: (np.asarray(state['initial']).reshape(-1) - center) @ axes[:2].T
        for state in contexts.values()
    }
    points = np.vstack([training, *[p.reshape(1, 2) for p in endpoints.values()],
                        *[p.reshape(1, 2) for p in starts.values()]])
    low = points.min(0)
    span = np.maximum(points.max(0) - low, 1e-6)
    low -= .06 * span
    span *= 1.12
    def xy(point):
        return 60 + 690 * (point[0] - low[0]) / span[0], 535 - 450 * (point[1] - low[1]) / span[1]
    names = [row['environment'] for row in rows if row['stage1_selected']]
    colors = {name: '#' + ''.join(f'{int(v*255):02x}' for v in colorsys.hsv_to_rgb(i/len(names), .7, .8))
              for i, name in enumerate(names)}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650"><rect width="1120" height="650" fill="white"/>',
        '<text x="40" y="30" font-family="sans-serif" font-size="18">Soft: family-mean initialization (1R in R)</text>',
    ]
    for row, point in zip(rows, training):
        x, y = xy(point)
        radius = 6 if row['stage1_selected'] else 4
        title = html.escape(f"{row['environment']} replica {row['replica']}")
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{colors[row["environment"]]}" stroke="black"><title>{title}</title></circle>')
    for name, point in endpoints.items():
        x, y = xy(point)
        parts.append(f'<polygon points="{x},{y-7} {x-6},{y+5} {x+6},{y+5}" fill="{colors[name]}" stroke="black"><title>{html.escape(name)} inference time</title></polygon>')
    for group, point in starts.items():
        x, y = xy(point)
        parts.append(f'<polygon points="{x},{y-9} {x+9},{y} {x},{y+9} {x-9},{y}" fill="none" stroke="black" stroke-width="2"><title>{html.escape(group)} family initial mean</title></polygon>')
    for i, name in enumerate(names):
        parts.append(f'<text x="785" y="{65+i*27}" fill="{colors[name]}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
    total = max(float(np.sum(singular**2)), 1e-12)
    parts.append(f'<text x="55" y="587" font-family="sans-serif" font-size="13">PC1 {100*singular[0]**2/total:.1f}%; PC2 {100*singular[1]**2/total:.1f}%; same 18-row training PCA plane.</text>')
    parts.append('<text x="55" y="609" font-family="sans-serif" font-size="12">Circles: training time. Black-bordered triangles: inference time. Hollow diamonds: family initial means.</text>')
    parts.append('<text x="55" y="630" font-family="sans-serif" font-size="12">Known-family prior ablation; no query ground truth used for adaptation.</text></svg>')
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
    initialization = json.loads((output / 'group_initialization.json').read_text())
    initials = {
        group: torch.as_tensor(item['context'], device=pipe.device, dtype=torch.float32).reshape(1, 32)
        for group, item in initialization['groups'].items()
    }
    contexts, results = {}, []
    for env in plan['environments']:
        name = env['environment']
        initial_group = initialization['environment_to_group'][name]
        initial = initials[initial_group].clone()
        context_path = output / 'contexts' / (name + '.json')
        if context_path.exists():
            state = json.loads(context_path.read_text())
            if (state['model_step'], state['table_step'], state['observed_frames']) != (config['model_step'], config['table_step'], 1):
                raise RuntimeError('Cannot reuse a different model/table/history context')
            if state.get('initialization_group') != initial_group or not np.array_equal(
                    np.asarray(state['initial'], dtype=np.float32), initial.cpu().numpy()):
                raise RuntimeError('Cannot reuse a context optimized from a different family initialization')
            adapted = torch.as_tensor(state['context'], device=pipe.device, dtype=torch.float32)
        else:
            support = [support_dataset[index] for index in env['support_indices']]
            set_global_seed(int(config['seed']) + int(env['environment_index']) * 1009)
            adapted, losses, _, metrics, trajectory = ttt._adapt_ttt_state(
                pipe, support, args, adapter_named_params=[], base_adapter_state={}, initial_context=initial,
                target_context=None, trajectory_meta={'environment': name, 'sample_index': env['query_indices'][0],
                                                       'friction_mu': float(env['environment_index'])})
            state = {'context': adapted.detach().cpu().tolist(), 'initial': initial.cpu().tolist(),
                     'initialization_group': initial_group,
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
        'method': 'ours_static_single_frame_balanced9_dual_family_mean', 'stage1_replica': 0,
        'stage2_initialization': config['stage2_initial'], 'known_family_prior': True, 'context_hard_bounds': None, 'model_step': config['model_step'], 'table_step': config['table_step'],
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
