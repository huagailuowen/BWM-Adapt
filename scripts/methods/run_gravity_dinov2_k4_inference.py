"""Independent K=4 Gravity evaluation; preserve the existing K=1 benchmark."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from scripts.methods import run_dinov2_standard_sim_inference as shared


def manifest_k4(config, train, metric, output):
    reference_config = dict(config, expected_queries=config['reference_expected_queries'])
    _, reference = shared.prepare_manifest(reference_config, train, metric, output / 'reference_k1_protocol')
    metadata = shared.jsonl(metric['metadata_jsonl'])
    rows = []
    for row in reference:
        source = int(row['source_index'])
        candidates = [source, *map(int, row['target_indices'])]
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError('Expected ten distinct trajectories per environment.')
        ordered = sorted(candidates, key=lambda i: (metadata[i][config['support_action_order_field']], i))
        rank = {index: position for position, index in enumerate(ordered)}
        supports = [source]
        while len(supports) < config['support_size']:
            remaining = [index for index in ordered if index not in supports]
            supports.append(max(remaining, key=lambda i: (min(abs(rank[i] - rank[j]) for j in supports), -rank[i])))
        queries = [int(index) for index in row['target_indices'] if int(index) not in supports]
        if len(queries) != 6 or set(supports) & set(queries):
            raise ValueError('Expected four supports and six disjoint queries.')
        record = dict(row, support_indices=supports, query_indices=queries, target_indices=queries,
                      support_sample_ids=[metadata[i]['sample_id'] for i in supports],
                      target_sample_ids=[metadata[i]['sample_id'] for i in queries],
                      support_action_ids=[metadata[i]['action_id'] for i in supports],
                      support_selection=config['support_selection'])
        record.pop('source_inner_step', None)
        rows.append(record)
    if sum(len(row['query_indices']) for row in rows) != config['expected_queries']:
        raise ValueError('Unexpected query count.')
    path = output / 'input_support_query_manifest.json'
    if path.exists() and json.loads(path.read_text()) != rows:
        raise RuntimeError('Refusing to change an existing K=4 protocol during resume.')
    shared.write_json(path, rows)
    return path, rows, metadata


def compose_grids(config, metric, output, rows, metadata):
    import imageio.v2 as imageio
    import numpy as np
    from scripts.evaluation import compose_context_transfer_support_grids as grid
    destination = output / 'grids_support_plus_queries'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        source = row['source_index']
        target = destination / f"{row['domain']}_env{row['environment_id']}_source{source:04d}_4support_6query.mp4"
        indices = row['support_indices'] + row['query_indices']
        pred_dir = grid._prediction_dir(output / 'transfer/raw', source)
        observations, predictions = [], []
        for index in indices:
            item = metadata[index]
            length = int(item.get('length', item['end_frame'] - item['start_frame'] + 1))
            observations.append(grid._read_gt_video(item, shared.resolve(metric['dataset_root']), 224, 224, length))
            predictions.append(None if index in row['support_indices'] else grid._read_pred_video(
                pred_dir / grid._default_pred_name(index, item), 224 * len(item['video']), 224))
        count = min([len(frames) for frames in observations] + [len(frames) for frames in predictions if frames is not None])
        partial = target.with_name(target.stem + '.partial.mp4')
        with imageio.get_writer(str(partial), fps=config['fps'], codec='libx264', quality=config['quality'], macro_block_size=1) as writer:
            for frame_id in range(count):
                cells = []
                for index, gt_frames, pred_frames in zip(indices, observations, predictions):
                    gt = grid._resize_rgb(gt_frames[frame_id], 224, 224)
                    action = grid._action_label(metadata[index])
                    if pred_frames is None:
                        top = grid._decorate(gt, f'Observed support: {action}', border=(255, 214, 0))
                        bottom = grid._decorate(gt, 'Used in pooled Z (K=4)', border=(255, 214, 0))
                    else:
                        top = grid._decorate(gt, f'GT query: {action}')
                        bottom = grid._decorate(grid._resize_rgb(pred_frames[frame_id], 224, 224), f'DINO K=4: {action}')
                    cells.append(np.concatenate([top, bottom], axis=0))
                writer.append_data(np.concatenate([np.concatenate(cells[:5], axis=1), np.concatenate(cells[5:], axis=1)], axis=0))
        os.replace(partial, target)
        records.append({'path': str(target), 'source_index': source, 'domain': row['domain'],
                        'support_indices': row['support_indices'], 'query_indices': row['query_indices'],
                        'grid_rows': 2, 'grid_columns': 5, 'frames': count})
    shared.write_json(destination / 'support_plus_query_grid_manifest.json', records)


def evaluate(config):
    train = yaml.safe_load(shared.resolve(config['train_config']).read_text())
    metric = yaml.safe_load(shared.resolve(config['metric_template']).read_text())
    base = shared.resolve(config['output_base'])
    selected = shared.select_checkpoint(config, base)
    output = base / f"step_{selected['step']}" / f"seed_{config['seed']}"
    flat = output / 'flat'
    flat.mkdir(parents=True, exist_ok=True)
    manifest, rows, metadata = manifest_k4(config, train, metric, output)
    shared.write_json(output / 'launch_protocol.json', {
        'launch_config': config, 'checkpoint': selected, 'training_config': train,
        'inference_support_size': 4, 'training_support_size_choices': [1, 2],
        'support_aggregation': 'mean_support_summary_before_output_head',
        'query_count': 60, 'query_support_disjoint': True,
        'support_selection_uses_outcomes': False,
        'action_candidates': 'four observed supports plus six predicted queries, matching the existing support-observation rule',
        'comparison_caveat': 'K=4 uses more observed trajectories and a smaller query set than K=1; do not replace the K=1 scoreboard.',
    })
    if not (output / 'rollouts.complete').exists():
        shared.quarantine_incomplete_videos(flat, int(train['video']['num_frames']))
        wan, dino, checkpoint, extracted = shared.stage_weights(config, train, selected)
        shared.run(shared.PYTHON, ROOT / 'scripts/methods/infer_dinov2_event80.py',
                   '--config', shared.resolve(config['train_config']),
                   '--dataset_metadata_path', shared.resolve(metric['metadata_jsonl']),
                   '--dataset_base_path', shared.resolve(metric['dataset_root']),
                   '--model_paths', wan, '--dinov2_model_path', dino,
                   '--dinov2_checkpoint_path', checkpoint, '--wan_checkpoint_output', extracted,
                   '--support_indices', ','.join(str(i) for row in rows for i in row['support_indices']),
                   '--sample_indices', ','.join(str(i) for row in rows for i in row['query_indices']),
                   '--expected_supports_per_environment', 4, '--output_path', flat,
                   '--num_inference_steps', config['num_inference_steps'], '--cfg_scale', config['cfg_scale'],
                   '--fps', config['fps'], '--quality', config['quality'], '--seed', config['seed'], '--skip_existing')
        (output / 'rollouts.complete').touch()
    shared.run(shared.PYTHON, ROOT / 'scripts/evaluation/organize_grouped_transfer_outputs.py',
               '--manifest', manifest, '--flat-root', flat, '--output-root', output, '--method-slug', config['method'])
    transfer = output / 'transfer'
    transfer_rows = json.loads((transfer / 'transfer_plan.json').read_text())
    for domain in ('id', 'ood'):
        shared.write_json(transfer / f'transfer_plan_{domain}.json', [row for row in transfer_rows if row['domain'] == domain])
    if not (output / 'grids.complete').exists():
        compose_grids(config, metric, output, rows, metadata)
        (output / 'grids.complete').touch()
    metric.update(method_name=config['method'], seed=config['seed'], support_size=4,
                  support_query_manifest=str(manifest), output_dir=str(output / 'action_evaluation'),
                  transfer_plans=[{'path': str(transfer / f'transfer_plan_{domain}.json'), 'domain': domain} for domain in ('id', 'ood')])
    metric_path = output / 'evaluation_config.yaml'
    metric_path.write_text(yaml.safe_dump(metric, sort_keys=False))
    if not (output / 'metrics.complete').exists():
        shared.run(shared.PYTHON, ROOT / 'scripts/evaluation/evaluate_sim_action_selection.py', '--config', metric_path)
        shared.run(shared.PYTHON, ROOT / 'scripts/evaluation/evaluate_sim_transfer_metrics.py', '--config', metric_path,
                   '--lpips', '--lpips-net', 'alex', '--lpips-device', 'cuda', '--lpips-batch-size', 8)
        (output / 'metrics.complete').touch()
    (output / 'inference.complete').touch()
    print(f'[done] K=4 Gravity output={output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_PARTITION') != 'yejin-lo':
        raise RuntimeError('Inference, video processing and LPIPS require a low-priority compute node.')
    os.chdir(ROOT)
    config = yaml.safe_load(shared.resolve(args.config).read_text())
    if config['support_size'] != 4:
        raise ValueError('This entry point is specifically for K=4.')
    base = shared.resolve(config['output_base'])
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluate(config)


if __name__ == '__main__':
    main()
