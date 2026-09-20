#!/usr/bin/env python3
"""Unchanged visible-tail tracking/action criteria for final Stick baselines."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path

import numpy as np

import score_real916_stick_final as engine
from real916_stick_final_common import ROOT, read_rows, write_json


def read(path):
    return json.loads(Path(path).read_text())


def prepare_tasks(config, output):
    method = config['method']
    run = ROOT / config['inference_output']
    complete = read(run / 'inference_complete.json')
    if (complete['queries'], complete['test_queries'], complete['environments']) != (63, 45, 9):
        raise RuntimeError('Incomplete inference population')
    if complete['method'] != method:
        raise RuntimeError('Inference method mismatch')
    prepared = ROOT / config['prepared']
    if (run / 'input_manifest/query.jsonl').read_bytes() != (prepared / 'query.jsonl').read_bytes():
        raise RuntimeError('Query manifest changed')
    plan = read(prepared / 'plan.json')
    old = {r['key']: r for r in read(ROOT / config['reference_metrics'] / 'per_query.json')}
    geometry = {**read(ROOT / config['geometry_config']), **read(ROOT / config['fusion_config'])}
    contact = {**read(ROOT / config['contact_config']), 'angle_thresholds_deg': [3, 5]}
    tasks = []
    for index, row in enumerate(read_rows(prepared / 'query.jsonl')):
        key = f"q{index:04d}_{row['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"
        prediction = read(run / 'completed' / (key + '.json'))
        standard = read(ROOT / config['standard'] / 'completed' / (key + '.json'))
        ours = read(ROOT / config['ours'] / 'completed' / (key + '.json'))
        if any(r['row'] != row for r in (prediction, standard, ours)):
            raise RuntimeError('Paired query identity mismatch')
        reference = old[key]
        phase = copy.deepcopy(reference['phase'])
        hi = row['evaluation_frame_indices'][-1]
        if hi - 2 < 1:
            raise RuntimeError('Insufficient real terminal frames')
        phase['decision'] = dict(terminal_window=dict(frame_interval=[hi - 2, hi]),
                                 horizon_status='visible_real_video_tail')
        tasks.append(dict(key=key, index=index, row=row,
            paths={'gt': (ours['paths']['gt'] if config.get('canonical_gt_codec_audit') else prediction['paths']['gt']), 'standard': standard['paths']['standard'],
                   'ours_stage2': ours['paths']['stage2'], method: prediction['paths'][method]},
            other_gt=(standard['paths']['gt'] if config.get('canonical_gt_codec_audit') else ours['paths']['gt']),
            baseline_export_gt=prediction['paths']['gt'], canonical_gt_codec_audit=config.get('canonical_gt_codec_audit', False),
            wanted=row['evaluation_frame_indices'], phase=phase,
            action_eligible=True, dataset_outcome_for_audit=reference['dataset_outcome_for_audit'],
            native_video=str(Path(plan['source_dataset']) / row['video'][0]), output=str(output),
            geometry=geometry, contact=contact))
    return tasks


def process_with_codec_audit(task):
    if task.get('canonical_gt_codec_audit'):
        # The old reference floors uint8 and encodes twice; the new export
        # rounds uint8 and encodes once. Keep both exports untouched. All
        # methods are scored against the same original reference GT, but only
        # after validating the alternate export across every decoded frame.
        baseline = np.asarray(engine.decode(task['baseline_export_gt']), dtype=np.float32)
        reference = np.asarray(engine.decode(task['paths']['gt']), dtype=np.float32)
        if baseline.shape != reference.shape:
            raise RuntimeError('GT geometry/frame-count mismatch: ' + task['key'])
        delta = baseline - reference
        frame_mae = np.abs(delta).mean(axis=(1, 2, 3))
        bias = delta.mean(axis=(1, 2), keepdims=True)
        residual = np.abs(delta - bias).mean(axis=(1, 2, 3))
        shifted = {
            str(shift): float(np.abs(baseline[max(0, shift):len(baseline) + min(shift, 0)]
                - reference[max(0, -shift):len(reference) - max(shift, 0)]).mean())
            for shift in (-1, 0, 1)
        }
        accepted = (float(frame_mae.max()) <= 3.0 and float(residual.max()) <= 1.5
                    and shifted['0'] <= min(shifted['-1'], shifted['1']) + 0.1)
        audit = dict(key=task['key'], source=task['baseline_export_gt'], canonical_gt=task['paths']['gt'],
            frame_count=len(baseline), all_frame_mae=frame_mae.tolist(),
            signed_channel_bias=bias.reshape(len(bias), 3).tolist(), centered_mae=residual.tolist(),
            temporal_shift_mae=shifted, accepted=accepted,
            gates=dict(max_frame_mae=3.0, max_bias_removed_frame_mae=1.5, temporal_tolerance=0.1),
            method='decoded all-frame codec consistency, not a relaxed tracking/action threshold')
        write_json(Path(task['output']) / 'gt_codec_audit' / (task['key'] + '.json'), audit)
        if not accepted:
            raise RuntimeError('GT discrepancy exceeds codec-consistency bounds: ' + task['key'])
        del baseline, reference, delta
    return engine.process(task)


def comparison(config, output, records):
    selected = read(ROOT / config['selected_cohort'])['selected_run']
    keys = {k for env in selected['selection'] for k in env['keys']}
    summary = read(output / 'summary.json')
    groups = {}
    for name, cohort in [('test45', [r for r in records if r['split'] == 'test']),
                         ('fixed_seed20260927_test36', [r for r in records if r['key'] in keys])]:
        if len(cohort) != (45 if name == 'test45' else 36):
            raise RuntimeError('Fixed scoring cohort changed')
        groups[name] = {}
        for method in engine.METHODS:
            positive = [r for r in cohort if r['action'][method]['5']['state'] == 'positive']
            if any(r['dataset_outcome_for_audit'] not in ('balanced', 'left_down', 'right_down') for r in cohort):
                raise RuntimeError('Unknown annotation ground truth')
            tp = sum(r['dataset_outcome_for_audit'] == 'balanced' for r in positive)
            actual = sum(r['dataset_outcome_for_audit'] == 'balanced' for r in cohort)
            metrics = dict(queries=len(cohort), selected=len(positive), true_balanced=tp,
                false_balanced=len(positive) - tp, precision=tp / len(positive) if positive else None,
                balanced_recall=tp / actual if actual else None,
                unknown_predictions=sum(r['action'][method]['5']['state'] == 'unknown' for r in cohort))
            for field in ('ade_px', 'fde_px', 'angle_mae_deg'):
                values = [r['object'][method][field] for r in cohort if r['object'][method][field] is not None]
                metrics[field] = sum(values) / len(values) if values else None
                metrics[field + '_valid_queries'] = len(values)
            if name == 'test45':
                metrics.update({k: summary['groups']['test'][method].get(k) for k in ('psnr', 'ssim', 'lpips')})
            groups[name][method] = metrics
    write_json(output / 'comparison.json', dict(groups=groups, gt_source='dataset outcome labels',
        action_rule='precision among predicted balanced; unchanged 5-degree visible-tail criterion',
        object_mask='common valid frames among GT and these three methods; coverage counts reported',
        primary_cohort='full_test45', post_hoc_seed_subset_secondary=True,
        main_table_modified=False, visual_audit_pending=True, lpips_complete=summary.get('lpips_complete', False)))
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['cpu', 'lpips'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    config = read(args.config)
    engine.METHODS = ('standard', 'ours_stage2', config['method'])
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ('.' + args.mode + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'protocol.json').exists() and read(output / 'protocol.json') != config:
        raise RuntimeError('Scoring protocol changed')
    write_json(output / 'protocol.json', config)
    tasks = prepare_tasks(config, output)
    if args.mode == 'cpu':
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            records = list(pool.map(process_with_codec_audit, tasks))
        summary = engine.summarize(records)
        summary['notes'] = ['Fixed full test45 primary; fixed post-hoc seed20260927 test36 secondary.',
                            'No outcome-based resampling; same tracking and visible-tail classifier.']
        if config.get('canonical_gt_codec_audit'):
            summary['notes'].append('All alternate GT exports audited; scoring uses the unchanged original reference GT for every method.')
        write_json(output / 'per_query.json', records)
        write_json(output / 'summary.json', summary)
        engine.export_table(summary, output)
        write_json(output / 'cpu_complete.json', dict(queries=len(records)))
    else:
        if not (output / 'cpu_complete.json').exists():
            raise RuntimeError('CPU scoring incomplete')
        records = read(output / 'per_query.json')
        engine.lpips_scores(tasks, output)
    comparison(config, output, records)


if __name__ == '__main__':
    main()
