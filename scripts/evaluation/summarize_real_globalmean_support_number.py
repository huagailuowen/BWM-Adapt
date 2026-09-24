#!/usr/bin/env python3
"""Formal K comparison with one global-mean initialization per real task."""
import csv
import json
import os
from pathlib import Path

import summarize_real97_simlr as formal

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/support_number_analysis/real/globalmean_20260922'


def source(task, k):
    if task == 'ball' and k == 1:
        run = ROOT / 'outputs/infer_real97_ball_dual6500_simlr_globalmean_k1l2_20260922_v1'
        metrics = ROOT / 'outputs/eval_real97_ball_dual6500_simlr_globalmean_k1l2_20260922_v1'
    elif task == 'door' and k == 2:
        run = ROOT / 'outputs/infer_real97_door_dual6500_simlr_globalmean_20260922_v1'
        metrics = ROOT / 'outputs/eval_real97_door_dual6500_simlr_globalmean_20260922_v1'
    elif task == 'soft' and k == 4:
        run = ROOT / 'outputs/infer_real97_soft_5500_simlr_globalmean_20260922_v1'
        metrics = ROOT / 'outputs/eval_real97_soft_5500_simlr_globalmean_20260922_v1'
    else:
        run = DEST / task / f'k{k}'
        metrics = run / 'metrics'
    return run, metrics


def score_door(report):
    protocol = formal.read(formal.TABLE / 'metrics/door_first_close_level_tolerance1_20260919.json')
    action = next(r for r in report['action'] if r['method'] == 'ours_stage2')
    indexed = {r['environment']: r for r in action['decisions']}
    results = []
    for env, gt in protocol['ground_truth_lowest_level'].items():
        states = indexed[env]['closed_by_level']
        if {int(x) for x in states} != set(range(1, 11)):
            raise RuntimeError('Incomplete Door action candidates')
        pred = next((level for level in range(1, 11) if states[str(level)] is True), None)
        error = abs(pred - gt) if pred is not None else None
        results.append({'environment': env, 'GT': gt, 'prediction': pred,
                        'score': 1 if error == 0 else .5 if error == 1 else 0})
    return 100 * sum(r['score'] for r in results) / len(results), len(results), results


def score_soft(metrics):
    cohort = formal.read(formal.TABLE / 'soft_shared6/per_query.json')
    old = {r['key']: r for r in formal.read(formal.TABLE / 'soft_sliding_onset_exploratory_v1/per_query.json')}
    import importlib.util
    path = formal.TABLE / 'soft_sliding_onset_exploratory_v1/compute.py'
    spec = importlib.util.spec_from_file_location('frozen_soft_onset', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    records = []
    for row in cohort:
        key = row['key']
        old_gt = old[key]['thresholds']['8']['events']['gt']['event']
        gt_tracks = formal.read(metrics / 'cases' / key / 'gt_tracks.json')['tracks']
        predicted = formal.read(metrics / 'cases' / key / 'stage2_tracks.json')['tracks']
        now_gt = engine.onset(gt_tracks, 8)['event']
        if (old_gt is None) != (now_gt is None) or (old_gt is not None and old_gt['native_frame'] != now_gt['native_frame']):
            raise RuntimeError('Soft GT onset differs from frozen test cohort')
        event = engine.onset(predicted, 8)['event']
        error = None
        if old_gt is not None and event is not None:
            angles = old[key]['rotation']['accumulated_rotation_deg']
            error = abs(angles[event['native_frame']] - old_gt['cumulative_angle_deg'])
        records.append({'key': key, 'GT_positive': old_gt is not None,
                        'angle_error': error, 'success6': error is not None and error <= 6})
    positives = [r for r in records if r['GT_positive']]
    if len(positives) != 11:
        raise RuntimeError('Soft formal action denominator changed')
    return 100 * sum(r['success6'] for r in positives) / 11, 11, records


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Aggregate on a compute node')
    results = []
    for task in ['ball', 'door']:
        for k in [1, 2, 4]:
            run, metrics = source(task, k)
            if task in ['ball', 'door']:
                report = formal.read(metrics / 'summary_provisional.json')
                image = next(r for r in report['image_object']
                             if r['split'] == 'test' and r['method'] == 'ours_stage2')
                scores = image['query_mean']
                if task == 'door':
                    action, denominator, decisions = score_door(report)
                else:
                    item = next(r for r in report['action'] if r['method'] == 'ours_stage2')
                    action, denominator, decisions = 100 * item['macro_score'], item['scorable_count'], item['decisions']
                record = dict(task=task, k=k, queries=image['queries'], action_pct=action,
                              action_denominator=denominator, **{metric: scores[metric]
                               for metric in ['psnr', 'ssim', 'lpips', 'ade_px', 'fde_px']})
            else:
                cohort = formal.read(formal.TABLE / 'soft_shared6/per_query.json')
                keys = {r['key'] for r in cohort}
                query = [r for r in formal.read(metrics / 'per_query.json') if r['key'] in keys]
                if len(query) != 12:
                    raise RuntimeError('Soft shared6/test12 query cohort incomplete')
                action, denominator, decisions = score_soft(metrics)
                lpips_path = metrics / 'lpips_complete.json'
                if not lpips_path.exists():
                    raise RuntimeError('Soft LPIPS missing: ' + str(lpips_path))
                lpips_rows = [r for r in formal.read(lpips_path)['records'] if r['key'] in keys]
                if len(lpips_rows) != 12:
                    raise RuntimeError('Soft LPIPS test12 incomplete')
                record = dict(task=task, k=k, queries=12, action_pct=action,
                              action_denominator=denominator,
                              psnr=formal.mean([r['image_metrics']['stage2']['psnr_db'] for r in query]),
                              ssim=formal.mean([r['image_metrics']['stage2']['ssim'] for r in query]),
                              lpips=formal.mean([r['mean'] for r in lpips_rows]),
                              ade_px=formal.mean([r['common_mask_metrics']['stage2']['center_ade_px'] for r in query]),
                              fde_px=formal.mean([r['metrics']['stage2']['center_fde_px'] for r in query]))
            record.update(inference_source=str(run), metric_source=str(metrics),
                          initialization='full training-table global mean',
                          action_protocol='published latest rule')
            results.append(record)
            formal.write(DEST / task / f'k{k}' / 'formal_action_decisions.json', decisions)
    formal.write(DEST / 'formal_summary.json', {'results': results,
        'door_k4_support_status': 'post-hoc selected after inspecting earlier failures',
        'comparison': 'Fixed pretrained weights, queries, timestep schedule; per-task global mean Z start'})
    columns = ['task', 'k', 'queries', 'action_pct', 'action_denominator',
               'psnr', 'ssim', 'lpips', 'ade_px', 'fde_px', 'inference_source', 'metric_source']
    temporary = DEST / 'formal_summary.csv.partial'
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(DEST / 'formal_summary.csv')


if __name__ == '__main__':
    main()
