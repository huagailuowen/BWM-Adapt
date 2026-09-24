#!/usr/bin/env python3
"""Aggregate existing real support-count metrics using the frozen formal cohorts."""
import csv
import json
import os
from pathlib import Path

import summarize_real97_simlr as formal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/support_number_analysis/real'


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run metric aggregation on a compute node')
    result = []
    door_protocol_path = formal.TABLE / 'metrics/door_first_close_level_tolerance1_20260919.json'
    door_protocol = formal.read(door_protocol_path)
    selection_path = OUT / 'selected_runs.json'
    selected_runs = formal.read(selection_path).get('runs', {}) if selection_path.exists() else {}
    for task, ks in [('ball', [1, 2, 4]), ('door', [1, 2, 4])]:
        for k in ks:
            path = OUT / task / f'k{k}' / 'metrics/summary_provisional.json'
            if task == 'door' and k == 2:
                path = ROOT / door_protocol['sources']['ours_simlr']
            if f'{task}/k{k}' in selected_runs:
                path = ROOT / selected_runs[f'{task}/k{k}']['metric_source']
            if not path.exists():
                continue
            report = formal.read(path)
            image = next(r for r in report['image_object']
                         if r['split'] == 'test' and r['method'] == 'ours_stage2')
            action = next(r for r in report['action'] if r['method'] == 'ours_stage2')
            if task == 'door':
                indexed = {d['environment']: d for d in action['decisions']}
                decisions = []
                for env, truth in door_protocol['ground_truth_lowest_level'].items():
                    states = indexed[env]['closed_by_level']
                    if {int(level) for level in states} != set(range(1, 11)):
                        raise RuntimeError('Incomplete Door action candidates: ' + env)
                    predicted = next((level for level in range(1, 11)
                                      if states[str(level)] is True), None)
                    error = abs(predicted - truth) if predicted is not None else None
                    score = 1.0 if error == 0 else 0.5 if error == 1 else 0.0
                    decisions.append(dict(environment=env, gt_lowest_level=truth,
                                          predicted_lowest_level=predicted, score=score))
                action = dict(macro_score=sum(d['score'] for d in decisions) / len(decisions),
                              scorable_count=len(decisions), decisions=decisions,
                              protocol_source=str(door_protocol_path),
                              metric=door_protocol['metric'])
                formal.write(OUT / task / f'k{k}' / 'formal_metrics/door_action_latest.json', action)
            result.append(dict(task=task, k=k, queries=image['queries'], **image['query_mean'],
                               action_pct=100 * action['macro_score'],
                               action_denominator=action['scorable_count'], source=str(path)))
    selection = formal.read(formal.TABLE / 'metrics/stick_action_seed20260927.json')['selected_run']['selection']
    stick_keys = {key for env in selection for key in env['keys']}
    for k in [1, 4]:
        directory = OUT / 'stick' / f'k{k}'
        tail = directory / 'metrics_visible_tail'
        if not (tail / 'summary.json').exists():
            continue
        cases = [r for r in formal.read(tail / 'per_query.json') if r['key'] in stick_keys]
        if len(cases) != 36 or len(stick_keys) != 36:
            raise RuntimeError('Stick fixed36 cohort incomplete')
        predicted = [r for r in cases if r['action']['ours_stage2']['5']['state'] == 'positive']
        hits = sum(r['dataset_outcome_for_audit'] == 'balanced' for r in predicted)
        action = dict(queries=36, predicted_balanced=len(predicted), true_balanced=hits,
                      precision=hits / len(predicted) if predicted else None,
                      subset_seed=20260927, keys=sorted(stick_keys))
        formal.write(directory / 'formal_metrics/stick_action_fixed36.json', action)
        image = formal.read(tail / 'summary.json')['groups']['test']['ours_stage2']
        lpips = formal.read(directory / 'metrics/summary.json')['groups']['test']['ours_stage2'].get('lpips')
        result.append(dict(task='stick', k=k, queries=45,
                           psnr=image['psnr'], ssim=image['ssim'], lpips=lpips,
                           ade_px=image['ade_px'], fde_px=image['fde_px'],
                           action_pct=100 * action['precision'] if predicted else None,
                           action_denominator=len(predicted),
                           source=str(directory / 'formal_metrics/stick_action_fixed36.json')))
    cohort = formal.read(formal.TABLE / 'soft_shared6/per_query.json')
    keys = {r['key'] for r in cohort}
    for k in [1, 2]:
        directory = OUT / 'soft' / f'k{k}'
        metrics = directory / 'metrics'
        if not (metrics / 'per_query.json').exists():
            continue
        selected = [r for r in formal.read(metrics / 'per_query.json') if r['key'] in keys]
        if len(selected) != 12:
            raise RuntimeError('Soft formal test cohort must contain exactly 12 queries')
        formal.SOFT = metrics
        formal.OUT = directory / 'formal_metrics'
        onset = formal.onset_scores(cohort)
        lpips_path = metrics / 'lpips_complete.json'
        lpips_value = None
        if lpips_path.exists():
            lpips_rows = [r for r in formal.read(lpips_path)['records'] if r['key'] in keys]
            if len(lpips_rows) != 12 or {r['key'] for r in lpips_rows} != keys:
                raise RuntimeError('Soft LPIPS formal cohort incomplete')
            lpips_value = formal.mean([r['mean'] for r in lpips_rows])
        result.append(dict(task='soft', k=k, queries=12,
            lpips=lpips_value,
            psnr=formal.mean([r['image_metrics']['stage2']['psnr_db'] for r in selected]),
            ssim=formal.mean([r['image_metrics']['stage2']['ssim'] for r in selected]),
            ade_px=formal.mean([r['common_mask_metrics']['stage2']['center_ade_px'] for r in selected]),
            fde_px=formal.mean([r['metrics']['stage2']['center_fde_px'] for r in selected]),
            action_pct=onset['success_pct'], action_denominator=11,
            source=str(formal.OUT / 'soft_onset6.json')))
    destination = OUT / 'formal_completed_summary.json'
    formal.write(destination, {'results': result, 'missing_results_are_not_zero': True})
    columns = ['task', 'k', 'queries', 'action_pct', 'action_denominator',
               'psnr', 'ssim', 'lpips', 'ade_px', 'fde_px', 'source']
    temporary = OUT / 'formal_completed_summary.csv.partial'
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result)
    temporary.replace(OUT / 'formal_completed_summary.csv')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
