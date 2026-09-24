#!/usr/bin/env python3
"""Recalculate the current Real97 main-table Door/Ball action rules."""

import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / 'results/real97_all_methods_main_table_v1'
OUTPUT = ROOT / 'outputs/eval_real97_globalmean_main_rules_20260922_v1'
SCORES = {
    'door_cluster': ROOT / 'outputs/eval_real97_door_simlr_20260918_v1',
    'door_global': ROOT / 'outputs/eval_real97_door_dual6500_simlr_globalmean_20260922_v1',
    'ball_original': ROOT / 'outputs/eval_real97_ball_simlr_20260918_v1',
    'ball_l7': ROOT / 'outputs/eval_real97_ball_dual6500_simlr_supportL7_ball7_ball9_20260920_v1',
    'ball_global': ROOT / 'outputs/eval_real97_ball_dual6500_simlr_globalmean_supportL7_20260922_v1',
}
L7_ENVS = {'ball-7', 'ball-9'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def near(actual, expected, label):
    if not math.isclose(actual, expected, abs_tol=1e-8):
        raise RuntimeError(f'{label}: expected {expected}, got {actual}')


def stage2_summary(path, expected_queries):
    report = read(path / 'summary_provisional.json')
    if report['queries_scored'] != expected_queries or report['queries_expected'] != expected_queries:
        raise RuntimeError(f'Incomplete source scoring: {path}')
    return report


def action_row(report):
    return next(row for row in report['action'] if row['method'] == 'ours_stage2')


def image_row(report):
    return next(row['query_mean'] for row in report['image_object']
                if row['split'] == 'test' and row['method'] == 'ours_stage2')


def door_action(report, truth):
    decisions = action_row(report)['decisions']
    chosen = {}
    for row in decisions:
        name = row['environment']
        if name not in truth:
            continue
        closed = row['closed_by_level']
        if set(closed) != {str(level) for level in range(1, 11)}:
            raise RuntimeError(f'Incomplete Door action levels: {name}')
        predicted = next((level for level in range(1, 11) if closed[str(level)] is True), None)
        actual = int(truth[name])
        score = 1.0 if predicted == actual else 0.5 if predicted is not None and abs(predicted - actual) == 1 else 0.0
        chosen[name] = dict(gt_lowest_level=actual, predicted_lowest_level=predicted,
                            score=score, closed_by_level=closed)
    if set(chosen) != set(truth):
        raise RuntimeError(f'Door action cohort mismatch: {set(chosen) ^ set(truth)}')
    return dict(score=sum(row['score'] for row in chosen.values()) / len(chosen),
                exact=sum(row['score'] == 1 for row in chosen.values()),
                within_one=sum(row['score'] > 0 for row in chosen.values()),
                environments=chosen)


def ball_queries(folder, environment):
    rows = {}
    for path in (folder / 'queries/ball').glob('*.json'):
        row = read(path)
        if row['environment'] != environment or row['split'] != 'test':
            continue
        level = int(row['level'])
        if level in rows:
            raise RuntimeError(f'Duplicate Ball test level: {environment} L{level}')
        rows[level] = row
    if set(rows) != set(range(1, 11)):
        raise RuntimeError(f'Incomplete Ball test levels: {environment}: {sorted(rows)}')
    return rows


def ball_action(rows, environment):
    reached = {'gt': {}, 'ours_stage2': {}}
    for level, row in rows.items():
        marker = row['marker']['center']
        if marker is None:
            raise RuntimeError(f'Missing blue marker: {environment} L{level}')
        threshold = float(marker[0]) - 30.0
        for method in reached:
            peak = row['trajectory_summary'][method]['peak_center']
            reached[method][level] = None if peak is None else bool(float(peak[0]) >= threshold)
    gt_first = next((level for level in range(1, 11) if reached['gt'][level] is True), None)
    pred_first = next((level for level in range(1, 11) if reached['ours_stage2'][level] is True), None)
    if gt_first is None or any(value is None for value in reached['gt'].values()):
        raise RuntimeError(f'Ball GT cannot define the main-table action: {environment}')
    gt_pair = [gt_first, gt_first + 1]
    pred_pair = [pred_first, pred_first + 1] if pred_first is not None else []
    return dict(gt_first_reach=gt_first, predicted_first_reach=pred_first,
                gt_pair=gt_pair, predicted_pair=pred_pair,
                score=len(set(gt_pair) & set(pred_pair)) / 2,
                reaches_by_level=reached)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run scoring on a compute allocation')
    door_truth = read(TABLE / 'metrics/door_first_close_level_tolerance1_20260919.json')
    door_truth = door_truth['ground_truth_lowest_level']
    door_old = stage2_summary(SCORES['door_cluster'], 120)
    door_new = stage2_summary(SCORES['door_global'], 120)
    door_old_action = door_action(door_old, door_truth)
    door_new_action = door_action(door_new, door_truth)
    near(door_old_action['score'], 6.5 / 9, 'Door main-table action')

    ball_old = stage2_summary(SCORES['ball_original'], 96)
    ball_l7 = stage2_summary(SCORES['ball_l7'], 96)
    ball_new = stage2_summary(SCORES['ball_global'], 96)
    environments = ('ball-0', 'ball-1', 'ball-1dot5', 'ball-2',
                    'ball-3', 'ball-4', 'ball-7', 'ball-9')
    old_decisions, new_decisions, sources = {}, {}, {}
    for name in environments:
        old_source = 'ball_l7' if name in L7_ENVS else 'ball_original'
        old_rows = ball_queries(SCORES[old_source], name)
        new_rows = ball_queries(SCORES['ball_global'], name)
        for level in range(1, 11):
            a, b = old_rows[level], new_rows[level]
            if a['sample_id'] != b['sample_id'] or a['marker']['center'] != b['marker']['center']:
                raise RuntimeError(f'Ball query/marker changed: {name} L{level}')
            if a['trajectory_summary']['gt']['peak_center'] != b['trajectory_summary']['gt']['peak_center']:
                raise RuntimeError(f'Ball GT tracking changed: {name} L{level}')
        old_decisions[name] = ball_action(old_rows, name)
        new_decisions[name] = ball_action(new_rows, name)
        sources[name] = str(SCORES[old_source])
    old_ball_score = sum(row['score'] for row in old_decisions.values()) / 8
    new_ball_score = sum(row['score'] for row in new_decisions.values()) / 8
    near(old_ball_score, 0.75, 'Ball main-table mixed support action')
    near(new_ball_score, action_row(ball_new)['macro_score'], 'Ball global-mean scored action')

    report = dict(rule_version='current_real97_main_table',
                  door=dict(rule='first_closing_level_exact1_adjacent0p5', denominator=9,
                            formal_cluster=door_old_action, global_mean=door_new_action,
                            formal_cluster_test_image_object=image_row(door_old),
                            global_mean_test_image_object=image_row(door_new)),
                  ball=dict(rule='first_reach_marker_minus_30px_and_successor_pair_overlap', denominator=8,
                            formal_cluster_score=old_ball_score, global_mean_score=new_ball_score,
                            formal_cluster_environments=old_decisions,
                            global_mean_environments=new_decisions,
                            formal_cluster_environment_sources=sources,
                            formal_cluster_test_image_object=image_row(ball_l7),
                            global_mean_test_image_object=image_row(ball_new)),
                  note='Ball formal action combines six original-cluster environment scores and Level-7-support scores for ball-7/ball-9; all query GT/marker tracks checked against the global-mean run.')
    write(OUTPUT / 'summary.json', report)
    print(json.dumps({'door_cluster':door_old_action['score'], 'door_global':door_new_action['score'],
                      'ball_cluster':old_ball_score, 'ball_global':new_ball_score,
                      'output':str(OUTPUT / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
