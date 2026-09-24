#!/usr/bin/env python3
"""Compare Ball K=1 Level-2 Stage2 runs with the formal mixed-support rule."""

import json
import math
import os
from pathlib import Path

from score_real97_globalmean_main_rules import ball_action, ball_queries, read, write


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/eval_real97_ball_k1l2_main_rules_20260922_v1'
NAMES = ('ball-0', 'ball-1', 'ball-1dot5', 'ball-2', 'ball-3', 'ball-4', 'ball-7', 'ball-9')


def run(variant):
    inference = ROOT / f'outputs/infer_real97_ball_dual6500_simlr_{variant}_k1l2_20260922_v1'
    score_root = ROOT / f'outputs/eval_real97_ball_dual6500_simlr_{variant}_k1l2_20260922_v1'
    done = read(inference / 'inference_complete.json')
    if (done['queries'], done['environments'], done['model_step'], done['table_step']) != (96, 8, 6500, 6500):
        raise RuntimeError(f'Incomplete inference: {inference}')
    report = read(score_root / 'summary_provisional.json')
    if report['queries_scored'] != 96 or report['queries_expected'] != 96:
        raise RuntimeError(f'Incomplete scoring: {score_root}')
    if not (score_root / 'lpips_complete.json').is_file():
        raise RuntimeError(f'LPIPS incomplete: {score_root}')
    prepared = Path(read(inference / 'support_override_prepared.json')['prepared'])
    plan = read(prepared / 'plan.json')
    support = [json.loads(line) for line in (prepared / 'support.jsonl').read_text().splitlines() if line]
    active = {}
    for environment in plan['environments']:
        name = environment['environment']
        if name not in ('ball-0', 'ball-1'):
            continue
        indices = environment['support_indices']
        if len(indices) != 1 or int(support[indices[0]]['action_level']) != 2:
            raise RuntimeError(f'{variant}/{name}: expected one Level-2 support')
        context = read(inference / 'contexts' / (name + '.json'))
        if context['support_indices'] != indices:
            raise RuntimeError(f'{variant}/{name}: adaptation used different support indices')
        active[name] = dict(indices=indices, episode=support[indices[0]]['episode_index'],
                            level=2, split=support[indices[0]]['dataset_split'])
    if set(active) != {'ball-0', 'ball-1'}:
        raise RuntimeError('Expected both replaced environments')
    queries = {name: ball_queries(score_root, name) for name in NAMES}
    decisions = {name: ball_action(queries[name], name) for name in NAMES}
    score = sum(row['score'] for row in decisions.values()) / len(decisions)
    recorded = next(row['macro_score'] for row in report['action'] if row['method'] == 'ours_stage2')
    if not math.isclose(score, recorded, abs_tol=1e-8):
        raise RuntimeError(f'{variant}: query-level score and aggregate disagree: {score} != {recorded}')
    image = next(row['query_mean'] for row in report['image_object']
                 if row['split'] == 'test' and row['method'] == 'ours_stage2')
    return dict(inference=str(inference), scoring=str(score_root), support=active,
                score=score, decisions=decisions, test_image_object=image, queries=queries)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run on a compute allocation')
    cluster, globalmean = run('cluster'), run('globalmean')
    for name in NAMES:
        for level in range(1, 11):
            a, b = cluster['queries'][name][level], globalmean['queries'][name][level]
            if a['sample_id'] != b['sample_id'] or a['marker']['center'] != b['marker']['center']:
                raise RuntimeError(f'Query or target marker changed: {name} L{level}')
            if a['trajectory_summary']['gt']['peak_center'] != b['trajectory_summary']['gt']['peak_center']:
                raise RuntimeError(f'GT tracking changed: {name} L{level}')
    for data in (cluster, globalmean):
        data.pop('queries')
    formal = read(ROOT / 'outputs/eval_real97_globalmean_main_rules_20260922_v1/summary.json')
    if not math.isclose(formal['ball']['formal_cluster_score'], 0.75, abs_tol=1e-8):
        raise RuntimeError('Formal mixed-support Ball reference changed')
    result = dict(rule='first_reach_marker_minus_30px_and_successor_pair_overlap',
                  original_formal_cluster_score=0.75, cluster_k1l2=cluster,
                  globalmean_k1l2=globalmean,
                  paired_query_gt_and_marker_check='passed for all eight environments and ten test levels each')
    write(OUTPUT / 'summary.json', result)
    print(json.dumps({'formal_cluster':0.75, 'cluster_k1l2':cluster['score'],
                      'globalmean_k1l2':globalmean['score'],
                      'output':str(OUTPUT / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
