#!/usr/bin/env python3
"""Ball dual6000 factual metrics with first-reaching-level successor overlap."""
from collections import defaultdict
from pathlib import Path
import score_real97_ball_door as engine

engine.METHODS = ('ours_stage1', 'ours_stage2')
original_aggregate = engine.aggregate


def freeze_manifest(config, output):
    path = output / 'manifest.json'
    if path.is_file():
        result = engine.read_json(path)
        if result['config'] != config:
            raise RuntimeError('Score configuration changed')
        return result
    baseline = engine.read_json(engine.ROOT / config['baseline_scores'] / 'manifest.json')
    run = engine.ROOT / config['tasks']['ball']['ours']
    done = engine.read_json(run / 'inference_complete.json')
    if (done['model_step'], done['table_step']) != (6000, 6000):
        raise RuntimeError('Expected model/table step6000 pair')
    indexed = {}
    for marker in sorted((run / 'completed').glob('q*.json')):
        record = engine.read_json(marker)
        identity = record['row']['sample_id']
        if identity in indexed:
            raise RuntimeError('Duplicate query')
        indexed[identity] = (record, marker)
    reference = [q for q in baseline['queries'] if q['task'] == 'ball']
    if set(indexed) != {q['row']['sample_id'] for q in reference} or done['queries'] != len(reference):
        raise RuntimeError('Ball dual and baseline factual query sets differ')
    queries = []
    for q in reference:
        record, marker = indexed[q['row']['sample_id']]
        for field in ('environment','dataset_split','episode_index','action_level','video','action',
                      'start_frame','end_frame','native_frame_indices','evaluation_frame_indices','length'):
            if record['row'].get(field) != q['row'].get(field):
                raise RuntimeError('Factual query mismatch: ' + field)
        paths = {'gt':record['paths']['gt'], 'ours_stage1':record['paths']['stage1'],
                 'ours_stage2':record['paths']['stage2']}
        if not all(Path(p).is_file() for p in paths.values()):
            raise RuntimeError('Missing raw video')
        queries.append({**q, 'paths':paths, 'source_markers':{'ours':str(marker)}})
    result = dict(config=config, queries=queries, methods=list(engine.METHODS), paired_gt_only=True,
                  checkpoint_steps={'ball':{'ours_stage1':6000,'ours_stage2':6000}},
                  fairness_note='Fixed factual queries and existing tracking calibration. Known training-cluster prior; cross-cluster replicas use z0 cluster. Counterfactual action sweeps excluded.')
    engine.save_json(path, result)
    return result


def aggregate(output, manifest):
    report = original_aggregate(output, manifest)
    rows = [engine.read_json(output/'queries/ball'/(q['key']+'.json')) for q in manifest['queries']
            if (output/'queries/ball'/(q['key']+'.json')).is_file()]
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'test':
            groups[row['environment']].append(row)
    actions = []
    for method in engine.METHODS:
        decisions = []
        for name, items in sorted(groups.items()):
            gt, pred, thresholds = {}, {}, {}
            for row in items:
                level = int(row['level'])
                if level in gt:
                    raise RuntimeError('Duplicate test level')
                marker = row['marker']['center'] if row.get('marker') else None
                threshold = marker[0] - 30.0 if marker is not None else None
                thresholds[level] = threshold
                for states, kind in ((gt,'gt'),(pred,method)):
                    peak = row['trajectory_summary'][kind]['peak_center']
                    states[level] = None if threshold is None or peak is None else bool(peak[0] >= threshold)
            complete = set(gt) == set(range(1,11)) and all(x is not None for x in gt.values())
            truth = next((l for l in sorted(gt) if gt[l] is True),None)
            chosen = next((l for l in sorted(pred) if pred[l] is True),None)
            gt_pair = [truth,truth+1] if truth is not None else []
            pred_pair = [chosen,chosen+1] if chosen is not None else []
            score = len(set(gt_pair)&set(pred_pair))/2 if complete and gt_pair else None
            decisions.append(dict(environment=name,gt_first_reach=truth,predicted_first_reach=chosen,
                gt_prefer=gt_pair,predicted_prefer=pred_pair,score=score,gt_complete=complete,
                gt_reaches_by_level=gt,predicted_reaches_by_level=pred,threshold_x_by_level=thresholds,
                unknown_prediction_levels=[l for l,v in pred.items() if v is None],
                level11_is_formal_successor_only=11 in gt_pair or 11 in pred_pair))
        actions.append(dict(task='ball',method=method,rule='first_reach_and_successor_pair_overlap',
            environment_count=len(decisions),scorable_count=sum(d['score'] is not None for d in decisions),
            macro_score=engine.mean([d['score'] for d in decisions]),
            conservative_score_fixed_denominator=sum(d['score'] or 0 for d in decisions)/len(decisions) if decisions else None,
            decisions=decisions))
    report['action'] = actions
    engine.save_json(output/'summary_provisional.json',report)
    engine.save_json(output/'ball_action_first_reach_successor_pair_overlap.json',actions)
    return report


engine.freeze_manifest = freeze_manifest
engine.aggregate = aggregate
if __name__ == '__main__':
    engine.main()
