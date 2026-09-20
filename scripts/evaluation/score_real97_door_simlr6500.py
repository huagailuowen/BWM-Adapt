#!/usr/bin/env python3
"""Score dual6500 against frozen factual Door queries and unchanged calibration."""
import json
from pathlib import Path
import score_real97_ball_door as engine

engine.METHODS = ('ours_stage1', 'ours_stage2')


def freeze_manifest(config, output):
    path = output / 'manifest.json'
    if path.exists():
        result = engine.read_json(path)
        if result['config'] != config:
            raise RuntimeError('Scoring configuration changed')
        return result
    baseline = engine.read_json(engine.ROOT / config['baseline_scores'] / 'manifest.json')
    run = engine.ROOT / config['tasks']['door']['ours']
    done = engine.read_json(run / 'inference_complete.json')
    if (done['model_step'], done['table_step']) != (6500, 6500):
        raise RuntimeError('Expected paired 6500-step model and table')
    indexed = {}
    for marker in sorted((run / 'completed').glob('q*.json')):
        record = engine.read_json(marker)
        identity = record['row']['sample_id']
        if identity in indexed:
            raise RuntimeError('Duplicate query identity')
        indexed[identity] = (record, marker)
    reference = [q for q in baseline['queries'] if q['task'] == 'door']
    if set(indexed) != {q['row']['sample_id'] for q in reference}:
        raise RuntimeError('New and baseline query sets differ')
    queries = []
    for q in reference:
        record, marker = indexed[q['row']['sample_id']]
        for field in ('environment','dataset_split','episode_index','action_level','video','action',
                      'start_frame','end_frame','native_frame_indices','evaluation_frame_indices','length'):
            if record['row'].get(field) != q['row'].get(field):
                raise RuntimeError('Mismatched factual query: '+field)
        paths = {'gt':record['paths']['gt'], 'ours_stage1':record['paths']['stage1'],
                 'ours_stage2':record['paths']['stage2']}
        if not all(Path(p).is_file() for p in paths.values()):
            raise RuntimeError('Missing prediction')
        queries.append({**q, 'paths':paths, 'source_markers':{'ours':str(marker)}})
    result = dict(config=config, queries=queries, methods=list(engine.METHODS), paired_gt_only=True,
                  checkpoint_steps={'door':{'ours_stage1':6500,'ours_stage2':6500}},
                  fairness_note='Fixed prior queries/calibration. Known training-cluster prior for Stage2; not a blind environment initialization. Excludes counterfactual sweeps.')
    engine.save_json(path,result)
    return result


engine.freeze_manifest = freeze_manifest
if __name__ == '__main__':
    engine.main()
