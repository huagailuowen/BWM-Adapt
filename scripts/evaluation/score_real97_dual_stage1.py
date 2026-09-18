#!/usr/bin/env python3
"""Score an explicit dual checkpoint's Stage1 against the published Ours row."""
import argparse
from pathlib import Path
import score_real97_ball_door as engine

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', required=True)
args, _ = parser.parse_known_args()
config = engine.read_json(engine.ROOT / args.config)
task, = config['tasks']
if task == 'ball':
    import score_real97_ball_dual6000  # Retain the exact first-reach pair metric.
engine.METHODS = ('ours_stage1',)
base_aggregate = engine.aggregate


def freeze_manifest(config, output):
    path = output / 'manifest.json'
    if path.exists():
        manifest = engine.read_json(path)
        if manifest['config'] != config:
            raise RuntimeError('Changed score configuration')
        return manifest
    baseline = engine.read_json(engine.ROOT / config['baseline_scores'] / 'manifest.json')
    run = engine.ROOT / config['tasks'][task]['ours']
    complete = engine.read_json(run / 'inference_complete.json')
    step = config['expected_checkpoint_step']
    if (complete['model_step'], complete['table_step']) != (step, step):
        raise RuntimeError('Wrong model/table pair')
    indexed = {}
    for marker in sorted((run / 'completed').glob('q*.json')):
        record = engine.read_json(marker)
        key = record['row']['sample_id']
        if key in indexed:
            raise RuntimeError('Duplicate query')
        indexed[key] = record, marker
    reference = [q for q in baseline['queries'] if q['task'] == task]
    if set(indexed) != {q['row']['sample_id'] for q in reference} or complete['queries'] != len(reference):
        raise RuntimeError('Query population differs from formal evaluation')
    queries = []
    for q in reference:
        record, marker = indexed[q['row']['sample_id']]
        for field in ('environment','dataset_split','episode_index','action_level','video','action',
                      'start_frame','end_frame','native_frame_indices','evaluation_frame_indices','length'):
            if record['row'].get(field) != q['row'].get(field):
                raise RuntimeError('Query mismatch: '+field)
        paths = {'gt': record['paths']['gt'], 'ours_stage1': record['paths']['stage1']}
        if not all(Path(p).is_file() for p in paths.values()):
            raise RuntimeError('Missing raw video')
        queries.append({**q, 'paths': paths, 'source_markers': {'ours': str(marker)}})
    manifest = {'config': config, 'queries': queries, 'methods': list(engine.METHODS),
                'paired_gt_only': True, 'checkpoint_steps': {task: {'ours_stage1': step}},
                'fairness_note': 'Identical factual query identities and calibration. Dual Stage1 uses known-environment training Z; published non-dual Ours uses Stage2 adaptation. Different training budget.'}
    engine.save_json(path, manifest)
    return manifest


def aggregate(output, manifest):
    report = base_aggregate(output, manifest)
    rows = [r for r in report['image_object'] if r['split']=='test' and r['method']=='ours_stage1']
    actions = [r for r in report['action'] if r['method']=='ours_stage1']
    if rows and actions:
        current = {k: rows[0]['environment_macro'].get(k) for k in ('psnr','ssim','lpips','ade_px','fde_px')}
        current['action_pair_overlap_pct'] = (actions[0]['macro_score'] * 100
                                               if actions[0]['macro_score'] is not None else None)
        engine.save_json(output / 'comparison_with_formal_ours.json', {
            'task': task, 'dual_stage1_step': manifest['config']['expected_checkpoint_step'],
            'dual_stage1': current, 'formal_nondual_ours_stage2': manifest['config']['formal_reference'],
            'queries_scored': report['queries_scored'], 'queries_expected': report['queries_expected'],
            'test_queries': rows[0]['queries'], 'valid_query_counts': rows[0]['valid_query_counts'],
            'fairness_note': manifest['fairness_note'], 'tracking_status': report['status'],
            'formal_results_unchanged': True})
    return report


engine.freeze_manifest = freeze_manifest
engine.aggregate = aggregate
if __name__ == '__main__':
    engine.main()
