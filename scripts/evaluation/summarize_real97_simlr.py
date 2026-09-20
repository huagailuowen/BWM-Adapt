#!/usr/bin/env python3
"""Standalone sim-LR results; never publish over the approved main table."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / 'results/real97_all_methods_main_table_v1'
OUT = ROOT / 'outputs/eval_real97_simlr_summary_20260918_v1'
SOFT = ROOT / 'outputs/eval_real97_soft_simlr_20260918_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def mean(values):
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else None


def soft_lpips():
    # Import only helpers; do not call the legacy publisher.
    import score_real97_soft_final_lpips as lp
    lp.torch.set_num_threads(4)
    lp.cv2.setNumThreads(1)
    model = lp.lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval().requires_grad_(False)
    records = read(SOFT / 'per_query.json')
    source = ROOT / 'outputs/infer_real97_soft_5500_simlr_family_20260918_v1'
    results = []
    for row in records:
        destination = SOFT / 'lpips' / (row['key'] + '.json')
        if destination.is_file():
            results.append(read(destination))
            continue
        indices = [i for i in row['row']['evaluation_frame_indices'] if i <= 28]
        gt = lp.frames(source / 'raw/gt' / (row['key'] + '.mp4'), indices)
        pred = lp.frames(source / 'raw/stage2' / (row['key'] + '.mp4'), indices)
        values = []
        with lp.torch.inference_mode():
            for start in range(0, len(indices), 8):
                values.extend(model(gt[start:start+8], pred[start:start+8]).flatten().cpu().tolist())
        result = dict(key=row['key'], environment=row['environment'], split=row['split'],
                      mean=mean(values), per_frame=values, indices=indices)
        write(destination, result)
        results.append(result)
        print('[soft_lpips]', row['key'], result['mean'], flush=True)
    write(SOFT / 'lpips_complete.json', dict(queries=len(results), records=results,
        protocol='Alex v0.1, 512x256 INTER_LINEAR, future native3..84, no tracking mask'))


def onset_scores(cohort):
    path = TABLE / 'soft_sliding_onset_exploratory_v1/compute.py'
    spec = importlib.util.spec_from_file_location('frozen_onset_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {r['key']: r for r in read(path.parent / 'per_query.json')}
    records = []
    for selected in cohort:
        key = selected['key']
        old = original[key]
        gt = old['thresholds']['8']['events']['gt']
        new_gt = read(SOFT / 'cases' / key / 'gt_tracks.json')['tracks']
        gt_event = module.onset(new_gt, 8)['event']
        expected = gt['event']
        if ((gt_event is None) != (expected is None) or
                (gt_event is not None and gt_event['native_frame'] != expected['native_frame'])):
            raise RuntimeError('GT onset changed; do not silently change the fixed denominator: ' + key)
        tracks = read(SOFT / 'cases' / key / 'stage2_tracks.json')['tracks']
        if [t['native_frame'] for t in tracks] != list(range(0, 85, 3)):
            raise RuntimeError('Onset timestamps differ from frozen evaluation')
        pred = module.onset(tracks, 8)
        event = pred['event']
        error = None
        if expected and event:
            angles = old['rotation']['accumulated_rotation_deg']
            error = abs(angles[event['native_frame']] - expected['cumulative_angle_deg'])
        records.append(dict(key=key, gt_positive=expected is not None,
            angle_error_deg=error, predicted_event=event,
            success6=bool(error is not None and error <= 6) if expected else None,
            missed=bool(expected and event is None),
            false_positive=bool(expected is None and gt['status']=='no_detected_onset' and event)))
    positives = [r for r in records if r['gt_positive']]
    if len(positives) != 11:
        raise RuntimeError('Expected frozen eleven GT-positive queries')
    result = dict(queries=12, positives=11, success_count=sum(r['success6'] for r in positives),
        success_pct=100*sum(r['success6'] for r in positives)/11,
        matched_angle_mae_deg=mean([r['angle_error_deg'] for r in positives]),
        missed=sum(r['missed'] for r in positives), false_positives=sum(r['false_positive'] for r in records),
        protocol='Fixed >8px x displacement for three frames; <=6deg cumulative target rotation error',
        threshold_selection='previously selected post hoc; unchanged for this experiment', records=records)
    write(OUT / 'soft_onset6.json', result)
    return result


def summarize():
    rows = []
    for task in ('door', 'ball'):
        report = read(ROOT / f'outputs/eval_real97_{task}_simlr_20260918_v1/summary_provisional.json')
        image = next(r for r in report['image_object'] if r['split']=='test' and r['method']=='ours_stage2')
        action = next(r for r in report['action'] if r['method']=='ours_stage2')
        m = image['query_mean']
        rows.append(dict(task=task, model_step=6500, queries=image['queries'],
            psnr=m.get('psnr'), ssim=m.get('ssim'), lpips=m.get('lpips'),
            ade_px=m.get('ade_px'), fde_px=m.get('fde_px'),
            action_pct=100*action['macro_score'] if action['macro_score'] is not None else None,
            action_denominator=action['scorable_count'], action_metric='preferred_level_pair_overlap'))
    stick_root = ROOT / 'outputs/eval_real916_stick_simlr_visible_tail_20260918_v1'
    stick = read(stick_root / 'summary.json')['groups']['test']['ours_stage2']
    selected = read(TABLE / 'metrics/stick_action_seed20260927.json')['selected_run']['selection']
    keys = {key for env in selected for key in env['keys']}
    cases = [r for r in read(stick_root / 'per_query.json') if r['key'] in keys]
    if len(cases) != 36 or len(keys) != 36:
        raise RuntimeError('Stick fixed subset incomplete')
    predicted = [r for r in cases if r['action']['ours_stage2']['5']['state']=='positive']
    hits = sum(r['dataset_outcome_for_audit']=='balanced' for r in predicted)
    action = dict(subset_seed=20260927, queries=36, predicted_balanced=len(predicted),
        true_balanced=hits, precision=hits/len(predicted) if predicted else None,
        unknown_predictions=sum(r['action']['ours_stage2']['5']['state']=='unknown' for r in cases),
        gt_source='unchanged dataset annotations, as in the frozen seed20260927 main-table protocol',
        subset_status='previously post-hoc selected; no reselection', keys=sorted(keys))
    write(OUT / 'stick_action_fixed36.json', action)
    lp_path = ROOT / 'outputs/eval_real916_stick_simlr_matched_20260918_v1/summary.json'
    lp = read(lp_path)['groups']['test']['ours_stage2'].get('lpips')
    rows.append(dict(task='stick', model_step=3900, queries=45, psnr=stick['psnr'], ssim=stick['ssim'],
        lpips=lp, ade_px=stick['ade_px'], fde_px=stick['fde_px'],
        action_pct=100*action['precision'] if action['precision'] is not None else None,
        action_denominator=len(predicted), action_metric='precision_among_predicted_balanced_fixed36'))
    cohort = read(TABLE / 'soft_shared6/per_query.json')
    soft_keys = {r['key'] for r in cohort}
    soft = [r for r in read(SOFT / 'per_query.json') if r['key'] in soft_keys]
    if len(soft) != 12:
        raise RuntimeError('Soft shared6 test12 incomplete')
    onset = onset_scores(cohort)
    lp_path = SOFT / 'lpips_complete.json'
    lp = mean([r['mean'] for r in read(lp_path)['records'] if r['key'] in soft_keys]) if lp_path.exists() else None
    rows.append(dict(task='soft', model_step=5500, queries=12,
        psnr=mean([r['image_metrics']['stage2']['psnr_db'] for r in soft]),
        ssim=mean([r['image_metrics']['stage2']['ssim'] for r in soft]), lpips=lp,
        ade_px=mean([r['common_mask_metrics']['stage2']['center_ade_px'] for r in soft]),
        fde_px=mean([r['metrics']['stage2']['center_fde_px'] for r in soft]),
        action_pct=onset['success_pct'], action_denominator=11, action_metric='sliding_onset_success6deg'))
    write(OUT / 'summary.json', dict(rows=rows, formal_table_modified=False,
        lr_schedule='3:10,1.5:10,0.5:10,0.15:10', roi=False,
        tracking_review_status='automatic_scoring_not_newly_manually_audited',
        notes=['Report tracking coverage alongside ADE/FDE; never treat missing tracks as zero error.',
               'Door/Ball use dual6500, not the old nondual main-table checkpoints.',
               'Stick also changes initialization to all-latent mean; not an isolated LR-only comparison.',
               'Soft common-mask ADE uses GT/Standard/Stage1/new Stage2 valid frames.',
               'Soft6deg and Stick36 cohorts retain their disclosed post-hoc selection history.']))
    with (OUT / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Real-world sim-LR inference scores', '',
        'Stage2 schedule: 3 / 1.5 / 0.5 / 0.15, ten steps each; no ROI.',
        'Separate experiment; original formal tables unchanged. Automated tracking, pending visual audit.', '',
        '| Task | Test queries | PSNR | SSIM | LPIPS | ADE px | FDE px | Action % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda v: 'pending/undefined' if v is None else f'{v:.4f}'
        lines.append('| '+r['task']+' | '+str(r['queries'])+' | '+' | '.join(fmt(r[k]) for k in
                     ('psnr','ssim','lpips','ade_px','fde_px','action_pct'))+' |')
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--soft-lpips', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run scoring on an allocated compute node')
    if args.soft_lpips:
        soft_lpips()
    summarize()
