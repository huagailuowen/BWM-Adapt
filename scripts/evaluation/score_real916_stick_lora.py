#!/usr/bin/env python3
"""Paired Stick LoRA/Standard/Ours metrics, using the unchanged visible-tail rule."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path

import score_real916_stick_final as engine
from real916_stick_final_common import ROOT, read_rows, write_json

engine.METHODS = ('standard','ours_stage2','lora')


def read(path):
    return json.loads(Path(path).read_text())


def prepare_tasks(config, output):
    run = ROOT/config['inference_output']
    complete = read(run/'inference_complete.json')
    if (complete['queries'],complete['test_queries'],complete['model_step'])!=(63,45,4559):
        raise RuntimeError('Incomplete/wrong LoRA population')
    prepared = ROOT/config['prepared']
    plan = read(prepared/'plan.json')
    old = {r['key']:r for r in read(ROOT/config['reference_metrics']/'per_query.json')}
    queries = read_rows(prepared/'query.jsonl')
    geometry = {**read(ROOT/config['geometry_config']),**read(ROOT/config['fusion_config'])}
    contact = {**read(ROOT/config['contact_config']),'angle_thresholds_deg':[3,5]}
    tasks = []
    for i,row in enumerate(queries):
        key = f"q{i:04d}_{row['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"
        lora = read(run/'completed'/(key+'.json'))
        standard = read(ROOT/config['standard']/'completed'/(key+'.json'))
        ours = read(ROOT/config['ours']/'completed'/(key+'.json'))
        if any(r['row']!=row for r in (lora,standard,ours)):
            raise RuntimeError('Query identity mismatch')
        reference = old[key]
        phase = copy.deepcopy(reference['phase'])
        hi = row['evaluation_frame_indices'][-1]
        if hi-2<1:raise RuntimeError('Insufficient real terminal frames')
        phase['decision'] = {'terminal_window':{'frame_interval':[hi-2,hi]},'horizon_status':'visible_real_video_tail'}
        tasks.append({'key':key,'index':i,'row':row,
            'paths':{'gt':lora['paths']['gt'],'standard':standard['paths']['standard'],
                     'ours_stage2':ours['paths']['stage2'],'lora':lora['paths']['lora']},
            'other_gt':ours['paths']['gt'],'wanted':row['evaluation_frame_indices'],'phase':phase,
            'action_eligible':True,'dataset_outcome_for_audit':reference['dataset_outcome_for_audit'],
            'native_video':str(Path(plan['source_dataset'])/row['video'][0]),'output':str(output),
            'geometry':geometry,'contact':contact})
    return tasks


def comparison(config, output, records):
    selected = read(ROOT/config['selected_cohort'])['selected_run']
    keys = {k for env in selected['selection'] for k in env['keys']}
    summary = read(output/'summary.json')
    groups = {}
    for name,rows in [('test45',[r for r in records if r['split']=='test']),
                      ('fixed_seed20260927_test36',[r for r in records if r['key'] in keys])]:
        expected = 45 if name=='test45' else 36
        if len(rows)!=expected:raise RuntimeError('Changed cohort')
        groups[name] = {}
        for method in engine.METHODS:
            positive = [r for r in rows if r['action'][method]['5']['state']=='positive']
            if any(r['dataset_outcome_for_audit'] not in ('balanced','left_down','right_down') for r in rows):
                raise RuntimeError('Unknown annotation GT')
            tp = sum(r['dataset_outcome_for_audit']=='balanced' for r in positive)
            actual = sum(r['dataset_outcome_for_audit']=='balanced' for r in rows)
            metrics = {'queries':len(rows),'selected':len(positive),'true_balanced':tp,
                'false_balanced':len(positive)-tp,'precision':tp/len(positive) if positive else None,
                'balanced_recall':tp/actual if actual else None,
                'unknown_predictions':sum(r['action'][method]['5']['state']=='unknown' for r in rows)}
            for field in ('ade_px','fde_px','angle_mae_deg'):
                values = [r['object'][method][field] for r in rows if r['object'][method][field] is not None]
                metrics[field] = sum(values)/len(values) if values else None
                metrics[field+'_valid_queries'] = len(values)
            if name=='test45':
                metrics.update({k:summary['groups']['test'][method].get(k) for k in ('psnr','ssim','lpips')})
            groups[name][method] = metrics
    write_json(output/'comparison.json',{'groups':groups,'gt_source':'dataset outcome labels',
        'action_rule':'precision among predicted balanced at unchanged 5-degree visible-tail criterion',
        'object_mask':'common valid frames among GT/Standard/Ours Stage2/LoRA; differs from historical table mask',
        'post_hoc_seed_subset_secondary':True,'main_table_modified':False,'visual_audit_pending':True,
        'lpips_complete':summary.get('lpips_complete',False)})
    print(json.dumps(groups,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--mode',choices=['cpu','lpips'],required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    config = read(args.config)
    output = ROOT/config['output']
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/('.'+args.mode+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'protocol.json').exists() and read(output/'protocol.json')!=config:
        raise RuntimeError('Score protocol changed')
    write_json(output/'protocol.json',config)
    tasks = prepare_tasks(config,output)
    if args.mode=='cpu':
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            records = list(pool.map(engine.process,tasks))
        summary = engine.summarize(records)
        summary['notes'] = ['Last real 0.3 seconds; no commanded-lift-end eligibility gate.',
            'Annotation-GT action scores are in comparison.json; summary classifier bounds are diagnostic.',
            'Common object tracking mask includes LoRA; predictions from Standard/Ours are unchanged.']
        write_json(output/'per_query.json',records)
        write_json(output/'summary.json',summary)
        engine.export_table(summary,output)
        write_json(output/'cpu_complete.json',{'queries':len(records)})
    else:
        if not (output/'cpu_complete.json').exists():raise RuntimeError('CPU scoring incomplete')
        records = read(output/'per_query.json')
        engine.lpips_scores(tasks,output)
    comparison(config,output,records)


if __name__=='__main__':main()
