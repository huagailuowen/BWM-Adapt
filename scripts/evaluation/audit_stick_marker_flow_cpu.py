#!/usr/bin/env python3
"""CPU-only marker optical-flow experiment, retaining absolute+ratio criteria."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
from scripts.evaluation.audit_stick_training_action_precision import read_frames, initial_frame
from scripts.evaluation.audit_stick_flow_ratio_absolute_cpu import measure, classify, summarize, plot
from wan_video_action.real97_eval.stick_marker_flow import MarkerFlowPair

BASE=ROOT/'outputs/evaluation_stick_fine_contact_job114006'
SELECTED={'full_stick-L0-R0_ep000000','full_stick-L0-R0_ep000006','full_stick-L0-R0_ep000020',
          'full_stick-L0-R3right_ep000017','q0021_stick-L0-R4right_gt','q0021_stick-L0-R4right_stage2'}


def analyze(task):
    item,reference,output,config=task
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260910)
    frames,fps=read_frames(item['video'])
    old=[json.loads(line) for line in (BASE/'cases'/item['key']/'corner_tracks.jsonl').read_text().splitlines() if line]
    if len(old)!=len(frames): raise ValueError('Frozen video length changed')
    tracker=MarkerFlowPair(initial_frame(reference),config)
    rows=[]; observations=[]; states=[]; counts=Counter()
    for index,frame in enumerate(frames):
        row=tracker.update(frame)
        row.update(frame=index,time_seconds=old[index]['time_seconds'],evaluated=old[index]['evaluated'])
        observation=measure(row)
        state,ratio=classify(observation,config['absolute_min_px'],config['minimum_rise_ratio'],angle=row['angle_change_deg'])
        row.update(state=state,rise_ratio=ratio)
        rows.append(row); observations.append(observation); states.append(state)
        if row['evaluated']:
            counts['frames']+=1; counts[state]+=1
            for side in row['sides']:
                counts['sides']+=1
                counts['valid_flow' if side.get('flow_valid') else side.get('reason','missing')]+=1
    decision=summarize(rows,states,config['hold_seconds'])
    directory=Path(output)/'cases'/item['key']; directory.mkdir(parents=True)
    with (directory/'marker_tracks.jsonl').open('w') as handle:
        for row in rows: handle.write(json.dumps(row)+'\n')
    result={'key':item['key'],'environment':item['environment'],'episode_index':item['episode_index'],
            'scope':item['scope'],'method':item['method'],'video':item['video'],
            'decision':decision,'counts':dict(counts),'manual_reviewed':False}
    (directory/'summary.json').write_text(json.dumps(result,indent=2))
    if item['key'] in SELECTED: plot(rows,observations,item['key'],directory/'marker_rise_ratio_angle.svg')
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU compute node required')
    config=json.loads((ROOT/'configs/evaluation/stick_marker_flow_ratio_v1.json').read_text())
    output=ROOT/'outputs'/('evaluation_stick_marker_flow_job'+os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    (output/'config_snapshot.json').write_text(json.dumps(config,indent=2))
    items=json.loads((BASE/'per_video.json').read_text())
    refs={(r['environment'],r['episode_index'],r['scope']):r['video'] for r in items if r['method']=='gt'}
    tasks=[(item,refs[(item['environment'],item['episode_index'],item['scope'])],str(output),config) for item in items]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(analyze,tasks,chunksize=1):
            results.append(result)
            if len(results)%10==0:
                progress={'completed':len(results),'expected':len(items)}
                (output/'progress.json').write_text(json.dumps(progress)); print(json.dumps(progress),flush=True)
    groups={}
    for result in results:
        group=groups.setdefault(result['scope']+'/'+result['method'],{'decisions':Counter(),'frames':Counter()})
        group['decisions'][result['decision']['status']]+=1
        group['frames'].update(result['counts'])
    report={'status':'marker_motion_proxy_not_human_validated','videos':len(results),'groups':groups,
            'selected_cases':[r for r in results if r['key'] in SELECTED],
            'formal_results_published':False,'training_modified':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    (output/'per_video.json').write_text(json.dumps(results,indent=2))
    page=['<!doctype html><meta charset="utf-8"><h1>Marker optical flow, absolute rise and ratio</h1>',
          '<p>Experimental marker-motion proxy with rotation uncertainty. Formal visual audit not yet performed.</p>']
    for key in sorted(SELECTED):
        rel='cases/'+key+'/marker_rise_ratio_angle.svg'
        page.append(f'<h2>{key}</h2><a href="{rel}"><img width="1060" src="{rel}"></a>')
    (output/'index.html').write_text('\n'.join(page))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
