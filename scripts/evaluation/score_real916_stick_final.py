#!/usr/bin/env python3
"""Matched 33-frame Stick image, motion and terminal-action measurements."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real916_stick_final_common import read_rows, write_json
from score_real97_stick_paired_lighting_cpu import track_video, angle_error, identity
from score_real97_stick_action_precision_cpu import video_decision, native_reference, aggregate

METHODS = ('standard', 'ours_stage1', 'ours_stage2')


def read(path):
    return json.loads(Path(path).read_text())


def average(values):
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else None


def decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (192, 256):
                raise RuntimeError(f'Unexpected frame geometry: {path}')
            frames.append(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR))
    finally:
        cap.release()
    if len(frames) != 33:
        raise RuntimeError(f'Expected 33 frames: {path}, got {len(frames)}')
    return frames


def setup(config, output):
    ours, standard, prepared = [ROOT / config[k] for k in ('ours', 'standard', 'prepared')]
    for run in (ours, standard):
        complete = read(run / 'inference_complete.json')
        if complete['queries'] != 63 or complete['test_queries'] != 45 or complete['environments'] != 9:
            raise RuntimeError('Inference cohort is incomplete')
    plan = read(prepared / 'plan.json')
    source = Path(plan['source_dataset'])
    annotations = {(r['environment'], int(r['episode_index'])): r
                   for r in read_rows(source / 'episode_split_manifest.jsonl')}
    query_rows = read_rows(prepared / 'query.jsonl')
    geometry = {**read(ROOT / config['geometry_config']), **read(ROOT / config['fusion_config'])}
    contact = {**read(ROOT / config['contact_config']), 'angle_thresholds_deg': config['angle_thresholds_deg']}
    tasks = []
    for index, row in enumerate(query_rows):
        key = f"q{index:04d}_{row['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"
        a, b = [read(run / 'completed' / (key + '.json')) for run in (ours, standard)]
        if identity(a['row']) != identity(b['row']) or identity(a['row']) != identity(row):
            raise RuntimeError(f'Unpaired query: {key}')
        paths = {'gt': a['paths']['gt'], 'ours_stage1': a['paths']['stage1'],
                 'ours_stage2': a['paths']['stage2'], 'standard': b['paths']['standard']}
        wanted = [i for i in range(1, 33) if row['start_frame'] + 3*i <= min(row['end_frame'], row['total_frames']-1)]
        if wanted != row['evaluation_frame_indices'] or not wanted:
            raise RuntimeError('Real future-frame mask mismatch')
        annotation = annotations[(row['environment'], int(row['episode_index']))]
        if annotation['split'] != row['dataset_split']:
            raise RuntimeError('Annotation split mismatch')
        lift = [s for s in annotation['segments'] if s['label'] == 'lift']
        if len(lift) != 1:
            raise RuntimeError('Ambiguous commanded lift interval')
        end = int(lift[0]['end_frame_exclusive'])-1
        start = int(row['start_frame'])
        last_native = start + 3*wanted[-1]
        covered = last_native >= end-2
        hi = min(wanted[-1], max(0, math.ceil((end-start)/3)))
        lo = hi - math.ceil(config['hold_seconds'] * 20 / 3 - 1e-8)
        eligible = covered and lo >= 1 and start + hi*3 < row['total_frames']
        phase = {'stride': 3, 'start_frame': start, 'source_fps': 20,
                 'native_reference_frame': max(0, int(annotation['lift_start'])-1),
                 'decision': {'terminal_window': {'frame_interval': [lo,hi]},
                              'horizon_status': 'covered_to_sampling_resolution' if eligible else 'incomplete_lift_horizon'}}
        tasks.append({'key':key, 'index':index, 'row':row, 'paths':paths, 'other_gt':b['paths']['gt'],
                      'wanted':wanted, 'phase':phase, 'action_eligible':eligible,
                      'dataset_outcome_for_audit':annotation.get('outcome'),
                      'native_video':str(source/row['video'][0]), 'output':str(output),
                      'geometry':geometry,'contact':contact})
    if len(tasks) != 63 or sum(t['row']['dataset_split']=='test' for t in tasks) != 45:
        raise RuntimeError('Unexpected paired population')
    return tasks


def signature(task):
    paths = {**task['paths'], 'other_gt':task['other_gt']}
    return {k: {'path':str(p),'bytes':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns}
            for k,p in paths.items()}


def process(task):
    from skimage.metrics import structural_similarity
    cv2.setNumThreads(1)
    output=Path(task['output']);key=task['key'];result_path=output/'per_query'/(key+'.json')
    sig=signature(task)
    if result_path.exists():
        old=read(result_path)
        if old['input_identity']!=sig:
            raise RuntimeError('A scored video changed: '+key)
        return old
    frames={k:decode(p) for k,p in task['paths'].items()}
    other=decode(task['other_gt'])
    gt_difference=max(float(np.mean(np.abs(a.astype(float)-b.astype(float)))) for a,b in zip(frames['gt'],other))
    if gt_difference>1.0:
        raise RuntimeError('GT differs across methods: '+key)
    tracks={}
    for method,video in frames.items():
        cv2.setRNGSeed(73019)
        tracks[method]=track_video(frames['gt'][0],video)
    wanted=task['wanted'];common=[i for i in wanted if all(t['flow'][i]['valid'] for t in tracks.values())]
    result={'key':key,'index':task['index'],'environment':task['row']['environment'],'split':task['row']['dataset_split'],
            'episode_index':task['row']['episode_index'],'wanted':wanted,'common_indices':common,
            'common_coverage':len(common)/len(wanted),'input_identity':sig,'gt_codec_difference':gt_difference,
            'method_coverage':{m:sum(t['flow'][i]['valid'] for i in wanted)/len(wanted) for m,t in tracks.items()},
            'image':{},'object':{},'action':{},'action_eligible':task['action_eligible'],
            'phase':task['phase'],'dataset_outcome_for_audit':task['dataset_outcome_for_audit']}
    for method in METHODS:
        psnr=[];ssim=[]
        for i in wanted:
            gt,pred=frames['gt'][i],frames[method][i]
            mse=float(np.mean(((gt.astype(np.float64)-pred)/255.)**2))
            psnr.append(float(-10*np.log10(max(mse,1e-12))))
            ssim.append(float(structural_similarity(gt,pred,channel_axis=-1,data_range=255,
                                                   gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
        result['image'][method]={'psnr':average(psnr),'ssim':average(ssim),'per_frame_psnr':psnr,'per_frame_ssim':ssim}
        scores={k:None for k in ['ade_px','fde_px','left_ade_px','right_ade_px','midpoint_ade_px','angle_mae_deg','final_angle_error_deg']}
        if common:
            gt=np.asarray([tracks['gt']['flow'][i]['centers'] for i in common])
            pred=np.asarray([tracks[method]['flow'][i]['centers'] for i in common])
            error=np.linalg.norm(pred-gt,axis=-1)
            angular=[angle_error(tracks[method]['flow'][i]['angle_deg'],tracks['gt']['flow'][i]['angle_deg']) for i in common]
            scores.update(ade_px=float(error.mean()),left_ade_px=float(error[:,0].mean()),right_ade_px=float(error[:,1].mean()),
                          midpoint_ade_px=float(np.linalg.norm(pred.mean(1)-gt.mean(1),axis=-1).mean()),angle_mae_deg=average(angular))
            if wanted[-1] in common:
                j=common.index(wanted[-1]);scores.update(fde_px=float(error[j].mean()),final_angle_error_deg=float(angular[j]))
        result['object'][method]=scores
    contact_audits={}
    if task['action_eligible']:
        reference=native_reference(task['native_video'],task['phase']['native_reference_frame'])
        for method,path in task['paths'].items():
            record,_,_=video_decision(path,reference,task['phase'],task['contact'],task['geometry'])
            result['action'][method]=record['decision']['variants']
            contact_audits[method]=record
    else:
        for method in task['paths']:
            result['action'][method]={str(t):{'state':'unknown','reason':'incomplete_lift_horizon'} for t in task['contact']['angle_thresholds_deg']}
    write_json(output/'tracks'/(key+'.json'),tracks)
    write_json(output/'contact'/(key+'.json'),contact_audits)
    times=sorted(set([0,wanted[len(wanted)//2],wanted[-1]]))
    panel=np.full((len(times)*270,4*320,3),245,np.uint8)
    for r,i in enumerate(times):
        for c,method in enumerate(('gt',)+METHODS):
            frame=frames[method][i].copy();state=tracks[method]['flow'][i]
            for side,point in enumerate(state['centers']):
                if state['side_valid'][side]:cv2.circle(frame,tuple(np.rint(point).astype(int)),7,(0,0,255) if side==0 else (255,0,0),2)
            x,y=c*320,r*270;panel[y+30:y+270,x:x+320]=cv2.resize(frame,(320,240))
            label=f"{method} f{i} flow={state['valid']} action5={result['action'][method]['5']['state']}"
            cv2.putText(panel,label,(x+3,y+18),0,.32,(25,25,25),1)
    review=output/'reviews'/(key+'.jpg');review.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(review),panel):raise RuntimeError('Could not write review')
    result['review']=str(review);write_json(result_path,result)
    return result


def summarize(records):
    groups={}
    for name,chosen in [('test',[r for r in records if r['split']=='test']),('train',[r for r in records if r['split']=='train'])]+[
            ('test/'+e,[r for r in records if r['split']=='test' and r['environment']==e]) for e in sorted({r['environment'] for r in records})]:
        methods={}
        for method in METHODS:
            metrics={}
            for section in ('image','object'):
                for key in chosen[0][section][method]:
                    if key.startswith('per_frame'):continue
                    vals=[r[section][method][key] for r in chosen]
                    metrics[key]=average(vals);metrics[key+'_queries']=sum(v is not None for v in vals)
            metrics.update(queries=len(chosen),mean_tracking_coverage=average([r['method_coverage'][method] for r in chosen]),
                           common_tracking_coverage=average([r['common_coverage'] for r in chosen]),
                           action_eligible_queries=sum(r['action_eligible'] for r in chosen))
            action_rows=[{'methods':r['action']} for r in chosen]
            metrics['action_precision']={str(t):aggregate(action_rows,method,str(t)) for t in (3,5)}
            # Dataset outcome annotations are a diagnostic cross-check only;
            # do not substitute them silently for the common video classifier.
            confusion={}
            for r in chosen:
                label=f"{r['dataset_outcome_for_audit']} -> {r['action']['gt']['5']['state']}"
                confusion[label]=confusion.get(label,0)+1
            metrics['gt_annotation_diagnostic']=confusion
            methods[method]=metrics
        groups[name]=methods
    return {'complete':True,'formal_metric_approved':False,'visual_audit_pending':True,'groups':groups,
            'primary_split':'test','primary_action_threshold_deg':5,'lpips_complete':False,
            'notes':['Missing tracking remains missing; no last-valid-frame FDE substitution.',
                     'Action success is terminal precision; zero predicted positives is undefined.',
                     'Unsupported lift horizons are unknown, not failures.']}


def lpips_scores(tasks, output):
    import torch
    import lpips
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('LPIPS needs one allocated GPU')
    summary=read(output/'summary.json')
    if not summary['complete']:raise RuntimeError('CPU scores incomplete')
    torch.set_num_threads(4)
    metric=lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval().requires_grad_(False)
    records=[]
    for task in tasks:
        sig=signature(task);path=output/'lpips'/(task['key']+'.json')
        if path.exists():
            record=read(path)
            if record['input_identity']!=sig:raise RuntimeError('LPIPS input changed')
        else:
            def tensor(kind):
                video=decode(task['paths'][kind]);rgb=np.stack([cv2.cvtColor(video[i],cv2.COLOR_BGR2RGB) for i in task['wanted']])
                return torch.from_numpy(rgb.transpose(0,3,1,2).copy()).cuda().float()/127.5-1
            gt=tensor('gt');scores={}
            for method in METHODS:
                pred=tensor(method);values=[]
                with torch.inference_mode():
                    for i in range(0,len(gt),8):values.extend(metric(gt[i:i+8],pred[i:i+8]).flatten().cpu().tolist())
                scores[method]={'mean':average(values),'per_frame':values}
            record={'key':task['key'],'environment':task['row']['environment'],'split':task['row']['dataset_split'],
                    'scores':scores,'input_identity':sig,'wanted':task['wanted']}
            write_json(path,record)
        records.append(record);print('[lpips]',len(records),'/63',task['key'],flush=True)
    for group,methods in summary['groups'].items():
        split=group.split('/')[0];env=group.split('/',1)[1] if '/' in group else None
        selected=[r for r in records if r['split']==split and (env is None or r['environment']==env)]
        for method in METHODS:
            methods[method]['lpips']=average([r['scores'][method]['mean'] for r in selected])
            methods[method]['lpips_queries']=len(selected)
    summary.update(lpips_complete=True,lpips_protocol='AlexNet v0.1, RGB [-1,1], 640x480; all real future frames, no tracking mask')
    write_json(output/'lpips_per_query.json',records);write_json(output/'summary.json',summary)
    write_json(output/'lpips_complete.json',{'queries':len(records)})
    export_table(summary,output)


def export_table(summary, output):
    rows=[]
    for group,methods in summary['groups'].items():
        for method,metrics in methods.items():
            rows.append({'group':group,'method':method,**{k:metrics.get(k) for k in ['queries','psnr','ssim','lpips','ade_px','fde_px','angle_mae_deg','common_tracking_coverage','action_eligible_queries']},
                         'action_precision':metrics['action_precision']['5']['precision'],
                         'predicted_balanced':metrics['action_precision']['5']['predicted_balanced'],
                         'action_precision_lower':metrics['action_precision']['5']['precision_lower_bound'],
                         'action_precision_upper':metrics['action_precision']['5']['precision_upper_bound']})
    with (output/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--mode',choices=['cpu','lpips'],required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    os.chdir(ROOT);config=read(args.config);output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/('.'+args.mode+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=setup(config,output)
    protocol={**config,'query_count':63,'test_queries':45,'train_queries':18,
              'model_steps':{'ours':3900,'standard':4559},'coordinate_system':'640x480 original-image pixels',
              'object_reference':'common GT chunk initial frame; material-marker LK flow with left/right half constraints',
              'object_common_mask':'GT, Standard, Stage1, Stage2 all valid; no fill/interpolation',
              'psnr_ssim':'640x480 INTER_LINEAR; SSIM Gaussian sigma1.5 population covariance; PSNR MSE floor1e-12',
              'fdes':'exact last real future frame, not earlier last detected frame',
              'action_reference':'shared native pre-lift frame; boundary-safe terminal footpoints and fused angle',
              'temporal_policy':'exclude initial frame and repeated padding; commanded lift end within stride-1 tolerance',
              'query_identity':[{'key':t['key'],'row':t['row']} for t in tasks]}
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and read(protocol_path)!=protocol:raise RuntimeError('Refusing changed score protocol')
    write_json(protocol_path,protocol)
    if args.mode=='lpips':
        lpips_scores(tasks,output);return
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process,tasks):
            records.append(result);print('[scored]',len(records),'/63',result['key'],flush=True)
    summary=summarize(records);write_json(output/'per_query.json',records);write_json(output/'summary.json',summary)
    export_table(summary,output);write_json(output/'cpu_complete.json',{'queries':len(records)})
    print(json.dumps(summary['groups']['test']),flush=True)


if __name__=='__main__':main()
