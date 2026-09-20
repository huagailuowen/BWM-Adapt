#!/usr/bin/env python3
"""Score new single-frame Soft Standard/LoRA against the selected Ours run."""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for site in (ROOT/'.venv-real97-eval-20260909/lib').glob('python*/site-packages'):
    sys.path.append(str(site))
import cv2
import numpy as np
import score_real97_soft_balanced9_static_matched as base

OUT = ROOT/'outputs/eval_real97_soft_static9_standard_lora_20260918_v1'
OLD = ROOT/'outputs/eval_real97_soft_simlr_20260918_v1'
RUNS = {
    'standard': ROOT/'outputs/infer_real97_soft_static9_standard_final118326_v1',
    'lora': ROOT/'outputs/infer_real97_soft_static9_lora_final118326_v1',
    'ours': ROOT/'outputs/infer_real97_soft_5500_simlr_family_20260918_v1',
}
TABLE = ROOT/'results/real97_all_methods_main_table_v1'
read, write = base.read, base.write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def setup():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated compute node')
    cv2.setNumThreads(1)
    rows=base.legacy.jsonl(RUNS['ours']/'query.jsonl')
    for method in ('standard','lora'):
        done=read(RUNS[method]/'inference_complete.json')
        if (done['model_step'],done['queries'],done['test_queries'],done['observed_frames'])!=(5148,36,18,1):
            raise RuntimeError('Wrong new Standard/LoRA checkpoint or observation contract')
        if base.legacy.jsonl(RUNS[method]/'query.jsonl')!=rows:
            raise RuntimeError('Do not mix different factual queries')
    frozen=read(OLD/'shard00/config.json')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(runs={k:str(v) for k,v in RUNS.items()},checkpoints=dict(standard=5148,lora=5148,ours=5500),
        future_native_frames=list(range(3,85,3)),observed_frames=1,tracking=frozen['tracking'],
        detector=frozen['detector'],filtering=frozen['filtering'],
        primary_cohort='fixed shared6 test12; also all9 test18 and all train18',
        object_metric='Visible-box centers; missing detections are not zero errors; exact final-frame FDE',
        roi=False,formal_table_modified=False,manual_review_pending=True)
    p=OUT/'protocol.json'
    # All array shards and the image scorer share this immutable protocol.
    # Serialize its creation; the legacy writer uses one .partial filename.
    with (OUT/'.protocol.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists():
            if read(p)!=protocol:raise RuntimeError('Protocol changed')
        else:
            write(p,protocol)
    return rows,frozen


def case(index,row):
    key=base.legacy.key_for(index,row)
    common=copy.deepcopy(row)
    common.update(length=29,native_frame_indices=row['native_frame_indices'][:29],
                  evaluation_frame_indices=[i for i in row['evaluation_frame_indices'] if i<=28])
    paths={'gt':RUNS['ours']/'raw/gt'/(key+'.mp4'),
           'ours':RUNS['ours']/'raw/stage2'/(key+'.mp4')}
    for kind in ('standard','lora'):
        marker=read(RUNS[kind]/'completed'/(key+'.json'))
        if marker['row']!=row or not marker['paired_gt'] or marker['observed_frames']!=1:
            raise RuntimeError('Wrong completion marker')
        paths[kind]=Path(marker['paths'][kind])
    return key,common,paths


def track(rows,frozen,shard):
    detector=None
    for i,row in enumerate(rows):
        if i%6!=shard:continue
        key,common,paths=case(i,row)
        for method in ('standard','lora'):
            destination=OUT/'cases'/key/(method+'_tracks.json')
            signature=dict(row=common,source=str(paths[method]),sha256=sha(paths[method]),
                tracking=frozen['tracking'],filtering=frozen['filtering'],detector=frozen['detector'])
            if destination.exists():
                if read(destination)['signature']!=signature:raise RuntimeError('Cached tracking source changed')
                continue
            if detector is None:detector=base.legacy.build_detector(frozen['detector'])
            frames=base.decode(paths[method])[:29]
            states=base.track(frames,common,detector,frozen['tracking'],frozen['filtering'])
            write(destination,dict(signature=signature,tracks=states))
            for n,indices in enumerate(((0,5,9,12),(18,21,25,28))):
                image=np.hstack([base.legacy.annotated(frames[j],states[j],method) for j in indices])
                cv2.imwrite(str(destination.parent/f'{method}_review{n}.jpg'),image)
            print('[tracked]',key,method,sum(s['measurement_valid'] for s in states),'/29',flush=True)
    write(OUT/f'track_shard{shard:02d}_complete.json',dict(shard=shard,complete=True))


def images(rows):
    import torch
    import lpips
    if not torch.cuda.is_available():raise RuntimeError('LPIPS requires allocated GPU')
    torch.set_num_threads(4)
    metric=lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval().requires_grad_(False)
    for i,row in enumerate(rows):
        key,common,paths=case(i,row)
        destination=OUT/'cases'/key/'image_metrics.json'
        hashes={k:sha(p) for k,p in paths.items()}
        if destination.exists():
            if read(destination)['video_sha256']!=hashes:raise RuntimeError('Scored video changed')
            continue
        frames={k:base.decode(p)[:29] for k,p in paths.items()}
        for method in ('standard','lora'):
            other=base.decode(RUNS[method]/'raw/gt'/(key+'.mp4'))[:29]
            difference=max(float(np.abs(a.astype(float)-b.astype(float)).mean()) for a,b in zip(frames['gt'],other))
            if difference>3:raise RuntimeError('GT video mismatch: '+key)
        indices=common['evaluation_frame_indices']
        def tensor(kind):
            rgb=np.stack([cv2.cvtColor(frames[kind][j],cv2.COLOR_BGR2RGB) for j in indices])
            return torch.from_numpy(rgb.transpose(0,3,1,2).copy()).cuda().float()/127.5-1
        gt=tensor('gt');scores={}
        for method in RUNS:
            scores[method]=base.image_score(frames['gt'],frames[method],indices)
            pred=tensor(method);values=[]
            with torch.inference_mode():
                for j in range(0,len(indices),8):values.extend(metric(gt[j:j+8],pred[j:j+8]).flatten().cpu().tolist())
            scores[method].update(lpips=mean(values),lpips_per_frame=values)
        write(destination,dict(scores=scores,video_sha256=hashes,
            lpips_protocol='Alex v0.1, RGB [-1,1], native512x256 INTER_LINEAR, future3..84, no tracking mask'))
        print('[image_metrics]',key,json.dumps({m:{k:v[k] for k in ('psnr_db','ssim','lpips')} for m,v in scores.items()}),flush=True)
    write(OUT/'image_metrics_complete.json',dict(queries=len(rows)))


def aggregate(rows,frozen):
    cohort=read(TABLE/'soft_shared6/per_query.json')
    shared_keys={r['key'] for r in cohort}
    onset_path=TABLE/'soft_sliding_onset_exploratory_v1/compute.py'
    spec=importlib.util.spec_from_file_location('onset_protocol',onset_path)
    onset_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(onset_module)
    onset_original={r['key']:r for r in read(onset_path.parent/'per_query.json')}
    records=[]
    for i,row in enumerate(rows):
        key,common,paths=case(i,row)
        tracks={}
        for method,old_kind in (('gt','gt'),('ours','stage2')):
            cached=read(OLD/'cases'/key/(old_kind+'_tracks.json'))
            sig=cached['signature']
            if sig['row']!=common or sig['tracking']!=frozen['tracking'] or sig['filtering']!=frozen['filtering']:
                raise RuntimeError('Reference query/tracker mismatch')
            if sha(sig['source'])!=sha(paths[method]):raise RuntimeError('Reference video changed')
            tracks[method]=cached['tracks']
        for method in ('standard','lora'):
            cached=read(OUT/'cases'/key/(method+'_tracks.json'))
            if cached['signature']['sha256']!=sha(paths[method]):raise RuntimeError('Tracked video changed')
            tracks[method]=cached['tracks']
        wanted=common['evaluation_frame_indices']
        valid=lambda kind,j: tracks[kind][j]['measurement_valid'] and tracks[kind][j]['center'] is not None
        indices=[j for j in wanted if all(valid(kind,j) for kind in tracks)]
        masked={**common,'evaluation_frame_indices':indices}
        scores={}
        for method in RUNS:
            natural=base.legacy.score(tracks['gt'],tracks[method],common,frozen['config'])
            matched=base.legacy.score(tracks['gt'],tracks[method],masked,frozen['config'])
            if not wanted or wanted[-1] not in indices:matched['center_fde_px']=None
            scores[method]=dict(natural=natural,common_mask=matched)
        onset={}
        if key in shared_keys:
            reference=onset_original[key]
            truth=reference['thresholds']['8']['events']['gt']['event']
            actual=onset_module.onset(tracks['gt'],8)['event']
            if (truth is None)!=(actual is None) or (truth and truth['native_frame']!=actual['native_frame']):
                raise RuntimeError('Frozen GT onset changed')
            angle=reference['rotation']['accumulated_rotation_deg']
            for method in RUNS:
                event=onset_module.onset(tracks[method],8)['event']
                error=abs(angle[event['native_frame']]-truth['cumulative_angle_deg']) if truth and event else None
                onset[method]=dict(gt_positive=truth is not None,angle_error_deg=error,
                    success6=bool(error is not None and error<=6) if truth else None,
                    missed=bool(truth and not event),false_positive=bool(not truth and event),predicted_event=event)
        images=read(OUT/'cases'/key/'image_metrics.json')['scores']
        result=dict(key=key,index=i,environment=row['environment'],split=row['dataset_split'],
                    shared6=key in shared_keys,metrics=scores,image_metrics=images,onset=onset,
                    common_frames=len(indices),requested_frames=len(wanted))
        write(OUT/'cases'/key/'summary.json',result);records.append(result)
    groups={
        'test_shared6':[r for r in records if r['shared6']],
        'test_all9':[r for r in records if r['split']=='test'],
        'train_all9':[r for r in records if r['split']=='train'],
    }
    for env in sorted({r['environment'] for r in records}):groups['test/'+env]=[r for r in records if r['split']=='test' and r['environment']==env]
    report={}
    for group,selected in groups.items():
        values={}
        for method in RUNS:
            values[method]=dict(queries=len(selected),
                psnr=mean([r['image_metrics'][method]['psnr_db'] for r in selected]),
                ssim=mean([r['image_metrics'][method]['ssim'] for r in selected]),
                lpips=mean([r['image_metrics'][method]['lpips'] for r in selected]),
                common_ade_px=mean([r['metrics'][method]['common_mask']['center_ade_px'] for r in selected]),
                natural_ade_px=mean([r['metrics'][method]['natural']['center_ade_px'] for r in selected]),
                fde_px=mean([r['metrics'][method]['natural']['center_fde_px'] for r in selected]),
                common_fde_px=mean([r['metrics'][method]['common_mask']['center_fde_px'] for r in selected]),
                fde_valid_queries=sum(r['metrics'][method]['natural']['center_fde_px'] is not None for r in selected),
                paired_coverage=sum(r['metrics'][method]['natural']['paired_frames'] for r in selected)/sum(r['requested_frames'] for r in selected),
                common_coverage=sum(r['common_frames'] for r in selected)/sum(r['requested_frames'] for r in selected))
            if group=='test_shared6':
                positives=[r['onset'][method] for r in selected if r['onset'][method]['gt_positive']]
                if len(positives)!=11:raise RuntimeError('Frozen onset denominator changed')
                values[method].update(onset_success6_pct=100*sum(r['success6'] for r in positives)/11,
                    onset_successes=sum(r['success6'] for r in positives),onset_denominator=11,
                    onset_angle_mae_matched_deg=mean([r['angle_error_deg'] for r in positives]),
                    onset_missed=sum(r['missed'] for r in positives),
                    onset_false_positives=sum(r['onset'][method]['false_positive'] for r in selected))
        report[group]=values
    write(OUT/'per_query.json',records)
    write(OUT/'summary.json',dict(complete=True,groups=report,formal_table_modified=False,
        manual_review_pending=True,checkpoints=dict(standard=5148,lora=5148,ours=5500),
        notes=['All models use one observed frame; common future3..84 retained for existing benchmark compatibility.',
               'Primary comparison is shared6 test12; report all9 test18 separately.',
               'Common-mask ADE compares GT/newStandard/newLoRA/currentOurs on identical valid frames; old-table masks may differ.',
               'Onset6deg keeps the prior post-hoc threshold and fixed11 positive denominator.']))
    fields=['cohort','method','queries','psnr','ssim','lpips','common_ade_px','natural_ade_px','fde_px','fde_valid_queries','paired_coverage','onset_success6_pct']
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for cohort,methods in report.items():
            for method,scores in methods.items():writer.writerow({k:({'cohort':cohort,'method':method,**scores}).get(k) for k in fields})
    write(OUT/'scoring_complete.json',dict(queries=len(records),test_queries=18,shared6_test_queries=12))
    print(json.dumps({k:report[k] for k in ('test_shared6','test_all9')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['track','images','aggregate'])
    parser.add_argument('--shard',type=int,default=0);args=parser.parse_args()
    rows,frozen=setup()
    if args.mode=='track':track(rows,frozen,args.shard)
    elif args.mode=='images':images(rows)
    else:aggregate(rows,frozen)
