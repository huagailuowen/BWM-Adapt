#!/usr/bin/env python3
"""Add TTT to frozen Soft scores; recompute a shared frame mask for every method."""
import argparse
import copy
import os
from pathlib import Path
import cv2
import numpy as np
import score_real97_soft_dino_matched as prior

base = prior.base
ROOT = prior.ROOT
RUN = ROOT / 'outputs/infer_real97_soft_ttt_static_reference_after117462_v1'
OUT = ROOT / 'outputs/eval_real97_soft_ttt_matched_20260917_v1'
METHODS = (*prior.METHODS, 'ttt')
SHARED = {'soft-1l','soft-2l','soft-1r','soft-5r','soft-2m','soft-8'}
read, write = prior.read, prior.write


def setup():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a compute allocation')
    cv2.setNumThreads(1)
    rows = base.legacy.jsonl(RUN / 'query.jsonl')
    if rows != base.legacy.jsonl(prior.DINO / 'query.jsonl'):
        raise RuntimeError('TTT and prior cohort mismatch')
    done = read(RUN / 'inference_complete.json')
    if done['queries'] != len(rows) or done['checkpoint_step'] != 3658:
        raise RuntimeError('Incomplete/unexpected TTT inference')
    OUT.mkdir(parents=True,exist_ok=True)
    return rows, read(prior.OLD / 'shard00/config.json')


def track(rows, frozen, shard):
    detect = None
    for index,row in enumerate(rows):
        if index % 6 != shard:
            continue
        key, common, paths = prior.case(index,row)
        source = RUN / 'raw/ttt' / (key+'.mp4')
        destination = OUT / 'cases' / key
        signature = dict(row=common,tracking=frozen['tracking'],filtering=frozen['filtering'],
                         detector=frozen['detector'],source=str(source),source_sha256=prior.sha(source))
        path = destination / 'ttt_tracks.json'
        if path.is_file():
            if read(path)['signature'] != signature:
                raise RuntimeError('TTT inputs changed')
            continue
        if detect is None:
            detect = base.legacy.build_detector(frozen['detector'])
        frames = base.decode(source)[:29]
        gt = base.decode(paths['gt'])[:29]
        exported_gt = base.decode(RUN/'raw/gt'/(key+'.mp4'))[:29]
        if len(frames)!=29 or len(exported_gt)!=29 or np.abs(np.asarray(gt,dtype=float)-np.asarray(exported_gt,dtype=float)).mean()>3:
            raise RuntimeError('TTT GT or frame contract mismatch')
        states = base.track(frames,common,detect,frozen['tracking'],frozen['filtering'])
        write(path,dict(signature=signature,tracks=states))
        print('TRACKED '+key,flush=True)
    write(OUT/f'track_shard{shard:02d}_complete.json',dict(complete=True))


def aggregate(rows,frozen):
    results=[]
    for index,row in enumerate(rows):
        key,common,paths=prior.case(index,row)
        tracks={}
        for kind,old_kind in [('gt','gt'),('standard','standard'),('stage1','stage1'),('ours','stage2')]:
            record=read(prior.OLD/'cases'/key/(old_kind+'_tracks.json'))
            if record['signature']['row']!=common or prior.sha(record['signature']['source'])!=prior.sha(paths[kind]):
                raise RuntimeError('Prior track cohort/source changed')
            tracks[kind]=record['tracks']
        tracks['dino']=read(prior.OUT/'cases'/key/'dino_tracks.json')['tracks']
        paths['ttt']=RUN/'raw/ttt'/(key+'.mp4')
        record=read(OUT/'cases'/key/'ttt_tracks.json')
        if record['signature']['source_sha256']!=prior.sha(paths['ttt']):
            raise RuntimeError('TTT video changed')
        tracks['ttt']=record['tracks']
        wanted=common['evaluation_frame_indices']
        valid=lambda k,i: tracks[k][i]['measurement_valid'] and tracks[k][i]['center'] is not None
        shared=[i for i in wanted if all(valid(k,i) for k in ('gt',)+METHODS)]
        mask=copy.deepcopy(common);mask['evaluation_frame_indices']=shared
        natural,matched={},{}
        for kind in METHODS:
            natural[kind]=base.legacy.score(tracks['gt'],tracks[kind],common,frozen['config'])
            matched[kind]=base.legacy.score(tracks['gt'],tracks[kind],mask,frozen['config'])
            if not wanted or wanted[-1] not in shared:
                matched[kind]['center_fde_px']=None
        result=dict(index=index,key=key,environment=row['environment'],split=row['dataset_split'],
                    requested_frames=len(wanted),common_frames=len(shared),natural=natural,common_mask=matched)
        results.append(result)
        panel=[]
        for kind in ('gt',)+METHODS:
            frames=base.decode(paths[kind]);frames=frames[4:] if kind=='standard' else frames[:29]
            panel.append(np.hstack([base.legacy.annotated(frames[i],tracks[kind][i],kind) for i in (0,9,18,28)]))
        cv2.imwrite(str(OUT/'cases'/key/'all_methods_contact_sheet.jpg'),np.vstack(panel))
    groups={'test_shared6':[r for r in results if r['split']=='test' and r['environment'] in SHARED],
            'test_all9':[r for r in results if r['split']=='test']}
    for name in sorted({r['environment'] for r in results}):
        groups['test/'+name]=[r for r in results if r['split']=='test' and r['environment']==name]
    summary={}
    for name,items in groups.items():
        n=sum(r['requested_frames'] for r in items)
        summary[name]={kind:dict(queries=len(items),
            common_ADE_px=prior.mean([r['common_mask'][kind]['center_ade_px'] for r in items]),
            common_final_FDE_px=prior.mean([r['common_mask'][kind]['center_fde_px'] for r in items]),
            natural_ADE_px=prior.mean([r['natural'][kind]['center_ade_px'] for r in items]),
            natural_FDE_px=prior.mean([r['natural'][kind]['center_fde_px'] for r in items]),
            paired_coverage=sum(r['natural'][kind]['paired_frames'] for r in items)/n,
            common_coverage=sum(r['common_frames'] for r in items)/n,
            final_valid_queries=sum(r['common_mask'][kind]['center_fde_px'] is not None for r in items)) for kind in METHODS}
    write(OUT/'per_query.json',results)
    write(OUT/'summary.json',dict(groups=summary,complete=True,manual_review_required=True,
          no_lpips=True,no_action_metric=True,common_mask_methods=list(METHODS),
          native_future_frames=list(range(3,85,3)),center='visible in-frame center',
          caveats=['TTT trained3658 updates, DINO5500; Standard trained6env and uses5 repeated initial frames; Ours known-family prior.']))
    print(summary['test_shared6'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['track','aggregate']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    rows,frozen=setup()
    if a.mode=='track':track(rows,frozen,a.shard)
    else:aggregate(rows,frozen)
