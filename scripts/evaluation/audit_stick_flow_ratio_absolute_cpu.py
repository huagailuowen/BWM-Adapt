#!/usr/bin/env python3
"""Compare absolute rise plus bilateral rise ratio using frozen optical flow."""

from collections import Counter
from pathlib import Path
import html
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/evaluation_stick_fine_contact_job114006"
CONFIG = ROOT / "configs/evaluation/stick_flow_ratio_absolute_v1.json"


def initial_scale(video):
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (92, 75, 25), (125, 255, 255))
    blue[:140] = 0
    heights = []
    for side in range(2):
        mask = blue.copy()
        if side == 0: mask[:, 256:] = 0
        else: mask[:, :384] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for label in range(1, n):
            x, y, w, h, area = map(int, stats[label])
            if area >= 40 and 12 <= h <= 120 and w >= 3 and h >= .6*w:
                candidates.append((area, h))
        if not candidates: return None
        heights.append(float(np.median([h for _, h in sorted(candidates, reverse=True)[:2]])))
    return heights


def measure(row):
    observations = []
    for side in row.get('sides', []):
        displacement = side.get('corner_displacement_px', [])
        visible = side.get('corner_visible', [])
        band = side.get('uncertainty_margin_px')
        if not side.get('flow_valid') or len(displacement) != 2 or len(visible) != 2 or band is None:
            observations.append(None)
            continue
        if not math.isfinite(band) or not all(map(math.isfinite, displacement)):
            observations.append(None)
            continue
        observations.append({'d': min(displacement), 'band': float(band),
                             'fully_visible': all(visible),
                             'visible_d': min((d for d, v in zip(displacement, visible) if v), default=None)})
    while len(observations) < 2: observations.append(None)
    return observations


def classify(observations, absolute, ratio_min, scales=None, require_scale=False, angle=None, angle_guard=False):
    if any(value is None or not value['fully_visible'] for value in observations):
        # Reliable visible contact at one end still rules out this frame's lift.
        contact = any(value is not None and value['visible_d'] is not None
                      and abs(value['visible_d']) <= value['band'] for value in observations)
        return ('below_absolute_or_noise' if contact else 'unknown'), None
    d = [value['d'] for value in observations]
    if any(value['d'] < absolute or value['d'] <= value['band'] for value in observations):
        return 'below_absolute_or_noise', None
    if require_scale and scales is None: return 'unknown', None
    scaled = [x/s for x, s in zip(d, scales)] if require_scale else d
    ratio = min(scaled)/max(scaled)
    if ratio < ratio_min: return 'asymmetric_rise', ratio
    if angle_guard and (angle is None or not math.isfinite(angle)): return 'unknown', ratio
    if angle_guard and abs(angle) > 5: return 'tilt_guard', ratio
    return 'balanced_lift_proposal', ratio


def summarize(rows, states, hold):
    accepted_start = possible_start = None
    accepted_longest = possible_longest = 0.0
    intervals = []
    counts = Counter()
    for row, state in zip(rows, states):
        if not row.get('evaluated'):
            accepted_start = possible_start = None
            continue
        t = float(row['time_seconds'])
        counts[state] += 1
        if state == 'balanced_lift_proposal':
            if accepted_start is None: accepted_start = t
            accepted_longest = max(accepted_longest, t-accepted_start)
        else:
            accepted_start = None
        if state in ('balanced_lift_proposal', 'unknown'):
            if possible_start is None: possible_start = t
            possible_longest = max(possible_longest, t-possible_start)
        else:
            possible_start = None
    if accepted_longest >= hold:
        status = 'balanced_lift_proposal'
    elif possible_longest >= hold or sum(counts.values()) < 2:
        status = 'unresolved'
    else:
        status = 'no_qualifying_lift_observed'
    return {'status': status, 'longest_accepted_seconds': accepted_longest,
            'longest_possible_seconds': possible_longest, 'frame_counts': dict(counts)}


def plot(rows, observations, key, path):
    W, H = 1060, 840
    left, right = 100, 1020
    duration = max(.01, float(rows[-1]['time_seconds']))
    panels = []
    series = [[], [], [], [], [], []]
    for row, sides in zip(rows, observations):
        t = float(row['time_seconds'])
        for i in range(2):
            if sides[i] is not None:
                series[i].append((t,sides[i]['d']))
                series[i+2].append((t,sides[i]['band']))
        if all(s is not None for s in sides) and min(s['d'] for s in sides)>0:
            d=[s['d'] for s in sides]
            series[4].append((t,min(d)/max(d)))
        angle=row.get('angle_change_deg')
        if angle is not None and math.isfinite(angle): series[5].append((t,float(angle)))
    rise_values=[v for s in series[:4] for _,v in s]
    low=min([0]+rise_values); high=max([3]+rise_values)
    pad=.08*max(3,high-low)
    panels=[('Cumulative minimum-contact rise (640x480 pixels)',low-pad,high+pad,
             [(series[0],'#176998',2.0,''),(series[1],'#d17939',2.0,''),
              (series[2],'#176998',1.0,'4 4'),(series[3],'#d17939',1.0,'4 4')],[1]),
            ('Smaller rise / larger rise (only meaningful after the absolute/noise gates)',0,1.05,
             [(series[4],'#31695a',2.0,'')],[.5]),
            ('Rod angle change (auxiliary; not a primary veto)',
             min([-6]+[v for _,v in series[5]])-1,max([6]+[v for _,v in series[5]])+1,
             [(series[5],'#6c6a60',1.5,'')],[-5,5])]
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#25343e">',
         f'<text x="25" y="27" font-size="17">{html.escape(key)}</text>',
         '<text x="25" y="51" font-size="12">Blue: left; orange: right. Dashed colored curves: flow error bands. No physical-height calibration.</text>']
    spacing=duration/max(1,len(rows)-1)
    for index,(title,lo,hi,curves,thresholds) in enumerate(panels):
        top=100+index*240; bottom=top+170
        def xy(t,v): return left+t/duration*(right-left), bottom-(v-lo)/(hi-lo)*(bottom-top)
        svg.append(f'<text x="25" y="{top-17}" font-size="14">{title}</text>')
        for k in range(5):
            value=lo+(hi-lo)*k/4; _,y=xy(0,value)
            svg.append(f'<path d="M {left} {y:.2f} H {right}" stroke="#e5e9eb"/>')
            svg.append(f'<text x="90" y="{y+4:.2f}" text-anchor="end" font-size="11">{value:.2f}</text>')
        for value in thresholds:
            _,y=xy(0,value)
            svg.append(f'<path d="M {left} {y:.2f} H {right}" stroke="#ac5d59" stroke-dasharray="6 5"/>')
            svg.append(f'<text x="{right}" y="{y-4:.2f}" text-anchor="end" font-size="11">{value:g}</text>')
        for curve,color,stroke,dash in curves:
            parts=[]; part=[]; last=None
            for t,value in curve:
                if last is not None and t-last>1.5*spacing:
                    parts.append(part); part=[]
                part.append(xy(t,value)); last=t
            if part: parts.append(part)
            for part in parts:
                if len(part)<2: continue
                points=' '.join(f'{x:.2f},{y:.2f}' for x,y in part)
                svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-dasharray="{dash}"/>')
        for k in range(6):
            t=duration*k/5; x,_=xy(t,lo)
            svg.append(f'<text x="{x:.2f}" y="{bottom+18}" text-anchor="middle" font-size="11">{t:.2f}s</text>')
    svg.append('</g></svg>'); path.write_text('\n'.join(svg))


def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU compute node required')
    cv2.setNumThreads(1)
    output=ROOT/'outputs'/('evaluation_stick_flow_ratio_absolute_job'+os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    cfg=json.loads(CONFIG.read_text())
    (output/'config_snapshot.json').write_text(json.dumps(cfg,indent=2))
    items=json.loads((BASE/'per_video.json').read_text())
    refs={(r['environment'],r['episode_index'],r['scope']):r['video'] for r in items if r['method']=='gt'}
    scale_cache={}; totals={}; results=[]
    selected={'full_stick-L0-R0_ep000000','full_stick-L0-R0_ep000006','full_stick-L0-R0_ep000020',
              'full_stick-L0-R3right_ep000017','q0021_stick-L0-R4right_gt','q0021_stick-L0-R4right_stage2'}
    variants=[]
    for absolute in cfg['absolute_min_sensitivity_px']:
        for normalized in (False,True):
            variants.append((f'abs{absolute:g}_ratio0.5_'+('scale_normalized' if normalized else 'raw'),absolute,.5,normalized,False))
    for ratio in cfg['ratio_sensitivity']:
        variants.append((f'abs1_ratio{ratio:.3f}_raw',1.0,ratio,False,False))
    variants.append(('abs1_ratio0.5_raw_tilt5_guard',1.0,.5,False,True))
    for item in items:
        rows=[json.loads(line) for line in (BASE/'cases'/item['key']/'corner_tracks.jsonl').read_text().splitlines() if line]
        observations=[measure(row) for row in rows]
        ref=refs[(item['environment'],item['episode_index'],item['scope'])]
        if ref not in scale_cache: scale_cache[ref]=initial_scale(ref)
        scales=scale_cache[ref]
        result={'key':item['key'],'environment':item['environment'],'scope':item['scope'],
                'method':item['method'],'blue_tape_height_proxy_px':scales,'variants':{}}
        directory=output/'cases'/item['key']; directory.mkdir(parents=True)
        primary_rows=[]
        for name,absolute,ratio,normalized,angle_guard in variants:
            states=[]
            for row, measured in zip(rows,observations):
                state,balance=classify(measured,absolute,ratio,scales,normalized,row.get('angle_change_deg'),angle_guard)
                states.append(state)
                if name=='abs1_ratio0.5_raw':
                    primary_rows.append({'frame':row['frame'],'time_seconds':row['time_seconds'],
                                         'evaluated':row['evaluated'],'sides':measured,'ratio':balance,
                                         'state':state,'angle_change_deg':row.get('angle_change_deg')})
            decision=summarize(rows,states,cfg['hold_seconds'])
            result['variants'][name]=decision
            group=item['scope']+'/'+item['method']+'/'+name
            totals.setdefault(group,Counter())[decision['status']]+=1
        with (directory/'primary_tracks.jsonl').open('w') as handle:
            for row in primary_rows: handle.write(json.dumps(row)+'\n')
        if item['key'] in selected: plot(rows,observations,item['key'],directory/'rise_ratio_angle.svg')
        (directory/'summary.json').write_text(json.dumps(result,indent=2))
        results.append(result)
    report={'status':'research_only_no_independently_validated_action_accuracy',
            'videos':len(results),'groups':totals,
            'selected_cases':[r for r in results if r['key'] in selected],
            'scale_proxy_ratios':[h[0]/h[1] for h in scale_cache.values() if h is not None],
            'formal_action_metric_updated':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    (output/'per_video.json').write_text(json.dumps(results,indent=2))
    page=['<!doctype html><meta charset="utf-8"><title>Stick rise ratio and absolute motion</title>',
          '<h1>Absolute rise + bilateral ratio</h1><p>Primary: both contacts at least 1 pixel above their initial position and beyond flow error; ratio at least 0.5 for 0.3 seconds. Research proposals, not certified labels.</p>']
    for key in sorted(selected):
        rel='cases/'+key+'/rise_ratio_angle.svg'
        page.append(f'<h2>{html.escape(key)}</h2><a href="{rel}"><img width="1060" src="{rel}"></a>')
    (output/'index.html').write_text('\n'.join(page))
    print('OUTPUT',output,flush=True)
    print(json.dumps({'videos':len(results),'groups':totals}),flush=True)


if __name__=='__main__': main()
