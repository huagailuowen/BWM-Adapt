#!/usr/bin/env python3
"""Frozen-track ablation: optical flow primary, no dynamic silhouette veto.

No new model inference, no training changes, and no certified action labels.
Initial contacts and optical flow are identical to v2, so this isolates the
effect of its dynamic silhouette-agreement requirement.
"""

from collections import Counter
from pathlib import Path
import html
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wan_video_action.real97_eval.stick_corner_contact import summarize_refined

BASE = ROOT / "outputs/evaluation_stick_fine_contact_job114006"
CONTOUR = ROOT / "outputs/evaluation_stick_visible_contact_job114128"


def flow_row(row, margin_multiplier=1.0):
    sides = []
    for original in row.get("sides", []):
        side = {"side": original.get("side"), "state": "uncertain"}
        if original.get("flow_valid"):
            motion = original.get("corner_displacement_px", [])
            visible = original.get("corner_visible", [])
            band = float(original.get("uncertainty_margin_px", float("nan"))) * margin_multiplier
            if len(motion) == 2 and len(visible) == 2 and math.isfinite(band) and all(map(math.isfinite, motion)):
                observed = [value for value, valid in zip(motion, visible) if valid]
                if observed and abs(min(observed)) <= band:
                    side["state"] = "contact_consistent"
                elif all(visible) and min(motion) > band:
                    side["state"] = "airborne_proposal"
                side.update(corner_displacement_px=motion, uncertainty_margin_px=band,
                            corner_visible=visible)
        sides.append(side)
    while len(sides) < 2:
        sides.append({"state": "uncertain"})
    angle = row.get("angle_change_deg")
    states = [side["state"] for side in sides]
    state = "uncertain"
    if all(value == "airborne_proposal" for value in states) and angle is not None and math.isfinite(angle):
        state = "both_airborne_balanced" if abs(angle) <= 5.0 else "both_airborne_tilted"
    elif "contact_consistent" in states:
        state = "visible_endpoint_contact_evidence"
    return {"frame": row["frame"], "time_seconds": row["time_seconds"],
            "evaluated": row["evaluated"], "angle_change_deg": angle,
            "state": state, "sides": sides}


def plot_motion(rows, contour, key, target):
    width, height = 1080, 700
    left, right = 100, 1030
    duration = max(0.01, max(float(r['time_seconds']) for r in rows))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" fill="#24323e">',
           f'<text x="30" y="30" font-size="18">{html.escape(key)}</text>',
           '<text x="30" y="54" font-size="12">Flow minimum (blue), opposite tracked contact (light blue), rejected v3 contour estimate (red dashed).</text>',
           '<text x="30" y="74" font-size="12">Gray band: flow uncertainty. Image-space displacement only; not calibrated physical height.</text>']
    for side_index, name in enumerate(('Left endpoint', 'Right endpoint')):
        top, bottom = 115 + side_index * 285, 325 + side_index * 285
        samples, old_samples, bands = [], [], []
        for row in rows:
            sides = row.get('sides', [])
            if len(sides) <= side_index: continue
            side = sides[side_index]
            d = side.get('corner_displacement_px', [])
            b = side.get('uncertainty_margin_px')
            if side.get('flow_valid') and len(d) == 2 and all(map(math.isfinite,d)) and b is not None and math.isfinite(b):
                samples.append((row['time_seconds'], min(d), max(d)))
                bands.append((row['time_seconds'], float(b)))
        for row in contour:
            sides = row.get('sides', [])
            if len(sides) <= side_index: continue
            gap = sides[side_index].get('gap_px')
            if gap is not None and math.isfinite(gap):
                old_samples.append((row['time_seconds'], float(gap)))
        values = [0.0] + [v for point in samples for v in point[1:]] + [v for _,v in old_samples]
        lower, upper = min(values), max(values)
        span = max(3.0, upper-lower)
        lower -= .1*span; upper += .1*span
        def point(t, value):
            return (left + t/duration*(right-left), bottom-(value-lower)/(upper-lower)*(bottom-top))
        svg.append(f'<text x="30" y="{top-13}" font-size="15">{name}</text>')
        for tick in range(5):
            value=lower+(upper-lower)*tick/4
            _, y=point(0,value)
            svg.append(f'<path d="M {left} {y:.2f} H {right}" stroke="#e5e9ed"/>')
            svg.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11">{value:.1f} px</text>')
        if bands:
            xy=[point(t,b) for t,b in bands]+[point(t,-b) for t,b in reversed(bands)]
            svg.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in xy)+'" fill="#bac3cb" opacity="0.3"/>')
        _,zero=point(0,0)
        svg.append(f'<path d="M {left} {zero:.2f} H {right}" stroke="#687984" stroke-dasharray="3 4"/>')
        sequences=[([(t,a) for t,a,b in samples],'#19639b','',2),
                   ([(t,b) for t,a,b in samples],'#75b0d2','',1.3),
                   (old_samples,'#b44c41','stroke-dasharray="6 4"',1.3)]
        # Separate segments across missing samples; never interpolate a lost track.
        times=[float(row['time_seconds']) for row in rows]
        spacing=duration/max(1,len(times)-1)
        for sequence,color,dash,stroke_width in sequences:
            segments=[]; current=[]; previous=None
            for t,value in sequence:
                if previous is not None and t-previous>1.5*spacing:
                    segments.append(current); current=[]
                current.append(point(t,value)); previous=t
            if current: segments.append(current)
            for segment in segments:
                if len(segment)<2: continue
                pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in segment)
                svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{stroke_width}" {dash}/>')
        for tick in range(6):
            t=duration*tick/5; x,_=point(t,0)
            svg.append(f'<text x="{x:.2f}" y="{bottom+20}" text-anchor="middle" font-size="11">{t:.2f}s</text>')
    svg.append('</g></svg>')
    target.write_text('\n'.join(svg))


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this audit on a CPU compute node')
    output=ROOT/'outputs'/('evaluation_stick_flow_primary_job'+os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    items=json.loads((BASE/'per_video.json').read_text())
    totals={}; results=[]
    selected={'full_stick-L0-R0_ep000000','full_stick-L0-R0_ep000006',
              'full_stick-L0-R0_ep000020','full_stick-L0-R3right_ep000017',
              'q0021_stick-L0-R4right_gt','q0021_stick-L0-R4right_stage2'}
    for item in items:
        rows=[json.loads(line) for line in (BASE/'cases'/item['key']/'corner_tracks.jsonl').read_text().splitlines() if line]
        result={'key':item['key'],'method':item['method'],'scope':item['scope'],
                'previous_v2':item['refined_decision'],'modes':{}}
        case=output/'cases'/item['key']; case.mkdir(parents=True)
        for multiplier in (1.0,2.0):
            mode='flow_only_margin_'+str(int(multiplier))+'x'
            derived=[flow_row(row,multiplier) for row in rows]
            decision=summarize_refined(derived,item['fps'],{'hold_seconds':.3,'maximum_tilt_deg':5.0})
            result['modes'][mode]=decision
            group=item['scope']+'/'+item['method']+'/'+mode
            counts=totals.setdefault(group,Counter())
            counts['videos']+=1
            counts[decision['status']]+=1
            if multiplier==1.0:
                with (case/'flow_states.jsonl').open('w') as handle:
                    for row in derived: handle.write(json.dumps(row)+'\n')
        if item['key'] in selected:
            contour=[json.loads(line) for line in (CONTOUR/'cases'/item['key']/'tracks.jsonl').read_text().splitlines() if line]
            plot_motion(rows,contour,item['key'],case/'flow_vs_contour.svg')
        (case/'summary.json').write_text(json.dumps(result,indent=2))
        results.append(result)
    report={'status':'frozen_flow_ablation_not_validated_action_metric','videos':len(results),
            'groups':totals,'selected_cases':[r for r in results if r['key'] in selected],
            'unchanged':'v2 initial contacts, flow correspondences, initial offset policy, temporal masks, hold duration and tilt threshold',
            'changed':'remove dynamic silhouette agreement veto; include 2x uncertainty sensitivity',
            'limitations':['reference points can still be inaccurate','2D affine motion cannot certify every hidden 3D contact','no independent labeled accuracy measurement'],
            'formal_action_metric_updated':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    (output/'per_video.json').write_text(json.dumps(results,indent=2))
    page=['<!doctype html><meta charset="utf-8"><title>Stick optical flow ablation</title>',
          '<h1>Optical-flow-primary diagnostic</h1><p>Research only. Red contour estimates include known bad-reference cases.</p>']
    for key in sorted(selected):
        rel='cases/'+key+'/flow_vs_contour.svg'
        page.append(f'<h2>{html.escape(key)}</h2><a href="{rel}"><img width="1080" src="{rel}"></a>')
    (output/'index.html').write_text('\n'.join(page))
    print('OUTPUT',output,flush=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
