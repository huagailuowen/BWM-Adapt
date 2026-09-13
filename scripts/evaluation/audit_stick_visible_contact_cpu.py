#!/usr/bin/env python3
"""Compare direct bottom contours against the frozen v2 contact calibration."""

from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import argparse
import html
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scripts.evaluation.audit_stick_training_action_precision import read_frames, initial_frame
from wan_video_action.real97_eval.stick_corner_contact import summarize_refined
from wan_video_action.real97_eval.stick_visible_contact import VisibleBottomContact

BASE = ROOT / "outputs/evaluation_stick_fine_contact_job114006"


def render_detail(frames, rows, indices, reference, destination):
    columns = []
    for index in indices:
        frame, row = frames[index], rows[index]
        panels = []
        for side in range(2):
            body = reference[side]
            if body is None:
                cx, cy = (80 if side == 0 else 550), 300
            else:
                cx = body["center"][0]
                cy = float(np.percentile(body["lower"][:, 1], 95)) - 25
            x0, y0 = int(np.clip(cx - 65, 0, 510)), int(np.clip(cy - 55, 0, 370))
            raw = frame[y0:y0+110, x0:x0+130]
            overlay = frame.copy()
            result = row["sides"][side]
            if result.get("hull"):
                cv2.polylines(overlay, [np.asarray(result["hull"], np.int32)], True, (0,200,0), 1)
            for point in result.get("contact_points", []):
                cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, (0,220,255), -1)
            m, b = row["ground_line"]
            if m is not None:
                cv2.line(overlay, (0,int(round(b))), (639,int(round(639*m+b))), (255,100,0), 1)
            for label, crop in [("RAW",raw), ("CONTOUR",overlay[y0:y0+110,x0:x0+130])]:
                header = np.full((62,390,3),255,np.uint8)
                lines = [f"{'L' if side==0 else 'R'} frame={index} {label}",result['state'],
                         f"gap={result.get('gap_px',float('nan')):.2f}px edge={result.get('edge_strength',0):.1f}"]
                for k, text in enumerate(lines):
                    cv2.putText(header,text,(6,16+19*k),cv2.FONT_HERSHEY_SIMPLEX,.42,(30,30,30),1,cv2.LINE_AA)
                panels.append(np.vstack([header,cv2.resize(crop,(390,330),interpolation=cv2.INTER_NEAREST)]))
        columns.append(np.vstack(panels))
    cv2.imwrite(str(destination),np.hstack(columns))


def analyze(task):
    item, output, reference_path = task
    cv2.setNumThreads(1)
    frames, fps = read_frames(item["video"])
    frozen = [json.loads(line) for line in (BASE/"cases"/item["key"]/"corner_tracks.jsonl").read_text().splitlines() if line]
    if len(frames) != len(frozen):
        raise ValueError(f"Frozen frame count changed: {item['key']}")
    tracker = VisibleBottomContact(initial_frame(reference_path))
    rows = []
    for index, frame in enumerate(frames):
        row = tracker.update(frame)
        row.update(frame=index, time_seconds=index/fps, evaluated=frozen[index]["evaluated"])
        rows.append(row)
    # Retain v2 temporal aggregation, 0.3s hold and five-degree limit. This
    # isolates the detector change; do not relax video labels to improve coverage.
    decision = summarize_refined(rows,fps,{"hold_seconds":.3,"maximum_tilt_deg":5.0})
    directory = Path(output)/"cases"/item["key"]
    directory.mkdir(parents=True)
    with (directory/"tracks.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row)+"\n")
    counts = Counter()
    for row in rows:
        if not row["evaluated"]: continue
        counts["frames"]+=1
        counts[row["state"]]+=1
        for side in row["sides"]:
            counts["sides"]+=1
            counts["body_detected"]+=int(side["body_detected"])
            counts["side_"+side["state"]]+=1
            counts["clipped_sides"]+=int(side.get("clipped",False))
            counts["crop_cut_sides"]+=int(side.get("crop_cut",False))
            counts["weak_edge_sides"]+=int(side.get("edge_strength",0)<12)
    result={"key":item["key"],"scope":item["scope"],"method":item["method"],
            "environment":item["environment"],"episode_index":item["episode_index"],
            "previous":item["refined_decision"],"proposed":decision,"counts":dict(counts)}
    if item.get("rendered_detail"):
        indices=item.get("detail_frames",[0,len(frames)//3,2*len(frames)//3,len(frames)-1])
        render_detail(frames,rows,indices,tracker.reference,directory/"contact_detail.jpg")
        result["detail"]=str(directory/"contact_detail.jpg")
    (directory/"summary.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--workers",default=4,type=int)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this video audit on a CPU compute node")
    args.output.mkdir(parents=True,exist_ok=False)
    items=json.loads((BASE/"per_video.json").read_text())
    # All generated methods use their matched GT first frame as the rest reference.
    references={(r['environment'],r['episode_index'],r['scope']):r['video']
                for r in items if r['method']=='gt'}
    tasks=[(r,str(args.output),references[(r['environment'],r['episode_index'],r['scope'])]) for r in items]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(analyze,tasks,chunksize=1):
            results.append(result)
            if len(results)%10==0:
                progress={"completed":len(results),"expected":len(items)}
                (args.output/"progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress),flush=True)
    groups={}
    for result in results:
        key=result['scope']+'/'+result['method']
        group=groups.setdefault(key,{"decisions":Counter(),"frames":Counter(),"transitions":Counter()})
        group['decisions'][result['proposed']['status']]+=1
        group['frames'].update(result['counts'])
        group['transitions'][result['previous']['status']+' -> '+result['proposed']['status']]+=1
    report={"status":"experimental_detector_not_human_validated","videos":len(results),"groups":groups,
            "temporal_aggregation":"unchanged from v2; no majority-vote relabeling",
            "large_candidate_inference_submitted":False}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    (args.output/"per_video.json").write_text(json.dumps(results,indent=2))
    page=['<!doctype html><meta charset="utf-8"><title>Stick visible-contact audit</title>',
          '<h1>Experimental direct contour audit</h1><p>Green: measured body contour. Yellow: lowest contact neighborhood. Blue: reference ground line. Not a certified action metric.</p>']
    for result in results:
        if 'detail' in result:
            rel=Path(result['detail']).relative_to(args.output)
            page.append(f"<h3>{html.escape(result['key'])}: {html.escape(result['proposed']['status'])}</h3><a href='{rel.as_posix()}'><img width='1000' src='{rel.as_posix()}'></a>")
    (args.output/"index.html").write_text('\n'.join(page))
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
