#!/usr/bin/env python3
"""CPU-only calibration / RGB audit. This entrypoint never loads a world model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from wan_video_action.real97_eval.calibrated_tracking import (
    build_templates, make_calibrated_tracker)
from wan_video_action.real97_eval.object_measurements import episode_diagnostics

CONFIG=None
TEMPLATES=None
LEGACY=None


def initialize(config):
    global CONFIG,TEMPLATES,LEGACY
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video calibration requires a Slurm CPU compute allocation")
    cv2.setNumThreads(1)
    CONFIG=config
    TEMPLATES=build_templates(config["soft"], ROOT/config["soft_dataset"])
    spec=importlib.util.spec_from_file_location(
        "real97_legacy_measure",ROOT/"scripts/evaluation/audit_real97_tracking.py")
    LEGACY=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LEGACY)
    LEGACY.make_tracker=lambda task,frame,**kw: make_calibrated_tracker(
        task,frame,CONFIG,TEMPLATES)


def measure(job):
    row,output=job
    result=LEGACY.measure((row,output,{}))
    states=[json.loads(s) for s in Path(result["track_path"]).read_text().splitlines()]
    result["calibration_v2"]=episode_diagnostics(row["task"],states,row["fps"],CONFIG)
    # Full-video measurements alone do not approve a support CHUNK.
    result["formal_support_approved"]=False
    result["tracker_version"]=CONFIG["soft"].get("tracker_version","v2") if row["task"]=="soft" else "calibration_v2"
    if row["task"]=="soft":
        result["support_candidate"]=bool(row["training_eligible"] and
            result["calibration_v2"].get("support_motion_proposal"))
        result["support_candidate_reason"]="motion_proposal_requires_identity_and_chunk_review"
    return result


def audit_landmarks(config,output):
    templates=build_templates(config["soft"],ROOT/config["soft_dataset"])
    records=[]
    out=output/"landmark_review"
    out.mkdir()
    for idx,label in enumerate(config["validation_landmarks"]):
        env,ep,f=label["environment"],label["episode_index"],label["frame"]
        path=(ROOT/config["soft_dataset"]/(env+"_lerobot")/"videos"/
              f"chunk-{ep//1000:03d}"/"observation.images.image"/f"episode_{ep:06d}.mp4")
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        ok,image=cap.read();cap.release()
        if not ok: raise RuntimeError(f"Cannot decode validation frame {path}:{f}")
        tracker=make_calibrated_tracker("soft",image,config,templates)
        prediction=tracker.update(image)
        # Independently check the sequential tracker at the same landmark.
        sequential_cap=cv2.VideoCapture(str(path))
        sequential_tracker=None
        sequential_prediction=None
        for frame_index in range(f+1):
            decoded,current=sequential_cap.read()
            if not decoded:
                raise RuntimeError(f"Cannot decode sequential landmark {path}:{frame_index}")
            if sequential_tracker is None:
                sequential_tracker=make_calibrated_tracker("soft",current,config,templates)
            sequential_prediction=sequential_tracker.update(current)
        sequential_cap.release()
        sequential_error=None if sequential_prediction.get("center") is None else float(np.linalg.norm(
            np.array(sequential_prediction["center"])-np.array(label["center"])))
        error=None if prediction.get("center") is None else float(np.linalg.norm(
            np.array(prediction["center"])-np.array(label["center"])))
        row=dict(label,prediction=prediction,sequential_prediction=sequential_prediction,
                 sequential_center_error_px=sequential_error,center_error_px=error,
                 pass_within_annotation_tolerance=error is not None and error<=label["tolerance_px"])
        cv2.circle(image,tuple(map(round,label["center"])),5,(0,255,255),1)
        if prediction.get("center") is not None:
            cv2.circle(image,tuple(map(round,prediction["center"])),4,(0,255,0),1)
        if sequential_prediction.get("center") is not None:
            cv2.drawMarker(image,tuple(map(round,sequential_prediction["center"])),(255,0,255),cv2.MARKER_CROSS,10,1)
        if prediction.get("polygon") is not None:
            cv2.polylines(image,[np.array(prediction["polygon"],np.int32)],True,(0,255,0),1)
        cv2.putText(image,f"{env} ep{ep} f{f} err={error}",(4,18),0,.4,(0,0,255),1)
        filename=out/f"sample{idx:03d}.png";cv2.imwrite(str(filename),image)
        row["review_path"]=str(filename);records.append(row)
    (output/"landmark_validation.json").write_text(json.dumps(records,indent=2))
    print("[landmarks] "+json.dumps({"count":len(records),"detected":sum(r["center_error_px"] is not None for r in records),
                                    "within_tolerance":sum(r["pass_within_annotation_tolerance"] for r in records)}),flush=True)


def summary(results,output):
    groups=defaultdict(list)
    for r in results: groups[r["task"]].append(r)
    report={}
    for task,rows in groups.items():
        report[task]=dict(episodes=len(rows),
                         all_frames_usable=sum(r["calibration_v2"]["usable_fraction"]==1. for r in rows),
                         no_usable_frames=sum(r["calibration_v2"]["usable_fraction"]==0. for r in rows),
                         mean_usable_fraction=float(np.mean([r["calibration_v2"]["usable_fraction"] for r in rows])),
                         motion_support_proposals=sum(r.get("support_candidate",False) for r in rows),
                         formally_certified=False)
    (output/"quality_summary.json").write_text(json.dumps(report,indent=2))
    review=[]
    for r in results:
        d=r["calibration_v2"]
        flagged=(d["usable_fraction"]<.98 or d["edge_frames"]>0 or
                 d.get("near_target_boundary",False) or
                 d.get("uncertain_closure_frames",0)>0)
        if flagged:
            review.append({k:r[k] for k in ("task","environment","episode_index","split",
                                          "track_path","track_review_path","calibration_v2")})
    (output/"review_queue.json").write_text(json.dumps(review,indent=2))
    # Index every example; not just successful detections.
    html=["<!doctype html><meta charset=utf-8><title>Real97 tracking calibration</title>",
          "<style>body{background:#eee;font-family:monospace}img{max-width:100%}article{margin:20px}</style>",
          "<h1>Calibration proposals, not certified gold</h1>"]
    for r in results:
        path=os.path.relpath(r["track_review_path"],output)
        html.append(f'<article><h3>{r["task"]} {r["environment"]} {r["split"]} ep{r["episode_index"]}</h3>'
                    f'<img loading="lazy" src="{path}"><pre>{json.dumps(r["calibration_v2"])}</pre></article>')
    (output/"review.html").write_text("\n".join(html))
    print("[quality] "+json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inventory",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--tasks",default="ball,door,stick,soft")
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit with run_real97_calibration_cpu.sh; login-node decoding is forbidden")
    cv2.setNumThreads(1)
    config=json.loads(args.calibration.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"calibration_snapshot.json").write_text(json.dumps(config,indent=2))
    rows=[json.loads(s) for s in args.inventory.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r["task"] in args.tasks.split(",")]
    (args.output/"measurement_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    audit_landmarks(config,args.output)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(config,)) as pool:
        with (args.output/"episode_outcomes.jsonl").open("w") as handle:
            for index,result in enumerate(pool.map(measure,[(r,str(args.output)) for r in rows]),1):
                results.append(result);handle.write(json.dumps(result)+"\n");handle.flush()
                if index%40==0 or index==len(rows):
                    print(f"[calibration] measured={index}/{len(rows)}",flush=True)
    summary(results,args.output)


if __name__=="__main__":
    main()
