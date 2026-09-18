#!/usr/bin/env python3
"""Action-only evaluation: unchanged 45 test + 30 train-negative candidates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real916_stick_final_common import read_rows, write_json
from score_real97_stick_action_precision_cpu import video_decision, native_reference, aggregate
METHODS = ("standard","ours_stage1","ours_stage2")


def read(path):
    return json.loads(Path(path).read_text())


def process(task):
    cv2.setNumThreads(1)
    key,row,paths,source,config=task
    output=ROOT/config["metrics_output"]
    target=output/"new_per_query"/(key+".json")
    if target.exists():
        return read(target)
    annotation={(r["environment"],r["episode_index"]):r for r in read_rows(
        Path(source)/"episode_split_manifest.jsonl")}[row["environment"],row["episode_index"]]
    hi=row["evaluation_frame_indices"][-1]
    lo=hi-2
    if lo<1:
        raise RuntimeError("No real 0.3-second terminal interval")
    phase={"stride":3,"start_frame":row["start_frame"],
           "native_reference_frame":max(0,int(annotation["lift_start"])-1),
           "decision":{"terminal_window":{"frame_interval":[lo,hi]},"horizon_status":"visible_real_video_tail"}}
    geometry={**read(ROOT/config["geometry_config"]),**read(ROOT/config["fusion_config"])}
    contact={**read(ROOT/config["contact_config"]),"angle_thresholds_deg":[3,5]}
    reference=native_reference(Path(source)/row["video"][0],phase["native_reference_frame"])
    actions={};audits={};panels=[]
    for method,path in paths.items():
        audit,ref,tail=video_decision(path,reference,phase,contact,geometry)
        actions[method]=audit["decision"]["variants"];audits[method]=audit
        tiles=[]
        for i,frame in enumerate([ref]+tail):
            tile=cv2.copyMakeBorder(cv2.resize(frame,(320,240)),25,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
            cv2.putText(tile,f"{method} {'reference' if i==0 else 'tail'+str(i)} {actions[method]['5']['state']}",
                        (4,17),0,.4,(20,20,20),1)
            tiles.append(tile)
        panels.append(np.hstack(tiles))
    review=output/"reviews"/(key+".jpg");review.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(review),np.vstack(panels)):raise RuntimeError("Review write failed")
    result={"key":key,"split":"train","environment":row["environment"],"episode_index":row["episode_index"],
            "action":actions,"review":str(review),"phase":phase,
            "dataset_outcome_for_audit":annotation["outcome"],"cohort_source":"additional_train15"}
    write_json(output/"contact"/(key+".json"),audits);write_json(target,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Compute node required")
    config=read(args.config);prepared=ROOT/config["prepared"]
    plan=read(prepared/"plan.json");selection=read(prepared/"mixed_pool_selection.json")
    original=read(ROOT/config["source_metrics"]/"per_query.json")
    retained=[r for r in original if r["key"] in set(selection["retained_train_keys"])]
    tests=[r for r in original if r["split"]=="test"]
    if len(retained)!=15 or len(tests)!=45:raise RuntimeError("Original pool mismatch")
    runs={m:ROOT/config[m+"_output"] for m in ("ours","standard")}
    for run in runs.values():
        if read(run/"inference_complete.json")["queries"]!=15:raise RuntimeError("Incomplete additional rollouts")
    tasks=[]
    for i,row in enumerate(read_rows(prepared/"query.jsonl")):
        key=f"q{i:04d}_{row['environment']}_train_ep{row['episode_index']:06d}"
        ours=read(runs["ours"]/"completed"/(key+".json"))
        std=read(runs["standard"]/"completed"/(key+".json"))
        if ours["row"]!=std["row"] or ours["row"]!=row:raise RuntimeError("Unpaired new query")
        paths={"gt":ours["paths"]["gt"],"standard":std["paths"]["standard"],
               "ours_stage1":ours["paths"]["stage1"],"ours_stage2":ours["paths"]["stage2"]}
        tasks.append((key,row,paths,plan["source_dataset"],config))
    with ProcessPoolExecutor(max_workers=8) as pool:
        new=list(pool.map(process,tasks))
    combined=tests+retained+new
    identities=[(r["environment"],r["episode_index"],r["split"]) for r in combined]
    if len(combined)!=75 or len(set(identities))!=75:raise RuntimeError("Duplicate or missing candidates")
    groups={}
    table=[]
    for name,rows in (("test_only",tests),("train_negative_candidates",retained+new),("test45_plus_train30",combined)):
        groups[name]={"queries":len(rows),"methods":{}}
        for method in METHODS:
            score=aggregate([{"methods":r["action"]} for r in rows],method,"5")
            score["selection_rate"]=score["predicted_balanced"]/len(rows)
            groups[name]["methods"][method]=score
            table.append({"group":name,"method":method,**score})
    output=ROOT/config["metrics_output"]
    write_json(output/"summary.json",{
        "groups":groups,"pool_selection":selection,"formal_metric_approved":False,
        "visual_audit_pending":True,"old_test_predictions_unchanged":True,
        "no_model_outcome_based_candidate_selection":True,
        "metric":"GT balanced among model-predicted balanced; unknown GT reported as bounds",
        "note":"Negative-enriched train/test mixture; not pure held-out performance"})
    write_json(output/"candidate_pool.json",combined)
    with (output/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write_json(output/"automatic_complete.json",{"test":45,"train":30,"total":75})
    print(json.dumps(groups,indent=2),flush=True)


if __name__=="__main__":main()
