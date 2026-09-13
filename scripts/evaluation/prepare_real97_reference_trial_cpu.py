#!/usr/bin/env python3
"""Freeze train-only supports and factual queries without changing any dataset."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
from real97_trial_common import ROOT, SilverDetector, native_frame, read_jsonl, require_compute, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["stick","soft"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    import cv2
    cv2.setNumThreads(1)
    protocol = json.loads((ROOT / "configs/evaluation/real97_reference_ttt_v3.json").read_text())
    setting = protocol["tasks"][args.task]
    training = ROOT / setting["training_run"]
    frozen = training / "input_manifest"
    summary = json.loads((frozen / "manifest_summary.json").read_text())
    dataset_root = Path(summary["dataset_base_path"])
    args.output.mkdir(parents=True, exist_ok=False)
    source_model = training / setting["model_file"]
    if not (training / ("."+setting["model_file"]+".complete")).is_file():
        raise RuntimeError("Refusing an unpublished model checkpoint")
    reference = args.output / "reference"
    reference.mkdir()
    # Pin immutable weights against future checkpoint pruning, without another 11 GB copy.
    os.link(source_model, reference / "model.safetensors")
    shutil.copy2(training / setting["table_file"], reference / "context_table.json")
    shutil.copy2(frozen / "action_stats.json", reference / "action_stats.json")
    shutil.copy2(ROOT / setting["training_config"], reference / "training_config.yaml")
    write_json(reference / "provenance.json", dict(setting=setting, manifest=summary, protocol=protocol))
    by_episode = defaultdict(list)
    for split in ["train","test"]:
        for row in read_jsonl(frozen / (split+".jsonl")):
            by_episode[(row["environment"],int(row["episode_index"]),split)].append(row)
    def window(key, position=.5):
        rows = by_episode[key]
        if args.task == "stick":
            rows = [row for row in rows if row["sampling_kind"] == "full_lift"]
        if not rows:
            raise ValueError(f"No training-compatible complete window for {key}")
        rows = sorted(rows, key=lambda row: row["start_frame"])
        return dict(rows[round((len(rows)-1)*position)])
    rng = random.Random(protocol["seed"])
    outcomes = {}
    if args.task == "stick":
        for row in read_jsonl(ROOT / "outputs/evaluation_real97_dataset_audit_20260909/calibration_v2_job113505/episode_outcomes.jsonl"):
            if row["task"] == "stick" and row["split"] == "train":
                outcomes[(row["environment"],row["episode_index"],"train")] = row
    detector = None
    cache = {}
    def soft_measure(row, frame):
        nonlocal detector
        key = (row["environment"], row["episode_index"])
        if key not in cache:
            path = ROOT / "outputs/evaluation_real97_dataset_audit_20260909/dino_fullvideo_job113664/visible_center/episodes" / f'{key[0]}_ep{key[1]:06d}' / "tracks.jsonl"
            cache[key] = {item["frame"]: item for item in read_jsonl(path)} if path.is_file() else {}
        if frame not in cache[key]:
            if detector is None:
                detector = SilverDetector()
            cache[key][frame] = detector(native_frame(dataset_root / row["video"][0], frame))
            cache[key][frame]["frame"] = frame
        return cache[key][frame]
    support_rows, query_rows, environments, unavailable = [], [], [], []
    env_names = sorted({key[0] for key in by_episode})
    for env in env_names:
        train_keys = sorted(key for key in by_episode if key[0] == env and key[2] == "train")
        rng.shuffle(train_keys)
        chosen = []
        notes = {}
        if args.task == "stick":
            candidates = []
            for key in train_keys:
                measured = outcomes.get(key,{})
                u, angle = measured.get("initial_support_fraction"), measured.get("tilt_change_deg")
                if u is None or angle is None or measured.get("missing_rate",1) > .02:
                    continue
                if abs(angle) > setting["maximum_support_tilt_deg"]:
                    continue
                candidates.append((float(u),float(angle),key))
            candidates.sort()
            crossings = [(a,b) for a,b in zip(candidates,candidates[1:]) if a[1]*b[1] <= 0 and b[0]>a[0]]
            if crossings:
                a,b = min(crossings,key=lambda pair:pair[1][0]-pair[0][0])
                balance = a[0]+(b[0]-a[0])*abs(a[1])/max(abs(a[1])+abs(b[1]),1e-9)
                left = [item for item in candidates if item[0]<balance and abs(item[1])>=setting["minimum_informative_tilt_deg"]]
                right = [item for item in candidates if item[0]>balance and abs(item[1])>=setting["minimum_informative_tilt_deg"]]
                if left and right:
                    selected = [max(left,key=lambda x:x[0]),min(right,key=lambda x:x[0])]
                    chosen = [window(item[2]) for item in selected]
                    notes = dict(train_only_balance_fraction=balance, support_u=[x[0] for x in selected],
                                 support_tilt_change_deg=[x[1] for x in selected])
        else:
            buckets = {"left":[],"right":[]}
            for key in train_keys:
                best = None
                for position in [.5,0.,1.]:
                    row = window(key,position)
                    frames = sorted({min(row["total_frames"]-1,row["start_frame"]+3*k)
                                     for k in [0,8,16,24,32]})
                    points = [soft_measure(row,f) for f in frames]
                    if not all(p.get("measurement_valid",False) and p.get("center") is not None for p in points):
                        continue
                    xy = np.asarray([p["center"] for p in points])
                    dx = float(xy[-1,0]-xy[0,0])
                    direction = "right" if dx>0 else "left"
                    excursion = float(np.max(xy[:,0]-xy[0,0]) if dx>0 else np.max(xy[0,0]-xy[:,0]))
                    if abs(dx)>=setting["minimum_net_dx_px"] and excursion>=setting["minimum_directional_excursion_px"]:
                        best = dict(row, support_direction=direction, support_dx_px=dx,
                                    support_excursion_px=excursion, support_measurement_frames=frames)
                        break
                if best is not None:
                    buckets[best["support_direction"]].append(best)
                if all(len(bucket)>=2 for bucket in buckets.values()):
                    break
            if all(buckets.values()):
                chosen = buckets["left"][:2]+buckets["right"][:2]
                if len(chosen)<4:
                    used = {row["episode_index"] for row in chosen}
                    rest = [row for rows in buckets.values() for row in rows if row["episode_index"] not in used]
                    chosen += rest[:4-len(chosen)]
                notes = dict(direction_counts=dict(Counter(row["support_direction"] for row in chosen)))
        if len(chosen) != setting["support_count"]:
            unavailable.append(dict(environment=env, reason="support_quality_or_bracketing_not_satisfied", found=len(chosen)))
            print("[unavailable] "+json.dumps(unavailable[-1]),flush=True)
            continue
        used = {row["episode_index"] for row in chosen}
        train_queries = [key for key in train_keys if key[1] not in used][:protocol["query_train_episodes_per_environment"]]
        test_queries = sorted(key for key in by_episode if key[0]==env and key[2]=="test")
        if len(train_queries)<2 or not test_queries:
            raise RuntimeError(f"Insufficient disjoint query episodes: {env}")
        start = len(support_rows)
        support_rows.extend(chosen)
        query_indices = []
        for key in train_queries+test_queries:
            row = window(key)
            row["sample_index"] = len(query_rows)
            query_indices.append(len(query_rows))
            query_rows.append(row)
        environments.append(dict(environment=env, environment_index=chosen[0]["environment_index"],
                                 support_indices=list(range(start,len(support_rows))), query_indices=query_indices, **notes))
        panels=[]
        for row in chosen:
            frames = [min(row["total_frames"]-1,row["start_frame"]+3*k) for k in [0,row["length"]//2,row["length"]-1]]
            strip=[]
            for f in frames:
                image=native_frame(dataset_root/row["video"][0],f)
                cv2.putText(image,f'{env} ep{row["episode_index"]} f{f}',(4,18),0,.42,(0,0,255),1)
                strip.append(image)
            panels.append(np.hstack(strip))
        review=args.output/"support_review";review.mkdir(exist_ok=True)
        cv2.imwrite(str(review/(env+".jpg")),np.vstack(panels))
        print("[environment_ready] "+json.dumps(environments[-1]),flush=True)
    if not environments:
        raise RuntimeError("No environment passed support preparation")
    write_jsonl(args.output/"support.jsonl",support_rows)
    write_jsonl(args.output/"query.jsonl",query_rows)
    stage_files=set()
    for row in support_rows+query_rows:
        stage_files.update(row["video"]+[row["action"],row["action"].split("/")[0]+"/meta/info.json"])
    (args.output/"stage_files.txt").write_text("\n".join(sorted(stage_files))+"\n")
    for (env,ep),frames in cache.items():
        destination=args.output/"support_measurements"/f"{env}_ep{ep:06d}.json"
        write_json(destination,list(frames.values()))
    plan=dict(task=args.task, protocol=protocol, source_dataset=str(dataset_root), prepared=str(args.output.resolve()),
              environments=environments, unavailable_environments=unavailable, expected_environments=len(env_names),
              support_count=len(support_rows), query_count=len(query_rows),
              query_splits=dict(Counter(row["dataset_split"] for row in query_rows)),
              stage1_reference="exact_training_environment_table_entry", stage2_initial="mean_training_table",
              no_query_GT_in_adaptation=True, counterfactual_actions=False, annotation_status="first_batch_proposals")
    write_json(args.output/"plan.json",plan)
    print("[prepared] "+json.dumps({k:v for k,v in plan.items() if k not in ("protocol","environments")}),flush=True)


if __name__=="__main__":
    main()
