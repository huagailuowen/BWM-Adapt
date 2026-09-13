#!/usr/bin/env python3
"""Measure full GT episodes and emit review artifacts, not final model scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from wan_video_action.real97_eval.tracking import make_tracker
from wan_video_action.real97_eval.protocol import (
    ball_support_eligibility, desired_door_support_levels, score_door_level,
)


def measure(job):
    row, output, seed = job
    cv2.setNumThreads(1)
    output = Path(output)
    cap = cv2.VideoCapture(row["video_path"])
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode {row['video_path']}")
    tracker = make_tracker(row["task"], first, **seed)
    states = []
    image = first
    peak_image = first.copy()
    peak_value = -float("inf")
    peak_frame = 0
    idx = 0
    while True:
        state = tracker.update(image)
        state["frame"] = idx
        states.append(state)
        if state.get("center") is not None and state["center"][0] > peak_value:
            peak_value = state["center"][0]
            peak_image = image.copy()
            peak_frame = idx
        ok, image = cap.read()
        if not ok:
            break
        idx += 1
    cap.release()
    folder = output / "tracks" / row["task"] / row["environment"]
    folder.mkdir(parents=True, exist_ok=True)
    track_path = folder / f"episode_{row['episode_index']:06d}.jsonl"
    with track_path.open("w") as handle:
        for state in states:
            handle.write(json.dumps(state) + "\n")
    valid = [s for s in states if s.get("center") is not None]
    result = {key: value for key, value in row.items() if key != "skill_annotation"}
    result.update(
        decoded_frames=len(states), track_path=str(track_path),
        missing_rate=1 - len(valid) / len(states),
        tracker_status="proposal_not_formally_certified",
        ambiguous_frames=sum(bool(s.get("ambiguous")) for s in states),
    )
    if valid:
        result.update(initial_center=valid[0]["center"], final_center=valid[-1]["center"],
                      min_x=min(s["center"][0] for s in valid),
                      max_x=max(s["center"][0] for s in valid))
    if row["task"] == "ball" and valid:
        peak = result["max_x"] + 80
        annotation = row.get("skill_annotation") or {}
        end = annotation.get("skill_end_frame")
        end_state = states[min(int(end), len(states)-1)] if end is not None else None
        end_x = end_state["center"][0] if end_state and end_state.get("center") else None
        result.update(
            peak_x_original_px=peak, peak_frame=peak_frame,
            peak_in_target=280 <= peak <= 330,
            boundary_seen=any(s.get("touches_right_edge", False) for s in states),
            post_action_roll_px=0.0 if end_x is None else max(0., result["max_x"]-end_x),
            initial_x_original_px=valid[0]["center"][0]+80,
        )
        eligible, reason = ball_support_eligibility(result)
        result.update(support_candidate=eligible, support_candidate_reason=reason)
        state = states[peak_frame]
        annotated = peak_image.copy()
        cv2.circle(annotated, tuple(np.round(state["center"]).astype(int)), 7, (0, 255, 0), 1)
        for x in (200, 250):
            cv2.line(annotated, (x, 0), (x, annotated.shape[0]-1), (255, 180, 0), 1)
        review = output / "peak_review" / row["environment"]
        review.mkdir(parents=True, exist_ok=True)
        filename = review / f"{row['split']}_level{row['level']:02d}_ep{row['episode_index']:06d}.jpg"
        cv2.imwrite(str(filename), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result["peak_review_path"] = str(filename)
    elif row["task"] == "door":
        result.update(score_door_level(row["environment"], row["level"]))
        result["preferred_support_levels"] = desired_door_support_levels(row["environment"])
        tail = states[-max(1, round(.3*row["fps"])):]
        result["stable_closed_proposal"] = (
            all(s.get("closed_proposal", False) for s in tail) if
            len(tail) >= round(.3*row["fps"]) else None
        )
        result["final_closure_gap_px"] = states[-1].get("closure_gap_px")
    elif row["task"] == "stick":
        good = [s for s in states if s.get("angle_deg") is not None]
        if good:
            base = float(np.median([s["angle_deg"] for s in good[:5]]))
            tail = good[-7:]
            result["initial_support_fraction"] = good[0].get("support_fraction")
            result["tilt_change_deg"] = float(np.median([s["angle_deg"] for s in tail])-base)
            result["left_lift_px"] = float(good[0]["left"][1]-np.median([s["left"][1] for s in tail]))
            result["right_lift_px"] = float(good[0]["right"][1]-np.median([s["right"][1] for s in tail]))
    elif row["task"] == "soft" and valid:
        initial_x = valid[0]["center"][0]
        result["left_motion_px"] = initial_x-result["min_x"]
        result["right_motion_px"] = result["max_x"]-initial_x
        result["seed_approved"] = bool(seed.get("seed_approved", False))
        result["support_candidate"] = (
            row["training_eligible"] and result["seed_approved"]
            and result["missing_rate"] <= .02
            and max(result["left_motion_px"], result["right_motion_px"]) >= 25
        )
    # Per-task native-frame overlays at start/middle/end allow independent audits.
    review = output / "track_review" / row["task"] / row["environment"]
    review.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(row["video_path"])
    frames = []
    for frame_idx in (0, len(states)//2, len(states)-1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = cap.read()
        if not ok:
            continue
        state = states[frame_idx]
        if state.get("center") is not None:
            cv2.circle(image, tuple(np.round(state["center"]).astype(int)), 6, (0, 255, 0), 2)
        if state.get("left") is not None:
            cv2.line(image, tuple(np.round(state["left"]).astype(int)),
                     tuple(np.round(state["right"]).astype(int)), (0, 255, 0), 2)
        if state.get("polygon") is not None:
            cv2.polylines(image, [np.round(state["polygon"]).astype(np.int32)], True, (0, 255, 0), 1)
        if state.get("closed_reference_x") is not None:
            x = round(state["closed_reference_x"])
            cv2.line(image, (x, 140), (x, 370), (255, 255, 0), 1)
        width = 400
        image = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])))
        cv2.putText(image, f"f{frame_idx} conf={state.get('confidence', 0):.2f}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
        frames.append(image)
    cap.release()
    if frames:
        review_path = review / f"{row['split']}_ep{row['episode_index']:06d}.jpg"
        cv2.imwrite(str(review_path), np.hstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 93])
        result["track_review_path"] = str(review_path)
    return result


def summarize(results, output):
    output = Path(output)
    groups = defaultdict(list)
    for row in results:
        if row["task"] == "ball":
            groups[(row["environment"], row["level"], row["split"])].append(row)
    table = []
    for (env, level, split), rows in sorted(groups.items()):
        valid = [r for r in rows if r.get("peak_x_original_px") is not None]
        peaks = [r["peak_x_original_px"] for r in valid]
        table.append({
            "environment": env, "level": level, "split": split,
            "episodes": len(rows), "detected_episodes": len(valid),
            "peak_x_min": min(peaks) if peaks else None,
            "peak_x_median": float(np.median(peaks)) if peaks else None,
            "peak_x_max": max(peaks) if peaks else None,
            "target_hits": sum(r.get("peak_in_target", False) for r in rows),
            "support_candidates": sum(r.get("support_candidate", False) for r in rows),
            "tracking_review_needed": sum(r["missing_rate"] > .02 or r["ambiguous_frames"] > 0 for r in rows),
        })
    if table:
        with (output/"ball_levels.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    support = [r for r in results if r["task"] == "ball" and r.get("support_candidate")]
    (output/"ball_train_support_candidates.json").write_text(json.dumps(support, indent=2))
    for env in sorted({row["environment"] for row in results if row["task"]=="ball"}):
        relevant = [r for r in table if r["environment"] == env]
        print(env, json.dumps({
            "test_peak_x_by_level": {r["level"]: round(r["peak_x_median"],1) if r["peak_x_median"] is not None else None
                                     for r in relevant if r["split"] == "test"},
            "test_target_levels": [r["level"] for r in relevant if r["split"]=="test" and r["target_hits"]],
            "train_support_candidate_levels": [r["level"] for r in relevant if r["split"]=="train" and r["support_candidates"]],
        }), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="ball,door,stick,soft")
    parser.add_argument("--other-per-env-split", type=int, default=1,
                        help="0 means all episodes; ball always measures all 320 episodes.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=Path)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Batch video detection requires a Slurm compute allocation; use run_real97_cpu_audit.sh.")
    rows=[json.loads(line) for line in args.inventory.read_text().splitlines() if line.strip()]
    seeds=json.loads(args.seeds.read_text()) if args.seeds else {}
    tasks=set(args.tasks.split(","))
    selected=[]
    counts=defaultdict(int)
    for row in rows:
        if row["task"] not in tasks:
            continue
        key=(row["task"],row["environment"],row["split"])
        if row["task"]!="ball" and args.other_per_env_split and counts[key]>=args.other_per_env_split:
            continue
        selected.append(row); counts[key]+=1
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"measurement_manifest.jsonl").open("w") as handle:
        for row in selected: handle.write(json.dumps(row)+"\n")
    jobs=[(row,str(args.output),seeds.get(f"{row['task']}/{row['environment']}/{row['episode_index']}",{}))
          for row in selected]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        with (args.output/"episode_outcomes.jsonl").open("w") as handle:
            for idx,result in enumerate(executor.map(measure,jobs),1):
                results.append(result)
                handle.write(json.dumps(result)+"\n"); handle.flush()
                if idx%40==0 or idx==len(jobs):
                    print(f"[audit] measured={idx}/{len(jobs)}",flush=True)
    summarize(results,args.output)


if __name__=="__main__":
    main()
