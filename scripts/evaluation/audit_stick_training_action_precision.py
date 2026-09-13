#!/usr/bin/env python3
"""CPU contact audit of all train episodes plus existing matched predictions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from wan_video_action.real97_eval.stick_lift_contact import StickLiftContact, classify_lift


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError(f"Cannot decode {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 480)) if frame.shape[:2] != (480, 640) else frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames, fps


def initial_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Missing common conditioning frame: {path}")
    return cv2.resize(frame, (640, 480)) if frame.shape[:2] != (480, 640) else frame


def annotate(frame, state, text):
    image = frame.copy()
    for hull in state.get("body_hulls", []):
        cv2.polylines(image, [np.asarray(hull, np.int32)], True, (0, 210, 255), 1, cv2.LINE_AA)
        if state.get("ground_line"):
            slope, intercept = state["ground_line"]
            xs = np.asarray(hull)[:, 0]
            x0, x1 = max(0, int(xs.min()) - 8), min(639, int(xs.max()) + 8)
            cv2.line(image, (x0, round(slope * x0 + intercept)), (x1, round(slope * x1 + intercept)), (255, 70, 0), 1)
    panel = np.full((545, 640, 3), 248, dtype=np.uint8)
    panel[65:] = image
    cv2.putText(panel, text[:85], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1, cv2.LINE_AA)
    if "left_clearance_px" in state:
        label = f"lower L={state['left_clearance_px']:.1f}px R={state['right_clearance_px']:.1f}px tilt={state['angle_change_deg']:.1f}deg"
    else:
        label = state.get("reason", "missing")
    cv2.putText(panel, label, (8, 47), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 160), 1, cv2.LINE_AA)
    return panel


def measure(job):
    cv2.setNumThreads(1)
    frames, fps = read_frames(job["video"])
    reference = initial_frame(job["reference_video"]) if job.get("reference_video") else frames[0]
    tracker = StickLiftContact(reference)
    states = []
    for index, frame in enumerate(frames):
        state = tracker.update(frame)
        state.update(frame=index, time_seconds=index / fps, evaluated=index > 0)
        if job.get("query"):
            query = job["query"]
            native = int(query["start_frame"]) + index * int(query["frame_stride"])
            state["evaluated"] = bool(index > 0 and native < int(query["total_frames"]))
            state["native_frame_index"] = native
        states.append(state)
    decision = classify_lift(states, fps, **job["config"]["success"])
    sensitivities = {}
    for clearance in job["config"]["sensitivity_clearance_px"]:
        for angle in job["config"]["sensitivity_tilt_deg"]:
            sensitivities[f"clearance{clearance:g}_tilt{angle:g}"] = classify_lift(
                states, fps, clearance_px=clearance, tilt_deg=angle,
                hold_seconds=job["config"]["success"]["hold_seconds"],
            )["status"]
    score = [state.get("minimum_clearance_px", -1000) - .5 * abs(state.get("angle_change_deg", 90)) for state in states]
    peak = int(np.argmax(score))
    selected = sorted(set([0, len(frames) // 2, peak, len(frames) - 1]))
    panels = [annotate(frames[index], states[index], f"{job['environment']} ep{job['episode_index']} {job['method']} f{index} {decision['status']}") for index in selected]
    directory = Path(job["output"]) / "cases" / job["key"]
    directory.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(directory / "review.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 94])
    with (directory / "tracks.jsonl").open("w") as handle:
        for state in states:
            handle.write(json.dumps(state) + "\n")
    result = {key: job[key] for key in ("key", "environment", "episode_index", "method", "scope", "video")}
    result.update(fps=fps, frames=len(frames), decision=decision, sensitivity=sensitivities,
                  review=str(directory / "review.jpg"), measurement="provisional_visible_lower_edge_proxy")
    if "query_index" in job:
        result["query_index"] = job["query_index"]
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def precision_report(results, method):
    pairs = defaultdict(dict)
    for row in results:
        if row["scope"] == "existing_train_query":
            pairs[row["query_index"]][row["method"]] = row
    selected = []
    by_env = defaultdict(Counter)
    for index, pair in pairs.items():
        if "gt" not in pair or method not in pair:
            continue
        prediction, truth = pair[method], pair["gt"]
        counts = by_env[prediction["environment"]]
        counts["candidates"] += 1
        counts["gt_" + truth["decision"]["status"]] += 1
        counts["pred_" + prediction["decision"]["status"]] += 1
        if prediction["decision"]["status"] == "success":
            selected.append({"query_index": index, "environment": prediction["environment"],
                             "episode_index": prediction["episode_index"], "gt_status": truth["decision"]["status"]})
            counts["selected"] += 1
            counts["selected_gt_" + truth["decision"]["status"]] += 1
    gt_counts = Counter(row["gt_status"] for row in selected)
    count = len(selected)
    known = gt_counts["success"] + gt_counts["failure"]
    return {"method": method, "selected_count": count, "selected": selected,
            "selected_gt_status": dict(gt_counts),
            "precision_known_only": gt_counts["success"] / known if known else None,
            "precision_lower_bound": gt_counts["success"] / count if count else None,
            "precision_upper_bound": (gt_counts["success"] + gt_counts["unknown"]) / count if count else None,
            "per_environment": {env: dict(values) for env, values in sorted(by_env.items())},
            "status": "preliminary_existing_16_train_queries_not_full_pool_and_not_certified"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full-video analysis must run on a Slurm CPU compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/evaluation/stick_training_action_precision_v1.json"
    config = json.loads(config_path.read_text())
    (args.output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n")
    prep = ROOT / "outputs/evaluation_real97_reference_ttt_v3_20260910/stick_prep_job113867"
    inference = ROOT / "outputs/infer_real97_stick_stage1table_stage2_v3_job113868"
    with (ROOT / "outputs/evaluation_real97_dataset_audit_20260909/episodes.jsonl").open() as handle:
        episodes = [row for line in handle if line.strip()
                    for row in [json.loads(line)] if row["task"] == "stick" and row["training_eligible"]]
    with (prep / "support.jsonl").open() as handle:
        support = [json.loads(line) for line in handle if line.strip()]
    support_keys = {(row["environment"], int(row["episode_index"])) for row in support}
    source = ROOT / "outputs/real97_train_real97_stick_target_c32_2gpu_20260909_v1_job112804/input_manifest/train.jsonl"
    candidates = defaultdict(list)
    with source.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dataset_split"] != "train":
                raise ValueError("Non-training candidate in training action pool")
            key = row["environment"], int(row["episode_index"])
            if key not in support_keys:
                candidates[key].append(row)
    frozen = []
    for key, rows in sorted(candidates.items()):
        full = [row for row in rows if row.get("sampling_kind") == "full_lift"]
        if not full:
            raise ValueError(f"No full-lift window for {key}; do not silently prune an episode")
        row = dict(min(full, key=lambda row: int(row["start_frame"])))
        row["sample_index"] = len(frozen)
        frozen.append(row)
    with (args.output / "candidate_train_full_lift.jsonl").open("w") as handle:
        for row in frozen:
            handle.write(json.dumps(row) + "\n")
    (args.output / "candidate_plan.json").write_text(json.dumps({
        "candidate_count": len(frozen), "full_train_episodes": len(episodes),
        "excluded_support_episodes": sorted([list(key) for key in support_keys]),
        "choice": "earliest annotated full_lift window; frozen before reading outcome videos",
        "contexts_to_reuse": str(inference / "contexts"),
        "bulk_candidate_generation_status": "not_submitted_pending_contact_calibration",
    }, indent=2) + "\n")
    jobs = []
    for episode in episodes:
        jobs.append({"key": f"full_{episode['environment']}_ep{episode['episode_index']:06d}",
                     "environment": episode["environment"], "episode_index": episode["episode_index"],
                     "method": "gt", "scope": "full_training_episode", "video": episode["video_path"],
                     "output": str(args.output), "config": config})
    with (prep / "query.jsonl").open() as handle:
        queries = [json.loads(line) for line in handle if line.strip()]
    for query in queries:
        if query["dataset_split"] != "train":
            continue
        index, env, ep = int(query["sample_index"]), query["environment"], int(query["episode_index"])
        name = f"q{index:04d}_{env}_train_ep{ep:06d}.mp4"
        for method in ("gt", "stage1", "stage2"):
            jobs.append({"key": f"q{index:04d}_{env}_{method}", "query_index": index,
                         "environment": env, "episode_index": ep, "method": method,
                         "scope": "existing_train_query", "video": str(inference / "raw" / method / name),
                         "reference_video": str(inference / "raw/gt" / name), "query": query,
                         "output": str(args.output), "config": config})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure, job): job["key"] for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 10 == 0 or len(results) == len(jobs):
                progress = {"completed": len(results), "expected": len(jobs), "last": result["key"]}
                (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda row: row["key"])
    full_counts = defaultdict(Counter)
    for row in results:
        if row["scope"] == "full_training_episode":
            full_counts[row["environment"]][row["decision"]["status"]] += 1
    summary = {"status": "provisional_contact_proxy_requires_visual_audit",
               "full_train_gt": {env: dict(counts) for env, counts in sorted(full_counts.items())},
               "candidate_pool_size": len(frozen), "processed_videos": len(results),
               "existing_train_query_selection": [precision_report(results, method) for method in ("stage1", "stage2")],
               "bulk_candidate_inference": "not_yet_generated",
               "limitations": ["monocular visible lower-edge proxy", "hidden/clipped contacts are unknown",
                               "thresholds must be audited before formal action success claims"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "per_video.json").write_text(json.dumps(results, indent=2) + "\n")
    body = ['<!doctype html><meta charset="utf-8"><title>Stick contact audit</title>',
            '<style>body{font-family:sans-serif;margin:28px;background:#fafaf7}img{max-width:100%}article{margin:24px 0;border-top:1px solid #aaa}</style>',
            '<h1>Stick: two-ended lift / action selection audit</h1>',
            '<p>Provisional labels, not certified action metrics. Orange outlines: bodies; blue lines: rest contact reference. Inspect lowest corners and shadows.</p>',
            '<p>All training episodes below. Bulk prediction candidate pool is frozen independently of GT outcomes.</p>']
    for row in results:
        relative = Path(row["review"]).relative_to(args.output)
        body.append(f'<article><h3>{html.escape(row["key"])}: {row["decision"]["status"]}</h3><img loading="lazy" src="{relative}"></article>')
    (args.output / "index.html").write_text("\n".join(body))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
