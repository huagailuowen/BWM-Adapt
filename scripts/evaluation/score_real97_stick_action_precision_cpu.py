#!/usr/bin/env python3
"""Paired terminal-balance precision on existing Stick videos; CPU allocations only."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.reevaluate_stick_boundary_safe_cpu import classify
from wan_video_action.real97_eval.stick_boundary_safe_edges import boundary_safe_search, legacy
from wan_video_action.real97_eval.stick_common_reference_rigid_review import RigidBody
from wan_video_action.real97_eval.stick_fused_terminal_review import SequentialBody, fuse_angles
from wan_video_action.real97_eval.stick_shape_flow_review import ReferenceBody, export_shape, rod_body
from wan_video_action.real97_eval.stick_terminal_silhouette import rod_angle


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=plain, allow_nan=False)
        stream.write("\n")


def native_reference(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        frame = None
        for _ in range(index + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Missing native reference {path} frame {index}")
        return frame
    finally:
        cap.release()


def video_decision(path, reference, phase, config, geometry):
    lo, hi = phase["decision"]["terminal_window"]["frame_interval"]
    stride, start = int(phase["stride"]), int(phase["start_frame"])
    cap = cv2.VideoCapture(str(path))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Missing prediction {path}")
    height, width = first.shape[:2]
    if (height, width) != (192, 256):
        cap.release()
        raise RuntimeError(f"Unexpected Stick geometry {path}: {first.shape}")
    # Both observed GT and all predictions use the same effective resolution.
    reference = cv2.resize(cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA), (640, 480))
    cv2.setRNGSeed(73019)
    bodies = [RigidBody(reference, side, geometry) for side in (0, 1)]
    shapes = [body.original for body in bodies]
    chains = [SequentialBody(body, cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)) for body in bodies]
    reference_rod = rod_angle(reference, [rod_body(shape) for shape in shapes], geometry)
    usable = all(shape.get("valid") for shape in shapes)
    slope = 0.0
    if usable:
        hulls = [np.asarray(shape["hull"]) for shape in shapes]
        slope = float((hulls[1][:, 1].max() - hulls[0][:, 1].max()) /
                      (hulls[1][:, 0].mean() - hulls[0][:, 0].mean()))
    rows, edges, tail_frames = [], [], []
    radii = config["search_modes"]["bounded_expand"]
    try:
        with boundary_safe_search(radii, config["search_boundary_margin_px"]) as search:
            feet = [legacy.reference_feet(reference, shape, side, slope, config) if usable else []
                    for side, shape in enumerate(shapes)]
            for index in range(hi + 1):
                if index == 0:
                    frame = first
                else:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Missing terminal frame {path}: {index}")
                frame = cv2.resize(frame, (640, 480))
                if start + index * stride < phase["native_reference_frame"]:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for chain in chains:
                    chain.update(gray)
                if index < lo:
                    for body in bodies:
                        if body.flow is not None:
                            body.flow.update(gray, hsv)
                    continue
                direct = [body.track(frame) for body in bodies]
                sequential = [chain.report() for chain in chains]
                current_shapes = [ReferenceBody(frame, side, geometry).reference for side in (0, 1)]
                current_rod = rod_angle(frame, [rod_body(shape) for shape in current_shapes], geometry)
                angle = fuse_angles(shapes, direct, sequential, reference_rod, current_rod, geometry)
                measured = []
                for side in (0, 1):
                    pose = direct[side] if direct[side].get("valid") else sequential[side]
                    measurement = legacy.measure(frame, shapes[side], feet[side], pose, side, slope, config)
                    measurement["pose_source"] = "direct" if direct[side].get("valid") else "sequential"
                    measured.append(measurement)
                rows.append({"frame": index, "source_frame": start + index * stride, "fused_angle": angle})
                edges.append({"frame": index, "sides": measured})
                tail_frames.append(frame)
            audit = {"frames": edges, "reference_footpoints": feet,
                     "search_summary": search.summary(), "search_events": search.events}
    finally:
        cap.release()
    if len(rows) != hi - lo + 1:
        raise RuntimeError(f"Terminal interval not fully measured: {path}")
    decision = classify({"terminal_frames": rows}, audit, config)
    return {"video": str(path), "reference_shapes": [export_shape(shape) for shape in shapes],
            "floor_slope_proxy": slope, "decision": decision, "audit": audit}, reference, tail_frames


def process(task):
    cv2.setNumThreads(1)
    config, geometry, contact = task["config"], task["geometry"], task["contact"]
    phase = task["phase"]
    native = task["native"]
    key = task["key"]
    phase = dict(phase, native_reference_frame=native["reference_frame"])
    window = phase["decision"].get("terminal_window", {})
    interval = window.get("frame_interval", [])
    covered = phase["decision"].get("horizon_status") == "covered_to_sampling_resolution"
    result = {"key": key, "environment": phase["environment"], "split": phase["split"],
              "episode_index": phase["episode_index"], "terminal_window": window,
              "reference_video": native["video"], "reference_frame": native["reference_frame"],
              "methods": {}, "formal_metric_approved": False}
    if not covered or len(interval) != 2 or interval[0] < 1:
        for method in task["videos"]:
            result["methods"][method] = {str(t): {"state": "unknown", "reason": "incomplete_lift_horizon"}
                                         for t in config["angle_thresholds_deg"]}
    else:
        lo, hi = interval
        if (hi - lo) * phase["stride"] / phase["source_fps"] + 1e-8 < config["hold_seconds"]:
            raise RuntimeError(f"Insufficient terminal duration: {key}")
        if phase["start_frame"] + hi * phase["stride"] >= phase["total_frames"]:
            raise RuntimeError(f"Padded frame in terminal window: {key}")
        reference = native_reference(native["video"], native["reference_frame"])
        panels = []
        for method, path in task["videos"].items():
            record, common_reference, frames = video_decision(path, reference, phase, contact, geometry)
            result["methods"][method] = record["decision"]["variants"]
            write(Path(task["output"]) / "measurements" / key / f"{method}.json", record)
            panel = cv2.resize(frames[-1], (320, 240))
            panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            labels = record["decision"]["variants"]
            title = f"{method}: 3deg {labels['3']['state']}; 5deg {labels['5']['state']}"
            cv2.putText(panel, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .34, (20, 20, 20), 1)
            panels.append(panel)
        review = Path(task["output"]) / "reviews" / f"{key}.jpg"
        review.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(review), np.concatenate(panels, axis=1)):
            raise RuntimeError(f"Cannot save review {review}")
        result["review"] = str(review)
    write(Path(task["output"]) / "per_query" / f"{key}.json", result)
    return result


def aggregate(records, method, threshold):
    pairs = [(r["methods"]["gt"][threshold]["state"], r["methods"][method][threshold]["state"])
             for r in records]
    selected = [gt for gt, prediction in pairs if prediction == "positive"]
    counts = Counter(selected)
    tp, fp, unknown = counts["positive"], counts["negative"], counts["unknown"]
    n = len(selected)
    return {"queries": len(records), "predicted_balanced": n,
            "true_balanced_selected": tp, "false_balanced_selected": fp,
            "selected_gt_unknown": unknown,
            "prediction_unknown": sum(p == "unknown" for _, p in pairs),
            "gt_balanced_total": sum(g == "positive" for g, _ in pairs),
            "gt_unknown_total": sum(g == "unknown" for g, _ in pairs),
            "precision": tp / n if n and not unknown else None,
            "precision_on_known_gt": tp / (tp + fp) if tp + fp else None,
            "precision_lower_bound": tp / n if n else None,
            "precision_upper_bound": (tp + unknown) / n if n else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video processing requires a CPU compute allocation")
    os.chdir(ROOT)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = read(args.config)
    geometry = {**read(config["geometry_config"]), **read(config["fusion_config"])}
    contact = {**read(config["contact_config"]), "angle_thresholds_deg": config["angle_thresholds_deg"]}
    phases = read(config["source_phase_records"])
    native = {(r["environment"], r["episode_index"]): r for r in phases if r["scope"] == "full_episode"}
    queries = {(r["environment"], r["episode_index"]): r for r in phases if r["scope"] == "query" and r["method"] == "gt"}
    tasks = []
    hashes = {}
    for path in sorted((Path(config["ours"]) / "completed").glob("q*.json")):
        other = Path(config["standard"]) / "completed" / path.name
        raw, other_raw = path.read_bytes(), other.read_bytes()
        ours, standard = json.loads(raw), json.loads(other_raw)
        row = ours["row"]
        identity = ["environment", "episode_index", "dataset_split", "start_frame", "end_frame", "frame_stride", "length", "total_frames"]
        if any(row[k] != standard["row"][k] for k in identity):
            raise RuntimeError(f"Unpaired query metadata: {path.name}")
        pair = row["environment"], row["episode_index"]
        phase = queries[pair]
        if phase["start_frame"] != row["start_frame"] or phase["stride"] != row["frame_stride"]:
            raise RuntimeError(f"Phase annotation does not match query: {path.name}")
        if not native[pair].get("has_pre_lift_reference"):
            raise RuntimeError(f"No valid resting reference: {path.name}")
        videos = {"gt": ours["paths"]["gt"], "ours_stage1": ours["paths"]["stage1"],
                  "ours_stage2": ours["paths"]["stage2"], "standard": standard["paths"]["standard"]}
        tasks.append({"key": path.stem, "phase": phase, "native": native[pair], "videos": videos,
                      "config": config, "geometry": geometry, "contact": contact, "output": str(output)})
        hashes[path.stem] = {"ours": hashlib.sha256(raw).hexdigest(), "standard": hashlib.sha256(other_raw).hexdigest()}
    write(output / "protocol.json", {"config": config, "geometry": geometry, "contact": contact,
                                      "completed_marker_sha256": hashes, "queries": len(tasks)})
    if not tasks:
        raise RuntimeError("No completed paired queries")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(json.dumps({"completed": len(records), "total": len(tasks), "key": result["key"]}), flush=True)
    records.sort(key=lambda r: r["key"])
    summary = []
    for split in ["train", "test", "all"]:
        chosen = [r for r in records if split == "all" or r["split"] == split]
        for threshold in config["angle_thresholds_deg"]:
            for method in ["standard", "ours_stage1", "ours_stage2"]:
                summary.append({"split": split, "angle_threshold_deg": threshold, "method": method,
                                **aggregate(chosen, method, str(threshold))})
    write(output / "summary.json", {"formal_metric_approved": False, "rows": summary,
                                    "note": "Precision is TP / predicted positives. Unknown GT yields bounds; zero selections are undefined."})
    with (output / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write(output / "per_query.json", records)
    write(output / "complete.json", {"queries": len(records), "videos_per_query": 4})
    print(json.dumps({"output": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
