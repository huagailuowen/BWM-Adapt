#!/usr/bin/env python3
"""Run fusion and audit artifacts on a CPU allocation; never modify legacy results."""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.reevaluate_stick_shape_flow_cpu import read_frames
from wan_video_action.real97_eval.stick_common_reference_rigid_review import RigidBody
from wan_video_action.real97_eval.stick_shape_flow_review import ReferenceBody, export_shape, rod_body
from wan_video_action.real97_eval.stick_terminal_silhouette import rod_angle
from wan_video_action.real97_eval.stick_fused_terminal_review import (
    SequentialBody, appearance_at_pose, clearance, decide, fuse_angles,
)

BRANCH = json.loads((ROOT / "configs/evaluation/stick_fused_terminal_review_v4.json").read_text())
CONFIG = {**json.loads((ROOT / BRANCH["base_config"]).read_text()), **BRANCH}
SOURCE = ROOT / "outputs/evaluation_stick_phase_interval_flow_job114259/per_video.json"


def save_review(reference, frames, rows, path):
    selected = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    panels = [("GT pre-lift", reference, None)] + [("terminal %s" % frames[i][0], frames[i][1], rows[i]) for i in selected]
    clean = np.full((780, 320 * len(panels), 3), 245, np.uint8)
    marked = clean.copy()
    for col, (title, image, row) in enumerate(panels):
        overlay = image.copy()
        if row:
            for side, geometry in enumerate(row["sides"]):
                for measurement in geometry.get("measurements", []):
                    if "hull" not in measurement:
                        continue
                    color = (0, 180, 0) if measurement["source"] == "sequential" else (255, 90, 0) if side == 0 else (0, 60, 255)
                    cv2.polylines(overlay, [np.asarray(measurement["hull"], np.int32).reshape(-1, 1, 2)], True, color, 1)
        x = col * 320
        for canvas, frame in ((clean, image), (marked, overlay)):
            canvas[30:370, x:x + 320] = frame[140:480, :320]
            canvas[420:760, x:x + 320] = frame[140:480, 320:640]
            cv2.putText(canvas, "LEFT " + title, (x + 3, 21), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            cv2.putText(canvas, "RIGHT " + title, (x + 3, 410), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    cv2.imwrite(str(path / "endpoint_review.jpg"), clean, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(str(path / "tracking_review.jpg"), marked, [cv2.IMWRITE_JPEG_QUALITY, 98])


def process(item):
    record, native, previous, destination = item
    cv2.setNumThreads(1)
    cv2.setRNGSeed(73019)
    reference_index = native["reference_frame"]
    reference = read_frames(native["video"], [reference_index])[reference_index]
    bodies = [RigidBody(reference, side, CONFIG) for side in (0, 1)]
    originals = [b.original for b in bodies]
    chains = [SequentialBody(body, cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)) for body in bodies]
    reference_rod = rod_angle(reference, [rod_body(s) for s in originals], CONFIG)
    lo, hi = record["decision"]["terminal_window"]["frame_interval"]
    stride, start = record["stride"], record["start_frame"]
    if (hi - lo) * stride / record["source_fps"] + 1e-8 < CONFIG["hold_seconds"]:
        raise ValueError("incomplete source-clock interval: " + record["key"])
    if start + hi * stride >= record["total_frames"]:
        raise ValueError("padding cannot certify success: " + record["key"])
    control = record["scope"] == "query_native_control"
    targets = {start + i * stride if control else i: i for i in range(lo, hi + 1)}
    first = reference_index if control else max(0, math.ceil((reference_index - start) / stride))
    prior = {r["frame"]: r for r in previous["terminal_frames"]}
    capture = cv2.VideoCapture(record["video"])
    frames, rows = [], []
    for index in range(max(targets) + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("missing video frame: " + record["key"])
        if index < first:
            continue
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for chain in chains:
            chain.update(gray)
        if index not in targets:
            continue
        old = prior[index]
        direct = old["bodies"]
        sequential = [chain.report() for chain in chains]
        appearances, gaps = [], []
        for side, (original, d, s) in enumerate(zip(originals, direct, sequential)):
            appearance = d.get("appearance", {"valid": False})
            if (not appearance.get("valid") or appearance.get("search_truncated")) and s.get("valid"):
                appearance = appearance_at_pose(frame, original, np.asarray(s["matrix"]), side, CONFIG)
            appearances.append(appearance)
            gaps.append(clearance(original, d, s, appearance, previous["floor_slope_proxy"], CONFIG))
        static_shapes = [ReferenceBody(frame, side, CONFIG).reference for side in (0, 1)]
        independent_rod = rod_angle(frame, [rod_body(s) for s in static_shapes], CONFIG)
        angle = fuse_angles(originals, direct, sequential, reference_rod, independent_rod, CONFIG)
        source_frame = index if control else start + index * stride
        rows.append({"frame": index, "source_frame": source_frame, "sides": gaps,
                     "fused_angle": angle, "independent_rod": independent_rod,
                     "direct": direct, "sequential": sequential, "appearances": appearances})
        frames.append((index, frame))
    capture.release()
    if len(rows) != hi - lo + 1:
        raise RuntimeError("terminal frame coverage mismatch: " + record["key"])
    result = {k: record[k] for k in ("key", "scope", "split", "method", "environment", "episode_index", "video")}
    result.update(reference_video=native["video"], reference_frame=reference_index,
                  reference_bodies=[export_shape(o) for o in originals], reference_rod=reference_rod,
                  terminal_window=record["decision"]["terminal_window"], terminal_frames=rows,
                  previous_variants=previous["diagnostic_variants"], diagnostic_variants=decide(rows, CONFIG),
                  formal_metric_approved=False)
    case = Path(destination) / "cases" / record["key"]
    case.mkdir(parents=True, exist_ok=False)
    (case / "analysis.json").write_text(json.dumps(result, indent=2))
    save_review(reference, frames, rows, case)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a CPU compute-node allocation")
    output = ROOT / ("outputs/reevaluation_stick_fused_terminal_job" + os.environ["SLURM_JOB_ID"])
    output.mkdir(exist_ok=False)
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2))
    records = json.loads(SOURCE.read_text())
    previous = {r["key"]: r for r in json.loads((ROOT / BRANCH["direct_measurements"]).read_text())}
    full = {(r["environment"], r["episode_index"]): r for r in records if r["scope"] == "full_episode"}
    tasks = []
    for r in records:
        native = full[(r["environment"], r["episode_index"])]
        if not native.get("has_pre_lift_reference"):
            raise ValueError("missing original resting reference: " + native["key"])
        tasks.append((r, native, previous[r["key"]], str(output)))
        if r["scope"] == "query" and r["method"] == "gt":
            control = dict(r, key=r["key"] + "_native_control", scope="query_native_control", video=native["video"])
            tasks.append((control, native, previous[control["key"]], str(output)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(process, tasks))
    counts = {str(t): defaultdict(Counter) for t in CONFIG["small_tilt_sensitivity_deg"]}
    tracking = defaultdict(Counter)
    changes, positives = [], []
    paired = defaultdict(dict)
    for r in results:
        group = r["scope"] + "/" + r["split"] + "/" + r["method"]
        frames = r["terminal_frames"]
        direct_ok = all(b.get("valid") for f in frames for b in f["direct"])
        fused_ok = all(d.get("valid") or s.get("valid") for f in frames for d, s in zip(f["direct"], f["sequential"]))
        tracking[group][str(direct_ok) + " -> " + str(fused_ok)] += 1
        for threshold, variant in r["diagnostic_variants"].items():
            counts[threshold][group][variant["status"]] += 1
            before = r["previous_variants"][threshold]["status"]
            if before != variant["status"]:
                changes.append({"key": r["key"], "threshold": threshold, "before": before, "after": variant["status"],
                                "all_negative_rule": variant["all_frames_negative_rule_status"]})
        if any(v["status"] == "balanced_lift_candidate" for v in r["diagnostic_variants"].values()):
            positives.append({"key": r["key"], "scope": r["scope"], "review": str(output / "cases" / r["key"] / "endpoint_review.jpg")})
        if r["scope"] in ("query", "query_native_control"):
            paired[r["key"].split("_")[0]]["native_gt" if r["scope"] == "query_native_control" else r["method"]] = r
    comparisons = {}
    control_mismatches = []
    for threshold in counts:
        comparison = {m: Counter() for m in ("stage1", "stage2", "native_gt")}
        for case, methods in paired.items():
            gt = methods["gt"]["diagnostic_variants"][threshold]["status"]
            for method, counter in comparison.items():
                other = methods[method]["diagnostic_variants"][threshold]["status"]
                counter[gt + " -> " + other] += 1
                if method == "native_gt" and other != gt:
                    control_mismatches.append({"case": case, "threshold": threshold, "sampled_gt": gt, "native_gt": other})
        comparisons[threshold] = {m: dict(c) for m, c in comparison.items()}
    summary = {"job": os.environ["SLURM_JOB_ID"], "videos": len(results),
               "counts": {t: {g: dict(c) for g, c in groups.items()} for t, groups in counts.items()},
               "tracking": {g: dict(c) for g, c in tracking.items()}, "paired": comparisons,
               "native_sampled_gt_disagreements": control_mismatches,
               "formal_metric_approved": False, "legacy_results_modified": False}
    for name, data in (("summary.json", summary), ("per_video.json", results), ("changed_cases.json", changes),
                       ("positive_review_manifest.json", positives)):
        (output / name).write_text(json.dumps(data, indent=2))
    print(json.dumps({"output": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
