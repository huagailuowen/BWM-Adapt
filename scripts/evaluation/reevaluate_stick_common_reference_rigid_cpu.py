#!/usr/bin/env python3
"""CPU-only diagnostic; retain legacy metrics and inference videos unchanged."""
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wan_video_action.real97_eval.stick_common_reference_rigid_review import evaluate
from scripts.evaluation.reevaluate_stick_shape_flow_cpu import read_frames

SOURCE = ROOT / "outputs/evaluation_stick_phase_interval_flow_job114259/per_video.json"
CONFIG_PATH = ROOT / "configs/evaluation/stick_common_reference_rigid_review_v3.json"
CONFIG = json.loads(CONFIG_PATH.read_text())


def draw_review(reference, frames, report, destination):
    samples = [("GT pre-lift reference", reference, None)]
    for (index, frame), row in zip(frames, report["terminal_frames"]):
        samples.append(("terminal frame %s" % index, frame, row))
    canvas = np.full((((len(samples) + 3) // 4) * 290, 1280, 3), 245, np.uint8)
    for n, (title, frame, row) in enumerate(samples):
        overlay = frame.copy()
        if row:
            for side, body in enumerate(row["bodies"]):
                if not body.get("valid"):
                    continue
                cv2.polylines(overlay, [np.asarray(body["rigid"]["hull"], np.int32).reshape(-1, 1, 2)],
                              True, (255, 120, 0) if side == 0 else (0, 80, 255), 2)
                shape = body["appearance"]
                if shape.get("valid"):
                    cv2.polylines(overlay, [np.asarray(shape["hull"], np.int32).reshape(-1, 1, 2)],
                                  True, (0, 220, 0), 1)
        x, y = (n % 4) * 320, (n // 4) * 290
        canvas[y:y + 240, x:x + 320] = cv2.resize(overlay, (320, 240))
        cv2.putText(canvas, title, (x + 4, y + 253), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
        if row:
            values = [round(g["gap_lower_px"], 1) if g.get("valid") else "?" for g in row["sides"]]
            angle = row["tracked_angle"]
            av = round(angle["relative_angle_deg"], 2) if angle.get("valid") else "?"
            cv2.putText(canvas, "lower gaps %s angle %s" % (values, av), (x + 4, y + 272),
                        cv2.FONT_HERSHEY_SIMPLEX, .34, (20, 20, 20), 1)
    cv2.imwrite(str(destination / "terminal_review.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])
    # Native-resolution endpoint views, without overlays hiding tiny contact gaps.
    positions = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    selected = [("reference", reference)] + [("terminal %s" % frames[i][0], frames[i][1]) for i in positions]
    clean = np.full((780, len(selected) * 320, 3), 245, np.uint8)
    for column, (label, frame) in enumerate(selected):
        x = column * 320
        clean[30:370, x:x + 320] = frame[140:480, :320]
        clean[420:760, x:x + 320] = frame[140:480, 320:640]
        cv2.putText(clean, "LEFT " + label, (x + 5, 21), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
        cv2.putText(clean, "RIGHT " + label, (x + 5, 410), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
    cv2.imwrite(str(destination / "endpoint_review.jpg"), clean, [cv2.IMWRITE_JPEG_QUALITY, 98])


def process(item):
    record, output = item
    cv2.setNumThreads(1)
    cv2.setRNGSeed(73019)
    window = record["decision"]["terminal_window"]
    lo, hi = window["frame_interval"]
    reference_index = record["shared_reference_frame"]
    reference = read_frames(record["shared_reference_video"], [reference_index])[reference_index]
    native_control = record["scope"] == "query_native_control"
    indices = [record["start_frame"] + i * record["stride"] if native_control else i for i in range(lo, hi + 1)]
    frames_by_index = read_frames(record["video"], indices)
    frames = [(i, frames_by_index[i]) for i in indices]
    duration = (hi - lo) * record["stride"] / record["source_fps"]
    if duration + 1e-8 < CONFIG["hold_seconds"]:
        raise ValueError("insufficient terminal source-clock coverage: " + record["key"])
    if record["start_frame"] + hi * record["stride"] >= record["total_frames"]:
        raise ValueError("padded frames cannot certify success: " + record["key"])
    report = {k: record[k] for k in ("key", "environment", "episode_index", "scope", "split", "method", "video")}
    report.update({"shared_reference_video": record["shared_reference_video"],
                   "shared_reference_frame": reference_index, "terminal_window": window,
                   "old_status": record["decision"]["status"], **evaluate(reference, frames, CONFIG)})
    destination = Path(output) / "cases" / record["key"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "analysis.json").write_text(json.dumps(report, indent=2))
    draw_review(reference, frames, report, destination)
    return report


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video evaluation must run on a CPU compute node")
    output = ROOT / ("outputs/reevaluation_stick_common_reference_rigid_job" + os.environ["SLURM_JOB_ID"])
    output.mkdir(exist_ok=False)
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2))
    records = json.loads(SOURCE.read_text())
    full = {(r["environment"], r["episode_index"]): r for r in records if r["scope"] == "full_episode"}
    tasks = []
    for source in records:
        record = dict(source)
        native = full[(record["environment"], record["episode_index"])]
        if not native.get("has_pre_lift_reference"):
            raise ValueError("Missing true GT pre-lift reference: " + native["key"])
        record["shared_reference_video"] = native["video"]
        record["shared_reference_frame"] = native["reference_frame"]
        tasks.append((record, str(output)))
        if record["scope"] == "query" and record["method"] == "gt":
            control = dict(record)
            control.update(key=record["key"] + "_native_control", scope="query_native_control", video=native["video"])
            tasks.append((control, str(output)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(process, tasks))
    counts = {str(int(t)): defaultdict(Counter) for t in CONFIG["small_tilt_sensitivity_deg"]}
    gates = defaultdict(Counter)
    pairs = defaultdict(dict)
    for row in results:
        group = row["scope"] + "/" + row["split"] + "/" + row["method"]
        for threshold, variant in row["diagnostic_variants"].items():
            counts[threshold][group][variant["status"]] += 1
        frames = row["terminal_frames"]
        sides = [s for f in frames for s in f["sides"]]
        for gate, value in {
            "flow_missing": any(not b.get("valid") for f in frames for b in f["bodies"]),
            "image_crop": any(s.get("actual_image_crop") for s in sides),
            "reference_roi_truncated": any(s.get("reference_roi_truncated") for s in sides),
            "shape_rigid_conflict": any(s.get("reason") == "appearance_rigid_disagreement" for s in sides),
            "angle_missing_or_conflicting": any(not f["tracked_angle"].get("valid") for f in frames),
            "all_bilateral_lower_positive": all(s.get("valid") and s.get("full_geometry_visible") and s["gap_lower_px"] >= 1 for s in sides),
        }.items():
            gates[group][gate] += int(value)
        if row["scope"] in ("query", "query_native_control"):
            pairs[row["key"].split("_")[0]]["native_gt" if row["scope"] == "query_native_control" else row["method"]] = row
    paired = {}
    controls = []
    for threshold in counts:
        counters = {m: Counter() for m in ("stage1", "stage2", "native_gt")}
        for key, methods in pairs.items():
            gt = methods["gt"]["diagnostic_variants"][threshold]["status"]
            for method, counter in counters.items():
                other = methods[method]["diagnostic_variants"][threshold]["status"]
                counter[gt + " -> " + other] += 1
                if method == "native_gt" and gt != other:
                    controls.append({"key": key, "tilt_limit": threshold, "sampled_gt": gt, "native_gt": other})
        paired[threshold] = {k: dict(v) for k, v in counters.items()}
    summary = {"job": os.environ["SLURM_JOB_ID"], "videos": len(results),
               "counts": {t: {g: dict(c) for g, c in groups.items()} for t, groups in counts.items()},
               "gate_counts": {g: dict(c) for g, c in gates.items()}, "paired": paired,
               "native_sampled_gt_disagreements": controls, "formal_metric_approved": False,
               "legacy_results_modified": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_video.json").write_text(json.dumps(results, indent=2))
    review = [{"key": r["key"], "scope": r["scope"], "old": r["old_status"],
               "variants": r["diagnostic_variants"], "review": str(output / "cases" / r["key"] / "endpoint_review.jpg")}
              for r in results if any(v["status"] == "balanced_lift_candidate" for v in r["diagnostic_variants"].values())]
    (output / "positive_review_manifest.json").write_text(json.dumps(review, indent=2))
    print(json.dumps({"output": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
