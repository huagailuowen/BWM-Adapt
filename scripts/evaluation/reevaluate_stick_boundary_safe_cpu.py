#!/usr/bin/env python3
"""Independent v8 diagnostic; all video decoding runs in a CPU allocation."""

import argparse
from collections import Counter, defaultdict
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
from reevaluate_stick_shape_flow_cpu import read_frames
from reevaluate_stick_contact_decision_cpu import blind_panel
from wan_video_action.real97_eval.stick_boundary_safe_edges import boundary_safe_search, legacy


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def side_state(side, config):
    if not side.get("valid") or side.get("visible_footpoint_count", 0) < config["minimum_visible_footpoints"]:
        return "unknown"
    gap = side["gap_px"]
    if config["contact_gap_min_px"] <= gap <= config["contact_gap_max_px"]:
        return "contact"
    if not side.get("reference_cropped") and gap >= config["lift_gap_min_px"]:
        return "off_table"
    return "unknown"


def classify(source, edge, config):
    measurements = {int(row["frame"]): row for row in edge.get("frames", [])}
    frames = []
    for row in source.get("terminal_frames", []):
        sides = measurements.get(int(row["frame"]), {}).get("sides", [{}, {}])
        frames.append({"frame": row["frame"], "states": [side_state(side, config) for side in sides],
                       "gaps_px": [side.get("gap_px") for side in sides], "angle": row.get("fused_angle", {})})
    variants = {}
    for threshold in config["angle_thresholds_deg"]:
        contact = sum("contact" in row["states"] for row in frames)
        excessive_tilt = sum(bool(row["angle"].get("valid")) and
                             row["angle"]["absolute_lower_deg"] > threshold + config["angle_failure_margin_deg"] for row in frames)
        positive = bool(frames) and all(row["states"] == ["off_table", "off_table"] and row["angle"].get("valid") and
                                      row["angle"]["absolute_upper_deg"] <= threshold for row in frames)
        required = max(2, math.ceil(len(frames) * config["negative_persistence_fraction"]))
        if positive:
            state, reason = "positive", "entire_terminal_window"
        elif contact >= required:
            state, reason = "negative", "persistent_visible_contact"
        elif excessive_tilt >= required:
            state, reason = "negative", "persistent_excessive_tilt"
        else:
            state, reason = "unknown", "insufficient_contact_or_angle_evidence"
        variants[str(threshold)] = {"state": state, "reason": reason}
    return {"variants": variants, "frames": frames}


def collect_keys(value, valid_keys):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in valid_keys:
                found.add(key)
            found.update(collect_keys(item, valid_keys))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_keys(item, valid_keys))
    elif isinstance(value, str) and value in valid_keys:
        found.add(value)
    return found


def add_contour_diagnostic(side):
    selected_x = set(side.get("minimum_group_reference_x", []))
    selected = sorted((point for point in side.get("points", []) if point["reference"]["x"] in selected_x),
                      key=lambda point: point["reference"]["x"])
    side["selected_contour_diagnostic_only"] = {
        "reference_max_adjacent_y_jump_px": max((abs(a["reference"]["y"] - b["reference"]["y"]) for a, b in zip(selected, selected[1:])), default=None),
        "current_max_adjacent_y_jump_px": max((abs(a["current"]["y"] - b["current"]["y"]) for a, b in zip(selected, selected[1:])), default=None),
        "affects_decision": False,
    }


def overlay(reference, current, audit, mode):
    images = [reference.copy(), current.copy()]
    last = audit.get("frames", [])[-1] if audit.get("frames") else {"sides": []}
    for side_index, side in enumerate(last["sides"]):
        selected = set(side.get("minimum_group_reference_x", []))
        for point in side.get("points", []):
            chosen = point["reference"]["x"] in selected
            for image, field in zip(images, ("reference", "current")):
                item = point[field]
                cv2.circle(image, (round(item["x"]), round(item["y"])), 3 if chosen else 1,
                           (0, 0, 255) if chosen else (0, 220, 220), -1)
        label = f"{'L' if side_index == 0 else 'R'} gap={side.get('gap_px')} valid={side.get('valid')}"
        cv2.putText(images[1], label, (10, 30 + 25 * side_index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 230), 1)
    for image, label in zip(images, ("Native reference", mode)):
        cv2.putText(image, label, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)
    return np.hstack(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video reevaluation must run in a Slurm compute allocation")
    os.chdir(ROOT)
    cv2.setNumThreads(4)
    config = read_json(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_config.json", config)
    sources = read_json(config["source_geometry"])
    by_key = {row["key"]: row for row in sources}
    manual_keys = collect_keys(read_json(config["source_manual_review"]), set(by_key))
    old_manifest = read_json(config["source_review_manifest"])
    counts = defaultdict(Counter)
    search_counts = defaultdict(Counter)
    records, changes = [], []
    for index, source in enumerate(sources):
        key = source["key"]
        old_path = Path(config["source_edges"]) / "cases" / key / "audit.json"
        old_edge = read_json(old_path) if old_path.exists() else {"frames": []}
        previous = classify(source, old_edge, config)
        record = {name: source.get(name) for name in ("key", "scope", "split", "method", "environment", "episode_index", "video")}
        record.update(formal_metric_approved=False, decisions={"v7_reproduced": previous})
        terminal = source.get("terminal_frames", [])
        shapes = source.get("reference_bodies", [])
        can_measure = bool(terminal) and len(shapes) == 2 and all(shape.get("valid") for shape in shapes)
        if can_measure:
            reference = read_frames(source["reference_video"], [source["reference_frame"]])[source["reference_frame"]]
            video_frames = read_frames(source["video"], [row["frame"] for row in terminal])
            hulls = [np.asarray(shape["hull"], dtype=float) for shape in shapes]
            slope = float((hulls[1][:, 1].max() - hulls[0][:, 1].max()) /
                          (hulls[1][:, 0].mean() - hulls[0][:, 0].mean()))
        for mode, radii in config["search_modes"].items():
            audit = {"key": key, "mode": mode, "frames": [], "formal_metric_approved": False}
            if can_measure:
                with boundary_safe_search(radii, config["search_boundary_margin_px"]) as search:
                    feet = [legacy.reference_feet(reference, shape, side, slope, config) for side, shape in enumerate(shapes)]
                    audit["reference_footpoints"] = feet
                    audit["reference_search_events"] = list(search.events)
                    offset = len(search.events)
                    for row in terminal:
                        measured = []
                        for side in range(2):
                            direct = row.get("direct", [{}, {}])[side]
                            sequential = row.get("sequential", [{}, {}])[side]
                            pose = direct if direct.get("valid") else sequential
                            result = legacy.measure(video_frames[row["frame"]], shapes[side], feet[side], pose, side, slope, config)
                            result["pose_source"] = "direct" if direct.get("valid") else "sequential"
                            result["search_events"] = search.events[offset:]
                            offset = len(search.events)
                            add_contour_diagnostic(result)
                            measured.append(result)
                        audit["frames"].append({"frame": row["frame"], "source_frame": row.get("source_frame"), "sides": measured})
                    audit["search_summary"] = search.summary()
                    search_counts[mode].update(search.summary()["outcomes"])
                audit["reference_floor_slope"] = slope
            else:
                audit["reason"] = "missing_terminal_or_reference_geometry"
            decision = classify(source, audit, config)
            record["decisions"][mode] = decision
            case_dir = output / "cases" / key / mode
            write_json(case_dir / "audit.json", audit)
            if can_measure and (key in manual_keys or previous["variants"]["5"]["state"] == "positive" or decision["variants"]["5"]["state"] == "positive"):
                cv2.imwrite(str(case_dir / "contact_edges_review.jpg"), overlay(reference, video_frames[terminal[-1]["frame"]], audit, mode))
            for threshold in config["angle_thresholds_deg"]:
                old = previous["variants"][str(threshold)]["state"]
                new = decision["variants"][str(threshold)]["state"]
                if old != new:
                    changes.append({"key": key, "mode": mode, "threshold_deg": threshold, "previous": old, "current": new,
                                    "reason": decision["variants"][str(threshold)]["reason"]})
        for mode, decision in record["decisions"].items():
            for threshold, variant in decision["variants"].items():
                bucket = f"{mode}/{source['scope']}/{source['method']}/{threshold}"
                counts[bucket][variant["state"]] += 1
        records.append(record)
        if (index + 1) % 25 == 0:
            print(json.dumps({"videos_completed": index + 1, "total": len(sources)}), flush=True)
    write_json(output / "per_video.json", records)
    write_json(output / "changed_cases.json", changes)
    baseline_mismatches = []
    for bucket, expected in config["expected_v7_counts"].items():
        actual = {state: counts[f"v7_reproduced/{bucket}"][state] for state in ("positive", "negative", "unknown")}
        if actual != expected:
            baseline_mismatches.append({"bucket": bucket, "expected": expected, "actual": actual})
    review = []
    seen = set()
    for original in old_manifest:
        key = original.get("key")
        if key not in manual_keys or key in seen or not by_key[key].get("terminal_frames"):
            continue
        label = Path(original.get("image", "")).stem or original.get("label", f"review{len(review):03d}")
        path = output / "blind_review" / f"{label}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), blind_panel(by_key[key], label))
        review.append({"label": label, "key": key, "image": str(path), "cohort": "same_previously_reviewed_52"})
        seen.add(key)
    for change in changes:
        key = change["key"]
        if key in seen or change["threshold_deg"] != 5 or by_key[key]["scope"] != "query":
            continue
        label = f"changed{len(review):03d}"
        path = output / "blind_review" / f"{label}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), blind_panel(by_key[key], label))
        review.append({"label": label, "key": key, "image": str(path), "cohort": "targeted_changed_query_not_random"})
        seen.add(key)
    write_json(output / "review_manifest.json", review)
    summary = {"videos": len(records), "counts": dict(counts), "search_outcomes": dict(search_counts),
               "changed_decisions": len(changes), "review_cases": len(review), "baseline_mismatches": baseline_mismatches,
               "formal_metric_approved": False,
               "interpretation": "Coverage is not accuracy. Expanded searches and small gaps still require visual review."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
