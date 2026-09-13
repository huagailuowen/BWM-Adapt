#!/usr/bin/env python3
"""Audit local footpoint measurements without changing formal success labels."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from real97_trial_common import ROOT, require_compute, write_json
from reevaluate_stick_shape_flow_cpu import read_frames
from wan_video_action.real97_eval.stick_local_contact_edges import reference_feet, measure


def quantiles(values):
    return dict(count=len(values), q10_q50_q90=np.quantile(values, [.1, .5, .9]).tolist() if values else [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    cv2.setNumThreads(4)
    config = json.loads((ROOT / "configs/evaluation/stick_local_contact_edges_v6.json").read_text())
    sources = json.loads((ROOT / config["source"]).read_text())
    manual = json.loads((ROOT / config["manual_labels"]).read_text())
    label_by_key = {row["key"]: row for row in manual["rows"]}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "manual_labels.json", manual)
    results = []
    groups = defaultdict(list)
    for index, source in enumerate(sources):
        terminal = source["terminal_frames"]
        record = {key: source[key] for key in ("key", "scope", "method", "environment", "episode_index", "video")}
        record.update(frames=[], formal_metric_approved=False)
        shapes = source["reference_bodies"]
        if not terminal or not all(shape.get("valid") for shape in shapes):
            record["reason"] = "missing_terminal_or_reference_shape"
            results.append(record)
            continue
        reference = read_frames(source["reference_video"], [source["reference_frame"]])[source["reference_frame"]]
        frames = read_frames(source["video"], sorted({int(row["frame"]) for row in terminal}))
        hulls = [np.asarray(shape["hull"], dtype=float).reshape(-1, 2) for shape in shapes]
        centers = [np.mean(hull[:, 0]) for hull in hulls]
        slope = float((hulls[1][:, 1].max() - hulls[0][:, 1].max()) / max(1, centers[1] - centers[0]))
        feet = [reference_feet(reference, shapes[side], side, slope, config) for side in range(2)]
        for row in terminal:
            current = frames[int(row["frame"])]
            sides = []
            for side in range(2):
                pose = row["direct"][side]
                pose_source = "direct"
                if not pose.get("valid"):
                    pose, pose_source = row["sequential"][side], "sequential"
                value = measure(current, shapes[side], feet[side], pose, side, slope, config)
                value["pose_source"] = pose_source
                sides.append(value)
            record["frames"].append(dict(frame=row["frame"], source_frame=row["source_frame"], sides=sides))
        record["reference_footpoints"] = feet
        record["reference_floor_slope"] = slope
        record["manual_contact_only"] = label_by_key.get(source["key"])
        record["terminal_side_summary"] = []
        for side, side_name in enumerate(("left", "right")):
            values = [row["sides"][side]["gap_px"] for row in record["frames"] if row["sides"][side].get("valid")]
            summary = dict(valid_frames=len(values), total_frames=len(terminal),
                           minimum_gap_px=min(values) if values else None,
                           maximum_gap_px=max(values) if values else None,
                           median_gap_px=float(np.median(values)) if values else None)
            record["terminal_side_summary"].append(summary)
            if record["manual_contact_only"] and values:
                groups[record["manual_contact_only"][side_name]].append(summary["median_gap_px"])
        directory = args.output / "cases" / source["key"]
        write_json(directory / "audit.json", record)
        # Independent diagnostic overlay, never replace the previous review images.
        last = frames[int(terminal[-1]["frame"])].copy()
        first = reference.copy()
        for side in range(2):
            for point in feet[side]:
                cv2.circle(first, (round(point["x"]), round(point["y"])), 2, (0, 255, 255), -1)
            for point in record["frames"][-1]["sides"][side]["points"]:
                cur = point["current"]
                cv2.circle(last, (round(cur["x"]), round(cur["y"])), 2, (0, 255, 255), -1)
        panel = np.hstack([first, last])
        cv2.putText(panel, "GT reference | terminal: bottom-edge candidates, NOT success labels", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(directory / "contact_edges_review.jpg"), panel)
        compact = {key: value for key, value in record.items() if key not in ("frames", "reference_footpoints")}
        results.append(compact)
        if (index + 1) % 40 == 0:
            print(f"[contact_edge_audit] {index+1}/{len(sources)}", flush=True)
    write_json(args.output / "per_video.json", results)
    write_json(args.output / "summary.json", dict(
        videos=len(results), manual_videos=len(manual["rows"]),
        manual_contact_label_gap_distributions={key: quantiles(values) for key, values in groups.items()},
        formal_metric_approved=False, success_labels_changed=False,
        limitations="Manual labels concern visible first/last contact only, not all terminal frames or calibrated angle. This is not a success-rate estimate.",
    ))
    print(f"[complete] {args.output}", flush=True)


if __name__ == "__main__":
    main()
