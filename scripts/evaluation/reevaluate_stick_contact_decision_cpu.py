#!/usr/bin/env python3
"""Frozen contact candidate, old angle intervals, independent blind review."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np

from real97_trial_common import ROOT, require_compute, write_json
from reevaluate_stick_shape_flow_cpu import read_frames


def side_state(side, config):
    if not side.get("valid"):
        return "unknown"
    if side.get("visible_footpoint_count", 0) < config["minimum_visible_footpoints"]:
        return "unknown"
    gap = float(side["gap_px"])
    if config["contact_gap_min_px"] <= gap <= config["contact_gap_max_px"]:
        return "contact"
    if not side.get("reference_cropped", True) and gap >= config["lift_gap_min_px"]:
        return "off_table"
    return "unknown"


def classify(source, edge, config):
    lookup = {int(row["frame"]): row for row in edge.get("frames", [])}
    frames = []
    for row in source["terminal_frames"]:
        measured = lookup.get(int(row["frame"]), {}).get("sides", [{}, {}])
        states = [side_state(value, config) for value in measured]
        frames.append(dict(frame=row["frame"], source_frame=row["source_frame"],
                           contacts=states, gap_px=[value.get("gap_px") for value in measured],
                           angle=row["fused_angle"]))
    variants = {}
    for threshold in config["angle_thresholds_deg"]:
        key = str(int(threshold))
        positive, contact_witness, tilt_witness = [], [], []
        for row in frames:
            angle = row["angle"]
            small = angle.get("valid") and angle["absolute_upper_deg"] <= threshold
            tilted = angle.get("valid") and angle["absolute_lower_deg"] > threshold + config["angle_failure_margin_deg"]
            positive.append(row["contacts"] == ["off_table", "off_table"] and bool(small))
            contact_witness.append("contact" in row["contacts"])
            tilt_witness.append(bool(tilted))
        required = max(2, math.ceil(len(frames) * config["negative_persistence_fraction"]))
        if frames and all(positive):
            status, reason = "positive", "both_ends_off_table_and_small_tilt_entire_terminal_window"
        elif sum(contact_witness) >= required:
            status, reason = "negative", "persistent_visible_contact"
        elif sum(tilt_witness) >= required:
            status, reason = "negative", "persistent_excessive_tilt"
        else:
            status, reason = "unknown", "insufficient_or_conflicting_terminal_evidence"
        variants[key] = dict(status=status, reason=reason, positive_frames=sum(positive),
                             contact_frames=sum(contact_witness), excessive_tilt_frames=sum(tilt_witness),
                             total_frames=len(frames), required_negative_frames=required)
    return dict(variants=variants, frames=frames)


def fit(image, width, height):
    factor = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*factor)), max(1, round(image.shape[0]*factor))))
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def blind_panel(source, label):
    indices = [int(row["frame"]) for row in source["terminal_frames"]]
    if not indices:
        return None
    reference = read_frames(source["reference_video"], [int(source["reference_frame"])])[int(source["reference_frame"])]
    frames = read_frames(source["video"], indices)
    top = np.hstack([fit(reference, 480, 360), fit(frames[indices[-1]], 480, 360)])
    cv2.putText(top, f"{label}: REFERENCE / FINAL", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 220), 1, cv2.LINE_AA)
    strips = []
    for side in range(2):
        shape = source["reference_bodies"][side]
        if not shape.get("valid"):
            strips.append(np.full((130, 960, 3), 230, np.uint8))
            continue
        hull = np.asarray(shape["hull"], dtype=float).reshape(-1, 2)
        points = [hull]
        for row in source["terminal_frames"]:
            pose = row["direct"][side]
            if not pose.get("valid"):
                pose = row["sequential"][side]
            if pose.get("valid"):
                matrix = np.asarray(pose["matrix"], dtype=float)
                points.append(hull @ matrix[:, :2].T + matrix[:, 2])
        cloud = np.vstack(points)
        xmin, xmax = (0, 320) if side == 0 else (320, 640)
        x0 = max(xmin, int(cloud[:, 0].min())-12)
        x1 = min(xmax, int(cloud[:, 0].max())+12)
        y0 = max(0, int(cloud[:, 1].min())-10)
        y1 = min(480, int(cloud[:, 1].max())+22)
        cells = []
        for j, image in enumerate([reference] + [frames[index] for index in indices]):
            cell = fit(image[y0:y1, x0:x1], 120, 112)
            header = np.full((18, 120, 3), 255, np.uint8)
            text = ("L" if side == 0 else "R") + (" REF" if j == 0 else f" {j}/{len(indices)}")
            cv2.putText(header, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            cells.append(np.vstack([header, cell]))
        while len(cells) < 8:
            cells.append(np.full((130, 120, 3), 255, np.uint8))
        strips.append(np.hstack(cells[:8]))
    return np.vstack([top] + strips)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    cv2.setNumThreads(4)
    config_path = ROOT / "configs/evaluation/stick_terminal_contact_decision_v7.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    sources = json.loads((ROOT / config["source_geometry"]).read_text())
    edge_root = ROOT / config["source_edges"]
    labels = json.loads((ROOT / config["development_labels"]).read_text())
    development = {row["key"] for row in labels["rows"]}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "frozen_config.json", config)
    results, counts, reasons = [], defaultdict(Counter), defaultdict(Counter)
    source_by_key = {row["key"]: row for row in sources}
    for source in sources:
        path = edge_root / "cases" / source["key"] / "audit.json"
        edge = json.loads(path.read_text()) if path.is_file() else {}
        record = {key: source[key] for key in ("key", "scope", "method", "environment", "episode_index", "video")}
        record.update(classify(source, edge, config))
        record["development_contact_reviewed"] = source["key"] in development
        record["formal_metric_approved"] = False
        record["previous_variants"] = source["diagnostic_variants"]
        results.append(record)
        for threshold, decision in record["variants"].items():
            group = f"{source['scope']}/{source['method']}/{threshold}deg"
            counts[group][decision["status"]] += 1
            reasons[group][decision["reason"]] += 1
    by_key = {row["key"]: row for row in results}
    grouped_native = defaultdict(list)
    for row in results:
        if row["scope"] == "full_episode" and row["key"] not in development:
            grouped_native[row["environment"]].append(row["key"])
    rng = random.Random(config["seed"])
    selected = []
    for environment, keys in sorted(grouped_native.items()):
        for key in rng.sample(sorted(keys), min(len(keys), config["native_blind_samples_per_environment"])):
            selected.append((key, "independent_random_native_holdout"))
    query_ids = sorted({row["key"].split("_")[0] for row in results if row["scope"] == "query"})
    sampled_ids = set(rng.sample(query_ids, min(len(query_ids), config["query_blind_count"])))
    for row in results:
        if row["scope"] == "query" and row["key"].split("_")[0] in sampled_ids:
            selected.append((row["key"], "paired_query_method_audit"))
    already = {key for key, _ in selected}
    for row in results:
        if row["scope"] == "full_episode" and row["variants"]["5"]["status"] in ("positive", "unknown") and row["key"] not in already:
            selected.append((row["key"], "targeted_positive_or_unknown_not_prevalence_sample"))
    rng.shuffle(selected)
    manifest = []
    for index, (key, purpose) in enumerate(selected):
        label = f"audit{index:03d}"
        panel = blind_panel(source_by_key[key], label)
        if panel is None:
            continue
        destination = args.output / "blind_review" / (label + ".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), panel)
        manifest.append(dict(label=label, key=key, purpose=purpose, image=str(destination),
                             candidate=by_key[key]["variants"]))
    pairs = defaultdict(dict)
    for row in results:
        if row["scope"] == "query":
            pairs[row["key"].split("_")[0]][row["method"]] = row["variants"]
    native_control_disagreements = []
    for row in results:
        if row["scope"] != "query_native_control":
            continue
        key = row["key"].removesuffix("_native_control")
        if key in by_key:
            differences = [threshold for threshold in ("3", "5") if row["variants"][threshold]["status"] != by_key[key]["variants"][threshold]["status"]]
            if differences:
                native_control_disagreements.append(dict(native_key=row["key"], query_key=key, thresholds=differences))
    write_json(args.output / "per_video.json", results)
    write_json(args.output / "paired_queries.json", pairs)
    write_json(args.output / "blind_review_manifest.json", manifest)
    write_json(args.output / "summary.json", dict(
        videos=len(results), counts=dict(counts), reasons=dict(reasons),
        source_metric_version="v4 geometry plus v6 local edges",
        native_control_disagreements=native_control_disagreements,
        independent_native_review_count=sum(row["purpose"] == "independent_random_native_holdout" for row in manifest),
        review_images=len(manifest), frozen_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        formal_metric_approved=False, review_complete=False,
        warning="Coverage is not correctness. Development/targeted review is not a population success estimate.",
    ))
    print(f"[complete] {args.output} videos={len(results)} blind_images={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
