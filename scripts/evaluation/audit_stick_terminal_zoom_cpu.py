"""Prepare terminal-only manual review; do not alter the decision thresholds."""

import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/evaluation_stick_phase_interval_flow_job114259"


def stable_order(record):
    return hashlib.sha256(record["key"].encode()).hexdigest()


def stratified(records, limit):
    groups = defaultdict(list)
    for record in sorted(records, key=stable_order):
        groups[(record["scope"], record["method"], record["environment"], record["split"])].append(record)
    keys = sorted(groups, key=lambda key: hashlib.sha256(repr(key).encode()).hexdigest())
    selected = []
    while len(selected) < limit and keys:
        remaining = []
        for key in keys:
            if len(selected) >= limit:
                break
            selected.append(groups[key].pop(0))
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return selected


def make_sheet(task):
    record, destination = task
    cv2.setNumThreads(1)
    interval = record["decision"]["terminal_window"]["frame_interval"]
    selected = [record["reference_frame"], interval[0], (interval[0] + interval[1]) // 2, interval[1]]
    capture = cv2.VideoCapture(record["video"])
    frames = {}
    wanted = set(selected)
    for index in range(max(selected) + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Missing review frame {index}: {record['video']}")
        if index in wanted:
            frames[index] = cv2.resize(frame, (640, 480))
    capture.release()
    canvas = np.full((820, 1280, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, record["key"], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "LEFT endpoint: upper row; RIGHT endpoint: lower row. Fixed image coordinates.", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for column, index in enumerate(selected):
        source_frame = record["start_frame"] + index * record["stride"]
        label = ("REFERENCE ONLY" if column == 0 else "TERMINAL") + f" f{index} src{source_frame}"
        cv2.putText(canvas, label, (column * 320 + 5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1)
        frame = frames[index]
        canvas[90:450, column * 320:(column + 1) * 320] = frame[120:480, :320]
        canvas[460:820, column * 320:(column + 1) * 320] = frame[120:480, 320:640]
    path = Path(destination) / f"{record['key']}.jpg"
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"Could not write {path}")
    return {"key": record["key"], "sheet": str(path), "automatic_decision": record["decision"]["status"],
            "reviewed_frames": selected, "terminal_window": record["decision"]["terminal_window"],
            "manual_label": None, "manual_audit_complete": False}


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run video decoding on a CPU compute node, not a login node")
    output = ROOT / f"outputs/stick_terminal_zoom_audit_job{os.environ['SLURM_JOB_ID']}"
    output.mkdir(exist_ok=False)
    sheets = output / "cases"
    sheets.mkdir()
    records = json.loads((SOURCE / "per_video.json").read_text())
    terminal_states = defaultdict(Counter)
    endpoint_failures = defaultdict(Counter)
    uncertainty_categories = Counter()
    record_by_identity = {}
    for record in records:
        decision = record["decision"]
        interval = decision.get("terminal_window", {}).get("frame_interval")
        rows = [json.loads(line) for line in (SOURCE / "cases" / record["key"] / "tracks.jsonl").read_text().splitlines()]
        tail = [row for row in rows if interval and interval[0] <= row["frame"] <= interval[1]]
        group = f"{record['scope']}/{record['split']}/{record['method']}"
        terminal_states[group].update(row["state"] for row in tail)
        for row in tail:
            for side in row["sides"]:
                if not side.get("valid"):
                    endpoint_failures[group][side.get("reason", "unspecified")] += 1
        if decision["status"] == "unresolved":
            states = {row["state"] for row in tail}
            uncertainty_categories["tracking" if "unknown_tracking" in states else
                                   "partial_geometry" if "unknown_partial_geometry" in states else
                                   "threshold_or_mixed_terminal_states"] += 1
        record_by_identity[(record["scope"], record["environment"], record["episode_index"], record["method"])] = record
    consistency = Counter()
    disagreements = []
    for record in records:
        if record["scope"] != "query" or record["method"] != "gt":
            continue
        full = record_by_identity.get(("full_episode", record["environment"], record["episode_index"], "gt"))
        if full:
            pair = (full["decision"]["status"], record["decision"]["status"])
            consistency[" -> ".join(pair)] += 1
            if pair[0] != pair[1]:
                disagreements.append({"full": full["key"], "query": record["key"], "statuses": pair})
    positives = [record for record in records if record["decision"]["status"] == "balanced_lift_proposal"]
    selected = list(positives)
    for status in ("no_qualifying_lift_observed", "unresolved"):
        selected.extend(stratified([record for record in records if record["decision"]["status"] == status], 24))
    selected_by_key = {record["key"]: record for record in selected}
    for pair in disagreements:
        for key in (pair["full"], pair["query"]):
            selected_by_key[key] = next(record for record in records if record["key"] == key)
    selected = sorted(selected_by_key.values(), key=stable_order)
    summary = {"status": "awaiting_manual_terminal_review", "source": str(SOURCE),
               "terminal_frame_states": terminal_states, "terminal_endpoint_failures": endpoint_failures,
               "unresolved_video_categories": uncertainty_categories,
               "native_vs_subsampled_gt_consistency": consistency,
               "native_vs_subsampled_gt_disagreements": disagreements,
               "selected_videos": len(selected), "all_automatic_positives_included": len(positives),
               "decision_accuracy_measured": False, "formal_results_published": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(make_sheet, ((record, str(sheets)) for record in selected)))
    (output / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "review_cases": len(manifest),
                      "unresolved_video_categories": uncertainty_categories,
                      "gt_consistency": consistency}), flush=True)


if __name__ == "__main__":
    main()
