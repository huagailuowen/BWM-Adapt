#!/usr/bin/env python3
"""Re-score existing Stick videos at their visible real-frame tail, no re-inference."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from real916_stick_final_common import write_json
from score_real97_stick_action_precision_cpu import video_decision, native_reference
from score_real916_stick_final import summarize, export_table

METHODS = ("standard", "ours_stage1", "ours_stage2")


def read(path):
    return json.loads(Path(path).read_text())


def process(task):
    config, original = task
    cv2.setNumThreads(1)
    output = ROOT / config["output"]
    key = original["key"]
    target = output / "per_query" / (key + ".json")
    if target.exists():
        return read(target)
    result = copy.deepcopy(original)
    for identity in result["input_identity"].values():
        stat = Path(identity["path"]).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise RuntimeError("Source video changed: " + key)
    wanted = result["wanted"]
    hi = wanted[-1]
    count = math.ceil(config["hold_seconds"] * 20 / 3 - 1e-8)
    lo = hi - count
    if lo < 1 or any(i not in wanted for i in range(lo, hi + 1)):
        raise RuntimeError("Insufficient real terminal frames: " + key)
    phase = copy.deepcopy(result["phase"])
    phase["decision"] = {
        "terminal_window": {"frame_interval": [lo, hi]},
        "horizon_status": "visible_real_video_tail",
    }
    result["previous_action"] = result["action"]
    result["previous_action_eligible"] = result["action_eligible"]
    result["previous_phase"] = result["phase"]
    result["phase"] = phase
    result["action_eligible"] = True
    result["action"] = {}
    result["action_protocol"] = "last_real_0.3_seconds_no_commanded_lift_end_gate"
    row = task[0]["query_rows"][str(result["index"])]
    reference = native_reference(
        Path(config["source_dataset"]) / row["video"][0],
        phase["native_reference_frame"],
    )
    geometry = {**read(ROOT / config["geometry_config"]), **read(ROOT / config["fusion_config"])}
    contact = {**read(ROOT / config["contact_config"]), "angle_thresholds_deg": [3, 5]}
    audits = {}
    method_rows = []
    for method in ("gt",) + METHODS:
        record, ref, terminal = video_decision(
            result["input_identity"][method]["path"], reference, phase, contact, geometry
        )
        result["action"][method] = record["decision"]["variants"]
        audits[method] = record
        panels = []
        for i, frame in enumerate([ref] + terminal):
            panel = cv2.copyMakeBorder(cv2.resize(frame, (320, 240)), 34, 0, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(245, 245, 245))
            if i == 0:
                label = method + " | common pre-lift reference"
            else:
                detail = record["decision"]["frames"][i-1]
                gaps = detail["gaps_px"]
                label = f"{method} f{lo+i-1} gaps=" + "/".join(
                    "?" if x is None else f"{x:.1f}" for x in gaps
                )
            cv2.putText(panel, label, (4, 14), 0, .36, (20, 20, 20), 1)
            cv2.putText(panel, "action5=" + record["decision"]["variants"]["5"]["state"],
                        (4, 29), 0, .36, (20, 20, 20), 1)
            panels.append(panel)
        method_rows.append(np.hstack(panels))
    review = output / "reviews" / (key + ".jpg")
    review.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(review), np.vstack(method_rows)):
        raise RuntimeError("Failed to save evidence")
    result["review"] = str(review)
    result["manual_review_status"] = "pending"
    write_json(output / "contact" / (key + ".json"), audits)
    write_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compute-node allocation required")
    os.chdir(ROOT)
    config = read(args.config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    frozen = output / "protocol.json"
    protocol = {**config, "config_sha256": digest, "no_video_regeneration": True,
                "reference": "same native pre-lift reference for GT and all methods",
                "decision_tail": "last 0.3 seconds of non-padding video frames",
                "commanded_lift_end_gate": False,
                "classification_thresholds_changed": False}
    if frozen.exists() and read(frozen) != protocol:
        raise RuntimeError("Refusing changed protocol")
    write_json(frozen, protocol)
    source = ROOT / config["source_metrics"]
    records = read(source / "per_query.json")
    original_summary = read(source / "summary.json")
    plan = read(ROOT / config["prepared"] / "plan.json")
    config["source_dataset"] = plan["source_dataset"]
    config["query_rows"] = {
        str(i): json.loads(line) for i, line in enumerate(
            (ROOT / config["prepared"] / "query.jsonl").read_text().splitlines()) if line.strip()
    }
    results = []
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        for result in pool.map(process, [(config, r) for r in records]):
            results.append(result)
            print("[rescored]", len(results), "/", len(records), result["key"],
                  result["action"]["gt"]["5"], flush=True)
    summary = summarize(results)
    for group, methods in summary["groups"].items():
        for method, metrics in methods.items():
            for field in ("lpips", "lpips_queries"):
                if field in original_summary["groups"][group][method]:
                    metrics[field] = original_summary["groups"][group][method][field]
    summary.update(
        lpips_complete=original_summary["lpips_complete"],
        lpips_protocol=original_summary.get("lpips_protocol"),
        action_protocol="last_real_0.3_seconds_no_commanded_lift_end_gate",
        unchanged_metrics_source=str(source),
        formal_metric_approved=False, visual_audit_pending=True,
        notes=["Only action labels were re-evaluated; image/object/LPIPS scores are unchanged.",
               "Unknown means insufficient visual evidence, never a metadata timing mismatch.",
               "GT outcome metadata is diagnostic only, not a replacement for video judgement."],
    )
    changed = []
    unknown = []
    for r in results:
        for method in ("gt",) + METHODS:
            old = r["previous_action"][method]["5"]
            new = r["action"][method]["5"]
            if old != new:
                changed.append({"key": r["key"], "method": method, "before": old, "after": new})
            if new["state"] == "unknown":
                unknown.append({"key": r["key"], "method": method, "reason": new["reason"],
                                "review": r["review"]})
    write_json(output / "per_query.json", results)
    write_json(output / "summary.json", summary)
    write_json(output / "changed_labels.json", changed)
    write_json(output / "unknown_cases.json", unknown)
    export_table(summary, output)
    # All 45 test GTs, not just favorable/changed cases, are included in review sheets.
    tests = [r for r in results if r["split"] == "test"]
    sheets = []
    for start in range(0, len(tests), 6):
        selected = tests[start:start+6]
        tiles = []
        for r in selected:
            evidence = cv2.imread(r["review"])
            gt = evidence[:274].copy()
            cv2.putText(gt, r["key"], (4, 270), 0, .45, (0, 0, 200), 1)
            tiles.append(gt)
        path = output / "gt_review" / f"page_{start//6:02d}.jpg"
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), np.vstack(tiles)):
            raise RuntimeError("Failed to save GT review sheet")
        sheets.append({"path": str(path), "keys": [r["key"] for r in selected]})
    write_json(output / "gt_review_manifest.json", sheets)
    write_json(output / "automatic_complete.json", {
        "queries": len(results), "test_queries": len(tests),
        "gt_test_counts": dict(Counter(r["action"]["gt"]["5"]["state"] for r in tests)),
        "manual_review_pending": True,
    })
    print(json.dumps(summary["groups"]["test"]), flush=True)


if __name__ == "__main__":
    main()
