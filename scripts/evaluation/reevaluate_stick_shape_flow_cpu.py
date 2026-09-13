"""Reevaluate existing full GT and GT/Stage1/Stage2 query videos on CPU."""

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan_video_action.real97_eval.stick_shape_flow_review import evaluate

SOURCE = ROOT / "outputs/evaluation_stick_phase_interval_flow_job114259"
CONFIG = json.loads((ROOT / "configs/evaluation/stick_shape_flow_terminal_review_v2.json").read_text())


def read_frames(path, indices):
    wanted = set(indices)
    capture = cv2.VideoCapture(str(path))
    frames = {}
    for index in range(max(wanted) + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Missing frame {index}: {path}")
        if index in wanted:
            frames[index] = cv2.resize(frame, (640, 480))
    capture.release()
    return frames


def process(task):
    record, destination = task
    cv2.setNumThreads(1)
    cv2.setRNGSeed(73019)
    window = record["decision"]["terminal_window"]
    lo, hi = window["frame_interval"]
    if window["duration_seconds"] + 1e-8 < CONFIG["hold_seconds"]:
        raise RuntimeError(f"Insufficient terminal duration: {record['key']}")
    terminal = list(range(lo, hi + 1))
    if any(record["start_frame"] + index * record["stride"] >= record["total_frames"] for index in terminal):
        raise RuntimeError(f"Padding in terminal window: {record['key']}")
    reference_index = 0 if record["scope"] == "query" else record["reference_frame"]
    reference_video = record.get("gt_condition_video", record["video"])
    if record["start_frame"] + reference_index * record["stride"] >= record["lift_start"]:
        raise RuntimeError(f"Reference is not before lift: {record['key']}")
    if reference_video == record["video"]:
        frames = read_frames(record["video"], [reference_index] + terminal)
        reference = frames[reference_index]
    else:
        frames = read_frames(record["video"], terminal)
        reference = read_frames(reference_video, [reference_index])[reference_index]
    analysis = evaluate(reference, [(index, frames[index]) for index in terminal], CONFIG)
    report = {"key": record["key"], "environment": record["environment"], "episode_index": record["episode_index"],
              "scope": record["scope"], "split": record["split"], "method": record["method"],
              "video": record["video"], "reference_video": reference_video, "reference_frame": reference_index,
              "reference_kind": "gt_conditioning_frame" if record["scope"] == "query" else "pre_lift_gt_frame",
              "terminal_window": window, "old_status": record["decision"]["status"], **analysis}
    directory = Path(destination) / record["key"]
    directory.mkdir()
    (directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    tiles = [(reference_index, reference, analysis["reference_bodies"], analysis["reference_rod"], "REFERENCE")]
    for row in analysis["terminal_frames"]:
        gaps = [f"{side['gap_lower_px']:.1f}" if side.get("valid") else "?" for side in row["sides"]]
        angle = row["relative_rod_angle_deg"]
        label = f"f{row['frame']} low-gap {'/'.join(gaps)} tilt " + (f"{angle:.1f}" if angle is not None else "?")
        tiles.append((row["frame"], frames[row["frame"]], row["bodies"], row["rod"], label))
    columns = 4
    canvas = np.full((260 * ((len(tiles) + columns - 1) // columns), 320 * columns, 3), 255, np.uint8)
    for position, (_, frame, bodies, rod, label) in enumerate(tiles):
        image = frame.copy()
        for side, body in enumerate(bodies):
            if body["valid"]:
                cv2.polylines(image, [np.rint(body["hull"]).astype(np.int32)], True,
                              (255, 160, 0) if side == 0 else (0, 80, 255), 1)
        for line in rod.get("lines", []):
            x0, y0, x1, y1 = [int(round(value)) for value in line["line"]]
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 255), 1)
        x, y = position % columns * 320, position // columns * 260
        cv2.putText(canvas, label, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 1)
        canvas[y + 20:y + 260, x:x + 320] = cv2.resize(image, (320, 240))
    if not cv2.imwrite(str(directory / "terminal_review.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"Failed to save review image: {record['key']}")
    return report


def main():
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("Use a CPU compute node")
    output = ROOT / f"outputs/reevaluation_stick_shape_flow_job{job}"
    output.mkdir(exist_ok=False)
    cases = output / "cases"
    cases.mkdir()
    records = json.loads((SOURCE / "per_video.json").read_text())
    queries = {record["key"].split("_")[0]: record for record in records if record["scope"] == "query" and record["method"] == "gt"}
    for record in records:
        if record["scope"] == "query":
            record["gt_condition_video"] = queries[record["key"].split("_")[0]]["video"]
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(process, ((record, str(cases)) for record in records)))
    groups = defaultdict(lambda: defaultdict(Counter))
    failures = Counter()
    pairs = defaultdict(dict)
    for report in reports:
        group = f"{report['scope']}/{report['split']}/{report['method']}"
        for threshold, status in report["diagnostic_variants"].items():
            groups[threshold][group][status] += 1
        for row in report["terminal_frames"]:
            for body in row["bodies"]:
                if not body["valid"]:
                    failures[body.get("reason", "unknown")] += 1
        if report["scope"] == "query":
            pairs[report["key"].split("_")[0]][report["method"]] = report
    agreement = defaultdict(lambda: defaultdict(Counter))
    for group in pairs.values():
        for method in ("stage1", "stage2"):
            for threshold in group["gt"]["diagnostic_variants"]:
                gt = group["gt"]["diagnostic_variants"][threshold]
                pred = group[method]["diagnostic_variants"][threshold]
                agreement[threshold][method][f"{gt} -> {pred}"] += 1
    summary = {"status": "review_before_formal_use", "videos": len(reports), "groups": groups,
               "gt_prediction_agreement": agreement, "body_tracking_failures": failures,
               "formal_metric_approved": False, "formal_results_published": False,
               "legacy_results_overwritten": False,
               "notes": ["No rise-ratio success gate.", "Query methods share the corresponding actual GT conditioning frame as reference.",
                         "3 and 5 degrees are sensitivity variants, not independently validated success definitions.",
                         "Unknown is not a negative and not a success."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "per_video.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
