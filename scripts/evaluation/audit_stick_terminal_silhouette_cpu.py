"""Run an independent terminal silhouette/rod audit on every full GT episode."""

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

from wan_video_action.real97_eval.stick_terminal_silhouette import analyze_frames

SOURCE = ROOT / "outputs/evaluation_stick_phase_interval_flow_job114259"
CONFIG = json.loads((ROOT / "configs/evaluation/stick_terminal_silhouette_diagnostic_v1.json").read_text())


def process(task):
    record, destination = task
    cv2.setNumThreads(1)
    interval = record["decision"]["terminal_window"]["frame_interval"]
    terminal = list(range(interval[0], interval[1] + 1))
    reference = record["reference_frame"]
    wanted = set([reference] + terminal)
    capture = cv2.VideoCapture(record["video"])
    frames = {}
    for index in range(max(wanted) + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Missing source frame {index}: {record['key']}")
        if index in wanted:
            frames[index] = cv2.resize(frame, (640, 480))
    capture.release()
    analysis = analyze_frames(frames, reference, terminal, CONFIG)
    report = {"key": record["key"], "environment": record["environment"], "episode_index": record["episode_index"],
              "split": record["split"], "video": record["video"], "reference_frame": reference,
              "terminal_window": record["decision"]["terminal_window"],
              "old_status": record["decision"]["status"], **analysis,
              "human_label": None, "human_audit_complete": False}
    directory = Path(destination) / record["key"]
    directory.mkdir()
    (directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    selected = [reference] + terminal
    width, height = 320, 260
    columns = 4
    canvas = np.full((height * ((len(selected) + columns - 1) // columns), width * columns, 3), 255, np.uint8)
    for position, index in enumerate(selected):
        image = frames[index].copy()
        if index == reference:
            bodies, rod = analysis["reference_bodies"], analysis["reference_rod"]
            label = f"REFERENCE f{index}"
        else:
            row = next(row for row in analysis["terminal_frames"] if row["frame"] == index)
            bodies, rod = row["bodies"], row["rod"]
            gaps = [f"{side['gap_lower_px']:.1f}" if side.get("valid") else "?" for side in row["sides"]]
            label = f"f{index} low-gap L/R {'/'.join(gaps)}"
        for side, body in enumerate(bodies):
            if body["candidates"]:
                candidate = min(body["candidates"], key=lambda item: abs(item["threshold"] - 85))
                cv2.polylines(image, [np.rint(candidate["hull"]).astype(np.int32)], True, (255, 160, 0) if side == 0 else (0, 80, 255), 1)
        for line in rod.get("lines", []):
            x0, y0, x1, y1 = map(lambda value: int(round(value)), line["line"])
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 255), 1)
        x, y = (position % columns) * width, (position // columns) * height
        cv2.putText(canvas, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        canvas[y + 20:y + height, x:x + width] = cv2.resize(image, (320, 240))
    if not cv2.imwrite(str(directory / "terminal_overlay.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"Cannot write review image: {record['key']}")
    return report


def main():
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("Bulk video analysis must run on a CPU compute node")
    output = ROOT / f"outputs/stick_terminal_silhouette_audit_job{job}"
    output.mkdir(exist_ok=False)
    cases = output / "cases"
    cases.mkdir()
    records = [record for record in json.loads((SOURCE / "per_video.json").read_text()) if record["scope"] == "full_episode"]
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(process, ((record, str(cases)) for record in records)))
    counts = defaultdict(Counter)
    cross = defaultdict(Counter)
    frames = Counter()
    candidates = []
    for report in reports:
        for threshold, status in report["diagnostic_variants"].items():
            counts[threshold][status] += 1
            cross[threshold][f"{report['old_status']} -> {status}"] += 1
        if "bilateral_clearance_small_tilt_candidate" in report["diagnostic_variants"].values():
            candidates.append(report["key"])
        for row in report["terminal_frames"]:
            frames["total"] += 1
            frames["bilateral_body_measurements_valid"] += int(all(side.get("valid") for side in row["sides"]))
            frames["bilateral_full_geometry"] += int(all(side.get("valid") and side.get("full_geometry") for side in row["sides"]))
            frames["direct_rod_angle_valid"] += int(row["relative_rod_angle_deg"] is not None)
    summary = {"status": "requires_visual_audit", "episodes": len(reports), "counts": counts, "old_to_new": cross,
               "terminal_frame_diagnostics": frames, "candidate_keys": candidates,
               "formal_results_published": False, "legacy_evaluator_changed": False,
               "warning": "These are uncalibrated image-space diagnostic candidates, not confirmed physical successes."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "per_episode.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
