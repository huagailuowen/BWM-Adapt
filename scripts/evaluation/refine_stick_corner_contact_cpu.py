#!/usr/bin/env python3
"""CPU-only contact calibration, retaining all old outputs and raw images."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluation.audit_stick_training_action_precision import read_frames, initial_frame
from wan_video_action.real97_eval.stick_corner_contact import RefinedStickContact, summarize_refined


def draw_detail(frame, row, side_index, bounds, label, overlay):
    x0, y0, x1, y1 = bounds
    image = frame.copy()
    if overlay and len(row["sides"]) > side_index:
        side = row["sides"][side_index]
        base = row["silhouette"]
        if base.get("ground_line"):
            slope, intercept = base["ground_line"]
            cv2.line(image, (x0, round(slope * x0 + intercept)),
                     (x1, round(slope * x1 + intercept)), (255, 100, 0), 1)
        if side.get("corners"):
            for point, visible in zip(side["corners"], side["corner_visible"]):
                cv2.circle(image, tuple(np.rint(point).astype(int)), 2,
                           (0, 200, 255) if visible else (0, 0, 255), 1, cv2.LINE_AA)
    crop = cv2.resize(image[y0:y1, x0:x1], (420, 330), interpolation=cv2.INTER_NEAREST)
    panel = np.full((390, 420, 3), 250, np.uint8)
    panel[60:] = crop
    cv2.putText(panel, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (25, 25, 25), 1)
    if len(row["sides"]) > side_index:
        side = row["sides"][side_index]
        cv2.putText(panel, side["state"], (5, 37), cv2.FONT_HERSHEY_SIMPLEX, .42, (25, 25, 25), 1)
        gaps = side.get("corner_displacement_px")
        text = f"corners {gaps[0]:.2f}, {gaps[1]:.2f}px  band {side['uncertainty_margin_px']:.2f}" if gaps else side.get("reason", "unknown")
        cv2.putText(panel, text[:62], (5, 54), cv2.FONT_HERSHEY_SIMPLEX, .38, (30, 30, 150), 1)
    return panel


def process(job):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(970910)
    source = job["record"]
    frames, fps = read_frames(source["video"])
    common = initial_frame(job["reference_video"]) if job.get("reference_video") else frames[0]
    tracker = RefinedStickContact(common, job["config"])
    rows = []
    for index, frame in enumerate(frames):
        row = tracker.update(frame)
        row.update(frame=index, time_seconds=index / fps, evaluated=index > 0)
        if job.get("query"):
            query = job["query"]
            row["evaluated"] = bool(index > 0 and int(query["start_frame"]) + index * int(query["frame_stride"]) < int(query["total_frames"]))
        rows.append(row)
    decision = summarize_refined(rows, fps, job["config"])
    output = Path(job["output"]) / "cases" / source["key"]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "corner_tracks.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = {key: source[key] for key in ("key", "environment", "episode_index", "method", "scope", "video")}
    report.update(old_decision=source["decision"], refined_decision=decision,
                  rendered_detail=bool(job["render"]), fps=fps)
    if job["render"] and tracker.references is not None:
        signals = []
        for index, row in enumerate(rows):
            sides = row["sides"]
            if index < len(rows) // 4 or len(sides) != 2 or not all(side.get("flow_valid") for side in sides):
                continue
            value = min(min(side["corner_displacement_px"]) for side in sides)
            signals.append((abs(value - 1.0), index))
        borderline = min(signals)[1] if signals else len(rows) // 2
        selected = sorted(set([0, max(0, borderline - 2), borderline, len(rows) - 1]))
        strips = []
        for side_index, reference in enumerate(tracker.references):
            cx = int(round(float(reference["center"][0])))
            floor_y = int(round(tracker.silhouette.slope * cx + tracker.silhouette.intercept))
            x0 = min(max(0, cx - 70), 500)
            y0 = min(max(0, floor_y - 80), 370)
            bounds = (x0, y0, x0 + 140, y0 + 110)
            for overlay in (False, True):
                panels = [draw_detail(frames[index], rows[index], side_index, bounds,
                                     f"{'L' if side_index == 0 else 'R'} f{index} {'overlay' if overlay else 'RAW 3x nearest'}",
                                     overlay) for index in selected]
                strips.append(np.hstack(panels))
        cv2.imwrite(str(output / "contact_detail.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 96])
        report["detail_frames"] = selected
        report["detail"] = str(output / "contact_detail.jpg")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run video/corner processing on a CPU compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/evaluation/stick_corner_contact_v2.json").read_text())
    source = ROOT / "outputs/evaluation_stick_training_action_contact_job113992"
    records = json.loads((source / "per_video.json").read_text())
    selected = set()
    groups = defaultdict(list)
    for record in records:
        if record["scope"] == "full_training_episode":
            groups[(record["environment"], record["decision"]["status"])].append(record)
        else:
            selected.add(record["key"])
    for group in groups.values():
        selected.add(sorted(group, key=lambda row: row["episode_index"])[0]["key"])
    selected.update(["full_stick-L0-R0_ep000020", "full_stick-L0-R0_ep000006", "full_stick-L0-R3right_ep000017"])
    prep = ROOT / "outputs/evaluation_real97_reference_ttt_v3_20260910/stick_prep_job113867"
    with (prep / "query.jsonl").open() as handle:
        queries = {int(row["sample_index"]): row for line in handle if line.strip() for row in [json.loads(line)]}
    existing_gt = {record["query_index"]: record["video"] for record in records
                   if record["scope"] == "existing_train_query" and record["method"] == "gt"}
    jobs = []
    for record in records:
        job = {"record": record, "output": str(args.output), "config": config, "render": record["key"] in selected}
        if record["scope"] == "existing_train_query":
            job.update(reference_video=existing_gt[record["query_index"]], query=queries[record["query_index"]])
        jobs.append(job)
    (args.output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(process, job) for job in jobs]
        for future in as_completed(pending):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(jobs):
                progress = {"completed": len(results), "expected": len(jobs)}
                (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda row: row["key"])
    transitions = Counter()
    for row in results:
        if row["scope"] == "full_training_episode":
            transitions[(row["old_decision"]["status"], row["refined_decision"]["status"])] += 1
    summary = {
        "status": "calibration_only_no_action_success_rate_claimed",
        "processed_videos": len(results),
        "gt_transitions": [{"old": a, "refined": b, "count": n} for (a, b), n in sorted(transitions.items())],
        "details_rendered": sum("detail" in row for row in results),
        "large_candidate_inference_submitted": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "per_video.json").write_text(json.dumps(results, indent=2) + "\n")
    page = ['<!doctype html><meta charset="utf-8"><title>Stick fine contact calibration</title>',
            '<style>body{font-family:sans-serif;margin:28px;background:#fafaf7}img{max-width:100%}article{margin-bottom:40px}</style>',
            '<h1>Stick fine contact calibration</h1>',
            '<p>RAW crops are nearest-neighbor magnifications of native frames, not enhanced or synthesized. L/R rows each contain raw and annotated versions. Inspect lowest corners, shadows, and the full episode before accepting labels.</p>',
            '<p>All classifications and uncertainty margins remain provisional. The detail crop is fixed around the rest contact region; a highly raised box may leave this crop.</p>']
    for row in results:
        if "detail" not in row:
            continue
        detail = Path(row["detail"]).relative_to(args.output)
        page.append(f'<article><h3>{html.escape(row["key"])}: {row["old_decision"]["status"]} to {row["refined_decision"]["status"]}</h3><img loading="lazy" src="{detail}"></article>')
    (args.output / "index.html").write_text("\n".join(page))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
