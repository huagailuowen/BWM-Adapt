#!/usr/bin/env python3
"""Replay saved DINO proposals with color/size ablations on a CPU compute node."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm allocation required; no login-node video analysis")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import numpy as np
    from wan_video_action.real97_eval.open_vocab_filter import filter_proposals

    cv2.setNumThreads(1)
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config_snapshot.json").write_text(json.dumps(config, indent=2))
    records = [json.loads(line) for line in (ROOT / config["source_detections"]).read_text().splitlines()
               if line.strip()]
    variants = {"baseline": [], "size_only": [], "color_only": [], "size_and_color": []}
    summary = dict(samples=len(records), calibration_only=True, formally_certified=False,
                   source_detections=config["source_detections"], variants={})
    thumbnails = []
    with (args.output / "filtered_detections.jsonl").open("w") as handle:
        for index, record in enumerate(records):
            if "image_path" in record:
                image = cv2.imread(record["image_path"])
            else:
                cap = cv2.VideoCapture(record["video_path"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, record["frame"])
                ok, image = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError(f"Cannot decode {record['video_path']} frame {record['frame']}")
            if image is None:
                raise RuntimeError(f"Missing sample image: {index}")
            proposals = filter_proposals(image, record["all_detections"], config)
            selections = dict(baseline=record["selected"])
            for name, flag in (("size_only", "size_ok"), ("color_only", "color_ok"),
                               ("size_and_color", "accepted")):
                selections[name] = next((row for row in proposals if row[flag]), None)
            reference = record.get("reference_center")
            for name, selection in selections.items():
                error = None if reference is None or selection is None else float(
                    np.linalg.norm(np.asarray(selection["center"]) - np.asarray(reference)))
                variants[name].append(dict(kind=record["kind"], detected=selection is not None,
                                           error_px=error))
            selected = selections["size_and_color"]
            output = dict(record, filtered_proposals=proposals, filtered_selected=selected,
                          ablation_selections=selections)
            canvas = image.copy()
            if selected is not None:
                x0, y0, x1, y1 = map(round, selected["box"])
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 0), 2)
                cv2.drawMarker(canvas, tuple(map(round, selected["center"])),
                               (0, 220, 0), cv2.MARKER_CROSS, 8, 1)
            if reference is not None:
                cv2.drawMarker(canvas, tuple(map(round, reference)),
                               (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
            label = (f"{record['environment']} ep{record['episode_index']} f{record['frame']} "
                     + ("FOUND" if selected is not None else "NO TARGET"))
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (245, 245, 245), -1)
            cv2.putText(canvas, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            path = args.output / f"sample{index:03d}.jpg"
            cv2.imwrite(str(path), canvas)
            output["filtered_review_path"] = str(path)
            handle.write(json.dumps(output) + "\n")
            thumbnails.append(canvas)
        for start in range(0, len(thumbnails), 10):
            tile = thumbnails[start:start+10]
            while len(tile) < 10:
                tile.append(np.full_like(thumbnails[0], 245))
            sheet = np.vstack([np.hstack(tile[i:i+2]) for i in range(0, 10, 2)])
            cv2.imwrite(str(args.output / f"contact_sheet_{start//10:02d}.jpg"), sheet)
    for name, rows in variants.items():
        landmarks = [row for row in rows if row["kind"] == "coarse_landmark"]
        negatives = [row for row in rows if row["kind"] == "synthetic_negative"]
        missed = [row for row in rows if row["kind"] == "previous_whole_episode_miss"]
        errors = [row["error_px"] for row in landmarks if row["error_px"] is not None]
        summary["variants"][name] = dict(
            landmarks=len(landmarks), landmarks_detected=len(errors),
            landmarks_within6=sum(error <= 6 for error in errors),
            landmark_mean_error_px=float(np.mean(errors)) if errors else None,
            negative_controls=len(negatives),
            negative_false_detections=sum(row["detected"] for row in negatives),
            previous_miss_frames=len(missed),
            previous_miss_frames_detected=sum(row["detected"] for row in missed))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
