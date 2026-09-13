#!/usr/bin/env python3
"""Measure possible floor-tape contamination and prepare blind contact review sheets."""
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.reevaluate_stick_shape_flow_cpu import read_frames
from wan_video_action.real97_eval.stick_common_reference_rigid_review import RigidBody
from wan_video_action.real97_eval.stick_contact_floor_audit import analyze_side

BRANCH = json.loads((ROOT / "configs/evaluation/stick_contact_floor_audit_v5.json").read_text())
CONFIG = {**json.loads((ROOT / BRANCH["base_config"]).read_text()), **BRANCH}


def fit_patch(image, bounds, width=160, height=180):
    x0, y0, x1, y1 = bounds
    patch = image[y0:y1, x0:x1]
    canvas = np.full((height, width, 3), 235, np.uint8)
    if not patch.size:
        return canvas
    scale = min(width / patch.shape[1], height / patch.shape[0])
    resized = cv2.resize(patch, (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale))))
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def save_contact_strip(reference, frames, bodies, measurements, record, path):
    # No automatic decision/angle is displayed: first review physical contact blind.
    strip = np.full((218, 960, 3), 245, np.uint8)
    title = record["environment"] + " ep%06d " % record["episode_index"] + record["method"]
    cv2.putText(strip, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
    for side in (0, 1):
        points = []
        if bodies[side].original.get("valid"):
            points.extend(np.asarray(bodies[side].original["hull"]).reshape(-1, 2).tolist())
        for measurement in measurements:
            data = measurement["sides"][side]
            if data.get("valid"):
                points.extend(data["projected_hull"])
        if points:
            p = np.asarray(points)
            x0 = max(side * 320, int(p[:, 0].min()) - 20)
            x1 = min((side + 1) * 320, int(p[:, 0].max()) + 21)
            y0 = max(120, int(p[:, 1].min()) - 15)
            y1 = min(480, int(p[:, 1].max()) + 35)
            bounds = (x0, y0, x1, y1)
        else:
            bounds = (side * 320, 140, (side + 1) * 320, 480)
        for position, (name, frame) in enumerate((("reference", reference), ("first", frames[0]), ("last", frames[-1]))):
            col = side * 3 + position
            x = col * 160
            strip[36:216, x:x + 160] = fit_patch(frame, bounds)
            cv2.putText(strip, ("L " if side == 0 else "R ") + name, (x + 4, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
    cv2.imwrite(str(path / "blind_contact_strip.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 98])


def process(item):
    record, output = item
    cv2.setNumThreads(1)
    reference = read_frames(record["reference_video"], [record["reference_frame"]])[record["reference_frame"]]
    bodies = [RigidBody(reference, side, CONFIG) for side in (0, 1)]
    slope = None
    if all(b.original.get("valid") for b in bodies):
        centers = [(s.original["bbox"][0] + s.original["bbox"][2] / 2,
                    s.original["bbox"][1] + s.original["bbox"][3]) for s in bodies]
        slope = (centers[1][1] - centers[0][1]) / max(1, centers[1][0] - centers[0][0])
    indices = [f["frame"] for f in record["terminal_frames"]]
    decoded = read_frames(record["video"], indices)
    rows = []
    last_floor = np.zeros((480, 640), np.uint8)
    for source in record["terminal_frames"]:
        index = source["frame"]
        sides = []
        floor_union = np.zeros((480, 640), np.uint8)
        for side in (0, 1):
            direct, sequential = source["direct"][side], source["sequential"][side]
            pose = dict(direct, source="direct") if direct.get("valid") else dict(sequential, source="sequential")
            result, floor = analyze_side(decoded[index], bodies[side].original, pose, slope, side, CONFIG, 9173 + index * 2 + side)
            sides.append(result)
            floor_union = cv2.bitwise_or(floor_union, floor)
        rows.append({"frame": index, "source_frame": source["source_frame"], "sides": sides})
        last_floor = floor_union
    path = Path(output) / "cases" / record["key"]
    path.mkdir(parents=True, exist_ok=False)
    result = {k: record[k] for k in ("key", "scope", "split", "method", "environment", "episode_index", "video")}
    result.update(reference_video=record["reference_video"], reference_frame=record["reference_frame"],
                  terminal_frames=rows, decisions_unchanged=True, physical_contact_certified=False)
    (path / "audit.json").write_text(json.dumps(result, indent=2))
    last = decoded[indices[-1]]
    baseline, filtered = last.copy(), last.copy()
    filtered[last_floor > 0] = (0, 230, 255)
    for side in rows[-1]["sides"]:
        if not side.get("valid"):
            continue
        for canvas, name, color in ((baseline, "baseline", (210, 0, 210)), (filtered, "filtered", (0, 220, 0))):
            shape = side[name]
            if shape.get("valid"):
                cv2.polylines(canvas, [np.asarray(shape["hull"], np.int32).reshape(-1, 1, 2)], True, color, 2)
    visual = np.full((510, 1280, 3), 245, np.uint8)
    visual[30:, :640] = baseline
    visual[30:, 640:] = filtered
    cv2.putText(visual, "baseline silhouette", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    cv2.putText(visual, "floor-blue exclusion diagnostic (yellow)", (650, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    cv2.imwrite(str(path / "segmentation_pair.jpg"), visual, [cv2.IMWRITE_JPEG_QUALITY, 98])
    save_contact_strip(reference, [decoded[i] for i in indices], bodies, rows, record, path)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run image analysis on a CPU compute node")
    output = ROOT / ("outputs/stick_contact_floor_audit_job" + os.environ["SLURM_JOB_ID"])
    output.mkdir(exist_ok=False)
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2))
    records = json.loads((ROOT / CONFIG["source"]).read_text())
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(process, [(r, str(output)) for r in records]))
    counters = Counter()
    largest = []
    for r in results:
        changed = []
        for frame in r["terminal_frames"]:
            for side in frame["sides"]:
                if not side.get("valid"):
                    continue
                counters["measured_endpoint_frames"] += 1
                if side["floor_overlap_pixels"]["baseline"]:
                    counters["baseline_frames_overlap_floor_candidate"] += 1
                delta = side["filtered_minus_baseline_gap_px"]
                if delta is not None and abs(delta) >= 2:
                    changed.append({"frame": frame["frame"], "side": side["side"], "gap_shift_px": delta,
                                    "rigid_gap_px": side["rigid_gap_px"], "gaps_px": side["gaps_px"]})
        if changed:
            counters["videos_with_gap_shift_at_least_2px"] += 1
            largest.append({"key": r["key"], "scope": r["scope"], "changes": changed,
                            "maximum_absolute_shift": max(abs(x["gap_shift_px"]) for x in changed)})
    largest.sort(key=lambda x: x["maximum_absolute_shift"], reverse=True)
    pages = []
    for scope in ("full_episode", "query"):
        subset = [r for r in results if r["scope"] == scope]
        subset.sort(key=lambda r: (r["environment"], r["episode_index"], r["method"]))
        for page_start in range(0, len(subset), 8):
            batch = subset[page_start:page_start + 8]
            sheet = np.full((4 * 242, 1920, 3), 245, np.uint8)
            mapping = []
            for i, r in enumerate(batch):
                strip = cv2.imread(str(output / "cases" / r["key"] / "blind_contact_strip.jpg"))
                if strip is None:
                    raise RuntimeError("missing contact review strip")
                x, y = (i % 2) * 960, (i // 2) * 242
                number = page_start + i
                cv2.putText(sheet, "%s #%03d" % (scope, number), (x + 5, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
                sheet[y + 24:y + 242, x:x + 960] = strip
                mapping.append({"index": number, "key": r["key"], "scope": r["scope"],
                                "left_contact_label": None, "right_contact_label": None, "review_note": None})
            page_path = output / ("blind_%s_page%02d.jpg" % (scope, page_start // 8))
            cv2.imwrite(str(page_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 98])
            pages.append({"path": str(page_path), "scope": scope, "cases": mapping})
    summary = {"job": os.environ["SLURM_JOB_ID"], "videos": len(results), "counts": dict(counters),
               "largest_changes": largest[:12], "review_pages": len(pages),
               "success_thresholds_modified": False, "formal_metric_approved": False}
    for name, data in (("summary.json", summary), ("per_video_audit.json", results), ("blind_review_pages.json", pages)):
        (output / name).write_text(json.dumps(data, indent=2))
    print(json.dumps({"output": str(output), "job": summary["job"], "videos": len(results),
                      "counts": dict(counters), "review_pages": len(pages),
                      "largest_change_keys": [{"key": r["key"], "shift_px": r["maximum_absolute_shift"]} for r in largest[:10]]}), flush=True)


if __name__ == "__main__":
    main()
