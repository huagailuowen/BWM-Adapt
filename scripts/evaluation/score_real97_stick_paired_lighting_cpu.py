#!/usr/bin/env python3
"""Paired Stick object-motion errors on clean and light-augmented predictions.

Uses the existing split-blind material-marker optical-flow tracker. Missing
measurements remain missing; no balanced-lift classifier is used as an error.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wan_video_action.real97_eval.stick_marker_flow import MarkerReference, locate_markers

RUNS = {
    "clean": {
        "ours": ROOT / "outputs/infer_real97_stick_near_table_unbounded_job114938",
        "standard": ROOT / "outputs/infer_real97_stick_standard_step5500_reference_job114478",
    },
    "augmented": {
        "ours": ROOT / "outputs/infer_real97_stick_light_aug_pair_job114973/ours",
        "standard": ROOT / "outputs/infer_real97_stick_light_aug_pair_job114973/standard",
    },
}
METRICS = ("center_ade_px", "center_fde_px", "angle_mae_deg", "final_angle_error_deg",
           "left_ade_px", "right_ade_px", "midpoint_ade_px", "terminal_center_error_px")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    content = path.read_bytes()
    return [json.loads(line) for line in content.decode().splitlines() if line.strip()], hashlib.sha256(content).hexdigest()


def identity(row):
    return tuple(row[k] for k in ("environment", "episode_index", "dataset_split", "start_frame",
                                  "end_frame", "total_frames", "frame_stride", "length"))


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode video: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (192, 256):
                raise RuntimeError(f"Unexpected model video geometry: {path}: {frame.shape}")
            frames.append(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR))
    finally:
        cap.release()
    if len(frames) != 41:
        raise RuntimeError(f"Expected 41 aligned frames: {path}: {len(frames)}")
    return frames


def make_state(centers, reasons, detail=None):
    good = []
    for side, center in enumerate(centers):
        valid = center is not None and np.isfinite(center).all()
        if valid:
            x, y = center
            valid = 0 <= y < 480 and (0 <= x < 320 if side == 0 else 320 <= x < 640)
        good.append(bool(valid))
    valid = all(good)
    if valid and np.linalg.norm(np.asarray(centers[1]) - centers[0]) < 180:
        valid = False
        reasons = list(reasons) + ["implausible_endpoint_span"]
    angle = None
    if valid:
        delta = np.asarray(centers[1]) - centers[0]
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    return {"valid": valid, "side_valid": good,
            "centers": [None if c is None else np.asarray(c, dtype=float).tolist() for c in centers],
            "angle_deg": angle, "reasons": reasons, "detail": detail}


def track_video(reference, frames):
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    settings = {"contact_lever_arm_px": 20.0, "flow_error_floor_px": 0.5}
    references = [MarkerReference(gray, hsv, side, 0.0, settings) for side in (0, 1)]
    tracks = {"flow": [], "color_check": []}
    for frame in frames:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        flow = [reference.update(current_gray, current_hsv) for reference in references]
        centers = [state.get("center") if state.get("flow_valid") else None for state in flow]
        tracks["flow"].append(make_state(centers, [s.get("reason") for s in flow], flow))
        detected = [locate_markers(current_hsv, side) for side in (0, 1)]
        centers = [None if state is None else state["center"] for state in detected]
        tracks["color_check"].append(make_state(centers, ["missing_color_marker" if s is None else None for s in detected]))
    return tracks


def angle_error(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def score_condition(states, wanted, forced=None):
    common = [i for i in wanted if all(states[method][i]["valid"] for method in ("gt", "ours", "standard"))]
    if forced is not None:
        common = [i for i in common if i in forced]
    coverage = {method: sum(states[method][i]["valid"] for i in wanted) / len(wanted)
                for method in states}
    result = {"requested_frames": len(wanted), "common_frames": len(common),
              "common_indices": common, "common_coverage": len(common) / len(wanted),
              "method_valid_fraction": coverage, "methods": {}}
    for method in ("ours", "standard"):
        metrics = {key: None for key in METRICS}
        if common:
            gt = np.asarray([states["gt"][i]["centers"] for i in common])
            prediction = np.asarray([states[method][i]["centers"] for i in common])
            errors = np.linalg.norm(prediction - gt, axis=-1)
            angular = [angle_error(states[method][i]["angle_deg"], states["gt"][i]["angle_deg"]) for i in common]
            metrics.update(center_ade_px=float(errors.mean()), left_ade_px=float(errors[:, 0].mean()),
                           right_ade_px=float(errors[:, 1].mean()), angle_mae_deg=float(np.mean(angular)),
                           midpoint_ade_px=float(np.linalg.norm(prediction.mean(axis=1) - gt.mean(axis=1), axis=1).mean()))
            if wanted[-1] in common:
                index = common.index(wanted[-1])
                metrics.update(center_fde_px=float(errors[index].mean()), final_angle_error_deg=float(angular[index]))
            tail = wanted[-3:]
            if all(i in common for i in tail):
                metrics["terminal_center_error_px"] = float(np.mean([errors[common.index(i)].mean() for i in tail]))
        result["methods"][method] = metrics
    return result


def draw_review(destination, frames, tracks, wanted, key):
    times = [0, wanted[len(wanted) // 2], wanted[-1]]
    canvas = np.full((3 * 270, 6 * 320, 3), 245, np.uint8)
    for r, index in enumerate(times):
        for c, (condition, method) in enumerate((c, m) for c in ("clean", "augmented") for m in ("gt", "ours", "standard")):
            frame = frames[condition][method][index].copy()
            flow = tracks[condition][method]["flow"][index]
            color = tracks[condition][method]["color_check"][index]
            for side, point in enumerate(flow["centers"]):
                if flow["side_valid"][side]:
                    cv2.circle(frame, tuple(np.rint(point).astype(int)), 7, (0, 0, 255) if side == 0 else (255, 0, 0), 2)
            for side, point in enumerate(color["centers"]):
                if color["side_valid"][side]:
                    cv2.drawMarker(frame, tuple(np.rint(point).astype(int)), (0, 210, 210), cv2.MARKER_CROSS, 9, 1)
            x, y = c * 320, r * 270
            canvas[y + 28:y + 268, x:x + 320] = cv2.resize(frame, (320, 240))
            label = f"{condition} {method} t={index} flow={'ok' if flow['valid'] else 'missing'}"
            cv2.putText(canvas, label, (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (20, 20, 20), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not save review: {key}")


def process(task):
    index, row, output_text = task
    output = Path(output_text)
    cv2.setNumThreads(1)
    key = f'q{index:04d}_{row["environment"]}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'
    last = min(int(row["end_frame"]), int(row["total_frames"]) - 1)
    wanted = [i for i in range(1, 41) if int(row["start_frame"]) + int(row["frame_stride"]) * i <= last]
    if not wanted:
        raise RuntimeError(f"No real future frames: {key}")
    frames, tracks, gt_differences = {}, {}, {}
    for condition, runs in RUNS.items():
        paths = {"gt": runs["ours"] / "raw/gt" / (key + ".mp4"),
                 "ours": runs["ours"] / "raw/stage2" / (key + ".mp4"),
                 "standard": runs["standard"] / "raw/standard" / (key + ".mp4")}
        frames[condition] = {method: read_video(path) for method, path in paths.items()}
        other_gt = read_video(runs["standard"] / "raw/gt" / (key + ".mp4"))
        errors = [float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
                  for a, b in zip(frames[condition]["gt"], other_gt)]
        gt_differences[condition] = {"mean_absolute_rgb_difference": float(np.mean(errors)),
                                   "maximum_frame_mean_absolute_rgb_difference": max(errors)}
        if max(errors) > 1.0:
            raise RuntimeError(f"GT videos differ beyond codec tolerance: {condition} {key}: {max(errors)}")
        reference = frames[condition]["gt"][0]
        tracks[condition] = {}
        for method in ("gt", "ours", "standard"):
            cv2.setRNGSeed(73019)
            tracks[condition][method] = track_video(reference, frames[condition][method])
    result = {"key": key, "index": index, "environment": row["environment"], "split": row["dataset_split"],
              "episode_index": row["episode_index"], "wanted_indices": wanted, "GT_consistency": gt_differences,
              "scores": {}, "six_way_common_scores": {}}
    for tracker in ("flow", "color_check"):
        paired = {condition: {m: t[tracker] for m, t in by_method.items()} for condition, by_method in tracks.items()}
        result["scores"][tracker] = {condition: score_condition(states, wanted) for condition, states in paired.items()}
        common = {i for i in wanted if all(states[method][i]["valid"] for states in paired.values()
                                          for method in ("gt", "ours", "standard"))}
        result["six_way_common_scores"][tracker] = {
            condition: score_condition(states, wanted, common) for condition, states in paired.items()}
    result["missing_reasons"] = {
        condition: {method: dict(Counter(reason for i in wanted for reason in by_tracker["flow"][i]["reasons"] if reason))
                    for method, by_tracker in by_method.items()}
        for condition, by_method in tracks.items()}
    write_json(output / "tracks" / (key + ".json"), tracks)
    review = output / "reviews" / (key + ".jpg")
    draw_review(review, frames, tracks, wanted, key)
    result["review"] = str(review)
    write_json(output / "per_query" / (key + ".json"), result)
    return result


def paired_statistics(items):
    if not items:
        return {"paired_queries": 0, "ours": None, "standard": None, "delta_ours_minus_standard": None}
    ours = np.asarray([item[1] for item in items], dtype=float)
    standard = np.asarray([item[2] for item in items], dtype=float)
    differences = ours - standard
    names = sorted({item[0] for item in items})
    sums = np.asarray([sum(differences[i] for i, item in enumerate(items) if item[0] == name) for name in names])
    counts = np.asarray([sum(item[0] == name for item in items) for name in names])
    rng = np.random.default_rng(20260912)
    bootstrap = rng.integers(0, len(names), size=(5000, len(names)))
    samples = sums[bootstrap].sum(axis=1) / counts[bootstrap].sum(axis=1)
    standard_mean = float(standard.mean())
    return {"paired_queries": len(items), "environments": len(names), "ours": float(ours.mean()),
            "standard": standard_mean, "delta_ours_minus_standard": float(differences.mean()),
            "ours_reduction_pct": float((standard.mean() - ours.mean()) / standard.mean() * 100) if standard_mean else None,
            "ours_wins": int((differences < -1e-9).sum()), "standard_wins": int((differences > 1e-9).sum()),
            "ties": int((np.abs(differences) <= 1e-9).sum()),
            "environment_cluster_bootstrap_delta_ci95": np.quantile(samples, [.025, .975]).tolist()}


def summarize(results):
    report = {}
    for tracker in ("flow", "color_check"):
        report[tracker] = {}
        for condition in RUNS:
            report[tracker][condition] = {}
            for split in ("all", "train", "test"):
                rows = [row for row in results if split == "all" or row["split"] == split]
                values = [row["scores"][tracker][condition] for row in rows]
                summary = {"queries": len(rows), "requested_frames": sum(v["requested_frames"] for v in values),
                           "common_frames": sum(v["common_frames"] for v in values),
                           "common_coverage": sum(v["common_frames"] for v in values) / sum(v["requested_frames"] for v in values),
                           "method_valid_fraction": {m: float(np.mean([v["method_valid_fraction"][m] for v in values]))
                                                     for m in ("gt", "ours", "standard")},
                           "queries_with_at_least_80pct_common_coverage": sum(v["common_coverage"] >= .8 for v in values),
                           "metrics": {}, "metrics_at_least_80pct_common_coverage": {}}
                for metric in METRICS:
                    items = [(row["environment"], value["methods"]["ours"][metric], value["methods"]["standard"][metric])
                             for row, value in zip(rows, values) if value["methods"]["ours"][metric] is not None
                             and value["methods"]["standard"][metric] is not None]
                    summary["metrics"][metric] = paired_statistics(items)
                    selected = [(row["environment"], value["methods"]["ours"][metric], value["methods"]["standard"][metric])
                                for row, value in zip(rows, values) if value["common_coverage"] >= .8
                                and value["methods"]["ours"][metric] is not None and value["methods"]["standard"][metric] is not None]
                    summary["metrics_at_least_80pct_common_coverage"][metric] = paired_statistics(selected)
                report[tracker][condition][split] = summary
    changes = {}
    for tracker in ("flow", "color_check"):
        changes[tracker] = {}
        for split in ("all", "train", "test"):
            selected = [r for r in results if split == "all" or r["split"] == split]
            changes[tracker][split] = {}
            for method in ("ours", "standard"):
                changes[tracker][split][method] = {}
                for metric in METRICS:
                    pairs = [(r["six_way_common_scores"][tracker]["clean"]["methods"][method][metric],
                              r["six_way_common_scores"][tracker]["augmented"]["methods"][method][metric]) for r in selected]
                    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                    changes[tracker][split][method][metric] = {
                        "paired_queries": len(pairs), "clean": float(np.mean([a for a, b in pairs])) if pairs else None,
                        "augmented": float(np.mean([b for a, b in pairs])) if pairs else None,
                        "augmented_minus_clean": float(np.mean([b - a for a, b in pairs])) if pairs else None,
                    }
    return {"method_comparison": report, "lighting_change_identical_six_way_frames": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full-video tracking must run on a CPU compute node")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_rows, hashes = {}, {}
    for condition, runs in RUNS.items():
        for method, run in runs.items():
            path = run / ("input_manifest/query.jsonl" if method == "ours" else "query.jsonl")
            rows, digest = read_rows(path)
            source_rows[(condition, method)] = rows
            hashes[condition + "/" + method] = digest
    queries = source_rows[("clean", "ours")]
    if len(queries) != 40 or Counter(r["dataset_split"] for r in queries) != {"train": 16, "test": 24}:
        raise RuntimeError("Unexpected original Query population")
    canonical = [identity(row) for row in queries]
    for key, rows in source_rows.items():
        if [identity(row) for row in rows] != canonical:
            raise RuntimeError(f"The four conditions do not share identical Query windows: {key}")
    protocol = {
        "source_runs": {c: {m: str(p) for m, p in runs.items()} for c, runs in RUNS.items()},
        "query_manifest_sha256": hashes, "queries": 40, "splits": {"train": 16, "test": 24},
        "coordinate_size_wh": [640, 480], "decoded_size_wh": [256, 192], "frames": 41,
        "frame_stride": 3, "source_fps": 20, "condition_frame_excluded": True, "padding_excluded": True,
        "primary_tracker": "existing MarkerReference, fixed GT initial-frame material markers, forward-backward LK plus RANSAC",
        "secondary_tracker": "existing split-blind locate_markers framewise color centers",
        "center_definition": "left/right object blue-material marker centers, not full-object silhouette centroids",
        "center_ade": "mean of the two endpoint Euclidean errors on GT/Ours/Standard common valid frames, then query macro mean",
        "center_fde": "two-endpoint error at the exact last real future frame, never an earlier last-valid frame",
        "angle": "signed line angle between the two material centers; wrapped 180-degree absolute error",
        "missing_policy": "no interpolation or forward filling; separate method and common coverage; 80pct subset also reported",
        "lighting_delta_policy": "identical six-way-valid frames across clean/augmented GT and both models",
        "confidence_interval": "5000 paired environment-cluster bootstrap draws",
        "known_environment_Z_initialization": True, "formal_metric_approved": False,
        "action_success_metric": "not computed; no balanced-lift threshold mixed into trajectory errors",
        "source_results_modified": False,
    }
    write_json(output / "protocol.json", protocol)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(process, [(i, q, str(output)) for i, q in enumerate(queries)]):
            results.append(row)
            print(f"[scored] {len(results)}/40 {row['key']}", flush=True)
    summary = {"job_id": os.environ["SLURM_JOB_ID"], "queries": len(results), "formal_metric_approved": False,
               **summarize(results)}
    write_json(output / "per_query.json", results)
    write_json(output / "summary.json", summary)
    print("[scoring_complete] " + str(output), flush=True)
    print(json.dumps({c: {s: summary["method_comparison"]["flow"][c][s] for s in ("train", "test")} for c in RUNS}), flush=True)


if __name__ == "__main__":
    main()
