#!/usr/bin/env python3
"""Opt-in scoring of frozen Real97 factual Ball/Door inference videos.

CPU: frame metrics, native-pixel tracking, action decisions, review sheets.
GPU: LPIPS on exactly the same manifest and non-conditioning/non-pad frames.
No model inference, source-video changes, or training changes are performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
METHODS = ("standard", "dino", "ours_stage1", "ours_stage2")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def freeze_manifest(config, output):
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["config"] != config:
            raise RuntimeError("Existing manifest uses another protocol; use a new output directory")
        return manifest
    queries = []
    for task, settings in config["tasks"].items():
        indexes = {}
        for method in ("ours", "standard", "dino"):
            index = {}
            for marker in sorted((ROOT / settings[method] / "completed").glob("q*.json")):
                record = read_json(marker)
                row = record["row"] if method == "ours" else record["query"]
                key = row["sample_id"]
                if key in index:
                    raise ValueError(f"Duplicate factual query {method}: {key}")
                index[key] = (record, row, str(marker))
            indexes[method] = index
        reference = indexes["ours"]
        if not reference:
            raise ValueError(f"No factual queries for {task}")
        for method in ("standard", "dino"):
            if set(reference) != set(indexes[method]):
                raise ValueError(f"Unmatched {task}/{method} query identities; do not silently intersect")
        for key, (record, row, marker) in reference.items():
            paths = {"gt": record["paths"]["gt"],
                     "ours_stage1": record["paths"]["stage1"],
                     "ours_stage2": record["paths"]["stage2"]}
            markers = {"ours": marker}
            for method in ("standard", "dino"):
                other, other_row, other_marker = indexes[method][key]
                for field in ("environment", "dataset_split", "action_level", "video",
                              "action", "native_frame_indices", "evaluation_frame_indices"):
                    if row.get(field) != other_row.get(field):
                        raise ValueError(f"Query mismatch {task}/{key}/{method}/{field}")
                paths[method] = other["prediction"]
                markers[method] = other_marker
            frame_ids = row["evaluation_frame_indices"]
            native_ids = row["native_frame_indices"]
            if not frame_ids or min(frame_ids) < 1 or len(set(frame_ids)) != len(frame_ids):
                raise ValueError(f"Invalid evaluation-frame mask: {key}")
            actual_ids = [native_ids[i] for i in frame_ids]
            if len(set(actual_ids)) != len(actual_ids):
                raise ValueError(f"Padding included in evaluation frames: {key}")
            for path in paths.values():
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            queries.append({"task": task, "key": Path(marker).stem,
                            "row": row, "paths": paths, "source_markers": markers,
                            "native_size": settings["native_size"]})
    manifest = {"config": config, "queries": queries,
                "methods": list(METHODS), "paired_gt_only": True,
                "checkpoint_steps": {"ball": {"ours": 5500, "standard": 5500, "dino": 5500},
                                     "door": {"ours": 3715, "standard": 5500, "dino": 5500}},
                "fairness_note": "Ours Stage2 uses the existing customized supports and initializations, including near-environment-table starts. This is not an equal-information or equal-training-step benchmark."}
    save_json(manifest_path, manifest)
    save_json(output / "protocol.json", config)
    return manifest


def content_box(width, height, native_size):
    nw, nh = native_size
    scale = min(width / nw, height / nh)
    rw, rh = min(width, max(1, round(nw * scale))), min(height, max(1, round(nh * scale)))
    left, top = (width - rw) // 2, (height - rh) // 2
    return left, top, left + rw, top + rh


def read_video(path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Empty or undecodable video: {path}")
    return np.stack(frames)


def native_frames(frames, size):
    import cv2
    left, top, right, bottom = content_box(frames.shape[2], frames.shape[1], size)
    return [cv2.resize(frame[top:bottom, left:right], tuple(size), interpolation=cv2.INTER_LINEAR)
            for frame in frames]


def blue_marker(first, settings):
    import cv2
    import numpy as np
    from wan_video_action.real97_eval.tracking import components
    hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= settings["hue_min"]) & (h <= settings["hue_max"]) &
            (s >= settings["saturation_min"]) & (v >= settings["value_min"]))
    mask[:, :int(first.shape[1] * settings["minimum_x_fraction"])] = False
    candidates = []
    for component in components(mask, settings["minimum_area"]):
        x, y, w, hh = component["bbox"]
        if (component["area"] <= settings["maximum_area"] and
                max(w, hh) <= settings["maximum_extent"]):
            candidates.append({"center": component["center"].tolist(), "area": component["area"],
                               "bbox": list(component["bbox"])})
    candidates.sort(key=lambda c: c["center"][0], reverse=True)
    return {"center": candidates[0]["center"] if candidates else None,
            "candidates": candidates, "needs_visual_audit": True,
            "policy": "rightmost blue component in the shared GT initial frame"}


def apply_door_closure_policy(state, config):
    """Optional one-sided closure rule; old configs retain the legacy rule."""
    policy = config.get("door_closure")
    if not policy or state.get("center") is None:
        return state
    if policy["policy"] != "one_sided_edge_gap":
        raise ValueError(f"Unknown Door closure policy: {policy['policy']}")
    result = dict(state)
    gap = result["raw_edge_seam_gap_px"]
    if gap < policy["minimum_plausible_gap_px"]:
        label = "uncertain"
        result["measurement_valid"] = False
    elif gap <= policy["closed_max_gap_px"]:
        label = "closed"
    elif gap >= policy["open_min_gap_px"]:
        label = "open"
    else:
        label = "uncertain"
    result.setdefault("legacy_closure_state", result.get("closure_state"))
    result.setdefault("legacy_calibrated_closure_excess_px", result.get("calibrated_closure_excess_px"))
    result.update(closure_state=label, closed_proposal=label == "closed",
                  calibrated_closure_excess_px=gap - policy["closed_max_gap_px"],
                  closed_outer_reference_x=result["closed_reference_x"] + policy["closed_max_gap_px"],
                  reason="train_calibrated_one_sided_" + label,
                  closure_policy=policy["policy"])
    return result


def track_video(task, frames, shared_first, calibration, config):
    import numpy as np
    from wan_video_action.real97_eval.calibrated_tracking import make_calibrated_tracker
    from wan_video_action.real97_eval.object_measurements import usable
    tracker = make_calibrated_tracker(task, shared_first, calibration, [])
    states, observations = [], []
    exit_state = None
    width = shared_first.shape[1]
    for i, frame in enumerate(frames):
        if exit_state is not None:
            state = dict(exit_state, reason="confirmed_exit_hold_last_visible", held_after_exit=True)
        else:
            state = tracker.update(frame)
            if task == "ball":
                state["held_after_exit"] = False
                if usable(state):
                    observations.append((i, state))
                elif observations and observations[-1][0] == i - 1:
                    last_i, last = observations[-1]
                    x = last["center"][0]
                    velocity = 0.0
                    if len(observations) >= 2:
                        prev_i, prev = observations[-2]
                        velocity = (x - prev["center"][0]) / (last_i - prev_i)
                    near_exit = (x >= width - config["ball_exit"]["near_boundary_px"] and
                                 velocity >= config["ball_exit"]["minimum_rightward_velocity_px"] and
                                 x + velocity >= width - 2)
                    if last.get("touches_right_edge") or near_exit:
                        exit_state = dict(last, last_visible_frame=last_i,
                                          exit_confirmation_frame=i, held_after_exit=True)
                        state = dict(exit_state, reason="confirmed_exit_hold_last_visible")
        if task == "door":
            state = apply_door_closure_policy(state, config)
        state["frame"] = i
        state["usable"] = bool(usable(state))
        states.append(state)
    return states


def trajectory_summary(task, states, frame_ids, marker, config):
    valid = [states[i] for i in frame_ids if states[i]["usable"]]
    result = {"valid_frames": len(valid), "frames": len(frame_ids),
              "coverage": len(valid) / len(frame_ids),
              "held_exit_frames": sum(s.get("held_after_exit", False) for s in valid)}
    if task == "ball":
        peak = max(valid, key=lambda s: s["center"][0]) if valid else None
        result["peak_center"] = peak["center"] if peak else None
        result["peak_frame"] = peak["frame"] if peak else None
        result["marker_distance_px"] = None
        result["marker_x_distance_px"] = None
        if peak is not None and marker["center"] is not None:
            result["marker_distance_px"] = math.dist(peak["center"], marker["center"])
            result["marker_x_distance_px"] = abs(peak["center"][0] - marker["center"][0])
    else:
        count = max(1, round(config["source_fps"] * .3))
        tail = [states[i] for i in frame_ids[-count:]]
        uncertain = (len(tail) < count or any(not s["usable"] or
                     s.get("closure_state") == "uncertain" for s in tail))
        result["tail_state"] = ("uncertain" if uncertain else
                                "closed" if all(s.get("closure_state") == "closed" for s in tail)
                                else "open_or_rebound")
        result["closed"] = None if uncertain else result["tail_state"] == "closed"
        result["tail_excess_px"] = mean([s.get("calibrated_closure_excess_px") for s in tail])
    return result


def image_metrics(gt, pred, frame_ids, size):
    import cv2
    import numpy as np
    left, top, right, bottom = content_box(gt.shape[2], gt.shape[1], size)
    rows = []
    for i in frame_ids:
        a = gt[i, top:bottom, left:right].astype(np.float64) / 255.
        b = pred[i, top:bottom, left:right].astype(np.float64) / 255.
        mse = float(np.square(a - b).mean())
        ux, uy = cv2.blur(a, (7, 7)), cv2.blur(b, (7, 7))
        vx = (cv2.blur(a * a, (7, 7)) - ux * ux) * (49 / 48)
        vy = (cv2.blur(b * b, (7, 7)) - uy * uy) * (49 / 48)
        cov = (cv2.blur(a * b, (7, 7)) - ux * uy) * (49 / 48)
        ssim_map = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2) /
                    ((ux * ux + uy * uy + .01 ** 2) * (vx + vy + .03 ** 2)))
        rows.append({"frame": i, "mse": mse, "psnr": -10 * math.log10(max(mse, 1e-12)),
                     "ssim": float(ssim_map[3:-3, 3:-3].mean())})
    return {"psnr": mean([r["psnr"] for r in rows]),
            "ssim": mean([r["ssim"] for r in rows]), "per_frame": rows}


def object_errors(task, gt, pred, frame_ids, size):
    errors, penalized, observed = [], [], []
    penalty = math.hypot(*size) if task == "ball" else size[0]
    for i in frame_ids:
        if not gt[i]["usable"]:
            continue
        if pred[i]["usable"]:
            error = math.dist(gt[i]["center"], pred[i]["center"])
            errors.append(error)
            penalized.append(error)
            if not gt[i].get("held_after_exit") and not pred[i].get("held_after_exit"):
                observed.append(error)
        else:
            penalized.append(penalty)
    final = frame_ids[-1]
    fde = (math.dist(gt[final]["center"], pred[final]["center"])
           if gt[final]["usable"] and pred[final]["usable"] else None)
    return {"ade_px": mean(errors), "fde_px": fde,
            "observed_only_ade_px": mean(observed),
            "missing_penalized_ade_px": mean(penalized),
            "paired_valid_frames": len(errors), "gt_valid_frames": len(penalized),
            "gt_coverage": sum(gt[i]["usable"] for i in frame_ids) / len(frame_ids),
            "prediction_coverage": sum(pred[i]["usable"] for i in frame_ids) / len(frame_ids),
            "paired_coverage": len(errors) / len(frame_ids),
            "measurement": "ball_visible_center" if task == "ball" else "door_outer_edge_not_centroid"}


def review_sheet(query, videos, tracks, marker, destination):
    import cv2
    import numpy as np
    frame_ids = query["row"]["evaluation_frame_indices"]
    chosen = [0, frame_ids[len(frame_ids) // 3], frame_ids[2 * len(frame_ids) // 3], frame_ids[-1]]
    strips = []
    for method in ("gt",) + METHODS:
        panels = []
        for idx in chosen:
            frame = videos[method][idx].copy()
            state = tracks[method][idx]
            if marker and marker["center"] is not None:
                x, y = map(round, marker["center"])
                cv2.drawMarker(frame, (x, y), (255, 200, 0), cv2.MARKER_CROSS, 14, 2)
            if state.get("center") is not None:
                x, y = map(round, state["center"])
                cv2.circle(frame, (x, y), 7, (0, 220, 0) if state["usable"] else (0, 0, 255), 2)
            if query["task"] == "door":
                ref = state.get("closed_outer_reference_x")
                if ref is not None:
                    cv2.line(frame, (round(ref), 150), (round(ref), 360), (255, 200, 0), 1)
            height = round(frame.shape[0] * 320 / frame.shape[1])
            frame = cv2.resize(frame, (320, height))
            label = np.full((38, 320, 3), 255, dtype=np.uint8)
            cv2.putText(label, f"{method} frame {idx}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            state_label = state.get("closure_state", state.get("reason", ""))[:43]
            cv2.putText(label, state_label, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
            panels.append(np.concatenate([label, frame], axis=0))
        strips.append(np.concatenate(panels, axis=1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.concatenate(strips, axis=0)):
        raise RuntimeError(f"Cannot write review image {destination}")


def score_query(payload):
    import cv2
    cv2.setNumThreads(1)
    query, config, calibration, output_text = payload
    output = Path(output_text)
    destination = output / "queries" / query["task"] / (query["key"] + ".json")
    if destination.exists():
        return read_json(destination)
    row = query["row"]
    task = query["task"]
    frame_ids = row["evaluation_frame_indices"]
    raw = {method: read_video(path) for method, path in query["paths"].items()}
    shape = raw["gt"].shape
    for method, frames in raw.items():
        if frames.shape != shape or frames.shape[0] != row["length"]:
            raise ValueError(f"Frame/geometry mismatch: {query['key']}/{method}: {frames.shape} vs {shape}")
    videos = {method: native_frames(frames, query["native_size"]) for method, frames in raw.items()}
    first = videos["gt"][0]
    marker = blue_marker(first, config["ball_marker"]) if task == "ball" else None
    tracks = {method: track_video(task, frames, first, calibration, config)
              for method, frames in videos.items()}
    summaries = {method: trajectory_summary(task, states, frame_ids, marker, config)
                 for method, states in tracks.items()}
    result = {"task": task, "key": query["key"], "environment": row["environment"],
              "split": row["dataset_split"], "level": row["action_level"],
              "sample_id": row["sample_id"], "frame_ids": frame_ids,
              "marker": marker, "trajectory_summary": summaries,
              "full_episode_peak_in_window": row.get("full_episode_peak_in_window"),
              "full_episode_end_in_window": row.get("full_episode_end_in_window"),
              "methods": {}}
    for method in METHODS:
        im = image_metrics(raw["gt"], raw[method], frame_ids, query["native_size"])
        obj = object_errors(task, tracks["gt"], tracks[method], frame_ids, query["native_size"])
        if task == "ball":
            a, b = summaries["gt"]["peak_center"], summaries[method]["peak_center"]
            obj["peak_x_error_px"] = abs(a[0] - b[0]) if a and b else None
        else:
            a, b = summaries["gt"]["closed"], summaries[method]["closed"]
            obj["closure_agreement_with_tracked_gt"] = float(a == b) if a is not None and b is not None else None
        result["methods"][method] = {"image": im, "object": obj}
    save_json(output / "tracks" / task / (query["key"] + ".json"), tracks)
    needs_review = any(s["coverage"] < .9 for s in summaries.values())
    chosen = row["dataset_split"] == "test" and row["action_level"] in config["audit_levels"]
    if chosen or needs_review or (marker and marker["center"] is None):
        name = output / "audit" / task / (query["key"] + ".jpg")
        review_sheet(query, videos, tracks, marker, name)
        result["review_sheet"] = str(name)
    save_json(destination, result)
    return result


def aggregate(output, manifest):
    config = manifest["config"]
    rows = []
    for query in manifest["queries"]:
        path = output / "queries" / query["task"] / (query["key"] + ".json")
        if path.exists():
            rows.append(read_json(path))
    report = {"status": config["tracking_status"], "queries_expected": len(manifest["queries"]),
              "queries_scored": len(rows), "image_object": [], "action": [],
              "checkpoint_steps": manifest["checkpoint_steps"],
              "fairness_note": manifest["fairness_note"]}
    for task in config["tasks"]:
        for split in ("train", "test"):
            selected = [r for r in rows if r["task"] == task and r["split"] == split]
            for method in METHODS:
                values = defaultdict(list)
                by_env = defaultdict(lambda: defaultdict(list))
                for row in selected:
                    for section in ("image", "object"):
                        for key, value in row["methods"][method][section].items():
                            if isinstance(value, (int, float)) or value is None:
                                values[key].append(value)
                                by_env[row["environment"]][key].append(value)
                    lp = output / "lpips" / task / (row["key"] + ".json")
                    if lp.exists():
                        value = read_json(lp)[method]["mean"]
                        values["lpips"].append(value)
                        by_env[row["environment"]]["lpips"].append(value)
                report["image_object"].append({"task": task, "split": split, "method": method,
                    "queries": len(selected), "environments": len(by_env),
                    "query_mean": {k: mean(v) for k, v in values.items()},
                    "valid_query_counts": {k: sum(v is not None for v in vv) for k, vv in values.items()},
                    "environment_macro": {k: mean([mean(e.get(k, [])) for e in by_env.values()]) for k in values}})
        test = [r for r in rows if r["task"] == task and r["split"] == "test"]
        environments = sorted({q["row"]["environment"] for q in manifest["queries"] if q["task"] == task})
        for method in METHODS:
            decisions = []
            for env in environments:
                candidates = [r for r in test if r["environment"] == env]
                levels = [r["level"] for r in candidates]
                if len(levels) != len(set(levels)):
                    raise ValueError(f"More than one test query per level: {task}/{env}; define level aggregation first")
                if task == "door":
                    if env in config["door_action_excluded"]:
                        continue
                    states = {r["level"]: r["trajectory_summary"][method]["closed"] for r in candidates}
                    pair = next(([l, l + 1] for l in range(1, 10)
                                 if states.get(l) is True and states.get(l + 1) is True), [])
                    gt = config["door_prefer"][env]
                    score = len(set(pair).intersection(gt)) / 2
                    decisions.append({"environment": env, "predicted_prefer": pair, "gt_prefer": gt,
                                      "score": score, "closed_by_level": states,
                                      "missing_levels": sorted(set(range(1, 11)) - set(levels)),
                                      "uncertain_levels": [l for l, s in states.items() if s is None]})
                else:
                    distances = {r["level"]: r["trajectory_summary"][method]["marker_distance_px"] for r in candidates}
                    truth = {r["level"]: r["trajectory_summary"]["gt"]["marker_distance_px"] for r in candidates}
                    complete_gt = len(truth) == 10 and all(d is not None for d in truth.values())
                    valid = {l: d for l, d in distances.items() if d is not None}
                    chosen = min(valid, key=lambda l: (valid[l], l)) if valid else None
                    best = min(truth, key=lambda l: (truth[l], l)) if complete_gt else None
                    decisions.append({"environment": env, "selected_level": chosen, "gt_best_level": best,
                                      "score": float(chosen == best) if best is not None else None,
                                      "gt_target_regret_px": truth[chosen] - truth[best] if best is not None and chosen in truth else None,
                                      "predicted_distances_px": distances, "gt_distances_px": truth,
                                      "missing_prediction_levels": sorted(set(range(1, 11)) - set(valid)),
                                      "gt_complete": complete_gt})
            report["action"].append({"task": task, "method": method,
                "environment_count": len(decisions), "scorable_count": sum(d["score"] is not None for d in decisions),
                "macro_score": mean([d["score"] for d in decisions]),
                "conservative_score_fixed_denominator": sum(d["score"] or 0 for d in decisions) / len(decisions) if decisions else None,
                "decisions": decisions})
    save_json(output / "summary_provisional.json", report)
    with (output / "image_object_metrics.csv").open("w") as stream:
        fields = ["task", "split", "method", "queries", "psnr", "ssim", "lpips", "ade_px", "fde_px",
                  "missing_penalized_ade_px", "prediction_coverage", "paired_coverage", "peak_x_error_px"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["image_object"]:
            writer.writerow({k: row.get(k, row["environment_macro"].get(k)) for k in fields})
    return report


def cpu_main(output, manifest, workers):
    config = manifest["config"]
    calibration = read_json(ROOT / config["calibration"])
    payloads = [(q, config, calibration, str(output)) for q in manifest["queries"]]
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_query, p): p[0]["key"] for p in payloads}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                row = future.result()
                print(f"SCORED {i}/{len(payloads)} {row['task']} {key}", flush=True)
            except Exception as exc:
                failures.append({"key": key, "error": repr(exc)})
                print(f"FAILED {key} {exc!r}", flush=True)
    save_json(output / "failures.json", failures)
    report = aggregate(output, manifest)
    print(json.dumps({"queries_scored": report["queries_scored"], "failures": len(failures),
                      "output": str(output)}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} queries failed; not a complete score")
    save_json(output / "cpu_complete.json", {"queries": len(payloads), "tracking_status": config["tracking_status"]})


def lpips_main(output, manifest):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from wan_video_action.metrics.global_video import LPIPSEvaluator
    torch.set_num_threads(4)
    evaluator = LPIPSEvaluator(net="alex", device="cuda")
    for i, query in enumerate(manifest["queries"], 1):
        destination = output / "lpips" / query["task"] / (query["key"] + ".json")
        if destination.exists():
            continue
        frame_ids = query["row"]["evaluation_frame_indices"]
        def load_frames(path):
            reader = imageio.get_reader(str(path))
            try:
                frames = np.stack([f for f in reader])
            finally:
                reader.close()
            if len(frames) != query["row"]["length"]:
                raise ValueError(f"Frame count mismatch in {path}")
            left, top, right, bottom = content_box(frames.shape[2], frames.shape[1], query["native_size"])
            return frames[frame_ids, top:bottom, left:right].astype(np.float32) / 255.
        gt = load_frames(query["paths"]["gt"])
        result = {}
        for method in METHODS:
            pred = load_frames(query["paths"][method])
            scores = evaluator(gt, pred, batch_size=16)
            result[method] = {"mean": float(np.mean(scores)), "frame_ids": frame_ids,
                              "per_frame": np.asarray(scores).tolist()}
        save_json(destination, result)
        print(f"LPIPS {i}/{len(manifest['queries'])} {query['key']}", flush=True)
    aggregate(output, manifest)
    save_json(output / "lpips_complete.json", {"queries": len(manifest["queries"]), "net": "alex"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/real97_ball_door_scores_20260912_v1.json")
    parser.add_argument("--mode", choices=("cpu", "lpips", "report"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode != "report" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video scoring must run on a compute node, not the login node")
    config = read_json(ROOT / args.config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest(config, output)
    if args.mode == "cpu":
        cpu_main(output, manifest, args.workers)
    elif args.mode == "lpips":
        lpips_main(output, manifest)
    else:
        report = aggregate(output, manifest)
        print(json.dumps({"scored": report["queries_scored"], "expected": report["queries_expected"]}))


if __name__ == "__main__":
    main()
