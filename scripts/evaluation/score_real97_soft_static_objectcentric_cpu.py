#!/usr/bin/env python3
"""Split-blind detector audit and missing-aware Soft static-start diagnostics."""

import argparse
from collections import Counter, defaultdict
import fcntl
import html
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv/lib/python3.10/site-packages"))
import cv2
import numpy as np
from track_real97_soft_dino_cpu import select_detection
from wan_video_action.real97_eval.open_vocab_filter import filter_proposals


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key_for(index, row):
    return f'q{index:04d}_{row["environment"]}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'


def build_detector(config):
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.set_num_interop_threads(1)
    cache = str(ROOT / config["cache_dir"])
    processor = AutoProcessor.from_pretrained(config["model_id"], cache_dir=cache, local_files_only=True)
    model_config = AutoConfig.from_pretrained(config["model_id"], cache_dir=cache, local_files_only=True)
    model_config.disable_custom_kernels = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config["model_id"], config=model_config, cache_dir=cache, local_files_only=True,
        use_safetensors=True, trust_remote_code=False).to("cpu").eval()
    threshold_name = "threshold" if "threshold" in inspect.signature(processor.post_process_grounded_object_detection).parameters else "box_threshold"

    def detect(frame):
        inputs = processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                           text=config["prompt"], return_tensors="pt", size=config["detector_resize"])
        with torch.inference_mode():
            prediction = model(**inputs)
        result = processor.post_process_grounded_object_detection(prediction, inputs.input_ids, **{
            threshold_name: config["box_threshold"], "text_threshold": config["text_threshold"],
            "target_sizes": [(frame.shape[0], frame.shape[1])]})[0]
        labels = result.get("text_labels", result.get("labels", []))
        proposals = []
        for index, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
            x0, y0, x1, y1 = map(float, box.tolist())
            proposals.append({"box": [x0, y0, x1, y1], "center": [(x0 + x1) / 2, (y0 + y1) / 2],
                              "score": float(score), "label": str(labels[index]) if index < len(labels) else ""})
        return sorted(proposals, key=lambda item: item["score"], reverse=True)
    return detect


def track(path, row, detect, config, tracking, filtering):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    frames, states, previous = [], [], None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index = len(frames)
            if index >= config["model_frames"] or [frame.shape[1], frame.shape[0]] != config["model_size"]:
                raise RuntimeError(f"Unexpected video geometry or length: {path}, {index}, {frame.shape}")
            frame = cv2.resize(frame, tuple(config["native_size"]), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            native = int(row["native_frame_indices"][index])
            state = {"frame": index, "native_frame": native, "timestamp_seconds": native / config["native_fps"],
                     "center": None, "measurement_valid": False, "boundary_partial": False,
                     "reason": "observed_padding_not_remeasured", "selected": None, "proposals": []}
            if index < config["observed_frames"] - 1:
                states.append(state)
                continue
            start = time.monotonic()
            proposals = filter_proposals(frame, detect(frame), filtering)
            selected, reason = select_detection(proposals, previous, native, tracking)
            state.update(reason=reason, proposals=proposals, detector_seconds=time.monotonic() - start)
            if selected is not None:
                x0, y0, x1, y1 = selected["box"]
                width, height = config["native_size"]
                visible = [max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)]
                center = [(visible[0] + visible[2]) / 2, (visible[1] + visible[3]) / 2]
                margin = tracking["boundary_margin_px"]
                boundary = bool(selected["boundary_clipped"] or min(x0, y0) <= margin or x1 >= width - margin or y1 >= height - margin)
                step = math.dist(center, previous[1]) if previous is not None else None
                delta = native - previous[0] if previous is not None else None
                state.update(center=center, original_center=[center[i] + config["original_image_crop_offset"][i] for i in range(2)],
                             visible_box=visible, measurement_valid=True, boundary_partial=boundary,
                             reason="visible_boundary_detection" if boundary else reason,
                             confidence=float(selected["score"]), selected=selected,
                             displacement_since_last_detection_px=step, native_frames_since_detection=delta,
                             displacement_per_native_frame_px=step / delta if step is not None and delta and delta > 0 else None,
                             jump_warning=bool(step is not None and step > tracking["jump_warning_px"]))
                previous = (native, center)
            states.append(state)
    finally:
        capture.release()
    if len(frames) != config["model_frames"]:
        raise RuntimeError(f"Incomplete decode: {path}: {len(frames)}")
    return frames, states


def score(gt, predicted, row, config):
    wanted = list(row["evaluation_frame_indices"])
    if len(wanted) != len(set(wanted)) or any(index < config["observed_frames"] or index >= config["model_frames"] for index in wanted):
        raise RuntimeError("Invalid prediction-only evaluation mask")
    native = [row["native_frame_indices"][index] for index in wanted]
    if len(native) != len(set(native)) or any(index <= 0 or index >= row["total_frames"] for index in native):
        raise RuntimeError("Observed or padded timestamps entered the evaluation horizon")
    good = lambda state: bool(state.get("measurement_valid")) and state.get("center") is not None
    pairs = [index for index in wanted if good(gt[index]) and good(predicted[index])]
    errors = [float(np.linalg.norm(np.asarray(predicted[index]["center"]) - gt[index]["center"])) for index in pairs]
    result = {"requested_frames": len(wanted), "paired_frames": len(pairs),
              "coverage": len(pairs) / len(wanted) if wanted else None,
              "gt_valid_frames": sum(good(gt[index]) for index in wanted),
              "prediction_valid_frames": sum(good(predicted[index]) for index in wanted),
              "center_ade_px": float(np.mean(errors)) if errors else None,
              "center_fde_px": errors[-1] if pairs and pairs[-1] == wanted[-1] else None,
              "complete_horizon": bool(wanted) and len(pairs) == len(wanted),
              "gt_boundary_frames": sum(gt[index].get("boundary_partial", False) for index in wanted),
              "prediction_boundary_frames": sum(predicted[index].get("boundary_partial", False) for index in wanted),
              "per_frame_errors": [{"model_frame": index, "native_frame": row["native_frame_indices"][index], "error_px": error}
                                   for index, error in zip(pairs, errors)],
              "static_baseline_ade_on_paired_frames_px": None, "displacement_ade_px": None,
              "gt_net_dx_px": None, "prediction_net_dx_px": None,
              "formal_metric_approved": False}
    if errors:
        result["center_ade_crop_diagonal_normalized"] = result["center_ade_px"] / math.hypot(*config["native_size"])
        result["center_ade_camera_diagonal_normalized"] = result["center_ade_px"] / math.hypot(*config["camera_size"])
    anchor = config["observed_frames"] - 1
    if good(gt[anchor]):
        origin = np.asarray(gt[anchor]["center"])
        baseline = [float(np.linalg.norm(np.asarray(gt[index]["center"]) - origin)) for index in pairs]
        result["static_baseline_ade_on_paired_frames_px"] = float(np.mean(baseline)) if baseline else None
        if wanted and good(gt[wanted[-1]]):
            result["gt_net_dx_px"] = float(gt[wanted[-1]]["center"][0] - origin[0])
        if good(predicted[anchor]):
            pred_origin = np.asarray(predicted[anchor]["center"])
            relative_errors = [float(np.linalg.norm((np.asarray(predicted[index]["center"]) - pred_origin) -
                                                   (np.asarray(gt[index]["center"]) - origin))) for index in pairs]
            result["displacement_ade_px"] = float(np.mean(relative_errors)) if relative_errors else None
            result["initial_center_difference_px"] = float(np.linalg.norm(pred_origin - origin))
            if wanted and good(predicted[wanted[-1]]):
                result["prediction_net_dx_px"] = float(predicted[wanted[-1]]["center"][0] - pred_origin[0])
    return result


def annotated(frame, state, label):
    canvas = cv2.copyMakeBorder(frame.copy(), 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    if state.get("measurement_valid"):
        x0, y0, x1, y1 = map(round, state["visible_box"])
        cv2.rectangle(canvas, (x0, y0 + 26), (x1, y1 + 26), (20, 180, 20), 1)
        center = (round(state["center"][0]), round(state["center"][1]) + 26)
        cv2.drawMarker(canvas, center, (0, 0, 220), cv2.MARKER_CROSS, 8, 1)
    cv2.putText(canvas, f"{label} | f{state['native_frame']} | {state['reason']}", (4, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)
    return canvas


def trajectory_svg(path, tracks):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="530">',
             '<rect width="760" height="530" fill="white"/>',
             '<text x="40" y="25" font-family="sans-serif">Visible-object centers: GT blue; Standard orange; missing detections are gaps.</text>']
    for coordinate, maximum, top in ((0, 512, 60), (1, 256, 295)):
        parts.append(f'<rect x="55" y="{top}" width="660" height="175" fill="#fafafa" stroke="#999"/>')
        parts.append(f'<text x="12" y="{top + 90}" font-family="sans-serif">{"x" if coordinate == 0 else "y"}</text>')
        for kind, color in (("gt", "#2474b5"), ("standard", "#d36b24")):
            segment = []
            for state in tracks[kind][4:] + [{"measurement_valid": False}]:
                if state["measurement_valid"]:
                    segment.append(f'{55 + 660 * state["native_frame"] / 84:.2f},{top + 175 * (1 - state["center"][coordinate] / maximum):.2f}')
                elif segment:
                    parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')
                    segment = []
    parts.append('<text x="55" y="505" font-family="sans-serif">Native frames 0 to 84 (0 to 4.2 seconds). Coordinates in the 512 x 256 dataset crop.</text></svg>')
    path.write_text("\n".join(parts))


def process(index, row, source, destination, detect, config, tracking, filtering):
    from infer_real97_reference_trial import write_video
    destination.mkdir(parents=True, exist_ok=False)
    key = key_for(index, row)
    marker = read_json(source / "completed" / (key + ".json"))
    if marker["observed_frames"] != 5 or not marker["paired_gt"]:
        raise RuntimeError("Expected a factual five-history-frame query")
    all_frames, all_tracks = {}, {}
    for kind in ("gt", "standard"):
        frames, states = track(Path(marker["paths"][kind]), row, detect, config, tracking, filtering)
        all_frames[kind], all_tracks[kind] = frames, states
        with (destination / f"{kind}_tracks.jsonl").open("w") as handle:
            for state in states:
                handle.write(json.dumps(state, allow_nan=False) + "\n")
        print(f"[tracked] {key} {kind} valid={sum(state['measurement_valid'] for state in states[5:])}/28", flush=True)
    metrics = score(all_tracks["gt"], all_tracks["standard"], row, config)
    summary = {"index": index, "key": key, "environment": row["environment"], "split": row["dataset_split"],
               "episode_index": row["episode_index"], "metrics": metrics, "row": row,
               "reasons": {kind: dict(Counter(state["reason"] for state in states[5:])) for kind, states in all_tracks.items()},
               "tracking_seconds": sum(state.get("detector_seconds", 0) for states in all_tracks.values() for state in states),
               "center_definition": "visible_box", "formal_metric_approved": False}
    panels = [np.hstack([annotated(all_frames[kind][frame], all_tracks[kind][frame], kind) for kind in ("gt", "standard")])
              for frame in range(4, 33)]
    write_video(destination / "tracking_overlay.mp4", (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in panels), config["playback_fps"])
    for label, frame in (("first", 0), ("middle", 14), ("last", 28)):
        cv2.imwrite(str(destination / f"{label}.png"), panels[frame])
    sampled = [panels[index] for index in (0, 5, 11, 17, 23, 28)]
    cv2.imwrite(str(destination / "contact_sheet.jpg"), np.vstack([np.hstack(sampled[index:index + 2]) for index in (0, 2, 4)]))
    trajectory_svg(destination / "trajectory.svg", all_tracks)
    write_json(destination / "summary.json", summary)
    print("[query_done] " + json.dumps({"key": key, "coverage": metrics["coverage"], "ADE": metrics["center_ade_px"], "FDE": metrics["center_fde_px"]}), flush=True)
    return summary


def aggregate(output, rows, config):
    summaries, missing = [], []
    for index, row in enumerate(rows):
        path = output / "cases" / key_for(index, row) / "summary.json"
        if path.exists():
            summaries.append(read_json(path))
        else:
            missing.append(key_for(index, row))
    def statistics(items):
        metrics = [item["metrics"] for item in items]
        wanted = sum(item["requested_frames"] for item in metrics)
        paired = sum(item["paired_frames"] for item in metrics)
        mean = lambda values: float(np.mean(values)) if values else None
        ade_sum = sum(item["center_ade_px"] * item["paired_frames"] for item in metrics if item["center_ade_px"] is not None)
        common_baselines = [item for item in metrics if item["static_baseline_ade_on_paired_frames_px"] is not None and item["center_ade_px"] is not None]
        return {"queries": len(items), "requested_frames": wanted, "paired_frames": paired,
                "paired_coverage": paired / wanted if wanted else None,
                "gt_coverage": sum(item["gt_valid_frames"] for item in metrics) / wanted if wanted else None,
                "prediction_coverage": sum(item["prediction_valid_frames"] for item in metrics) / wanted if wanted else None,
                "frame_weighted_ADE_px": ade_sum / paired if paired else None,
                "episode_mean_ADE_px": mean([item["center_ade_px"] for item in metrics if item["center_ade_px"] is not None]),
                "mean_FDE_px": mean([item["center_fde_px"] for item in metrics if item["center_fde_px"] is not None]),
                "valid_final_queries": sum(item["center_fde_px"] is not None for item in metrics),
                "complete_horizon_queries": sum(item["complete_horizon"] for item in metrics),
                "complete_horizon_mean_ADE_px": mean([item["center_ade_px"] for item in metrics if item["complete_horizon"]]),
                "same_queries_model_ADE_px": mean([item["center_ade_px"] for item in common_baselines]),
                "same_queries_static_baseline_ADE_px": mean([item["static_baseline_ade_on_paired_frames_px"] for item in common_baselines]),
                "queries_with_static_baseline": len(common_baselines)}
    groups = defaultdict(list)
    for item in summaries:
        groups["split/" + item["split"]].append(item)
        groups["environment/" + item["environment"]].append(item)
    result = {"expected_queries": len(rows), "completed_queries": len(summaries), "missing_queries": missing,
              "overall": statistics(summaries), "groups": {key: statistics(items) for key, items in groups.items()},
              "formal_metric_approved": False,
              "caution": "Detector-derived center errors are provisional. Coverage is not accuracy; GT and generated tracking must be visually audited."}
    write_json(output / "summary.json", result)
    write_json(output / "per_query.json", summaries)
    rng = random.Random(config["seed"])
    review, seen = [], set()
    for split in ("train", "test"):
        pool = sorted([item for item in summaries if item["split"] == split], key=lambda item: item["index"])
        for item in rng.sample(pool, min(len(pool), config["random_review_per_split"])):
            review.append({"key": item["key"], "purpose": "fixed_seed_random_" + split})
            seen.add(item["key"])
    targeted = sorted(summaries, key=lambda item: (item["metrics"]["coverage"], -(item["metrics"]["center_ade_px"] or 0)))[:8]
    targeted += sorted(summaries, key=lambda item: -(item["metrics"]["center_ade_px"] or 0))[:4]
    for item in targeted:
        if item["key"] not in seen:
            review.append({"key": item["key"], "purpose": "targeted_low_coverage_or_high_error_not_random"})
            seen.add(item["key"])
    for item in review:
        item["contact_sheet"] = str(output / "cases" / item["key"] / "contact_sheet.jpg")
    write_json(output / "review_manifest.json", review)
    page = ['<!doctype html><html><meta charset="utf-8"><title>Soft static-start tracking audit</title><body>',
            '<h1>Soft static-start: GT and Standard visible-center tracking</h1>',
            '<p>Initial five frames excluded. Green box/red center. Missing detections remain missing. No action-selection metric. NOT formally certified.</p>',
            '<p><a href="summary.json">Summary</a> | <a href="per_query.json">Per-query metrics</a> | <a href="review_manifest.json">Review cohort</a></p>']
    for item in summaries:
        key = html.escape(item["key"])
        page.append(f'<h2>{key}</h2><p>Coverage {item["metrics"]["coverage"]:.1%}; ADE {item["metrics"]["center_ade_px"]}</p>')
        page.append(f'<a href="cases/{key}/tracking_overlay.mp4">Tracking video</a> | <a href="cases/{key}/trajectory.svg">Trajectory</a>')
        page.append(f'<p><img loading="lazy" width="1024" src="cases/{key}/contact_sheet.jpg"></p>')
    (output / "index.html").write_text("\n".join(page + ['</body></html>']))
    print("[aggregate] " + json.dumps(result), flush=True)
    if missing:
        raise RuntimeError(f"Incomplete evaluation: {len(missing)} missing queries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm allocation required; never decode videos on the login node")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(ROOT)
    cv2.setNumThreads(1)
    config = read_json(args.config)
    source = ROOT / config["source_inference"]
    rows = jsonl(source / "query.jsonl")
    if len(rows) != config["expected_queries"] or any(row["num_history_frames"] != 5 or row["observed_native_frame_indices"] != [0] * 5 for row in rows):
        raise RuntimeError("Not the expected sixty static-start queries")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(output, rows, config)
        return
    if not 0 <= args.shard < config["shards"]:
        raise RuntimeError("Invalid shard")
    tracking = read_json(ROOT / config["tracking_config"])
    filtering = read_json(ROOT / config["filter_config"])
    detector = read_json(ROOT / config["detector_config"])
    shard = output / f"shard{args.shard:02d}"
    shard.mkdir(exist_ok=args.resume)
    # Keep the descriptor alive until this process exits. A requeued process
    # acquires the released lock; a concurrent duplicate cannot overwrite files.
    lock = (shard / ".writer.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns shard {args.shard}; refusing concurrent writes") from error
    snapshot = {"experiment": config, "tracking": tracking, "filtering": filtering, "detector": detector}
    frozen = shard / "frozen_config.json"
    if frozen.exists():
        if not args.resume or read_json(frozen) != snapshot:
            raise RuntimeError("Existing shard configuration differs; completed results must not be mixed")
    else:
        write_json(frozen, snapshot)
    completed, failures = [], []
    pending = []
    for index, row in enumerate(rows):
        if index % config["shards"] != args.shard:
            continue
        key = key_for(index, row)
        destination = output / "cases" / key
        final = destination / "summary.json"
        if final.exists():
            if not args.resume:
                raise RuntimeError(f"Completed result already exists: {key}")
            saved = read_json(final)
            if saved.get("key") != key or saved.get("row") != row or saved.get("center_definition") != "visible_box":
                raise RuntimeError(f"Completed result identity mismatch: {key}")
            required = ("gt_tracks.jsonl", "standard_tracks.jsonl", "tracking_overlay.mp4",
                        "first.png", "middle.png", "last.png", "contact_sheet.jpg", "trajectory.svg")
            missing = [name for name in required if not (destination / name).is_file() or (destination / name).stat().st_size == 0]
            if missing:
                raise RuntimeError(f"Completed marker has missing artifacts; preserving result for review: {key}: {missing}")
            completed.append(key)
            print(f"[resume_skip] {key}", flush=True)
        else:
            if destination.exists() and not args.resume:
                raise RuntimeError(f"Incomplete output exists; explicit --resume required: {key}")
            pending.append((index, row))
    attempt = f"job{os.environ['SLURM_JOB_ID']}_restart{os.environ.get('SLURM_RESTART_COUNT', '0')}_{time.time_ns()}"
    write_json(shard / "attempts" / (attempt + ".json"), {
        "resume": args.resume, "reused_completed_queries": completed,
        "pending_queries": [key_for(index, row) for index, row in pending],
        "measurement_algorithm_changed": False,
    })
    write_json(shard / "progress.json", {"completed": completed, "failures": failures, "pending_count": len(pending)})
    print(f"[resume_plan] shard={args.shard} completed={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        return
    detect = build_detector(detector)
    print(f"[model_ready] Grounding DINO Tiny CPU; shard {args.shard}", flush=True)
    for index, row in pending:
        key = key_for(index, row)
        destination = output / "cases" / key
        try:
            if destination.exists():
                # Only an incomplete per-query directory can arrive here.
                # Rename it into an audit archive; never delete completed data.
                if (destination / "summary.json").exists():
                    raise RuntimeError(f"A completed result appeared unexpectedly: {key}")
                archive = output / "interrupted_attempts" / f"{key}_{attempt}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    raise RuntimeError(f"Archive collision; refusing replacement: {archive}")
                destination.rename(archive)
                print(f"[archived_incomplete] {key} -> {archive}", flush=True)
            completed.append(process(index, row, source, destination, detect, config, tracking, filtering)["key"])
        except Exception as error:
            import traceback
            traceback.print_exc()
            failures.append({"key": key, "error": repr(error)})
        write_json(shard / "progress.json", {"completed": completed, "failures": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} queries failed on shard {args.shard}")


if __name__ == "__main__":
    main()
