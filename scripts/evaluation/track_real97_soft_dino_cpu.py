#!/usr/bin/env python3
"""Dense CPU DINO audit of complete Soft pull videos, isolated from training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv/lib/python3.10/site-packages"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def sample_inventory(config, detector_config):
    records = [json.loads(line) for line in (ROOT / config["inventory"]).read_text().splitlines()
               if line.strip()]
    groups = defaultdict(list)
    for row in records:
        if row["task"] == "soft":
            groups[row["environment"]].append(row)
    rng = random.Random(config["seed"])
    selected = {}
    for environment, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["episode_index"])
        if rows[0]["domain"] == "id":
            chosen = []
            for split, count in (("train", config["id_train_episodes"]),
                                 ("test", config["id_test_episodes"])):
                pool = [row for row in rows if row["split"] == split]
                if len(pool) < count:
                    raise ValueError(f"Insufficient {environment} {split} episodes")
                chosen.extend(rng.sample(pool, count))
        else:
            chosen = rng.sample(rows, min(len(rows), config["episodes_per_environment"]))
        for row in chosen:
            selected[(environment, row["episode_index"])] = dict(row, sampling_reason="stratified_random")
    lookup = {(row["environment"], row["episode_index"]): row for row in records if row["task"] == "soft"}
    failures = set()
    if config["include_previous_failure_episodes"]:
        failures = {tuple(pair) for pair in detector_config["failure_episodes"]}
        for key in failures:
            selected[key] = dict(lookup[key], sampling_reason="previous_detector_failure")
    # Interleave environments and place known difficult cases first for early review.
    result = sorted(selected.values(), key=lambda row: (
        (row["environment"], row["episode_index"]) not in failures,
        row["episode_index"], row["environment"]))
    for index, row in enumerate(result):
        row["sample_index"] = index
    return result


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def select_detection(proposals, previous, frame, config):
    candidates = []
    for row in proposals:
        if row["accepted"] and not any(box_iou(row["box"], other["box"]) >= config["nms_iou"]
                                       for other in candidates):
            candidates.append(row)
    if not candidates:
        return None, "no_filtered_detection"
    if previous is not None and frame-previous[0] <= config["association_memory_frames"]:
        gate = config["max_displacement_per_frame_px"] * (frame-previous[0])
        candidates = [row for row in candidates if math.dist(row["center"], previous[1]) <= gate]
        if not candidates:
            return None, "temporal_jump_rejected"
    selected = candidates[0]
    for alternative in candidates[1:]:
        if (selected["score"]-alternative["score"] < config["ambiguous_score_margin"] and
                math.dist(selected["center"], alternative["center"]) > config["distinct_center_distance_px"]):
            return None, "multiple_plausible_objects"
    return selected, "dense_dino_detection"


def trajectory_svg(path, tracks, title, center_definition="full_box_strict"):
    coordinate_note = ("Native crop coordinates; visible boundary-box centers included; gaps are not interpolated."
                       if center_definition == "visible_box" else
                       "Native crop coordinates; gaps are not interpolated; partial boundary boxes excluded.")
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="430" viewBox="0 0 800 430">',
             '<rect width="800" height="430" fill="white"/>',
             f'<text x="35" y="28" font-family="sans-serif" font-size="16">{html.escape(title)}</text>',
             f'<text x="35" y="50" font-family="sans-serif" font-size="12">{coordinate_note}</text>',
             '<rect x="50" y="80" width="680" height="300" fill="#fafafa" stroke="#999"/>']
    n = max(1, len(tracks)-1)
    for key, color, denominator in (("x", "#2366b0", 512), ("y", "#d36723", 256)):
        segment = []
        segments = []
        for row in tracks:
            if row["measurement_valid"]:
                coordinate = row["center"][0 if key == "x" else 1]
                segment.append(f'{50+680*row["frame"]/n:.2f},{380-300*coordinate/denominator:.2f}')
            elif segment:
                segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for segment in segments:
            lines.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    lines.extend(['<text x="50" y="410" font-family="sans-serif" font-size="12" fill="#2366b0">x / 512</text>',
                  '<text x="150" y="410" font-family="sans-serif" font-size="12" fill="#d36723">y / 256</text>',
                  '<text x="550" y="410" font-family="sans-serif" font-size="12">horizontal axis: full episode frames</text>', '</svg>'])
    path.write_text("\n".join(lines))


def audit_episode(sample, output, detect, config, filter_config):
    import cv2
    import numpy as np
    from wan_video_action.real97_eval.open_vocab_filter import filter_proposals

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "episode.json", sample)
    cap = cv2.VideoCapture(sample["video_path"])
    fps = float(sample["fps"])
    width, height = filter_config["native_size"]
    video = output / "overlay.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"Cannot open video decoder/writer: {sample['video_path']}")
    review_indices = set(np.linspace(0, sample["length"]-1, 9).round().astype(int).tolist())
    reviews = []
    tracks = []
    previous = None
    recent_line = []
    started = time.monotonic()
    print("[episode_start] " + json.dumps(dict(environment=sample["environment"],
          episode=sample["episode_index"], expected_frames=sample["length"])), flush=True)
    try:
        with (output / "tracks.jsonl").open("w") as handle:
            frame = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if image.shape[:2] != (height, width):
                    raise ValueError(f"Unexpected native image size: {image.shape}")
                begin = time.monotonic()
                proposals = filter_proposals(image, detect(image), filter_config)
                selection, reason = select_detection(proposals, previous, frame, config)
                center = selection["center"] if selection is not None else None
                boundary = False
                jump = None
                reacquired = False
                if selection is not None:
                    x0, y0, x1, y1 = selection["box"]
                    margin = config["boundary_margin_px"]
                    boundary = (selection["boundary_clipped"] or min(x0, y0) <= margin or
                                x1 >= width-margin or y1 >= height-margin)
                    if config.get("center_definition", "full_box_strict") == "visible_box":
                        center = [(max(0.0, x0)+min(float(width), x1))/2,
                                  (max(0.0, y0)+min(float(height), y1))/2]
                        selection = dict(selection, original_detector_center=selection["center"], center=center)
                    if previous is not None:
                        if frame-previous[0] == 1:
                            jump = math.dist(center, previous[1])
                        else:
                            reacquired = True
                    previous = (frame, center)
                visible_mode = config.get("center_definition", "full_box_strict") == "visible_box"
                valid = selection is not None and (visible_mode or not boundary)
                row = dict(frame=frame, timestamp_seconds=frame/fps, center=center,
                           original_center=None if center is None else [
                               center[i]+config["original_image_crop_offset"][i] for i in range(2)],
                           measurement_valid=valid, boundary_partial=boundary,
                           reason=("visible_boundary_detection" if visible_mode else "boundary_partial") if boundary else reason,
                           confidence=None if selection is None else selection["score"],
                           consecutive_step_px=jump,
                           jump_warning=jump is not None and jump > config["jump_warning_px"],
                           reacquired_after_gap=reacquired, selected=selection,
                           all_proposals=proposals, detector_seconds=time.monotonic()-begin)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                # Keep only compact track rows in memory; raw proposals remain on disk.
                tracks.append({key: value for key, value in row.items() if key not in ("all_proposals", "selected")})
                canvas = image.copy()
                if valid:
                    recent_line.append(tuple(map(round, center)))
                    recent_line = recent_line[-30:]
                    if len(recent_line) > 1:
                        cv2.polylines(canvas, [np.asarray(recent_line, dtype=np.int32)], False, (255, 170, 0), 1)
                else:
                    recent_line = []
                if selection is not None:
                    x0, y0, x1, y1 = map(round, selection["box"])
                    color = (0, 200, 0) if valid else (0, 165, 255)
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)
                    cv2.drawMarker(canvas, tuple(map(round, center)), color, cv2.MARKER_CROSS, 8, 1)
                label = (f'{sample["environment"]} ep{sample["episode_index"]} '
                         f'{sample["domain"]}/{sample["split"]} f{frame} {row["reason"]}')
                cv2.rectangle(canvas, (0, 0), (width, 20), (245, 245, 245), -1)
                cv2.putText(canvas, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 0, 0), 1)
                writer.write(canvas)
                if frame in review_indices:
                    reviews.append(canvas)
                frame += 1
                if frame % 50 == 0:
                    print(f'[frame] {sample["environment"]} ep{sample["episode_index"]} {frame}/{sample["length"]}', flush=True)
    finally:
        cap.release()
        writer.release()
    if not tracks:
        raise RuntimeError("Video decoded zero frames")
    if shutil.which("ffmpeg"):
        converted = output / "overlay_h264.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
                        "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted)], check=True)
        os.replace(converted, video)
    if reviews:
        while len(reviews) < 9:
            reviews.append(np.full_like(reviews[0], 245))
        cv2.imwrite(str(output / "contact_sheet.jpg"), np.vstack([
            np.hstack(reviews[i:i+3]) for i in range(0, 9, 3)]))
    with (output / "trajectory.csv").open("w", newline="") as handle:
        table = csv.writer(handle)
        table.writerow(["frame", "time_s", "x_crop", "y_crop", "x_original", "y_original", "valid", "reason", "score"])
        for row in tracks:
            table.writerow([row["frame"], row["timestamp_seconds"], *(row["center"] or [None, None]),
                            *(row["original_center"] or [None, None]), row["measurement_valid"],
                            row["reason"], row["confidence"]])
    trajectory_svg(output / "trajectory.svg", tracks, f'{sample["environment"]} episode {sample["episode_index"]}',
                   config.get("center_definition", "full_box_strict"))
    runs = []
    gap = 0
    for row in tracks:
        if not row["measurement_valid"]:
            gap += 1
        elif gap:
            runs.append(gap)
            gap = 0
    if gap:
        runs.append(gap)
    valid_count = sum(row["measurement_valid"] for row in tracks)
    elapsed = time.monotonic()-started
    summary = dict(sample_index=sample["sample_index"], environment=sample["environment"],
                   episode_index=sample["episode_index"], split=sample["split"], domain=sample["domain"],
                   sampling_reason=sample["sampling_reason"], frames=len(tracks), expected_frames=sample["length"],
                   decode_complete=len(tracks) == sample["length"], valid_frames=valid_count,
                   usable_frame_fraction=valid_count/len(tracks),
                   detected_frames=sum(row["center"] is not None for row in tracks),
                   boundary_partial_frames=sum(row["boundary_partial"] for row in tracks),
                   missing_runs=len(runs), longest_missing_run_frames=max(runs, default=0),
                   jump_warnings=sum(row["jump_warning"] for row in tracks),
                   reacquisitions=sum(row["reacquired_after_gap"] for row in tracks),
                   reasons=dict(Counter(row["reason"] for row in tracks)), elapsed_seconds=elapsed,
                   mean_detector_seconds=sum(row["detector_seconds"] for row in tracks)/len(tracks),
                   formally_certified=False, center_definition=config.get("center_definition", "full_box_strict"),
                   overlay=str(video), contact_sheet=str(output / "contact_sheet.jpg"),
                   trajectory=str(output / "trajectory.svg"))
    write_json(output / "summary.json", summary)
    print("[episode_done] " + json.dumps(summary), flush=True)
    return summary


def aggregate(run, samples, center_definition="full_box_strict"):
    summaries = []
    missing = []
    for sample in samples:
        path = run / "episodes" / f'{sample["environment"]}_ep{sample["episode_index"]:06d}' / "summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text()))
        else:
            missing.append(dict(environment=sample["environment"], episode_index=sample["episode_index"]))
    frames = sum(row["frames"] for row in summaries)
    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["domain"] + "/" + row["split"]].append(row)
    def stats(rows):
        total = sum(row["frames"] for row in rows)
        return dict(episodes=len(rows), frames=total,
                    valid_frames=sum(row["valid_frames"] for row in rows),
                    frame_weighted_coverage=sum(row["valid_frames"] for row in rows)/total if total else None,
                    episode_mean_coverage=sum(row["usable_frame_fraction"] for row in rows)/len(rows) if rows else None,
                    jump_warnings=sum(row["jump_warnings"] for row in rows),
                    complete_decodes=sum(row["decode_complete"] for row in rows))
    summary = dict(expected_episodes=len(samples), completed_episodes=len(summaries),
                   missing_episodes=missing, frames=frames, all_episodes_complete=not missing,
                   overall=stats(summaries), by_domain_split={key: stats(rows) for key, rows in grouped.items()},
                   formally_certified=False, center_definition=center_definition,
                   caveat="Coverage and temporal smoothness do not establish tracking accuracy; visually review overlays, including low-coverage and jump cases.")
    write_json(run / "summary.json", summary)
    write_json(run / "episode_summaries.json", summaries)
    lines = ['<!doctype html><html><head><meta charset="utf-8"><title>Soft pull dense DINO audit</title></head>',
             '<body style="font-family:sans-serif;max-width:1400px;margin:30px auto">',
             '<h1>Soft pull full-video tracking audit</h1>',
             ('<p>Green: visible-object box center, including boundary boxes. Missing frames are not filled. Coverage is not accuracy.</p>'
              if center_definition == "visible_box" else
              '<p>Green: usable detection. Orange: partial boundary detection. Missing frames are not filled. Coverage is not accuracy.</p>'),
             f'<p>Completed {len(summaries)} / {len(samples)} episodes, {frames} native frames.</p>']
    for row in sorted(summaries, key=lambda row: (row["usable_frame_fraction"], -row["jump_warnings"])):
        relative = Path(row["overlay"]).parent.relative_to(run).as_posix()
        lines.append(f'<h2>{html.escape(row["environment"])} ep{row["episode_index"]} '
                     f'{row["domain"]}/{row["split"]}: coverage {row["usable_frame_fraction"]:.1%}, '
                     f'jump warnings {row["jump_warnings"]}</h2>')
        lines.append(f'<p><a href="{relative}/overlay.mp4">Full overlay video</a> | '
                     f'<a href="{relative}/trajectory.svg">Trajectory</a> | '
                     f'<a href="{relative}/tracks.jsonl">Per-frame detections</a></p>')
        lines.append(f'<img loading="lazy" width="1000" src="{relative}/contact_sheet.jpg" alt="Episode audit contact sheet">')
    lines.append('</body></html>')
    (run / "index.html").write_text("\n".join(lines))
    print("[aggregate] " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm CPU allocation required; do not run video analysis on the login node")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = json.loads(args.config.read_text())
    detector_config = json.loads((ROOT / config["detector_config"]).read_text())
    filter_config = json.loads((ROOT / config["filter_config"]).read_text())
    if config["frame_stride"] != 1:
        raise ValueError("This audit must decode and detect every native video frame")
    samples = sample_inventory(config, detector_config)
    args.run.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args.run, samples)
        return
    if not 0 <= args.shard < config["shards"]:
        raise ValueError("Invalid shard")
    shard_output = args.run / f"shard{args.shard:02d}"
    shard_output.mkdir(exist_ok=False)
    write_json(shard_output / "config_snapshot.json", dict(tracking=config, detector=detector_config, filtering=filter_config))
    write_json(shard_output / "sample_manifest.json", samples)
    assigned = [sample for sample in samples if sample["sample_index"] % config["shards"] == args.shard]
    print("[plan] " + json.dumps(dict(total_episodes=len(samples), environments=len({row['environment'] for row in samples}),
          total_frames=sum(row["length"] for row in samples), shard=args.shard, assigned_episodes=len(assigned),
          assigned_frames=sum(row["length"] for row in assigned))), flush=True)
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection

    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.set_num_interop_threads(1)
    cache = str(ROOT / detector_config["cache_dir"])
    processor = AutoProcessor.from_pretrained(detector_config["model_id"], cache_dir=cache, local_files_only=True)
    model_config = AutoConfig.from_pretrained(detector_config["model_id"], cache_dir=cache, local_files_only=True)
    model_config.disable_custom_kernels = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_config["model_id"], config=model_config, cache_dir=cache, local_files_only=True,
        use_safetensors=True, trust_remote_code=False).to("cpu").eval()
    threshold_name = ("threshold" if "threshold" in inspect.signature(
        processor.post_process_grounded_object_detection).parameters else "box_threshold")
    print("[model_ready] Grounding DINO Tiny CPU, local cache, no GPU", flush=True)

    def detect(image):
        inputs = processor(images=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                           text=detector_config["prompt"], return_tensors="pt", size=detector_config["detector_resize"])
        with torch.inference_mode():
            output = model(**inputs)
        detections = processor.post_process_grounded_object_detection(output, inputs.input_ids, **{
            threshold_name: detector_config["box_threshold"], "text_threshold": detector_config["text_threshold"],
            "target_sizes": [(image.shape[0], image.shape[1])]})[0]
        labels = detections.get("text_labels")
        if labels is None:
            labels = detections.get("labels", [])
        proposals = []
        for index, (box, score) in enumerate(zip(detections["boxes"], detections["scores"])):
            x0, y0, x1, y1 = map(float, box.tolist())
            proposals.append(dict(box=[x0, y0, x1, y1], center=[(x0+x1)/2, (y0+y1)/2],
                                  score=float(score), label=str(labels[index]) if index < len(labels) else ""))
        return sorted(proposals, key=lambda row: row["score"], reverse=True)

    failures = []
    completed = []
    for sample in assigned:
        destination = args.run / "episodes" / f'{sample["environment"]}_ep{sample["episode_index"]:06d}'
        try:
            completed.append(audit_episode(sample, destination, detect, config, filter_config))
        except Exception as error:
            traceback.print_exc()
            failure = dict(environment=sample["environment"], episode_index=sample["episode_index"], error=repr(error))
            failures.append(failure)
            write_json(shard_output / "failures.json", failures)
            print("[episode_failed] " + json.dumps(failure), flush=True)
        write_json(shard_output / "progress.json", dict(completed=completed, failures=failures,
                   expected_episodes=len(assigned), all_done=len(completed)+len(failures) == len(assigned)))
    if failures:
        raise RuntimeError(f"{len(failures)} episodes failed; see shard failures.json")


if __name__ == "__main__":
    main()
