#!/usr/bin/env python3
"""Build leakage-free LightSwitch windows around physical button presses.

The event frame is always read from the dataset generator's
``episodes[].events[].step``.  That value is the physical button-trigger frame
and exists whether or not the pressed button controls the lamp.  Lamp state,
lamp transitions, and causal labels are annotations only and never influence
window timing.

This script deliberately does not modify the legacy event33 metadata, whose
outcome-dependent detector is retained solely for experiment reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


NUM_FRAMES = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--legacy-metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-index-min", type=int, default=11)
    parser.add_argument("--event-index-max", type=int, default=22)
    parser.add_argument("--window-seed", type=int, default=20260827)
    parser.add_argument(
        "--camera-mode", choices=("both", "main"), default="both"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def deterministic_index(sample_key: str, low: int, high: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_key}".encode("utf-8")).digest()
    return low + int.from_bytes(digest[:8], "big") % (high - low + 1)


def main() -> None:
    args = parse_args()
    if not 0 <= args.event_index_min <= args.event_index_max < NUM_FRAMES:
        raise ValueError(
            f"Event-index range must be inside [0, {NUM_FRAMES - 1}], got "
            f"[{args.event_index_min}, {args.event_index_max}]."
        )

    dataset_root = args.dataset_base_path.resolve()
    generator_path = (
        dataset_root / "robomme_light_switch_independent_controls_metadata.json"
    )
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    generator_episodes = {
        int(episode["episode_index"]): episode for episode in generator["episodes"]
    }
    legacy_path = (
        args.legacy_metadata_dir / "event_centered_group20_train.jsonl"
    )
    legacy_rows = read_jsonl(legacy_path)
    legacy_by_event = {
        (int(row["episode_index"]), int(row["press_number"])): row
        for row in legacy_rows
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_lengths: dict[int, int] = {}
    output_rows: list[dict] = []

    for episode_index in sorted(generator_episodes):
        episode = generator_episodes[episode_index]
        events = episode["events"]
        if len(events) != 8:
            raise ValueError(
                f"Episode {episode_index} has {len(events)} physical events; expected 8."
            )
        for press_number, physical_event in enumerate(events, start=1):
            key = (episode_index, press_number)
            legacy = legacy_by_event[key]
            button_color = str(physical_event["button_color"])
            if button_color != str(legacy["button_color"]):
                raise ValueError(
                    f"Button mismatch at episode {episode_index}, press {press_number}: "
                    f"generator={button_color}, legacy={legacy['button_color']}"
                )

            if episode_index not in episode_lengths:
                action_path = dataset_root / str(legacy["action"])
                episode_lengths[episode_index] = pq.ParquetFile(
                    action_path
                ).metadata.num_rows
            frame_count = episode_lengths[episode_index]
            physical_press_frame = int(physical_event["step"])
            sample_key = f"ep{episode_index}:press{press_number}:{button_color}"
            event_window_index = deterministic_index(
                sample_key,
                args.event_index_min,
                args.event_index_max,
                args.window_seed,
            )
            start_frame = physical_press_frame - event_window_index
            end_frame = start_frame + NUM_FRAMES - 1
            if start_frame < 0 or end_frame >= frame_count:
                feasible = [
                    index
                    for index in range(
                        args.event_index_min, args.event_index_max + 1
                    )
                    if 0 <= physical_press_frame - index
                    and physical_press_frame - index + NUM_FRAMES <= frame_count
                ]
                if not feasible:
                    raise ValueError(
                        f"No valid 33-frame physical-event window for {sample_key}."
                    )
                selector = deterministic_index(
                    sample_key, 0, len(feasible) - 1, args.window_seed + 1
                )
                event_window_index = feasible[selector]
                start_frame = physical_press_frame - event_window_index
                end_frame = start_frame + NUM_FRAMES - 1

            videos = list(legacy["video"])
            if args.camera_mode == "main":
                videos = [
                    path
                    for path in videos
                    if "observation.images.image/" in str(path)
                ]
                if len(videos) != 1:
                    raise ValueError(
                        f"Expected one main-camera path for {sample_key}, got {videos}."
                    )

            controls_lamp = bool(physical_event["controls_lamp"])
            lamp_transition_frame = (
                int(legacy["event_frame"])
                if bool(legacy["lamp_toggled"])
                else None
            )
            row = dict(legacy)
            row.update(
                {
                    "video": videos,
                    "event_frame": physical_press_frame,
                    "event_type": "physical_button_trigger",
                    "event_definition": "generator_physical_button_trigger",
                    "event_window_index": event_window_index,
                    "event_window_index_min": args.event_index_min,
                    "event_window_index_max": args.event_index_max,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "requested_core_start_frame": start_frame,
                    "requested_core_end_frame": end_frame,
                    "physical_press_frame": physical_press_frame,
                    "physical_press_depth_m": float(
                        physical_event["button_depth_m"]
                    ),
                    "controls_lamp_annotation": controls_lamp,
                    "lamp_transition_frame_annotation": lamp_transition_frame,
                    "legacy_detector_event_frame": int(legacy["event_frame"]),
                    "legacy_detector_event_type": str(legacy["event_type"]),
                    "window_alignment_seed": args.window_seed,
                    "camera_mode": args.camera_mode,
                    "grouping_name": (
                        "physical_press_event33_jitter11to22_"
                        "random_episode_group20"
                    ),
                    "sample_id": (
                        f"lightswitch:physicalpress33jitter11to22:"
                        f"{args.camera_mode}:g{int(legacy['context_group_id']):02d}:"
                        f"ep{episode_index:06d}:press{press_number}:"
                        f"idx{event_window_index:02d}:"
                        f"raw{start_frame:04d}-{end_frame:04d}"
                    ),
                }
            )
            output_rows.append(row)

    if len(output_rows) != 1600:
        raise ValueError(f"Generated {len(output_rows)} rows; expected 1600.")
    write_jsonl(output_dir / "physical_press_event_group20_train.jsonl", output_rows)
    write_jsonl(output_dir / "physical_press_event_report.jsonl", output_rows)
    csv_fields = [
        "episode_index",
        "causal_class",
        "context_group_id",
        "press_number",
        "button_color",
        "controls_lamp_annotation",
        "physical_press_frame",
        "lamp_transition_frame_annotation",
        "legacy_detector_event_frame",
        "legacy_detector_event_type",
        "event_window_index",
        "start_frame",
        "end_frame",
        "camera_mode",
    ]
    with (output_dir / "physical_press_event_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    for filename in ("action_stats.json", "group20_manifest.json"):
        shutil.copy2(
            args.legacy_metadata_dir / filename, output_dir / filename
        )

    summary = {
        "dataset_base_path": str(dataset_root),
        "generator_metadata": str(generator_path),
        "legacy_metadata_used_only_for_group_and_annotations": str(legacy_path),
        "event_definition": "episodes[].events[].step physical button trigger",
        "forbidden_timing_inputs": [
            "lamp_on",
            "lamp transition",
            "controls_lamp",
            "causal_class",
        ],
        "sample_count": len(output_rows),
        "episode_count": len(episode_lengths),
        "camera_mode": args.camera_mode,
        "window": {
            "num_frames": NUM_FRAMES,
            "frame_stride": 1,
            "event_window_index_min": args.event_index_min,
            "event_window_index_max": args.event_index_max,
            "window_seed": args.window_seed,
        },
        "event_window_index_counts": dict(
            sorted(Counter(row["event_window_index"] for row in output_rows).items())
        ),
        "sample_counts_by_causal_class": dict(
            sorted(Counter(row["causal_class"] for row in output_rows).items())
        ),
        "sample_counts_by_group": dict(
            sorted(Counter(str(row["context_group_id"]) for row in output_rows).items())
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
