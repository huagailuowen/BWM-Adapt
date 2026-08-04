#!/usr/bin/env python3
"""Build exact lamp-transition/button-contact windows for LightSwitch Stage1.

Each episode contains eight button-press task segments.  A controlled press is
centered on the exact frame where ``lamp_on`` changes.  A non-controlling press
is centered on maximum button depth and is retained as a negative causal
example.  The 31-frame requested core (15 before, event, 15 after) is padded by
one boundary frame on each side to produce the Wan-compatible length 33.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAUSAL_CLASSES = ("neither", "red_only", "blue_only", "both")
VIDEO_KEYS = ("observation.images.image", "observation.images.wrist_image")
PRESS_RE = re.compile(r"^random_press_(\d+)_(red|blue)$")
PROMPT = (
    "predict the robot and lamp response while the causal relationship between "
    "the red and blue buttons and the lamp is hidden"
)
NUM_FRAMES = 33
CORE_RADIUS = 15
BOUNDARY_RADIUS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def relative_data_path(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def relative_video_path(episode_index: int, key: str) -> str:
    return (
        f"videos/chunk-{episode_index // 1000:03d}/{key}/"
        f"episode_{episode_index:06d}.mp4"
    )


def task_map(dataset_root: Path) -> dict[int, str]:
    return {
        int(row["task_index"]): str(row["task"])
        for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    }


def segment_bounds(values: list[int]) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            segments.append((start, index - 1, int(values[start])))
            start = index
    return segments


def centered_window(event_frame: int, frame_count: int) -> tuple[int, int, int]:
    if frame_count < NUM_FRAMES:
        raise ValueError(f"Episode has only {frame_count} frames; need {NUM_FRAMES}")
    start = min(max(event_frame - BOUNDARY_RADIUS, 0), frame_count - NUM_FRAMES)
    end = start + NUM_FRAMES - 1
    return start, end, event_frame - start


def classify(events: list[dict]) -> tuple[str, bool, bool]:
    red = any(event["button_color"] == "red" and event["lamp_toggled"] for event in events)
    blue = any(event["button_color"] == "blue" and event["lamp_toggled"] for event in events)
    if red and blue:
        return "both", red, blue
    if red:
        return "red_only", red, blue
    if blue:
        return "blue_only", red, blue
    return "neither", red, blue


def scan_episode(
    dataset_root: Path,
    episode_index: int,
    tasks: dict[int, str],
) -> tuple[list[dict], np.ndarray]:
    rel_data = relative_data_path(episode_index)
    table = pq.read_table(
        dataset_root / rel_data,
        columns=["frame_index", "task_index", "observation.button_lamp_state", "action"],
    )
    frames = [int(value) for value in table["frame_index"].to_pylist()]
    task_indices = [int(value) for value in table["task_index"].to_pylist()]
    states = np.asarray(table["observation.button_lamp_state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    lamp = np.rint(states[:, 8]).astype(np.int8)
    events: list[dict] = []

    for segment_start, segment_end, task_index in segment_bounds(task_indices):
        task = tasks[task_index]
        match = PRESS_RE.match(task)
        if match is None:
            continue
        press_number = int(match.group(1))
        color = match.group(2)
        transition_frames = [
            index
            for index in range(max(1, segment_start), segment_end + 1)
            if lamp[index] != lamp[index - 1]
        ]
        if len(transition_frames) > 1:
            raise ValueError(
                f"episode {episode_index} task {task} has multiple lamp transitions: "
                f"{transition_frames}"
            )
        toggled = bool(transition_frames)
        if toggled:
            event_frame = transition_frames[0]
            lamp_before = int(lamp[event_frame - 1])
            lamp_after = int(lamp[event_frame])
            event_type = "lamp_toggle"
        else:
            depth_column = 6 if color == "red" else 7
            local_peak = int(np.argmax(states[segment_start : segment_end + 1, depth_column]))
            event_frame = segment_start + local_peak
            lamp_before = int(lamp[max(segment_start, event_frame - 1)])
            lamp_after = int(lamp[min(segment_end, event_frame + 1)])
            event_type = "button_press_no_toggle"

        window_start, window_end, event_window_index = centered_window(event_frame, len(frames))
        events.append(
            {
                "episode_index": episode_index,
                "press_number": press_number,
                "action_id": press_number - 1,
                "task_index": task_index,
                "task_name": task,
                "button_color": color,
                "event_type": event_type,
                "lamp_toggled": toggled,
                "lamp_before": lamp_before,
                "lamp_after": lamp_after,
                "segment_start_frame": segment_start,
                "segment_end_frame": segment_end,
                "event_frame": event_frame,
                "event_window_index": event_window_index,
                "requested_core_start_frame": event_frame - CORE_RADIUS,
                "requested_core_end_frame": event_frame + CORE_RADIUS,
                "start_frame": window_start,
                "end_frame": window_end,
                "length": NUM_FRAMES,
                "frame_stride": 1,
                "raw_frame_span": NUM_FRAMES - 1,
                "source_fps": 30,
                "target_fps": 30,
                "action": rel_data,
                "video": [relative_video_path(episode_index, key) for key in VIDEO_KEYS],
            }
        )

    events.sort(key=lambda event: event["event_frame"])
    if len(events) != 8:
        raise ValueError(f"episode {episode_index} has {len(events)} press events; expected 8")
    return events, actions


def assign_random_groups(
    events_by_episode: dict[int, list[dict]],
    seed: int,
) -> tuple[dict[int, int], list[dict]]:
    episodes_by_class: dict[str, list[int]] = defaultdict(list)
    for episode_index, events in events_by_episode.items():
        causal_class, _, _ = classify(events)
        episodes_by_class[causal_class].append(episode_index)

    assignments: dict[int, int] = {}
    groups: list[dict] = []
    for class_index, causal_class in enumerate(CAUSAL_CLASSES):
        episode_indices = sorted(episodes_by_class[causal_class])
        if len(episode_indices) != 50:
            raise ValueError(
                f"{causal_class} has {len(episode_indices)} episodes; expected exactly 50"
            )
        random.Random(seed + 100003 * class_index).shuffle(episode_indices)
        for local_group in range(5):
            group_id = class_index * 5 + local_group
            members = episode_indices[local_group * 10 : (local_group + 1) * 10]
            for episode_index in members:
                assignments[episode_index] = group_id
            groups.append(
                {
                    "context_group_id": group_id,
                    "friction_mu": float(group_id),
                    "causal_class": causal_class,
                    "causal_class_index": class_index,
                    "local_group_index": local_group,
                    "episode_indices": members,
                    "episode_count": len(members),
                }
            )
    return assignments, groups


def action_stats(actions: list[np.ndarray]) -> dict:
    values = np.concatenate(actions, axis=0).astype(np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "count": int(values.shape[0]),
        "frame_stride": 1,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_base_path.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = task_map(dataset_root)
    events_by_episode: dict[int, list[dict]] = {}
    all_actions: list[np.ndarray] = []
    for episode in episodes_meta:
        episode_index = int(episode["episode_index"])
        events, actions = scan_episode(dataset_root, episode_index, tasks)
        events_by_episode[episode_index] = events
        all_actions.append(actions)

    assignments, groups = assign_random_groups(events_by_episode, args.seed)
    train_rows: list[dict] = []
    report_rows: list[dict] = []
    for episode_index in sorted(events_by_episode):
        events = events_by_episode[episode_index]
        causal_class, red_controls, blue_controls = classify(events)
        group_id = assignments[episode_index]
        for temporal_event_index, event in enumerate(events):
            common = {
                **event,
                "temporal_event_index": temporal_event_index,
                "causal_class": causal_class,
                "causal_class_index": CAUSAL_CLASSES.index(causal_class),
                "red_controls_lamp": red_controls,
                "blue_controls_lamp": blue_controls,
                "context_group_id": group_id,
                "friction_mu": float(group_id),
                "grouping_name": "event_centered_random_episode_group20",
            }
            report_rows.append(common)
            train_rows.append(
                {
                    **common,
                    "sample_id": (
                        f"lightswitch:event33:g{group_id:02d}:ep{episode_index:06d}:"
                        f"press{event['press_number']}:raw{event['start_frame']:04d}-"
                        f"{event['end_frame']:04d}"
                    ),
                    "task": "hidden_button_lamp_causal_dynamics",
                    "prompt": PROMPT,
                    "covered_button_colors": [event["button_color"]],
                    "covered_event_count": 1,
                }
            )

    write_jsonl(output_dir / "event_centered_group20_train.jsonl", train_rows)
    write_jsonl(output_dir / "event_report.jsonl", report_rows)
    csv_fields = [
        "episode_index",
        "causal_class",
        "context_group_id",
        "temporal_event_index",
        "press_number",
        "button_color",
        "task_name",
        "event_type",
        "lamp_toggled",
        "lamp_before",
        "lamp_after",
        "segment_start_frame",
        "segment_end_frame",
        "event_frame",
        "requested_core_start_frame",
        "requested_core_end_frame",
        "start_frame",
        "end_frame",
        "event_window_index",
    ]
    with (output_dir / "event_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)

    manifest = {
        "seed": args.seed,
        "grouping": "seeded random episode shuffle, five disjoint groups of ten per environment",
        "causal_class_order": list(CAUSAL_CLASSES),
        "groups": groups,
        "curriculum": {
            "initial_groups": 4,
            "groups_added_per_wave": 4,
            "waves": 3,
            "active_groups_after_training": 12,
            "held_out_groups": 8,
            "strata": 4,
            "requirement": "one group from each causal environment per wave",
        },
    }
    write_json(output_dir / "group20_manifest.json", manifest)
    write_json(output_dir / "action_stats.json", action_stats(all_actions))

    event_types = Counter(row["event_type"] for row in report_rows)
    class_counts = Counter(row["causal_class"] for row in report_rows)
    group_counts = Counter(row["context_group_id"] for row in report_rows)
    summary = {
        "dataset_base_path": str(dataset_root),
        "seed": args.seed,
        "episode_count": len(events_by_episode),
        "sample_count": len(train_rows),
        "event_type_counts": dict(sorted(event_types.items())),
        "sample_counts_by_causal_class": dict(sorted(class_counts.items())),
        "sample_counts_by_group": {str(key): group_counts[key] for key in sorted(group_counts)},
        "window": {
            "num_frames": NUM_FRAMES,
            "frame_stride": 1,
            "source_fps": 30,
            "target_fps": 30,
            "requested_core": "15 frames before + event frame + 15 frames after",
            "wan_boundary_context": "one additional frame on each side",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
