#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/ball_friction"
)
DEFAULT_OUTPUT = Path(
    "data/real_ball_friction_7env_60f_1prefix9impact_1view_20260815"
)
ENVIRONMENTS = (
    ("ball-0_lerobot", 0),
    ("ball-1_lerobot", 1),
    ("ball-2_lerobot", 2),
    ("ball-3_lerobot", 3),
    ("ball-4_lerobot", 4),
    ("ball-7_lerobot", 7),
    ("ball-9_lerobot", 9),
)
VIDEO_KEYS = (
    "observation.images.image",
)
PHYSICAL_CHUNK_FRAMES = 60
MODEL_FRAMES = 61
PREFIX_WINDOWS_PER_EPISODE = 1
IMPACT_WINDOWS_PER_EPISODE = 9
PRE_RIGHT_CONTEXT_FRAMES = 8
PROMPT = "predict the robot swing, ball impact, and subsequent ball motion"
ACCEPTED_DETECTION_STATUSES = frozenset(
    {
        "unique_match",
        "model_fit_unique_match",
        "model_fit_speed_out_of_tolerance",
    }
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def relative_data_path(environment: str, episode_index: int) -> str:
    return f"{environment}/data/chunk-000/episode_{episode_index:06d}.parquet"


def relative_video_path(environment: str, episode_index: int, video_key: str) -> str:
    return (
        f"{environment}/videos/chunk-000/{video_key}/"
        f"episode_{episode_index:06d}.mp4"
    )


def evenly_spaced_integer_starts(low: int, high: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if low > high:
        raise ValueError(f"Invalid start range [{low}, {high}].")
    if high - low + 1 < count:
        raise ValueError(
            f"Start range [{low}, {high}] has fewer than {count} distinct frames."
        )
    if count == 1:
        return [low]
    starts = [round(low + index * (high - low) / (count - 1)) for index in range(count)]
    if len(set(starts)) != count:
        raise RuntimeError(f"Non-unique evenly spaced starts: {starts}")
    return starts


def build_rows(source_root: Path) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    environment_summary: dict[str, dict] = {}
    for environment_index, (environment, ball_id) in enumerate(ENVIRONMENTS):
        environment_root = source_root / environment
        episodes = {
            int(row["episode_index"]): row
            for row in read_jsonl(environment_root / "meta" / "episodes.jsonl")
        }
        annotations = read_jsonl(
            environment_root / "meta" / "ball_friction_skill_annotations.jsonl"
        )
        if len(annotations) != len(episodes):
            raise ValueError(
                f"{environment}: {len(annotations)} annotations for {len(episodes)} episodes."
            )

        gear_episode_counts: Counter[int] = Counter()
        sampling_counts: Counter[str] = Counter()
        for annotation in sorted(annotations, key=lambda item: int(item["episode_index"])):
            episode_index = int(annotation["episode_index"])
            if episode_index not in episodes:
                raise ValueError(f"{environment}: missing episode metadata {episode_index}.")
            detection_status = annotation.get("detection_status")
            if detection_status not in ACCEPTED_DETECTION_STATUSES:
                raise ValueError(
                    f"{environment} episode {episode_index}: unsupported skill annotation "
                    f"status {detection_status!r}; expected one of "
                    f"{sorted(ACCEPTED_DETECTION_STATUSES)}."
                )

            episode = episodes[episode_index]
            total_frames = int(episode["length"])
            skill_gear = int(annotation["skill_gear"])
            if not 1 <= skill_gear <= 10:
                raise ValueError(f"Invalid skill gear {skill_gear} in {environment}.")
            segments = {segment["label"]: segment for segment in annotation["segments"]}
            right = segments["right_motion"]
            right_start = int(right["start_frame_index"])
            right_end = int(right["end_frame_index_inclusive"])
            right_frames = int(right["frame_count"])
            required_right_frames = max(1, math.ceil(right_frames / 3))

            earliest_impact_start = max(0, right_start - PRE_RIGHT_CONTEXT_FRAMES)
            latest_impact_start = right_end - required_right_frames + 1
            impact_starts = evenly_spaced_integer_starts(
                earliest_impact_start,
                latest_impact_start,
                IMPACT_WINDOWS_PER_EPISODE,
            )
            candidates = [("prefix", 0)] + [
                ("impact", start) for start in impact_starts
            ]

            action_rel = relative_data_path(environment, episode_index)
            videos = [
                relative_video_path(environment, episode_index, key)
                for key in VIDEO_KEYS
            ]
            for relative in [action_rel, *videos]:
                if not (source_root / relative).is_file():
                    raise FileNotFoundError(source_root / relative)

            global_episode_index = environment_index * 100000 + episode_index
            for candidate_index, (sampling_kind, start_frame) in enumerate(candidates):
                end_frame = start_frame + PHYSICAL_CHUNK_FRAMES - 1
                right_overlap = max(
                    0,
                    min(end_frame, right_end) - max(start_frame, right_start) + 1,
                )
                if sampling_kind == "impact" and right_overlap < required_right_frames:
                    raise RuntimeError(
                        f"{environment} episode {episode_index}: impact window "
                        f"{start_frame}-{end_frame} contains {right_overlap}/{right_frames} "
                        "right-motion frames."
                    )
                valid_physical_frames = max(
                    1,
                    min(PHYSICAL_CHUNK_FRAMES, total_frames - start_frame),
                )
                rows.append(
                    {
                        "sample_id": (
                            f"real_ball_friction:{ball_id}:gear{skill_gear}:"
                            f"ep{episode_index:06d}:{sampling_kind}{candidate_index}:"
                            f"frames{start_frame:04d}-{end_frame:04d}"
                        ),
                        "environment": environment,
                        "environment_index": environment_index,
                        "ball_id": ball_id,
                        "friction_mu": float(environment_index),
                        "episode_index": global_episode_index,
                        "source_episode_index": episode_index,
                        "source_directory": annotation.get("source_directory"),
                        "action_id": skill_gear,
                        "skill_gear": skill_gear,
                        "skill_speed_scale": float(annotation["skill_speed_scale"]),
                        "expected_commanded_angular_speed_deg_s": float(
                            annotation["expected_commanded_angular_speed_deg_s"]
                        ),
                        "sampling_kind": sampling_kind,
                        "candidate_index": candidate_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "length": MODEL_FRAMES,
                        "physical_chunk_frames": PHYSICAL_CHUNK_FRAMES,
                        "valid_physical_frames": valid_physical_frames,
                        "total_frames": total_frames,
                        "frame_stride": 1,
                        "pad_short": True,
                        "alignment_padding": "repeat final frame from 60 to 61",
                        "right_motion_start": right_start,
                        "right_motion_end": right_end,
                        "right_motion_frames": right_frames,
                        "required_right_motion_frames": required_right_frames,
                        "window_right_motion_frames": right_overlap,
                        "video": videos,
                        "action": action_rel,
                        "action_semantics": "joint_state_action",
                        "state_action_layout": "observation.state[:7] || action[:7]",
                        "prompt": PROMPT,
                        "task": "real_ball_friction_impact_world_modeling",
                        "episode_tasks": list(episode.get("tasks", [])),
                    }
                )
                sampling_counts[sampling_kind] += 1
            gear_episode_counts[skill_gear] += 1

        if len(gear_episode_counts) < 6:
            raise ValueError(
                f"{environment}: expected at least 6 distinct skill levels, "
                f"got {sorted(gear_episode_counts)}."
            )
        environment_summary[environment] = {
            "ball_id": ball_id,
            "context_group_value": float(environment_index),
            "episodes": len(episodes),
            "gear_episode_counts": {
                str(key): value for key, value in sorted(gear_episode_counts.items())
            },
            "candidate_windows": sum(sampling_counts.values()),
            "sampling_counts": dict(sorted(sampling_counts.items())),
        }
    return rows, environment_summary


def compute_state_action_stats(source_root: Path) -> dict:
    arrays = []
    source_frames = 0
    for environment, _ in ENVIRONMENTS:
        environment_root = source_root / environment
        episodes = read_jsonl(environment_root / "meta" / "episodes.jsonl")
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            parquet_path = source_root / relative_data_path(environment, episode_index)
            table = pq.read_table(
                parquet_path,
                columns=["observation.state", "action"],
            )
            payload = table.to_pydict()
            state = np.asarray(payload["observation.state"], dtype=np.float32)
            action = np.asarray(payload["action"], dtype=np.float32)
            if state.ndim != 2 or action.ndim != 2 or state.shape[1] < 7 or action.shape[1] < 7:
                raise ValueError(
                    f"Expected state/action widths >=7 in {parquet_path}; "
                    f"got state={state.shape}, action={action.shape}."
                )
            if state.shape[0] != action.shape[0]:
                raise ValueError(
                    f"State/action length mismatch in {parquet_path}: "
                    f"{state.shape[0]} vs {action.shape[0]}."
                )
            arrays.append(np.concatenate([state[:, :7], action[:, :7]], axis=1))
            source_frames += state.shape[0]
    values = np.concatenate(arrays, axis=0)
    stat = {
        "shape": [14],
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "p01": np.percentile(values, 1.0, axis=0).tolist(),
        "p99": np.percentile(values, 99.0, axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }
    return {"joint_state_action": stat, "source_frames": source_frames}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    rows, environment_summary = build_rows(source_root)
    group_counts = Counter(float(row["friction_mu"]) for row in rows)
    action_counts = Counter(int(row["action_id"]) for row in rows)
    sampling_counts = Counter(str(row["sampling_kind"]) for row in rows)
    if len(group_counts) != len(ENVIRONMENTS):
        raise ValueError(f"Expected {len(ENVIRONMENTS)} groups: {group_counts}")
    if set(action_counts) != set(range(1, 11)):
        raise ValueError(f"Expected action levels 1-10: {action_counts}")
    expected_prefix = sum(item["episodes"] for item in environment_summary.values())
    if sampling_counts != {
        "prefix": expected_prefix,
        "impact": expected_prefix * IMPACT_WINDOWS_PER_EPISODE,
    }:
        raise ValueError(f"Unexpected 10/90 sampling counts: {sampling_counts}")

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_state_action_stats(source_root)
    (output_dir / "action_stats.json").write_text(
        json.dumps({"joint_state_action": stats["joint_state_action"]}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "environments": environment_summary,
        "train_samples": len(rows),
        "group_candidate_counts": {
            str(key): value for key, value in sorted(group_counts.items())
        },
        "action_level_candidate_counts": {
            str(key): value for key, value in sorted(action_counts.items())
        },
        "sampling_counts": dict(sorted(sampling_counts.items())),
        "prefix_probability_by_candidate_count": (
            sampling_counts["prefix"] / len(rows)
        ),
        "source_frames": stats["source_frames"],
        "physical_chunk_frames": PHYSICAL_CHUNK_FRAMES,
        "model_frames": MODEL_FRAMES,
        "temporal_padding": (
            "repeat source final frame for short clips, then repeat the 60th frame once "
            "to satisfy Wan 4k+1 temporal alignment"
        ),
        "impact_window_policy": (
            "nine starts uniformly spanning 8 frames before right_motion through the "
            "latest start retaining at least one third of right_motion"
        ),
        "action_sampling_policy": (
            "sample distinct skill levels first, then sample one episode/window within "
            "each selected (environment, skill level) cell"
        ),
        "state_action_layout": "observation.state[:7] || action[:7]",
        "resize_policy": "per-view aspect-preserving letterbox to 224x224 at load time",
        "video_keys": list(VIDEO_KEYS),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
