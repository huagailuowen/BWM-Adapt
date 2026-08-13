#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/stick_balance"
)
DEFAULT_OUTPUT = Path(
    "data/real_stick_balance_8env_120raw_stride3_1general4lift_20260810"
)

# Ordered from the largest left offset to the largest right offset. With the
# grouped-context seed below, the first four curriculum groups span this range.
ENVIRONMENTS = (
    ("stick-L3left-R0_lerobot", -3),
    ("stick-L2left-R0_lerobot", -2),
    ("stick-L1left-R0_lerobot", -1),
    ("stick-L0-R0_lerobot", 0),
    ("stick-L0-R1right_lerobot", 1),
    ("stick-L0-R2right_lerobot", 2),
    ("stick-L0-R3right_lerobot", 3),
    ("stick-L0-R4right_lerobot", 4),
)
VIDEO_KEY = "observation.images.image"
NUM_FRAMES = 41
FRAME_STRIDE = 3
RAW_SPAN = (NUM_FRAMES - 1) * FRAME_STRIDE
GENERAL_WINDOWS_PER_EPISODE = 1
LIFT_WINDOWS_PER_EPISODE = 4
PROMPT = "predict the robot reaching, lifting, balancing, and lowering the stick"


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


def relative_video_path(environment: str, episode_index: int) -> str:
    return (
        f"{environment}/videos/chunk-000/{VIDEO_KEY}/"
        f"episode_{episode_index:06d}.mp4"
    )


def evenly_spaced_starts(low: int, high: int, count: int) -> list[int]:
    if low > high:
        raise ValueError(f"Invalid start range [{low}, {high}].")
    if count <= 1:
        return [low]
    return [round(low + index * (high - low) / (count - 1)) for index in range(count)]


def deterministic_general_start(
    max_start: int, environment_index: int, episode_index: int
) -> int:
    if max_start <= 0:
        return 0
    rng = np.random.default_rng(20260810 + environment_index * 10000 + episode_index)
    return int(rng.integers(0, max_start + 1))


def sampled_overlap(start: int, segment_start: int, segment_end_exclusive: int) -> int:
    return sum(
        segment_start <= start + index * FRAME_STRIDE < segment_end_exclusive
        for index in range(NUM_FRAMES)
    )


def build_rows(source_root: Path) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    environment_summary: dict[str, dict] = {}

    for environment_index, (environment, balance_offset) in enumerate(ENVIRONMENTS):
        environment_root = source_root / environment
        episodes = {
            int(row["episode_index"]): row
            for row in read_jsonl(environment_root / "meta" / "episodes.jsonl")
        }
        annotations = {
            int(row["episode_index"]): row
            for row in read_jsonl(environment_root / "meta" / "action_segments.jsonl")
        }
        if set(episodes) != set(annotations):
            raise ValueError(
                f"{environment}: episode/segment IDs differ: "
                f"episodes={sorted(episodes)}, annotations={sorted(annotations)}"
            )

        sampling_counts: Counter[str] = Counter()
        lift_starts: list[int] = []
        episode_lengths: list[int] = []
        minimum_lift_samples = NUM_FRAMES

        for episode_index in sorted(episodes):
            episode = episodes[episode_index]
            total_frames = int(episode["length"])
            episode_lengths.append(total_frames)
            lift_segments = [
                segment
                for segment in annotations[episode_index]["segments"]
                if segment["label"] == "lift"
            ]
            if len(lift_segments) != 1:
                raise ValueError(
                    f"{environment} episode {episode_index}: expected one lift segment, "
                    f"got {len(lift_segments)}."
                )
            lift = lift_segments[0]
            lift_start = int(lift["start_frame"])
            lift_end = int(lift["end_frame_exclusive"])
            if not 0 <= lift_start < lift_end <= total_frames:
                raise ValueError(
                    f"{environment} episode {episode_index}: invalid lift interval "
                    f"[{lift_start}, {lift_end}) for {total_frames} frames."
                )
            if lift_end - lift_start > RAW_SPAN + 1:
                raise ValueError(
                    f"{environment} episode {episode_index}: lift length "
                    f"{lift_end - lift_start} exceeds the {RAW_SPAN + 1}-frame window."
                )
            lift_starts.append(lift_start)

            # A 41-frame stride-3 sequence spans 120 raw-frame intervals. Short
            # episodes start at zero and are padded by repeating their final frame.
            max_start = max(0, total_frames - 1 - RAW_SPAN)
            full_lift_low = max(0, lift_end - 1 - RAW_SPAN)
            full_lift_high = min(lift_start, max_start)
            if full_lift_low > full_lift_high:
                raise RuntimeError(
                    f"{environment} episode {episode_index}: cannot cover lift interval "
                    f"[{lift_start}, {lift_end}) with raw span {RAW_SPAN}."
                )
            lift_window_starts = evenly_spaced_starts(
                full_lift_low, full_lift_high, LIFT_WINDOWS_PER_EPISODE
            )
            general_start = deterministic_general_start(
                max_start, environment_index, episode_index
            )
            candidates = [("general", general_start)] + [
                ("lift", start) for start in lift_window_starts
            ]

            action_rel = relative_data_path(environment, episode_index)
            video_rel = relative_video_path(environment, episode_index)
            for relative in (action_rel, video_rel):
                if not (source_root / relative).is_file():
                    raise FileNotFoundError(source_root / relative)

            global_episode_index = environment_index * 100000 + episode_index
            for candidate_index, (sampling_kind, start_frame) in enumerate(candidates):
                end_frame = start_frame + RAW_SPAN
                lift_samples = sampled_overlap(start_frame, lift_start, lift_end)
                if sampling_kind == "lift":
                    minimum_lift_samples = min(minimum_lift_samples, lift_samples)
                valid_sampled_frames = min(
                    NUM_FRAMES,
                    max(1, (total_frames - 1 - start_frame) // FRAME_STRIDE + 1),
                )
                rows.append(
                    {
                        "sample_id": (
                            f"real_stick_balance:offset{balance_offset:+d}:"
                            f"ep{episode_index:06d}:{sampling_kind}{candidate_index}:"
                            f"raw{start_frame:04d}-{end_frame:04d}:s{FRAME_STRIDE}"
                        ),
                        "environment": environment,
                        "environment_index": environment_index,
                        "balance_offset": balance_offset,
                        "friction_mu": float(environment_index),
                        "episode_index": global_episode_index,
                        "source_episode_index": episode_index,
                        # Retained for traceability. Training uses independent-window
                        # sampling, so no cross-environment episode-ID intersection is
                        # imposed.
                        "action_id": episode_index,
                        "sampling_kind": sampling_kind,
                        "candidate_index": candidate_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "length": NUM_FRAMES,
                        "raw_frame_span": RAW_SPAN,
                        "frame_stride": FRAME_STRIDE,
                        "source_fps": 20,
                        "target_fps": 20.0 / FRAME_STRIDE,
                        "total_frames": total_frames,
                        "valid_sampled_frames": valid_sampled_frames,
                        "pad_short": True,
                        "alignment_padding": "repeat final source frame when needed",
                        "lift_start": lift_start,
                        "lift_end_exclusive": lift_end,
                        "lift_raw_frames": lift_end - lift_start,
                        "window_lift_sampled_frames": lift_samples,
                        "video": [video_rel],
                        "action": action_rel,
                        "action_semantics": "joint_state_action",
                        "state_action_layout": "observation.state[:7] || action[:7]",
                        "camera_mode": "observer_only",
                        "prompt": PROMPT,
                        "task": "real_stick_balance_world_modeling",
                        "episode_tasks": list(episode.get("tasks", [])),
                    }
                )
                sampling_counts[sampling_kind] += 1

        environment_summary[environment] = {
            "balance_offset": balance_offset,
            "context_group_value": float(environment_index),
            "episodes": len(episodes),
            "episode_frames": {
                "min": min(episode_lengths),
                "max": max(episode_lengths),
                "mean": float(np.mean(episode_lengths)),
            },
            "lift_start_frames": {
                "min": min(lift_starts),
                "max": max(lift_starts),
                "mean": float(np.mean(lift_starts)),
            },
            "candidate_windows": sum(sampling_counts.values()),
            "sampling_counts": dict(sorted(sampling_counts.items())),
            "minimum_lift_samples_in_lift_window": minimum_lift_samples,
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
        "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
    }
    return {"joint_state_action": stat, "source_frames": source_frames}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if any("old" in environment.lower() for environment, _ in ENVIRONMENTS):
        raise ValueError("Old environments must not be included.")
    rows, environment_summary = build_rows(source_root)
    group_counts = Counter(float(row["friction_mu"]) for row in rows)
    action_sets = {
        environment: {
            int(row["action_id"])
            for row in rows
            if row["environment"] == environment
        }
        for environment, _ in ENVIRONMENTS
    }
    common_actions = sorted(set.intersection(*action_sets.values()))
    sampling_counts = Counter(str(row["sampling_kind"]) for row in rows)
    total_episodes = sum(item["episodes"] for item in environment_summary.values())
    expected_sampling = {
        "general": total_episodes * GENERAL_WINDOWS_PER_EPISODE,
        "lift": total_episodes * LIFT_WINDOWS_PER_EPISODE,
    }
    if len(group_counts) != len(ENVIRONMENTS):
        raise ValueError(f"Expected {len(ENVIRONMENTS)} context groups: {group_counts}")
    if sampling_counts != expected_sampling:
        raise ValueError(f"Unexpected 20/80 sampling counts: {sampling_counts}")
    if len(common_actions) < 6:
        raise ValueError(f"Need at least six common action slots: {common_actions}")

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    stats = compute_state_action_stats(source_root)
    (output_dir / "action_stats.json").write_text(
        json.dumps(
            {"joint_state_action": stats["joint_state_action"]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "excluded_environment_pattern": "*Old*",
        "environments": environment_summary,
        "train_samples": len(rows),
        "source_episodes": total_episodes,
        "source_frames": stats["source_frames"],
        "group_candidate_counts": {
            str(key): value for key, value in sorted(group_counts.items())
        },
        "common_action_ids": common_actions,
        "common_action_count": len(common_actions),
        "sampling_counts": dict(sorted(sampling_counts.items())),
        "lift_candidate_probability": sampling_counts["lift"] / len(rows),
        "window": {
            "model_frames": NUM_FRAMES,
            "frame_stride": FRAME_STRIDE,
            "raw_frame_span": RAW_SPAN,
            "source_fps": 20,
            "target_fps": 20.0 / FRAME_STRIDE,
            "padding": "repeat final source frame for short episodes",
        },
        "lift_window_policy": (
            "four evenly spaced starts from the feasible interval that contains the "
            "complete annotated lift segment"
        ),
        "general_window_policy": (
            "one deterministically seeded uniform start over the complete episode"
        ),
        "action_sampling_policy": (
            "sample six independent windows per selected environment; every episode has "
            "exactly five candidates, so episodes are uniform and lift/general probability "
            "is exactly 4/5 versus 1/5"
        ),
        "state_action_layout": "observation.state[:7] || action[:7]",
        "resize_policy": "observer-view aspect-preserving letterbox to 224x224",
        "video_keys": [VIDEO_KEY],
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
