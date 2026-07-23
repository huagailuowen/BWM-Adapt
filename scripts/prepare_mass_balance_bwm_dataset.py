#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/"
    "libero_mass_balance_workspace_random_20ratio_15support_300eps_"
    "direct_approach_absolute_eef_lerobot_2026-07-22_hai-machine"
)
DEFAULT_OUTPUT = Path("data/mass_balance_20ratio_stride2_41f_20260722")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
WINDOWS = (
    (0, 80, "approach"),
    (40, 120, "insert_lift"),
    (70, 150, "lift_hold"),
)
PROMPT = (
    "predict the robot and mass-balanced object motion conditioned on the "
    "absolute end-effector action sequence"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def data_path(episode_index: int) -> str:
    return f"data/chunk-000/episode_{episode_index:06d}.parquet"


def video_path(episode_index: int, video_key: str) -> str:
    return f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"


def build_rows(
    source_root: Path,
    episodes: list[dict],
    mass_metadata: list[dict],
) -> list[dict]:
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    mass_by_id = {int(row["episode_index"]): row for row in mass_metadata}
    if set(episode_by_id) != set(mass_by_id):
        raise ValueError("Episode metadata and mass-balance metadata do not match.")

    rows = []
    for episode_index in sorted(episode_by_id):
        episode = episode_by_id[episode_index]
        physical = mass_by_id[episode_index]
        frame_count = int(episode["length"])
        if frame_count != 150:
            raise ValueError(
                f"Episode {episode_index} has {frame_count} frames; expected 150."
            )
        ratio = float(physical["right_to_left_mass_ratio"])
        action_rel = data_path(episode_index)
        videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
        for relative in [action_rel, *videos]:
            if not (source_root / relative).is_file():
                raise FileNotFoundError(source_root / relative)

        for window_index, (start_frame, end_frame, window_name) in enumerate(WINDOWS):
            required_pool = window_index in (1, 2)
            rows.append(
                {
                    **physical,
                    "sample_id": (
                        f"mass_balance:ratio{int(physical['ratio_index']):02d}:"
                        f"ep{episode_index:06d}:frames{start_frame:04d}-{end_frame:04d}"
                    ),
                    "episode_index": episode_index,
                    "source_dataset": source_root.name,
                    "source_split": "train_all_300eps",
                    "friction_mu": ratio,
                    "mass_ratio": ratio,
                    "right_to_left_mass_ratio": ratio,
                    "action_id": int(physical["support_bin_index"]),
                    "window_index": window_index,
                    "window_name": window_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "length": 41,
                    "valid_frames": 40,
                    "raw_frame_span": 80,
                    "total_frames": frame_count,
                    "frame_stride": 2,
                    "pad_short": True,
                    "sampling_required_pool": required_pool,
                    "sampling_required_count": 4,
                    "video": videos,
                    "action": action_rel,
                    "prompt": PROMPT,
                    "task": "mass_balance_absolute_eef_action_behavior_cloning",
                    "episode_tasks": list(episode.get("tasks", [])),
                }
            )
    return rows


def compute_action_stats(source_root: Path, episodes: list[dict]) -> dict:
    arrays = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = pq.read_table(source_root / data_path(episode_index), columns=["action"])
        arrays.append(np.asarray(table.to_pydict()["action"], dtype=np.float32))
    actions = np.concatenate(arrays, axis=0)
    minimum = actions.min(axis=0)
    maximum = actions.max(axis=0)
    stat = {
        "shape": [int(actions.shape[1])],
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "p01": minimum.tolist(),
        "p99": maximum.tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
    }
    return {
        "action_pose": stat,
        "eef_delta": stat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    mass_metadata = read_jsonl(
        source_root / "meta" / "mass_balance_episode_metadata.jsonl"
    )
    if len(episodes) != 300 or len(mass_metadata) != 300:
        raise ValueError(
            f"Expected 300 episodes, found {len(episodes)} episode rows and "
            f"{len(mass_metadata)} mass rows."
        )

    rows = build_rows(source_root, episodes, mass_metadata)
    ratio_counts = Counter(float(row["mass_ratio"]) for row in rows)
    if len(ratio_counts) != 20 or set(ratio_counts.values()) != {45}:
        raise ValueError(f"Expected 20 ratios with 45 windows each: {ratio_counts}")

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_stats.json").write_text(
        json.dumps(compute_action_stats(source_root, episodes), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "episodes": len(episodes),
        "environments": len(ratio_counts),
        "train_samples": len(rows),
        "raw_windows": [[start, end] for start, end, _ in WINDOWS],
        "raw_window_frames": 80,
        "frame_stride": 2,
        "unique_sampled_frames": 40,
        "model_frames": 41,
        "temporal_padding": "repeat each window's final sampled frame once",
        "required_late_windows_per_six": 4,
        "action_semantics": "absolute EEF setpoint synchronized with video stride",
        "video_keys": list(VIDEO_KEYS),
        "mass_ratios": sorted(ratio_counts),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
