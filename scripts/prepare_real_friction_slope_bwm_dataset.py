#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/friction_slope"
)
DEFAULT_OUTPUT = Path(
    "data/real_friction_slope_2env_stride2_41f_allwindows_20260801"
)
ENVIRONMENTS = (
    ("sandpaper-box-120_lerobot", 120),
    ("sandpaper-box-240_lerobot", 240),
)
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
RAW_WINDOW_FRAMES = 80
FRAME_STRIDE = 2
UNIQUE_MODEL_FRAMES = 40
MODEL_FRAMES = 41
PROMPT = "predict the robot and box motion on the inclined friction surface"


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


def build_rows(source_root: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    environment_summary = {}
    for environment_index, (environment, grit) in enumerate(ENVIRONMENTS):
        environment_root = source_root / environment
        episodes = read_jsonl(environment_root / "meta" / "episodes.jsonl")
        window_count = 0
        skipped_short = []
        for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
            source_episode_index = int(episode["episode_index"])
            frame_count = int(episode["length"])
            if frame_count < RAW_WINDOW_FRAMES:
                skipped_short.append(source_episode_index)
                continue
            action_rel = relative_data_path(environment, source_episode_index)
            videos = [
                relative_video_path(environment, source_episode_index, key)
                for key in VIDEO_KEYS
            ]
            for relative in [action_rel, *videos]:
                if not (source_root / relative).is_file():
                    raise FileNotFoundError(source_root / relative)

            global_episode_index = environment_index * 100000 + source_episode_index
            for start_frame in range(frame_count - RAW_WINDOW_FRAMES + 1):
                end_frame = start_frame + RAW_WINDOW_FRAMES - 1
                rows.append(
                    {
                        "sample_id": (
                            f"real_friction_slope:{grit}:ep{source_episode_index:06d}:"
                            f"frames{start_frame:04d}-{end_frame:04d}"
                        ),
                        "environment": environment,
                        "environment_index": environment_index,
                        "sandpaper_grit": grit,
                        "friction_mu": float(grit),
                        "episode_index": global_episode_index,
                        "source_episode_index": source_episode_index,
                        "action_id": source_episode_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "length": MODEL_FRAMES,
                        "valid_frames": UNIQUE_MODEL_FRAMES,
                        "raw_frame_span": RAW_WINDOW_FRAMES,
                        "total_frames": frame_count,
                        "frame_stride": FRAME_STRIDE,
                        "pad_short": True,
                        "video": videos,
                        "action": action_rel,
                        "action_semantics": "joint_state_action",
                        "state_action_layout": "observation.state[:7] || action[:7]",
                        "prompt": PROMPT,
                        "task": "real_friction_slope_joint_transition_world_modeling",
                        "episode_tasks": list(episode.get("tasks", [])),
                    }
                )
                window_count += 1
        environment_summary[environment] = {
            "grit": grit,
            "episodes": len(episodes),
            "windows": window_count,
            "skipped_short_episodes": skipped_short,
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
    return {
        "joint_state_action": stat,
        "source_frames": source_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    rows, environment_summary = build_rows(source_root)
    group_counts = Counter(float(row["friction_mu"]) for row in rows)
    if len(group_counts) != len(ENVIRONMENTS):
        raise ValueError(f"Expected {len(ENVIRONMENTS)} environment groups: {group_counts}")

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
        "group_window_counts": {str(key): value for key, value in sorted(group_counts.items())},
        "source_frames": stats["source_frames"],
        "raw_window_frames": RAW_WINDOW_FRAMES,
        "frame_stride": FRAME_STRIDE,
        "unique_sampled_frames": UNIQUE_MODEL_FRAMES,
        "model_frames": MODEL_FRAMES,
        "temporal_padding": "repeat each window's final sampled frame once",
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
