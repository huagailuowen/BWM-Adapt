#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/various-action/"
    "libero_mu0100_workspace_rich_eef_absolute_xyz_action_400eps_lerobot_2026-07-16_hai-machine"
)
DEFAULT_OUTPUT = Path("data/workspace_rich_mu0100_actionbc_41f_stride40_20260716")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PROMPT = "predict the robot and object motion conditioned on the absolute end-effector action sequence"


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


def window_starts(frame_count: int, num_frames: int, chunk_stride: int) -> list[int]:
    last_start = int(frame_count) - int(num_frames)
    if last_start < 0:
        raise ValueError(f"Episode has {frame_count} frames; need at least {num_frames}.")
    starts = list(range(0, last_start + 1, int(chunk_stride)))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_rows(
    source_root: Path,
    episodes: list[dict],
    *,
    num_frames: int,
    chunk_stride: int,
) -> list[dict]:
    rows = []
    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        episode_index = int(episode["episode_index"])
        frame_count = int(episode["length"])
        action_rel = data_path(episode_index)
        videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
        for relative in [action_rel, *videos]:
            if not (source_root / relative).is_file():
                raise FileNotFoundError(source_root / relative)
        for window_index, start_frame in enumerate(
            window_starts(frame_count, num_frames, chunk_stride)
        ):
            end_frame = start_frame + int(num_frames) - 1
            rows.append(
                {
                    "sample_id": (
                        f"workspace_rich:ep{episode_index:06d}:"
                        f"frames{start_frame:04d}-{end_frame:04d}"
                    ),
                    "episode_index": episode_index,
                    "source_dataset": "workspace_rich_mu0100_eef_absolute_xyz_action",
                    "source_split": "train_all_400eps",
                    "friction_mu": 0.1,
                    "action_id": window_index,
                    "window_index": window_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "length": int(num_frames),
                    "valid_frames": int(num_frames),
                    "total_frames": frame_count,
                    "frame_stride": 1,
                    "video": videos,
                    "action": action_rel,
                    "prompt": PROMPT,
                    "task": "workspace_rich_absolute_eef_action_behavior_cloning",
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
        # Preserve rare workspace-extreme actions instead of clipping them.
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
    parser.add_argument("--num-frames", type=int, default=41)
    parser.add_argument("--chunk-stride", type=int, default=40)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    if len(episodes) != 400:
        raise ValueError(f"Expected 400 episodes, found {len(episodes)}.")
    rows = build_rows(
        source_root,
        episodes,
        num_frames=int(args.num_frames),
        chunk_stride=int(args.chunk_stride),
    )
    stats = compute_action_stats(source_root, episodes)
    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "episodes": len(episodes),
        "source_frames": sum(int(row["length"]) for row in episodes),
        "train_samples": len(rows),
        "model_frames": int(args.num_frames),
        "chunk_stride": int(args.chunk_stride),
        "chunk_overlap": int(args.num_frames) - int(args.chunk_stride),
        "video_keys": list(VIDEO_KEYS),
        "action_semantics": "absolute eef target XYZ plus relative axis-angle and gripper",
        "friction_mu": 0.1,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
