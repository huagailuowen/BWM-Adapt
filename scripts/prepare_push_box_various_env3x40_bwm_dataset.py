#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SUBSET = "hidden_straight_lerobot"
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PROMPT = "observe how the object slides after a short robot push on the table; no target is shown"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def episode_path(kind: str, episode_index: int, video_key: str | None = None) -> str:
    chunk = f"chunk-{int(episode_index) // 1000:03d}"
    if kind == "action":
        return f"{SUBSET}/data/{chunk}/episode_{episode_index:06d}.parquet"
    if kind == "video" and video_key is not None:
        return f"{SUBSET}/videos/{chunk}/{video_key}/episode_{episode_index:06d}.mp4"
    raise ValueError(f"Unsupported episode path: kind={kind!r}, video_key={video_key!r}")


def action_stats(source_root: Path, episodes: list[dict]) -> dict:
    arrays = []
    for episode in episodes:
        path = source_root / episode_path("action", int(episode["episode_index"]))
        table = pq.read_table(path, columns=["action"])
        arrays.append(np.asarray(table.to_pydict()["action"], dtype=np.float32))
    action = np.concatenate(arrays, axis=0)
    minimum = action.min(axis=0)
    maximum = action.max(axis=0)
    stats = {
        "shape": [int(action.shape[1])],
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "p01": minimum.tolist(),
        "p99": maximum.tolist(),
        "mean": action.mean(axis=0).tolist(),
        "std": action.std(axis=0).tolist(),
    }
    return {"action_pose": stats, "eef_delta": stats}


def validate_grid(episodes: list[dict]) -> tuple[list[int], list[float], list[int]]:
    environment_indices = sorted({int(row["environment_index"]) for row in episodes})
    if environment_indices != [0, 1, 2]:
        raise ValueError(f"Expected environment indices [0, 1, 2], got {environment_indices}.")
    reference_mu = None
    reference_actions = None
    for environment_index in environment_indices:
        subset = [row for row in episodes if int(row["environment_index"]) == environment_index]
        mu_values = sorted({float(row["mu"]) for row in subset})
        action_ids = sorted({int(row["action_id"]) for row in subset})
        if len(mu_values) != 40 or action_ids != list(range(10)) or len(subset) != 400:
            raise ValueError(
                f"Environment {environment_index} is not a 40-friction x 10-action grid: "
                f"episodes={len(subset)} frictions={len(mu_values)} actions={action_ids}."
            )
        if reference_mu is None:
            reference_mu = mu_values
            reference_actions = action_ids
        elif mu_values != reference_mu or action_ids != reference_actions:
            raise ValueError("The three environments do not share identical friction/action tables.")
    return environment_indices, reference_mu or [], reference_actions or []


def build_rows(source_root: Path, episodes: list[dict], start_frame: int, num_frames: int) -> tuple[list[dict], list[dict]]:
    rows = []
    group_records = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        environment_index = int(episode["environment_index"])
        mu_index = int(episode["mu_index"])
        action_id = int(episode["action_id"])
        context_group_id = environment_index * 40 + mu_index
        physical_mu = float(episode["mu"])
        metrics = episode["metrics"]
        phase_counts = metrics["phase_counts"]
        push_start = int(phase_counts["approach"]) + int(phase_counts["descend"])
        push_end = push_start + int(phase_counts["push"])
        total_frames = int(metrics["steps"])
        valid_frames = max(0, min(int(num_frames), total_frames - int(start_frame)))
        action_path = episode_path("action", episode_index)
        video_paths = [
            episode_path("video", episode_index, video_key)
            for video_key in VIDEO_KEYS
        ]
        for relative_path in [action_path, *video_paths]:
            if not (source_root / relative_path).is_file():
                raise FileNotFoundError(source_root / relative_path)

        group_key = f"env{environment_index:02d}_mu{mu_index:02d}"
        rows.append(
            {
                "sample_id": f"{SUBSET}:ep{episode_index:06d}:frames{start_frame:04d}-{start_frame + num_frames - 1:04d}",
                "episode_index": episode_index,
                "source_dataset": SUBSET,
                "source_split": "train",
                "pair_id": group_key,
                "case_id": str(episode["case_id"]),
                "environment_group": group_key,
                "environment_index": environment_index,
                "environment_id": str(episode["environment_id"]),
                "context_group_id": context_group_id,
                "friction_mu": float(context_group_id),
                "physical_friction_mu": physical_mu,
                "mu_index": mu_index,
                "action_id": action_id,
                "action_amplitude": float(episode["A"]),
                "push_action_peak_x": float(episode["A"]),
                "push_start": push_start,
                "push_end": push_end,
                "push_steps": int(phase_counts["push"]),
                "chunk_type": "fixed_65_105",
                "start_frame": int(start_frame),
                "end_frame": int(start_frame) + int(num_frames) - 1,
                "length": int(num_frames),
                "valid_frames": valid_frames,
                "total_frames": total_frames,
                "pad_short": valid_frames < int(num_frames),
                "video": video_paths,
                "action": action_path,
                "prompt": PROMPT,
                "task": "libero_plus_push_box_various_environment_physical_observation",
            }
        )
        group_records[context_group_id] = {
            "context_group_id": context_group_id,
            "environment_group": group_key,
            "environment_index": environment_index,
            "environment_id": str(episode["environment_id"]),
            "mu_index": mu_index,
            "physical_friction_mu": physical_mu,
        }
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows, [group_records[index] for index in sorted(group_records)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 3-style x 40-friction x 10-action push-box BWM manifest.")
    parser.add_argument(
        "--source-root",
        default="/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox_various_env/libero_plus_push_box_event_tap_segmented40_10action_3env_hidden_lerobot_A500_offset160_stop_2026-07-27_hai-machine",
    )
    parser.add_argument(
        "--output-dir",
        default="data/push_box_bwm_various_env3x40_10action_65_105_20260727",
    )
    parser.add_argument("--start-frame", type=int, default=65)
    parser.add_argument("--num-frames", type=int, default=41)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    episodes = read_jsonl(source_root / SUBSET / "meta" / "push_box_episode_metadata.jsonl")
    environment_indices, friction_values, action_ids = validate_grid(episodes)
    rows, group_records = build_rows(source_root, episodes, args.start_frame, args.num_frames)
    counts = Counter(int(row["context_group_id"]) for row in rows)
    if len(counts) != 120 or set(counts.values()) != {10}:
        raise ValueError(f"Expected 120 groups with 10 episodes each, got {dict(counts)}.")

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    with (output_dir / "action_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(action_stats(source_root, episodes), handle, indent=2, sort_keys=True)
    with (output_dir / "context_group_map.json").open("w", encoding="utf-8") as handle:
        json.dump(group_records, handle, indent=2, sort_keys=True)
    summary = {
        "source_root": str(source_root),
        "episodes": len(episodes),
        "train_samples": len(rows),
        "environment_indices": environment_indices,
        "friction_values": friction_values,
        "action_ids": action_ids,
        "context_groups": len(group_records),
        "samples_per_context_group": sorted(set(counts.values())),
        "context_lookup_field": "friction_mu",
        "context_lookup_encoding": "environment_index * 40 + mu_index",
        "physical_friction_field": "physical_friction_mu",
        "start_frame": int(args.start_frame),
        "end_frame": int(args.start_frame) + int(args.num_frames) - 1,
        "num_frames": int(args.num_frames),
    }
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
