#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/gravity/change_gravity_lerobot_v21"
)
DEFAULT_OUTPUT = Path("data/gravity_bwm_full61_20260710")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
NUM_FRAMES = 61
PROMPT = (
    "observe a blue cube launched from a smooth elevated platform and predict "
    "its trajectory and first landing point; the environment parameter is hidden"
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


def read_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    return np.asarray(table.to_pydict()["action"], dtype=np.float32)


def compute_action_stats(source_root: Path, rows: list[dict]) -> dict:
    actions = np.concatenate(
        [read_actions(source_root / data_path(int(row["episode_index"]))) for row in rows],
        axis=0,
    )
    stat = {
        "shape": [int(actions.shape[1])],
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "p01": np.quantile(actions, 0.01, axis=0).tolist(),
        "p99": np.quantile(actions, 0.99, axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
    }
    return {"action_pose": stat, "eef_delta": stat}


def build_row(source_root: Path, meta: dict) -> dict:
    episode_index = int(meta["episode_index"])
    total_frames = int(meta["frame_count"])
    gravity = float(meta["gravity_mps2"])
    gravity_index = int(meta["gravity_index"])
    speed_index = int(meta["speed_index"])
    action_rel = data_path(episode_index)
    videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
    for relative in [action_rel, *videos]:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)

    return {
        "sample_id": f"gravity_v21:ep{episode_index:06d}:frames0000-0060",
        "episode_index": episode_index,
        "case_id": str(meta["case_id"]),
        "pair_id": f"g{gravity_index:02d}_v{speed_index:02d}",
        "source_dataset": "change_gravity_lerobot_v21",
        "source_split": "gravity80_speed10_hidden",
        "gravity_index": gravity_index,
        "gravity_mps2": gravity,
        "physical_parameter_name": "gravity_mps2",
        # Compatibility alias used by the existing grouped-context trainer.
        # It is a grouping identifier only and is never exposed to the model.
        "friction_mu": gravity,
        "speed_index": speed_index,
        "initial_speed_mps": float(meta["initial_speed_mps"]),
        "action_id": speed_index,
        "launch_frame": int(meta["launch_frame"]),
        "first_table_contact_frame": int(meta["first_table_contact_frame"]),
        "first_table_contact_x_m": float(meta["first_table_contact_x_m"]),
        "theoretical_landing_x_m": float(meta["theoretical_landing_x_m"]),
        "start_frame": 0,
        "end_frame": NUM_FRAMES - 1,
        "length": NUM_FRAMES,
        "valid_frames": total_frames,
        "total_frames": total_frames,
        "pad_short": total_frames < NUM_FRAMES,
        "chunk_type": "full_launch_to_landing",
        "video": videos,
        "action": action_rel,
        "prompt": PROMPT,
        "task": "hidden_gravity_ballistic_dynamics",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir
    episode_meta = read_jsonl(source_root / "meta/gravity_episode_metadata.jsonl")
    if len(episode_meta) != 800:
        raise ValueError(f"Expected 800 episodes, found {len(episode_meta)}.")
    if not all(bool(row.get("quality_pass")) for row in episode_meta):
        raise ValueError("Dataset contains episodes that failed the quality gate.")

    rows = [build_row(source_root, row) for row in episode_meta]
    write_jsonl(output_dir / "train.jsonl", rows)
    action_stats = compute_action_stats(source_root, episode_meta)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_stats.json").write_text(
        json.dumps(action_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gravity_counts = Counter(float(row["gravity_mps2"]) for row in rows)
    speed_counts = Counter(int(row["speed_index"]) for row in rows)
    summary = {
        "source_root": str(source_root),
        "num_samples": len(rows),
        "num_frames": NUM_FRAMES,
        "valid_frame_range": [
            min(int(row["valid_frames"]) for row in rows),
            max(int(row["valid_frames"]) for row in rows),
        ],
        "gravity_group_count": len(gravity_counts),
        "speed_action_count": len(speed_counts),
        "samples_per_gravity": sorted(set(gravity_counts.values())),
        "samples_per_speed": sorted(set(speed_counts.values())),
        "group_field": "gravity_mps2",
        "trainer_compatibility_group_field": "friction_mu",
        "action_group_field": "action_id (alias of speed_index)",
        "gravity_is_model_input": False,
        "short_episodes_are_tail_padded": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] samples={len(rows)} gravity_groups={len(gravity_counts)} "
        f"speed_actions={len(speed_counts)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
