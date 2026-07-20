#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass/"
    "libero_two_box_collision_9speed_20mass_180eps_lerobot_2026-07-16_hai-machine"
)
DEFAULT_OUTPUT = Path("data/mass_collision_bwm_full61_20260716")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
NUM_FRAMES = 61
PROMPT = (
    "observe a moving box collide with a second box on a smooth table and predict "
    "both boxes' post-collision motion; the environment parameter is hidden"
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


def compute_action_stats(source_root: Path, metadata: list[dict]) -> dict:
    actions = np.concatenate(
        [read_actions(source_root / data_path(int(row["episode_index"]))) for row in metadata],
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


def build_row(source_root: Path, meta: dict, total_frames: int) -> dict:
    episode_index = int(meta["episode_index"])
    mass_index = int(meta["target_mass_index"])
    target_mass_kg = float(meta["target_mass_kg"])
    action_id = int(meta["action_id"])
    action_rel = data_path(episode_index)
    videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
    for relative in [action_rel, *videos]:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)

    metrics = meta.get("metrics", {})
    return {
        "sample_id": f"mass_collision:ep{episode_index:06d}:frames0000-0060",
        "episode_index": episode_index,
        "case_id": str(meta["case_id"]),
        "pair_id": f"m{mass_index:02d}_a{action_id:02d}",
        "source_dataset": source_root.name,
        "source_split": "mass20_speed9_hidden",
        "target_mass_index": mass_index,
        "target_mass_kg": target_mass_kg,
        "target_mass_g": float(meta["target_mass_g"]),
        "physical_parameter_name": "target_mass_kg",
        # Compatibility alias used only to index the grouped latent table.
        # The target mass is never exposed directly to the model.
        "friction_mu": target_mass_kg,
        "action_id": action_id,
        "speed_index": action_id,
        "action_amplitude": float(meta["A"]),
        "preimpact_speed_mps": float(meta["calibrated_preimpact_vx_mps"]),
        "push_steps": int(meta["push_steps"]),
        "first_collision_frame": int(metrics.get("first_block_collision_frame", -1)),
        "separation_frame": int(metrics.get("separation_frames", -1)),
        "start_frame": 0,
        "end_frame": NUM_FRAMES - 1,
        "length": NUM_FRAMES,
        "valid_frames": total_frames,
        "total_frames": total_frames,
        "pad_short": total_frames < NUM_FRAMES,
        "chunk_type": "full_two_box_collision_rollout",
        "video": videos,
        "action": action_rel,
        "prompt": PROMPT,
        "task": "hidden_target_mass_collision_dynamics",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir
    metadata = read_jsonl(source_root / "meta/push_box_episode_metadata.jsonl")
    episodes = read_jsonl(source_root / "meta/episodes.jsonl")
    episode_lengths = {
        int(record["episode_index"]): int(record["length"])
        for record in episodes
    }
    if len(metadata) != 180 or len(episode_lengths) != 180:
        raise ValueError(
            f"Expected 180 metadata rows and episodes, found {len(metadata)} and "
            f"{len(episode_lengths)}."
        )
    if set(episode_lengths.values()) != {60}:
        raise ValueError(f"Expected all source episodes to have 60 frames: {episode_lengths}")

    pair_counts = Counter(
        (int(row["target_mass_index"]), int(row["action_id"]))
        for row in metadata
    )
    mass_values = sorted({float(row["target_mass_kg"]) for row in metadata})
    action_ids = sorted({int(row["action_id"]) for row in metadata})
    if len(mass_values) != 20 or len(action_ids) != 9:
        raise ValueError(
            f"Expected 20 masses x 9 actions, found {len(mass_values)} x {len(action_ids)}."
        )
    if len(pair_counts) != 180 or set(pair_counts.values()) != {1}:
        raise ValueError("Mass/action Cartesian product is incomplete or duplicated.")

    rows = [
        build_row(source_root, row, episode_lengths[int(row["episode_index"])])
        for row in metadata
    ]
    write_jsonl(output_dir / "train.jsonl", rows)
    action_stats = compute_action_stats(source_root, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_stats.json").write_text(
        json.dumps(action_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_root": str(source_root),
        "num_samples": len(rows),
        "num_frames": NUM_FRAMES,
        "valid_frames": 60,
        "mass_group_count": len(mass_values),
        "mass_values_kg": mass_values,
        "speed_action_count": len(action_ids),
        "action_ids": action_ids,
        "samples_per_mass": 9,
        "samples_per_action": 20,
        "group_field": "target_mass_kg",
        "trainer_compatibility_group_field": "friction_mu",
        "action_group_field": "action_id",
        "mass_is_model_input": False,
        "short_episodes_are_tail_padded": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] samples={len(rows)} mass_groups={len(mass_values)} "
        f"speed_actions={len(action_ids)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
