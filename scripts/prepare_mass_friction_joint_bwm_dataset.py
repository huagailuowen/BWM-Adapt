#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass-fric/"
    "libero_two_box_mass_friction_balanced100env_9action_900eps_lerobot_2026-07-18_hai-machine"
)
DEFAULT_OUTPUT = Path("data/mass_friction_joint100_bwm_full61_20260718")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
NUM_FRAMES = 61
PROMPT = (
    "observe a moving box collide with a second box and predict both boxes' "
    "post-collision motion; the environment properties are hidden"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def environment_group_id(environment_id: str) -> int:
    prefix, value = str(environment_id).rsplit("_", 1)
    if prefix != "env":
        raise ValueError(f"Unexpected environment_id={environment_id!r}.")
    return int(value)


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
    group_id = environment_group_id(meta["environment_id"])
    action_id = int(meta["action_id"])
    target_mass_kg = float(meta["target_mass_kg"])
    target_friction = float(meta["target_table_friction_mu"])
    action_rel = data_path(episode_index)
    videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
    for relative in [action_rel, *videos]:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)

    metrics = meta.get("metrics", {})
    return {
        "sample_id": f"mass_friction_joint:ep{episode_index:06d}:frames0000-0060",
        "episode_index": episode_index,
        "case_id": str(meta["case_id"]),
        "pair_id": f"env{group_id:03d}_a{action_id:02d}",
        "source_dataset": source_root.name,
        "source_split": "massfriction100_action9_hidden",
        "environment_id": str(meta["environment_id"]),
        "environment_group_id": group_id,
        "mass_index": int(meta["mass_index"]),
        "target_mass_kg": target_mass_kg,
        "target_mass_g": float(meta["target_mass_g"]),
        "friction_index": int(meta["friction_index"]),
        "target_table_friction_mu": target_friction,
        "projectile_table_friction_mu": float(meta["projectile_table_friction_mu"]),
        "physical_parameter_names": ["target_mass_kg", "target_table_friction_mu"],
        # Compatibility scalar used only to index one C32 per joint environment.
        # It is an environment ID, not a physical friction value, and is never
        # exposed to the model.
        "friction_mu": float(group_id),
        "action_id": action_id,
        "action_amplitude": float(meta["A"]),
        "push_steps": int(meta["push_steps"]),
        "matching_index": int(meta["matching_index"]),
        "matching_name": str(meta["matching_name"]),
        "first_collision_frame": int(metrics.get("first_block_collision_frame", -1)),
        "separation_frame": int(metrics.get("separation_frames", -1)),
        "start_frame": 0,
        "end_frame": NUM_FRAMES - 1,
        "length": NUM_FRAMES,
        "valid_frames": total_frames,
        "total_frames": total_frames,
        "pad_short": total_frames < NUM_FRAMES,
        "chunk_type": "full_two_box_mass_friction_collision_rollout",
        "video": videos,
        "action": action_rel,
        "prompt": PROMPT,
        "task": "hidden_joint_mass_friction_collision_dynamics",
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
    if len(metadata) != 900 or len(episode_lengths) != 900:
        raise ValueError(
            f"Expected 900 metadata rows and episodes, found {len(metadata)} and "
            f"{len(episode_lengths)}."
        )
    if set(episode_lengths.values()) != {60}:
        raise ValueError("Expected every source episode to contain exactly 60 frames.")

    by_environment: dict[int, list[dict]] = defaultdict(list)
    for row in metadata:
        by_environment[environment_group_id(row["environment_id"])].append(row)
    if sorted(by_environment) != list(range(100)):
        raise ValueError("Expected complete joint environment IDs 0 through 99.")

    environment_records = []
    mass_counts = Counter()
    friction_counts = Counter()
    for group_id in sorted(by_environment):
        group_rows = by_environment[group_id]
        actions = sorted(int(row["action_id"]) for row in group_rows)
        pairs = {
            (float(row["target_mass_kg"]), float(row["target_table_friction_mu"]))
            for row in group_rows
        }
        if actions != list(range(9)) or len(pairs) != 1:
            raise ValueError(
                f"Environment {group_id} is not one fixed mass/friction pair with actions 0-8."
            )
        mass, friction = next(iter(pairs))
        mass_counts[mass] += 1
        friction_counts[friction] += 1
        first = group_rows[0]
        environment_records.append(
            {
                "environment_group_id": group_id,
                "environment_id": f"env_{group_id:03d}",
                "trainer_compatibility_friction_mu": float(group_id),
                "target_mass_kg": mass,
                "target_mass_g": float(first["target_mass_g"]),
                "target_table_friction_mu": friction,
                "mass_index": int(first["mass_index"]),
                "friction_index": int(first["friction_index"]),
                "matching_index": int(first["matching_index"]),
                "matching_name": str(first["matching_name"]),
            }
        )
    if set(mass_counts.values()) != {5} or set(friction_counts.values()) != {5}:
        raise ValueError("Joint design is not balanced five times per mass and friction level.")

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
    (output_dir / "environment_table.json").write_text(
        json.dumps({"records": environment_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "num_samples": len(rows),
        "num_frames": NUM_FRAMES,
        "valid_frames": 60,
        "joint_environment_count": len(environment_records),
        "mass_level_count": len(mass_counts),
        "friction_level_count": len(friction_counts),
        "environments_per_mass_level": sorted(set(mass_counts.values())),
        "environments_per_friction_level": sorted(set(friction_counts.values())),
        "actions_per_environment": 9,
        "group_fields": ["target_mass_kg", "target_table_friction_mu"],
        "trainer_compatibility_group_field": "friction_mu (environment_group_id)",
        "physical_parameters_are_model_inputs": False,
        "short_episodes_are_tail_padded": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] samples={len(rows)} joint_environments={len(environment_records)} "
        f"actions_per_environment=9 output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
