#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/"
    "pnp_various_machine_dynamics/"
    "libero_last_joint_source_state_action_bank_20machine_3position_10source_"
    "600eps_lerobot_2026-08-20_hai-machine"
)
DEFAULT_OUTPUT = Path("data/pnp_machine20_position3_jointabs_121raw_stride3_phase1_20260820")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
NUM_FRAMES = 41
FRAME_STRIDE = 3
RAW_SPAN = (NUM_FRAMES - 1) * FRAME_STRIDE
PHASE1_WINDOWS_PER_EPISODE = 4
GENERAL_WINDOWS_PER_EPISODE = 1
PROMPT = "predict the robot grasping and placing the object under the commanded joint trajectory"


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


def evenly_spaced_starts(low: int, high: int, count: int) -> list[int]:
    if low > high:
        raise ValueError(f"Invalid start range [{low}, {high}].")
    if count <= 1:
        return [low]
    return [round(low + index * (high - low) / (count - 1)) for index in range(count)]


def deterministic_general_start(max_start: int, episode_index: int) -> int:
    if max_start <= 0:
        return 0
    rng = np.random.default_rng(20260820 + int(episode_index) * 104729)
    return int(rng.integers(0, max_start + 1))


def build_rows(source_root: Path, episodes: list[dict], configs: list[dict]) -> list[dict]:
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    config_by_id = {int(row["episode_index"]): row for row in configs}
    if set(episode_by_id) != set(config_by_id):
        raise ValueError("Episode and machine-configuration metadata do not match.")

    rows = []
    for episode_index in sorted(episode_by_id):
        episode = episode_by_id[episode_index]
        config = config_by_id[episode_index]
        total_frames = int(episode["length"])
        action_rel = data_path(episode_index)
        videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
        parquet_path = source_root / action_rel
        for relative in [action_rel, *videos]:
            if not (source_root / relative).is_file():
                raise FileNotFoundError(source_root / relative)

        phase_values = np.asarray(
            pq.read_table(parquet_path, columns=["phase_index"])["phase_index"].to_pylist(),
            dtype=np.int64,
        )
        phase1_indices = np.flatnonzero(phase_values == 1)
        if phase1_indices.size == 0 or np.any(np.diff(phase1_indices) != 1):
            raise ValueError(f"Episode {episode_index} has invalid phase-1 annotation.")
        phase1_start = int(phase1_indices[0])
        phase1_end_exclusive = int(phase1_indices[-1]) + 1
        max_start = total_frames - 1 - RAW_SPAN
        if max_start < 0:
            raise ValueError(f"Episode {episode_index} is shorter than {RAW_SPAN + 1} frames.")
        focus_low = max(0, phase1_end_exclusive - 1 - RAW_SPAN)
        focus_high = min(phase1_start, max_start)
        if focus_low > focus_high:
            raise ValueError(
                f"Episode {episode_index}: phase 1 [{phase1_start}, "
                f"{phase1_end_exclusive}) cannot fit in a 121-frame window."
            )
        candidates = [
            ("phase1", start)
            for start in evenly_spaced_starts(
                focus_low, focus_high, PHASE1_WINDOWS_PER_EPISODE
            )
        ]
        candidates.append(("general", deterministic_general_start(max_start, episode_index)))

        target_machine = config["target_machine"]
        source_machine = config["source_machine"]
        machine_index = int(target_machine["index"])
        position_index = int(config["grasp_position_index"])
        action_index = int(config["candidate_action_index"])
        for candidate_index, (sampling_kind, start_frame) in enumerate(candidates):
            rows.append(
                {
                    "sample_id": (
                        f"pnp_machine:m{machine_index:02d}:p{position_index}:a{action_index:02d}:"
                        f"{sampling_kind}{candidate_index}:"
                        f"raw{start_frame:04d}-{start_frame + RAW_SPAN:04d}:s3"
                    ),
                    "source_dataset": source_root.name,
                    "source_split": "train_all_600eps",
                    "episode_index": episode_index,
                    "target_machine_index": machine_index,
                    "target_machine_id": str(target_machine["machine_id"]),
                    "joint6_initial_offset_deg": float(
                        target_machine["joint6_initial_offset_deg"]
                    ),
                    "joint6_command_response_gain": float(
                        target_machine["joint6_command_response_gain"]
                    ),
                    "source_machine_index": int(source_machine["index"]),
                    "source_machine_id": str(source_machine["machine_id"]),
                    "grasp_position_index": position_index,
                    "grasp_position_label": str(config["grasp_position_label"]),
                    "source_plate_xy_m": list(config["source_plate_xy_m"]),
                    "candidate_action_index": action_index,
                    "friction_mu": float(machine_index),
                    "action_id": action_index,
                    "sampling_position_id": position_index,
                    "sampling_action_id": action_index,
                    "sampling_kind": sampling_kind,
                    "candidate_index": candidate_index,
                    "phase1_start": phase1_start,
                    "phase1_end_exclusive": phase1_end_exclusive,
                    "start_frame": start_frame,
                    "end_frame": start_frame + RAW_SPAN,
                    "length": NUM_FRAMES,
                    "valid_frames": NUM_FRAMES,
                    "raw_frame_span": RAW_SPAN,
                    "frame_stride": FRAME_STRIDE,
                    "source_fps": 20,
                    "target_fps": 20.0 / FRAME_STRIDE,
                    "total_frames": total_frames,
                    "pad_short": False,
                    "video": videos,
                    "action": action_rel,
                    "action_semantics": (
                        "source-native absolute 7-joint targets plus absolute gripper command"
                    ),
                    "prompt": PROMPT,
                    "task": "pnp_target_machine_dynamics_world_modeling",
                    "episode_tasks": list(episode.get("tasks", [])),
                }
            )
    return rows


def compute_action_stats(source_root: Path, episodes: list[dict]) -> dict:
    arrays = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = pq.read_table(source_root / data_path(episode_index), columns=["action"])
        arrays.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
    values = np.concatenate(arrays, axis=0)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"Expected 8-D absolute joint/gripper action, got {values.shape}.")
    stat = {
        "shape": [8],
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "p01": values.min(axis=0).tolist(),
        "p99": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
    }
    return {"action_joint": stat, "joint_delta": stat}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    configs = read_jsonl(
        source_root / "meta" / "episode_configs_and_outcomes_2026-08-20_hai-machine.jsonl"
    )
    if len(episodes) != 600 or len(configs) != 600:
        raise ValueError(
            f"Expected 600 episodes/configs, found {len(episodes)} and {len(configs)}."
        )
    rows = build_rows(source_root, episodes, configs)
    combo_counts = Counter(
        (
            int(row["target_machine_index"]),
            int(row["grasp_position_index"]),
            int(row["candidate_action_index"]),
            str(row["sampling_kind"]),
        )
        for row in rows
    )
    expected = {
        "phase1": PHASE1_WINDOWS_PER_EPISODE,
        "general": GENERAL_WINDOWS_PER_EPISODE,
    }
    if len(rows) != 3000 or len(combo_counts) != 1200:
        raise ValueError(
            f"Unexpected candidate structure: rows={len(rows)} combos={len(combo_counts)}"
        )
    for key, count in combo_counts.items():
        if count != expected[key[-1]]:
            raise ValueError(f"Unexpected candidate count {key}: {count}")

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
        "target_machines": 20,
        "grasp_positions_per_machine": 3,
        "actions_per_position": 10,
        "candidate_rows": len(rows),
        "phase1_windows_per_episode": PHASE1_WINDOWS_PER_EPISODE,
        "general_windows_per_episode": GENERAL_WINDOWS_PER_EPISODE,
        "model_frames": NUM_FRAMES,
        "raw_frame_span_intervals": RAW_SPAN,
        "raw_frames_inclusive": RAW_SPAN + 1,
        "frame_stride": FRAME_STRIDE,
        "per_rank_update": "3 machines x 2 positions x 3 distinct actions",
        "focused_actions_per_machine": 4,
        "general_actions_per_machine": 2,
        "action_semantics": "absolute 7-joint target plus absolute gripper command",
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
