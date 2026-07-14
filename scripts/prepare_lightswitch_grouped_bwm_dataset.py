#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/"
    "robomme_light_switch_independent_controls_random8_200eps_hai-machine_lerobot"
)
DEFAULT_OUTPUT = Path("data/lightswitch_bwm_10fps41_20260714")
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
CAUSAL_CLASSES = ("neither", "red_only", "blue_only", "both")
NUM_MODEL_FRAMES = 41
FRAME_STRIDE = 3
RAW_FRAME_SPAN = (NUM_MODEL_FRAMES - 1) * FRAME_STRIDE
WINDOW_START_STRIDE = 80
SUBGROUP_SIZE = 5
GROUPS_PER_CLASS = 10
ACTIVE_GROUPS = 20
PROMPT = (
    "predict the robot and lamp response while the causal relationship between "
    "the red and blue buttons and the lamp is hidden"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def data_path(episode_index: int) -> str:
    return f"data/chunk-000/episode_{episode_index:06d}.parquet"


def video_path(episode_index: int, video_key: str) -> str:
    return f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"


def nested_uniform_group_order(num_groups: int, initial_groups: int, total_groups: int) -> list[int]:
    total = min(total_groups, num_groups)
    initial = min(max(initial_groups, 1), total)
    selected: list[int] = []

    def add(index: int) -> None:
        index = max(0, min(num_groups - 1, int(index)))
        if index not in selected:
            selected.append(index)

    if initial == 1:
        add((num_groups - 1) // 2)
    else:
        for i in range(initial):
            add(round(i * (num_groups - 1) / max(initial - 1, 1)))
    while len(selected) < initial:
        candidates = [index for index in range(num_groups) if index not in selected]
        add(max(candidates, key=lambda index: min(abs(index - old) for old in selected)))
    while len(selected) < total:
        candidates = [index for index in range(num_groups) if index not in selected]
        center = (num_groups - 1) / 2.0
        add(
            max(
                candidates,
                key=lambda index: (
                    min(abs(index - old) for old in selected),
                    -abs(float(index) - center),
                    -index,
                ),
            )
        )
    return selected


def window_starts(frame_count: int) -> list[int]:
    max_start = frame_count - RAW_FRAME_SPAN - 1
    if max_start < 0:
        raise ValueError(
            f"Episode has only {frame_count} frames; need at least {RAW_FRAME_SPAN + 1}."
        )
    starts = list(range(0, max_start + 1, WINDOW_START_STRIDE))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def make_group_assignments(episodes: list[dict], seed: int) -> tuple[dict, dict, dict]:
    by_class = {
        causal_class: sorted(
            int(row["episode_index"])
            for row in episodes
            if row["causal_class"] == causal_class
        )
        for causal_class in CAUSAL_CLASSES
    }
    if any(len(indices) != 50 for indices in by_class.values()):
        raise ValueError(f"Expected 50 episodes per causal class, got {Counter(row['causal_class'] for row in episodes)}")

    env4_assignment: dict[int, int] = {}
    for class_index, causal_class in enumerate(CAUSAL_CLASSES):
        for episode_index in by_class[causal_class]:
            env4_assignment[episode_index] = class_index

    rng = random.Random(seed)
    subgroups: dict[int, list[list[int]]] = {}
    for class_index, causal_class in enumerate(CAUSAL_CLASSES):
        shuffled = list(by_class[causal_class])
        rng.shuffle(shuffled)
        subgroups[class_index] = [
            shuffled[offset : offset + SUBGROUP_SIZE]
            for offset in range(0, len(shuffled), SUBGROUP_SIZE)
        ]
        if len(subgroups[class_index]) != GROUPS_PER_CLASS:
            raise ValueError(f"Unexpected subgroup count for {causal_class}: {len(subgroups[class_index])}")

    curriculum_order = nested_uniform_group_order(40, initial_groups=5, total_groups=ACTIVE_GROUPS)
    desired_active_classes = (
        [0, 0, 1, 2, 3]
        + [0, 1, 1, 2, 3]
        + [0, 1, 2, 2, 3]
        + [0, 1, 2, 3, 3]
    )
    next_subgroup = [0, 0, 0, 0]
    group_records: dict[int, dict] = {}

    for activation_rank, (group_id, class_index) in enumerate(
        zip(curriculum_order, desired_active_classes), start=1
    ):
        subgroup_index = next_subgroup[class_index]
        next_subgroup[class_index] += 1
        group_records[group_id] = {
            "group_id": group_id,
            "causal_class": CAUSAL_CLASSES[class_index],
            "class_index": class_index,
            "class_subgroup_index": subgroup_index,
            "episode_indices": subgroups[class_index][subgroup_index],
            "active_stage1": True,
            "activation_rank": activation_rank,
            "activation_wave": (activation_rank - 1) // 5 + 1,
        }

    unused_group_ids = [group_id for group_id in range(40) if group_id not in group_records]
    remaining_subgroups = [
        (class_index, subgroup_index)
        for class_index in range(len(CAUSAL_CLASSES))
        for subgroup_index in range(next_subgroup[class_index], GROUPS_PER_CLASS)
    ]
    for group_id, (class_index, subgroup_index) in zip(unused_group_ids, remaining_subgroups):
        group_records[group_id] = {
            "group_id": group_id,
            "causal_class": CAUSAL_CLASSES[class_index],
            "class_index": class_index,
            "class_subgroup_index": subgroup_index,
            "episode_indices": subgroups[class_index][subgroup_index],
            "active_stage1": False,
            "activation_rank": None,
            "activation_wave": None,
        }

    group40_assignment: dict[int, int] = {}
    for group_id, record in group_records.items():
        for episode_index in record["episode_indices"]:
            group40_assignment[int(episode_index)] = int(group_id)

    manifest = {
        "seed": seed,
        "causal_classes": list(CAUSAL_CLASSES),
        "curriculum_group_order": curriculum_order,
        "active_group_ids": curriculum_order[:ACTIVE_GROUPS],
        "ood_group_ids": [group_id for group_id in range(40) if group_id not in curriculum_order],
        "groups": [group_records[group_id] for group_id in range(40)],
    }
    return env4_assignment, group40_assignment, manifest


def build_rows(
    source_root: Path,
    episodes: list[dict],
    assignment: dict[int, int],
    grouping_name: str,
) -> list[dict]:
    rows: list[dict] = []
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        episode_index = int(episode["episode_index"])
        frame_count = int(episode["frame_count"])
        group_id = int(assignment[episode_index])
        action_rel = data_path(episode_index)
        videos = [video_path(episode_index, key) for key in VIDEO_KEYS]
        for relative in [action_rel, *videos]:
            if not (source_root / relative).is_file():
                raise FileNotFoundError(source_root / relative)

        for window_index, start_frame in enumerate(window_starts(frame_count)):
            end_frame = start_frame + RAW_FRAME_SPAN
            covered_events = [
                event
                for event in episode["events"]
                if start_frame <= int(event["step"]) <= end_frame
            ]
            rows.append(
                {
                    "sample_id": (
                        f"lightswitch:{grouping_name}:g{group_id:02d}:ep{episode_index:06d}:"
                        f"raw{start_frame:04d}-{end_frame:04d}"
                    ),
                    "episode_index": episode_index,
                    "causal_class": episode["causal_class"],
                    "red_controls_lamp": bool(episode["red_controls_lamp"]),
                    "blue_controls_lamp": bool(episode["blue_controls_lamp"]),
                    "context_group_id": group_id,
                    "grouping_name": grouping_name,
                    # Compatibility alias used only to index the latent-C table.
                    "friction_mu": float(group_id),
                    "action_id": window_index,
                    "window_index": window_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "length": NUM_MODEL_FRAMES,
                    "raw_frame_span": RAW_FRAME_SPAN,
                    "frame_stride": FRAME_STRIDE,
                    "source_fps": 30,
                    "target_fps": 10,
                    "covered_event_count": len(covered_events),
                    "covered_button_colors": sorted(
                        {str(event["button_color"]) for event in covered_events}
                    ),
                    "video": videos,
                    "action": action_rel,
                    "prompt": PROMPT,
                    "task": "hidden_button_lamp_causal_dynamics",
                }
            )
    return rows


def compute_action_stats(source_root: Path, episodes: list[dict]) -> dict:
    arrays = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = pq.read_table(source_root / data_path(episode_index), columns=["action"])
        arrays.append(np.asarray(table.to_pydict()["action"], dtype=np.float32)[::FRAME_STRIDE])
    actions = np.concatenate(arrays, axis=0)
    stat = {
        "shape": [int(actions.shape[1])],
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "p01": np.quantile(actions, 0.01, axis=0).tolist(),
        "p99": np.quantile(actions, 0.99, axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
    }
    return {
        "action_joint": stat,
        "joint_delta": stat,
        "action_pose": stat,
        "eef_delta": stat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir
    metadata = json.loads(
        (source_root / "robomme_light_switch_independent_controls_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    episodes = metadata["episodes"]
    if len(episodes) != 200:
        raise ValueError(f"Expected 200 episodes, found {len(episodes)}.")

    env4_assignment, group40_assignment, manifest = make_group_assignments(episodes, args.seed)
    env4_rows = build_rows(source_root, episodes, env4_assignment, "environment4")
    group40_rows = build_rows(source_root, episodes, group40_assignment, "episode5_group40")
    write_jsonl(output_dir / "environment4_train.jsonl", env4_rows)
    write_jsonl(output_dir / "group40_train.jsonl", group40_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "group40_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    action_stats = compute_action_stats(source_root, episodes)
    (output_dir / "action_stats.json").write_text(
        json.dumps(action_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "source_fps": 30,
        "target_fps": 10,
        "frame_stride": FRAME_STRIDE,
        "model_frames": NUM_MODEL_FRAMES,
        "raw_frame_span": RAW_FRAME_SPAN,
        "window_start_stride": WINDOW_START_STRIDE,
        "raw_overlap": RAW_FRAME_SPAN - WINDOW_START_STRIDE,
        "environment4_samples": len(env4_rows),
        "group40_samples": len(group40_rows),
        "environment4_group_count": len(set(env4_assignment.values())),
        "group40_group_count": len(set(group40_assignment.values())),
        "active_group_count": ACTIVE_GROUPS,
        "ood_group_count": 40 - ACTIVE_GROUPS,
        "causal_class_counts": dict(Counter(row["causal_class"] for row in episodes)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] environment4_samples={len(env4_rows)} group40_samples={len(group40_rows)} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
