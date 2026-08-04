#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare_lightswitch_grouped_bwm_dataset import (
    CAUSAL_CLASSES,
    FRAME_STRIDE,
    NUM_MODEL_FRAMES,
    RAW_FRAME_SPAN,
    WINDOW_START_STRIDE,
    build_rows,
    compute_action_stats,
    write_jsonl,
)


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/"
    "robomme_light_switch_independent_controls_random8_fixed_close_buttons_no_pause_"
    "random_initial_absolute_eef_200eps_hai-machine_lerobot"
)
DEFAULT_OUTPUT = Path(
    "data/lightswitch_randominitial_absolute_eef_environment8_bwm_10fps41_20260729"
)
GROUPS_PER_CLASS = 2
EPISODES_PER_GROUP = 25


def make_environment8_assignment(
    episodes: list[dict], seed: int
) -> tuple[dict[int, int], dict]:
    counts = Counter(str(row["causal_class"]) for row in episodes)
    expected = {causal_class: 50 for causal_class in CAUSAL_CLASSES}
    if dict(counts) != expected:
        raise ValueError(f"Expected 50 episodes per causal class, got {dict(counts)}")

    rng = random.Random(seed)
    assignment: dict[int, int] = {}
    groups: list[dict] = []
    for class_index, causal_class in enumerate(CAUSAL_CLASSES):
        episode_indices = sorted(
            int(row["episode_index"])
            for row in episodes
            if row["causal_class"] == causal_class
        )
        rng.shuffle(episode_indices)
        for subgroup_index in range(GROUPS_PER_CLASS):
            group_id = class_index * GROUPS_PER_CLASS + subgroup_index
            begin = subgroup_index * EPISODES_PER_GROUP
            selected = episode_indices[begin : begin + EPISODES_PER_GROUP]
            if len(selected) != EPISODES_PER_GROUP:
                raise ValueError(
                    f"Group {group_id} has {len(selected)} episodes, "
                    f"expected {EPISODES_PER_GROUP}."
                )
            for episode_index in selected:
                assignment[episode_index] = group_id
            groups.append(
                {
                    "group_id": group_id,
                    "causal_class": causal_class,
                    "class_index": class_index,
                    "class_subgroup_index": subgroup_index,
                    "episode_indices": sorted(selected),
                    "active_stage1": True,
                }
            )

    if len(assignment) != len(episodes):
        raise ValueError(
            f"Assigned {len(assignment)} episodes, expected {len(episodes)}."
        )
    manifest = {
        "seed": seed,
        "causal_classes": list(CAUSAL_CLASSES),
        "groups_per_causal_class": GROUPS_PER_CLASS,
        "episodes_per_group": EPISODES_PER_GROUP,
        "active_group_ids": list(range(len(CAUSAL_CLASSES) * GROUPS_PER_CLASS)),
        "groups": groups,
    }
    return assignment, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
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

    assignment, manifest = make_environment8_assignment(episodes, args.seed)
    rows = build_rows(source_root, episodes, assignment, "environment8")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "environment8_train.jsonl", rows)
    (output_dir / "environment8_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "action_stats.json").write_text(
        json.dumps(
            compute_action_stats(source_root, episodes),
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
        "sample_count": len(rows),
        "environment_count": len(CAUSAL_CLASSES),
        "context_group_count": len(CAUSAL_CLASSES) * GROUPS_PER_CLASS,
        "groups_per_environment": GROUPS_PER_CLASS,
        "episodes_per_context_group": EPISODES_PER_GROUP,
        "all_groups_active": True,
        "causal_class_counts": dict(
            Counter(str(row["causal_class"]) for row in episodes)
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] samples={len(rows)} context_groups=8 "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
