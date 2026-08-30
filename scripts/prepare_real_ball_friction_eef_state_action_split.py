#!/usr/bin/env python3
"""Create an auditable episode-level train/test split for EEF-state-action training."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_real_ball_friction_7env_1view_bwm_dataset import (
    ENVIRONMENTS,
    build_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between zero and one.")

    source_root = args.source_root.resolve()
    rows, environment_summary = build_rows(source_root)
    episode_rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["environment"]), int(row["source_episode_index"]))
        episode_rows[key].append(row)

    episodes = []
    for key, chunks in sorted(episode_rows.items()):
        ball_ids = {int(row["ball_id"]) for row in chunks}
        levels = {int(row["skill_gear"]) for row in chunks}
        if len(ball_ids) != 1 or len(levels) != 1:
            raise ValueError(f"Episode {key} spans multiple ball/level labels.")
        if len(chunks) != 10:
            raise ValueError(f"Episode {key} has {len(chunks)} chunks instead of 10.")
        episodes.append(
            {
                "key": key,
                "environment": key[0],
                "source_episode_index": key[1],
                "global_episode_index": int(chunks[0]["episode_index"]),
                "ball_id": next(iter(ball_ids)),
                "skill_gear": next(iter(levels)),
                "chunk_count": len(chunks),
            }
        )

    stratum_total = Counter((e["ball_id"], e["skill_gear"]) for e in episodes)
    target_test = round(len(episodes) * args.test_fraction)
    rng = random.Random(args.split_seed)
    selected: set[tuple[str, int]] = set()
    stratum_selected: Counter[tuple[int, int]] = Counter()

    def can_select(episode: dict) -> bool:
        stratum = (episode["ball_id"], episode["skill_gear"])
        return (
            episode["key"] not in selected
            and stratum_selected[stratum] < stratum_total[stratum] - 1
        )

    def select_one(candidates: list[dict]) -> None:
        candidates = [episode for episode in candidates if can_select(episode)]
        if not candidates:
            raise RuntimeError("Unable to satisfy randomized split coverage constraints.")
        episode = rng.choice(candidates)
        selected.add(episode["key"])
        stratum_selected[(episode["ball_id"], episode["skill_gear"])] += 1

    requirements = [
        *(('ball', ball_id) for _, ball_id in ENVIRONMENTS),
        *(('level', level) for level in range(1, 11)),
    ]
    rng.shuffle(requirements)
    for kind, value in requirements:
        if kind == "ball" and any(
            episode["ball_id"] == value and episode["key"] in selected
            for episode in episodes
        ):
            continue
        if kind == "level" and any(
            episode["skill_gear"] == value and episode["key"] in selected
            for episode in episodes
        ):
            continue
        select_one(
            [
                episode
                for episode in episodes
                if episode["ball_id"] == value
                if kind == "ball"
            ]
            if kind == "ball"
            else [episode for episode in episodes if episode["skill_gear"] == value]
        )

    remaining = episodes[:]
    rng.shuffle(remaining)
    for episode in remaining:
        if len(selected) >= target_test:
            break
        if can_select(episode):
            selected.add(episode["key"])
            stratum_selected[(episode["ball_id"], episode["skill_gear"])] += 1
    if len(selected) != target_test:
        raise RuntimeError(f"Selected {len(selected)}/{target_test} requested test episodes.")

    train_rows, test_rows = [], []
    for row in rows:
        key = (str(row["environment"]), int(row["source_episode_index"]))
        split = "test" if key in selected else "train"
        updated = dict(row)
        updated.update(
            {
                "dataset_split": split,
                "episode_split_seed": args.split_seed,
                "action_semantics": "eef_state_action",
                "state_action_layout": (
                    "observation.eef_state[t] || observation.eef_state[t+1]"
                ),
                "next_eef_policy": (
                    "next sampled parquet row; repeat final pose only at episode end"
                ),
            }
        )
        (test_rows if split == "test" else train_rows).append(updated)

    train_episode_counts = Counter(
        (e["ball_id"], e["skill_gear"])
        for e in episodes
        if e["key"] not in selected
    )
    if any(train_episode_counts[stratum] < 1 for stratum in stratum_total):
        raise RuntimeError("At least one ball/level stratum was removed from training.")
    if {e["ball_id"] for e in episodes if e["key"] in selected} != {
        ball_id for _, ball_id in ENVIRONMENTS
    }:
        raise RuntimeError("Test split does not cover every ball.")
    if {e["skill_gear"] for e in episodes if e["key"] in selected} != set(range(1, 11)):
        raise RuntimeError("Test split does not cover every skill level.")

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    split_records = []
    for episode in episodes:
        split_records.append(
            {
                **{key: value for key, value in episode.items() if key != "key"},
                "split": "test" if episode["key"] in selected else "train",
            }
        )
    write_jsonl(output_dir / "episode_split.jsonl", split_records)

    strata = []
    for ball_id, skill_gear in sorted(stratum_total):
        total = stratum_total[(ball_id, skill_gear)]
        train = train_episode_counts[(ball_id, skill_gear)]
        strata.append(
            {
                "ball_id": ball_id,
                "skill_gear": skill_gear,
                "total_episodes": total,
                "train_episodes": train,
                "test_episodes": total - train,
            }
        )
    summary = {
        "source_root": str(source_root),
        "split_unit": "episode",
        "split_algorithm": (
            "seeded random selection with all-ball/all-level test coverage and at "
            "least one train episode retained in every observed ball-by-level stratum"
        ),
        "split_seed": args.split_seed,
        "requested_test_fraction": args.test_fraction,
        "actual_test_episode_fraction": len(selected) / len(episodes),
        "total_episodes": len(episodes),
        "train_episodes": len(episodes) - len(selected),
        "test_episodes": len(selected),
        "chunks_per_episode": 10,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "action_representation": "eef_state_action",
        "state_action_layout": (
            "observation.eef_state[t] || observation.eef_state[t+1]"
        ),
        "environment_summary_before_split": environment_summary,
        "ball_level_episode_counts": strata,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
