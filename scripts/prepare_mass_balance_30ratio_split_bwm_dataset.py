#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_mass_balance_bwm_dataset import (
    VIDEO_KEYS,
    WINDOWS,
    build_rows,
    compute_action_stats,
    read_jsonl,
    write_jsonl,
)


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/"
    "libero_mass_balance_fixed_pose_30ratio_15support_450eps_combined_"
    "train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine"
)
DEFAULT_OUTPUT = Path(
    "data/mass_balance_fixed_pose_30ratio_train20_stride2_41f_20260902"
)

# Preserve exactly the 20 environments used by the 2026-07-22 fixed-pose run.
TRAIN_RATIOS = (
    0.125,
    0.1666666667,
    0.2,
    0.25,
    0.3333333333,
    0.4,
    0.5,
    0.6666666667,
    0.8,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
)


def is_training_ratio(value: float) -> bool:
    return any(abs(float(value) - expected) <= 1e-8 for expected in TRAIN_RATIOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    metadata = read_jsonl(
        source_root / "meta" / "mass_balance_episode_metadata.jsonl"
    )
    if len(episodes) != 450 or len(metadata) != 450:
        raise ValueError(
            f"Expected 450 episodes, found {len(episodes)} episode rows and "
            f"{len(metadata)} mass rows."
        )

    train_metadata = [
        row
        for row in metadata
        if is_training_ratio(float(row["right_to_left_mass_ratio"]))
    ]
    test_metadata = [
        row
        for row in metadata
        if not is_training_ratio(float(row["right_to_left_mass_ratio"]))
    ]
    episodes_by_id = {int(row["episode_index"]): row for row in episodes}
    train_episodes = [episodes_by_id[int(row["episode_index"])] for row in train_metadata]
    test_episodes = [episodes_by_id[int(row["episode_index"])] for row in test_metadata]

    train_rows = build_rows(source_root, train_episodes, train_metadata)
    test_rows = build_rows(source_root, test_episodes, test_metadata)
    train_counts = Counter(float(row["mass_ratio"]) for row in train_rows)
    test_counts = Counter(float(row["mass_ratio"]) for row in test_rows)
    if len(train_counts) != 20 or set(train_counts.values()) != {45}:
        raise ValueError(f"Expected 20 train ratios with 45 windows each: {train_counts}")
    if len(test_counts) != 10 or set(test_counts.values()) != {45}:
        raise ValueError(f"Expected 10 test ratios with 45 windows each: {test_counts}")

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_stats.json").write_text(
        json.dumps(
            compute_action_stats(source_root, train_episodes),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_root": str(source_root),
        "episodes": len(episodes),
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "train_environments": len(train_counts),
        "test_environments": len(test_counts),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_ratios": sorted(train_counts),
        "test_ratios": sorted(test_counts),
        "raw_windows": [[start, end] for start, end, _ in WINDOWS],
        "raw_window_frames": 80,
        "frame_stride": 2,
        "unique_sampled_frames": 40,
        "model_frames": 41,
        "temporal_padding": "repeat each window's final sampled frame once",
        "required_late_windows_per_six": 4,
        "video_keys": list(VIDEO_KEYS),
        "action_stats_scope": "training 20 ratios only",
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
