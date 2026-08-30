#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path(
    "data/real_stick_balance_8env_120raw_stride3_1general4lift_20260810"
)
DEFAULT_OUTPUT = Path(
    "data/real_stick_balance_8env_120raw_stride3_1general4lift_"
    "episode90train10test_seed20260829"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir
    rows = read_jsonl(source_dir / "train.jsonl")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be strictly between zero and one.")

    rows_by_environment_episode: dict[int, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    environment_names: dict[int, str] = {}
    for row in rows:
        group = int(row["environment_index"])
        episode = int(row["source_episode_index"])
        rows_by_environment_episode[group][episode].append(row)
        environment_names[group] = str(row["environment"])

    test_episodes_by_group: dict[int, set[int]] = {}
    split_records: list[dict] = []
    for group in sorted(rows_by_environment_episode):
        episodes = sorted(rows_by_environment_episode[group])
        test_count = max(1, round(len(episodes) * args.test_fraction))
        rng = random.Random(args.seed + group * 1_000_003)
        held_out = set(rng.sample(episodes, test_count))
        test_episodes_by_group[group] = held_out

        for episode in episodes:
            episode_rows = rows_by_environment_episode[group][episode]
            kinds = Counter(str(row["sampling_kind"]) for row in episode_rows)
            if len(episode_rows) != 5 or kinds != {"general": 1, "lift": 4}:
                raise ValueError(
                    f"Environment {group} episode {episode} has unexpected candidates: "
                    f"rows={len(episode_rows)} kinds={dict(kinds)}"
                )
            split_records.append(
                {
                    "environment_index": group,
                    "environment": environment_names[group],
                    "source_episode_index": episode,
                    "split": "test" if episode in held_out else "train",
                    "candidate_windows": len(episode_rows),
                    "general_windows": kinds["general"],
                    "lift_windows": kinds["lift"],
                }
            )

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for row in rows:
        group = int(row["environment_index"])
        episode = int(row["source_episode_index"])
        split = "test" if episode in test_episodes_by_group[group] else "train"
        output_row = dict(row)
        output_row["dataset_split"] = split
        output_row["episode_split_seed"] = args.seed
        output_row["episode_test_fraction"] = args.test_fraction
        (test_rows if split == "test" else train_rows).append(output_row)

    train_episode_keys = {
        (int(row["environment_index"]), int(row["source_episode_index"]))
        for row in train_rows
    }
    test_episode_keys = {
        (int(row["environment_index"]), int(row["source_episode_index"]))
        for row in test_rows
    }
    overlap = train_episode_keys & test_episode_keys
    if overlap:
        raise RuntimeError(f"Episode leakage across train/test: {sorted(overlap)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    shutil.copy2(source_dir / "action_stats.json", output_dir / "action_stats.json")
    if (source_dir / "manifest_summary.json").is_file():
        shutil.copy2(
            source_dir / "manifest_summary.json",
            output_dir / "source_manifest_summary.json",
        )

    with (output_dir / "episode_split.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "environment_index\tenvironment\tsource_episode_index\tsplit\t"
            "candidate_windows\tgeneral_windows\tlift_windows\n"
        )
        for record in split_records:
            handle.write(
                f"{record['environment_index']}\t{record['environment']}\t"
                f"{record['source_episode_index']}\t{record['split']}\t"
                f"{record['candidate_windows']}\t{record['general_windows']}\t"
                f"{record['lift_windows']}\n"
            )

    per_environment = {}
    for group in sorted(rows_by_environment_episode):
        all_episodes = sorted(rows_by_environment_episode[group])
        test_episodes = sorted(test_episodes_by_group[group])
        train_episodes = [episode for episode in all_episodes if episode not in test_episodes]
        per_environment[str(group)] = {
            "environment": environment_names[group],
            "total_episodes": len(all_episodes),
            "train_episode_count": len(train_episodes),
            "test_episode_count": len(test_episodes),
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
        }

    total_episodes = len(train_episode_keys) + len(test_episode_keys)
    summary = {
        "source_metadata": str((source_dir / "train.jsonl").resolve()),
        "split_unit": "complete source episode within each environment",
        "selection": "deterministic seeded uniform sample without replacement per environment",
        "seed": args.seed,
        "requested_test_fraction": args.test_fraction,
        "actual_test_fraction": len(test_episode_keys) / total_episodes,
        "train_episode_count": len(train_episode_keys),
        "test_episode_count": len(test_episode_keys),
        "total_episode_count": total_episodes,
        "train_candidate_rows": len(train_rows),
        "test_candidate_rows": len(test_rows),
        "leakage_episode_count": len(overlap),
        "test_usage_policy": "never used by training or fixed validation; reserved for inference",
        "per_environment": per_environment,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
