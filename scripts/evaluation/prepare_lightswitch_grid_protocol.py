#!/usr/bin/env python3
"""Freeze a disjoint support/query LightSwitch environment-grid protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--existing-support-plan", type=Path, required=True)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries-per-environment", type=int, default=15)
    parser.add_argument("--inner-steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.metadata_path)
    old_support_plans = read_jsonl(args.existing_support_plan)
    source_indices = [int(value) for value in args.source_indices.split(",") if value.strip()]
    if len(source_indices) != 4:
        raise ValueError("The formal LightSwitch protocol requires exactly four environments.")

    expected_classes = {4.0: "neither", 8.0: "red_only", 12.0: "blue_only", 19.0: "both"}
    frozen_support_plans = []
    transfer_plan = []
    environment_records = []

    for source_index in source_indices:
        source = rows[source_index]
        group = float(source["friction_mu"])
        causal_class = str(source["causal_class"])
        source_episode = int(source["episode_index"])
        if expected_classes.get(group) != causal_class:
            raise ValueError(
                f"Unexpected source mapping source={source_index} group={group} class={causal_class}."
            )

        matches = []
        for plan in old_support_plans:
            key = plan.get("group_key", [])
            if len(key) >= 5 and key[-2] == "exclude":
                try:
                    same_group = abs(float(key[2]) - group) <= 1e-8
                    same_episode = int(key[-1]) == source_episode
                except (TypeError, ValueError):
                    continue
                if same_group and same_episode:
                    matches.append(plan)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one frozen support plan for source={source_index}; found {len(matches)}."
            )
        support_plan = dict(matches[0])
        support_indices = [int(value) for value in support_plan["support_indices"]]
        support_episodes = {int(rows[index]["episode_index"]) for index in support_indices}
        if len(support_indices) != 8:
            raise ValueError(f"source={source_index} has {len(support_indices)} supports, expected 8.")
        frozen_support_plans.append(support_plan)

        bucket_order = (("red", 0), ("blue", 0), ("red", 1), ("blue", 1))
        buckets: dict[tuple[str, int], list[int]] = {key: [] for key in bucket_order}
        for index, row in enumerate(rows):
            if abs(float(row["friction_mu"]) - group) > 1e-8:
                continue
            if str(row["causal_class"]) != causal_class:
                continue
            episode = int(row["episode_index"])
            if index == source_index or index in support_indices:
                continue
            if episode in support_episodes:
                continue
            key = (str(row["button_color"]), int(row["lamp_before"]))
            if key in buckets:
                buckets[key].append(index)
        for key in bucket_order:
            buckets[key].sort(
                key=lambda index: (
                    abs(float(rows[index]["event_window_index"]) - 16.5),
                    int(rows[index]["action_id"]),
                    int(rows[index]["episode_index"]),
                    index,
                )
            )

        selected = []
        while len(selected) < int(args.queries_per_environment):
            progress = False
            for key in bucket_order:
                while buckets[key]:
                    index = buckets[key].pop(0)
                    selected.append(index)
                    progress = True
                    break
                if len(selected) == int(args.queries_per_environment):
                    break
            if not progress:
                raise ValueError(
                    f"Could not select {args.queries_per_environment} disjoint queries for {causal_class}."
                )

        transfer_plan.append(
            {
                "source_index": source_index,
                "source_sample_id": source.get("sample_id"),
                "source_friction_mu": group,
                "source_inner_step": int(args.inner_steps),
                "target_indices": selected,
                "target_sample_ids": [rows[index].get("sample_id") for index in selected],
            }
        )
        environment_records.append(
            {
                "causal_class": causal_class,
                "context_group_id": group,
                "source_index": source_index,
                "source_episode_index": source_episode,
                "support_indices": support_indices,
                "support_episode_indices": sorted(support_episodes),
                "query_indices": selected,
                "query_episode_indices": [int(rows[index]["episode_index"]) for index in selected],
                "unique_query_episode_indices": sorted(
                    {int(rows[index]["episode_index"]) for index in selected}
                ),
                "query_condition_counts": {
                    f"{color}_lamp{lamp}": sum(
                        str(rows[index]["button_color"]) == color
                        and int(rows[index]["lamp_before"]) == lamp
                        for index in selected
                    )
                    for color, lamp in bucket_order
                },
            }
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_text(
        output / "support_plan.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in frozen_support_plans),
    )
    write_json(output / "transfer_plan.json", transfer_plan)
    write_json(output / "support_query_manifest.json", environment_records)
    write_text(output / "source_indices.txt", ",".join(map(str, source_indices)) + "\n")
    write_text(
        output / "target_indices_by_source.txt",
        ";".join(
            f"{row['source_index']}:" + ",".join(map(str, row["target_indices"]))
            for row in transfer_plan
        )
        + "\n",
    )
    protocol = {
        "version": 1,
        "protocol_id": "lightswitch_physicalpress33_all4env_support8_query15_v1",
        "metadata_jsonl": str(args.metadata_path),
        "support": {
            "chunks_per_environment": 8,
            "source_indices": source_indices,
            "context_inner_steps": int(args.inner_steps),
        },
        "query": {
            "environments": ["neither", "red_only", "blue_only", "both"],
            "chunks_per_environment": int(args.queries_per_environment),
            "total_chunks": len(source_indices) * int(args.queries_per_environment),
            "disjoint_from_support_episodes": True,
            "query_episodes_are_disjoint_from_support": True,
            "stratified_by": ["button_color", "lamp_before"],
        },
        "metrics": {
            "global_environment_values": ["neither", "red_only", "blue_only", "both"],
            "action_environment_values": ["red_only", "blue_only"],
        },
    }
    write_text(output / "protocol.yaml", yaml.safe_dump(protocol, sort_keys=False))
    print(
        f"[done] protocol={output} environments={len(source_indices)} "
        f"supports={8 * len(source_indices)} queries={len(source_indices) * args.queries_per_environment}"
    )


if __name__ == "__main__":
    main()
