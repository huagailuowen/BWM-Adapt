#!/usr/bin/env python3
"""Build a 2x2 Light Switch K=2 support-state sweep with fixed queries."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path


VARIANTS = {
    "b0r0": {"blue": 0, "red": 0},
    "b0r1": {"blue": 0, "red": 1},
    "b1r0": {"blue": 1, "red": 0},
    "b1r1": {"blue": 1, "red": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def choose_pair(
    rows: list[dict], environment: dict, desired: dict[str, int]
) -> tuple[list[int], int]:
    causal_class = str(environment["causal_class"])
    context_group_id = float(environment["context_group_id"])
    original_action_id = int(environment["support_selection"]["matched_action_id"])
    excluded_episodes = {int(value) for value in environment["unique_query_episode_indices"]}
    eligible_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row.get("causal_class")) == causal_class
        and float(row.get("context_group_id", -1)) == context_group_id
        and int(row.get("episode_index", -1)) not in excluded_episodes
    ]
    complete_actions = []
    for candidate_action in sorted({int(row["action_id"]) for _, row in eligible_rows}):
        counts = [
            sum(
                int(row["action_id"]) == candidate_action
                and str(row["button_color"]) == color
                and int(row["lamp_before"]) == lamp
                for _, row in eligible_rows
            )
            for color in ("blue", "red")
            for lamp in (0, 1)
        ]
        if all(counts):
            complete_actions.append((candidate_action, min(counts)))
    if not complete_actions:
        raise RuntimeError(f"No complete factorial action level for {context_group_id=}")
    action_id, _ = min(
        complete_actions,
        key=lambda item: (abs(item[0] - original_action_id), -item[1], item[0]),
    )
    ranked: dict[str, list[tuple[float, int]]] = {}
    for color in ("blue", "red"):
        candidates = [
            (index, row)
            for index, row in eligible_rows
            if int(row.get("action_id", -1)) == action_id
            and str(row.get("button_color")) == color
            and int(row.get("lamp_before", -1)) == desired[color]
            and int(row.get("episode_index", -1)) not in excluded_episodes
        ]
        if not candidates:
            raise RuntimeError(f"No {causal_class=} {action_id=} {color=} lamp={desired[color]}")
        depths = [float(row.get("physical_press_depth_m", 0.0)) for _, row in candidates]
        median = statistics.median(depths)
        ranked[color] = sorted(
            (abs(float(row.get("physical_press_depth_m", 0.0)) - median), index)
            for index, row in candidates
        )

    valid_pairs = []
    for blue_distance, blue_index in ranked["blue"]:
        for red_distance, red_index in ranked["red"]:
            if int(rows[blue_index]["episode_index"]) == int(rows[red_index]["episode_index"]):
                continue
            valid_pairs.append((blue_distance + red_distance, blue_index, red_index))
    if not valid_pairs:
        raise RuntimeError(f"No cross-episode support pair for {causal_class=} {desired=}")
    _, blue_index, red_index = min(valid_pairs)
    return [blue_index, red_index], action_id


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    base = json.loads(args.base_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, desired in VARIANTS.items():
        manifest = copy.deepcopy(base)
        for environment in manifest:
            support_indices, action_id = choose_pair(rows, environment, desired)
            environment["support_indices"] = support_indices
            environment["support_episode_indices"] = [
                int(rows[index]["episode_index"]) for index in support_indices
            ]
            environment["support_selection"] = {
                "rule": "factorial_lamp_before_matched_action_nearest_depth_median",
                "matched_action_id": action_id,
                "button_order": ["blue", "red"],
                "lamp_before": desired,
            }
        output = args.output_dir / f"support_query_manifest_redblue_k2_{variant}.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{variant}: {output}")


if __name__ == "__main__":
    main()
