#!/usr/bin/env python3
"""Derive reproducible Ours K=1 and K=4 Light Switch protocols from formal K=2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--k2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    metadata = [
        json.loads(line)
        for line in args.metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    k2_rows = json.loads(args.k2_manifest.read_text(encoding="utf-8"))
    k1_rows: list[dict[str, Any]] = []
    k4_rows: list[dict[str, Any]] = []
    k8_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for base in k2_rows:
        old_supports = [int(value) for value in base["support_indices"]]
        if len(old_supports) != 2:
            raise ValueError(f"Expected K=2 supports, got {old_supports}")
        by_color = {str(metadata[index]["button_color"]): index for index in old_supports}
        if set(by_color) != {"red", "blue"}:
            raise ValueError(f"Expected red+blue supports, got {by_color}")

        query_indices = [int(value) for value in base["query_indices"]]
        excluded = set(old_supports) | set(query_indices) | {int(base["source_index"])}
        causal_class = str(base["causal_class"])
        extras: dict[str, int] = {}
        for color in ("red", "blue"):
            reference = metadata[by_color[color]]
            desired_before = 1 - int(reference["lamp_before"])
            reference_action = int(reference["action_id"])
            candidates = [
                (index, row)
                for index, row in enumerate(metadata)
                if index not in excluded
                and str(row.get("causal_class")) == causal_class
                and str(row.get("button_color")) == color
                and int(row.get("lamp_before", -1)) == desired_before
            ]
            if not candidates:
                raise ValueError(
                    f"causal_class={causal_class} color={color}: no opposite-state support"
                )
            extra_index, _ = min(
                candidates,
                key=lambda pair: (
                    int(pair[1].get("action_id", -999)) != reference_action,
                    abs(int(pair[1].get("action_id", -999)) - reference_action),
                    pair[0],
                ),
            )
            extras[color] = extra_index
            excluded.add(extra_index)

        k1_supports = [by_color["red"]]
        k4_supports = old_supports + [extras["red"], extras["blue"]]
        if set(k4_supports) & set(query_indices):
            raise ValueError(f"causal_class={causal_class}: support/query overlap")

        k8_supports = list(k4_supports)
        k8_excluded = excluded | set(k4_supports)
        for color in ("red", "blue"):
            for lamp_before in (0, 1):
                existing_actions = [
                    int(metadata[index]["action_id"])
                    for index in k8_supports
                    if str(metadata[index]["button_color"]) == color
                    and int(metadata[index]["lamp_before"]) == lamp_before
                ]
                candidates = [
                    (index, row)
                    for index, row in enumerate(metadata)
                    if index not in k8_excluded
                    and str(row.get("causal_class")) == causal_class
                    and str(row.get("button_color")) == color
                    and int(row.get("lamp_before", -1)) == lamp_before
                ]
                if not candidates:
                    raise ValueError(
                        f"causal_class={causal_class} color={color} lamp_before={lamp_before}: "
                        "no second support"
                    )
                extra_index, _ = max(
                    candidates,
                    key=lambda pair: (
                        min(
                            abs(int(pair[1]["action_id"]) - action)
                            for action in existing_actions
                        ),
                        int(pair[1]["action_id"]),
                        -pair[0],
                    ),
                )
                k8_supports.append(extra_index)
                k8_excluded.add(extra_index)

        def record(supports: list[int], policy: str) -> dict[str, Any]:
            return {
                **base,
                "support_indices": supports,
                "support_sample_ids": [str(metadata[index]["sample_id"]) for index in supports],
                "query_indices": query_indices,
                "target_indices": query_indices,
                "target_sample_ids": [str(metadata[index]["sample_id"]) for index in query_indices],
                "support_selection": policy,
            }

        k1_rows.append(record(k1_supports, "fixed_red_support_from_formal_k2"))
        k4_rows.append(
            record(
                k4_supports,
                "formal_red_blue_k2_plus_opposite_initial_state_per_button_color",
            )
        )
        k8_rows.append(
            record(
                k8_supports,
                "two_supports_per_button_color_and_initial_lamp_state",
            )
        )
        selections.append(
            {
                "causal_class": causal_class,
                "source_index": int(base["source_index"]),
                "query_count": len(query_indices),
                "k1_support_indices": k1_supports,
                "k4_support_indices": k4_supports,
                "k8_support_indices": k8_supports,
                "k4_support_details": [
                    {
                        "index": index,
                        "button_color": str(metadata[index]["button_color"]),
                        "lamp_before": int(metadata[index]["lamp_before"]),
                        "lamp_after": int(metadata[index]["lamp_after"]),
                        "action_id": str(metadata[index]["action_id"]),
                    }
                    for index in k4_supports
                ],
            }
        )

    atomic_json(args.output_dir / "support_query_manifest_red_k1.json", k1_rows)
    atomic_json(args.output_dir / "support_query_manifest_redblue_opposite_state_k4.json", k4_rows)
    atomic_json(args.output_dir / "support_query_manifest_redblue_opposite_state_k8.json", k8_rows)
    atomic_json(args.output_dir / "support_selection_k1_k4.json", selections)
    atomic_json(
        args.output_dir / "protocol_k1_k4.json",
        {
            "version": 1,
            "task": "lightswitch",
            "environments": [str(row["causal_class"]) for row in k2_rows],
            "query_policy": "same 15 formal queries for K=1, K=2, and K=4",
            "k1": {"support_size": 1, "policy": "fixed red support from formal K=2"},
            "k4": {
                "support_size": 4,
                "policy": "red and blue, each with lamp-before 0 and 1",
            },
            "k8": {
                "support_size": 8,
                "policy": "two action-diverse supports for each button-color/lamp-before pair",
            },
            "support_query_disjoint": True,
        },
    )
    print(f"[done] environments={len(k2_rows)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
