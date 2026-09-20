#!/usr/bin/env python3
"""Build the oracle-informed K=4 Mass Balance support/query protocol.

Each environment keeps its formal K=2 supports, adds one fixed-seed random
unbalanced trajectory, and adds the balanced trajectory whose terminal tilt
is closest to zero. All four supports are excluded from the query set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidate-outcomes", type=Path, required=True)
    parser.add_argument("--k2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--balanced-threshold-deg", type=float, default=3.0)
    parser.add_argument("--random-seed", type=int, default=20260919)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    metadata = read_jsonl(args.metadata)
    outcomes = read_jsonl(args.candidate_outcomes)
    k2_rows = json.loads(args.k2_manifest.read_text(encoding="utf-8"))
    threshold = float(args.balanced_threshold_deg)

    by_ratio: dict[int, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        indices = [int(value) for value in outcome.get("sample_indices", [])]
        if len(indices) != 1 or outcome.get("ground_truth_value") is None:
            continue
        index = indices[0]
        row = metadata[index]
        if int(row.get("window_index", 1)) != 1:
            continue
        by_ratio.setdefault(int(row["ratio_index"]), []).append(
            {
                "index": index,
                "action_id": str(outcome["action_id"]),
                "action_order": float(outcome["action_order"]),
                "gt_tilt_deg": float(outcome["ground_truth_value"]),
            }
        )

    manifest: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for k2 in k2_rows:
        old_supports = [int(value) for value in k2["support_indices"]]
        if len(old_supports) != 2:
            raise ValueError(f"Expected K=2 input, got supports={old_supports}")
        ratio_index = int(metadata[old_supports[0]]["ratio_index"])
        candidates = sorted(
            by_ratio.get(ratio_index, []), key=lambda item: item["action_order"]
        )
        if len(candidates) != 15:
            raise ValueError(
                f"ratio_index={ratio_index}: expected 15 candidates, got {len(candidates)}"
            )

        balanced = [item for item in candidates if abs(item["gt_tilt_deg"]) <= threshold]
        remaining_unbalanced = [
            item
            for item in candidates
            if abs(item["gt_tilt_deg"]) > threshold
            and int(item["index"]) not in set(old_supports)
        ]
        if not balanced or not remaining_unbalanced:
            raise ValueError(
                f"ratio_index={ratio_index}: balanced={len(balanced)} "
                f"remaining_unbalanced={len(remaining_unbalanced)}"
            )

        balanced_support = min(
            balanced,
            key=lambda item: (
                abs(item["gt_tilt_deg"]),
                abs(item["action_order"]),
                item["action_id"],
            ),
        )
        rng = random.Random(int(args.random_seed) + 1009 * ratio_index)
        random_unbalanced = rng.choice(remaining_unbalanced)
        added = [random_unbalanced, balanced_support]
        support_indices = old_supports + [int(item["index"]) for item in added]
        if len(set(support_indices)) != 4:
            raise ValueError(f"ratio_index={ratio_index}: duplicate K=4 support")

        support_set = set(support_indices)
        query_indices = [
            int(item["index"])
            for item in candidates
            if int(item["index"]) not in support_set
        ]
        query_indices.sort(key=lambda index: int(metadata[index]["support_bin_index"]))
        if len(query_indices) != 11 or support_set & set(query_indices):
            raise ValueError(f"Invalid K=4 split for ratio_index={ratio_index}")

        source = support_indices[0]
        source_row = metadata[source]
        record = {
            **k2,
            "protocol_source_index": source,
            "source_index": source,
            "source_ratio_index": ratio_index,
            "source_mass_ratio": float(source_row["right_to_left_mass_ratio"]),
            "source_sample_id": str(source_row["sample_id"]),
            "support_indices": support_indices,
            "support_sample_ids": [
                str(metadata[index]["sample_id"]) for index in support_indices
            ],
            "query_indices": query_indices,
            "target_indices": query_indices,
            "target_sample_ids": [
                str(metadata[index]["sample_id"]) for index in query_indices
            ],
            "support_selection": "k2_plus_seeded_random_unbalanced_plus_most_balanced",
            "contains_balanced_support": True,
            "random_seed": int(args.random_seed),
        }
        manifest.append(record)
        support_rows.extend(metadata[index] for index in support_indices)

        candidate_by_index = {int(item["index"]): item for item in candidates}
        selections.append(
            {
                "domain": str(k2["domain"]),
                "ratio_index": ratio_index,
                "mass_ratio": float(source_row["right_to_left_mass_ratio"]),
                "selection_rule": record["support_selection"],
                "support_indices": support_indices,
                "support_action_ids": [
                    candidate_by_index[index]["action_id"] for index in support_indices
                ],
                "support_action_orders": [
                    candidate_by_index[index]["action_order"] for index in support_indices
                ],
                "support_gt_tilts_deg": [
                    candidate_by_index[index]["gt_tilt_deg"] for index in support_indices
                ],
                "added_random_unbalanced_index": int(random_unbalanced["index"]),
                "added_balanced_index": int(balanced_support["index"]),
                "query_indices": query_indices,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "input_support_query_manifest.json", manifest)
    atomic_json(args.output_dir / "support_selection.json", selections)
    atomic_text(
        args.output_dir / "support_metadata.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in support_rows),
    )
    atomic_json(
        args.output_dir / "protocol.json",
        {
            "version": 1,
            "task": "mass_balance",
            "dataset": "workspace_random_30ratio_noleak",
            "environment_selection": "same_5_id_5_ood_as_formal_k1_and_k2",
            "support_size": 4,
            "queries_per_environment": 11,
            "balanced_interval_deg": [-threshold, threshold],
            "support_policy": "formal K2 plus one seeded-random unbalanced and one most-balanced sample",
            "balanced_support_forbidden": False,
            "oracle_informed_support": True,
            "random_seed": int(args.random_seed),
        },
    )
    print(
        f"[done] environments={len(manifest)} supports={len(support_rows)} "
        f"queries={sum(len(row['query_indices']) for row in manifest)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
