#!/usr/bin/env python3
"""Build a non-leaking K=2 Mass Balance support/query protocol.

For each environment retained by the formal K=1 protocol, this selects the
closest unbalanced left-tilting and right-tilting trajectories around the GT
balanced action region. If the finite action set contains unbalanced samples
on only one side, it uses the two closest unbalanced samples on that side and
records the environment as a single-side fallback. Balanced trajectories are
never selected as support.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidate-outcomes", type=Path, required=True)
    parser.add_argument("--k1-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--balanced-threshold-deg", type=float, default=3.0)
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
    k1_rows = json.loads(args.k1_manifest.read_text(encoding="utf-8"))
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
        ratio_index = int(row["ratio_index"])
        by_ratio.setdefault(ratio_index, []).append(
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
    for k1 in k1_rows:
        old_support = int(k1["support_indices"][0])
        ratio_index = int(metadata[old_support]["ratio_index"])
        candidates = sorted(
            by_ratio.get(ratio_index, []), key=lambda item: item["action_order"]
        )
        if len(candidates) != 15:
            raise ValueError(
                f"ratio_index={ratio_index}: expected 15 candidates, got {len(candidates)}"
            )

        balanced = [item for item in candidates if abs(item["gt_tilt_deg"]) <= threshold]
        negative = [item for item in candidates if item["gt_tilt_deg"] < -threshold]
        positive = [item for item in candidates if item["gt_tilt_deg"] > threshold]
        if not balanced:
            raise ValueError(f"ratio_index={ratio_index}: no balanced action")
        balanced_orders = [item["action_order"] for item in balanced]

        def boundary_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
            return (
                min(abs(item["action_order"] - order) for order in balanced_orders),
                abs(abs(item["gt_tilt_deg"]) - threshold),
                abs(item["gt_tilt_deg"]),
                item["action_id"],
            )

        if negative and positive:
            pair = [min(negative, key=boundary_key), min(positive, key=boundary_key)]
            rule = "nearest_negative_and_positive_unbalanced"
            strict_bidirectional = True
        else:
            one_side = negative or positive
            if len(one_side) < 2:
                raise ValueError(
                    f"ratio_index={ratio_index}: fewer than two non-balanced support candidates"
                )
            pair = sorted(one_side, key=boundary_key)[:2]
            pair.sort(key=lambda item: item["action_order"])
            rule = "two_nearest_unbalanced_single_side_fallback"
            strict_bidirectional = False

        support_indices = [int(item["index"]) for item in pair]
        query_indices = [
            int(item["index"])
            for item in candidates
            if int(item["index"]) not in set(support_indices)
        ]
        query_indices.sort(key=lambda index: int(metadata[index]["support_bin_index"]))
        if len(query_indices) != 13 or set(support_indices) & set(query_indices):
            raise ValueError(f"Invalid K=2 split for ratio_index={ratio_index}")

        source = support_indices[0]
        domain = str(k1["domain"])
        source_row = metadata[source]
        record = {
            **k1,
            "domain": domain,
            "protocol_source_index": source,
            "source_index": source,
            "source_ratio_index": ratio_index,
            "source_mass_ratio": float(source_row["right_to_left_mass_ratio"]),
            "source_sample_id": str(source_row["sample_id"]),
            "support_indices": support_indices,
            "support_sample_ids": [str(metadata[index]["sample_id"]) for index in support_indices],
            "query_indices": query_indices,
            "target_indices": query_indices,
            "target_sample_ids": [str(metadata[index]["sample_id"]) for index in query_indices],
            "support_selection": rule,
            "strict_bidirectional_support": strict_bidirectional,
        }
        manifest.append(record)
        support_rows.extend(metadata[index] for index in support_indices)
        selections.append(
            {
                "domain": domain,
                "ratio_index": ratio_index,
                "mass_ratio": float(source_row["right_to_left_mass_ratio"]),
                "selection_rule": rule,
                "strict_bidirectional_support": strict_bidirectional,
                "support_indices": support_indices,
                "support_action_ids": [item["action_id"] for item in pair],
                "support_action_orders": [item["action_order"] for item in pair],
                "support_gt_tilts_deg": [item["gt_tilt_deg"] for item in pair],
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
            "environment_selection": "same_5_id_5_ood_as_formal_k1",
            "support_size": 2,
            "queries_per_environment": 13,
            "balanced_interval_deg": [-threshold, threshold],
            "support_policy": "nearest negative and positive unbalanced samples",
            "fallback_policy": "two nearest unbalanced samples on the available side",
            "balanced_support_forbidden": True,
            "strict_bidirectional_environment_count": sum(
                bool(row["strict_bidirectional_support"]) for row in selections
            ),
            "single_side_fallback_environment_count": sum(
                not bool(row["strict_bidirectional_support"]) for row in selections
            ),
        },
    )
    print(
        f"[done] environments={len(manifest)} supports={len(support_rows)} "
        f"strict_bidirectional={sum(bool(row['strict_bidirectional_support']) for row in selections)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
