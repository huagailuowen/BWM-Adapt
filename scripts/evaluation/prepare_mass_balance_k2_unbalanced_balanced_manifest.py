#!/usr/bin/env python3
"""Build K=2 Mass Balance supports from K=1 unbalanced plus one balanced sample."""

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
    parser.add_argument(
        "--source-indices",
        default="",
        help="Optional comma-separated protocol source indices to retain.",
    )
    parser.add_argument(
        "--balanced-action-overrides",
        default="",
        help="Optional comma-separated SOURCE_INDEX:ACTION_ID overrides.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    source_indices = {
        int(value) for value in args.source_indices.split(",") if value.strip()
    }
    balanced_action_overrides: dict[int, str] = {}
    for value in args.balanced_action_overrides.split(","):
        if not value.strip():
            continue
        source_index, action_id = value.split(":", maxsplit=1)
        balanced_action_overrides[int(source_index)] = str(int(action_id))

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

    manifest, selections, support_rows = [], [], []
    for k1 in k1_rows:
        unbalanced = int(k1["support_indices"][0])
        if source_indices and unbalanced not in source_indices:
            continue
        ratio_index = int(metadata[unbalanced]["ratio_index"])
        candidates = sorted(by_ratio.get(ratio_index, []), key=lambda item: item["action_order"])
        if len(candidates) != 15:
            raise ValueError(f"ratio_index={ratio_index}: expected 15 candidates, got {len(candidates)}")
        unbalanced_item = next(item for item in candidates if int(item["index"]) == unbalanced)
        if abs(unbalanced_item["gt_tilt_deg"]) <= threshold:
            raise ValueError(f"ratio_index={ratio_index}: K1 support is unexpectedly balanced")
        balanced_candidates = [item for item in candidates if abs(item["gt_tilt_deg"]) <= threshold]
        if not balanced_candidates:
            raise ValueError(f"ratio_index={ratio_index}: no balanced action")
        if unbalanced in balanced_action_overrides:
            action_id = balanced_action_overrides[unbalanced]
            matches = [item for item in candidates if item["action_id"] == action_id]
            if len(matches) != 1:
                raise ValueError(
                    f"source_index={unbalanced}: expected one action_id={action_id}, got {len(matches)}"
                )
            balanced = matches[0]
            if abs(balanced["gt_tilt_deg"]) > threshold:
                raise ValueError(
                    f"source_index={unbalanced}: overridden action_id={action_id} is not balanced "
                    f"(tilt={balanced['gt_tilt_deg']:.6f} deg)"
                )
        else:
            balanced = min(
                balanced_candidates,
                key=lambda item: (
                    abs(item["gt_tilt_deg"]),
                    abs(item["action_order"]),
                    item["action_id"],
                ),
            )
        support_indices = [unbalanced, int(balanced["index"])]
        support_set = set(support_indices)
        query_indices = [int(item["index"]) for item in candidates if int(item["index"]) not in support_set]
        query_indices.sort(key=lambda index: int(metadata[index]["support_bin_index"]))
        if len(query_indices) != 13 or support_set & set(query_indices):
            raise ValueError(f"ratio_index={ratio_index}: invalid K=2 split")

        source_row = metadata[unbalanced]
        record = {
            **k1,
            "protocol_source_index": unbalanced,
            "source_index": unbalanced,
            "source_ratio_index": ratio_index,
            "source_mass_ratio": float(source_row["right_to_left_mass_ratio"]),
            "source_sample_id": str(source_row["sample_id"]),
            "support_indices": support_indices,
            "support_sample_ids": [str(metadata[index]["sample_id"]) for index in support_indices],
            "query_indices": query_indices,
            "target_indices": query_indices,
            "target_sample_ids": [str(metadata[index]["sample_id"]) for index in query_indices],
            "support_selection": (
                "formal_k1_nearest_unbalanced_plus_explicit_balanced_action"
                if unbalanced in balanced_action_overrides
                else "formal_k1_nearest_unbalanced_plus_most_balanced"
            ),
            "strict_bidirectional_support": False,
            "contains_balanced_support": True,
            "oracle_informed_support": True,
        }
        manifest.append(record)
        support_rows.extend(metadata[index] for index in support_indices)
        selections.append(
            {
                "domain": str(k1["domain"]),
                "ratio_index": ratio_index,
                "mass_ratio": float(source_row["right_to_left_mass_ratio"]),
                "support_indices": support_indices,
                "support_action_ids": [unbalanced_item["action_id"], balanced["action_id"]],
                "support_action_orders": [unbalanced_item["action_order"], balanced["action_order"]],
                "support_gt_tilts_deg": [unbalanced_item["gt_tilt_deg"], balanced["gt_tilt_deg"]],
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
            "support_policy": "formal K1 nearest-unbalanced plus most-balanced action",
            "source_indices": sorted(source_indices),
            "balanced_action_overrides": balanced_action_overrides,
            "support_query_disjoint": True,
            "oracle_informed_support": True,
        },
    )
    print(f"[done] environments={len(manifest)} supports={len(support_rows)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
