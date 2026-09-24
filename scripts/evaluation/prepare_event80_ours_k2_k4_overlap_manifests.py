#!/usr/bin/env python3
"""Build Event80 K=2/K=4 support-size diagnostics with frozen K=1 queries.

Additional supports are selected from the original K=1 query set by greedy
farthest-point sampling in action-id space. The original query set is retained
for evaluation, so these manifests intentionally record support/query overlap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--k1-manifest", type=Path, required=True)
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
    base = json.loads(args.k1_manifest.read_text(encoding="utf-8"))
    outputs = {2: copy.deepcopy(base), 4: copy.deepcopy(base)}
    selection_rows: list[dict[str, Any]] = []

    for env_index, environment in enumerate(base["environments"]):
        original = int(environment["support_indices"][0])
        original_queries = [int(value) for value in environment["query_indices"]]
        chosen = [original]
        remaining = list(original_queries)
        while len(chosen) < 4:
            chosen_actions = [int(metadata[index]["action_id"]) for index in chosen]
            candidate = max(
                remaining,
                key=lambda index: (
                    min(
                        abs(int(metadata[index]["action_id"]) - action)
                        for action in chosen_actions
                    ),
                    int(metadata[index]["action_id"]),
                ),
            )
            chosen.append(candidate)
            remaining.remove(candidate)

        for support_size in (2, 4):
            row = outputs[support_size]["environments"][env_index]
            supports = chosen[:support_size]
            overlap = [index for index in supports if index in set(original_queries)]
            row["support_indices"] = supports
            row["query_indices"] = original_queries
            row["support_selection"] = "k1_informative_plus_greedy_action_farthest_points"
            row["query_overlap_support_indices"] = overlap
            row["support_query_disjoint"] = False
        selection_rows.append(
            {
                "domain": str(environment["domain"]),
                "environment_id": str(environment["environment_id"]),
                "friction_mu": float(environment["friction_mu"]),
                "original_k1_support": original,
                "original_k1_support_action": int(metadata[original]["action_id"]),
                "k2_support_indices": chosen[:2],
                "k2_support_actions": [int(metadata[index]["action_id"]) for index in chosen[:2]],
                "k4_support_indices": chosen[:4],
                "k4_support_actions": [int(metadata[index]["action_id"]) for index in chosen[:4]],
                "frozen_query_indices": original_queries,
            }
        )

    for support_size, manifest in outputs.items():
        manifest["evaluation_id"] = f"event80_ours_k{support_size}_frozen_k1_query_overlap_v1"
        manifest["support_size"] = support_size
        manifest["protocol"] = (
            "Ours support-size diagnostic: retain the formal K=1 query set; "
            "additional supports are query-derived greedy action farthest points."
        )
        manifest["support_query_disjoint"] = False
        manifest["query_overlap_diagnostic"] = True
        atomic_json(args.output_dir / f"support_query_manifest_k{support_size}.json", manifest)

    atomic_json(args.output_dir / "support_selection_k2_k4.json", selection_rows)
    atomic_json(
        args.output_dir / "protocol.json",
        {
            "version": 1,
            "task": "pushbox_friction_event80",
            "base_protocol": str(args.k1_manifest),
            "environment_count": len(base["environments"]),
            "environment_selection": "frozen formal 5 ID and 5 OOD",
            "query_policy": "retain all nine formal K=1 queries per environment",
            "support_policy": "K=1 informative support plus greedy action-id farthest points",
            "support_sizes": [2, 4],
            "support_query_disjoint": False,
            "query_overlap_diagnostic": True,
        },
    )
    print(f"[done] environments={len(base['environments'])} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
