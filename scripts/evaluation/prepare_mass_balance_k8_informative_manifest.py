#!/usr/bin/env python3
"""Extend the frozen Mass Balance K4 supports with four action-space-farthest clips."""

import argparse
import json
import os
from pathlib import Path


def write_atomic(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if jsonl:
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in value)
    else:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--k4-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    k4_rows = json.loads(args.k4_manifest.read_text())
    if len(k4_rows) != 10:
        raise ValueError(f"Expected 10 environments, got {len(k4_rows)}")

    manifest = []
    selections = []
    support_rows = []
    for old in k4_rows:
        chosen = [int(index) for index in old["support_indices"]]
        pool = chosen + [int(index) for index in old["query_indices"]]
        if len(chosen) != 4 or len(pool) != 15 or len(set(pool)) != 15:
            raise ValueError(f"Invalid K4 action pool for source {old['source_index']}")
        added = []
        while len(chosen) < 8:
            remaining = [index for index in pool if index not in chosen]
            selected = max(
                remaining,
                key=lambda index: (
                    min(
                        abs(int(metadata[index]["support_bin_index"]) - int(metadata[prior]["support_bin_index"]))
                        for prior in chosen
                    ),
                    -int(metadata[index]["support_bin_index"]),
                ),
            )
            chosen.append(selected)
            added.append(selected)
        queries = sorted(
            (index for index in pool if index not in chosen),
            key=lambda index: int(metadata[index]["support_bin_index"]),
        )
        if len(queries) != 7 or set(chosen) & set(queries):
            raise ValueError(f"Invalid K8 split for source {old['source_index']}")
        record = {
            **old,
            "support_indices": chosen,
            "support_sample_ids": [str(metadata[index]["sample_id"]) for index in chosen],
            "query_indices": queries,
            "target_indices": queries,
            "target_sample_ids": [str(metadata[index]["sample_id"]) for index in queries],
            "support_selection": "frozen_k4_plus_four_greedy_farthest_action_bins",
            "added_action_space_indices": added,
        }
        manifest.append(record)
        support_rows.extend(metadata[index] for index in chosen)
        selections.append({
            "source_index": int(old["source_index"]),
            "domain": old["domain"],
            "ratio_index": int(old["source_ratio_index"]),
            "frozen_k4_support_indices": [int(index) for index in old["support_indices"]],
            "added_action_space_indices": added,
            "support_indices": chosen,
            "query_indices": queries,
        })

    output = args.output_dir
    write_atomic(output / "input_support_query_manifest.json", manifest)
    write_atomic(output / "support_selection.json", selections)
    write_atomic(output / "support_metadata.jsonl", support_rows, jsonl=True)
    write_atomic(output / "protocol.json", {
        "version": 1,
        "task": "mass_balance",
        "dataset": "workspace_random_30ratio_noleak",
        "environment_selection": "same_5_id_5_ood_as_formal_k1_k2_k4",
        "support_size": 8,
        "queries_per_environment": 7,
        "support_policy": "frozen K4 plus four greedy-farthest action bins; no additional GT-based selection",
        "oracle_informed_support": True,
        "reference_k4_manifest": str(args.k4_manifest),
    })
    print(f"[done] environments={len(manifest)} supports={len(support_rows)} queries=70 output={output}", flush=True)


if __name__ == "__main__":
    main()
