#!/usr/bin/env python3
"""Freeze the joint-2-gain ID5 support-to-nine-query evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ENV_INDICES = (0, 5, 10, 14, 19)
SUPPORT_SPEC = {
    "position": 1,
    "action": 5,
    "sampling_kind": "phase1",
    "candidate_index": 1,
}
QUERY_SPECS = (
    {"position": 0, "action": 0, "sampling_kind": "phase1", "candidate_index": 0},
    {"position": 1, "action": 1, "sampling_kind": "phase1", "candidate_index": 1},
    {"position": 2, "action": 2, "sampling_kind": "phase1", "candidate_index": 2},
    {"position": 0, "action": 3, "sampling_kind": "phase1", "candidate_index": 0},
    {"position": 1, "action": 4, "sampling_kind": "phase1", "candidate_index": 1},
    {"position": 2, "action": 6, "sampling_kind": "general", "candidate_index": 3},
    {"position": 0, "action": 7, "sampling_kind": "general", "candidate_index": 4},
    {"position": 1, "action": 8, "sampling_kind": "general", "candidate_index": 3},
    {"position": 2, "action": 9, "sampling_kind": "general", "candidate_index": 4},
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def select_one(rows: list[dict], machine_index: int, spec: dict) -> dict:
    matches = [
        row
        for row in rows
        if int(row["target_machine_index"]) == machine_index
        and int(row["grasp_position_index"]) == int(spec["position"])
        and int(row["candidate_action_index"]) == int(spec["action"])
        and str(row["sampling_kind"]) == str(spec["sampling_kind"])
        and int(row["candidate_index"]) == int(spec["candidate_index"])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one row for machine={machine_index}, spec={spec}; got {len(matches)}."
        )
    return dict(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_metadata)
    protocol_dir = args.result_root / "protocol"
    eval_rows: list[dict] = []
    selection: list[dict] = []
    execution_groups: list[dict] = []

    for group_index, machine_index in enumerate(ENV_INDICES):
        group_records = []
        support = select_one(source_rows, machine_index, SUPPORT_SPEC)
        gain = float(support["joint1_command_response_gain"])
        for role, query_order, row in [
            ("support", -1, support),
            *(
                ("query", query_order, select_one(source_rows, machine_index, spec))
                for query_order, spec in enumerate(QUERY_SPECS)
            ),
        ]:
            sample_index = len(eval_rows)
            frozen = dict(row)
            frozen["evaluation_group_index"] = group_index
            frozen["evaluation_role"] = role
            frozen["evaluation_query_order"] = query_order
            frozen["evaluation_protocol"] = args.result_root.name
            eval_rows.append(frozen)
            record = {
                "group_index": group_index,
                "machine_index": machine_index,
                "joint1_command_response_gain": gain,
                "role": role,
                "query_order": query_order,
                "sample_index": sample_index,
                "episode_index": int(frozen["episode_index"]),
                "position_index": int(frozen["grasp_position_index"]),
                "action_index": int(frozen["candidate_action_index"]),
                "sampling_kind": str(frozen["sampling_kind"]),
                "candidate_index": int(frozen["candidate_index"]),
                "start_frame": int(frozen["start_frame"]),
                "end_frame": int(frozen["end_frame"]),
                "sample_id": str(frozen["sample_id"]),
            }
            selection.append(record)
            group_records.append(record)
        execution_groups.append(
            {
                "group_index": group_index,
                "machine_index": machine_index,
                "joint1_command_response_gain": gain,
                "support": group_records[0],
                "queries": group_records[1:],
            }
        )

    if len(eval_rows) != 50:
        raise RuntimeError(f"Expected 50 frozen rows, got {len(eval_rows)}.")
    if any(sum(row["role"] == "support" for row in selection if row["group_index"] == group) != 1 for group in range(5)):
        raise RuntimeError("Every environment must have exactly one support.")

    write_jsonl(protocol_dir / "frozen_eval_metadata.jsonl", eval_rows)
    write_jsonl(
        protocol_dir / "frozen_query_manifest.jsonl",
        [row for row in selection if row["role"] == "query"],
    )
    with (protocol_dir / "selection.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(selection[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(selection)

    execution_plan = {
        "name": args.result_root.name,
        "method": "ours",
        "evaluation_modes": ["stage1_training_time", "stage2_inference_time"],
        "environment_count": 5,
        "environment_indices": list(ENV_INDICES),
        "protocol": "one support learns one environment Z; the fixed Z reconstructs support and transfers to nine held-out queries",
        "support_per_environment": 1,
        "queries_per_environment": 9,
        "query_action_policy": "all nine action IDs not used by support, balanced three per position",
        "query_window_mix": {"phase1": 5, "general": 4},
        "model_frames": 41,
        "raw_frame_span": 120,
        "frame_stride": 3,
        "views": ["observation.images.image"],
        "groups": execution_groups,
    }
    (protocol_dir / "execution_plan.json").write_text(
        json.dumps(execution_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[protocol] environments=5 supports=5 queries=45 rows={len(eval_rows)} "
        f"root={args.result_root}"
    )


if __name__ == "__main__":
    main()
