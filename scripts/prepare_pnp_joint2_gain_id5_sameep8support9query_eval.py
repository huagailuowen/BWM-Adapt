#!/usr/bin/env python3
"""Freeze an ID5 PnP protocol with eight same-episode support windows per env."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ENV_INDICES = (0, 5, 10, 14, 19)
RAW_SPAN = 120
NUM_FRAMES = 41
FRAME_STRIDE = 3
STAGE_PHASES = (1, 3, 5, 7)
SUPPORT_POSITION = 1
SUPPORT_ACTION = 5
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


def support_rows(base_row: dict, source_root: Path) -> list[dict]:
    phase_values = np.asarray(
        pq.read_table(source_root / base_row["action"], columns=["phase_index"])[
            "phase_index"
        ].to_pylist(),
        dtype=np.int64,
    )
    total_frames = len(phase_values)
    max_start = total_frames - 1 - RAW_SPAN
    if max_start < 0:
        raise ValueError(f"Episode {base_row['episode_index']} is too short.")

    stage_starts = []
    phase_ranges = {}
    for phase in STAGE_PHASES:
        indices = np.flatnonzero(phase_values == phase)
        if indices.size == 0 or np.any(np.diff(indices) != 1):
            raise ValueError(
                f"Episode {base_row['episode_index']} has invalid phase {phase}."
            )
        first, last = int(indices[0]), int(indices[-1])
        center = (first + last) / 2.0
        start = int(round(center - RAW_SPAN / 2.0))
        start = min(max(start, 0), max_start)
        stage_starts.append(start)
        phase_ranges[phase] = [first, last]

    rng = np.random.default_rng(20260902 + int(base_row["episode_index"]) * 104729)
    available = np.asarray(
        [start for start in range(max_start + 1) if start not in set(stage_starts)],
        dtype=np.int64,
    )
    general_starts = [int(value) for value in rng.choice(available, size=4, replace=False)]

    result = []
    candidates = [
        ("stage", start, phase) for start, phase in zip(stage_starts, STAGE_PHASES)
    ] + [("general", start, None) for start in general_starts]
    for order, (kind, start, phase) in enumerate(candidates):
        row = dict(base_row)
        row["sample_id"] = (
            f"pnp_j2gain:m{int(row['target_machine_index']):02d}:"
            f"support_ep{int(row['episode_index']):06d}:{kind}{order}:"
            f"raw{start:04d}-{start + RAW_SPAN:04d}:s3:main"
        )
        row["sampling_kind"] = kind
        row["candidate_index"] = order
        row["support_window_order"] = order
        row["support_target_phase"] = phase
        row["support_target_phase_range"] = None if phase is None else phase_ranges[phase]
        row["start_frame"] = start
        row["end_frame"] = start + RAW_SPAN
        row["length"] = NUM_FRAMES
        row["valid_frames"] = NUM_FRAMES
        row["raw_frame_span"] = RAW_SPAN
        row["frame_stride"] = FRAME_STRIDE
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_metadata)
    protocol_dir = args.result_root / "protocol"
    eval_rows: list[dict] = []
    selection: list[dict] = []
    execution_groups: list[dict] = []

    for group_index, machine_index in enumerate(ENV_INDICES):
        base_support = select_one(
            source_rows,
            machine_index,
            {
                "position": SUPPORT_POSITION,
                "action": SUPPORT_ACTION,
                "sampling_kind": "phase1",
                "candidate_index": 1,
            },
        )
        supports = support_rows(base_support, args.source_root)
        queries = [select_one(source_rows, machine_index, spec) for spec in QUERY_SPECS]
        gain = float(base_support["joint1_command_response_gain"])
        group_records = []
        for role, order, row in [
            *(("support", order, row) for order, row in enumerate(supports)),
            *(("query", order, row) for order, row in enumerate(queries)),
        ]:
            sample_index = len(eval_rows)
            frozen = dict(row)
            frozen["evaluation_group_index"] = group_index
            frozen["evaluation_role"] = role
            frozen["evaluation_order"] = order
            frozen["evaluation_protocol"] = args.result_root.name
            eval_rows.append(frozen)
            record = {
                "group_index": group_index,
                "machine_index": machine_index,
                "joint1_command_response_gain": gain,
                "role": role,
                "order": order,
                "sample_index": sample_index,
                "episode_index": int(frozen["episode_index"]),
                "position_index": int(frozen["grasp_position_index"]),
                "action_index": int(frozen["candidate_action_index"]),
                "sampling_kind": str(frozen["sampling_kind"]),
                "target_phase": frozen.get("support_target_phase"),
                "candidate_index": int(frozen["candidate_index"]),
                "start_frame": int(frozen["start_frame"]),
                "end_frame": int(frozen["end_frame"]),
                "sample_id": str(frozen["sample_id"]),
            }
            selection.append(record)
            group_records.append(record)
        support_records = [row for row in group_records if row["role"] == "support"]
        query_records = [row for row in group_records if row["role"] == "query"]
        if len({row["episode_index"] for row in support_records}) != 1:
            raise RuntimeError(f"Machine {machine_index} support spans multiple episodes.")
        execution_groups.append(
            {
                "group_index": group_index,
                "machine_index": machine_index,
                "joint1_command_response_gain": gain,
                "support_episode_index": support_records[0]["episode_index"],
                "support_anchor_index": support_records[0]["sample_index"],
                "supports": support_records,
                "queries": query_records,
            }
        )

    if len(eval_rows) != 85:
        raise RuntimeError(f"Expected 85 rows, got {len(eval_rows)}.")
    write_jsonl(protocol_dir / "frozen_eval_metadata.jsonl", eval_rows)
    write_jsonl(
        protocol_dir / "frozen_support_manifest.jsonl",
        [row for row in selection if row["role"] == "support"],
    )
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
        "support_episode_count_per_environment": 1,
        "support_windows_per_environment": 8,
        "support_window_policy": {
            "stage": 4,
            "stage_target_phases": list(STAGE_PHASES),
            "general": 4,
            "general_selection": "deterministic uniform random over the full episode",
        },
        "query_windows_per_environment": 9,
        "query_action_policy": "all nine action IDs not used by support, balanced across positions",
        "query_window_mix": {"phase1": 5, "general": 4},
        "stage2_support_loss": "mean over eight chunks via sequential gradient accumulation",
        "model_frames": NUM_FRAMES,
        "raw_frame_span": RAW_SPAN,
        "frame_stride": FRAME_STRIDE,
        "views": ["observation.images.image"],
        "groups": execution_groups,
    }
    (protocol_dir / "execution_plan.json").write_text(
        json.dumps(execution_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[protocol] envs=5 supports=40 queries=45 rows={len(eval_rows)} "
        f"root={args.result_root}"
    )


if __name__ == "__main__":
    main()
