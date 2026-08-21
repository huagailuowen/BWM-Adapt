#!/usr/bin/env python3
"""Evaluate collision action selection from candidate-rollout JSONL records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.evaluation.io import write_json_atomic, write_jsonl_atomic  # noqa: E402
from wan_video_action.metrics.action_metrics import (  # noqa: E402
    CollisionActionCandidate,
    CollisionActionSettings,
    TargetInterval,
    evaluate_collision_action_decision,
    summarize_collision_action_decisions,
)


GROUP_FIELDS = ("method", "decision_id", "seed", "target_area_id")
OUTPUT_CONTEXT_FIELDS = (
    "dataset",
    "task",
    "method",
    "checkpoint_id",
    "split",
    "domain",
    "support_size",
    "seed",
    "environment_id",
    "decision_id",
    "target_area_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select each collision action from predicted target-object displacement, "
            "then score the matching GT outcome."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Candidate-action JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-on-table", action="store_true")
    parser.add_argument("--require-in-workspace", action="store_true")
    parser.add_argument("--lateral-tolerance-m", type=float, default=None)
    parser.add_argument("--success-tolerance-m", type=float, default=0.0)
    parser.add_argument("--invalid-cost-margin-m", type=float, default=None)
    parser.add_argument(
        "--allow-oracle-unreachable",
        action="store_true",
        help="Keep decisions whose candidate set cannot reach the target; default is strict.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            row["_source_line"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"candidate manifest is empty: {path}")
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = [field for field in GROUP_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"manifest line {row['_source_line']} is missing grouping fields: {missing}"
            )
        key = tuple(str(row[field]) for field in GROUP_FIELDS)
        grouped[key].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _consistent_value(rows: list[dict[str, Any]], field: str, *, required: bool = False) -> Any:
    values = [row.get(field) for row in rows]
    if required and any(value is None for value in values):
        lines = [row["_source_line"] for row in rows if row.get(field) is None]
        raise ValueError(f"field {field!r} is missing on manifest lines {lines}")
    present = [value for value in values if value is not None]
    if not present:
        return None
    canonical = json.dumps(present[0], sort_keys=True)
    if any(json.dumps(value, sort_keys=True) != canonical for value in present[1:]):
        lines = [row["_source_line"] for row in rows]
        raise ValueError(f"field {field!r} is inconsistent in decision lines {lines}")
    return present[0]


def main() -> None:
    args = parse_args()
    settings = CollisionActionSettings(
        require_clean_collision=True,
        require_on_table=args.require_on_table,
        require_in_workspace=args.require_in_workspace,
        lateral_tolerance_m=args.lateral_tolerance_m,
        success_tolerance_m=args.success_tolerance_m,
        invalid_cost_margin_m=args.invalid_cost_margin_m,
    )

    decision_rows: list[dict[str, Any]] = []
    for rows in _group_rows(_load_jsonl(args.manifest)):
        target = TargetInterval(
            target_area_id=str(_consistent_value(rows, "target_area_id", required=True)),
            low_m=float(_consistent_value(rows, "target_low_m", required=True)),
            high_m=float(_consistent_value(rows, "target_high_m", required=True)),
        )
        candidates = [CollisionActionCandidate.from_mapping(row) for row in rows]
        result = evaluate_collision_action_decision(candidates, target, settings)
        if not args.allow_oracle_unreachable and not result["collision_oracle_reachable"]:
            decision_id = _consistent_value(rows, "decision_id", required=True)
            raise ValueError(
                f"decision {decision_id!r} cannot reach target {target.target_area_id!r}; "
                "remove it from the eligible set or pass --allow-oracle-unreachable"
            )

        context = {
            field: _consistent_value(rows, field)
            for field in OUTPUT_CONTEXT_FIELDS
            if any(field in row for row in rows)
        }
        decision_rows.append({**context, **result})

    summary = summarize_collision_action_decisions(decision_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "collision_action_per_decision.jsonl", decision_rows)
    write_json_atomic(args.output_dir / "collision_action_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
