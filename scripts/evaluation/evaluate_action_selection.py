#!/usr/bin/env python3
"""Evaluate generic scalar or LightSwitch candidate-action manifests."""

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
    LIGHTSWITCH_ACTION_METRIC_NAMES,
    SCALAR_ACTION_METRIC_NAMES,
    LightSwitchActionCandidate,
    ScalarActionCandidate,
    ScalarActionSettings,
    TargetInterval,
    evaluate_lightswitch_action_decision,
    evaluate_scalar_action_decision,
    summarize_action_decisions,
)


GROUP_FIELDS = ("method", "decision_id", "seed", "target_area_id")
CONTEXT_FIELDS = (
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("scalar_interval", "lightswitch_discrete"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-tolerance", type=float, default=0.0)
    parser.add_argument("--invalid-cost-margin", type=float, default=None)
    parser.add_argument("--allow-oracle-unreachable", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
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
        raise ValueError(f"action candidate manifest is empty: {path}")
    return rows


def _groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = [field for field in GROUP_FIELDS if field not in row]
        if missing:
            raise ValueError(f"line {row['_source_line']} is missing grouping fields: {missing}")
        grouped[tuple(str(row[field]) for field in GROUP_FIELDS)].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _consistent(rows: list[dict[str, Any]], field: str, required: bool = False) -> Any:
    values = [row.get(field) for row in rows]
    if required and any(value is None for value in values):
        raise ValueError(f"field {field!r} is required on every candidate row")
    present = [value for value in values if value is not None]
    if not present:
        return None
    canonical = json.dumps(present[0], sort_keys=True)
    if any(json.dumps(value, sort_keys=True) != canonical for value in present[1:]):
        raise ValueError(f"field {field!r} is inconsistent within one decision")
    return present[0]


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    for rows in _groups(_load_rows(args.manifest)):
        if args.kind == "scalar_interval":
            result = evaluate_scalar_action_decision(
                [ScalarActionCandidate.from_mapping(row) for row in rows],
                TargetInterval(
                    str(_consistent(rows, "target_area_id", True)),
                    float(_consistent(rows, "target_low", True)),
                    float(_consistent(rows, "target_high", True)),
                ),
                ScalarActionSettings(
                    success_tolerance=args.success_tolerance,
                    invalid_cost_margin=args.invalid_cost_margin,
                ),
            )
            metric_names = SCALAR_ACTION_METRIC_NAMES
        else:
            result = evaluate_lightswitch_action_decision(
                [LightSwitchActionCandidate.from_mapping(row) for row in rows],
                target_area_id=str(_consistent(rows, "target_area_id", True)),
                desired_light_on=bool(_consistent(rows, "desired_light_on", True)),
            )
            metric_names = LIGHTSWITCH_ACTION_METRIC_NAMES
        if not args.allow_oracle_unreachable and not result["action_oracle_reachable"]:
            raise ValueError(
                f"decision {_consistent(rows, 'decision_id', True)!r} is not GT-reachable"
            )
        context = {
            field: _consistent(rows, field)
            for field in CONTEXT_FIELDS
            if any(field in row for row in rows)
        }
        results.append({**context, **result})

    summary = summarize_action_decisions(results, metric_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "action_per_decision.jsonl", results)
    write_json_atomic(args.output_dir / "action_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
