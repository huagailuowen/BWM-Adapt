#!/usr/bin/env python3
"""Score already-frozen model decisions against a scorer-only GT action table."""

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
    TargetInterval,
    evaluate_lightswitch_action_decision,
    evaluate_scalar_action_decision,
    summarize_action_decisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-actions", type=Path, required=True)
    parser.add_argument("--gt-action-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-oracle-unreachable", action="store_true")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def main() -> None:
    args = parse_args()
    selections = _load_jsonl(args.selected_actions)
    gt_rows = _load_jsonl(args.gt_action_table)
    gt_by_environment: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        key = (
            str(row["benchmark"]),
            str(row["environment_id"]),
            None if "target_area_id" not in row else str(row["target_area_id"]),
        )
        gt_by_environment[key].append(row)

    scored: list[dict[str, Any]] = []
    metric_names: tuple[str, ...] | None = None
    for selection in selections:
        kind = str(selection["kind"])
        target_key = str(selection["target_area_id"]) if kind == "lightswitch_discrete" else None
        key = (str(selection["benchmark"]), str(selection["environment_id"]), target_key)
        gt_candidates = gt_by_environment.get(key)
        if not gt_candidates:
            raise ValueError(f"no GT action table rows match frozen decision key {key}")
        predictions = {str(row["action_id"]): row for row in selection["candidates"]}
        if kind == "scalar_interval":
            candidates = [
                ScalarActionCandidate(
                    action_id=str(row["action_id"]),
                    predicted_outcome=float(predictions[str(row["action_id"])]["predicted_outcome"]),
                    gt_outcome=float(row["gt_outcome"]),
                    valid=bool(row["valid"]),
                )
                for row in gt_candidates
            ]
            result = evaluate_scalar_action_decision(
                candidates,
                TargetInterval(
                    str(selection["target_area_id"]),
                    float(selection["target_low"]),
                    float(selection["target_high"]),
                ),
            )
            metric_names = SCALAR_ACTION_METRIC_NAMES
        elif kind == "lightswitch_discrete":
            candidates = [
                LightSwitchActionCandidate(
                    action_id=str(row["action_id"]),
                    button_color=str(row["button_color"]),
                    predicted_final_light_on_probability=float(
                        predictions[str(row["action_id"])][
                            "predicted_final_light_on_probability"
                        ]
                    ),
                    gt_final_light_on=bool(row["gt_final_light_on"]),
                    valid_press=bool(row["valid_press"]),
                )
                for row in gt_candidates
            ]
            result = evaluate_lightswitch_action_decision(
                candidates,
                target_area_id=str(selection["target_area_id"]),
                desired_light_on=bool(selection["desired_light_on"]),
            )
            metric_names = LIGHTSWITCH_ACTION_METRIC_NAMES
        else:
            raise ValueError(f"unsupported frozen decision kind: {kind!r}")
        if result["selected_action_id"] != str(selection["selected_action_id"]):
            raise ValueError("scorer recomputation does not match the frozen prediction-only decision")
        if not args.allow_oracle_unreachable and not result["action_oracle_reachable"]:
            raise ValueError(f"frozen decision is not GT-reachable: {key}")
        context = {key: value for key, value in selection.items() if key != "candidates"}
        scored.append({**context, **result})

    if metric_names is None:
        raise ValueError("no frozen decisions were scored")
    if len({row["kind"] for row in selections}) != 1:
        raise ValueError("one scoring invocation cannot mix action task kinds")
    summary = summarize_action_decisions(scored, metric_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "action_per_decision.jsonl", scored)
    write_json_atomic(args.output_dir / "action_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
