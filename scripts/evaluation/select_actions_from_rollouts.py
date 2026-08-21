#!/usr/bin/env python3
"""Freeze model decisions from prediction-only candidate rollout artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.evaluation.io import write_json_atomic, write_jsonl_atomic  # noqa: E402
from wan_video_action.metrics.action_metrics import TargetInterval  # noqa: E402
from wan_video_action.metrics.action_outcomes import extract_predicted_outcome  # noqa: E402


GROUP_FIELDS = ("method", "decision_id", "seed", "target_area_id")
FORBIDDEN_GT_FIELDS = {
    "gt_outcome",
    "gt_final_light_on",
    "gt_state_path",
    "gt_video_path",
    "oracle_action_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("scalar_interval", "lightswitch_discrete"), required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--extractor-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
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
            leaked = sorted(FORBIDDEN_GT_FIELDS.intersection(row))
            if leaked:
                raise ValueError(
                    f"prediction-only manifest line {line_number} contains forbidden GT fields: {leaked}"
                )
            row["_source_line"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"prediction manifest is empty: {path}")
    return rows


def _consistent(rows: list[dict[str, Any]], field: str) -> Any:
    if any(field not in row for row in rows):
        raise ValueError(f"field {field!r} is required on every candidate")
    values = [row[field] for row in rows]
    canonical = json.dumps(values[0], sort_keys=True)
    if any(json.dumps(value, sort_keys=True) != canonical for value in values[1:]):
        raise ValueError(f"field {field!r} is inconsistent within a decision")
    return values[0]


def _predicted_value(row: dict[str, Any], kind: str, extractor: dict[str, Any] | None) -> float:
    field = "predicted_outcome" if kind == "scalar_interval" else "predicted_final_light_on_probability"
    if field in row:
        return float(row[field])
    if extractor is None or "pred_state_path" not in row:
        raise ValueError(f"candidate requires {field!r} or pred_state_path plus extractor config")
    return extract_predicted_outcome(row["pred_state_path"], extractor)


def main() -> None:
    args = parse_args()
    extractor = None
    if args.extractor_config:
        payload = yaml.safe_load(args.extractor_config.read_text(encoding="utf-8"))
        extractor = payload.get("prediction_extractor", payload)
    source_bytes = args.prediction_manifest.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    rows = _load_rows(args.prediction_manifest)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in GROUP_FIELDS)].append(row)

    decisions: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        action_ids = [str(row["action_id"]) for row in candidates]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError(f"duplicate action IDs in decision {key}: {action_ids}")
        predicted = [_predicted_value(row, args.kind, extractor) for row in candidates]
        if args.kind == "scalar_interval":
            target = TargetInterval(
                str(_consistent(candidates, "target_area_id")),
                float(_consistent(candidates, "target_low")),
                float(_consistent(candidates, "target_high")),
            )
            errors = [target.distance_m(value) for value in predicted]
            target_payload = {"target_low": target.low_m, "target_high": target.high_m}
            value_field = "predicted_outcome"
        else:
            desired = bool(_consistent(candidates, "desired_light_on"))
            errors = [abs(value - float(desired)) for value in predicted]
            target_payload = {"desired_light_on": desired}
            value_field = "predicted_final_light_on_probability"
        selected_index = min(range(len(candidates)), key=lambda index: (errors[index], action_ids[index]))
        first = candidates[0]
        decisions.append(
            {
                **{
                    field: first.get(field)
                    for field in (
                        "dataset", "task", "benchmark", "method", "checkpoint_id", "split",
                        "domain", "support_size", "support_ids", "seed", "environment_id",
                        "decision_id", "target_area_id",
                    )
                    if field in first
                },
                **target_payload,
                "kind": args.kind,
                "selected_action_id": action_ids[selected_index],
                "selected_predicted_target_error": errors[selected_index],
                "prediction_manifest_sha256": source_sha256,
                "candidates": [
                    {"action_id": action_id, value_field: value, "predicted_target_error": error}
                    for action_id, value, error in zip(action_ids, predicted, errors)
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "selected_actions.prediction_only.jsonl", decisions)
    write_json_atomic(
        args.output_dir / "selection_summary.json",
        {
            "kind": args.kind,
            "decision_count": len(decisions),
            "prediction_manifest": str(args.prediction_manifest.resolve()),
            "prediction_manifest_sha256": source_sha256,
            "gt_fields_rejected": sorted(FORBIDDEN_GT_FIELDS),
        },
    )
    print(f"froze {len(decisions)} prediction-only action decisions")


if __name__ == "__main__":
    main()
