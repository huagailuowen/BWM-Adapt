"""Dataset-level target definitions and GT reachability filtering."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wan_video_action.metrics.action_metrics import TargetInterval


_MISSING = object()


def load_metadata_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load JSONL records or a list nested in a JSON document."""

    paths = metadata.get("paths")
    if paths is None:
        paths = [metadata["path"]]
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(str(raw_path))
        source_format = str(metadata.get("format", "auto"))
        if source_format == "auto":
            source_format = "jsonl" if path.suffix == ".jsonl" else "json"
        if source_format == "jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    records.append({**row, "_source_path": str(path), "_source_line": line_number})
        elif source_format == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records_key = metadata.get("records_key")
            payload = _nested_get(payload, str(records_key)) if records_key else payload
            if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
                raise ValueError(f"{path} does not contain a list of metadata objects")
            records.extend(
                {**row, "_source_path": str(path), "_source_line": index + 1}
                for index, row in enumerate(payload)
            )
        else:
            raise ValueError(f"unsupported metadata format: {source_format!r}")
    if not records:
        raise ValueError("target metadata contains no records")
    return records


def build_target_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(config["kind"])
    records = load_metadata_records(config["metadata"])
    if kind == "scalar_interval":
        return _build_scalar_registry(config, records)
    if kind == "lightswitch_discrete":
        return _build_lightswitch_registry(config, records)
    raise ValueError(f"unsupported action target kind: {kind!r}")


def flatten_eligible_environments(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in registry["target_areas"]:
        rows.extend(target["eligible_environments"])
    return rows


def flatten_gt_action_outcomes(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the scorer-only GT action table stored in a target registry."""

    return [dict(row) for row in registry["gt_action_table"]]


def _build_scalar_registry(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    environment_field = str(config["environment_field"])
    action_field = str(config["action_field"])
    outcome_field = str(config["outcome"]["field"])
    rules = list(config.get("validity_rules", []))
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_stable_id(_nested_get(row, environment_field))].append(row)

    gt_action_table: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str]] = set()
    for row in records:
        environment_id = _stable_id(_nested_get(row, environment_field))
        action_id = _stable_id(_nested_get(row, action_field))
        key = (environment_id, action_id)
        if key in seen_actions:
            raise ValueError(f"duplicate GT action for environment/action pair: {key}")
        seen_actions.add(key)
        gt_action_table.append(
            {
                "task": config["task"],
                "benchmark": config["benchmark"],
                "environment_id": environment_id,
                "action_id": action_id,
                "gt_outcome": float(_nested_get(row, outcome_field)),
                "valid": _matches_rules(row, rules),
                "source_path": row.get("_source_path"),
                "source_line": row.get("_source_line"),
            }
        )

    target_results: list[dict[str, Any]] = []
    for target_spec in config["target_areas"]:
        target = TargetInterval(
            str(target_spec["id"]), float(target_spec["low"]), float(target_spec["high"])
        )
        eligible: list[dict[str, Any]] = []
        ineligible_ids: list[str] = []
        for environment_id, candidates in sorted(grouped.items()):
            valid_candidates = [row for row in candidates if _matches_rules(row, rules)]
            successful = [
                row
                for row in valid_candidates
                if target.contains(float(_nested_get(row, outcome_field)))
            ]
            if not successful:
                ineligible_ids.append(environment_id)
                continue
            gt_errors = [
                target.distance_m(float(_nested_get(row, outcome_field)))
                for row in valid_candidates
            ]
            min_error = min(gt_errors)
            oracle = [
                row
                for row, error in zip(valid_candidates, gt_errors)
                if abs(error - min_error) <= 1e-12
            ]
            eligible.append(
                {
                    "task": config["task"],
                    "benchmark": config["benchmark"],
                    "target_area_id": target.target_area_id,
                    "environment_id": environment_id,
                    "candidate_count": len(candidates),
                    "valid_candidate_count": len(valid_candidates),
                    "successful_candidate_count": len(successful),
                    "successful_action_ids": _sorted_values(
                        _nested_get(row, action_field) for row in successful
                    ),
                    "oracle_action_ids": _sorted_values(
                        _nested_get(row, action_field) for row in oracle
                    ),
                }
            )
        reference_count = target_spec.get("reference_eligible_environment_count")
        target_results.append(
            {
                **dict(target_spec),
                "outcome_name": config["outcome"]["name"],
                "outcome_unit": config["outcome"].get("unit"),
                "eligible_environment_count": len(eligible),
                "ineligible_environment_count": len(ineligible_ids),
                "reference_eligible_environment_count": reference_count,
                "reference_count_delta": (
                    None if reference_count is None else len(eligible) - int(reference_count)
                ),
                "eligible_environments": eligible,
                "ineligible_environment_ids": ineligible_ids,
            }
        )

    return {
        "schema_version": 1,
        "kind": "scalar_interval",
        "task": config["task"],
        "benchmark": config["benchmark"],
        "dataset": config["dataset"],
        "metadata_record_count": len(records),
        "environment_count": len(grouped),
        "environment_field": environment_field,
        "action_field": action_field,
        "outcome": dict(config["outcome"]),
        "validity_rules": rules,
        "gt_action_table": gt_action_table,
        "target_areas": target_results,
    }


def _build_lightswitch_registry(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    environment_field = str(config["environment_field"])
    episode_field = str(config["episode_field"])
    red_field = str(config["control_fields"]["red"])
    blue_field = str(config["control_fields"]["blue"])
    allowed = {str(value) for value in config["eligible_environment_values"]}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        environment_id = _stable_id(_nested_get(row, environment_field))
        if environment_id in allowed:
            grouped[environment_id].append(row)

    environments: dict[str, dict[str, Any]] = {}
    for environment_id, rows in sorted(grouped.items()):
        red_values = {_as_bool(_nested_get(row, red_field)) for row in rows}
        blue_values = {_as_bool(_nested_get(row, blue_field)) for row in rows}
        if len(red_values) != 1 or len(blue_values) != 1:
            raise ValueError(f"inconsistent LightSwitch controls for {environment_id}")
        red_controls = next(iter(red_values))
        blue_controls = next(iter(blue_values))
        if int(red_controls) + int(blue_controls) != 1:
            raise ValueError(
                f"LightSwitch evaluation requires exactly one controlling button: {environment_id}"
            )
        environments[environment_id] = {
            "red_controls_lamp": red_controls,
            "blue_controls_lamp": blue_controls,
            "episode_ids": _sorted_values(_nested_get(row, episode_field) for row in rows),
            "episode_count": len(rows),
            "oracle_button_color": "red" if red_controls else "blue",
        }

    missing = sorted(allowed - set(environments))
    if missing:
        raise ValueError(f"LightSwitch metadata is missing required causal classes: {missing}")

    target_results: list[dict[str, Any]] = []
    gt_action_table: list[dict[str, Any]] = []
    for target_spec in config["target_areas"]:
        initial_light_on = _as_bool(target_spec["initial_light_on"])
        desired_light_on = _as_bool(target_spec["desired_light_on"])
        eligible = [
            {
                "task": config["task"],
                "benchmark": config["benchmark"],
                "target_area_id": target_spec["id"],
                "environment_id": environment_id,
                **environment,
            }
            for environment_id, environment in sorted(environments.items())
        ]
        for environment_id, environment in sorted(environments.items()):
            for button_color in ("red", "blue"):
                controls = bool(environment[f"{button_color}_controls_lamp"])
                gt_final_light_on = desired_light_on if controls else initial_light_on
                gt_action_table.append(
                    {
                        "task": config["task"],
                        "benchmark": config["benchmark"],
                        "target_area_id": target_spec["id"],
                        "environment_id": environment_id,
                        "action_id": button_color,
                        "button_color": button_color,
                        "gt_final_light_on": gt_final_light_on,
                        "valid_press": True,
                    }
                )
        target_results.append(
            {
                **dict(target_spec),
                "eligible_environment_count": len(eligible),
                "eligible_environments": eligible,
                "ineligible_environment_ids": [],
            }
        )

    return {
        "schema_version": 1,
        "kind": "lightswitch_discrete",
        "task": config["task"],
        "benchmark": config["benchmark"],
        "dataset": config["dataset"],
        "metadata_record_count": len(records),
        "environment_count": len(environments),
        "eligible_causal_classes": sorted(environments),
        "gt_action_table": gt_action_table,
        "target_areas": target_results,
    }


def _matches_rules(row: Mapping[str, Any], rules: Sequence[Mapping[str, Any]]) -> bool:
    for rule in rules:
        value = _nested_get(row, str(rule["field"]), default=_MISSING)
        operation = str(rule["op"])
        expected = rule.get("value")
        if operation == "exists":
            passed = value is not _MISSING
        elif operation == "not_null":
            passed = value is not _MISSING and value is not None
        elif value is _MISSING:
            passed = False
        elif operation == "equals":
            passed = value == expected
        elif operation == "not_equals":
            passed = value != expected
        elif operation == "lte":
            passed = float(value) <= float(expected)
        elif operation == "gte":
            passed = float(value) >= float(expected)
        elif operation == "abs_lte":
            passed = abs(float(value)) <= float(expected)
        elif operation == "in":
            passed = value in expected
        else:
            raise ValueError(f"unsupported validity operation: {operation!r}")
        if not passed:
            return False
    return True


def _nested_get(value: Any, path: str, *, default: Any = _MISSING) -> Any:
    current = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            if default is not _MISSING:
                return default
            raise KeyError(f"metadata field {path!r} is missing")
        current = current[key]
    return current


def _stable_id(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _sorted_values(values: Iterable[Any]) -> list[Any]:
    unique = {json.dumps(value, sort_keys=True): value for value in values}
    return [unique[key] for key in sorted(unique)]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"expected boolean metadata value, got {value!r}")
