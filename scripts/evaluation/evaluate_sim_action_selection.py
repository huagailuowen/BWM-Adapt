#!/usr/bin/env python3
"""Evaluate one sim task/method from an existing same-environment transfer plan."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.io import read_video_frames
from wan_video_action.evaluation.sim_task_extractors import extract_sim_task_state
from wan_video_action.evaluation.task_state import load_task_state, save_task_state
from wan_video_action.metrics.task_action_selection import TaskTarget, evaluate_task_action_choice


SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)
SOURCE_RE = re.compile(r"source[_-]?(\d+)", re.IGNORECASE)
OBJECT_OUTCOME_TASKS = {
    "gravity",
    "mass_collision",
    "collision",
    "mass_friction",
    "joint_mass_friction",
}
DEFAULT_OBJECT_ROLES = {
    "gravity": {"object": 0},
    "mass_collision": {
        "struck_object": 0,
        "target": 0,
        "striker": 1,
        "projectile": 1,
    },
    "collision": {
        "struck_object": 0,
        "target": 0,
        "striker": 1,
        "projectile": 1,
    },
    "mass_friction": {
        "struck_object": 0,
        "target": 0,
        "striker": 1,
        "projectile": 1,
        "driver": 1,
    },
    "joint_mass_friction": {
        "struck_object": 0,
        "target": 0,
        "striker": 1,
        "projectile": 1,
        "driver": 1,
    },
}


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_sample_index"] = int(row.get("sample_index", ordinal))
        rows.append(row)
    by_index = {int(row["_sample_index"]): row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError(f"Duplicate sample indices in {path}.")
    maximum = max(by_index, default=-1)
    output = [None] * (maximum + 1)
    for index, row in by_index.items():
        output[index] = row
    return output


def _video_path(row: Mapping[str, Any], dataset_root: Path) -> Path:
    values = row.get("video", [])
    if isinstance(values, str):
        values = [values]
    preferred = [value for value in values if "wrist" not in str(value).lower()]
    for value in [*preferred, *values]:
        path = Path(value)
        candidate = path if path.is_absolute() else dataset_root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No main-camera video for sample {row.get('_sample_index')}.")


def _gt_frames(row: Mapping[str, Any], dataset_root: Path) -> np.ndarray:
    frames = np.asarray(read_video_frames(_video_path(row, dataset_root)))
    start = int(row.get("start_frame", 0))
    stride = int(row.get("frame_stride", 1))
    length = int(row.get("length", len(frames)))
    indices = start + np.arange(length, dtype=np.int64) * stride
    indices = np.clip(indices, 0, len(frames) - 1)
    return frames[indices]


def _prediction_index(plan_root: Path) -> dict[tuple[int, int], Path]:
    choices: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
    for path in plan_root.rglob("*.mp4"):
        # Restrict filtering to the transfer-plan subtree.  Experiment roots can
        # legitimately contain tokens such as ``gt_stage1_stage2`` in their
        # names; inspecting the absolute path would then discard every raw
        # prediction below that root.
        lower = path.relative_to(plan_root).as_posix().lower()
        if any(token in lower for token in ("/grids/", "comparison", "audit", "gt_stage")):
            continue
        sample_match, source_match = SAMPLE_RE.search(path.name), SOURCE_RE.search(lower)
        if sample_match is None or source_match is None:
            continue
        rank = 0
        rank += 8 if "stage2_adapted_raw" in lower else 0
        rank += 5 if "/raw/" in lower else 0
        choices[(int(source_match.group(1)), int(sample_match.group(1)))].append(
            (rank, path.resolve())
        )
    result = {}
    for key, values in choices.items():
        values.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
        result[key] = values[0][1]
    return result


def _state(
    task: str,
    frames: np.ndarray,
    cache_path: Path,
    extractor: Mapping[str, Any],
):
    if cache_path.is_file():
        return load_task_state(cache_path)
    state = extract_sim_task_state(
        task,
        frames,
        fps=float(extractor.get("fps", 10.0)),
        main_view_width=int(extractor.get("main_view_width", 224)),
        min_area=int(extractor.get("min_area", 8)),
        max_area=int(extractor.get("max_area", 3000)),
        edge_margin=int(extractor.get("edge_margin", 16)),
        light_roi=tuple(extractor.get("light_roi", (98, 108, 151, 166))),
        yellow_threshold=float(extractor.get("yellow_threshold", 0.35)),
    )
    save_task_state(cache_path, state)
    return state


def _terminal_indices(length: int, window: int) -> np.ndarray:
    return np.arange(max(0, length - max(1, window)), length)


def _resolved_targets(
    task: str,
    outcome_settings: Mapping[str, Any],
    values: Sequence[Mapping[str, Any]],
) -> list[TaskTarget]:
    name = task.lower().replace("-", "_")
    roles = dict(DEFAULT_OBJECT_ROLES.get(name, {}))
    roles.update(
        {
            str(role): int(index)
            for role, index in outcome_settings.get("object_roles", {}).items()
        }
    )
    default_index = int(outcome_settings.get("object_index", 0))
    output = []
    for raw in values:
        value = dict(raw)
        role = value.get("object_role")
        explicit_index = value.get("object_index")
        if role is not None:
            role = str(role)
            if role not in roles:
                raise ValueError(
                    f"Unknown object_role {role!r} for task {task!r}; "
                    f"available roles are {sorted(roles)}."
                )
            role_index = int(roles[role])
            if explicit_index is not None and int(explicit_index) != role_index:
                raise ValueError(
                    f"Target {value.get('id')!r} gives object_role={role!r} "
                    f"but conflicting object_index={explicit_index}."
                )
            value["object_index"] = role_index
        elif name in OBJECT_OUTCOME_TASKS:
            value["object_index"] = (
                default_index if explicit_index is None else int(explicit_index)
            )
        output.append(TaskTarget.from_mapping(value))
    return output


def _terminal_object_outcome(state, indices: np.ndarray, object_index: int) -> dict[str, Any]:
    if state.centroids is None or not 0 <= object_index < state.centroids.shape[1]:
        return {"value": None, "details": {"valid": False, "object_index": object_index}}
    points = np.asarray(state.centroids[indices, object_index], dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    if not finite.any():
        return {"value": None, "details": {"valid": False, "object_index": object_index}}
    point = np.median(points[finite], axis=0)
    offscreen = bool(
        state.events
        and "offscreen" in state.events
        and state.events["offscreen"][-1, object_index]
    )
    exit_side = None
    if state.events:
        exit_side = next(
            (
                key.removeprefix("exit_")
                for key, value in state.events.items()
                if key.startswith("exit_") and value[-1, object_index]
            ),
            None,
        )
    return {
        "value": [float(point[0] / state.image_width), float(point[1] / state.image_height)],
        "details": {
            "valid": True,
            "object_index": object_index,
            "offscreen": offscreen,
            "exit_side": exit_side,
        },
    }


def _outcome(task: str, state, settings: Mapping[str, Any]) -> dict[str, Any]:
    window = int(settings.get("terminal_window", 5))
    indices = _terminal_indices(state.frame_count, window)
    name = task.lower().replace("-", "_")
    if name in OBJECT_OUTCOME_TASKS:
        object_index = int(settings.get("object_index", 0))
        count = 0 if state.centroids is None else int(state.centroids.shape[1])
        outcomes = {
            str(index): _terminal_object_outcome(state, indices, index)
            for index in range(count)
        }
        selected = outcomes.get(
            str(object_index),
            {"value": None, "details": {"valid": False, "object_index": object_index}},
        )
        return {
            "value": selected["value"],
            "details": selected["details"],
            "object_values": {
                index: outcome["value"] for index, outcome in outcomes.items()
            },
            "object_details": {
                index: outcome["details"] for index, outcome in outcomes.items()
            },
        }
    if name in {"mass_balance", "balance"}:
        values = np.degrees(np.asarray(state.angles_rad[indices, 0], dtype=np.float64))
        values = values[np.isfinite(values)]
        return {
            "value": float(np.median(values)) if len(values) else None,
            "details": {"valid": bool(len(values)), "unit": "image_axis_degrees"},
        }
    values = np.asarray(state.light_score[indices, 0], dtype=np.float64)
    values = values[np.isfinite(values)]
    score = float(np.median(values)) if len(values) else None
    threshold = float(settings.get("yellow_threshold", 0.35))
    return {
        "value": None if score is None else bool(score >= threshold),
        "details": {"valid": score is not None, "yellow_fraction": score},
    }


def _aggregate_value(values: Sequence[Any], kind: str) -> Any:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    if kind == "point":
        return np.median(np.asarray(valid, dtype=np.float64), axis=0).tolist()
    if kind == "scalar":
        return float(np.median(np.asarray(valid, dtype=np.float64)))
    return bool(fmean(float(bool(value)) for value in valid) >= 0.5)


def _aggregate_object_values(values: Sequence[Any], kind: str) -> dict[str, Any]:
    mappings = [value for value in values if isinstance(value, Mapping)]
    keys = sorted({str(key) for value in mappings for key in value})
    return {
        key: _aggregate_value(
            [value.get(key, value.get(int(key))) for value in mappings],
            kind,
        )
        for key in keys
    }


def _value_kind(task: str) -> str:
    name = task.lower().replace("-", "_")
    if name in {"gravity", "mass_collision", "collision", "mass_friction", "joint_mass_friction"}:
        return "point"
    return "scalar" if name in {"mass_balance", "balance"} else "binary"


def _domain(plan_spec: Mapping[str, Any], plan_path: Path, source_index: int) -> str:
    mapping = {str(key): value for key, value in plan_spec.get("domain_by_source", {}).items()}
    if str(source_index) in mapping:
        return str(mapping[str(source_index)]).lower()
    if "ood" in str(plan_path).lower():
        return "ood"
    return str(plan_spec.get("domain", "id")).lower()


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return fmean(values) if values else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    complete = [row for row in valid if row.get("candidate_set_complete")]
    reachable = [row for row in complete if row.get("oracle_reachable")]

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "task_success_rate": _mean(group, "task_success"),
            "selected_is_oracle_rate": _mean(group, "selected_is_oracle"),
            "mean_regret": _mean(group, "regret"),
            "mean_action_coverage": _mean(group, "action_coverage"),
        }

    grouped = {}
    for domain, target_id in sorted({(row["domain"], row["target"]["id"]) for row in valid}):
        group = [
            row
            for row in valid
            if row["domain"] == domain and row["target"]["id"] == target_id
        ]
        grouped[f"{domain}/{target_id}"] = metrics(group)
    return {
        "all_valid_decisions": metrics(valid),
        "complete_candidate_set_decisions": metrics(complete),
        "headline_oracle_reachable_complete_decisions": metrics(reachable),
        "by_domain_and_target": grouped,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = str(config["task"])
    method = str(config.get("method_name", "unknown"))
    metadata_path = _resolve(config["metadata_jsonl"])
    dataset_root = _resolve(config["dataset_root"])
    output_dir = (args.output_dir or _resolve(config["output_dir"])).resolve()
    extractor = config.get("extractor", {})
    outcome_settings = config.get("outcome", {})
    targets = _resolved_targets(task, outcome_settings, config["targets"])
    rows = _metadata(metadata_path)
    action_field = str(config.get("action_field", "action_id"))
    partition_field = config.get("partition_field")
    eligible_field = config.get("eligible_environment_field")
    eligible_values = {str(value) for value in config.get("eligible_environment_values", [])}
    report_fields = list(config.get("environment_report_fields", []))
    action_report_fields = list(config.get("action_report_fields", []))
    action_order_field = config.get("action_order_field")
    selection_strategy = str(config.get("selection_strategy", "nearest"))
    minimum_actions = int(config.get("minimum_actions", 2))
    value_kind = _value_kind(task)

    candidate_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for plan_value in config["transfer_plans"]:
        plan_spec = {"path": plan_value} if isinstance(plan_value, str) else dict(plan_value)
        plan_path = _resolve(plan_spec["path"])
        predictions = _prediction_index(plan_path.parent)
        for environment in _read_json(plan_path):
            source_index = int(environment["source_index"])
            source_row = rows[source_index]
            if source_row is None:
                raise KeyError(f"Missing metadata sample {source_index}.")
            if eligible_field and eligible_values and str(source_row.get(eligible_field)) not in eligible_values:
                skipped.append(
                    {
                        "source_index": source_index,
                        "reason": "ineligible_environment",
                        "field": eligible_field,
                        "value": source_row.get(eligible_field),
                    }
                )
                continue
            support_indices = [
                int(index)
                for index in environment.get("support_indices", [source_index])
            ]
            if source_index not in support_indices:
                support_indices.insert(0, source_index)
            support_indices = list(dict.fromkeys(support_indices))
            support_index_set = set(support_indices)
            indices = list(
                dict.fromkeys(
                    [*support_indices, *map(int, environment["target_indices"])]
                )
            )
            raw = []
            for index in indices:
                row = rows[index]
                if row is None:
                    raise KeyError(f"Missing metadata sample {index}.")
                gt_cache = output_dir / "states" / "sim_rgb_v1" / "ground_truth" / f"sample{index:04d}.npz"
                gt_state = _state(task, _gt_frames(row, dataset_root), gt_cache, extractor)
                gt_outcome = _outcome(task, gt_state, outcome_settings)
                is_support = index in support_index_set
                prediction_path = None if is_support else predictions.get((source_index, index))
                if is_support:
                    selection_outcome = gt_outcome
                    selection_source = "observed_support"
                elif prediction_path is not None:
                    pred_cache = (
                        output_dir
                        / "states"
                        / "sim_rgb_v1"
                        / "prediction"
                        / f"source{source_index:04d}_sample{index:04d}.npz"
                    )
                    pred_state = _state(
                        task,
                        np.asarray(read_video_frames(prediction_path)),
                        pred_cache,
                        extractor,
                    )
                    selection_outcome = _outcome(task, pred_state, outcome_settings)
                    selection_source = "model_prediction"
                else:
                    selection_outcome = {
                        "value": None,
                        "details": {"valid": False},
                        "object_values": {},
                        "object_details": {},
                    }
                    selection_source = "missing_prediction"
                raw.append(
                    {
                        "sample_index": index,
                        "is_support": is_support,
                        "partition": row.get(partition_field) if partition_field else "all",
                        "action_id": str(row.get(action_field, row.get("action_id", index))),
                        "action_order": (
                            row.get(action_order_field) if action_order_field else None
                        ),
                        "selection_source": selection_source,
                        "selection_value": selection_outcome["value"],
                        "selection_details": selection_outcome["details"],
                        "selection_object_values": selection_outcome.get(
                            "object_values", {}
                        ),
                        "selection_object_details": selection_outcome.get(
                            "object_details", {}
                        ),
                        "ground_truth_value": gt_outcome["value"],
                        "ground_truth_details": gt_outcome["details"],
                        "ground_truth_object_values": gt_outcome.get(
                            "object_values", {}
                        ),
                        "ground_truth_object_details": gt_outcome.get(
                            "object_details", {}
                        ),
                        "prediction_path": None if prediction_path is None else str(prediction_path),
                        "ground_truth_path": str(_video_path(row, dataset_root)),
                        "action_metadata": {field: row.get(field) for field in action_report_fields},
                    }
                )

            partitions = sorted({str(record["partition"]) for record in raw if not record["is_support"]})
            if not partitions:
                partitions = ["all"]
            for partition in partitions:
                subset = [
                    record
                    for record in raw
                    if str(record["partition"]) == partition
                    or (record["is_support"] and str(record["partition"]) == partition)
                ]
                by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for record in subset:
                    by_action[record["action_id"]].append(record)
                aggregated = []
                for action_id, records_for_action in sorted(by_action.items()):
                    model_records = [
                        record
                        for record in records_for_action
                        if record["selection_source"] == "model_prediction"
                        and record["selection_value"] is not None
                    ]
                    selection_records = model_records or [
                        record
                        for record in records_for_action
                        if record["selection_source"] == "observed_support"
                    ]
                    query_gt = [record for record in records_for_action if not record["is_support"]]
                    gt_records = query_gt or records_for_action
                    aggregate = {
                        "action_id": action_id,
                        "sample_indices": [record["sample_index"] for record in records_for_action],
                        "selection_source": (
                            "model_prediction" if model_records else "observed_support"
                        ),
                        "selection_value": _aggregate_value(
                            [record["selection_value"] for record in selection_records], value_kind
                        ),
                        "ground_truth_value": _aggregate_value(
                            [record["ground_truth_value"] for record in gt_records], value_kind
                        ),
                        "selection_object_values": _aggregate_object_values(
                            [
                                record["selection_object_values"]
                                for record in selection_records
                            ],
                            value_kind,
                        ),
                        "ground_truth_object_values": _aggregate_object_values(
                            [
                                record["ground_truth_object_values"]
                                for record in gt_records
                            ],
                            value_kind,
                        ),
                        "action_order": _aggregate_value(
                            [record["action_order"] for record in records_for_action],
                            "scalar",
                        ) if action_order_field else None,
                        "member_records": records_for_action,
                    }
                    aggregated.append(aggregate)
                    candidate_rows.append(
                        {
                            "method": method,
                            "task": task,
                            "source_index": source_index,
                            "environment_id": (
                                f"source{source_index:04d}"
                                if partition == "all"
                                else f"source{source_index:04d}:{partition_field}={partition}"
                            ),
                            "domain": _domain(plan_spec, plan_path, source_index),
                            "partition": partition,
                            **aggregate,
                        }
                    )
                if len(aggregated) < minimum_actions:
                    skipped.append(
                        {
                            "source_index": source_index,
                            "partition": partition,
                            "reason": "too_few_actions",
                            "action_count": len(aggregated),
                        }
                    )
                    continue
                environment_id = (
                    f"source{source_index:04d}"
                    if partition == "all"
                    else f"source{source_index:04d}:{partition_field}={partition}"
                )
                for target in targets:
                    decision = evaluate_task_action_choice(
                        aggregated,
                        target,
                        selection_strategy=selection_strategy,
                    )
                    decision.update(
                        {
                            "method": method,
                            "task": task,
                            "environment_id": environment_id,
                            "source_index": source_index,
                            "partition": partition,
                            "domain": _domain(plan_spec, plan_path, source_index),
                            "environment_metadata": {
                                field: source_row.get(field) for field in report_fields
                            },
                        }
                    )
                    decisions.append(decision)

    _write_jsonl(output_dir / "candidate_outcomes.jsonl", candidate_rows)
    _write_jsonl(output_dir / "decisions.jsonl", decisions)
    _write_json(output_dir / "summary.json", _summary(decisions))
    _write_jsonl(output_dir / "skipped.jsonl", skipped)
    _write_json(
        output_dir / "protocol.json",
        {
            "config": str(config_path),
            "task": task,
            "method": method,
            "selection_uses_query_ground_truth": False,
            "observed_support_is_selectable": True,
            "candidate_count_is_protocol_defined": True,
            "selection_strategy": selection_strategy,
            "action_order_field": action_order_field,
            "targets": [target.as_dict() for target in targets],
            "transfer_plans": config["transfer_plans"],
        },
    )


if __name__ == "__main__":
    main()
