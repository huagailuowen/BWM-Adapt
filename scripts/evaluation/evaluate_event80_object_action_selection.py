#!/usr/bin/env python3
"""Post-hoc Event80 action evaluation over existing ID/OOD rollouts.

For every environment and target, this program selects an action using only
predicted object endpoints. It then reads the endpoint for that same action in
the ground-truth rollout and reports task success and oracle regret.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from wan_video_action.evaluation.event80_pushbox import track_event80_block
from wan_video_action.evaluation.io import read_video_frames
from wan_video_action.metrics.object_action import RectangleTarget, evaluate_action_choice


SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)
SUPPORT_KEYS = ("support_indices", "support_sample_indices", "support")
QUERY_KEYS = ("query_indices", "query_sample_indices", "queries", "query")
PATH_KEYS = (
    "prediction_path",
    "predicted_video_path",
    "prediction_video",
    "output_video",
    "video_path",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _first_key(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _sample_index(value: Any) -> int:
    if isinstance(value, Mapping):
        for key in ("sample_index", "sample_idx", "index", "episode_index", "episode_id"):
            if key in value:
                return int(value[key])
    return int(value)


def _indices(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        value = [value]
    return [_sample_index(item) for item in value]


def _environment_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            support = _first_key(node, SUPPORT_KEYS)
            query = _first_key(node, QUERY_KEYS)
            if support is not None and query is not None:
                records.append(dict(node))
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    if not records:
        raise ValueError("No support/query environment records found in protocol")
    return records


def _normalise_protocol(path: Path) -> list[dict[str, Any]]:
    raw = _read_json(path) if path.suffix != ".jsonl" else _read_jsonl(path)
    output = []
    for ordinal, record in enumerate(_environment_records(raw)):
        support = _indices(_first_key(record, SUPPORT_KEYS))
        query = _indices(_first_key(record, QUERY_KEYS))
        all_indices = list(dict.fromkeys(support + query))
        output.append(
            {
                "environment_id": str(
                    record.get("environment_id", record.get("env_id", record.get("id", ordinal)))
                ),
                "domain": str(record.get("domain", record.get("split", "unknown"))).lower(),
                "support_indices": support,
                "query_indices": query,
                "candidate_indices": all_indices,
            }
        )
    return output


def _metadata_index(path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for row in _read_jsonl(path):
        index = _sample_index(row)
        result[index] = row
    return result


def _action_id(record: Mapping[str, Any], sample_index: int) -> str:
    for key in ("action_id", "action_index", "action_idx", "action"):
        value = record.get(key)
        if isinstance(value, (str, int, float)):
            return str(value)
    return str(sample_index)


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, Sequence):
        for child in value:
            yield from _flatten_strings(child)


def _resolve_path(value: str, roots: Sequence[Path]) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    for root in roots:
        candidate = root / value
        if candidate.is_file():
            return candidate.resolve()
    return None


def _ground_truth_path(
    record: Mapping[str, Any], metadata_path: Path, dataset_root: Path | None
) -> Path | None:
    roots = [metadata_path.parent]
    if dataset_root is not None:
        roots.insert(0, dataset_root)
    paths = []
    for value in _flatten_strings(record):
        if value.lower().endswith((".mp4", ".avi", ".mov")):
            resolved = _resolve_path(value, roots)
            if resolved is not None:
                paths.append(resolved)
    if not paths:
        return None

    def score(path: Path) -> tuple[int, str]:
        name = str(path).lower()
        rank = 0
        rank += 4 if any(token in name for token in ("agentview", "main", "front")) else 0
        rank -= 6 if any(token in name for token in ("wrist", "eye_in_hand")) else 0
        return (-rank, name)

    return sorted(set(paths), key=score)[0]


def _path_from_result(record: Mapping[str, Any], result_path: Path, roots: Sequence[Path]) -> Path | None:
    for key in PATH_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            resolved = _resolve_path(value, [result_path.parent, *roots])
            if resolved is not None:
                return resolved
    return None


def _prediction_index(method_root: Path, wanted: set[int]) -> dict[int, Path]:
    found: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for result_path in method_root.rglob("results.jsonl"):
        for row in _read_jsonl(result_path):
            try:
                index = _sample_index(row)
            except (KeyError, TypeError, ValueError):
                continue
            if index not in wanted:
                continue
            path = _path_from_result(row, result_path, [method_root])
            if path is not None:
                found[index].append((20, path))

    for path in method_root.rglob("*.mp4"):
        match = SAMPLE_RE.search(path.name)
        if match is None or int(match.group(1)) not in wanted:
            continue
        lower = str(path).lower()
        if any(token in lower for token in ("comparison", "grid", "contact_sheet", "ground_truth")):
            continue
        rank = 0
        rank += 8 if "prediction" in lower or "predictions" in lower else 0
        rank += 3 if "pred" in path.name.lower() else 0
        rank -= 8 if "support" in lower else 0
        found[int(match.group(1))].append((rank, path.resolve()))

    output = {}
    for index, matches in found.items():
        matches.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
        output[index] = matches[0][1]
    return output


def _coerce_centers(value: Any) -> np.ndarray | None:
    candidates = []
    if isinstance(value, Mapping):
        for key in ("centers", "centroids", "center_xy", "centroid_xy"):
            if key in value:
                candidates.append(value[key])
    else:
        for key in ("centers", "centroids", "center_xy", "centroid_xy"):
            if hasattr(value, key):
                candidates.append(getattr(value, key))
        if isinstance(value, Sequence):
            candidates.extend(value)

    for candidate in candidates:
        try:
            rows = []
            for point in candidate:
                if point is None or len(point) < 2:
                    rows.append((np.nan, np.nan))
                else:
                    rows.append((float(point[0]), float(point[1])))
            array = np.asarray(rows, dtype=np.float64)
            if array.ndim == 2 and array.shape[1] == 2:
                return array
        except (TypeError, ValueError, IndexError):
            continue
    return None


def _track_endpoint(task: tuple[str, int, int]) -> tuple[str, dict[str, Any]]:
    path_value, main_view_width, terminal_window = task
    path = Path(path_value)
    frames = read_video_frames(path)
    if isinstance(frames, tuple):
        frames = frames[0]
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Unexpected video tensor shape {frames.shape} for {path}")
    if main_view_width > 0 and frames.shape[2] > main_view_width:
        frames = frames[:, :, :main_view_width]
    height, width = int(frames.shape[1]), int(frames.shape[2])
    centers = _coerce_centers(track_event80_block(frames))
    if centers is None or not len(centers):
        return path_value, {"xy": None, "valid": False, "width": width, "height": height}
    tail = centers[-max(1, terminal_window) :]
    valid = np.isfinite(tail).all(axis=1)
    if not valid.any():
        valid_all = np.isfinite(centers).all(axis=1)
        if not valid_all.any():
            return path_value, {"xy": None, "valid": False, "width": width, "height": height}
        tail = centers[valid_all][-1:]
    else:
        tail = tail[valid]
    x, y = np.median(tail, axis=0)
    return path_value, {
        "xy": [float(x / width), float(y / height)],
        "xy_px": [float(x), float(y)],
        "valid": True,
        "width": width,
        "height": height,
    }


def _track_paths(paths: Iterable[Path], width: int, window: int, workers: int) -> dict[str, dict[str, Any]]:
    unique = sorted({str(path.resolve()) for path in paths if path is not None})
    tasks = [(path, width, window) for path in unique]
    if workers <= 1:
        rows = map(_track_endpoint, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = pool.map(_track_endpoint, tasks)
    return dict(rows)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return fmean(values) if values else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    protocol_valid = [row for row in valid if row.get("candidate_set_complete")]
    reachable = [row for row in protocol_valid if row.get("oracle_reachable")]

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "task_success_rate": _mean(group, "task_success"),
            "selected_is_oracle_rate": _mean(group, "selected_is_oracle"),
            "mean_regret": _mean(group, "regret"),
            "mean_action_coverage": _mean(group, "action_coverage"),
        }

    grouped: dict[str, Any] = {}
    keys = sorted({(row["domain"], row["target"]["id"]) for row in valid})
    for domain, target_id in keys:
        group = [
            row
            for row in valid
            if row["domain"] == domain and row["target"]["id"] == target_id
        ]
        grouped[f"{domain}/{target_id}"] = metrics(group)
    return {
        "all_valid_decisions": metrics(valid),
        "complete_action_set_decisions": metrics(protocol_valid),
        "headline_oracle_reachable_complete_decisions": metrics(reachable),
        "incomplete_prediction_decisions": len(valid) - len(protocol_valid),
        "by_domain_and_target": grouped,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.target_config.read_text(encoding="utf-8"))
    targets = [RectangleTarget.from_mapping(value) for value in config["targets"]]
    configured_expected = config.get("expected_action_count")
    expected = None if configured_expected is None else int(configured_expected)
    width = int(config.get("main_view_width", 224))
    terminal_window = int(config.get("terminal_window", 5))

    protocol = _normalise_protocol(args.protocol)
    metadata = _metadata_index(args.metadata_jsonl)
    wanted = {index for env in protocol for index in env["candidate_indices"]}
    missing_metadata = sorted(wanted - set(metadata))
    if missing_metadata:
        raise KeyError(f"Missing metadata for sample indices: {missing_metadata}")

    methods_root = args.benchmark_root / "methods"
    methods = args.method or sorted(path.name for path in methods_root.iterdir() if path.is_dir())
    output_root = args.output_root or args.benchmark_root / "metrics" / "object_action_selection"

    gt_paths = {
        index: _ground_truth_path(metadata[index], args.metadata_jsonl, args.dataset_root)
        for index in wanted
    }
    gt_tracks = _track_paths(
        (path for path in gt_paths.values() if path is not None), width, terminal_window, args.workers
    )

    benchmark_summary: dict[str, Any] = {}
    for method in methods:
        method_root = methods_root / method
        predictions = _prediction_index(method_root, wanted)
        pred_tracks = _track_paths(predictions.values(), width, terminal_window, args.workers)
        candidates_by_env: dict[str, list[dict[str, Any]]] = {}
        decisions: list[dict[str, Any]] = []

        for env in protocol:
            candidates = []
            for index in env["candidate_indices"]:
                pred_path = predictions.get(index)
                gt_path = gt_paths.get(index)
                pred_track = None if pred_path is None else pred_tracks.get(str(pred_path.resolve()))
                gt_track = None if gt_path is None else gt_tracks.get(str(gt_path.resolve()))
                is_support = index in env["support_indices"]
                if is_support and gt_track is not None:
                    selection_xy = gt_track.get("xy")
                    selection_source = "observed_support"
                else:
                    selection_xy = None if pred_track is None else pred_track.get("xy")
                    selection_source = "model_prediction"
                candidates.append(
                    {
                        "environment_id": env["environment_id"],
                        "domain": env["domain"],
                        "sample_index": index,
                        "action_id": _action_id(metadata[index], index),
                        "is_support": is_support,
                        "selection_source": selection_source,
                        "prediction_path": None if pred_path is None else str(pred_path),
                        "ground_truth_path": None if gt_path is None else str(gt_path),
                        "selection_xy": selection_xy,
                        "model_predicted_xy": None if pred_track is None else pred_track.get("xy"),
                        "ground_truth_xy": None if gt_track is None else gt_track.get("xy"),
                    }
                )
            candidates_by_env[env["environment_id"]] = candidates
            for target in targets:
                decision = evaluate_action_choice(
                    candidates, target, expected_action_count=expected
                )
                decision.update(
                    {
                        "method": method,
                        "environment_id": env["environment_id"],
                        "domain": env["domain"],
                    }
                )
                decisions.append(decision)

        method_output = output_root / "methods" / method
        candidate_rows = [row for rows in candidates_by_env.values() for row in rows]
        _write_jsonl(method_output / "candidate_outcomes.jsonl", candidate_rows)
        _write_jsonl(method_output / "decisions.jsonl", decisions)
        summary = _summary(decisions)
        _write_json(method_output / "summary.json", summary)
        benchmark_summary[method] = summary

    _write_json(
        output_root / "protocol.json",
        {
            "source_protocol": str(args.protocol.resolve()),
            "target_config": str(args.target_config.resolve()),
            "selection_uses_ground_truth": False,
            "observed_support_is_a_selectable_candidate": True,
            "ground_truth_is_read_after_selection": True,
            "expected_action_count": expected,
            "methods": methods,
            "targets": [target.as_dict() for target in targets],
        },
    )
    _write_json(output_root / "summary.json", benchmark_summary)


if __name__ == "__main__":
    main()
