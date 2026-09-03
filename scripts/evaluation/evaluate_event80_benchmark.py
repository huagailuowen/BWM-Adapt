#!/usr/bin/env python3
"""Evaluate all Event80 methods under one frozen support/query protocol.

This evaluator computes global appearance, object-centroid, and action-choice
metrics from the same decoded rollouts. Query ground truth is never used to
select an action: it is read only after the predicted action has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.event80_pushbox import (  # noqa: E402
    Event80TrackerConfig,
    track_event80_block,
)
from wan_video_action.evaluation.multibackground_pushbox import (  # noqa: E402
    MultiBackgroundTrackerConfig,
    track_multibackground_block,
)
from wan_video_action.evaluation.io import read_video_frames  # noqa: E402
from wan_video_action.metrics.global_video import (  # noqa: E402
    LPIPSEvaluator,
    frame_psnr,
    frame_ssim,
)
from wan_video_action.metrics.object_action import (  # noqa: E402
    RectangleTarget,
    evaluate_action_choice,
)


SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)
VIDEO_METRICS = (
    "psnr_main",
    "psnr_wrist",
    "psnr_multiview",
    "ssim_main",
    "ssim_wrist",
    "ssim_multiview",
    "lpips_multiview",
    "centroid_ade_px",
    "centroid_fde_px",
    "centroid_ade_normalized",
    "centroid_fde_normalized",
    "centroid_prediction_missing_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def as_frames(value: Any) -> np.ndarray:
    frames = value[0] if isinstance(value, tuple) else value
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video [T,H,W,3], got {frames.shape}")
    return frames


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames.shape[1:3] == (height, width):
        return frames
    try:
        import cv2

        return np.stack([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            for frame in frames
        ])
    except ImportError:
        from PIL import Image

        return np.stack([
            np.asarray(
                Image.fromarray(frame).resize(
                    (width, height), resample=Image.Resampling.BICUBIC
                )
            )
            for frame in frames
        ])


def float_frames(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) / 255.0


def terminal_xy(centers: np.ndarray, *, width: int, height: int, window: int) -> list[float] | None:
    tail = centers[-max(1, window) :]
    valid = np.isfinite(tail).all(axis=1)
    if not valid.any():
        valid_all = np.isfinite(centers).all(axis=1)
        if not valid_all.any():
            return None
        tail = centers[valid_all][-1:]
    else:
        tail = tail[valid]
    x, y = np.median(tail, axis=0)
    return [float(x / width), float(y / height)]


def mean_rows(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        output[name] = fmean(values) if values else None
    return output


def prediction_index(method_root: Path, wanted: set[int]) -> dict[int, Path]:
    prediction_root = method_root / "predictions"
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Missing prediction directory: {prediction_root}")
    found: dict[int, list[Path]] = {}
    for path in prediction_root.glob("*.mp4"):
        match = SAMPLE_RE.search(path.name)
        if match is None:
            continue
        sample_index = int(match.group(1))
        if sample_index in wanted:
            found.setdefault(sample_index, []).append(path.resolve())
    duplicates = {index: paths for index, paths in found.items() if len(paths) != 1}
    if duplicates:
        raise ValueError(f"Duplicate predictions under {prediction_root}: {duplicates}")
    missing = sorted(wanted - set(found))
    if missing:
        raise FileNotFoundError(
            f"{method_root} is missing {len(missing)} query predictions: {missing}"
        )
    return {index: paths[0] for index, paths in found.items()}


def load_ground_truth(
    record: Mapping[str, Any],
    dataset_root: Path,
    tracker,
    tracker_config,
    terminal_window: int,
) -> dict[str, Any]:
    video_paths = [dataset_root / value for value in record["video"]]
    if len(video_paths) != 2:
        raise ValueError(f"Expected main and wrist videos for {record['sample_id']}")
    start = int(record["start_frame"])
    count = int(record["length"])
    main = as_frames(read_video_frames(video_paths[0], start_frame=start, num_frames=count))
    wrist = as_frames(read_video_frames(video_paths[1], start_frame=start, num_frames=count))
    count = min(count, len(main), len(wrist))
    main = main[:count]
    wrist = wrist[:count]
    track = tracker(main, tracker_config)
    height, width = main.shape[1:3]
    return {
        "main": main,
        "wrist": wrist,
        "centers": track.centers,
        "endpoint_xy": terminal_xy(
            track.centers, width=width, height=height, window=terminal_window
        ),
        "paths": [str(path.resolve()) for path in video_paths],
        "start_frame": start,
        "num_frames": count,
    }


def environment_mu_index(environment: Mapping[str, Any]) -> int:
    for key in ("mu_index", "friction_index", "context_group_id"):
        value = environment.get(key)
        if value is not None:
            return int(value)
    suffix = str(environment["environment_id"]).rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError as error:
        raise KeyError(
            f"Cannot determine friction index for {environment['environment_id']!r}"
        ) from error


def evaluate_prediction(
    *,
    method: str,
    environment: Mapping[str, Any],
    sample_index: int,
    metadata: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    prediction_path: Path,
    tracker,
    tracker_config,
    terminal_window: int,
    future_start: int,
    lpips_evaluator: LPIPSEvaluator | None,
    lpips_batch_size: int,
) -> tuple[dict[str, Any], list[float] | None]:
    gt_main = ground_truth["main"]
    gt_wrist = ground_truth["wrist"]
    pred_full = as_frames(read_video_frames(prediction_path))
    count = min(len(gt_main), len(gt_wrist), len(pred_full))
    if count <= future_start:
        raise ValueError(f"No future frames for {prediction_path}")

    main_width = gt_main.shape[2]
    wrist_width = gt_wrist.shape[2]
    required_width = main_width + wrist_width
    if pred_full.shape[2] < required_width:
        raise ValueError(
            f"Prediction {prediction_path} has width {pred_full.shape[2]}, "
            f"but the two GT views require at least {required_width}"
        )
    pred_main = resize_frames(
        pred_full[:count, :, :main_width], gt_main.shape[1], main_width
    )
    pred_wrist = resize_frames(
        pred_full[:count, :, main_width:required_width], gt_wrist.shape[1], wrist_width
    )
    gt_main = gt_main[:count]
    gt_wrist = gt_wrist[:count]

    evaluation_slice = slice(future_start, count)
    gt_main_float = float_frames(gt_main[evaluation_slice])
    pred_main_float = float_frames(pred_main[evaluation_slice])
    gt_wrist_float = float_frames(gt_wrist[evaluation_slice])
    pred_wrist_float = float_frames(pred_wrist[evaluation_slice])
    gt_multiview = np.concatenate((gt_main_float, gt_wrist_float), axis=2)
    pred_multiview = np.concatenate((pred_main_float, pred_wrist_float), axis=2)

    gt_centers = np.asarray(ground_truth["centers"][:count], dtype=np.float64)
    pred_centers = tracker(pred_main, tracker_config).centers
    height, width = gt_main.shape[1:3]
    diagonal = float(np.hypot(height, width))
    centroid_errors: list[float] = []
    missing = 0
    excluded_gt = 0
    for frame_index in range(future_start, count):
        gt_center = gt_centers[frame_index]
        if not np.isfinite(gt_center).all():
            excluded_gt += 1
            continue
        pred_center = pred_centers[frame_index]
        if np.isfinite(pred_center).all():
            centroid_errors.append(float(np.linalg.norm(gt_center - pred_center)))
        else:
            missing += 1
            centroid_errors.append(diagonal)
    if not centroid_errors:
        raise ValueError(f"No trackable GT object frames for sample {sample_index}")

    row = {
        "method": method,
        "sample_index": sample_index,
        "sample_id": metadata["sample_id"],
        "action_id": int(metadata["action_id"]),
        "environment_id": environment["environment_id"],
        "mu_index": environment_mu_index(environment),
        "friction_mu": float(environment["friction_mu"]),
        "domain": environment["domain"],
        "support_size": 1,
        "prediction_path": str(prediction_path),
        "ground_truth_paths": ground_truth["paths"],
        "gt_start_frame": int(ground_truth["start_frame"]),
        "pred_start_frame": 0,
        "evaluated_frames": count - future_start,
        "object_valid_gt_frames": len(centroid_errors),
        "object_excluded_gt_frames": excluded_gt,
        "psnr_main": float(np.mean(frame_psnr(gt_main_float, pred_main_float))),
        "psnr_wrist": float(np.mean(frame_psnr(gt_wrist_float, pred_wrist_float))),
        "psnr_multiview": float(np.mean(frame_psnr(gt_multiview, pred_multiview))),
        "ssim_main": float(np.mean(frame_ssim(gt_main_float, pred_main_float))),
        "ssim_wrist": float(np.mean(frame_ssim(gt_wrist_float, pred_wrist_float))),
        "ssim_multiview": float(np.mean(frame_ssim(gt_multiview, pred_multiview))),
        "lpips_multiview": (
            None
            if lpips_evaluator is None
            else float(np.mean(lpips_evaluator(
                gt_multiview, pred_multiview, batch_size=lpips_batch_size
            )))
        ),
        "centroid_ade_px": float(np.mean(centroid_errors)),
        "centroid_fde_px": float(centroid_errors[-1]),
        "centroid_ade_normalized": float(np.mean(centroid_errors) / diagonal),
        "centroid_fde_normalized": float(centroid_errors[-1] / diagonal),
        "centroid_prediction_missing_rate": float(missing / len(centroid_errors)),
    }
    endpoint = terminal_xy(
        pred_centers, width=width, height=height, window=terminal_window
    )
    return row, endpoint


def aggregate_video_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    environments: list[dict[str, Any]] = []
    for environment_id in sorted({row["environment_id"] for row in rows}):
        group = [row for row in rows if row["environment_id"] == environment_id]
        environments.append({
            "environment_id": environment_id,
            "domain": group[0]["domain"],
            "mu_index": group[0]["mu_index"],
            "friction_mu": group[0]["friction_mu"],
            "query_count": len(group),
            **mean_rows(group, VIDEO_METRICS),
        })
    domains: dict[str, Any] = {}
    for domain in ("id", "ood"):
        group = [row for row in environments if row["domain"] == domain]
        domains[domain] = {
            "environment_count": len(group),
            "query_count": sum(int(row["query_count"]) for row in group),
            **mean_rows(group, VIDEO_METRICS),
        }
    return {
        "aggregation": "query mean within environment, then equal-weight environment mean",
        "query_count": len(rows),
        "environment_count": len(environments),
        "overall": mean_rows(environments, VIDEO_METRICS),
        "by_domain": domains,
        "per_environment": environments,
    }


def action_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    complete = [row for row in valid if row.get("candidate_set_complete")]
    eligible = [row for row in complete if row.get("oracle_reachable")]
    random_rates = [
        len(row.get("oracle_feasible_action_ids", []))
        / int(row["protocol_action_count"])
        for row in eligible
        if int(row.get("protocol_action_count", 0)) > 0
    ]
    return {
        "decision_count": len(rows),
        "valid_decision_count": len(valid),
        "complete_decision_count": len(complete),
        "oracle_reachable_complete_count": len(eligible),
        "headline_task_success_rate": (
            fmean(float(row["task_success"]) for row in eligible) if eligible else None
        ),
        "selected_is_oracle_rate": (
            fmean(float(row["selected_is_oracle"]) for row in eligible) if eligible else None
        ),
        "mean_regret": (
            fmean(float(row["regret"]) for row in eligible if row.get("regret") is not None)
            if any(row.get("regret") is not None for row in eligible)
            else None
        ),
        "mean_action_coverage": (
            fmean(float(row["action_coverage"]) for row in valid) if valid else None
        ),
        "uniform_random_success_rate": (
            fmean(random_rates) if random_rates else None
        ),
    }


def aggregate_action_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain = {
        domain: action_statistics([row for row in rows if row["domain"] == domain])
        for domain in ("id", "ood")
    }
    by_target = {
        target_id: action_statistics([
            row for row in rows if row["target"]["id"] == target_id
        ])
        for target_id in sorted({row["target"]["id"] for row in rows})
    }
    by_domain_target = {
        f"{domain}/{target_id}": action_statistics([
            row
            for row in rows
            if row["domain"] == domain and row["target"]["id"] == target_id
        ])
        for domain in ("id", "ood")
        for target_id in sorted({row["target"]["id"] for row in rows})
    }
    return {
        "selection_protocol": (
            "observed support endpoint plus model-predicted query endpoints; "
            "query GT is read only after action selection"
        ),
        "headline_subset": "complete candidate sets with a GT-reachable target",
        "overall": action_statistics(rows),
        "by_domain": by_domain,
        "by_target": by_target,
        "by_domain_and_target": by_domain_target,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    benchmark_root = resolve_path(config["benchmark_root"])
    protocol_path = resolve_path(config["protocol_path"])
    metadata_path = resolve_path(config["metadata_path"])
    dataset_root = resolve_path(config["dataset_root"])
    output_root = resolve_path(config["output_root"])
    target_config_path = resolve_path(config["action_target_config"])

    metric_config = config.get("metrics", {})
    lpips_config = metric_config.get("lpips", {})
    use_lpips = bool(lpips_config.get("enabled", True))
    if use_lpips and bool(config.get("require_slurm_for_lpips", True)):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("LPIPS is restricted to a Slurm compute allocation")
        required_partition = config.get("required_slurm_partition")
        actual_partition = os.environ.get("SLURM_JOB_PARTITION")
        if required_partition and actual_partition != required_partition:
            raise RuntimeError(
                f"LPIPS requires partition {required_partition!r}, got {actual_partition!r}"
            )

    protocol = read_json(protocol_path)
    environments = list(protocol["environments"])
    metadata = {int(row["episode_index"]): row for row in read_jsonl(metadata_path)}
    query_indices = {
        int(index) for environment in environments for index in environment["query_indices"]
    }
    candidate_indices = {
        int(index)
        for environment in environments
        for index in environment["support_indices"] + environment["query_indices"]
    }
    missing_metadata = sorted(candidate_indices - set(metadata))
    if missing_metadata:
        raise KeyError(f"Missing metadata for samples: {missing_metadata}")

    tracker_type = str(config.get("tracker_type", "event80")).strip().lower()
    if tracker_type == "event80":
        tracker = track_event80_block
        tracker_config = Event80TrackerConfig(**config.get("tracker", {}))
    elif tracker_type == "multi_background":
        tracker = track_multibackground_block
        tracker_config = MultiBackgroundTrackerConfig(**config.get("tracker", {}))
    else:
        raise ValueError(f"Unknown PushBox tracker type: {tracker_type!r}")
    terminal_window = int(config.get("terminal_window", 5))
    future_start = int(config.get("future_start_frame", 1))
    ground_truth = {
        index: load_ground_truth(
            metadata[index], dataset_root, tracker, tracker_config, terminal_window
        )
        for index in sorted(candidate_indices)
    }

    lpips_evaluator = None
    if use_lpips:
        lpips_evaluator = LPIPSEvaluator(
            net=str(lpips_config.get("network", "alex")),
            device=str(lpips_config.get("device", "cuda")),
        )
    lpips_batch_size = int(lpips_config.get("batch_size", 8))
    target_config = yaml.safe_load(target_config_path.read_text(encoding="utf-8"))
    targets = [RectangleTarget.from_mapping(value) for value in target_config["targets"]]

    scoreboard: list[dict[str, Any]] = []
    benchmark_summary: dict[str, Any] = {}
    for method, relative_method_root in config["methods"].items():
        method_root = benchmark_root / relative_method_root
        predictions = prediction_index(method_root, query_indices)
        query_rows: list[dict[str, Any]] = []
        predicted_endpoints: dict[int, list[float] | None] = {}
        for environment in environments:
            for value in environment["query_indices"]:
                sample_index = int(value)
                row, endpoint = evaluate_prediction(
                    method=method,
                    environment=environment,
                    sample_index=sample_index,
                    metadata=metadata[sample_index],
                    ground_truth=ground_truth[sample_index],
                    prediction_path=predictions[sample_index],
                    tracker=tracker,
                    tracker_config=tracker_config,
                    terminal_window=terminal_window,
                    future_start=future_start,
                    lpips_evaluator=lpips_evaluator,
                    lpips_batch_size=lpips_batch_size,
                )
                query_rows.append(row)
                predicted_endpoints[sample_index] = endpoint

        video_summary = aggregate_video_rows(query_rows)
        candidate_rows: list[dict[str, Any]] = []
        decision_rows: list[dict[str, Any]] = []
        for environment in environments:
            support_indices = {int(index) for index in environment["support_indices"]}
            candidates: list[dict[str, Any]] = []
            for value in environment["support_indices"] + environment["query_indices"]:
                sample_index = int(value)
                is_support = sample_index in support_indices
                candidate = {
                    "method": method,
                    "environment_id": environment["environment_id"],
                    "domain": environment["domain"],
                    "mu_index": environment_mu_index(environment),
                    "friction_mu": float(environment["friction_mu"]),
                    "sample_index": sample_index,
                    "action_id": int(metadata[sample_index]["action_id"]),
                    "is_support": is_support,
                    "selection_source": "observed_support" if is_support else "model_prediction",
                    "selection_xy": (
                        ground_truth[sample_index]["endpoint_xy"]
                        if is_support
                        else predicted_endpoints[sample_index]
                    ),
                    "ground_truth_xy": ground_truth[sample_index]["endpoint_xy"],
                    "prediction_path": (
                        None if is_support else str(predictions[sample_index])
                    ),
                }
                candidates.append(candidate)
                candidate_rows.append(candidate)
            for target in targets:
                decision = evaluate_action_choice(
                    candidates, target, expected_action_count=len(candidates)
                )
                decision.update({
                    "method": method,
                    "environment_id": environment["environment_id"],
                    "domain": environment["domain"],
                    "mu_index": environment_mu_index(environment),
                    "friction_mu": float(environment["friction_mu"]),
                })
                decision_rows.append(decision)

        action_summary = aggregate_action_rows(decision_rows)
        method_output = output_root / "methods" / method
        write_jsonl(method_output / "video_object_per_query.jsonl", query_rows)
        write_jsonl(
            method_output / "video_object_per_environment.jsonl",
            video_summary.pop("per_environment"),
        )
        write_json(method_output / "video_object_summary.json", video_summary)
        write_jsonl(method_output / "action_candidates.jsonl", candidate_rows)
        write_jsonl(method_output / "action_decisions.jsonl", decision_rows)
        write_json(method_output / "action_summary.json", action_summary)

        overall = video_summary["overall"]
        action_overall = action_summary["overall"]
        scoreboard_row = {
            "method": method,
            "query_count": video_summary["query_count"],
            "environment_count": video_summary["environment_count"],
            **{name: overall[name] for name in VIDEO_METRICS},
            "action_success_all": action_overall["headline_task_success_rate"],
            "action_success_id": action_summary["by_domain"]["id"]["headline_task_success_rate"],
            "action_success_ood": action_summary["by_domain"]["ood"]["headline_task_success_rate"],
            "action_selected_oracle_all": action_overall["selected_is_oracle_rate"],
            "action_mean_regret_all": action_overall["mean_regret"],
            "action_mean_coverage_all": action_overall["mean_action_coverage"],
            "action_eligible_decisions_all": action_overall["oracle_reachable_complete_count"],
        }
        scoreboard.append(scoreboard_row)
        benchmark_summary[method] = {
            "video_object": video_summary,
            "action_selection": action_summary,
        }
        print(
            f"[method complete] {method}: queries={len(query_rows)} "
            f"action_decisions={len(decision_rows)}",
            flush=True,
        )

    write_csv(output_root / "scoreboard.csv", scoreboard)
    write_json(output_root / "scoreboard.json", scoreboard)
    write_json(output_root / "benchmark_summary.json", benchmark_summary)
    write_json(output_root / "protocol.json", {
        "source_protocol": str(protocol_path),
        "metadata_path": str(metadata_path),
        "dataset_root": str(dataset_root),
        "action_target_config": str(target_config_path),
        "support_size": int(protocol["support_size"]),
        "support_is_excluded_from_query_metrics": True,
        "query_ground_truth_used_for_action_selection": False,
        "ground_truth_window": "metadata start_frame and length (65-105 for Event80)",
        "future_start_frame": future_start,
        "aggregation": "equal weight per environment after averaging its nine queries",
        "object_metric": "main-camera pushed-block centroid ADE/FDE",
        "offscreen_handling": "hold final observed centroid after confirmed exit",
        "tracker_type": tracker_type,
        "lpips_implementation": "official lpips package",
        "lpips_network": lpips_config.get("network", "alex"),
        "methods": config["methods"],
    })
    print(f"[done] scoreboard={output_root / 'scoreboard.csv'}", flush=True)


if __name__ == "__main__":
    main()
