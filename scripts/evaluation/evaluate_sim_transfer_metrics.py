#!/usr/bin/env python3
"""Evaluate appearance and physical-state metrics on action-transfer queries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (ROOT, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from evaluate_sim_action_selection import (
    _gt_frames,
    _metadata,
    _prediction_index,
    _read_json,
    _resolve,
    _video_path,
)
from wan_video_action.evaluation.io import (
    read_video_frames,
    write_json_atomic,
    write_jsonl_atomic,
)
from wan_video_action.evaluation.manifest import EvaluationRecord
from wan_video_action.evaluation.sim_task_extractors import extract_sim_task_state
from wan_video_action.evaluation.task_state import save_task_state
from wan_video_action.metrics.aggregation import aggregate_query_metrics
from wan_video_action.metrics.global_video import (
    LPIPSEvaluator,
    _align_frames,
    frame_psnr,
    frame_ssim,
)
from wan_video_action.metrics.task_metrics import evaluate_task_record


TASK_ALIASES = {"mass_friction": "joint_mass_friction"}
OBJECT_NAMES = {
    "gravity": ["object"],
    "mass_collision": ["struck_object", "striker"],
    "mass_friction": ["pushed_object", "driver"],
    "mass_balance": ["bar"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--lpips-device", default="cuda")
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    return parser.parse_args()


def _domain(plan: Mapping[str, Any], source_index: int) -> str:
    by_source = plan.get("domain_by_source", {})
    value = by_source.get(source_index, by_source.get(str(source_index)))
    domain = str(value if value is not None else plan.get("domain", "id"))
    if domain not in {"id", "ood"}:
        raise ValueError(f"Invalid domain {domain!r} for source {source_index}.")
    return domain


def _extractor_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(config.get("extractor", {}))
    allowed = {
        "fps",
        "main_view_width",
        "min_area",
        "max_area",
        "edge_margin",
        "light_roi",
        "yellow_threshold",
    }
    output = {key: value for key, value in raw.items() if key in allowed}
    if "light_roi" in output:
        output["light_roi"] = tuple(output["light_roi"])
    return output


def _task_settings(task: str, config: Mapping[str, Any]) -> dict[str, Any]:
    canonical = TASK_ALIASES.get(task, task)
    settings: dict[str, Any] = {"task": canonical}
    if task in OBJECT_NAMES:
        settings.update({
            "object_names": OBJECT_NAMES[task],
            "primary_object_indices": [0],
            "kinematic_object_index": 0,
        })
    settings.update(config.get("task_metrics", {}))
    return settings


def _main_view(frames: np.ndarray, width: int) -> np.ndarray:
    return frames[:, :, :min(width, frames.shape[2])]


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("sample_id", row.get("id", f"sample{index:04d}")))


def main() -> None:
    args = parse_args()
    if args.lpips and not __import__("os").environ.get("SLURM_JOB_ID"):
        raise SystemExit("LPIPS is restricted to a scheduled compute node.")

    config_path = _resolve(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = str(config["task"])
    canonical_task = TASK_ALIASES.get(task, task)
    method = str(config.get("method_name", "unknown"))
    rows = _metadata(_resolve(config["metadata_jsonl"]))
    dataset_root = _resolve(config["dataset_root"])
    if args.output_dir:
        output_dir = _resolve(args.output_dir)
    else:
        output_dir = _resolve(config["output_dir"]).parent / "video_metrics"
    state_root = output_dir / "states" / "sim_rgb_v1"
    extractor = _extractor_settings(config)
    main_view_width = int(extractor.get("main_view_width", 224))
    task_settings = _task_settings(task, config)
    lpips = (
        LPIPSEvaluator(net=args.lpips_net, device=args.lpips_device)
        if args.lpips
        else None
    )

    manifest_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_metric_names: set[str] = set()
    skipped: list[dict[str, Any]] = []

    for plan_value in config["transfer_plans"]:
        plan = {"path": plan_value} if isinstance(plan_value, str) else dict(plan_value)
        plan_path = _resolve(plan["path"])
        predictions = _prediction_index(plan_path.parent)
        for environment in _read_json(plan_path):
            source_index = int(environment["source_index"])
            source_row = rows[source_index]
            if source_row is None:
                raise KeyError(f"Missing source metadata row {source_index}.")
            source_id = _sample_id(source_row, source_index)
            domain = _domain(plan, source_index)
            for target_index in map(int, environment["target_indices"]):
                row = rows[target_index]
                if row is None:
                    raise KeyError(f"Missing target metadata row {target_index}.")
                prediction_path = predictions.get((source_index, target_index))
                if prediction_path is None:
                    skipped.append({
                        "source_index": source_index,
                        "target_index": target_index,
                        "reason": "missing_prediction",
                    })
                    continue

                gt_frames = _gt_frames(row, dataset_root)
                pred_frames = np.asarray(read_video_frames(prediction_path))
                gt_global, pred_global = _align_frames(
                    _main_view(gt_frames, main_view_width),
                    _main_view(pred_frames, main_view_width),
                )
                count = min(len(gt_global), len(pred_global))
                gt_global, pred_global = gt_global[:count], pred_global[:count]
                target_id = _sample_id(row, target_index)
                common = {
                    "sample_id": target_id,
                    "environment_id": f"source{source_index:04d}",
                    "method": method,
                    "split": "evaluation",
                    "domain": domain,
                    "support_size": 1,
                    "seed": int(config.get("seed", 0)),
                    "source_index": source_index,
                    "target_index": target_index,
                    "evaluated_frames": count,
                }
                appearance = dict(common)
                appearance.update({
                    "psnr": float(np.mean(frame_psnr(gt_global, pred_global))),
                    "ssim": float(np.mean(frame_ssim(gt_global, pred_global))),
                })
                if lpips is not None:
                    appearance["lpips"] = float(np.mean(lpips(
                        gt_global,
                        pred_global,
                        batch_size=args.lpips_batch_size,
                    )))
                global_rows.append(appearance)

                gt_state = extract_sim_task_state(task, gt_frames, **extractor)
                pred_state = extract_sim_task_state(task, pred_frames, **extractor)
                stem = f"source{source_index:04d}_sample{target_index:04d}"
                gt_state_path = state_root / "ground_truth" / f"{stem}.npz"
                pred_state_path = state_root / "prediction" / f"{stem}.npz"
                save_task_state(gt_state_path, gt_state)
                save_task_state(pred_state_path, pred_state)

                record = EvaluationRecord(
                    sample_id=target_id,
                    environment_id=f"source{source_index:04d}",
                    method=method,
                    split="evaluation",
                    domain=domain,
                    support_size=1,
                    seed=int(config.get("seed", 0)),
                    gt_video_path=str(_video_path(row, dataset_root)),
                    pred_video_path=str(prediction_path),
                    support_ids=(source_id,),
                    gt_start_frame=int(row.get("start_frame", 0)),
                    pred_start_frame=0,
                    num_frames=count,
                    gt_frame_stride=int(row.get("frame_stride", 1)),
                    pred_frame_stride=1,
                    gt_state_start_frame=0,
                    pred_state_start_frame=0,
                    gt_state_path=str(gt_state_path),
                    pred_state_path=str(pred_state_path),
                    task=canonical_task,
                    metadata={
                        "source_index": source_index,
                        "target_index": target_index,
                        "action_id": row.get(config.get("action_field", "action_id")),
                    },
                )
                manifest_rows.append(asdict(record))
                result = evaluate_task_record(record, task_settings)
                task_rows.append(result.values)
                task_metric_names.update(result.metric_names)

    if not manifest_rows:
        raise RuntimeError("No predicted transfer queries were available for evaluation.")

    global_names = ["psnr", "ssim"] + (["lpips"] if lpips is not None else [])
    sorted_task_names = sorted(task_metric_names)
    object_names = [
        name for name in sorted_task_names
        if "centroid" in name or name.endswith("_missing_rate")
    ]
    physical_names = [name for name in sorted_task_names if name not in object_names]
    write_jsonl_atomic(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl_atomic(output_dir / "global" / "global_per_query.jsonl", global_rows)
    write_json_atomic(
        output_dir / "global" / "global_summary.json",
        aggregate_query_metrics(global_rows, global_names),
    )
    write_jsonl_atomic(output_dir / "task_specific" / "task_per_query.jsonl", task_rows)
    write_json_atomic(output_dir / "object_centric" / "object_summary.json", {
        "metric_names": object_names,
        "aggregation": (
            aggregate_query_metrics(task_rows, object_names) if object_names else {}
        ),
    })
    write_json_atomic(output_dir / "task_specific" / "task_summary.json", {
        "task": canonical_task,
        "metric_names": physical_names,
        "aggregation": (
            aggregate_query_metrics(task_rows, physical_names) if physical_names else {}
        ),
    })
    write_jsonl_atomic(output_dir / "skipped.jsonl", skipped)
    write_json_atomic(output_dir / "protocol.json", {
        "config": str(config_path),
        "extractor": "sim_rgb_v1",
        "global_metrics": global_names,
        "method": method,
        "object_primary_indices": task_settings.get("primary_object_indices", []),
        "query_ground_truth_used_for_selection": False,
        "task": canonical_task,
        "transfer_plans": config["transfer_plans"],
    })


if __name__ == "__main__":
    main()
