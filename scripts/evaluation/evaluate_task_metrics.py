#!/usr/bin/env python3
"""Evaluate task-specific physical metrics from frozen task-state artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation import EvaluationResultLayout, load_evaluation_config, load_manifest
from wan_video_action.evaluation.io import write_json_atomic, write_jsonl_atomic
from wan_video_action.metrics.aggregation import aggregate_query_metrics
from wan_video_action.metrics.task_metrics import evaluate_task_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_evaluation_config(args.config)
    manifest_path = args.manifest or config.get("manifest_path")
    if not manifest_path:
        raise SystemExit("manifest_path is required.")
    settings = config.get("task_metrics", {})
    evaluated = [evaluate_task_record(record, settings) for record in load_manifest(manifest_path)]
    rows = [result.values for result in evaluated]
    metric_names = sorted({name for result in evaluated for name in result.metric_names})
    aggregation = aggregate_query_metrics(rows, metric_names)
    if args.output_dir or config.get("output_dir"):
        destination = Path(args.output_dir or config["output_dir"])
    else:
        layout = EvaluationResultLayout.from_config(config)
        layout.create_shared_directories()
        destination = layout.comparison_metric_dir("task_specific")
    write_jsonl_atomic(destination / "task_per_query.jsonl", rows)
    write_json_atomic(destination / "task_summary.json", {
        "task": settings.get("task"),
        "metric_names": metric_names,
        "aggregation": aggregation,
    })


if __name__ == "__main__":
    main()
