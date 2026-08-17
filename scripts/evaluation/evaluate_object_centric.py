#!/usr/bin/env python3
"""Evaluate primary object-centric metrics from precomputed masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation import (
    EvaluationResultLayout,
    load_evaluation_config,
    load_manifest,
)
from wan_video_action.evaluation.io import write_json_atomic, write_jsonl_atomic
from wan_video_action.metrics.aggregation import aggregate_query_metrics
from wan_video_action.metrics.object_centric import evaluate_object_record


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
    if args.output_dir or config.get("output_dir"):
        output_dir = Path(args.output_dir or config["output_dir"])
    else:
        layout = EvaluationResultLayout.from_config(config)
        layout.create_shared_directories()
        output_dir = layout.comparison_metric_dir("object_centric")

    settings = config.get("object_metrics", {})
    rows = [
        evaluate_object_record(
            record,
            mask_key=settings.get("mask_key", "masks"),
        )
        for record in load_manifest(manifest_path)
    ]
    metric_names = (
        "mean_iou",
        "final_iou",
        "centroid_ade",
        "centroid_fde",
        "missing_mask_rate",
    )
    aggregation = aggregate_query_metrics(rows, metric_names)
    destination = Path(output_dir)
    write_jsonl_atomic(destination / "object_per_query.jsonl", rows)
    write_json_atomic(destination / "object_summary.json", aggregation)


if __name__ == "__main__":
    main()
