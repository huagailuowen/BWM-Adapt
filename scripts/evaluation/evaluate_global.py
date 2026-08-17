#!/usr/bin/env python3
"""Evaluate secondary global video metrics from a frozen JSONL manifest."""

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
from wan_video_action.metrics.global_video import (
    LPIPSEvaluator,
    evaluate_global_record,
)


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
        output_dir = layout.comparison_metric_dir("global")

    settings = config.get("global_metrics", {})
    use_lpips = bool(settings.get("lpips", True))
    lpips_evaluator = None
    if use_lpips:
        lpips_evaluator = LPIPSEvaluator(
            net=settings.get("lpips_net", "alex"),
            device=settings.get("device", "cuda"),
        )

    rows = [
        evaluate_global_record(
            record,
            compute_psnr=bool(settings.get("psnr", True)),
            lpips_evaluator=lpips_evaluator,
            lpips_batch_size=int(settings.get("lpips_batch_size", 8)),
        )
        for record in load_manifest(manifest_path)
    ]
    metric_names = [
        name for name in ("psnr", "lpips") if name in rows[0]
    ]
    aggregation = aggregate_query_metrics(rows, metric_names)
    destination = Path(output_dir)
    write_jsonl_atomic(destination / "global_per_query.jsonl", rows)
    write_json_atomic(destination / "global_summary.json", aggregation)


if __name__ == "__main__":
    main()
