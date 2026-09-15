#!/usr/bin/env python3
"""Score TTT with the frozen Real97 metrics; never modify baseline results."""
from collections import defaultdict
import json
from pathlib import Path

import score_real97_ball_door as engine

engine.METHODS = ("ttt",)
original_aggregate = engine.aggregate


def freeze_manifest(config, output):
    path = output / "manifest.json"
    if path.exists():
        manifest = engine.read_json(path)
        if manifest["config"] != config:
            raise RuntimeError("Output belongs to a different score configuration")
        return manifest
    baseline = engine.read_json(engine.ROOT / config["baseline_scores"] / "manifest.json")
    queries, steps = [], {}
    for task, setting in config["tasks"].items():
        run = engine.ROOT / setting["ttt"]
        completion = engine.read_json(run / "inference_complete.json")
        if completion["checkpoint_step"] != setting["ttt_checkpoint_step"]:
            raise RuntimeError(f"Unexpected TTT checkpoint: {task}")
        indexed = {}
        for marker in sorted((run / "completed").glob("q*.json")):
            record = engine.read_json(marker)
            row = record["query"]
            identity = row["sample_id"]
            if identity in indexed:
                raise RuntimeError(f"Duplicate TTT query: {identity}")
            indexed[identity] = (record, row, marker)
        reference = [q for q in baseline["queries"] if q["task"] == task]
        if set(indexed) != {q["row"]["sample_id"] for q in reference}:
            raise RuntimeError(f"TTT and baseline query sets differ for {task}; no silent intersection")
        for query in reference:
            row = query["row"]
            record, other, marker = indexed[row["sample_id"]]
            for field in ("environment", "dataset_split", "episode_index", "action_level",
                          "video", "action", "start_frame", "end_frame", "frame_stride",
                          "native_frame_indices", "evaluation_frame_indices", "length"):
                if row.get(field) != other.get(field):
                    raise RuntimeError(f"Query mismatch: {task}/{row['sample_id']}/{field}")
            paths = {"gt": query["paths"]["gt"], "ttt": record["prediction"]}
            for video in paths.values():
                if not Path(video).is_file():
                    raise FileNotFoundError(video)
            queries.append({**query, "paths": paths, "source_markers": {
                "ttt": str(marker), "baseline": query.get("source_markers", {})}})
        if completion["queries"] != len(reference):
            raise RuntimeError(f"Incomplete TTT query publication: {task}")
        steps[task] = {"ttt": completion["checkpoint_step"]}
    manifest = {
        "config": config, "queries": queries, "methods": ["ttt"], "paired_gt_only": True,
        "checkpoint_steps": steps,
        "baseline_manifest": str(engine.ROOT / config["baseline_scores"] / "manifest.json"),
        "fairness_note": (
            "Identical factual query identities, GT videos, frame masks and tracking thresholds. "
            "TTT Door step3656 and Ball step4144 differ from baseline training steps. "
            "Historical Ours Stage2 includes known-environment initialization. "
            "Tracking remains provisional pending the existing held-out visual audit."
        ),
    }
    engine.save_json(path, manifest)
    engine.save_json(output / "protocol.json", config)
    return manifest


def ball_action(rows, method):
    grouped = defaultdict(list)
    for row in rows:
        if row["task"] == "ball" and row["split"] == "test":
            grouped[row["environment"]].append(row)
    decisions = []
    for environment, group in sorted(grouped.items()):
        gt, pred, thresholds = {}, {}, {}
        for row in group:
            level = int(row["level"])
            if level in gt:
                raise RuntimeError(f"Multiple test queries per level: {environment}/{level}")
            marker = row["marker"]["center"] if row.get("marker") else None
            threshold = marker[0] - 30.0 if marker is not None else None
            thresholds[level] = threshold
            for output, variant in ((gt, "gt"), (pred, method)):
                peak = row["trajectory_summary"][variant]["peak_center"]
                output[level] = None if threshold is None or peak is None else bool(peak[0] >= threshold)
        complete = set(gt) == set(range(1, 11)) and all(value is not None for value in gt.values())
        truth = next((level for level in sorted(gt) if gt[level] is True), None)
        chosen = next((level for level in sorted(pred) if pred[level] is True), None)
        gt_pair = [truth, truth + 1] if truth is not None else []
        pred_pair = [chosen, chosen + 1] if chosen is not None else []
        score = len(set(gt_pair).intersection(pred_pair)) / 2 if complete and gt_pair else None
        decisions.append({
            "environment": environment, "gt_first_reach": truth, "predicted_first_reach": chosen,
            "gt_prefer": gt_pair, "predicted_prefer": pred_pair, "score": score,
            "gt_reaches_by_level": gt, "predicted_reaches_by_level": pred,
            "threshold_x_by_level": thresholds, "gt_complete": complete,
            "unknown_prediction_levels": [level for level, value in pred.items() if value is None],
            "level11_is_formal_successor_only": 11 in gt_pair or 11 in pred_pair,
        })
    return {
        "task": "ball", "method": method, "rule": "first_reach_and_successor_pair_overlap",
        "environment_count": len(decisions),
        "scorable_count": sum(row["score"] is not None for row in decisions),
        "macro_score": engine.mean([row["score"] for row in decisions]),
        "conservative_score_fixed_denominator": (
            sum(row["score"] or 0 for row in decisions) / len(decisions) if decisions else None),
        "decisions": decisions,
    }


def aggregate(output, manifest):
    report = original_aggregate(output, manifest)
    scored = [engine.read_json(output / "queries" / q["task"] / (q["key"] + ".json"))
              for q in manifest["queries"]
              if (output / "queries" / q["task"] / (q["key"] + ".json")).is_file()]
    report["action"] = [row for row in report["action"] if row["task"] != "ball"]
    report["action"].append(ball_action(scored, "ttt"))
    engine.save_json(output / "summary_provisional.json", report)
    engine.save_json(output / "ball_action_first_reach_successor_pair_overlap.json", report["action"][-1])
    config = manifest["config"]
    source = engine.ROOT / config["baseline_scores"]
    baseline_report = engine.read_json(source / "summary_provisional.json")
    baseline_manifest = engine.read_json(source / "manifest.json")
    baseline_rows = [engine.read_json(source / "queries" / q["task"] / (q["key"] + ".json"))
                     for q in baseline_manifest["queries"] if q["task"] == "ball"]
    baseline_actions = [row for row in baseline_report["action"] if row["task"] == "door"]
    baseline_actions += [ball_action(baseline_rows, method) for method in baseline_manifest["methods"]]
    comparison = {
        "status": config["tracking_status"], "ttt_queries_expected": len(manifest["queries"]),
        "ttt_queries_scored": len(scored), "ttt_lpips_complete": all(
            (output / "lpips" / q["task"] / (q["key"] + ".json")).is_file() for q in manifest["queries"]),
        "image_object": baseline_report["image_object"] + report["image_object"],
        "action": baseline_actions + report["action"],
        "checkpoint_steps": {"baseline": baseline_manifest["checkpoint_steps"],
                             "ttt": manifest["checkpoint_steps"]},
        "fairness_note": manifest["fairness_note"],
        "baseline_results_modified": False, "excluded_inputs": "unpaired fixed-initial-frame action sweeps",
        "door_calibration_caveat": "Nominal door-4d prefer [3,4] is retained; tracked held-out GT previously gave [4,5].",
    }
    engine.save_json(output / "comparison_with_baselines.json", comparison)
    print(json.dumps({"scored": len(scored), "ball_ttt_action": report["action"][-1]["macro_score"],
                      "lpips_complete": comparison["ttt_lpips_complete"]}), flush=True)
    return report


engine.freeze_manifest = freeze_manifest
engine.aggregate = aggregate

if __name__ == "__main__":
    engine.main()
