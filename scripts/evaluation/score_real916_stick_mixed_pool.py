#!/usr/bin/env python3
"""Report a frozen test-plus-train-negative action candidate pool separately."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real916_stick_final_common import write_json
from score_real97_stick_action_precision_cpu import aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    source = ROOT / config["source_metrics"]
    records = json.loads((source / "per_query.json").read_text())
    candidates = set(config["added_train_keys"])
    train = [r for r in records if r["key"] in candidates]
    test = [r for r in records if r["split"] == "test"]
    if len(train) != len(candidates) or len(test) != 45:
        raise RuntimeError("Candidate pool mismatch")
    if any(r["split"] != "train" or r["dataset_outcome_for_audit"] not in
           ("left_down", "right_down") for r in train):
        raise RuntimeError("Added candidates differ from frozen annotation selection")
    methods = ("standard", "ours_stage1", "ours_stage2")
    groups = {}
    for name, selected in (("test_only", test), ("added_train_only", train),
                           ("test_plus_train_negative_candidates", test + train)):
        metrics = {}
        for method in methods:
            metrics[method] = {}
            for threshold in ("3", "5"):
                value = aggregate([{"methods": r["action"]} for r in selected], method, threshold)
                value["selection_rate"] = value["predicted_balanced"] / len(selected)
                metrics[method][threshold] = value
        groups[name] = {"queries": len(selected), "methods": metrics}
    output = ROOT / config["output"]
    pool = [{
        "key": r["key"], "split": r["split"], "environment": r["environment"],
        "selection_annotation": r["dataset_outcome_for_audit"] if r["split"] == "train" else None,
        "gt_visual_decision": r["action"]["gt"]["5"],
        "method_predictions": {m: r["action"][m]["5"] for m in methods},
        "review": r["review"],
    } for r in test + train]
    summary = {
        "groups": groups, "primary_threshold_deg": 5,
        "metric": "true balanced among predicted balanced",
        "test_only_reported_separately": True,
        "selection_uses_model_prediction": False,
        "new_video_inference": False,
        "train_candidates_selected_by": "pre-existing episode outcome annotation",
        "scoring_gt": "actual visible final 0.3 seconds, not whole-episode annotation",
        "mixed_pool_is_not_heldout_test_performance": True,
        "unknown_selected_gt": "report precision bounds; never silently drop or count as failure",
        "formal_metric_approved": False, "visual_audit_pending": True,
    }
    write_json(output / "frozen_pool_config.json", config)
    write_json(output / "candidate_pool.json", pool)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
