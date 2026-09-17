#!/usr/bin/env python3
"""Paired, cache-only action-selection sensitivity analysis; no video decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan_video_action.metrics.task_action_selection import (
    TaskTarget,
    evaluate_task_action_choice,
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def group_key(row):
    return str(row["domain"]), int(row["source_index"]), str(row.get("partition", "all"))


def read_candidates(path):
    payload = path.read_bytes()
    groups = defaultdict(list)
    for line in payload.splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("partition", "all") == "all":
                groups[group_key(row)].append(row)
    return dict(groups), hashlib.sha256(payload).hexdigest()


def object_value(row, field, object_index):
    values = row.get(field.replace("_value", "_object_values"), {})
    value = values.get(str(object_index), row.get(field))
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"Missing {field} for action {row['action_id']}")
    point = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"Non-finite {field} for action {row['action_id']}")
    return point


def check_paired(groups, reference, object_indices, atol, method):
    if groups.keys() != reference.keys():
        raise ValueError(f"{method}: environment/partition mismatch; no silent filtering allowed")
    for key, rows in groups.items():
        actions = {str(row["action_id"]): row for row in rows}
        ref_actions = {str(row["action_id"]): row for row in reference[key]}
        if len(actions) != len(rows) or actions.keys() != ref_actions.keys():
            raise ValueError(f"{method} {key}: duplicate or mismatched candidate actions")
        for action, row in actions.items():
            if sorted(row["sample_indices"]) != sorted(ref_actions[action]["sample_indices"]):
                raise ValueError(f"{method} {key} {action}: candidate trajectory mismatch")
            for index in object_indices:
                gt = object_value(row, "ground_truth_value", index)
                ref_gt = object_value(ref_actions[action], "ground_truth_value", index)
                if any(abs(a - b) > atol for a, b in zip(gt, ref_gt)):
                    raise ValueError(f"{method} {key} {action}: GT cache mismatch")


def center_error_px(point, target, width, height):
    region = target["region"]
    cx = (float(region["x_min"]) + float(region["x_max"])) / 2
    cy = (float(region["y_min"]) + float(region["y_max"])) / 2
    return math.hypot((point[0] - cx) * width, (point[1] - cy) * height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this evaluator on a Slurm compute node, not a login node.")
    config_path = resolve(args.config)
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    template_path = resolve(config["protocol_config"])
    template_text = template_path.read_text()
    template = yaml.safe_load(template_text)
    if config["task"] != "mass_friction" or template["task"] != "mass_friction":
        raise ValueError("This analysis is restricted to mass_friction")
    targets = []
    for raw_target in template["targets"]:
        target = dict(raw_target)
        target["object_index"] = template["outcome"]["object_roles"].get(
            target.get("object_role"), template["outcome"]["object_index"]
        )
        targets.append(target)
    precision_ids = set(config["precision_target_ids"])
    if not precision_ids.issubset({target["id"] for target in targets}):
        raise ValueError("Unknown precision target")
    indices = {target["object_index"] for target in targets}
    caches, sources = {}, {}
    for method, directory in config["methods"].items():
        path = resolve(directory) / "candidate_outcomes.jsonl"
        caches[method], digest = read_candidates(path)
        sources[method] = {"path": str(path), "sha256": digest}
    reference = caches[config["reference_method"]]
    if len(reference) != int(config["expected_environments"]):
        raise ValueError("Reference environment count differs from the fixed protocol")
    for method, groups in caches.items():
        check_paired(groups, reference, indices, float(config["ground_truth_match_atol"]), method)

    cohort, skipped = [], []
    for key, candidates in sorted(reference.items()):
        for target in targets:
            original = evaluate_task_action_choice(candidates, TaskTarget.from_mapping(target))
            if original["status"] != "ok" or not original["candidate_set_complete"]:
                raise ValueError(f"Incomplete reference predictions: {key} {target['id']}")
            item = {"domain": key[0], "source_index": key[1], "partition": key[2], "target_id": target["id"]}
            if original["oracle_reachable"]:
                cohort.append((key, target))
            else:
                skipped.append({**item, "reason": "unreachable_in_ground_truth"})
    if len(cohort) != int(config["expected_reachable_environment_targets"]):
        raise ValueError("Reachable cohort differs from the fixed protocol; refusing to change the denominator")

    width, height = int(config["image_width"]), int(config["image_height"])
    records = []
    for method, groups in caches.items():
        for key, target in cohort:
            candidates = groups[key]
            for rule in ("original", "target_center_tiebreak"):
                effective = dict(target)
                if rule != "original" and target["id"] in precision_ids:
                    effective["selection_strategy"] = config["selection_strategy"]
                result = evaluate_task_action_choice(candidates, TaskTarget.from_mapping(effective))
                if result["status"] != "ok" or not result["candidate_set_complete"]:
                    raise ValueError(f"{method} {key}: incomplete predictions; no exclusion allowed")
                if not result["oracle_reachable"]:
                    raise ValueError(f"{method} {key}: inconsistent GT reachability")
                center_error = center_regret = None
                if target["id"] in precision_ids:
                    center_error = center_error_px(result["selected_ground_truth_value"], target, width, height)
                    oracle_center_error = min(
                        center_error_px(object_value(row, "ground_truth_value", target["object_index"]), target, width, height)
                        for row in candidates
                    )
                    center_regret = max(0.0, center_error - oracle_center_error)
                records.append({
                    **result, "method": method, "rule": rule,
                    "domain": key[0], "source_index": key[1], "partition": key[2],
                    "target_id": target["id"], "center_error_px": center_error,
                    "center_regret_px": center_regret,
                })

    scores = []
    for method in caches:
        for rule in ("original", "target_center_tiebreak"):
            for domain in ("all", "id", "ood"):
                rows = [r for r in records if r["method"] == method and r["rule"] == rule and (domain == "all" or r["domain"] == domain)]
                precision_rows = [r for r in rows if r["center_error_px"] is not None]
                if not rows:
                    continue
                scores.append({
                    "method": method, "rule": rule, "domain": domain,
                    "count": len(rows), "successes": sum(bool(r["task_success"]) for r in rows),
                    "action_success": fmean(float(r["task_success"]) for r in rows),
                    "mean_region_regret": fmean(r["regret"] for r in rows),
                    "precision_target_count": len(precision_rows),
                    "mean_center_error_px": fmean(r["center_error_px"] for r in precision_rows) if precision_rows else None,
                    "mean_center_regret_px": fmean(r["center_regret_px"] for r in precision_rows) if precision_rows else None,
                })
    output = resolve(config["output_dir"])
    if output in {resolve(path) for path in config["methods"].values()}:
        raise ValueError("Cannot overwrite an original action evaluation directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "exploratory_post_hoc_not_a_replacement_for_primary_results",
        "selection_uses_ground_truth": False,
        "original_success_policies_preserved": True,
        "long_target_selection_preserved": True,
        "support_candidate_handling": "Unchanged from the frozen candidate caches for every method",
        "config": config, "resolved_targets": targets, "sources": sources,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "protocol_config_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
        "cohort": [{"domain": k[0], "source_index": k[1], "partition": k[2], "target_id": t["id"]} for k, t in cohort],
        "skipped_unreachable": skipped,
        "center_regret_definition": "Selected GT center distance minus the minimum GT center distance over the same candidate actions; short/medium targets only",
    }
    for name, value in (("protocol.json", protocol), ("summary.json", scores)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (output / "decisions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    with (output / "scoreboard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    lines = [
        "# Mass friction: exploratory action-selection sensitivity analysis", "",
        "Outcome-informed supplementary analysis. Original formal scores remain unchanged.", "",
        "Target regions, complete candidate sets, reachability, and success policies are fixed.",
        "Only short/medium predicted-distance ties use proximity to the target center.",
        "Long-range minimum-reaching selection is unchanged. Center metrics cover short/medium targets only.", "",
        "| Method | Rule | Success | Center error (px) | Center regret (px) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in scores:
        if row["domain"] == "all":
            lines.append(f"| {row['method']} | {row['rule']} | {row['successes']}/{row['count']} ({100 * row['action_success']:.2f}%) | {row['mean_center_error_px']:.3f} | {row['mean_center_regret_px']:.3f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
