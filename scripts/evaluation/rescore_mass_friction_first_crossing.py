#!/usr/bin/env python3
"""Fixed-protocol first-crossing action calibration from cached terminal points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from datetime import datetime, timezone

import yaml

from rescore_mass_friction_target_precision import (
    check_paired,
    object_value,
    read_candidates,
    resolve,
)


def isotonic_projection(values):
    """Equal-weight PAVA, using predicted outcomes only, never ground truth."""
    blocks = []
    for index, value in enumerate(values):
        blocks.append([float(value), 1, index, index + 1])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1], left[2], right[3]])
    result = [0.0] * len(values)
    for total, count, start, end in blocks:
        result[start:end] = [total / count] * (end - start)
    return result


def first_crossing(values, threshold):
    return next((rank for rank, value in enumerate(values) if value > threshold), None)


def action_order(row):
    value = float(row["action_order"])
    if not math.isfinite(value):
        raise ValueError("Action order must be finite")
    return value


def amplitude(row):
    values = {
        float(member["action_metadata"]["action_amplitude"])
        for member in row.get("member_records", [])
        if member.get("action_metadata", {}).get("action_amplitude") is not None
    }
    if len(values) > 1:
        raise ValueError(f"Action {row['action_id']} has multiple force amplitudes")
    return next(iter(values)) if values else None


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this cache-only analysis on a Slurm compute node.")
    config_path = resolve(args.config)
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    if config["coordinate"] != "terminal_y_normalized" or config["comparison"] != "strictly_greater":
        raise ValueError("This evaluator uses terminal normalized y and strict > only")
    if config["aggregation"] != "equal_threshold_weight_then_equal_reachable_environment_weight":
        raise ValueError("Unsupported aggregation")
    if config["missing_crossing_rank_error"] != "maximum_action_rank_gap":
        raise ValueError("Unsupported missing-crossing penalty")
    if not config["preserve_formal_results"]:
        raise ValueError("Exploratory evaluation must preserve original formal results")
    selectors = config["selectors"]
    if set(selectors) != {"raw_first_crossing", "isotonic_first_crossing"}:
        raise ValueError("Report both raw and monotonic-projection selectors")
    threshold_sets = config["threshold_sets"]
    for name, thresholds in threshold_sets.items():
        if not thresholds or len(thresholds) != len(set(thresholds)):
            raise ValueError(f"Invalid thresholds in {name}")
        if not all(math.isfinite(float(t)) and 0 <= float(t) <= 1 for t in thresholds):
            raise ValueError("Threshold coordinates must lie in [0, 1]")
    parent_path = resolve(config["candidate_protocol_config"])
    parent_text = parent_path.read_text()
    parent = yaml.safe_load(parent_text)
    if parent["task"] != "mass_friction":
        raise ValueError("Only the frozen mass-friction protocol is supported")
    object_index = int(config["object_index"])
    caches, sources = {}, {}
    for method, directory in parent["methods"].items():
        path = resolve(directory) / "candidate_outcomes.jsonl"
        caches[method], digest = read_candidates(path)
        sources[method] = {"path": str(path), "sha256": digest}
    reference = caches[parent["reference_method"]]
    if len(reference) != int(parent["expected_environments"]):
        raise ValueError("Unexpected environment count")
    for method, groups in caches.items():
        check_paired(groups, reference, {object_index}, float(parent["ground_truth_match_atol"]), method)

    scores_by_error = {int(k): float(v) for k, v in config["score_by_absolute_rank_error"].items()}
    records, curves = [], []
    for method, groups in caches.items():
        for key, candidates in sorted(groups.items()):
            ordered = sorted(candidates, key=action_order)
            orders = [action_order(row) for row in ordered]
            if len(orders) < 2 or len(set(orders)) != len(orders):
                raise ValueError(f"{method} {key}: expected distinct ordered force levels")
            gt_y = [object_value(row, "ground_truth_value", object_index)[1] for row in ordered]
            pred_y = [object_value(row, "selection_value", object_index)[1] for row in ordered]
            amplitudes = [amplitude(row) for row in ordered]
            if all(value is not None for value in amplitudes) and any(b <= a for a, b in zip(amplitudes, amplitudes[1:])):
                raise ValueError(f"{method} {key}: action order is not increasing in force")
            projected_y = isotonic_projection(pred_y)
            curves.append({
                "method": method, "domain": key[0], "source_index": key[1],
                "action_ids": [str(row["action_id"]) for row in ordered],
                "action_order": orders, "action_amplitude": amplitudes,
                "ground_truth_terminal_y": gt_y, "predicted_terminal_y": pred_y,
                "isotonic_predicted_terminal_y": projected_y,
                "gt_decreasing_adjacent_pairs": sum(b < a for a, b in zip(gt_y, gt_y[1:])),
                "pred_decreasing_adjacent_pairs": sum(b < a for a, b in zip(pred_y, pred_y[1:])),
                "candidate_sources": [row.get("selection_source") for row in ordered],
            })
            for set_name, thresholds in threshold_sets.items():
                for threshold in thresholds:
                    gt_rank = first_crossing(gt_y, threshold)
                    for selector in selectors:
                        effective_y = pred_y if selector == "raw_first_crossing" else projected_y
                        pred_rank = first_crossing(effective_y, threshold)
                        eligible = gt_rank is not None
                        missed = eligible and pred_rank is None
                        signed_error = pred_rank - gt_rank if eligible and pred_rank is not None else None
                        absolute_error = abs(signed_error) if signed_error is not None else None
                        if not eligible:
                            score = None
                        elif missed:
                            score = float(config["prediction_never_crosses_score"])
                        else:
                            score = scores_by_error.get(absolute_error, float(config["other_rank_error_score"]))
                        records.append({
                            "method": method, "selector": selector, "threshold_set": set_name,
                            "threshold": threshold, "domain": key[0], "source_index": key[1],
                            "partition": key[2], "candidate_count": len(ordered),
                            "eligible": eligible,
                            "status": "unreachable_in_gt" if not eligible else "prediction_never_crosses" if missed else "scored",
                            "gt_first_rank": gt_rank, "pred_first_rank": pred_rank,
                            "gt_first_action_id": str(ordered[gt_rank]["action_id"]) if eligible else None,
                            "pred_first_action_id": str(ordered[pred_rank]["action_id"]) if pred_rank is not None else None,
                            "signed_rank_error": signed_error, "absolute_rank_error": absolute_error,
                            "rank_error_with_missing_penalty": (len(ordered) - 1 if missed else absolute_error) if eligible else None,
                            "graded_score": score,
                            "exact_match": eligible and absolute_error == 0,
                            "within_one_level": eligible and absolute_error is not None and absolute_error <= 1,
                            "prediction_never_crosses": bool(missed),
                            "selected_action_gt_reaches": eligible and pred_rank is not None and gt_y[pred_rank] > threshold,
                        })

    reference_eligibility = {
        (r["domain"], r["source_index"], r["threshold_set"], r["threshold"]): r["eligible"]
        for r in records if r["method"] == parent["reference_method"]
    }
    for row in records:
        key = row["domain"], row["source_index"], row["threshold_set"], row["threshold"]
        if row["eligible"] != reference_eligibility[key]:
            raise ValueError("Method-dependent eligibility is forbidden")
    metrics = {
        "graded_score": "graded_score",
        "exact_match_rate": "exact_match",
        "within_one_level_rate": "within_one_level",
        "mean_rank_error_with_missing_penalty": "rank_error_with_missing_penalty",
        "no_crossing_rate": "prediction_never_crosses",
        "execution_reach_rate": "selected_action_gt_reaches",
    }
    per_threshold = []
    for method in caches:
        for selector in selectors:
            for set_name, thresholds in threshold_sets.items():
                for domain in ("all", "id", "ood"):
                    for threshold in thresholds:
                        rows = [r for r in records if r["method"] == method and r["selector"] == selector and r["threshold_set"] == set_name and r["threshold"] == threshold and (domain == "all" or r["domain"] == domain)]
                        eligible = [r for r in rows if r["eligible"]]
                        entry = {
                            "method": method, "selector": selector, "threshold_set": set_name,
                            "domain": domain, "threshold": threshold,
                            "environment_count": len(rows), "reachable_count": len(eligible),
                            "unreachable_count": len(rows) - len(eligible),
                        }
                        entry.update({name: fmean(float(r[field]) for r in eligible) if eligible else None for name, field in metrics.items()})
                        per_threshold.append(entry)
    summaries = []
    for method in caches:
        for selector in selectors:
            for set_name, thresholds in threshold_sets.items():
                for domain in ("all", "id", "ood"):
                    rows = [r for r in per_threshold if r["method"] == method and r["selector"] == selector and r["threshold_set"] == set_name and r["domain"] == domain]
                    usable = [r for r in rows if r["reachable_count"]]
                    entry = {
                        "method": method, "selector": selector, "threshold_set": set_name,
                        "domain": domain, "threshold_count": len(thresholds),
                        "reachable_threshold_count": len(usable),
                        "reachable_environment_threshold_pairs": sum(r["reachable_count"] for r in rows),
                    }
                    entry.update({name: fmean(r[name] for r in usable) if usable else None for name in metrics})
                    summaries.append(entry)

    output = resolve(config["output_dir"])
    if output in {resolve(value) for value in parent["methods"].values()} or output == resolve(parent["output_dir"]):
        raise ValueError("Do not overwrite original or previous exploratory scores")
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "summary.json", summaries)
    dump_json(output / "protocol.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "exploratory_post_hoc_not_primary_results",
        "config": config, "candidate_protocol": parent, "sources": sources,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "candidate_protocol_sha256": hashlib.sha256(parent_text.encode()).hexdigest(),
        "threshold_selection": "Fixed original target lines plus the full predeclared uniform grid; no outcome-based selection",
        "ground_truth_used_in_action_selection": False,
        "measurement": "Cached last-five-frame terminal y, not metric-world displacement or first-passage time",
        "missing_prediction_policy": "No predicted crossing on a GT-reachable case scores zero and receives the maximum rank-gap penalty",
        "isotonic_note": "Auxiliary prediction-only increasing fit. The GT first-crossing oracle remains raw. Raw and projected results are both reported.",
        "limitations": "Outcome-informed supplementary analysis; correlated thresholds do not create independent trials. Physical force-spacing and rank-spacing need not be equal.",
    })
    for filename, rows in (("decisions.jsonl", records), ("action_curves.jsonl", curves)):
        with (output / filename).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_csv(output / "scoreboard.csv", summaries)
    write_csv(output / "per_threshold.csv", per_threshold)
    lines = [
        "# Mass friction first-crossing action-level analysis", "",
        "Exploratory post-hoc analysis. Original formal scores are not replaced.",
        "Exact rank: 1; one rank apart: 0.5; otherwise or no predicted crossing: 0.",
        "Thresholds are terminal screen-y target lines, not physical displacement.",
        "Average reachable environments within each threshold, then weight thresholds equally.",
        "Unreachable GT cases are excluded identically for all methods; see per_threshold.csv for coverage.", "",
        "| Threshold set | Selector | Method | Graded score | Exact | Within one | Rank error (missing penalized) | No crossing | Pairs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["domain"] == "all" and row["graded_score"] is not None:
            lines.append(f"| {row['threshold_set']} | {row['selector']} | {row['method']} | {row['graded_score']:.4f} | {100 * row['exact_match_rate']:.2f}% | {100 * row['within_one_level_rate']:.2f}% | {row['mean_rank_error_with_missing_penalty']:.3f} | {100 * row['no_crossing_rate']:.2f}% | {row['reachable_environment_threshold_pairs']} |")
    lines.extend([
        "", "The 0/0.5/1 graded score is not binary task success.",
        "The isotonic selector is an auxiliary monotonicity assumption, not a tuned replacement.",
        "Report the entire threshold sweep and both selectors, not only favorable rows.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
