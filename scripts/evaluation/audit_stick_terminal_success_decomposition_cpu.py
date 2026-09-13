"""Separate lift evidence, rise symmetry, and image-plane tilt for manual audit.

This is an opt-in diagnostic, not a replacement for an approved action metric.
"""

import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from audit_stick_terminal_zoom_cpu import make_sheet


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/evaluation_stick_phase_interval_flow_job114259"


def decompose(record):
    interval = record["decision"].get("terminal_window", {}).get("frame_interval")
    if not interval:
        return {"key": record["key"], "eligible_for_diagnostic": False, "reason": "no_terminal_interval"}
    path = SOURCE / "cases" / record["key"] / "tracks.jsonl"
    rows = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if interval[0] <= row["frame"] <= interval[1]:
                rows.append(row)
    valid = bool(rows) and all(
        len(row["sides"]) == 2
        and all(side.get("valid") and side.get("all_contacts_visible") for side in row["sides"])
        for row in rows
    )
    report = {"key": record["key"], "environment": record["environment"], "split": record["split"],
              "episode_index": record["episode_index"], "automatic_status": record["decision"]["status"],
              "terminal_window": record["decision"]["terminal_window"], "eligible_for_diagnostic": valid,
              "frame_states": dict(Counter(row["state"] for row in rows)),
              "manual_label": None, "manual_audit_complete": False}
    if not valid:
        report["reason"] = "terminal_tracking_or_visibility_incomplete"
        return report
    full = all(all(side.get("full_contact_geometry_visible", True) for side in row["sides"]) for row in rows)
    rises = np.asarray([[side["minimum_rise_px"] for side in row["sides"]] for row in rows], dtype=float)
    errors = np.asarray([[side["uncertainty_px"] for side in row["sides"]] for row in rows], dtype=float)
    angles = []
    for row in rows:
        reference = [np.asarray(side["reference_points"], dtype=float).mean(axis=0) for side in row["sides"]]
        current = [np.asarray(side["contact_points"], dtype=float).mean(axis=0) for side in row["sides"]]
        ref_vector, cur_vector = reference[1] - reference[0], current[1] - current[0]
        if ref_vector[0] <= 0 or cur_vector[0] <= 0:
            raise RuntimeError(f"Left/right ordering violation: {record['key']}")
        delta = math.degrees(math.atan2(cur_vector[1], cur_vector[0]) - math.atan2(ref_vector[1], ref_vector[0]))
        angles.append(abs((delta + 180.0) % 360.0 - 180.0))
    raw_both = bool(np.all(rises >= 1.0))
    confident_both = bool(np.all(rises - errors >= 1.0))
    ratios = np.maximum(0.0, rises.min(axis=1)) / np.maximum(1e-8, rises.max(axis=1))
    report.update({
        "full_contact_geometry_visible": full,
        "raw_bilateral_rise_ge1_every_terminal_frame": raw_both,
        "confidence_bilateral_rise_ge1_every_terminal_frame": confident_both,
        "left_right_terminal_median_rise_px": np.median(rises, axis=0).tolist(),
        "left_right_terminal_min_lower_rise_px": (rises - errors).min(axis=0).tolist(),
        "left_right_terminal_median_error_px": np.median(errors, axis=0).tolist(),
        "minimum_raw_rise_ratio": float(ratios.min()),
        "median_raw_rise_ratio": float(np.median(ratios)),
        "max_baseline_relative_endpoint_angle_deg_proxy": max(angles),
        "raw_ratio_sensitivity": {str(threshold): bool(raw_both and np.all(ratios >= threshold)) for threshold in (0.25, 1.0 / 3.0, 0.5)},
        "tilt_sensitivity_confident_bilateral_full_geometry": {
            str(threshold): bool(full and confident_both and max(angles) <= threshold) for threshold in (3.0, 5.0, 8.0)
        },
        "angle_is_proxy_not_calibrated_3d_or_direct_rod_measurement": True,
    })
    return report


def main():
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("Run on a CPU compute node")
    output = ROOT / f"outputs/stick_terminal_success_decomposition_job{job}"
    output.mkdir(exist_ok=False)
    sheets = output / "cases"
    sheets.mkdir()
    records = [record for record in json.loads((SOURCE / "per_video.json").read_text()) if record["scope"] == "full_episode"]
    reports = [decompose(record) for record in records]
    counts = Counter()
    groups = defaultdict(Counter)
    priority = []
    for report in reports:
        counts["total"] += 1
        groups[report["environment"]][report["automatic_status"]] += 1
        if not report["eligible_for_diagnostic"]:
            counts["terminal_tracking_or_visibility_incomplete"] += 1
            continue
        if report["confidence_bilateral_rise_ge1_every_terminal_frame"]:
            counts["confidence_bilateral_rise_ge1"] += 1
            if report["automatic_status"] != "balanced_lift_proposal":
                priority.append(report)
        for name in ("raw_ratio_sensitivity", "tilt_sensitivity_confident_bilateral_full_geometry"):
            for threshold, passed in report[name].items():
                if passed:
                    counts[f"{name}/{threshold}"] += 1
    priority.sort(key=lambda report: (report["automatic_status"] != "no_qualifying_lift_observed", report["key"]))
    summary = {"status": "diagnostic_awaiting_visual_labels", "source": str(SOURCE), "counts": counts,
               "automatic_counts_by_environment": groups,
               "priority_keys": [report["key"] for report in priority],
               "formal_metric_changed": False, "formal_results_published": False,
               "notes": ["Lift evidence and symmetry are separate axes, not interchangeable success labels.",
                         "Image motion and endpoint-angle proxies do not certify physical ground clearance.",
                         "Threshold sensitivity is for identifying audit cases, not maximizing the success count.",
                         "Unknown remains distinct from negative; no changes to existing evaluator or configs."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "per_episode_diagnostics.json").write_text(json.dumps(reports, indent=2) + "\n")
    (output / "priority_review.json").write_text(json.dumps(priority, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(make_sheet, ((record, str(sheets)) for record in records)))
    (output / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": counts, "priority_keys": summary["priority_keys"],
                      "review_images": len(manifest), "formal_metric_changed": False}), flush=True)


if __name__ == "__main__":
    main()
