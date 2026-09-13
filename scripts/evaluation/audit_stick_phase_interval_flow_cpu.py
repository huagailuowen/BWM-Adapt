"""CPU-only, opt-in interval evidence over phase-referenced box optical flow.

No training changes, no replacement of previous detectors, no formal metrics.
The marker-footprint fallback is explicitly a proxy, not a physical contact label.
"""
import collections
import concurrent.futures
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_stick_lift_phase_flow_cpu as base
from wan_video_action.real97_eval.stick_terminal_criteria import terminal_lift_summary

CONFIG = json.loads(Path(os.environ.get("STICK_PHASE_FLOW_CONFIG", str(ROOT / "configs/evaluation/stick_phase_interval_flow_v1.json"))).read_text())
SOURCE = ROOT / "outputs/evaluation_stick_lift_phase_flow_job114226"
REVIEW = ROOT / "outputs/stick_marker_manual_review_job114222/manifest.json"


class IntervalFlow(base.ContactFlow):
    def __init__(self, reference, side, slope, geometry_band):
        super().__init__(reference, side, slope, geometry_band)
        self.geometry_kind = "sampled_reference_lower_contour"
        self.full_contact_geometry_visible = True
        self.reference_error = float(geometry_band)
        self.measurement_floor = CONFIG["query_flow_floor_px" if geometry_band > 1 else "native_flow_floor_px"]
        self.original_failure = self.failure
        if self.failure != "reference_contact_clipped_or_roi_cut":
            return
        hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        marker = base.locate_box(hsv, side)
        if marker is None:
            return
        y, x = np.nonzero(marker["mask"])
        margin = CONFIG["fallback_contact_horizontal_margin_px"]
        x0, x1 = float(x.min()) - margin, float(x.max()) + margin
        sites = np.asarray(marker["sites"], np.float64)
        sites = sites[np.argsort(sites[:, 0])]
        if sites[1, 0] - sites[0, 0] < 3:
            self.failure = "reference_marker_span_too_small"
            return
        slope_bottom = (sites[1, 1] - sites[0, 1]) / (sites[1, 0] - sites[0, 0])
        xs = np.linspace(x0, x1, 8)
        ys = sites[0, 1] + slope_bottom * (xs - sites[0, 0]) + CONFIG["fallback_contact_offset_px"]
        feet = np.column_stack((xs, ys)).astype(np.float32)
        if (not base.points_in_half(feet, side, reference.shape[1]) or
                x0 < 2 or x1 >= reference.shape[1] - 2 or
                ys.min() < 2 or ys.max() >= reference.shape[0] - 2):
            if not CONFIG.get("allow_partial_contact_negative_witness", False):
                self.failure = "reference_footprint_actually_clipped"
                return
            # Keep observable material sites without pretending an unseen outer
            # foot exists inside the image. Partial geometry cannot certify a
            # positive lift or provide the larger-end lower bound for asymmetry.
            observed_x0 = max(2.0, float(x.min()) + 2.0)
            observed_x1 = min(reference.shape[1] - 3.0, float(x.max()) - 2.0)
            if observed_x1 - observed_x0 < 9:
                self.failure = "insufficient_visible_reference_span"
                return
            xs = np.linspace(observed_x0, observed_x1, 8)
            ys = sites[0, 1] + slope_bottom * (xs - sites[0, 0]) + CONFIG["fallback_contact_offset_px"]
            feet = np.column_stack((xs, ys)).astype(np.float32)
            if (not base.points_in_half(feet, side, reference.shape[1]) or
                    ys.min() < 2 or ys.max() >= reference.shape[0] - 2):
                self.failure = "visible_reference_geometry_unavailable"
                return
            self.full_contact_geometry_visible = False
        self.feet, self.failure = feet, None
        self.geometry_kind = ("bounded_marker_footprint_proxy" if self.full_contact_geometry_visible
                              else "partial_visible_marker_negative_witness_only")
        self.reference_error = CONFIG["fallback_reference_uncertainty_px"]

    def update(self, gray, hsv, index):
        row = super().update(gray, hsv, index)
        row["geometry_kind"] = self.geometry_kind
        row["reference_segmentation_failure"] = self.original_failure
        row["full_contact_geometry_visible"] = self.full_contact_geometry_visible
        row["physical_contact_certified"] = False
        if not row.get("valid"):
            return row
        p = np.asarray(row["reference_points"], np.float64)
        q = np.asarray(row["contact_points"], np.float64)
        i, j = np.unravel_index(np.argmax(np.sum((p[:, None] - p[None, :]) ** 2, axis=2)), (len(p), len(p)))
        u, v = p[j] - p[i], q[j] - q[i]
        denominator = float(u @ u)
        if denominator < 9:
            return {**row, "valid": False, "reason": "degenerate_reference_footprint"}
        a = float(u @ v) / denominator
        b = float(u[0] * v[1] - u[1] * v[0]) / denominator
        linear = np.array([[a, -b], [b, a]])
        normal = np.array([self.slope, -1.0])
        # Reference localization error is shared by both times. Its displacement
        # contribution is n^T(A-I)e, not an independent constant error per frame.
        geometry_error = self.reference_error * float(np.linalg.norm(normal @ (linear - np.eye(2))))
        fit_error = max(0.0, row["uncertainty_px"] - self.geometry_band - base.CONFIG["flow_error_floor_px"])
        row["uncertainty_px"] = self.measurement_floor + fit_error + geometry_error
        row["fit_and_anchor_error_px"] = fit_error
        row["propagated_geometry_error_px"] = geometry_error
        row["scale"] = math.hypot(a, b)
        row["rise_lower_px"] = row["minimum_rise_px"] - row["uncertainty_px"]
        row["rise_upper_px"] = row["minimum_rise_px"] + row["uncertainty_px"]
        return row


def decide(sides):
    if [s.get("side") for s in sides] != ["left", "right"]:
        raise RuntimeError("Two distinct ordered box identities are required")
    visible = [s.get("valid") and s.get("all_contacts_visible") for s in sides]
    for ok, s in zip(visible, sides):
        if ok and s["rise_upper_px"] < CONFIG["absolute_min_px"]:
            return "below_absolute", None, None, None
    if not all(visible):
        return "unknown_tracking", None, None, None
    lo = np.array([s["rise_lower_px"] for s in sides])
    hi = np.array([s["rise_upper_px"] for s in sides])
    d = np.array([s["minimum_rise_px"] for s in sides])
    ratio = float(max(0, d.min()) / d.max()) if d.max() > 0 else None
    ratio_low = float(max(0, lo.min()) / hi.max()) if hi.max() > 0 else 0.0
    ratio_high = float(min(1.0, max(0, hi.min()) / lo.max())) if lo.max() > 0 else 1.0
    r = CONFIG["minimum_rise_ratio"]
    larger_end_fully_observed = sides[int(np.argmax(lo))].get("full_contact_geometry_visible", True)
    if larger_end_fully_observed and lo.max() >= CONFIG["absolute_min_px"] and hi.min() < r * lo.max():
        return "asymmetric_rise", ratio, ratio_low, ratio_high
    if not all(s.get("full_contact_geometry_visible", True) for s in sides):
        return "unknown_partial_geometry", ratio, ratio_low, ratio_high
    if lo.min() >= CONFIG["absolute_min_px"] and lo.min() >= r * hi.max():
        return "balanced_lift_proposal", ratio, ratio_low, ratio_high
    return "unknown_resolution_or_boundary", ratio, ratio_low, ratio_high


def run_length(rows, allowed):
    start, longest, interval = None, 0.0, None
    for row in rows:
        if row["state"] in allowed:
            if start is None:
                start = row
            duration = row["time_seconds"] - start["time_seconds"]
            if duration > longest:
                longest, interval = duration, [start["frame"], row["frame"]]
        else:
            start = None
    return longest, interval


def summarize(rows, record=None):
    if CONFIG.get("success_time_scope") == "terminal_lift_phase":
        if record is None:
            raise ValueError("Terminal criterion requires the source clock and phase metadata")
        return terminal_lift_summary(rows, record, CONFIG["hold_seconds"])
    evaluated = [r for r in rows if r["evaluated"]]
    accepted, interval = run_length(evaluated, {"balanced_lift_proposal"})
    possible, _ = run_length(evaluated, {"balanced_lift_proposal", "unknown_tracking", "unknown_resolution_or_boundary", "unknown_partial_geometry"})
    negative, negative_interval = run_length(evaluated, {"asymmetric_rise", "below_absolute"})
    if accepted + 1e-8 >= CONFIG["hold_seconds"]:
        status = "balanced_lift_proposal"
    elif len(evaluated) < 2 or possible + 1e-8 >= CONFIG["hold_seconds"]:
        status = "unresolved"
    else:
        status = "no_qualifying_lift_observed"
    return {
        "status": status, "longest_accepted_seconds": accepted,
        "accepted_interval": interval, "longest_possible_seconds": possible,
        "longest_negative_seconds": negative, "negative_interval": negative_interval,
        "frame_counts": dict(collections.Counter(r["state"] for r in evaluated)),
        "bilateral_tracking_fraction": sum(all(s.get("valid") for s in r["sides"]) and len(r["sides"]) == 2 for r in evaluated) / max(1, len(evaluated)),
        "note": "No negative promotion by majority voting or by ignoring early unknown intervals"
    }


def draw_chart(rows, path, key):
    width, height = 1100, 480
    canvas = np.full((height, width, 3), 250, np.uint8)
    values = [s[k] for r in rows for s in r["sides"] if s.get("valid") for k in ("rise_lower_px", "rise_upper_px")]
    lower, upper = min([-5] + values), max([10] + values)
    times = [r["time_seconds"] for r in rows]
    t0, t1 = times[0], max(times[-1], times[0] + .01)
    def point(t, v):
        return int(65 + 995 * (t - t0) / (t1 - t0)), int(420 - 350 * (v - lower) / max(1, upper - lower))
    cv2.putText(canvas, key, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
    for v in np.linspace(lower, upper, 6):
        y = point(t0, v)[1]
        cv2.line(canvas, (65, y), (1060, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{v:.1f}", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1)
    for side, color in enumerate(((220, 95, 35), (35, 90, 225))):
        previous = None
        for row in rows:
            s = row["sides"][side]
            if not s.get("valid"):
                previous = None
                continue
            t = row["time_seconds"]
            p = point(t, s["minimum_rise_px"])
            cv2.line(canvas, point(t, s["rise_lower_px"]), point(t, s["rise_upper_px"]), tuple(int(.45*c + .55*255) for c in color), 1)
            if previous is not None:
                cv2.line(canvas, previous, p, color, 2)
            previous = p
        cv2.putText(canvas, "left" if side == 0 else "right", (700 + 150 * side, 45), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    cv2.putText(canvas, "time (seconds); rise in native 640x480 pixels; bars = error bounds", (65, 458), cv2.FONT_HERSHEY_SIMPLEX, .52, (35, 35, 35), 1)
    cv2.imwrite(str(path), canvas)


def evidence(frames, rows, result, directory):
    ref = result["reference_frame"]
    choices = sorted(set([ref, len(frames) - 1] + np.linspace(ref, len(frames) - 1, 7, dtype=int).tolist()))
    sheets = np.full((len(choices) * 284 + 30, 960, 3), 245, np.uint8)
    cv2.putText(sheets, result["key"] + "  fixed y=120:480; left / right", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1)
    for j, index in enumerate(choices):
        frame = frames[index].copy()
        for s in rows[index]["sides"]:
            if s.get("valid"):
                pts = np.rint(s["contact_points"]).astype(np.int32)
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 1)
        for side in range(2):
            crop = frames[index][120:480, side * 320:(side + 1) * 320]
            crop = cv2.resize(crop, (240, 270))
            overlay = cv2.resize(frame[120:480, side * 320:(side + 1) * 320], (240, 270))
            top, left = 30 + j * 284, side * 480
            sheets[top:top + 270, left:left + 240] = crop
            sheets[top:top + 270, left + 240:left + 480] = overlay
            cv2.putText(sheets, f"f{index} raw | track {rows[index]['state']}", (left + 2, top + 282), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
    cv2.imwrite(str(directory / "fixed_coordinates.jpg"), sheets)
    draw_chart(rows, directory / "rise_intervals.png", result["key"])
    interval = (result["decision"]["accepted_interval"] or result["decision"]["negative_interval"]
                or result["decision"].get("terminal_window", {}).get("frame_interval"))
    center = sum(interval) // 2 if interval else (ref + len(frames) - 1) // 2
    dense = list(range(max(ref, center - 5), min(len(frames), center + 7)))
    tile = np.full((math.ceil(len(dense) / 3) * 256 + 28, 960, 3), 245, np.uint8)
    cv2.putText(tile, "Consecutive frames, no time subsampling: " + result["key"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1)
    for j, index in enumerate(dense):
        y, x = 28 + (j // 3) * 256, (j % 3) * 320
        tile[y:y + 240, x:x + 320] = cv2.resize(frames[index], (320, 240))
        cv2.putText(tile, f"f{index} {rows[index]['state']}", (x + 2, y + 253), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
    cv2.imwrite(str(directory / "consecutive_frames.jpg"), tile)


def process(item):
    record, output, show = item
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(record["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 480)))
    cap.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Cannot decode video: {record['video']}")
    source = np.minimum(record["start_frame"] + np.arange(len(frames)) * record["stride"], record["total_frames"] - 1)
    before = np.flatnonzero(source < record["lift_start"])
    ref = int(before[-1]) if len(before) else 0
    hsv = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2HSV)
    markers = [base.locate_box(hsv, side) for side in ("left", "right")]
    slope = 0.0
    if all(m is not None for m in markers):
        l, r = [np.asarray(m["center"]) for m in markers]
        slope = float((r[1] - l[1]) / max(1, r[0] - l[0]))
    geometry = base.CONFIG["generated_geometry_band_px"] if record["scope"] == "query" else base.CONFIG["reference_geometry_band_px"]
    trackers = [IntervalFlow(frames[ref], side, slope, geometry) for side in ("left", "right")]
    rows, failures = [], collections.Counter()
    for i, frame in enumerate(frames):
        ratio = ratio_low = ratio_high = None
        if i < ref or not len(before):
            sides = [{"side": s, "valid": False, "reason": "no_pre_lift_reference" if not len(before) else "before_reference"} for s in ("left", "right")]
            state = "unknown_tracking"
        else:
            gray, hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sides = [t.update(gray, hsv, i - ref) for t in trackers]
            state, ratio, ratio_low, ratio_high = decide(sides)
        evaluated = bool(source[i] >= record["lift_start"])
        if evaluated:
            failures.update(s["reason"] for s in sides if not s.get("valid"))
        time_seconds = (i * record["stride"] / record["source_fps"]
                        if CONFIG.get("success_time_scope") == "terminal_lift_phase" else i / fps)
        rows.append({"frame": i, "source_frame": int(source[i]), "time_seconds": time_seconds,
                     "evaluated": evaluated, "state": state, "ratio": ratio,
                     "ratio_lower": ratio_low, "ratio_upper": ratio_high, "sides": sides})
    result = {**record, "reference_frame": ref, "reference_source_frame": int(source[ref]),
              "has_pre_lift_reference": bool(len(before)), "decision": summarize(rows, record),
              "flow_failures": dict(failures), "formal_metric_approved": False,
              "measurement_kind": "image_motion_with_geometry_and_registration_error_bounds"}
    directory = Path(output) / "cases" / record["key"]
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "tracks.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if show:
        evidence(frames, rows, result, directory)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this bulk video audit on a CPU compute node through Slurm")
    source_records = json.loads((SOURCE / "per_video.json").read_text())
    fields = ("key", "environment", "episode_index", "method", "scope", "split", "video", "start_frame", "stride", "total_frames", "lift_start")
    records = [{k: r[k] for k in fields} for r in source_records]
    if CONFIG.get("success_time_scope") == "terminal_lift_phase":
        events = [json.loads(line) for line in (base.DATA / "chunk_events_v1.jsonl").read_text().splitlines() if line.strip()]
        by_episode = {(e["environment"], e["episode_index"]): e for e in events}
        for record in records:
            event = by_episode[(record["environment"], record["episode_index"])]
            record["source_fps"] = float(event["fps"])
            record["lift_last_frame"] = int(event["events"]["lift_last_frame"]["frame"])
    out = ROOT / f"outputs/evaluation_stick_phase_interval_flow_job{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    selected = {"full_stick-L0-R0_ep000027", "full_stick-L0-R0_ep000020", "full_stick-L3left-R0_ep000027", "full_stick-L0-R1right_ep000022", "full_stick-L2left-R0_ep000025", "full_stick-L0-R3right_ep000014"}
    items = [(r, str(out), r["key"] in selected or r["scope"] == "query" or r["split"] == "test") for r in records]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(process, items):
            results.append(result)
            if len(results) % 20 == 0:
                print(f"completed {len(results)}/{len(records)}", flush=True)
                (out / "progress.json").write_text(json.dumps({"done": len(results), "total": len(records)}))
    groups = collections.defaultdict(collections.Counter)
    fractions = collections.defaultdict(list)
    for r in results:
        group = "/".join(r[k] for k in ("scope", "split", "method"))
        groups[group][r["decision"]["status"]] += 1
        fractions[group].append(r["decision"]["bilateral_tracking_fraction"])
    summary = {"status": "requires_visual_audit", "videos": len(results), "groups": dict(groups),
               "mean_bilateral_tracking_fraction": {k: float(np.mean(v)) for k, v in fractions.items()},
               "flow_failures": dict(sum((collections.Counter(r["flow_failures"]) for r in results), collections.Counter())),
               "tracking_coverage_is_not_decision_accuracy": True, "formal_results_published": False}
    (out / "per_video.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
