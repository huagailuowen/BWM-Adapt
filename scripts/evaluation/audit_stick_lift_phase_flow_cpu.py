"""Opt-in lift-phase/contact-boundary optical-flow experiment. CPU only."""
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
from wan_video_action.real97_eval.stick_marker_flow import MarkerReference, locate_markers

DATA = ROOT.parent.parent / "datasets_real/stick_balance_9_7"
PREP = ROOT / "outputs/evaluation_real97_reference_ttt_v3_20260910/stick_prep_job113867"
INFER = ROOT / "outputs/infer_real97_stick_stage1table_stage2_v3_job113868"
CONFIG = json.loads((ROOT / "configs/evaluation/stick_lift_phase_flow_v1.json").read_text())


def side_index(side):
    if side not in ("left", "right"):
        raise ValueError(f"Unknown box identity: {side!r}")
    return 0 if side == "left" else 1


def points_in_half(points, side, width):
    index = side_index(side)
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        return False
    x = points[:, 0]
    inside = (x >= 0) & (x < width)
    half = x < width / 2 if index == 0 else x >= width / 2
    return bool(np.all(inside & half))


def check_marker_identity(marker, side, width):
    side_index(side)
    if marker is None:
        return
    yy, xx = np.nonzero(marker["mask"])
    for points in (np.column_stack((xx, yy)), marker["sites"], [marker["center"]]):
        if not points_in_half(points, side, width):
            raise RuntimeError(f"Reference box identity violation: {side} outside its image half")


def locate_box(hsv, side):
    marker = locate_markers(hsv, side_index(side))
    check_marker_identity(marker, side, hsv.shape[1])
    return marker


def box_reference(gray, hsv, side, slope):
    reference = MarkerReference(gray, hsv, side_index(side), slope, CONFIG)
    check_marker_identity(reference.reference, side, gray.shape[1])
    if reference.points is not None and not points_in_half(reference.points, side, gray.shape[1]):
        raise RuntimeError(f"Reference optical-flow features crossed the {side} image half")
    return reference


def contact_reference(hsv, side, marker):
    if marker is None:
        return None, "no_reference_marker"
    check_marker_identity(marker, side, hsv.shape[1])
    blue = marker["mask"] > 0
    yy, xx = np.where(blue)
    x0, x1 = max(0, int(xx.min()) - 18), min(640, int(xx.max()) + 19)
    boundary = hsv.shape[1] // 2
    if side_index(side) == 0:
        x1 = min(x1, boundary)
    else:
        x0 = max(x0, boundary)
    y0, y1 = max(0, int(yy.min()) - 8), min(480, int(yy.max()) + 13)
    # Restrict component growth around the box markers; never include the whole arm.
    mask = ((hsv[..., 2] < CONFIG["reference_bottom_dark_value"]) | blue).astype(np.uint8)
    green = cv2.inRange(hsv, np.array([35, 50, 20]), np.array([90, 255, 255]))
    mask[cv2.dilate(green, np.ones((3, 3), np.uint8)) > 0] = 0
    region = np.zeros_like(mask)
    region[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    candidates = [(int(np.count_nonzero(blue & (labels == k))), k) for k in range(1, n)]
    if not candidates or max(candidates)[0] < 25:
        return None, "reference_body_missing"
    _, best = max(candidates)
    body = labels == best
    by, bx = np.where(body)
    if len(bx) < 80:
        return None, "reference_body_too_small"
    # Artificial ROI boundaries, clipping and gripper merges invalidate geometry.
    if bx.min() <= x0 or bx.max() >= x1 - 1 or by.max() >= y1 - 1:
        return None, "reference_contact_clipped_or_roi_cut"
    if stats[best, cv2.CC_STAT_WIDTH] > 1.8 * (xx.max() - xx.min() + 1) + 10:
        return None, "reference_body_merge"
    bottom = []
    for x in range(int(bx.min()), int(bx.max()) + 1):
        ys = np.flatnonzero(body[:, x])
        if len(ys):
            bottom.append((x, int(ys[-1])))
    bottom = np.asarray(bottom, np.float32)
    # Keep a material contour, not extrapolated virtual bounding-box corners.
    bins = np.array_split(np.arange(len(bottom)), min(12, len(bottom)))
    feet = np.array([bottom[b[np.argmax(bottom[b, 1])]] for b in bins if len(b)], np.float32)
    if not points_in_half(feet, side, hsv.shape[1]):
        raise RuntimeError(f"Reference contact points crossed the {side} image half")
    return feet, None


def similarity_from_sites(row):
    p = np.asarray(row["reference_marker_sites"], np.float64)
    q = np.asarray(row["lower_marker_sites"], np.float64)
    u, v = p[1] - p[0], q[1] - q[0]
    denominator = float(u @ u)
    if denominator < 9:
        return None
    a = float(u @ v) / denominator
    b = float(u[0] * v[1] - u[1] * v[0]) / denominator
    matrix = np.array([[a, -b, 0], [b, a, 0], [0, 0, 1]], np.float64)
    matrix[:2, 2] = q[0] - matrix[:2, :2] @ p[0]
    return matrix


class ContactFlow:
    def __init__(self, reference, side, slope, geometry_band):
        hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        marker = locate_box(hsv, side)
        self.feet, self.failure = contact_reference(hsv, side, marker)
        self.side, self.slope = side, slope
        self.geometry_band = geometry_band
        self.anchors = [(box_reference(gray, hsv, side, slope), np.eye(3), 0.0, 0)]
        self.last_anchor = 0

    def update(self, gray, hsv, index):
        if self.feet is None:
            return {"valid": False, "reason": self.failure, "side": self.side}
        failures, best = [], None
        # Prefer direct-reference correspondences, then recent verified anchors.
        for anchor, reference_matrix, accumulated_error, anchor_index in self.anchors:
            row = anchor.update(gray, hsv)
            if row.get("side") != self.side:
                raise RuntimeError(f"Optical-flow box identity mismatch: expected {self.side}, got {row.get('side')}")
            if not row.get("flow_valid"):
                failures.append(row.get("reason", "flow_failure"))
                continue
            if not points_in_half(row["reference_marker_sites"], self.side, gray.shape[1]):
                raise RuntimeError(f"Optical-flow reference sites crossed the {self.side} image half")
            if not all(points_in_half(points, self.side, gray.shape[1]) for points in
                       (row["lower_marker_sites"], [row["center"]])):
                failures.append("tracked_marker_outside_own_half")
                continue
            transform = similarity_from_sites(row)
            if transform is None:
                failures.append("degenerate_marker_sites")
                continue
            transform = transform @ reference_matrix
            error = accumulated_error + float(row["fit_error_p90_px"]) + float(row["fb_error_p90_px"])
            best = (transform, error, anchor_index)
            break
        if best is None:
            return {"valid": False, "side": self.side, "reason": ";".join(failures)}
        transform, error, anchor_index = best
        projected = self.feet @ transform[:2, :2].T + transform[:2, 2]
        if not points_in_half(projected, self.side, gray.shape[1]):
            return {"valid": False, "side": self.side, "reason": "tracked_contacts_outside_own_half"}
        displacement = self.slope * (projected[:, 0] - self.feet[:, 0]) - (projected[:, 1] - self.feet[:, 1])
        visible = (projected[:, 0] >= 1) & (projected[:, 0] < 639) & (projected[:, 1] >= 1) & (projected[:, 1] < 479)
        if index - self.last_anchor >= CONFIG["anchor_period"] and np.all(visible):
            fresh = (box_reference(gray, hsv, self.side, self.slope), transform.copy(), error + 0.25, index)
            self.anchors = [self.anchors[0], fresh] + self.anchors[1:2]
            self.last_anchor = index
        return {
            "valid": True, "side": self.side, "displacement_px": displacement.tolist(),
            "contact_points": projected.tolist(), "reference_points": self.feet.tolist(),
            "minimum_rise_px": float(displacement.min()), "all_contacts_visible": bool(np.all(visible)),
            "uncertainty_px": self.geometry_band + CONFIG["flow_error_floor_px"] + error,
            "anchor_index": anchor_index, "rotation_deg": math.degrees(math.atan2(transform[1, 0], transform[0, 0])),
        }


def decide(sides):
    if [side.get("side") for side in sides] != ["left", "right"]:
        raise RuntimeError("Bilateral lift requires distinct, ordered left and right box identities")
    for side in sides:
        if side.get("valid") and side["all_contacts_visible"]:
            d, band = side["minimum_rise_px"], side["uncertainty_px"]
            # A rise compatible with measurement noise is uncertain, not grounded.
            if d + band < CONFIG["absolute_min_px"]:
                return "below_absolute", None
    if any(not s.get("valid") or not s["all_contacts_visible"] for s in sides):
        return "unknown", None
    d = [s["minimum_rise_px"] for s in sides]
    band = [s["uncertainty_px"] for s in sides]
    if any(v <= e or v < CONFIG["absolute_min_px"] for v, e in zip(d, band)):
        return "unknown", None
    ratio = min(d) / max(d)
    if ratio < CONFIG["minimum_rise_ratio"]:
        return "asymmetric_rise", ratio
    return "balanced_lift_proposal", ratio


def summarize(rows, fps):
    longest, possible, start, possible_start = 0.0, 0.0, None, None
    counts = collections.Counter()
    for r in rows:
        if not r["evaluated"]:
            start = possible_start = None
            continue
        state, t = r["state"], r["time_seconds"]
        counts[state] += 1
        if state == "balanced_lift_proposal":
            start = t if start is None else start
            longest = max(longest, t - start)
        else:
            start = None
        if state in ("balanced_lift_proposal", "unknown"):
            possible_start = t if possible_start is None else possible_start
            possible = max(possible, t - possible_start)
        else:
            possible_start = None
    status = "balanced_lift_proposal" if longest >= CONFIG["hold_seconds"] else "unresolved" if possible >= CONFIG["hold_seconds"] or sum(counts.values()) < 2 else "no_qualifying_lift_observed"
    return {"status": status, "longest_accepted_seconds": longest, "longest_possible_seconds": possible, "frame_counts": dict(counts)}


def montage(frames, rows, record, path):
    indices = set(np.linspace(0, len(frames) - 1, 8).round().astype(int).tolist())
    for state in ("balanced_lift_proposal", "asymmetric_rise", "unknown"):
        pool = [r["frame"] for r in rows if r["evaluated"] and r["state"] == state]
        if pool:
            indices.add(pool[len(pool) // 2])
    indices.add(record["reference_frame"])
    indices = sorted(indices)[:12]
    canvas = np.full((42 + math.ceil(len(indices) / 3) * 386, 1440, 3), 248, np.uint8)
    cv2.putText(canvas, record["key"], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    for j, index in enumerate(indices):
        x, y = (j % 3) * 480, 42 + (j // 3) * 386
        canvas[y:y + 360, x:x + 480] = cv2.resize(frames[index], (480, 360))
        cv2.putText(canvas, "frame %d%s" % (index, " REF" if index == record["reference_frame"] else ""), (x + 8, y + 379), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])


def process(args):
    record, directory, show = args
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260911)
    cap = cv2.VideoCapture(record["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 480)) if frame.shape[:2] != (480, 640) else frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames: " + record["video"])
    source = np.minimum(record["start_frame"] + np.arange(len(frames)) * record["stride"], record["total_frames"] - 1)
    before = np.flatnonzero(source < record["lift_start"])
    ref = int(before[-1]) if len(before) else 0
    record["reference_frame"] = ref
    record["reference_source_frame"] = int(source[ref])
    record["has_pre_lift_reference"] = bool(len(before))
    hsv = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2HSV)
    markers = [locate_box(hsv, side) for side in ("left", "right")]
    slope = 0.0
    if all(m is not None for m in markers):
        left, right = [np.asarray(m["center"]) for m in markers]
        slope = float((right[1] - left[1]) / max(1.0, right[0] - left[0]))
    floor = CONFIG["generated_geometry_band_px"] if record["scope"] == "query" else CONFIG["reference_geometry_band_px"]
    trackers = [ContactFlow(frames[ref], side, slope, floor) for side in ("left", "right")]
    rows, failures = [], collections.Counter()
    for i, frame in enumerate(frames):
        evaluated = bool(source[i] >= record["lift_start"])
        if i < ref or not len(before):
            sides, state, ratio = [], "unknown", None
        else:
            gray, hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sides = [t.update(gray, hsv, i - ref) for t in trackers]
            state, ratio = decide(sides)
            for side in sides:
                if not side.get("valid"):
                    failures[side.get("reason", "unknown")] += 1
        rows.append({"frame": i, "source_frame": int(source[i]), "time_seconds": i / fps, "evaluated": evaluated, "state": state, "ratio": ratio, "sides": sides})
    case = Path(directory) / "cases" / record["key"]
    case.mkdir(parents=True, exist_ok=False)
    with (case / "tracks.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    result = {**record, "decision": summarize(rows, fps), "flow_failures": dict(failures), "camera_compensation": "none_fixed_camera_assumption", "formal_metric_approved": False}
    (case / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if show:
        montage(frames, rows, record, case / "overview.jpg")
        # Contact coordinates and reference frame are recorded for later targeted audit.
        chosen = sorted(set([ref, len(frames) - 1] + [i for i in range(ref, len(frames), max(1, (len(frames) - ref) // 5))]))
        tiles = np.full((len(chosen) * 270 + 36, 1280, 3), 248, np.uint8)
        for j, i in enumerate(chosen):
            raw = frames[i]
            for k, (x0, x1) in enumerate(((0, 256), (384, 640))):
                yy = 280
                if len(rows[i]["sides"]) == 2 and rows[i]["sides"][k].get("valid"):
                    yy = int(np.max(np.asarray(rows[i]["sides"][k]["contact_points"])[:, 1]))
                y0 = min(360, max(0, yy - 70))
                tile = cv2.resize(raw[y0:y0 + 120, x0:x1], (512, 240), interpolation=cv2.INTER_NEAREST)
                tiles[36 + j * 270:36 + j * 270 + 240, k * 640:k * 640 + 512] = tile
                cv2.putText(tiles, "f%d %s y%d%s" % (i, ("L", "R")[k], y0, " REF" if i == ref else ""), (k * 640 + 4, 36 + j * 270 + 260), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
        cv2.imwrite(str(case / "contacts.png"), tiles)
    return result


def main():
    events = [json.loads(line) for line in (DATA / "chunk_events_v1.jsonl").read_text().splitlines() if line.strip()]
    event_map = {(e["environment"], e["episode_index"]): e for e in events}
    queries = [json.loads(line) for line in (PREP / "query.jsonl").read_text().splitlines() if line.strip()]
    reviewed = json.loads((ROOT / "outputs/stick_marker_manual_review_job114222/manifest.json").read_text())
    show_keys = {r["key"] for r in reviewed}
    records = []
    for e in events:
        env, ep = e["environment"], e["episode_index"]
        records.append({"key": "full_%s_ep%06d" % (env, ep), "environment": env, "episode_index": ep, "method": "gt", "scope": "full_episode", "split": e["split"], "video": str(DATA / e["lerobot_dataset"] / "videos/chunk-000/observation.images.image" / ("episode_%06d.mp4" % ep)), "start_frame": 0, "stride": 1, "total_frames": e["num_frames"], "lift_start": e["events"]["lift_start"]["frame"]})
    for q in queries:
        env, ep, sample, split = q["environment"], q["episode_index"], q["sample_index"], q["dataset_split"]
        e = event_map[(env, ep)]
        for method in ("gt", "stage1", "stage2"):
            name = "q%04d_%s_%s_ep%06d.mp4" % (sample, env, split, ep)
            records.append({"key": "q%04d_%s_%s" % (sample, env, method), "environment": env, "episode_index": ep, "method": method, "scope": "query", "split": split, "video": str(INFER / "raw" / method / name), "start_frame": q["start_frame"], "stride": q["frame_stride"], "total_frames": q["total_frames"], "lift_start": e["events"]["lift_start"]["frame"]})
    out = ROOT / ("outputs/evaluation_stick_lift_phase_flow_job" + os.environ["SLURM_JOB_ID"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "config_snapshot.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(process, (r, str(out), r["key"] in show_keys or (r["scope"] == "full_episode" and r["split"] == "test"))): r for r in records}
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result())
            if len(results) % 20 == 0:
                (out / "progress.json").write_text(json.dumps({"done": len(results), "total": len(records)}) + "\n")
    groups, failures = collections.defaultdict(collections.Counter), collections.Counter()
    for r in results:
        groups["/".join((r["scope"], r["split"], r["method"]))][r["decision"]["status"]] += 1
        failures.update(r["flow_failures"])
    summary = {"status": "requires_visual_audit", "videos": len(results), "groups": {k: dict(v) for k, v in sorted(groups.items())}, "flow_failures": dict(failures.most_common(12)), "formal_results_published": False, "training_modified": False}
    (out / "per_video.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
