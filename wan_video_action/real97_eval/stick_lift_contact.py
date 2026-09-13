"""Split-blind, auditable two-ended lift proposals in native 640x480 pixels.

This is a monocular image-plane contact proxy, not proof of hidden 3-D contact.
Track the lower silhouette, not just the center of the blue endpoint boxes.
Thresholds are provisional until the generated review images are audited.
"""

import math

import cv2
import numpy as np


def endpoint_bodies(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue = ((hue >= 92) & (hue <= 125) & (saturation >= 75) & (value >= 30)).astype(np.uint8)
    blue[:150] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    seeds = np.zeros_like(blue)
    # Remove small blue table markers before joining the tall tape strips.
    for index in range(1, n):
        x, y, w, h, area = stats[index]
        if area >= 35 and h >= 13 and w >= 3:
            seeds[labels == index] = 1
    joined = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, np.ones((3, 17), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(joined, 8)
    candidates = []
    for index in range(1, n):
        x, y, w, h, area = map(int, stats[index])
        if area >= 100 and 15 <= w <= 140 and 14 <= h <= 110:
            candidates.append((area, index, x, y, w, h, centers[index]))
    candidates.sort(reverse=True, key=lambda item: item[0])
    if len(candidates) < 2:
        return None
    pair = sorted(candidates[:2], key=lambda item: item[-1][0])
    if np.linalg.norm(pair[1][-1] - pair[0][-1]) < 180:
        return None
    result = []
    for _, label, x, y, w, h, center in pair:
        x0, x1 = max(0, x - 10), min(frame.shape[1], x + w + 10)
        y0, y1 = max(0, y - 6), min(frame.shape[0], y + h + 9)
        seed = joined[y0:y1, x0:x1].astype(bool) & (labels[y0:y1, x0:x1] == label)
        neutral_body = (saturation[y0:y1, x0:x1] < 115) & (value[y0:y1, x0:x1] < 140)
        mask = cv2.morphologyEx((seed | neutral_body).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, body_labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
        body = np.zeros_like(mask, dtype=bool)
        for index in range(1, count):
            component = body_labels == index
            if np.count_nonzero(component & seed) >= 20:
                body |= component
        yy, xx = np.nonzero(body)
        if len(xx) < 100:
            return None
        lower_x, lower_y = [], []
        for column in np.unique(xx):
            lower_x.append(int(column + x0))
            lower_y.append(int(np.max(yy[xx == column]) + y0))
        hull = cv2.convexHull(np.column_stack([xx + x0, yy + y0]).astype(np.int32)).reshape(-1, 2)
        result.append({
            "center": np.asarray(center, dtype=float),
            "lower_x": np.asarray(lower_x, dtype=float),
            "lower_y": np.asarray(lower_y, dtype=float),
            "hull": hull.tolist(), "area": int(len(xx)),
            "clipped": bool(np.min(xx) + x0 <= 1 or np.max(xx) + x0 >= frame.shape[1] - 2),
        })
    return result


def _clearances(body, slope, intercept):
    gaps = slope * body["lower_x"] + intercept - body["lower_y"]
    thirds = [float(np.quantile(part, .05)) for part in np.array_split(gaps, 3) if len(part)]
    return min(float(np.quantile(gaps, .02)), min(thirds)), thirds


class StickLiftContact:
    def __init__(self, initial_frame):
        self.reference = endpoint_bodies(initial_frame)
        self.slope = self.intercept = self.angle0 = None
        self.offsets = []
        if self.reference is None:
            return
        left, right = self.reference
        x = [float(body["center"][0]) for body in self.reference]
        y = [float(np.quantile(body["lower_y"], .98)) for body in self.reference]
        self.slope = (y[1] - y[0]) / (x[1] - x[0])
        self.intercept = y[0] - self.slope * x[0]
        vector = right["center"] - left["center"]
        self.angle0 = float(np.degrees(np.arctan2(vector[1], vector[0])))
        self.span0 = float(np.linalg.norm(vector))
        self.offsets = [_clearances(body, self.slope, self.intercept)[0] for body in self.reference]

    def update(self, frame):
        bodies = endpoint_bodies(frame)
        if self.reference is None or bodies is None:
            return {"valid": False, "reason": "missing_endpoint_or_rest_reference"}
        vector = bodies[1]["center"] - bodies[0]["center"]
        span = float(np.linalg.norm(vector))
        angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
        delta = (angle - self.angle0 + 180) % 360 - 180
        gaps, thirds, areas = [], [], []
        for body, reference, offset in zip(bodies, self.reference, self.offsets):
            minimum, parts = _clearances(body, self.slope, self.intercept)
            gaps.append(float(minimum - offset))
            thirds.append([float(part - offset) for part in parts])
            areas.append(body["area"] / reference["area"])
        clipped = any(body["clipped"] for body in bodies + self.reference)
        geometry_ok = .75 <= span / self.span0 <= 1.15 and all(.5 <= ratio <= 1.7 for ratio in areas)
        return {
            "valid": bool(not clipped and geometry_ok),
            "reason": "clipped_contact_edge" if clipped else ("unstable_body_geometry" if not geometry_ok else "visible_contact_proxy"),
            "left_clearance_px": gaps[0], "right_clearance_px": gaps[1],
            "minimum_clearance_px": min(gaps), "lower_edge_thirds_px": thirds,
            "angle_change_deg": delta, "absolute_angle_deg": angle,
            "ground_line": [self.slope, self.intercept],
            "centers": [body["center"].tolist() for body in bodies],
            "body_hulls": [body["hull"] for body in bodies],
            "body_area_ratios": areas,
        }


def longest_run(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def classify_lift(rows, fps, *, clearance_px=3.0, tilt_deg=5.0, hold_seconds=.3):
    considered = [row for row in rows if row.get("evaluated", True)]
    required = max(2, int(math.ceil(hold_seconds * fps - 1e-6)) + 1)
    accepted, possible = [], []
    for row in considered:
        known = row.get("valid", False)
        ok = bool(known and row["minimum_clearance_px"] >= clearance_px and abs(row["angle_change_deg"]) <= tilt_deg)
        accepted.append(ok)
        possible.append(ok or not known)
    positive_run = longest_run(accepted)
    if positive_run >= required:
        status = "success"
    elif len(considered) < required or longest_run(possible) >= required:
        status = "unknown"
    else:
        status = "failure"
    return {
        "status": status, "evaluated_frames": len(considered),
        "valid_frames": sum(row.get("valid", False) for row in considered),
        "coverage": sum(row.get("valid", False) for row in considered) / max(1, len(considered)),
        "accepted_run_frames": positive_run, "required_run_frames": required,
        "accepted_duration_seconds": max(0, positive_run - 1) / fps,
        "clearance_threshold_px": clearance_px, "tilt_threshold_deg": tilt_deg,
        "hold_seconds": hold_seconds,
    }
