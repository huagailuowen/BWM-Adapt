"""Experimental visible-bottom detector, independent of optical-flow corners.

This estimates image-space contact, not metric 3-D height. A missing or hidden
bottom stays uncertain. It does not turn every detected box into a lift label.
"""

import cv2
import numpy as np


def detect_body(frame, side):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue = cv2.inRange(hsv, (92, 75, 25), (125, 255, 255))
    blue[:140] = 0
    if side == 0:
        blue[:, int(width * 0.40):] = 0
    else:
        blue[:, :int(width * 0.60)] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    vertical = np.zeros_like(blue)
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        # Flat tabletop tape is not an object seed. Do this before joining.
        if area >= 30 and h >= 12 and w >= 3 and h >= 0.6 * w:
            vertical[labels == label] = 255
    joined = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    candidates = []
    for label in range(1, n):
        x, y, w, h, area = map(int, stats[label])
        seeds = int(np.count_nonzero(vertical[labels == label]))
        if seeds >= 80 and 10 <= w <= 150 and 15 <= h <= 130:
            candidates.append((seeds, label, x, y, w, h))
    if not candidates:
        return None
    _, label, x, y, w, h = max(candidates)
    seed = (labels == label) & (vertical > 0)
    mx, my = max(18, int(w * 0.6)), max(16, int(h * 0.4))
    x0, x1 = max(0, x - mx), min(width, x + w + mx)
    y0, y1 = max(0, y - my), min(height, y + h + my)
    local_hsv = hsv[y0:y1, x0:x1]
    local_blue = blue[y0:y1, x0:x1] > 0
    local_seed = seed[y0:y1, x0:x1]
    value = local_hsv[:, :, 2]
    saturation = local_hsv[:, :, 1]
    background = value[(saturation < 70) & (value > 70)]
    bg_value = float(np.percentile(background, 75)) if len(background) else 170.0
    dark_limit = min(110.0, max(65.0, 0.62 * bg_value))
    neutral_dark = (saturation < 150) & (value < dark_limit)
    mask = ((neutral_dark | local_blue) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, component, _, _ = cv2.connectedComponentsWithStats(mask)
    overlaps = [(int(np.count_nonzero(local_seed & (component == k))), k)
                for k in range(1, n)]
    if not overlaps:
        return None
    overlap, chosen = max(overlaps)
    if overlap < 60:
        return None
    body = component == chosen
    ys, xs = np.nonzero(body)
    if len(xs) < 150:
        return None
    full_x, full_y = xs + x0, ys + y0
    xs_unique = np.unique(xs)
    profile = []
    for col in xs_unique:
        column = ys[xs == col]
        if len(column) >= 6 and column.max() - column.min() >= 8:
            profile.append((int(col + x0), int(column.max() + y0)))
    if len(profile) < 10:
        return None
    lower = np.asarray(profile, dtype=np.float32)
    # Snap only near the observed silhouette to a real dark-to-light edge.
    # Never extrapolate a fitted line into the empty space beside a box.
    subpixel, strengths = [], []
    for px, py in lower:
        xx, yy = int(px), int(py)
        sample_y = np.arange(max(1, yy - 3), min(height - 1, yy + 4))
        if len(sample_y) < 3:
            continue
        gradient = gray[sample_y + 1, xx].astype(float) - gray[sample_y - 1, xx].astype(float)
        peak = int(np.argmax(gradient))
        delta = 0.0
        if 0 < peak < len(gradient) - 1:
            a, b, c = gradient[peak - 1:peak + 2]
            denom = a - 2 * b + c
            if abs(denom) > 1e-6:
                delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        subpixel.append([float(px), float(sample_y[peak]) + delta])
        strengths.append(float(gradient[peak]))
    if len(subpixel) < 10:
        return None
    subpixel = np.asarray(subpixel, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    # Preserve weak edges as uncertainty, rather than treating them as clear gaps.
    hull = cv2.convexHull(np.stack([full_x, full_y], axis=1).astype(np.int32)).reshape(-1, 2)
    seed_y, seed_x = np.nonzero(seed)
    clipped = bool(full_x.min() <= 1 or full_x.max() >= width - 2)
    crop_cut = bool((xs.min() == 0 and x0 > 0) or (xs.max() == body.shape[1] - 1 and x1 < width)
                    or (ys.max() == body.shape[0] - 1 and y1 < height))
    green = cv2.inRange(hsv, (35, 60, 30), (90, 255, 255))
    covered = []
    for px, py in subpixel:
        xx, yy = int(round(px)), int(round(py))
        region = green[max(0, yy-4):min(height, yy+5), max(0, xx-3):min(width, xx+4)]
        covered.append(bool(region.size and np.mean(region > 0) > 0.15))
    return {
        "center": [float(seed_x.mean()), float(seed_y.mean())],
        "lower": subpixel, "edge_strength": strengths,
        "edge_occluded": np.asarray(covered, dtype=bool),
        "hull": hull.tolist(), "area": int(len(xs)), "clipped": clipped,
        "crop_cut": crop_cut, "dark_threshold": dark_limit,
        "seed_coverage": overlap / max(1, int(local_seed.sum())),
    }


def bottom_measurement(body, slope, intercept):
    points = body["lower"]
    distances = slope * points[:, 0] + intercept - points[:, 1]
    # Use the actual lowest contour neighborhood; a raised opposite corner
    # cannot hide the near-ground corner in an average over the whole box.
    order = np.argsort(distances)
    count = max(2, int(np.ceil(len(order) * 0.04)))
    contact = order[:count]
    return {
        "gap": float(np.median(distances[contact])),
        "points": points[contact].tolist(),
        "strength": float(np.median(body["edge_strength"][contact])),
        "occluded": bool(np.any(body["edge_occluded"][contact])),
    }


class VisibleBottomContact:
    def __init__(self, reference_frame):
        self.reference = [detect_body(reference_frame, side) for side in range(2)]
        self.reference_offsets = [None, None]
        self.initial_offsets = [None, None]
        self.slope = self.intercept = self.angle0 = None
        self.initial_angle_offset = None
        if any(body is None for body in self.reference):
            return
        a, b = self.reference
        x0, x1 = a["center"][0], b["center"][0]
        y0 = float(np.percentile(a["lower"][:, 1], 96))
        y1 = float(np.percentile(b["lower"][:, 1], 96))
        self.slope = (y1 - y0) / max(1.0, x1 - x0)
        self.intercept = y0 - self.slope * x0
        self.angle0 = float(np.degrees(np.arctan2(b["center"][1] - a["center"][1], x1 - x0)))
        for i, body in enumerate(self.reference):
            self.reference_offsets[i] = bottom_measurement(body, self.slope, self.intercept)["gap"]

    def update(self, frame):
        bodies = [detect_body(frame, side) for side in range(2)]
        sides = []
        for i, body in enumerate(bodies):
            row = {"side": "left" if i == 0 else "right", "state": "uncertain",
                   "body_detected": body is not None}
            if body is None or self.slope is None:
                row["reason"] = "missing_body_or_reference"
                sides.append(row)
                continue
            observed = bottom_measurement(body, self.slope, self.intercept)
            raw = observed["gap"] - self.reference_offsets[i]
            if self.initial_offsets[i] is None:
                self.initial_offsets[i] = raw
            displacement = min(raw, raw - self.initial_offsets[i])
            # Deliberately no claim of calibrated subpixel physical accuracy.
            band = 1.0
            reliable = observed["strength"] >= 12 and not observed["occluded"] and not body["crop_cut"]
            if reliable and abs(displacement) <= band:
                row["state"] = "contact_consistent"
            elif reliable and displacement > band and not body["clipped"]:
                row["state"] = "airborne_proposal"
            row.update(
                gap_px=displacement, raw_gap_px=raw, first_frame_offset_px=self.initial_offsets[i],
                uncertainty_margin_px=band, contact_points=observed["points"],
                edge_strength=observed["strength"], edge_occluded=observed["occluded"],
                clipped=body["clipped"], crop_cut=body["crop_cut"],
                hull=body["hull"], center=body["center"],
                dark_threshold=body["dark_threshold"], seed_coverage=body["seed_coverage"],
            )
            sides.append(row)
        angle = None
        if all(body is not None for body in bodies) and self.angle0 is not None:
            a, b = [body["center"] for body in bodies]
            raw_angle = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))) - self.angle0
            if self.initial_angle_offset is None:
                self.initial_angle_offset = raw_angle
            angle = raw_angle - self.initial_angle_offset
        states = [side["state"] for side in sides]
        state = "uncertain"
        if all(value == "airborne_proposal" for value in states) and angle is not None:
            state = "both_airborne_balanced" if abs(angle) <= 5 else "both_airborne_tilted"
        elif "contact_consistent" in states:
            state = "visible_endpoint_contact_evidence"
        return {"state": state, "sides": sides, "angle_change_deg": angle,
                "ground_line": [self.slope, self.intercept],
                "meaning": "experimental image-space visible contact, not calibrated 3D height"}
