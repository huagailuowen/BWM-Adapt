"""Independent image-space checks for terminal lift review, not a 3D oracle."""

import math

import cv2
import numpy as np

from .stick_marker_flow import locate_markers


def body_candidates(frame, side, config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    marker = locate_markers(hsv, side)
    if marker is None:
        return {"valid": False, "reason": "no_marker_seed", "candidates": []}
    height, width = frame.shape[:2]
    yy, xx = np.nonzero(marker["mask"])
    half_lo, half_hi = (0, width // 2) if side == 0 else (width // 2, width)
    x0 = max(half_lo, int(xx.min()) - config["roi_pad_x"])
    x1 = min(half_hi, int(xx.max()) + config["roi_pad_x"] + 1)
    y0 = max(140, int(yy.min()) - config["roi_pad_top"])
    y1 = min(height, int(yy.max()) + config["roi_pad_bottom"] + 1)
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[y0:y1, x0:x1] = 255
    blue = marker["mask"] > 0
    green = cv2.inRange(hsv, (35, 60, 25), (90, 255, 255)) > 0
    candidates = []
    for cutoff in config["dark_value_thresholds"]:
        binary = (((hsv[:, :, 2] <= cutoff) | blue) & (roi > 0) & ~green).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        scored = [(int(np.count_nonzero((labels == index) & blue)), index) for index in range(1, count)]
        if not scored or max(scored)[0] < 20:
            continue
        _, chosen = max(scored)
        dark = ((labels == chosen) & (hsv[:, :, 2] <= cutoff) & ~blue).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        by, bx = np.nonzero(dark)
        if len(bx) < config["minimum_dark_pixels"]:
            continue
        hull = cv2.convexHull(np.column_stack((bx, by)).astype(np.float32))[:, 0]
        x, y, w, h = cv2.boundingRect(hull)
        if w < 10 or h < 12 or w > 190 or h > 145:
            continue
        clipped = bool(x <= x0 + 1 or x + w >= x1 - 1 or y <= y0 + 1 or y + h >= y1 - 1)
        candidates.append({"threshold": cutoff, "hull": hull.tolist(), "bbox": [x, y, w, h],
                           "clipped": clipped, "dark_pixels": len(bx)})
    return {"valid": len(candidates) >= config["minimum_threshold_agreement"],
            "reason": None if len(candidates) >= config["minimum_threshold_agreement"] else "insufficient_body_threshold_agreement",
            "marker_center": marker["center"].tolist(), "roi": [x0, y0, x1, y1], "candidates": candidates}


def clearance(reference, current, slope, config):
    if not reference["valid"] or not current["valid"]:
        return {"valid": False, "reason": "body_segmentation_unavailable"}
    original = {candidate["threshold"]: candidate for candidate in reference["candidates"]}
    values = []
    full = True
    for candidate in current["candidates"]:
        prior = original.get(candidate["threshold"])
        if prior is None:
            continue
        a, b = np.asarray(prior["hull"]), np.asarray(candidate["hull"])
        reference_bottom = float(np.max(a[:, 1] - slope * a[:, 0]))
        current_bottom = float(np.max(b[:, 1] - slope * b[:, 0]))
        values.append({"threshold": candidate["threshold"], "gap_px": reference_bottom - current_bottom})
        full = full and not prior["clipped"] and not candidate["clipped"]
    if len(values) < config["minimum_threshold_agreement"]:
        return {"valid": False, "reason": "insufficient_matched_body_thresholds"}
    gaps = [value["gap_px"] for value in values]
    spread = max(gaps) - min(gaps)
    band = config["silhouette_error_floor_px"]
    return {"valid": spread <= config["maximum_threshold_gap_spread_px"],
            "reason": None if spread <= config["maximum_threshold_gap_spread_px"] else "body_boundary_threshold_sensitive",
            "gap_by_threshold": values, "gap_median_px": float(np.median(gaps)),
            "gap_lower_px": min(gaps) - band, "gap_upper_px": max(gaps) + band,
            "gap_spread_px": spread, "full_geometry": full,
            "measurement_kind": "current_dark_body_lower_envelope_relative_to_initial_local_floor_line",
            "physical_clearance_certified": False}


def rod_angle(frame, bodies, config):
    if not all(body["valid"] for body in bodies):
        return {"valid": False, "reason": "no_bilateral_body_rois"}
    chosen = [min(body["candidates"], key=lambda candidate: abs(candidate["threshold"] - 85)) for body in bodies]
    endpoints = []
    for candidate in chosen:
        x, y, w, h = candidate["bbox"]
        endpoints.append(np.asarray([x + w / 2.0, y + config["rod_attachment_offset_px"]]))
    left, right = endpoints
    if right[0] - left[0] < 100:
        return {"valid": False, "reason": "implausible_endpoint_span"}
    height, width = frame.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    expected = left[1] + (right[1] - left[1]) * (xx - left[0]) / (right[0] - left[0])
    corridor = (xx >= left[0] + 12) & (xx <= right[0] - 12) & (np.abs(yy - expected) <= config["rod_corridor_half_width_px"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(hsv, (15, 65, 25), (90, 255, 255))
    corridor &= cv2.dilate(colored, np.ones((7, 7), np.uint8)) == 0
    for candidate in chosen:
        x, y, w, h = candidate["bbox"]
        corridor[max(0, y - 3):min(height, y + h + 3), max(0, x - 3):min(width, x + w + 3)] = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    edges[~corridor] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720.0, threshold=20, minLineLength=35, maxLineGap=8)
    found = []
    if lines is not None:
        for item in lines[:, 0]:
            x0, y0, x1, y1 = map(float, item)
            if x1 < x0:
                x0, x1, y0, y1 = x1, x0, y1, y0
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            if abs(angle) <= 20:
                found.append({"line": [x0, y0, x1, y1], "angle_deg": angle, "length": math.hypot(x1 - x0, y1 - y0)})
    if not found:
        return {"valid": False, "reason": "no_visible_rod_line", "lines": []}
    center = float(np.median([line["angle_deg"] for line in found]))
    retained = [line for line in found if abs(line["angle_deg"] - center) <= config["rod_angle_agreement_deg"]]
    length = sum(line["length"] for line in retained)
    if length < config["minimum_rod_line_support_px"]:
        return {"valid": False, "reason": "insufficient_consistent_rod_edges", "lines": found}
    estimate = sum(line["angle_deg"] * line["length"] for line in retained) / length
    spread = max(abs(line["angle_deg"] - estimate) for line in retained)
    return {"valid": True, "angle_deg": estimate, "angle_spread_deg": spread,
            "summed_line_support_px": length, "lines": retained,
            "measurement_kind": "visible_rod_edge_angle_in_image_plane_not_3d_angle"}


def analyze_frames(frames, reference_index, terminal_indices, config):
    reference_frame = frames[reference_index]
    original = [body_candidates(reference_frame, side, config) for side in (0, 1)]
    if all("marker_center" in body for body in original):
        a, b = [body["marker_center"] for body in original]
        slope = (b[1] - a[1]) / max(1.0, b[0] - a[0])
    else:
        slope = None
    original_rod = rod_angle(reference_frame, original, config)
    records = []
    for index in terminal_indices:
        current = [body_candidates(frames[index], side, config) for side in (0, 1)]
        sides = [clearance(a, b, slope, config) if slope is not None else {"valid": False, "reason": "no_reference_floor_slope"}
                 for a, b in zip(original, current)]
        rod = rod_angle(frames[index], current, config)
        delta = abs(rod["angle_deg"] - original_rod["angle_deg"]) if rod["valid"] and original_rod["valid"] else None
        records.append({"frame": index, "bodies": current, "sides": sides, "rod": rod, "relative_rod_angle_deg": delta})
    variants = {}
    for threshold in config["small_tilt_sensitivity_deg"]:
        positive = bool(records) and all(
            all(side.get("valid") and side.get("full_geometry") and side["gap_lower_px"] >= config["minimum_clearance_px"] for side in row["sides"])
            and row["relative_rod_angle_deg"] is not None and row["relative_rod_angle_deg"] <= threshold
            for row in records
        )
        contact = bool(records) and all(
            any(side.get("valid") and side["gap_upper_px"] <= config["minimum_clearance_px"] for side in row["sides"])
            for row in records
        )
        variants[str(threshold)] = "bilateral_clearance_small_tilt_candidate" if positive else "persistent_contact_like_evidence" if contact else "unresolved"
    return {"reference_bodies": original, "reference_rod": original_rod, "floor_slope_proxy": slope,
            "terminal_frames": records, "diagnostic_variants": variants,
            "physical_success_certified": False, "fixed_camera_local_floor_assumption": True}
