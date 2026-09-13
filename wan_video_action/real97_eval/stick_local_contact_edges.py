"""Diagnostic local bottom-edge evidence, NOT an approved contact classifier."""

import cv2
import numpy as np


def edge_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.6)
    smooth = cv2.boxFilter(smooth, -1, (3, 3), normalize=True)
    indices = np.arange(len(smooth))
    return smooth[np.minimum(indices + 2, len(smooth) - 1)] - smooth[np.maximum(indices - 2, 0)]


def floor_tape_mask(frame, shape, slope, config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (92, 65, 20), (125, 255, 255))
    tall = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((17, 3), np.uint8))
    tall = cv2.dilate(tall, np.ones((5, 7), np.uint8))
    short = cv2.bitwise_and(blue, cv2.bitwise_not(tall))
    hull = np.asarray(shape["hull"], dtype=float).reshape(-1, 2)
    floor = np.max(hull[:, 1] - slope * hull[:, 0])
    yy, xx = np.indices(short.shape)
    short[np.abs(yy - slope * xx - floor) > config["floor_band_px"]] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(short)
    selected = np.zeros_like(short)
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if area >= 6 and width >= 5 and height <= 18 and width >= 0.65 * height:
            selected[labels == index] = 255
    return cv2.dilate(selected, np.ones((3, 3), np.uint8))


def find_edge(response, excluded, x, expected_y, config):
    height, width = response.shape
    ix = int(round(x))
    if not 2 <= ix < width - 2:
        return None
    radius = int(config["search_radius_px"])
    lo = max(3, int(np.floor(expected_y)) - radius)
    hi = min(height - 4, int(np.ceil(expected_y)) + radius)
    if lo > hi:
        return None
    yy = np.arange(lo, hi + 1)
    strength = response[yy, ix]
    valid = (strength >= config["minimum_edge_contrast"]) & (excluded[yy, ix] == 0)
    if not np.any(valid):
        return None
    score = strength - float(config["distance_penalty"]) * np.abs(yy - expected_y)
    score[~valid] = -np.inf
    best = int(np.argmax(score))
    y = int(yy[best])
    lower, center, upper = map(float, response[y-1:y+2, ix])
    denominator = lower - 2 * center + upper
    offset = 0.0 if denominator >= -1e-5 else float(np.clip(0.5 * (lower - upper) / denominator, -0.5, 0.5))
    local = response[max(0, y-4):min(height, y+5), ix]
    broadness = int(np.sum(local >= 0.8 * center))
    competitors = strength[np.abs(yy - y) >= 3]
    return dict(x=float(x), y=float(y + offset), contrast=center,
                prominence=center-float(np.max(competitors)) if len(competitors) else center,
                localization_radius_px=max(0.5, broadness / 2),
                displacement_from_pose_px=float(y + offset - expected_y))


def reference_feet(frame, shape, side, slope, config):
    if not shape.get("valid"):
        return []
    height, width = frame.shape[:2]
    hull = np.asarray(shape["hull"], dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 255)
    response = edge_image(frame)
    excluded = floor_tape_mask(frame, shape, slope, config)
    left, right = (3, width // 2 - 3) if side == 0 else (width // 2 + 3, width - 3)
    left = max(left, int(np.ceil(hull[:, 0].min())) + 2)
    right = min(right, int(np.floor(hull[:, 0].max())) - 2)
    points = []
    for x in range(left, right, int(config["footpoint_stride_px"])):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            point = find_edge(response, excluded, x, int(ys[-1]), config)
            if point is not None:
                points.append(point)
    return points


def measure(frame, shape, feet, pose, side, slope, config):
    if not pose.get("valid") or len(feet) < 3:
        return dict(valid=False, reason="missing_pose_or_reference_edge", points=[])
    matrix = np.asarray(pose["matrix"], dtype=float).reshape(2, 3)
    response = edge_image(frame)
    excluded = floor_tape_mask(frame, shape, slope, config)
    width = frame.shape[1]
    lo, hi = (3, width // 2 - 3) if side == 0 else (width // 2 + 3, width - 3)
    points = []
    for foot in feet:
        current = matrix[:, :2] @ [foot["x"], foot["y"]] + matrix[:, 2]
        if not lo < current[0] < hi:
            continue
        detected = find_edge(response, excluded, current[0], current[1], config)
        if detected is None:
            continue
        gap = foot["y"] - detected["y"] + slope * (detected["x"] - foot["x"])
        points.append(dict(reference=foot, current=detected, gap_px=float(gap),
                           pose_xy=current.tolist(),
                           localization_sum_px=foot["localization_radius_px"] + detected["localization_radius_px"]))
    triples = []
    for start in range(len(points)-2):
        triple = points[start:start+3]
        xs = [p["reference"]["x"] for p in triple]
        if max(np.diff(xs)) <= 2 * int(config["footpoint_stride_px"]):
            triples.append((float(np.median([p["gap_px"] for p in triple])), start))
    if not triples:
        return dict(valid=False, reason="insufficient_adjacent_bottom_edges", points=points)
    gap, start = min(triples)
    selected = points[start:start+3]
    return dict(valid=True, gap_px=gap,
                median_gap_px=float(np.median([p["gap_px"] for p in points])),
                minimum_group_reference_x=[p["reference"]["x"] for p in selected],
                localization_radius_px=float(np.median([p["localization_sum_px"] for p in selected])),
                points=points, visible_footpoint_count=len(points),
                reference_cropped=bool(shape.get("actual_image_crop") or shape.get("reference_roi_truncated")),
                measurement_kind="visible_bottom_edge_vertical_gap_diagnostic_not_3d_contact")
