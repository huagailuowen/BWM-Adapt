"""Reference-shape/flow constrained terminal review, without a rise-ratio gate."""

import math

import cv2
import numpy as np

from .stick_marker_flow import MarkerReference, locate_markers
from .stick_terminal_silhouette import rod_angle


def shape_from_mask(mask, roi, partial=False):
    yy, xx = np.nonzero(mask)
    if len(xx) < 60:
        return {"valid": False, "reason": "foreground_too_small"}
    hull = cv2.convexHull(np.column_stack((xx, yy)).astype(np.float32))[:, 0]
    x, y, w, h = cv2.boundingRect(hull)
    x0, y0, x1, y1 = roi
    clipped = bool(partial or x <= x0 + 1 or x + w >= x1 - 1 or y <= y0 + 1 or y + h >= y1 - 1)
    return {"valid": True, "mask": mask, "hull": hull, "bbox": [x, y, w, h], "clipped": clipped,
            "roi": list(roi), "foreground_pixels": len(xx)}


def segment(frame, roi, seeds, probable, definite_background, config):
    x0, y0, x1, y1 = roi
    crop = frame[y0:y1, x0:x1]
    labels = np.full(crop.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    labels[probable[y0:y1, x0:x1] > 0] = cv2.GC_PR_FGD
    labels[definite_background[y0:y1, x0:x1] > 0] = cv2.GC_BGD
    labels[seeds[y0:y1, x0:x1] > 0] = cv2.GC_FGD
    labels[:1] = cv2.GC_BGD
    labels[-1:] = cv2.GC_BGD
    labels[:, :1] = cv2.GC_BGD
    labels[:, -1:] = cv2.GC_BGD
    if np.count_nonzero(labels == cv2.GC_FGD) < 10 or np.count_nonzero(labels == cv2.GC_BGD) < 10:
        return {"valid": False, "reason": "insufficient_grabcut_seeds"}
    try:
        cv2.grabCut(crop, labels, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    config["grabcut_iterations"], cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        return {"valid": False, "reason": "grabcut_failed", "detail": str(error)[:160]}
    foreground = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, components, _, _ = cv2.connectedComponentsWithStats(foreground)
    local_seeds = seeds[y0:y1, x0:x1] > 0
    scores = [(int(np.count_nonzero((components == index) & local_seeds)), index) for index in range(1, count)]
    if not scores or max(scores)[0] < 10:
        return {"valid": False, "reason": "seeded_body_component_missing"}
    _, chosen = max(scores)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = (components == chosen).astype(np.uint8) * 255
    return shape_from_mask(mask, roi)


def background_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (15, 65, 25), (90, 255, 255))


class ReferenceBody:
    def __init__(self, frame, side, config):
        self.side = side
        self.config = config
        self.reference_frame = frame
        self.reference = {"valid": False, "reason": "reference_marker_missing"}
        self.flow = None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        marker = locate_markers(hsv, side)
        if marker is None:
            return
        height, width = frame.shape[:2]
        yy, xx = np.nonzero(marker["mask"])
        half_lo, half_hi = (0, width // 2) if side == 0 else (width // 2, width)
        roi = [max(half_lo, int(xx.min()) - 22), max(140, int(yy.min()) - 12),
               min(half_hi, int(xx.max()) + 23), min(height, int(yy.max()) + 15)]
        seeds = marker["mask"].copy()
        seeds[np.indices(seeds.shape)[0] > np.quantile(yy, 0.82)] = 0
        probable = np.zeros((height, width), np.uint8)
        probable[max(140, int(yy.min()) - 5):min(height, int(np.quantile(yy, 0.97)) + 5),
                 max(half_lo, int(xx.min()) - 12):min(half_hi, int(xx.max()) + 13)] = 255
        self.reference = segment(frame, roi, seeds, probable, background_colors(frame), config)
        if not self.reference["valid"]:
            return
        self.reference["clipped"] |= bool(xx.min() <= half_lo + 1 or xx.max() >= half_hi - 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.flow = MarkerReference(gray, hsv, side, 0.0,
                                    {"contact_lever_arm_px": 20.0, "flow_error_floor_px": 0.5})
        feature_mask = cv2.bitwise_and(self.reference["mask"], cv2.dilate(marker["mask"], np.ones((5, 5), np.uint8)))
        self.flow.points = cv2.goodFeaturesToTrack(gray, maxCorners=90, qualityLevel=0.005,
                                                  minDistance=3, blockSize=3, mask=feature_mask)

    def update(self, frame):
        if self.flow is None or not self.reference["valid"]:
            return {"valid": False, "reason": self.reference["reason"]}
        row = self.flow.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        if not row.get("flow_valid"):
            return {"valid": False, "reason": row.get("reason", "flow_failed")}
        source = np.asarray(row["reference_marker_sites"], dtype=np.float64)
        target = np.asarray(row["lower_marker_sites"], dtype=np.float64)
        u, v = source[1] - source[0], target[1] - target[0]
        denominator = float(u @ u)
        if denominator < 9:
            return {"valid": False, "reason": "degenerate_reference_sites"}
        a, b = float(u @ v / denominator), float((u[0] * v[1] - u[1] * v[0]) / denominator)
        rotation_scale = np.asarray([[a, -b], [b, a]])
        translation = target[0] - rotation_scale @ source[0]
        matrix = np.column_stack((rotation_scale, translation))
        height, width = frame.shape[:2]
        projected_hull = self.reference["hull"] @ rotation_scale.T + translation
        half_lo, half_hi = (0, width // 2) if self.side == 0 else (width // 2, width)
        if not np.isfinite(projected_hull).all() or (self.side == 0 and np.any(projected_hull[:, 0] >= half_hi)) or (self.side == 1 and np.any(projected_hull[:, 0] < half_lo)):
            return {"valid": False, "reason": "projected_body_crossed_image_half"}
        warped = cv2.warpAffine(self.reference["mask"], matrix, (width, height), flags=cv2.INTER_NEAREST)
        radius = self.config["shape_halo_px"]
        allowed = cv2.dilate(warped, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        allowed[:, :half_lo] = 0
        allowed[:, half_hi:] = 0
        yy, xx = np.nonzero(allowed)
        if len(xx) < 60:
            return {"valid": False, "reason": "body_outside_view"}
        pad = self.config["background_ring_px"]
        roi = [max(half_lo, int(xx.min()) - pad), max(140, int(yy.min()) - pad),
               min(half_hi, int(xx.max()) + pad + 1), min(height, int(yy.max()) + pad + 1)]
        interior = cv2.erode(warped, np.ones((7, 7), np.uint8))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (92, 65, 20), (125, 255, 255))
        seeds = cv2.bitwise_and(interior, blue)
        if np.count_nonzero(seeds) < 10:
            return {"valid": False, "reason": "no_current_material_seed_inside_reference_shape"}
        forbidden = cv2.bitwise_or(cv2.bitwise_not(allowed), background_colors(frame))
        shape = segment(frame, roi, seeds, warped, forbidden, self.config)
        if not shape["valid"]:
            return shape
        shape["clipped"] |= self.reference["clipped"] or bool(np.any(projected_hull[:, 0] < half_lo + 1) or np.any(projected_hull[:, 0] >= half_hi - 1))
        allowed_border = cv2.subtract(allowed, cv2.erode(allowed, np.ones((3, 3), np.uint8)))
        boundary_pixels = int(np.count_nonzero((shape["mask"] > 0) & (allowed_border > 0)))
        shape["shape_halo_boundary_pixels"] = boundary_pixels
        shape["clipped"] |= boundary_pixels > self.config["maximum_halo_boundary_pixels"]
        shape["fit_error_px"] = float(row["fit_error_p90_px"] + row["fb_error_p90_px"])
        shape["box_rotation_deg"] = row["box_rotation_deg"]
        shape["flow_inliers"] = row["feature_inliers"]
        return shape


def rod_body(shape):
    return {"valid": shape["valid"], "candidates": [{"threshold": 85, "bbox": shape["bbox"], "hull": shape["hull"].tolist(),
                                                    "clipped": shape["clipped"]}] if shape["valid"] else []}


def export_shape(shape):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in shape.items() if key != "mask"}


def evaluate(reference_frame, frames, config):
    bodies = [ReferenceBody(reference_frame, side, config) for side in (0, 1)]
    originals = [body.reference for body in bodies]
    reference_rod = rod_angle(reference_frame, [rod_body(shape) for shape in originals], config)
    slope = None
    if all(shape["valid"] for shape in originals):
        centers = [np.mean(shape["hull"], axis=0) for shape in originals]
        slope = float((centers[1][1] - centers[0][1]) / max(1.0, centers[1][0] - centers[0][0]))
    rows = []
    for index, frame in frames:
        current = [body.update(frame) for body in bodies]
        angle = rod_angle(frame, [rod_body(shape) for shape in current], config)
        sides = []
        for original, shape in zip(originals, current):
            if not original["valid"] or not shape["valid"] or slope is None:
                sides.append({"valid": False, "reason": shape.get("reason", "reference_unavailable")})
                continue
            ref = original["hull"]
            now = shape["hull"]
            gap = float(np.max(ref[:, 1] - slope * ref[:, 0]) - np.max(now[:, 1] - slope * now[:, 0]))
            error = config["clearance_error_floor_px"] + shape["fit_error_px"]
            sides.append({"valid": True, "gap_px": gap, "gap_lower_px": gap - error,
                          "gap_upper_px": gap + error, "uncertainty_px": error,
                          "full_geometry": not original["clipped"] and not shape["clipped"]})
        delta = abs(angle["angle_deg"] - reference_rod["angle_deg"]) if angle["valid"] and reference_rod["valid"] else None
        rows.append({"frame": index, "bodies": [export_shape(shape) for shape in current], "sides": sides,
                     "rod": angle, "relative_rod_angle_deg": delta})
    variants = {}
    for threshold in config["small_tilt_sensitivity_deg"]:
        success = bool(rows) and all(all(side.get("valid") and side["full_geometry"] and side["gap_lower_px"] >= config["minimum_clearance_px"] for side in row["sides"])
                                    and row["relative_rod_angle_deg"] is not None and row["relative_rod_angle_deg"] <= threshold for row in rows)
        contact = bool(rows) and all(any(side.get("valid") and side["gap_upper_px"] <= config["minimum_clearance_px"] for side in row["sides"]) for row in rows)
        tilted = bool(rows) and all(row["relative_rod_angle_deg"] is not None and row["relative_rod_angle_deg"] > threshold + config["tilt_failure_margin_deg"] for row in rows)
        variants[str(threshold)] = "balanced_lift_candidate" if success else "no_balanced_lift_candidate" if contact or tilted else "unresolved"
    return {"reference_bodies": [export_shape(shape) for shape in originals], "reference_rod": reference_rod,
            "floor_slope_proxy": slope, "terminal_frames": rows, "diagnostic_variants": variants,
            "rise_ratio_required": False, "physical_label_certified": False}
