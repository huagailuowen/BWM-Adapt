"""Independent diagnostic combining sequential and anchored body tracking."""
import math

import cv2
import numpy as np

from .stick_common_reference_rigid_review import (
    actual_crop, tracked_angle, transform, unwrap_angle,
)
from .stick_shape_flow_review import background_colors, export_shape, segment


class SequentialBody:
    """Keep reference correspondences while following every intervening frame."""

    def __init__(self, body, gray):
        self.body = body
        self.gray = gray
        self.reason = None
        self.matrix = None
        self.residual = None
        self.steps = 0
        if body.flow is None or body.flow.points is None:
            self.points = None
            self.reason = "missing_reference"
            return
        self.points = np.asarray(body.flow.points, np.float32).reshape(-1, 1, 2).copy()
        self.anchors = self.points.copy()

    def update(self, gray):
        if self.reason:
            return
        self.steps += 1
        if len(self.points) < 6:
            self.reason = "insufficient_surviving_features"
            return
        options = dict(winSize=(25, 25), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
        target, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        if target is None:
            self.reason = "forward_flow_missing"
            return
        backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, target, None, **options)
        if backward is None:
            self.reason = "backward_flow_missing"
            return
        xy = target.reshape(-1, 2)
        fb = np.linalg.norm(backward.reshape(-1, 2) - self.points.reshape(-1, 2), axis=1)
        good = (ok.ravel() > 0) & (back_ok.ravel() > 0) & np.isfinite(xy).all(axis=1) & (fb <= 1.5)
        good &= (xy[:, 1] > 1) & (xy[:, 1] < 479)
        if self.body.side_index == 0:
            good &= (xy[:, 0] > 1) & (xy[:, 0] < 319)
        else:
            good &= (xy[:, 0] > 321) & (xy[:, 0] < 639)
        source, target = self.anchors[good], target[good]
        if len(source) < 6:
            self.reason = "insufficient_surviving_features"
            return
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5,
            maxIters=2000, confidence=.99,
        )
        if matrix is None or inliers is None or inliers.sum() < 6:
            self.reason = "global_rigid_fit_failed"
            return
        ii = inliers.ravel() > 0
        source, target = source[ii], target[ii]
        span = np.ptp(source.reshape(-1, 2), axis=0)
        if span[0] < 4 or span[1] < 6:
            self.reason = "insufficient_feature_spatial_coverage"
            return
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        rotation = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        if not .75 <= scale <= 1.3 or rotation > 35:
            self.reason = "invalid_rigid_motion"
            return
        residual = np.linalg.norm(transform(source, matrix) - target.reshape(-1, 2), axis=1)
        self.residual = float(np.quantile(residual, .9)) + float(np.quantile(fb[good][ii], .9))
        self.matrix = matrix
        self.anchors, self.points, self.gray = source, target, gray

    def report(self):
        if self.reason or self.matrix is None:
            return {"valid": False, "reason": self.reason or "no_steps", "steps": self.steps}
        return {"valid": True, "matrix": self.matrix.tolist(), "fit_error_px": self.residual,
                "features": len(self.points), "steps": self.steps}


def appearance_at_pose(frame, original, matrix, side, config):
    height, width = frame.shape[:2]
    half = (0, width // 2) if side == 0 else (width // 2, width)
    warped = cv2.warpAffine(original["mask"], matrix, (width, height), flags=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (92, 65, 20), (125, 255, 255))
    seeds = cv2.bitwise_and(cv2.erode(warped, np.ones((7, 7), np.uint8)), blue)
    if np.count_nonzero(seeds) < 10:
        return {"valid": False, "reason": "insufficient_body_seeds"}
    appearance = {"valid": False, "reason": "empty_search_region"}
    for halo in config["shape_halos_px"]:
        allowed = cv2.dilate(warped, np.ones((halo * 2 + 1,) * 2, np.uint8))
        allowed[:, :half[0]] = 0
        allowed[:, half[1]:] = 0
        yy, xx = np.where(allowed > 0)
        if len(xx) < 60:
            continue
        ring = config["background_ring_px"]
        roi = (max(half[0], int(xx.min()) - ring), max(140, int(yy.min()) - ring),
               min(half[1], int(xx.max()) + ring + 1), min(height, int(yy.max()) + ring + 1))
        forbidden = cv2.bitwise_or(cv2.bitwise_not(allowed), background_colors(frame))
        appearance = segment(frame, roi, seeds, warped, forbidden, config)
        if not appearance.get("valid"):
            continue
        border = cv2.subtract(allowed, cv2.erode(allowed, np.ones((3, 3), np.uint8)))
        hits = int(np.count_nonzero(cv2.bitwise_and(border, appearance["mask"])))
        appearance.update(search_halo_px=halo, search_boundary_pixels=hits,
                          search_truncated=hits > config["maximum_halo_boundary_pixels"],
                          actual_image_crop=actual_crop(appearance, side, width, height),
                          area_ratio=float(np.count_nonzero(appearance["mask"])) / max(1, np.count_nonzero(warped)))
        if not appearance["search_truncated"]:
            break
    return export_shape(appearance)


def polygon_iou(a, b):
    masks = []
    for polygon in (a, b):
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(mask, np.asarray(polygon, np.int32).reshape(-1, 2), 1)
        masks.append(mask.astype(bool))
    return float(np.count_nonzero(masks[0] & masks[1])) / max(1, np.count_nonzero(masks[0] | masks[1]))


def clearance(original, direct, sequential, appearance, slope, config):
    if not original.get("valid") or slope is None:
        return {"valid": False, "reason": "reference_unavailable"}
    ref_hull = np.asarray(original["hull"]).reshape(-1, 2)
    ref_edge = float(np.max(ref_hull[:, 1] - slope * ref_hull[:, 0]))
    measurements = []
    full_visible = not original.get("actual_image_crop", True) and not original.get("reference_roi_truncated", True)
    for name, pose in (("direct", direct), ("sequential", sequential)):
        if not pose.get("valid"):
            continue
        hull = transform(ref_hull, np.asarray(pose["matrix"]))
        gap = ref_edge - float(np.max(hull[:, 1] - slope * hull[:, 0]))
        error = config["clearance_error_floor_px"] + pose["fit_error_px"]
        cropped = bool((hull[:, 0] <= 1).any() or (hull[:, 0] >= 639).any()
                       or (hull[:, 1] <= 1).any() or (hull[:, 1] >= 479).any())
        full_visible &= not cropped
        measurements.append({"source": name, "gap_px": gap, "lower_px": gap - error,
                             "upper_px": gap + error, "hull": hull.tolist()})
    if not measurements:
        return {"valid": False, "reason": "both_tracking_methods_failed"}
    appearance_ok = (appearance.get("valid") and not appearance.get("search_truncated")
                     and not appearance.get("actual_image_crop")
                     and config["minimum_shape_area_ratio"] <= appearance.get("area_ratio", 0) <= config["maximum_shape_area_ratio"])
    ious = []
    if appearance_ok:
        ious = [polygon_iou(m["hull"], appearance["hull"]) for m in measurements]
        appearance_ok = min(ious) >= config["minimum_appearance_iou"]
    if appearance_ok:
        hull = np.asarray(appearance["hull"]).reshape(-1, 2)
        gap = ref_edge - float(np.max(hull[:, 1] - slope * hull[:, 0]))
        error = max(m["upper_px"] - m["gap_px"] for m in measurements)
        measurements.append({"source": "appearance", "gap_px": gap,
                             "lower_px": gap - error, "upper_px": gap + error})
    # Union bounds retain disagreement without a blind fixed-difference veto.
    # A single valid pose requires an independent visible-body check.
    crosschecked = bool(appearance_ok or len(measurements) >= 2)
    return {"valid": crosschecked, "reason": None if crosschecked else "missing_independent_geometry_check",
            "gap_lower_px": min(m["lower_px"] for m in measurements),
            "gap_upper_px": max(m["upper_px"] for m in measurements),
            "full_geometry_visible": bool(full_visible), "appearance_check_available": bool(appearance_ok),
            "appearance_ious": ious, "measurements": measurements}


def fuse_angles(originals, direct, sequential, reference_rod, current_rod, config):
    estimates = []
    mixed = [d if d.get("valid") else s for d, s in zip(direct, sequential)]
    for name, poses in (("direct_geometry", direct), ("sequential_geometry", sequential), ("mixed_geometry", mixed)):
        if not all(p.get("valid") for p in poses):
            continue
        adapted = [dict(p, matrix=np.asarray(p["matrix"])) for p in poses]
        result = tracked_angle(originals, adapted, config)
        if result.get("valid"):
            estimates.append({"source": name, "signed_lower_deg": result["signed_lower_deg"],
                              "signed_upper_deg": result["signed_upper_deg"],
                              "signed_change_deg": result["signed_change_deg"]})
    if reference_rod.get("valid") and current_rod.get("valid"):
        change = unwrap_angle(current_rod["angle_deg"] - reference_rod["angle_deg"])
        error = max(config["minimum_independent_angle_error_deg"],
                    reference_rod.get("angle_spread_deg", 1.0) + current_rod.get("angle_spread_deg", 1.0))
        estimates.append({"source": "independent_visible_rod", "signed_lower_deg": change - error,
                          "signed_upper_deg": change + error, "signed_change_deg": change})
    if not estimates:
        return {"valid": False, "reason": "all_angle_methods_unavailable"}
    lower = min(e["signed_lower_deg"] for e in estimates)
    upper = max(e["signed_upper_deg"] for e in estimates)
    return {"valid": True, "signed_lower_deg": lower, "signed_upper_deg": upper,
            "absolute_lower_deg": 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper)),
            "absolute_upper_deg": max(abs(lower), abs(upper)), "estimates": estimates}


def decide(rows, config):
    variants = {}
    for threshold in config["small_tilt_sensitivity_deg"]:
        states = []
        witnesses = []
        for row in rows:
            sides, angle = row["sides"], row["fused_angle"]
            positive = (all(s.get("valid") and s.get("full_geometry_visible")
                            and s["gap_lower_px"] >= config["minimum_clearance_px"] for s in sides)
                        and angle.get("valid") and angle["absolute_upper_deg"] <= threshold)
            contact = [i for i, s in enumerate(sides) if s.get("valid") and s.get("full_geometry_visible")
                       and s["gap_upper_px"] <= config["minimum_clearance_px"]]
            tilted = angle.get("valid") and angle["absolute_lower_deg"] > threshold + config["tilt_failure_margin_deg"]
            state = "positive" if positive else "negative" if contact or tilted else "unresolved"
            states.append(state)
            if state == "negative":
                witnesses.append({"frame": row["frame"], "source_frame": row["source_frame"],
                                  "contact_sides": contact, "excessive_tilt": bool(tilted)})
        all_positive = bool(states) and all(s == "positive" for s in states)
        any_negative = any(s == "negative" for s in states)
        all_negative = bool(states) and all(s == "negative" for s in states)
        status = ("balanced_lift_candidate" if all_positive else "no_balanced_lift_candidate" if any_negative else "unresolved")
        old_rule = ("balanced_lift_candidate" if all_positive else "no_balanced_lift_candidate" if all_negative else "unresolved")
        variants[str(int(threshold))] = {"status": status, "frame_states": states,
                                       "failure_witnesses": witnesses, "all_frames_negative_rule_status": old_rule}
    return variants
