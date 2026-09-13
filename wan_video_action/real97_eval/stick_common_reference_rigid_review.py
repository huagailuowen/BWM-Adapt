"""Opt-in diagnostic: shared GT reference, rigid contact geometry and angle bounds."""
import itertools
import math

import cv2
import numpy as np

from .stick_shape_flow_review import (
    ReferenceBody, background_colors, export_shape, rod_body, segment,
    shape_from_mask,
)
from .stick_terminal_silhouette import rod_angle


def actual_crop(shape, side, width, height):
    if not shape.get("valid"):
        return True
    x, y, w, h = shape["bbox"]
    return bool(x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1)


def affine_from_flow(row):
    source = np.asarray(row["reference_marker_sites"], np.float64)
    target = np.asarray(row["lower_marker_sites"], np.float64)
    u, v = source[1] - source[0], target[1] - target[0]
    denom = float(u @ u)
    if denom < 9:
        return None
    a = float(u @ v) / denom
    b = float(u[0] * v[1] - u[1] * v[0]) / denom
    linear = np.array([[a, -b], [b, a]])
    return np.column_stack((linear, target[0] - linear @ source[0]))


def transform(points, matrix):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def unwrap_angle(angle):
    return (angle + 90.0) % 180.0 - 90.0


def vector_angle(left, right):
    delta = np.asarray(right) - np.asarray(left)
    return math.degrees(math.atan2(float(delta[1]), float(delta[0])))


class RigidBody(ReferenceBody):
    def __init__(self, frame, side, config):
        super().__init__(frame, side, config)
        self.side_index = side
        self.settings = config
        self.original = self.reference
        if self.original.get("valid"):
            # Keep search-ROI truncation separate from actual image clipping.
            self.original["actual_image_crop"] = actual_crop(
                self.original, side, frame.shape[1], frame.shape[0]
            )
            self.original["reference_roi_truncated"] = bool(
                self.original.get("clipped", False)
                and not self.original["actual_image_crop"]
            )

    def track(self, frame):
        if not self.original.get("valid") or self.flow is None:
            return {"valid": False, "reason": "reference_unavailable"}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        flow = self.flow.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), hsv)
        if not flow.get("flow_valid"):
            return {"valid": False, "reason": flow.get("reason", "flow_unavailable")}
        matrix = affine_from_flow(flow)
        if matrix is None:
            return {"valid": False, "reason": "degenerate_reference_sites"}
        height, width = frame.shape[:2]
        half = (0, width // 2) if self.side_index == 0 else (width // 2, width)
        hull = transform(self.original["hull"], matrix)
        if not np.isfinite(hull).all():
            return {"valid": False, "reason": "nonfinite_rigid_geometry"}
        # A tracked left object may never be replaced by the right object.
        if hull[:, 0].mean() < half[0] or hull[:, 0].mean() >= half[1]:
            return {"valid": False, "reason": "wrong_image_half"}
        warped = cv2.warpAffine(
            self.original["mask"], matrix, (width, height), flags=cv2.INTER_NEAREST
        )
        original_crop = self.original["actual_image_crop"]
        cropped = bool(original_crop or (hull[:, 0] <= 1).any()
                       or (hull[:, 0] >= width - 1).any()
                       or (hull[:, 1] <= 1).any()
                       or (hull[:, 1] >= height - 1).any())
        rigid = shape_from_mask(warped, (0, 0, width, height), partial=cropped)
        if not rigid.get("valid"):
            return {"valid": False, "reason": "empty_projected_body"}
        rigid["hull"] = hull
        blue = cv2.inRange(hsv, (92, 65, 20), (125, 255, 255))
        interior = cv2.erode(warped, np.ones((7, 7), np.uint8))
        seeds = cv2.bitwise_and(interior, blue)
        appearance = {"valid": False, "reason": "insufficient_body_seeds"}
        if np.count_nonzero(seeds) >= 10:
            for halo in self.settings["shape_halos_px"]:
                allowed = cv2.dilate(warped, np.ones((halo * 2 + 1,) * 2, np.uint8))
                allowed[:, :half[0]] = 0
                allowed[:, half[1]:] = 0
                yy, xx = np.where(allowed > 0)
                if len(xx) < 60:
                    continue
                ring = self.settings["background_ring_px"]
                roi = (max(half[0], int(xx.min()) - ring), max(140, int(yy.min()) - ring),
                       min(half[1], int(xx.max()) + ring + 1), min(height, int(yy.max()) + ring + 1))
                forbidden = cv2.bitwise_or(cv2.bitwise_not(allowed), background_colors(frame))
                appearance = segment(frame, roi, seeds, warped, forbidden, self.settings)
                if not appearance.get("valid"):
                    continue
                edge = cv2.subtract(allowed, cv2.erode(allowed, np.ones((3, 3), np.uint8)))
                hits = int(np.count_nonzero(cv2.bitwise_and(edge, appearance["mask"])))
                appearance["search_halo_px"] = halo
                appearance["search_boundary_pixels"] = hits
                appearance["search_truncated"] = hits > self.settings["maximum_halo_boundary_pixels"]
                appearance["actual_image_crop"] = actual_crop(appearance, self.side_index, width, height)
                appearance["clipped"] = appearance["actual_image_crop"]
                appearance["area_ratio"] = float(np.count_nonzero(appearance["mask"])) / max(1, np.count_nonzero(warped))
                if not appearance["search_truncated"]:
                    break
        error = float(flow.get("fit_error_p90_px", 0)) + float(flow.get("fb_error_p90_px", 0))
        return {"valid": True, "matrix": matrix, "rigid": rigid, "appearance": appearance,
                "fit_error_px": error, "flow_inliers": flow.get("feature_inliers"),
                "box_rotation_deg": flow.get("box_rotation_deg"), "scale": flow.get("scale"),
                "actual_image_crop": cropped,
                "reference_roi_truncated": self.original["reference_roi_truncated"]}


def attachment_points(shape):
    x, y, w, h = shape["bbox"]
    # Sweep plausible attachment locations; their identity is held fixed over time.
    return np.array([(x + w * fx, y + h * fy)
                     for fx, fy in itertools.product((0.3, 0.5, 0.7), (0.0, 0.12, 0.24))])


def tracked_angle(originals, tracks, config):
    if not all(t.get("valid") for t in tracks):
        return {"valid": False, "reason": "body_tracking_unavailable"}
    points = [attachment_points(s) for s in originals]
    targets = [transform(p, t["matrix"]) for p, t in zip(points, tracks)]
    changes = []
    for il, ir in itertools.product(range(len(points[0])), range(len(points[1]))):
        before = vector_angle(points[0][il], points[1][ir])
        after = vector_angle(targets[0][il], targets[1][ir])
        changes.append(unwrap_angle(after - before))
    signed = float(np.median(changes))
    span = float(np.linalg.norm(targets[1].mean(axis=0) - targets[0].mean(axis=0)))
    error = math.degrees(math.atan2(sum(t["fit_error_px"] + 1 for t in tracks), max(1, span)))
    lower, upper = min(changes) - error, max(changes) + error
    return {"valid": True, "signed_change_deg": signed, "relative_angle_deg": abs(signed),
            "signed_lower_deg": lower, "signed_upper_deg": upper,
            "absolute_lower_deg": 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper)),
            "absolute_upper_deg": max(abs(lower), abs(upper)),
            "point_and_fit_uncertainty_deg": max(signed - lower, upper - signed),
            "kind": "fixed_attachment_geometry_sweep_with_optical_flow"}


def body_gap(original, track, slope, config):
    if not track.get("valid") or slope is None:
        return {"valid": False, "reason": track.get("reason", "floor_reference_unavailable")}
    def lower_edge(points):
        p = np.asarray(points).reshape(-1, 2)
        return float(np.max(p[:, 1] - slope * p[:, 0]))
    reference_edge = lower_edge(original["hull"])
    rigid_gap = reference_edge - lower_edge(track["rigid"]["hull"])
    shape = track["appearance"]
    shape_gap = reference_edge - lower_edge(shape["hull"]) if shape.get("valid") else None
    reliable_shape = bool(shape.get("valid") and not shape.get("search_truncated")
                          and not shape.get("actual_image_crop")
                          and config["minimum_shape_area_ratio"] <= shape.get("area_ratio", 0) <= config["maximum_shape_area_ratio"])
    disagreement = abs(shape_gap - rigid_gap) if reliable_shape else None
    conflict = bool(disagreement is not None and disagreement > config["maximum_shape_gap_disagreement_px"])
    samples = [rigid_gap] + ([shape_gap] if reliable_shape else [])
    error = config["clearance_error_floor_px"] + track["fit_error_px"]
    visible = not track["actual_image_crop"] and not track["reference_roi_truncated"]
    return {"valid": not conflict, "reason": "appearance_rigid_disagreement" if conflict else None,
            "rigid_gap_px": rigid_gap, "appearance_gap_px": shape_gap,
            "appearance_check_available": reliable_shape, "gap_disagreement_px": disagreement,
            "gap_lower_px": min(samples) - error, "gap_upper_px": max(samples) + error,
            "full_geometry_visible": bool(visible), "fit_error_px": track["fit_error_px"],
            "actual_image_crop": track["actual_image_crop"],
            "reference_roi_truncated": track["reference_roi_truncated"]}


def evaluate(reference_frame, frames, config):
    bodies = [RigidBody(reference_frame, side, config) for side in (0, 1)]
    originals = [b.original for b in bodies]
    slope = None
    if all(o.get("valid") for o in originals):
        boxes = [o["bbox"] for o in originals]
        centers = [(x + w / 2, y + h) for x, y, w, h in boxes]
        slope = (centers[1][1] - centers[0][1]) / max(1, centers[1][0] - centers[0][0])
    reference_rod = rod_angle(reference_frame, [rod_body(o) for o in originals], config)
    rows = []
    for index, frame in frames:
        tracks = [b.track(frame) for b in bodies]
        gaps = [body_gap(o, t, slope, config) for o, t in zip(originals, tracks)]
        angle = tracked_angle(originals, tracks, config)
        shapes = [t.get("rigid", {"valid": False}) for t in tracks]
        direct = rod_angle(frame, [rod_body(s) for s in shapes], config)
        if angle.get("valid") and direct.get("valid") and reference_rod.get("valid"):
            change = unwrap_angle(direct["angle_deg"] - reference_rod["angle_deg"])
            angle["direct_signed_change_deg"] = change
            angle["direct_disagreement_deg"] = abs(unwrap_angle(change - angle["signed_change_deg"]))
            if angle["direct_disagreement_deg"] > config["maximum_direct_angle_disagreement_deg"]:
                angle["valid"] = False
                angle["reason"] = "direct_rod_body_geometry_disagreement"
        serialized_tracks = []
        for t in tracks:
            if not t.get("valid"):
                serialized_tracks.append(t)
                continue
            serialized_tracks.append({**{k: v for k, v in t.items() if k not in ("matrix", "rigid", "appearance")},
                                      "matrix": t["matrix"].tolist(), "rigid": export_shape(t["rigid"]),
                                      "appearance": export_shape(t["appearance"])})
        rows.append({"frame": index, "sides": gaps, "tracked_angle": angle,
                     "direct_rod": direct, "bodies": serialized_tracks})
    variants = {}
    for threshold in config["small_tilt_sensitivity_deg"]:
        states = []
        for row in rows:
            gaps, angle = row["sides"], row["tracked_angle"]
            positive = (all(g.get("valid") and g.get("full_geometry_visible")
                            and g["gap_lower_px"] >= config["minimum_clearance_px"] for g in gaps)
                        and angle.get("valid") and angle["absolute_upper_deg"] <= threshold)
            contact = any(g.get("valid") and g.get("full_geometry_visible")
                          and g["gap_upper_px"] <= config["minimum_clearance_px"] for g in gaps)
            tilted = angle.get("valid") and angle["absolute_lower_deg"] > threshold + config["tilt_failure_margin_deg"]
            states.append("positive" if positive else "negative" if contact or tilted else "unresolved")
        status = ("balanced_lift_candidate" if states and all(s == "positive" for s in states)
                  else "no_balanced_lift_candidate" if states and all(s == "negative" for s in states)
                  else "unresolved")
        variants[str(int(threshold))] = {"status": status, "frame_states": states}
    return {"reference_bodies": [export_shape(o) for o in originals], "reference_rod": reference_rod,
            "floor_slope_proxy": slope, "terminal_frames": rows, "diagnostic_variants": variants,
            "rise_ratio_required": False, "physical_label_certified": False}
