"""Independent endpoint evidence plus subpixel rigid-foot tracking.

The outputs are measurement proposals with heuristic uncertainty margins,
not calibrated probabilities or proof of invisible three-dimensional contact.
"""

import cv2
import numpy as np

from .stick_lift_contact import StickLiftContact, endpoint_bodies, longest_run


class FootReference:
    def __init__(self, image, body, table_slope, config):
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.body = body
        self.slope = float(table_slope)
        self.config = config
        hull = np.asarray(body["hull"], np.int32)
        lower = np.column_stack([body["lower_x"], body["lower_y"]]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(lower, cv2.DIST_HUBER, 0, .01, .01).reshape(-1)
        slope = float(vy / max(float(vx), 1e-6))
        residual = lower[:, 1] - (slope * (lower[:, 0] - x0) + y0)
        inliers = lower[np.abs(residual - np.median(residual)) <= 1.5]
        if len(inliers) < 6:
            inliers = lower
        edge_x = np.asarray([inliers[:, 0].min(), inliers[:, 0].max()])
        edge_y = slope * (edge_x - x0) + y0
        self.corners = np.column_stack([edge_x, edge_y]).astype(np.float32)
        self.initial_bias = None
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = ((hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 125)
                & (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 30)).astype(np.uint8)
        near_tape = cv2.dilate(blue, np.ones((5, 5), np.uint8))
        mask[near_tape == 0] = 0
        self.features = cv2.goodFeaturesToTrack(
            self.gray, mask=mask, maxCorners=70, qualityLevel=.01,
            minDistance=3, blockSize=5,
        )
        green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
                 & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 30))
        self.visible = []
        for x, y in self.corners:
            xi, yi = int(round(float(x))), int(round(float(y)))
            patch = green[max(0, yi - 5):min(480, yi + 6), max(0, xi - 5):min(640, xi + 6)]
            self.visible.append(bool(2 < x < 637 and 2 < y < 477 and patch.size and patch.mean() < .15))
        if body["clipped"]:
            self.visible[0 if hull[:, 0].min() <= 1 else 1] = False

    def measure(self, gray):
        if self.features is None or len(self.features) < 4:
            return {"flow_valid": False, "reason": "too_few_box_texture_features"}
        params = dict(winSize=(25, 25), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
        current, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.features, None, **params)
        if current is None:
            return {"flow_valid": False, "reason": "forward_flow_failed"}
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, **params)
        if backward is None:
            return {"flow_valid": False, "reason": "backward_flow_failed"}
        fb = np.linalg.norm(backward[:, 0] - self.features[:, 0], axis=1)
        good = (status[:, 0] != 0) & (back_status[:, 0] != 0) & (fb <= self.config["maximum_forward_backward_error_px"])
        if int(good.sum()) < 4:
            return {"flow_valid": False, "reason": "insufficient_consistent_box_features"}
        source, target = self.features[good, 0], current[good, 0]
        transform, inlier = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5,
            maxIters=2000, confidence=.99,
        )
        if transform is None or inlier is None or int(inlier.sum()) < 4:
            return {"flow_valid": False, "reason": "rigid_fit_failed"}
        inlier = inlier[:, 0].astype(bool)
        fitted = cv2.transform(source[None], transform)[0]
        error = np.linalg.norm(fitted - target, axis=1)[inlier]
        scale = float(np.linalg.norm(transform[0, :2]))
        fit_error = float(np.quantile(error, .9))
        if not .75 <= scale <= 1.25 or fit_error > self.config["maximum_rigid_fit_error_px"]:
            return {"flow_valid": False, "reason": "unstable_box_projection"}
        corners = cv2.transform(self.corners[None], transform)[0]
        change = corners - self.corners
        raw = self.slope * change[:, 0] - change[:, 1]
        if self.initial_bias is None:
            self.initial_bias = raw.copy()
        relative = raw - self.initial_bias
        # Require displacement above both the common input reference and this
        # stream's initial appearance: a static first-frame offset is not lift.
        displacement = np.minimum(raw, relative)
        uncertainty = max(
            self.config["flow_uncertainty_floor_px"],
            fit_error + float(np.quantile(fb[good][inlier], .9)),
        )
        visible = [bool(visible and 2 < point[0] < 637 and 2 < point[1] < 477)
                   for visible, point in zip(self.visible, corners)]
        return {
            "flow_valid": True, "corners": corners.tolist(),
            "corner_visible": visible, "corner_displacement_px": displacement.tolist(),
            "common_reference_displacement_px": raw.tolist(),
            "initial_appearance_offset_px": self.initial_bias.tolist(),
            "uncertainty_margin_px": uncertainty, "fit_error_p90_px": fit_error,
            "feature_inliers": int(inlier.sum()), "scale": scale,
            "box_rotation_deg": float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))),
        }


class RefinedStickContact:
    def __init__(self, initial_frame, config):
        self.config = config
        self.silhouette = StickLiftContact(initial_frame)
        self.references = self.silhouette.reference
        self.feet = [] if self.references is None else [
            FootReference(initial_frame, body, self.silhouette.slope, config)
            for body in self.references
        ]
        self.edge_bias = None
        self.angle_bias = None

    def update(self, image):
        base = self.silhouette.update(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if "left_clearance_px" in base:
            raw_edges = [base["left_clearance_px"], base["right_clearance_px"]]
            if self.edge_bias is None:
                self.edge_bias = raw_edges
                self.angle_bias = base["angle_change_deg"]
            edge = [min(raw, raw - bias) for raw, bias in zip(raw_edges, self.edge_bias)]
            tilt = base["angle_change_deg"] - self.angle_bias
        else:
            edge, tilt = [None, None], None
        sides = []
        for index, foot in enumerate(self.feet):
            result = foot.measure(gray)
            result["side"] = "left" if index == 0 else "right"
            result["silhouette_displacement_px"] = edge[index]
            result["state"] = "uncertain"
            if result["flow_valid"]:
                visible = np.asarray(result["corner_visible"], dtype=bool)
                gaps = np.asarray(result["corner_displacement_px"])
                margin = result["uncertainty_margin_px"]
                result["corner_lower_margin_px"] = (gaps - margin).tolist()
                result["corner_upper_margin_px"] = (gaps + margin).tolist()
                if visible.any() and edge[index] is not None:
                    minimum = float(gaps[visible].min())
                    silhouette_margin = self.config["silhouette_uncertainty_floor_px"]
                    agreement = abs(minimum - edge[index]) <= max(1.5, 3 * margin, .1 * abs(minimum))
                    result["flow_silhouette_agree"] = bool(agreement)
                    if (visible.all() and agreement and minimum > margin
                            and edge[index] > silhouette_margin):
                        result["state"] = "airborne_proposal"
                    elif agreement and abs(minimum) <= margin and abs(edge[index]) <= silhouette_margin:
                        # Contact-consistent, not mathematical proof of zero gap.
                        # This endpoint remains useful when the OTHER end is hidden.
                        result["state"] = "contact_consistent"
            sides.append(result)
        state = "uncertain"
        if len(sides) == 2:
            if all(side["state"] == "airborne_proposal" for side in sides):
                state = "both_airborne_balanced" if tilt is not None and abs(tilt) <= self.config["maximum_tilt_deg"] else "both_airborne_tilted"
            elif any(side["state"] == "contact_consistent" for side in sides):
                state = "visible_endpoint_contact_evidence"
        return {"state": state, "sides": sides, "angle_change_deg": tilt,
                "silhouette": base,
                "meaning": "uncertainty margins are heuristic, not calibrated confidence intervals"}


def summarize_refined(rows, fps, config):
    evaluated = [row for row in rows if row["evaluated"]]
    count = max(2, int(np.ceil(config["hold_seconds"] * fps - 1e-6)) + 1)
    clear = [row["state"] == "both_airborne_balanced" for row in evaluated]
    possible = [row["state"] in ("both_airborne_balanced", "uncertain") for row in evaluated]
    if longest_run(clear) >= count:
        status = "balanced_lift_proposal"
    elif len(evaluated) >= count and longest_run(possible) < count:
        status = "contact_or_tilt_supported_nonlift"
    else:
        status = "unresolved"
    return {"status": status, "evaluated_frames": len(evaluated),
            "both_clear_frames": sum(clear),
            "contact_evidence_frames": sum(row["state"] == "visible_endpoint_contact_evidence" for row in evaluated),
            "unresolved_frames": sum(row["state"] == "uncertain" for row in evaluated),
            "longest_balanced_interval_seconds": max(0, longest_run(clear) - 1) / fps,
            "status_is_provisional": True}
