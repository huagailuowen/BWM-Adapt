"""Opt-in, boundary-safe search for the v6 terminal-contact diagnostic.

The legacy module is replaced only inside a process-local context and restored
on exit. No v6/v7 caller or on-disk source is changed. This context is intended
for the single-threaded CPU evaluation driver, not concurrent Python callers.
"""

from collections import Counter
from contextlib import contextmanager
import math

import numpy as np

from wan_video_action.real97_eval import stick_local_contact_edges as legacy


class BoundarySafeSearch:
    def __init__(self, radii, margin):
        self.radii = tuple(int(radius) for radius in radii)
        self.margin = int(margin)
        self.events = []

    def __call__(self, response, excluded, x, expected_y, config):
        height, width = response.shape
        ix = int(round(x))
        event = {"x": float(x), "expected_y": float(expected_y), "attempts": []}
        self.events.append(event)
        if not 2 <= ix < width - 2 or not math.isfinite(expected_y):
            event["outcome"] = "invalid_coordinate"
            return None
        for radius in self.radii:
            lo = max(3, math.floor(expected_y) - radius)
            hi = min(height - 4, math.ceil(expected_y) + radius)
            attempt = {"radius_px": radius, "window": [lo, hi]}
            event["attempts"].append(attempt)
            if hi < lo:
                attempt["reason"] = "empty_window"
                break
            yy = np.arange(lo, hi + 1)
            strength = response[yy, ix]
            valid = (strength >= config["minimum_edge_contrast"]) & (excluded[yy, ix] == 0)
            if not np.any(valid):
                attempt["reason"] = "no_unmasked_edge"
                break
            score = strength - config["distance_penalty"] * np.abs(yy - expected_y)
            score = np.where(valid, score, -np.inf)
            y = int(yy[int(np.argmax(score))])
            attempt.update(peak_y=y, contrast=float(response[y, ix]))
            if min(y - lo, hi - y) < self.margin:
                attempt["reason"] = "search_boundary_peak"
                continue
            lower, center, upper = (float(v) for v in response[y - 1:y + 2, ix])
            if np.any(excluded[y - 1:y + 2, ix] != 0):
                attempt["reason"] = "masked_peak_bracket"
                break
            if center < lower or center < upper or center == lower == upper:
                attempt["reason"] = "unbracketed_gradient_peak"
                break
            denom = lower - 2 * center + upper
            offset = 0.0 if denom >= -1e-5 else float(np.clip(0.5 * (lower - upper) / denom, -0.5, 0.5))
            broadness = int(np.count_nonzero(response[max(0, y - 4):min(height, y + 5), ix] >= 0.8 * center))
            competitors = strength[np.abs(yy - y) >= 3]
            point = {
                "x": float(x), "y": float(y + offset), "contrast": center,
                "prominence": center - float(np.max(competitors)) if competitors.size else center,
                "localization_radius_px": max(0.5, broadness / 2),
                "displacement_from_pose_px": float(y + offset - expected_y),
                "search_radius_px": radius,
                "search_expanded": radius != self.radii[0],
                "search_boundary_distance_px": min(y - lo, hi - y),
                "initial_peak_y": event["attempts"][0].get("peak_y"),
            }
            attempt["reason"] = "accepted"
            event["outcome"] = "accepted_expanded" if point["search_expanded"] else "accepted_initial"
            event["point"] = point
            return point
        event["outcome"] = event["attempts"][-1]["reason"] if event["attempts"] else "no_window"
        return None

    def summary(self):
        return {
            "calls": len(self.events),
            "outcomes": dict(Counter(event["outcome"] for event in self.events)),
            "attempt_reasons": dict(Counter(attempt["reason"] for event in self.events for attempt in event["attempts"])),
        }


@contextmanager
def boundary_safe_search(radii, margin):
    search = BoundarySafeSearch(radii, margin)
    previous = legacy.find_edge
    legacy.find_edge = search
    try:
        yield search
    finally:
        legacy.find_edge = previous
