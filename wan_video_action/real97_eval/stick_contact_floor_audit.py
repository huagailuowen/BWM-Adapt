"""Paired silhouette diagnostic; never changes model or success decisions."""
import cv2
import numpy as np

from .stick_common_reference_rigid_review import actual_crop, transform
from .stick_shape_flow_review import background_colors, export_shape, segment


def floor_blue_candidates(frame, original, warped, slope, side, config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (92, 65, 20), (125, 255, 255))
    width, height = config["vertical_blue_kernel"]
    tall = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((height, width), np.uint8))
    width, height = config["vertical_blue_margin"]
    tall = cv2.dilate(tall, np.ones((height, width), np.uint8))
    candidates = cv2.bitwise_and(blue, cv2.bitwise_not(tall))
    x, y, w, h = original["bbox"]
    yy, xx = np.indices(blue.shape)
    hull = np.asarray(original["hull"]).reshape(-1, 2)
    floor_offset = float(np.max(hull[:, 1] - slope * hull[:, 0]))
    near_floor = np.abs(yy - slope * xx - floor_offset) <= config["floor_band_half_width_px"]
    near_body_x = (xx >= x - config["reference_x_margin_px"]) & (xx <= x + w + config["reference_x_margin_px"])
    own_half = xx < 320 if side == 0 else xx >= 320
    candidates[~(near_floor & near_body_x & own_half)] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    selected = np.zeros_like(candidates)
    for label in range(1, count):
        _, _, cw, ch, area = stats[label]
        if (area >= config["floor_component_min_area"] and cw >= config["floor_component_min_width_px"]
                and ch <= config["floor_component_max_height_px"]
                and cw >= config["floor_component_min_width_height_ratio"] * ch):
            selected[labels == label] = 255
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8))
    # Do not erase blue pixels inside the tracked box merely because they are static.
    interior = cv2.erode(warped, np.ones((5, 5), np.uint8))
    selected[interior > 0] = 0
    return selected


def candidate_shape(frame, original, matrix, side, extra_background, config, seed):
    height, width = frame.shape[:2]
    half = (0, 320) if side == 0 else (320, 640)
    warped = cv2.warpAffine(original["mask"], matrix, (width, height), flags=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (92, 65, 20), (125, 255, 255))
    seeds = cv2.bitwise_and(cv2.erode(warped, np.ones((7, 7), np.uint8)), blue)
    seeds[extra_background > 0] = 0
    if np.count_nonzero(seeds) < 10:
        return {"valid": False, "reason": "insufficient_body_seeds"}
    result = {"valid": False, "reason": "empty_allowed_region"}
    for halo in config["shape_halos_px"]:
        allowed = cv2.dilate(warped, np.ones((2 * halo + 1,) * 2, np.uint8))
        allowed[:, :half[0]] = 0
        allowed[:, half[1]:] = 0
        yy, xx = np.where(allowed > 0)
        if len(xx) < 60:
            continue
        ring = config["background_ring_px"]
        roi = (max(half[0], int(xx.min()) - ring), max(140, int(yy.min()) - ring),
               min(half[1], int(xx.max()) + ring + 1), min(height, int(yy.max()) + ring + 1))
        forbidden = cv2.bitwise_or(cv2.bitwise_not(allowed), background_colors(frame))
        forbidden = cv2.bitwise_or(forbidden, extra_background)
        cv2.setRNGSeed(seed + halo)
        result = segment(frame, roi, seeds, warped, forbidden, config)
        if not result.get("valid"):
            continue
        edge = cv2.subtract(allowed, cv2.erode(allowed, np.ones((3, 3), np.uint8)))
        hits = int(np.count_nonzero(cv2.bitwise_and(edge, result["mask"])))
        result.update(search_halo_px=halo, search_boundary_pixels=hits,
                      search_truncated=hits > config["maximum_halo_boundary_pixels"],
                      actual_image_crop=actual_crop(result, side, width, height),
                      area_ratio=float(np.count_nonzero(result["mask"])) / max(1, np.count_nonzero(warped)))
        if not result["search_truncated"]:
            break
    return result


def analyze_side(frame, original, pose, slope, side, config, seed):
    if not original.get("valid") or not pose.get("valid") or slope is None:
        return {"valid": False, "reason": "missing_reference_or_pose"}, np.zeros(frame.shape[:2], np.uint8)
    matrix = np.asarray(pose["matrix"])
    warped = cv2.warpAffine(original["mask"], matrix, (640, 480), flags=cv2.INTER_NEAREST)
    floor = floor_blue_candidates(frame, original, warped, slope, side, config)
    baseline = candidate_shape(frame, original, matrix, side, np.zeros_like(floor), config, seed)
    filtered = candidate_shape(frame, original, matrix, side, floor, config, seed)
    hull = np.asarray(original["hull"]).reshape(-1, 2)
    ref_edge = float(np.max(hull[:, 1] - slope * hull[:, 0]))
    projected = transform(hull, matrix)
    rigid_gap = ref_edge - float(np.max(projected[:, 1] - slope * projected[:, 0]))
    gaps = {}
    overlaps = {}
    for name, shape in (("baseline", baseline), ("filtered", filtered)):
        if shape.get("valid"):
            points = np.asarray(shape["hull"]).reshape(-1, 2)
            gaps[name] = ref_edge - float(np.max(points[:, 1] - slope * points[:, 0]))
            overlaps[name] = int(np.count_nonzero(cv2.bitwise_and(shape["mask"], floor)))
        else:
            gaps[name] = None
            overlaps[name] = None
    shift = gaps["filtered"] - gaps["baseline"] if all(v is not None for v in gaps.values()) else None
    return {"valid": True, "side": side, "pose_source": pose["source"], "rigid_gap_px": rigid_gap,
            "floor_candidate_pixels": int(np.count_nonzero(floor)), "floor_overlap_pixels": overlaps,
            "gaps_px": gaps, "filtered_minus_baseline_gap_px": shift,
            "baseline": export_shape(baseline), "filtered": export_shape(filtered),
            "projected_hull": projected.tolist()}, floor
