"""Frozen eligibility and support-quality policy; never reads query outcomes."""

DOOR_ACCEPTED_LEVELS = {
    "door-0": (1, 2),
    "door-2u-1d": (2, 3),
    "door-3u": (3, 4),
    "door-2u-4d": (4, 5),
    "door-4d": (3, 4),
    "door-3u-4d": (5, 6),
    "door-6u-7d": (7, 8),
    "door-6u-8d": (8, 9),
    "door-12u-Half": (9, 10),
    "door-12u-Half-11d-Half": (),
}


def score_door_level(environment, selected_level):
    accepted = DOOR_ACCEPTED_LEVELS[environment]
    return {
        "action_selection_eligible": bool(accepted),
        "accepted_levels": list(accepted),
        "action_selection_correct": int(selected_level) in accepted if accepted else None,
        # Physical closure is measured separately. In particular a critical
        # level-3 door-4d recording can rebound without changing the accepted set.
    }


def ball_support_eligibility(record):
    if not record.get("training_eligible", False):
        return False, "not_eligible_train"
    if record.get("missing_rate", 1.0) > .02 or record.get("ambiguous_frames", 1) > 0:
        return False, "tracking_requires_review"
    peak = record.get("peak_x_original_px")
    if peak is None:
        return False, "no_ball_peak"
    if 275 <= peak <= 335:
        return False, "target_or_target_boundary"
    if peak > 580 or record.get("boundary_seen"):
        return False, "too_far_or_boundary"
    if record.get("post_action_roll_px", 0) < 25:
        return False, "insufficient_free_roll"
    return True, "candidate_requires_chunk_and_visual_audit"


def desired_door_support_levels(environment):
    accepted = DOOR_ACCEPTED_LEVELS[environment]
    if not accepted:
        return [9, 10]
    minimum = min(accepted)
    return list(dict.fromkeys([2] + ([minimum - 1] if minimum > 1 else [])))


def support_source_allowed(row):
    return row["split"] == "train" and row["domain"] == "id"
