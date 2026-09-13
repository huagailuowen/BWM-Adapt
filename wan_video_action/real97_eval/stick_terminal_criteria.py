"""Source-clock, padding-aware terminal lift criterion; no video decoding."""
from bisect import bisect_right
from collections import Counter


def terminal_lift_summary(rows, record, hold_seconds=0.3):
    """Require sustained evidence at the commanded lift's final sampled window.

    Encoded MP4 frame rates (e.g. 6.67 instead of 20/3) are not the measurement
    clock. Geometry/flow uncertainty remains in the supplied per-frame states.
    An unobserved phase ending is censored, never promoted to action failure.
    """
    fps = float(record["source_fps"])
    stride = int(record["stride"])
    start = int(record["start_frame"])
    total = int(record["total_frames"])
    lift_start, lift_last = int(record["lift_start"]), int(record["lift_last_frame"])
    if fps <= 0 or stride <= 0 or hold_seconds <= 0 or lift_last < lift_start:
        raise ValueError("Invalid source clock or commanded lift interval")
    real, padding = [], 0
    for row in rows:
        source = start + int(row["frame"]) * stride
        if source >= total:
            padding += 1
            continue
        if record["scope"] == "query" and int(row["frame"]) == 0:
            continue
        if lift_start <= source <= lift_last:
            real.append((source, row))
    result = {
        "status": "unresolved", "criterion": "terminal_lift_phase",
        "longest_accepted_seconds": 0.0, "longest_possible_seconds": 0.0,
        "longest_negative_seconds": 0.0, "accepted_interval": None,
        "negative_interval": None, "frame_counts": {},
        "frame_count_scope": "terminal_window_only",
        "duration_scope": "terminal_window_only_not_whole_video_maximum",
        "bilateral_tracking_fraction": sum(
            len(r["sides"]) == 2 and all(s.get("valid") for s in r["sides"])
            for _, r in real) / max(1, len(real)),
        "source_fps": fps, "source_stride": stride,
        "excluded_padding_frames": padding, "real_lift_frames": len(real),
        "required_lift_last_frame": lift_last,
        "horizon_status": "unobserved", "terminal_window": {},
        "note": "Any earlier transient balance does not satisfy this terminal criterion"
    }
    if not real:
        result["reason"] = "no_real_nonconditioning_lift_frames"
        return result
    sources = [s for s, _ in real]
    if any(b <= a for a, b in zip(sources, sources[1:])):
        raise ValueError("Source-frame indices must be strictly increasing")
    end = sources[-1]
    # The final sample represents the phase end only to the documented sampling
    # cadence. Larger missing tails are explicitly incomplete prediction horizons.
    result["terminal_endpoint_gap_frames"] = lift_last - end
    covered = lift_last - end < stride
    result["horizon_status"] = "covered_to_sampling_resolution" if covered else "incomplete_lift_horizon"
    cutoff = end - hold_seconds * fps
    first = bisect_right(sources, cutoff + 1e-8) - 1
    if first < 0:
        result["reason"] = "less_than_required_real_lift_duration"
        return result
    tail = real[first:]
    duration = (tail[-1][0] - tail[0][0]) / fps
    interval = [int(tail[0][1]["frame"]), int(tail[-1][1]["frame"])]
    result["terminal_window"] = {
        "frame_interval": interval, "source_frame_interval": [tail[0][0], tail[-1][0]],
        "duration_seconds": duration, "required_seconds": hold_seconds,
        "sample_count": len(tail)
    }
    result["frame_counts"] = dict(Counter(r["state"] for _, r in tail))
    if not covered:
        result["reason"] = "prediction_does_not_observe_commanded_lift_end"
        return result
    if duration + 1e-9 < hold_seconds or any(b - a != stride for a, b in zip(sources[first:], sources[first + 1:])):
        result["reason"] = "insufficient_duration_or_missing_source_samples"
        return result
    states = [r["state"] for _, r in tail]
    if all(s == "balanced_lift_proposal" for s in states):
        result.update(status="balanced_lift_proposal", accepted_interval=interval,
                      longest_accepted_seconds=duration, reason="sustained_terminal_balance")
    elif all(s in ("below_absolute", "asymmetric_rise") for s in states):
        result.update(status="no_qualifying_lift_observed", negative_interval=interval,
                      longest_negative_seconds=duration, reason="sustained_terminal_unbalance")
    else:
        result.update(reason="terminal_evidence_uncertain_or_not_sustained",
                      longest_possible_seconds=duration)
    return result
