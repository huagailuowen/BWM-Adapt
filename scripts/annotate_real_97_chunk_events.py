#!/usr/bin/env python3
"""Create non-destructive event sidecars for the September 7 real datasets.

No training, video editing, split changes, or original metadata rewrites occur.
Door has no per-frame target: its annotations explicitly use measured EEF motion.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow.parquet as pq


DATASETS = {
    "ball": ("ball_friction_9_7_crop", 61, 1),
    "door": ("door_close_9_7", 49, 1),
    "stick": ("stick_balance_9_7", 41, 3),
}
OUTPUT_NAMES = (
    "chunk_events_v1.jsonl",
    "chunk_events_v1_summary.json",
    "chunk_events_v1_examples.svg",
    "CHUNK_EVENTS_V1.md",
)


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def publish_new(path, content):
    """Atomically publish a complete new file, never replacing an existing file."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".chunk-events-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def rotation_about_negative_x(poses):
    q = np.asarray(poses, dtype=np.float64)[:, 3:7].copy()
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(q).all() or (norm < 1e-10).any():
        raise ValueError("Invalid wxyz quaternion")
    q /= norm
    for index in range(1, len(q)):
        if np.dot(q[index - 1], q[index]) < 0:
            q[index] *= -1
    # q(t) * inverse(q(0)), expressed in the base frame, not Euler roll.
    w = q[:, 0] * q[0, 0] + (q[:, 1:] * q[0, 1:]).sum(axis=1)
    v = -q[:, :1] * q[0, 1:] + q[0, 0] * q[:, 1:] - np.cross(q[:, 1:], q[0, 1:])
    return np.rad2deg(np.unwrap(-2 * np.arctan2(v[:, 0], w)))


def sustained_start(signal, direction, after=1, threshold=0.05):
    """First moving frame, requiring two consecutive signed moving intervals."""
    delta = np.diff(signal, prepend=signal[0]) * direction
    for index in range(max(1, after), len(signal) - 1):
        if delta[index] > threshold and delta[index + 1] > threshold:
            return index
    return None


def event(frame, fps, signal=None):
    if frame is None:
        return None
    result = {"frame": int(frame), "time_s": round(int(frame) / fps, 6)}
    if signal is not None:
        result["signal_value"] = round(float(signal[frame]), 7)
    return result


def start_range(low, high, probability):
    low, high = int(low), int(high)
    return {
        "probability": probability,
        "start_min_inclusive": low,
        "start_max_inclusive": high,
        "candidate_count": max(0, high - low + 1),
    }


def annotate_ball(row, table, fps):
    field = "raw.target_action.gello_state_raw"
    signal = rotation_about_negative_x(table[field].to_pylist())
    source = row["skill_annotation"]
    anchors = {
        "left_start": int(source["skill_start_frame"]),
        "left_extreme": int(source["left_extreme_frame"]),
        "right_start": int(source["right_start_frame"]),
        "right_end": int(source["skill_end_frame"]),
    }
    if not (0 <= anchors["left_start"] <= anchors["left_extreme"]
            <= anchors["right_start"] <= anchors["right_end"] < len(signal)):
        raise ValueError("Existing target motion anchors are out of order")
    right = anchors["right_start"]
    if signal[right] >= 0:
        raise ValueError("Right swing does not start to the left of the initial angle")
    crossing = np.flatnonzero(signal[right:anchors["right_end"] + 1] >= 0)
    if not len(crossing):
        raise ValueError("Target right swing never crosses its initial orientation")
    crossing = int(right + crossing[0])
    anchors.update(return_to_initial=crossing, pre_return=crossing - 3)
    if anchors["pre_return"] < anchors["left_start"]:
        raise ValueError("Three-frame pre-return margin precedes left swing")
    ranges = {
        "precise": start_range(anchors["right_start"], crossing - 3, 0.5),
        "wide": start_range(0, crossing - 3, 0.5),
    }
    # Publish the documented ranges and a separate, explicit coverage guard.
    guarded = {
        name: start_range(max(branch["start_min_inclusive"], crossing - 60),
                          branch["start_max_inclusive"], branch["probability"])
        for name, branch in ranges.items()
    }
    return {
        "status": "target_derived",
        "source": field,
        "anchor_source": "meta/episodes.jsonl:skill_annotation",
        "signal_unit": "degree_about_base_negative_x_relative_to_first_frame",
        "events": {key: event(value, fps, signal) for key, value in anchors.items()},
        "documented_start_ranges": ranges,
        "start_ranges": guarded,
        "coverage_guard": "start + 60 >= return_to_initial.frame",
        "warnings": [],
    }, signal


def annotate_door(row, table, fps):
    field = "observation.eef_state"
    signal = rotation_about_negative_x(table[field].to_pylist())
    left = sustained_start(signal, -1)
    if left is None:
        raise ValueError("No sustained measured left swing")
    right = sustained_start(signal, 1)
    warnings = ["No recorded per-frame target; all events are measured-motion proxies."]
    if right is None or right >= left:
        right = None
        warnings.append("No sustained right preparation before left swing.")
    trough = left + int(np.argmin(signal[left:]))
    crossing = np.flatnonzero(signal[left:trough + 1] <= 0)
    if not len(crossing):
        raise ValueError("Measured left swing never crosses its initial orientation")
    crossing = int(left + crossing[0])
    before_zero = max(left, crossing - 1)
    if crossing == left:
        warnings.append("Crossing occurs on first left-moving frame; precise range is a singleton.")
    anchors = {
        "right_preparation_start": right,
        "right_extreme": int(np.argmax(signal[:left + 1])) if right is not None else None,
        "left_strike_start": left,
        "before_zero_crossing": before_zero,
        "zero_crossing": crossing,
        "left_extreme": trough,
        "rebound_start": sustained_start(signal, 1, after=trough + 1),
    }
    ranges = {"precise": start_range(left, before_zero, 0.5),
              "wide": start_range(0, before_zero, 0.5)}
    guarded = {
        name: start_range(max(branch["start_min_inclusive"], crossing - 48),
                          branch["start_max_inclusive"], branch["probability"])
        for name, branch in ranges.items()
    }
    return {
        "status": "measured_proxy",
        "source": field,
        "target_events_available": False,
        "requires_target_policy_approval": True,
        "signal_unit": "degree_about_base_negative_x_relative_to_first_frame",
        "detector": {"threshold_deg_per_frame": 0.05, "consecutive_intervals": 2},
        "events": {key: event(value, fps, signal) for key, value in anchors.items()},
        "documented_start_ranges": ranges,
        "start_ranges": guarded,
        "coverage_guard": "start + 48 >= zero_crossing.frame",
        "warnings": warnings,
    }, signal


def annotate_stick(row, table, fps, segments):
    signal = np.asarray(table["action"].to_pylist(), dtype=np.float64)[:, 2]
    lifts = [item for item in segments if item["label"] == "lift"]
    if not lifts:
        raise ValueError("No existing target lift segment")
    for item in segments:
        if not 0 <= item["start_frame"] < item["end_frame_exclusive"] <= len(signal):
            raise ValueError("Existing target segment is out of bounds")
    begin = min(item["start_frame"] for item in lifts)
    end = max(item["end_frame_exclusive"] for item in lifts)
    forwards = [item for item in segments if item["label"] == "forward"]
    max_start = max(0, len(signal) - 121)
    precise = start_range(max(0, end - 1 - 120), min(begin, max_start), 0.7)
    if precise["candidate_count"] == 0:
        raise ValueError("Full lift envelope cannot fit a 41x3 window")
    events = {"lift_start": event(begin, fps, signal),
              "lift_last_frame": event(end - 1, fps, signal)}
    if forwards:
        events["forward_start"] = event(min(item["start_frame"] for item in forwards), fps, signal)
        events["forward_last_frame"] = event(max(item["end_frame_exclusive"] for item in forwards) - 1, fps, signal)
    return {
        "status": "existing_target_segments",
        "source": "meta/action_segments.jsonl; action[:,2]",
        "signal_unit": "target_z_meter",
        "events": events,
        "source_segments": segments,
        "lift_interval": {"start_inclusive": begin, "end_exclusive": end},
        "lift_reaches_recording_end": end == len(signal),
        "lift_target_delta_z_m": float(signal[end - 1] - signal[begin]),
        "start_ranges": {"general": start_range(0, max_start, 0.3), "full_lift": precise},
        "warnings": [],
    }, signal


def example_svg(rows, task):
    usable = [row for row in rows if row["status"] != "needs_review"]
    ordered = sorted(usable, key=lambda row: row["num_frames"])
    selected = [ordered[index] for index in np.linspace(0, len(ordered) - 1, min(6, len(ordered))).astype(int)] if ordered else []
    output = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="640" viewBox="0 0 1500 640">',
              '<rect width="1500" height="640" fill="white"/>',
              '<style>text{font-family:monospace;font-size:11px}</style>',
              f'<text x="20" y="22">{html.escape(task)}: event annotations; blue=precise/full-lift start range, gray=wide/general</text>']
    for index, row in enumerate(selected):
        ox, oy = 30 + (index % 3) * 500, 65 + (index // 3) * 290
        values = row["reference_signal_values"]
        low, high = min(values), max(values)
        span = max(high - low, 1e-6)
        px = lambda frame: ox + 40 + frame * 390 / max(1, len(values) - 1)
        py = lambda value: oy + 165 - (value - low) * 140 / span
        title = f'{row["environment"]} ep{row["episode_index"]:06d} level={row["level"]} N={row["num_frames"]}'
        output.append(f'<text x="{ox}" y="{oy - 15}">{html.escape(title)}</text>')
        points = ' '.join(f'{px(i):.2f},{py(value):.2f}' for i, value in enumerate(values))
        output.append(f'<path d="M{ox+40},{oy+15} V{oy+175} H{ox+430}" stroke="#888" fill="none"/>')
        output.append(f'<polyline points="{points}" stroke="#222" stroke-width="1.5" fill="none"/>')
        for j, (name, branch) in enumerate(row["start_ranges"].items()):
            color = '#1384a5' if name in ('precise', 'full_lift') else '#888888'
            a, b = branch["start_min_inclusive"], branch["start_max_inclusive"]
            output.append(f'<path d="M{px(a):.2f},{oy+190+j*22} H{px(b):.2f}" stroke="{color}" stroke-width="7"/>')
            output.append(f'<text x="{ox+40}" y="{oy+205+j*22}">{name}: [{a}, {b}]</text>')
        labels = []
        for name, node in row["events"].items():
            if node is not None:
                f = node['frame']
                output.append(f'<circle cx="{px(f):.2f}" cy="{py(values[f]):.2f}" r="3" fill="#b84923"/>')
                labels.append(f'{name}={f}')
        for j in range(0, len(labels), 2):
            output.append(f'<text x="{ox}" y="{oy+250+(j//2)*12}">{html.escape("  ".join(labels[j:j+2]))}</text>')
    output.append('</svg>')
    return '\n'.join(output)


def build(root, task, frames, stride):
    rows = []
    split_map = {}
    if task == "ball":
        splits = json.loads((root / "TRAIN_TEST_SPLIT.json").read_text())
        for key in ("train_episodes", "test_episodes"):
            for item in splits[key]:
                split_map[(item["lerobot_dataset"], item["lerobot_episode_index"])] = item["split"]
    for info_path in sorted(root.glob("*_lerobot/meta/info.json")):
        env = info_path.parent.parent
        info = json.loads(info_path.read_text())
        fps = float(info["fps"])
        if fps != 20:
            raise ValueError(f"Expected 20 Hz: {env}")
        episodes = jsonl(env / "meta/episodes.jsonl")
        segment_map = {item["episode_index"]: item["segments"] for item in jsonl(env / "meta/action_segments.jsonl")} if task == "stick" else {}
        for episode in episodes:
            index = episode["episode_index"]
            n = int(episode["length"])
            fields = {"ball": ["raw.target_action.gello_state_raw"],
                      "door": ["observation.eef_state"], "stick": ["action"]}[task]
            matches = list((env / "data").glob(f"*/episode_{index:06d}.parquet"))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one parquet for {env.name}/{index}")
            table = pq.read_table(matches[0], columns=fields)
            if len(table) != n or n < 1:
                raise ValueError(f"Episode length mismatch: {matches[0]}")
            row = {
                "schema_version": 1, "task": task,
                "environment": env.name.removesuffix("_lerobot"),
                "lerobot_dataset": env.name,
                "episode_index": index,
                "source_episode_index": episode.get("source_episode_index"),
                "level": episode.get("level", episode.get("skill_gear")),
                "split": split_map.get((env.name, index), episode.get("split")),
                "num_frames": n, "fps": fps, "model_frames": frames,
                "stride": stride, "native_span_frames": 1 + (frames - 1) * stride,
                "padding": "repeat_last_row_for_video_action_state",
                "object_contact_frame": None, "object_motion_end_frame": None,
            }
            try:
                if task == "ball":
                    details, signal = annotate_ball(episode, table, fps)
                elif task == "door":
                    details, signal = annotate_door(episode, table, fps)
                else:
                    details, signal = annotate_stick(episode, table, fps, segment_map[index])
                row.update(details)
                row["reference_signal_values"] = np.round(signal, 7).tolist()
                for branch in row["start_ranges"].values():
                    a, b = branch["start_min_inclusive"], branch["start_max_inclusive"]
                    if not 0 <= a <= b < n:
                        raise ValueError(f"Invalid start range: {branch}")
                    branch["padded_model_frames_at_first_start"] = int(np.count_nonzero(a + stride * np.arange(frames) >= n))
                    branch["padded_model_frames_at_last_start"] = int(np.count_nonzero(b + stride * np.arange(frames) >= n))
            except (ValueError, KeyError, IndexError) as error:
                row.update(status="needs_review", error=str(error), start_ranges={})
            rows.append(row)
    if not rows:
        raise ValueError(f"No episodes found in {root}")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root), "task": task,
        "environments": len({row["environment"] for row in rows}), "episodes": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        "model_frames": frames, "stride": stride,
        "native_span_frames": 1 + (frames - 1) * stride,
        "short_episodes_retained": sum(row["num_frames"] < row["native_span_frames"] for row in rows),
        "episodes_removed": 0,
        "review_episodes": [{k: row[k] for k in ("environment", "episode_index", "error")} for row in rows if row["status"] == "needs_review"],
        "coverage_guard_changed_episodes": sum(
            "documented_start_ranges" in row and any(
                branch["start_min_inclusive"] != row["documented_start_ranges"][key]["start_min_inclusive"]
                for key, branch in row["start_ranges"].items()) for row in rows),
    }
    for key in sorted({key for row in rows for key in row.get("events", {})}):
        values = [row["events"][key]["frame"] for row in rows if row.get("events", {}).get(key) is not None]
        summary.setdefault("event_frame_statistics", {})[key] = {
            "count": len(values), "min": int(min(values)),
            "median": float(np.median(values)), "max": int(max(values)),
        }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    args = parser.parse_args()
    # Fail before doing any work if this would overwrite an earlier annotation.
    for folder, _, _ in DATASETS.values():
        for name in OUTPUT_NAMES:
            path = args.datasets_root / folder / name
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
    for task, (folder, frames, stride) in DATASETS.items():
        root = args.datasets_root / folder
        rows, summary = build(root, task, frames, stride)
        payloads = [
            ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows),
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            example_svg(rows, task),
            f"# Chunk event annotations v1: {task}\n\n"
            f"Episodes: {len(rows)}. Model frames: {frames}. Native-frame stride: {stride}.\n\n"
            "Generated by `BWM-Adapt/scripts/annotate_real_97_chunk_events.py`.\n"
            "See `BWM-Adapt/docs/real_97_chunk_sampling.md` for the proposed sampling contract.\n\n"
            "All frame indices are zero-based. Start ranges are inclusive; segment ends explicitly use exclusive indices. "
            "Sample indices are `min(start + stride*k, num_frames-1)` for video, action and state alike. "
            "All short episodes are retained. Existing train/test splits are unchanged.\n\n"
            "Ball anchors come from existing target labels; return crossing comes from recorded target quaternions. "
            "Door events are MEASURED EEF proxies, NOT recorded target events. "
            "Stick lift intervals preserve the existing target segmentation.\n\n"
            "Ball sampling uses 50% uniform starts in [right_start, pre_return] and "
            "50% in [0, pre_return], where pre_return = return_to_initial - 3. "
            "If right_start exceeds pre_return, the precise range is empty and the "
            "episode is marked needs_review, not silently removed or assigned a different rule.\n\n"
            "These are robot-motion annotations, not visual annotations of ball contact, door closure, or object rest. "
            "Unknown object events remain null. A `needs_review` row remains in the file, never silently removed. "
            "No training loader or training job is changed by this command.\n",
        ]
        for name, content in zip(OUTPUT_NAMES, payloads):
            publish_new(root / name, content)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
