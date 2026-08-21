#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_ROOT = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/"
    "ball_friction_8_17_align_medium"
)
ENVIRONMENTS = (
    "ball-0_lerobot",
    "ball-1_lerobot",
    "ball-2_lerobot",
    "ball-3_lerobot",
    "ball-4_lerobot",
    "ball-7_lerobot",
    "ball-9_lerobot",
)
ANNOTATION_NAME = "ball_friction_skill_annotations.jsonl"
SUMMARY_NAME = "ball_friction_skill_annotation_summary.json"
SCHEMA_VERSION = "ball-friction-pivot-skill-v1"
ROTATION_AXIS_BASE = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
EXPECTED_FULL_SPEED_DEG_S = 86.666666667
LOW_MOTION_FRACTION = 0.01
HIGH_MOTION_FRACTION = 0.99


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lv = left[0], left[1:]
    rw, rv = right[0], right[1:]
    return np.concatenate(
        [
            np.asarray([lw * rw - np.dot(lv, rv)]),
            lw * rv + rw * lv + np.cross(lv, rv),
        ]
    )


def base_axis_rotation_deg(eef_state: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(eef_state[:, 3:7], dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Encountered a zero-norm end-effector quaternion.")
    quaternions /= norms
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1

    initial_inverse = np.concatenate(
        [quaternions[0, :1], -quaternions[0, 1:]]
    )
    angles = []
    for quaternion in quaternions:
        relative = quaternion_multiply(quaternion, initial_inverse)
        if relative[0] < 0:
            relative *= -1
        vector_norm = float(np.linalg.norm(relative[1:]))
        if vector_norm < 1e-12:
            rotation_vector = np.zeros(3, dtype=np.float64)
        else:
            angle = 2.0 * math.atan2(vector_norm, float(relative[0]))
            rotation_vector = relative[1:] / vector_norm * angle
        angles.append(float(np.dot(rotation_vector, ROTATION_AXIS_BASE)))
    return np.unwrap(np.asarray(angles, dtype=np.float64)) * 180.0 / math.pi


def detect_motion_segments(rotation_deg: np.ndarray) -> dict:
    frame_count = len(rotation_deg)
    if frame_count < 12:
        raise ValueError(f"Episode has only {frame_count} frames.")
    smoothed = np.convolve(
        np.pad(rotation_deg, (1, 1), mode="edge"),
        np.ones(3, dtype=np.float64) / 3.0,
        mode="valid",
    )
    platform_frames = max(3, min(7, frame_count // 8))
    initial_angle = float(np.median(smoothed[:platform_frames]))
    final_angle = float(np.median(smoothed[-platform_frames:]))
    turning_frame = int(np.argmin(smoothed))
    turning_angle = float(smoothed[turning_frame])
    left_magnitude = initial_angle - turning_angle
    rightward_travel = final_angle - turning_angle
    if not 5.0 <= left_magnitude <= 12.0:
        raise ValueError(
            f"Implausible left magnitude {left_magnitude:.3f} deg."
        )
    if not 12.0 <= rightward_travel <= 24.0:
        raise ValueError(
            f"Implausible rightward travel {rightward_travel:.3f} deg."
        )

    frame_indices = np.arange(frame_count)
    left_progress = (initial_angle - smoothed) / left_magnitude
    right_progress = (smoothed - turning_angle) / rightward_travel
    left_start = int(
        np.where(
            (frame_indices <= turning_frame)
            & (left_progress >= LOW_MOTION_FRACTION)
        )[0][0]
    )
    left_end = int(
        np.where(
            (frame_indices <= turning_frame)
            & (left_progress >= HIGH_MOTION_FRACTION)
        )[0][0]
    )
    right_start = int(
        np.where(
            (frame_indices >= turning_frame)
            & (right_progress >= LOW_MOTION_FRACTION)
        )[0][0]
    )
    right_end = int(
        np.where(
            (frame_indices >= turning_frame)
            & (right_progress >= HIGH_MOTION_FRACTION)
        )[0][0]
    )

    # Keep the physical turning point as a one-frame pause even when reversal
    # is immediate at 20 Hz.
    left_end = min(left_end, turning_frame - 1)
    right_start = max(right_start, turning_frame + 1)
    if not 0 < left_start <= left_end < right_start <= right_end < frame_count - 1:
        raise ValueError(
            "Invalid motion order: "
            f"left={left_start}-{left_end}, right={right_start}-{right_end}, "
            f"turn={turning_frame}, frames={frame_count}."
        )
    return {
        "smoothed_rotation_deg": smoothed,
        "left_start": left_start,
        "left_end": left_end,
        "pause_start": left_end + 1,
        "pause_end": right_start - 1,
        "right_start": right_start,
        "right_end": right_end,
        "turning_frame": turning_frame,
        "left_magnitude_deg": left_magnitude,
        "rightward_travel_deg": rightward_travel,
        "final_right_offset_deg": final_angle - initial_angle,
    }


def motion_segment(
    label: str,
    start: int,
    end: int,
    rotation_deg: np.ndarray,
    timestamps: np.ndarray,
    dataset_offset: int,
    seconds_per_frame: float,
) -> dict:
    frame_count = end - start + 1
    velocities = np.diff(rotation_deg[start : end + 1]) / seconds_per_frame
    if len(velocities) == 0:
        velocities = np.zeros(1, dtype=np.float64)
    return {
        "commanded_angle_delta_deg": float(rotation_deg[end] - rotation_deg[start]),
        "commanded_angular_speed_median_deg_s": float(np.median(velocities)),
        "commanded_angular_speed_peak_abs_deg_s": float(
            np.max(np.abs(velocities))
        ),
        "commanded_angular_speed_q75_abs_deg_s": float(
            np.percentile(np.abs(velocities), 75.0)
        ),
        "duration_s": float(frame_count * seconds_per_frame),
        "end_dataset_index_inclusive": dataset_offset + end,
        "end_frame_index_inclusive": end,
        "end_time_s": float(timestamps[start] + frame_count * seconds_per_frame),
        "frame_count": frame_count,
        "label": label,
        "start_dataset_index": dataset_offset + start,
        "start_frame_index": start,
        "start_time_s": float(timestamps[start]),
    }


def pause_segment(
    start: int,
    end: int,
    timestamps: np.ndarray,
    dataset_offset: int,
    seconds_per_frame: float,
) -> dict:
    frame_count = end - start + 1
    return {
        "duration_s": float(frame_count * seconds_per_frame),
        "end_dataset_index_inclusive": dataset_offset + end,
        "end_frame_index_inclusive": end,
        "end_time_s": float(timestamps[start] + frame_count * seconds_per_frame),
        "frame_count": frame_count,
        "label": "inter_direction_pause",
        "start_dataset_index": dataset_offset + start,
        "start_frame_index": start,
        "start_time_s": float(timestamps[start]),
    }


def annotate_environment(environment_root: Path) -> tuple[list[dict], dict]:
    episodes = read_jsonl(environment_root / "meta" / "episodes.jsonl")
    augmentation = json.loads(
        (environment_root / "augmentation_metadata.json").read_text(
            encoding="utf-8"
        )
    )["episodes"]
    conversion_rows = json.loads(
        (environment_root / "conversion_metadata.json").read_text(
            encoding="utf-8"
        )
    )["episodes"]
    conversion = {
        int(row["output_episode_index"]): row for row in conversion_rows
    }
    expected_indices = {int(row["episode_index"]) for row in episodes}
    if set(conversion) != expected_indices:
        raise ValueError(
            f"{environment_root.name}: conversion metadata episode mismatch."
        )
    if {int(key) for key in augmentation} != expected_indices:
        raise ValueError(
            f"{environment_root.name}: augmentation metadata episode mismatch."
        )

    annotations = []
    dataset_offset = 0
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        episode_index = int(episode["episode_index"])
        expected_length = int(episode["length"])
        parquet_path = (
            environment_root
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(
            parquet_path,
            columns=["observation.eef_state", "timestamp"],
        )
        payload = table.to_pydict()
        eef_state = np.asarray(payload["observation.eef_state"], dtype=np.float64)
        timestamps = np.asarray(payload["timestamp"], dtype=np.float64)
        if len(eef_state) != expected_length or len(timestamps) != expected_length:
            raise ValueError(
                f"{parquet_path}: expected {expected_length} rows, got "
                f"eef={len(eef_state)} timestamp={len(timestamps)}."
            )
        time_differences = np.diff(timestamps)
        seconds_per_frame = (
            float(np.median(time_differences))
            if len(time_differences)
            else 0.05
        )
        if not 0.045 <= seconds_per_frame <= 0.055:
            raise ValueError(
                f"{parquet_path}: unexpected frame interval {seconds_per_frame}."
            )

        rotation_deg = base_axis_rotation_deg(eef_state)
        detected = detect_motion_segments(rotation_deg)
        smoothed = detected["smoothed_rotation_deg"]
        left = motion_segment(
            "left_motion",
            detected["left_start"],
            detected["left_end"],
            smoothed,
            timestamps,
            dataset_offset,
            seconds_per_frame,
        )
        pause = pause_segment(
            detected["pause_start"],
            detected["pause_end"],
            timestamps,
            dataset_offset,
            seconds_per_frame,
        )
        right = motion_segment(
            "right_motion",
            detected["right_start"],
            detected["right_end"],
            smoothed,
            timestamps,
            dataset_offset,
            seconds_per_frame,
        )
        active_speeds = np.concatenate(
            [
                np.abs(
                    np.diff(
                        smoothed[
                            detected["left_start"] : detected["left_end"] + 1
                        ]
                    )
                    / seconds_per_frame
                ),
                np.abs(
                    np.diff(
                        smoothed[
                            detected["right_start"] : detected["right_end"] + 1
                        ]
                    )
                    / seconds_per_frame
                ),
            ]
        )
        detected_speed = float(np.percentile(active_speeds, 75.0))
        augmentation_row = augmentation[f"{episode_index:06d}"]
        skill_gear = int(augmentation_row["skill_level"])
        if not 1 <= skill_gear <= 10:
            raise ValueError(
                f"{environment_root.name} episode {episode_index}: "
                f"invalid skill level {skill_gear}."
            )
        expected_speed = EXPECTED_FULL_SPEED_DEG_S * skill_gear / 10.0
        source = conversion[episode_index]
        before_end = detected["left_start"] - 1
        after_start = detected["right_end"] + 1
        annotation = {
            "annotation_schema_version": SCHEMA_VERSION,
            "context_ranges": {
                "after_skill": {
                    "end_dataset_index_inclusive": dataset_offset
                    + expected_length
                    - 1,
                    "end_frame_index_inclusive": expected_length - 1,
                    "start_dataset_index": dataset_offset + after_start,
                    "start_frame_index": after_start,
                },
                "before_skill": {
                    "end_dataset_index_inclusive": dataset_offset + before_end,
                    "end_frame_index_inclusive": before_end,
                    "start_dataset_index": dataset_offset,
                    "start_frame_index": 0,
                },
            },
            "detected_commanded_angular_speed_q75_deg_s": detected_speed,
            "detection_status": "unique_match",
            "episode_index": episode_index,
            "expected_commanded_angular_speed_deg_s": expected_speed,
            "gear_fit_relative_error": abs(detected_speed - expected_speed)
            / expected_speed,
            "rotation_axis_base": ROTATION_AXIS_BASE.tolist(),
            "sampled_geometry": {
                "final_right_offset_deg": detected["final_right_offset_deg"],
                "inter_direction_pause_s": pause["duration_s"],
                "left_magnitude_deg": detected["left_magnitude_deg"],
                "rightward_travel_deg": detected["rightward_travel_deg"],
            },
            "segments": [left, pause, right],
            "skill_gear": skill_gear,
            "skill_key": str(skill_gear),
            "skill_span": {
                "end_dataset_index_inclusive": dataset_offset
                + detected["right_end"],
                "end_frame_index_inclusive": detected["right_end"],
                "end_time_s": right["end_time_s"],
                "start_dataset_index": dataset_offset + detected["left_start"],
                "start_frame_index": detected["left_start"],
                "start_time_s": left["start_time_s"],
            },
            "skill_speed_scale": skill_gear / 10.0,
            "source_directory": source["source_directory"],
            "source_episode_index": int(source["source_episode_index"]),
            "level_annotation_source": augmentation_row.get("skill_source"),
            "motion_segmentation_source": (
                "measured_eef_base_axis_rotation_1pct_99pct"
            ),
        }
        annotations.append(annotation)
        dataset_offset += expected_length

    level_counts = Counter(row["skill_gear"] for row in annotations)
    if len(level_counts) < 6:
        raise ValueError(
            f"{environment_root.name}: only {len(level_counts)} skill levels."
        )
    return annotations, {
        "environment": environment_root.name,
        "episodes": len(annotations),
        "frames": dataset_offset,
        "level_counts": {
            str(key): value for key, value in sorted(level_counts.items())
        },
    }


def validate_all(
    all_annotations: dict[str, list[dict]], summaries: list[dict]
) -> dict:
    rows = [row for values in all_annotations.values() for row in values]
    if len(rows) != 314:
        raise ValueError(f"Expected 314 annotations, got {len(rows)}.")
    global_levels = Counter(row["skill_gear"] for row in rows)
    if set(global_levels) != set(range(1, 11)):
        raise ValueError(f"Expected global levels 1-10, got {global_levels}.")

    left_magnitudes = []
    rightward_travels = []
    pause_frames = []
    before_frames = []
    after_frames = []
    speeds_by_level: dict[int, list[float]] = defaultdict(list)
    for environment, values in all_annotations.items():
        seen = set()
        for row in values:
            episode_index = int(row["episode_index"])
            if episode_index in seen:
                raise ValueError(
                    f"{environment}: duplicate episode {episode_index}."
                )
            seen.add(episode_index)
            segments = {segment["label"]: segment for segment in row["segments"]}
            left = segments["left_motion"]
            pause = segments["inter_direction_pause"]
            right = segments["right_motion"]
            if not (
                left["start_frame_index"]
                <= left["end_frame_index_inclusive"]
                < pause["start_frame_index"]
                <= pause["end_frame_index_inclusive"]
                < right["start_frame_index"]
                <= right["end_frame_index_inclusive"]
            ):
                raise ValueError(
                    f"{environment} episode {episode_index}: invalid segment order."
                )
            left_magnitudes.append(row["sampled_geometry"]["left_magnitude_deg"])
            rightward_travels.append(
                row["sampled_geometry"]["rightward_travel_deg"]
            )
            pause_frames.append(pause["frame_count"])
            before = row["context_ranges"]["before_skill"]
            after = row["context_ranges"]["after_skill"]
            before_frames.append(
                before["end_frame_index_inclusive"]
                - before["start_frame_index"]
                + 1
            )
            after_frames.append(
                after["end_frame_index_inclusive"]
                - after["start_frame_index"]
                + 1
            )
            speeds_by_level[int(row["skill_gear"])].append(
                float(row["detected_commanded_angular_speed_q75_deg_s"])
            )

    levels = []
    speeds = []
    speed_summary = {}
    for level in range(1, 11):
        values = np.asarray(speeds_by_level[level], dtype=np.float64)
        levels.extend([level] * len(values))
        speeds.extend(values.tolist())
        speed_summary[str(level)] = {
            "count": len(values),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    speed_level_correlation = float(np.corrcoef(levels, speeds)[0, 1])
    if speed_level_correlation < 0.65:
        raise ValueError(
            f"Weak level/speed correlation: {speed_level_correlation:.4f}."
        )

    def quantiles(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "p10": float(np.percentile(array, 10.0)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90.0)),
            "max": float(array.max()),
        }

    return {
        "annotation_schema_version": SCHEMA_VERSION,
        "environments": summaries,
        "episodes": len(rows),
        "global_level_counts": {
            str(key): value for key, value in sorted(global_levels.items())
        },
        "left_magnitude_deg": quantiles(left_magnitudes),
        "rightward_travel_deg": quantiles(rightward_travels),
        "pause_frames": quantiles(pause_frames),
        "before_skill_frames": quantiles(before_frames),
        "after_skill_frames": quantiles(after_frames),
        "detected_speed_by_level_deg_s": speed_summary,
        "skill_level_detected_speed_correlation": speed_level_correlation,
        "level_source": "augmentation_metadata.json:episodes.*.skill_level",
        "motion_source": "observation.eef_state quaternion about base axis [-1,0,0]",
        "motion_thresholds": {
            "start_fraction": LOW_MOTION_FRACTION,
            "end_fraction": HIGH_MOTION_FRACTION,
        },
        "validation": "passed",
    }


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    all_annotations = {}
    summaries = []
    for environment in ENVIRONMENTS:
        environment_root = root / environment
        if not environment_root.is_dir():
            raise FileNotFoundError(environment_root)
        annotations, summary = annotate_environment(environment_root)
        all_annotations[environment] = annotations
        summaries.append(summary)
    validation_summary = validate_all(all_annotations, summaries)

    if args.write:
        for environment, annotations in all_annotations.items():
            meta_root = root / environment / "meta"
            atomic_write_jsonl(meta_root / ANNOTATION_NAME, annotations)
            environment_summary = next(
                item
                for item in summaries
                if item["environment"] == environment
            )
            atomic_write_json(meta_root / SUMMARY_NAME, environment_summary)
        atomic_write_json(root / SUMMARY_NAME, validation_summary)
    print(json.dumps(validation_summary, indent=2, sort_keys=True))
    print(f"[result] write={args.write} root={root}")


if __name__ == "__main__":
    main()
