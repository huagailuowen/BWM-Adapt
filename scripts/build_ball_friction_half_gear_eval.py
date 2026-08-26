#!/usr/bin/env python3
"""Build rigorously time-warped half-gear conditions for ball-friction evaluation.

The source command is an absolute joint-position trajectory, not a velocity vector.
Consequently, multiplying the action values would change robot geometry instead of
speed.  This script keeps one recorded geometric trajectory and continuously
resamples its rightward swing in time.  Observation state and action are warped by
the same phase map so the model never receives contradictory future conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TEMPORAL_VECTOR_COLUMNS = ("observation.state", "observation.eef_state", "action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--query-metadata", type=Path, required=True)
    parser.add_argument("--support-metadata", type=Path, required=True)
    parser.add_argument("--selection-tsv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--audit-plot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--support-gear", type=int, default=6)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def choose_impact_row(rows: list[dict], group: int, gear: int, target_ratio: float) -> dict:
    candidates = [
        row
        for row in rows
        if row.get("sampling_kind") == "impact"
        and int(row["environment_index"]) == group
        and int(row["skill_gear"]) == gear
    ]
    if not candidates:
        raise RuntimeError(f"No impact candidate for environment_index={group}, gear={gear}.")

    def score(row: dict) -> tuple:
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        right_start = int(row["right_motion_start"])
        right_end = int(row["right_motion_end"])
        edited_right_end = right_start + (right_end - right_start) / target_ratio
        pre_frames = right_start - start
        post_frames = end - edited_right_end
        return (
            pre_frames < 4,
            post_frames < 12,
            abs(pre_frames - 8),
            -post_frames,
            int(row["source_episode_index"]),
            int(row.get("candidate_index", 0)),
        )

    selected = min(candidates, key=score)
    if int(selected["start_frame"]) >= int(selected["right_motion_start"]):
        raise RuntimeError(f"Selected window does not preserve a pre-swing first frame: {selected['sample_id']}")
    return dict(selected)


def build_time_map(length: int, right_start: int, right_end: int, speed_ratio: float) -> tuple[np.ndarray, float]:
    if not 0 <= right_start < right_end < length:
        raise ValueError(
            f"Invalid rightward interval [{right_start}, {right_end}] for trajectory length {length}."
        )
    if speed_ratio <= 0:
        raise ValueError(f"speed_ratio must be positive, got {speed_ratio}.")

    target_end = right_start + (right_end - right_start) / speed_ratio
    output_time = np.arange(length, dtype=np.float64)
    source_time = output_time.copy()
    active = (output_time > right_start) & (output_time <= target_end)
    source_time[active] = right_start + (output_time[active] - right_start) * speed_ratio
    after = output_time > target_end
    source_time[after] = right_end + (output_time[after] - target_end)
    np.clip(source_time, 0.0, float(length - 1), out=source_time)
    return source_time, target_end


def interpolate_vectors(values: np.ndarray, source_time: np.ndarray) -> np.ndarray:
    source_axis = np.arange(values.shape[0], dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    for dimension in range(values.shape[1]):
        result[:, dimension] = np.interp(source_time, source_axis, values[:, dimension])
    return result


def interpolate_eef(values: np.ndarray, source_time: np.ndarray) -> np.ndarray:
    result = interpolate_vectors(values, source_time)
    if values.shape[1] < 7:
        return result

    source_quaternions = values[:, 3:7].astype(np.float64, copy=True)
    source_quaternions /= np.maximum(np.linalg.norm(source_quaternions, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(source_quaternions)):
        if np.dot(source_quaternions[index - 1], source_quaternions[index]) < 0:
            source_quaternions[index] *= -1
    result[:, 3:7] = interpolate_vectors(source_quaternions, source_time)
    result[:, 3:7] /= np.maximum(np.linalg.norm(result[:, 3:7], axis=1, keepdims=True), 1e-12)
    return result


def quaternion_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def replace_vector_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(f"Missing required temporal column {name!r}.")
    field = table.schema.field(column_index)
    replacement = pa.array(values.astype(np.float32).tolist(), type=field.type)
    return table.set_column(column_index, field, replacement)


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def make_audit_plot(rows: list[dict], output_path: Path) -> None:
    groups = sorted({int(row["environment_index"]) for row in rows})
    palette = ("#30123b", "#3957d0", "#20a4f3", "#34c759", "#f4d03f", "#f47c20", "#b40426")
    colors = {group: palette[index % len(palette)] for index, group in enumerate(groups)}
    width, height = 1200, 900
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f5"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.axis{font-size:13px}.tick{font-size:11px;fill:#5f6368}.legend{font-size:12px}.grid{stroke:#d9d7d0;stroke-width:1}.frame{stroke:#333;stroke-width:1.4;fill:white}</style>',
    ]

    def panel(
        left: float,
        top: float,
        title: str,
        x_label: str,
        y_label: str,
        x_key: str,
        y_key: str,
        equality: bool = False,
        legend: bool = False,
    ) -> None:
        panel_width, panel_height = 500.0, 335.0
        plot_left, plot_top = left + 72.0, top + 42.0
        plot_width, plot_height = panel_width - 92.0, panel_height - 94.0
        x_values = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
        y_values = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
        x_min, x_max = float(x_values.min()), float(x_values.max())
        y_min, y_max = float(y_values.min()), float(y_values.max())
        if equality:
            lower = min(x_min, y_min)
            upper = max(x_max, y_max)
            x_min = y_min = lower
            x_max = y_max = upper
        x_pad = max((x_max - x_min) * 0.06, 1e-6)
        y_pad = max((y_max - y_min) * 0.08, 1e-6)
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

        def project(x_value: float, y_value: float) -> tuple[float, float]:
            x_pixel = plot_left + (x_value - x_min) / (x_max - x_min) * plot_width
            y_pixel = plot_top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height
            return x_pixel, y_pixel

        elements.append(f'<rect class="frame" x="{plot_left:.1f}" y="{plot_top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}"/>')
        elements.append(f'<text class="title" x="{left + panel_width / 2:.1f}" y="{top + 23:.1f}" text-anchor="middle">{title}</text>')
        for tick_index in range(6):
            fraction = tick_index / 5.0
            x_value = x_min + fraction * (x_max - x_min)
            y_value = y_min + fraction * (y_max - y_min)
            x_pixel = plot_left + fraction * plot_width
            y_pixel = plot_top + plot_height - fraction * plot_height
            elements.append(f'<line class="grid" x1="{x_pixel:.1f}" y1="{plot_top:.1f}" x2="{x_pixel:.1f}" y2="{plot_top + plot_height:.1f}"/>')
            elements.append(f'<line class="grid" x1="{plot_left:.1f}" y1="{y_pixel:.1f}" x2="{plot_left + plot_width:.1f}" y2="{y_pixel:.1f}"/>')
            elements.append(f'<text class="tick" x="{x_pixel:.1f}" y="{plot_top + plot_height + 18:.1f}" text-anchor="middle">{x_value:.2g}</text>')
            elements.append(f'<text class="tick" x="{plot_left - 8:.1f}" y="{y_pixel + 4:.1f}" text-anchor="end">{y_value:.2g}</text>')
        elements.append(f'<text class="axis" x="{plot_left + plot_width / 2:.1f}" y="{top + panel_height - 10:.1f}" text-anchor="middle">{x_label}</text>')
        elements.append(f'<text class="axis" x="{left + 16:.1f}" y="{plot_top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {left + 16:.1f} {plot_top + plot_height / 2:.1f})">{y_label}</text>')
        if equality:
            start = project(max(x_min, y_min), max(x_min, y_min))
            end = project(min(x_max, y_max), min(x_max, y_max))
            elements.append(f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#111" stroke-width="1.5" stroke-dasharray="7 5"/>')

        for group in groups:
            selected = sorted(
                (row for row in rows if int(row["environment_index"]) == group),
                key=lambda row: float(row[x_key]),
            )
            points = [project(float(row[x_key]), float(row[y_key])) for row in selected]
            point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            elements.append(f'<polyline points="{point_string}" fill="none" stroke="{colors[group]}" stroke-width="2" opacity="0.8"/>')
            for x_pixel, y_pixel in points:
                elements.append(f'<circle cx="{x_pixel:.1f}" cy="{y_pixel:.1f}" r="3.4" fill="{colors[group]}" stroke="white" stroke-width="0.8"/>')
            if legend:
                legend_index = groups.index(group)
                legend_x = plot_left + 8 + (legend_index % 4) * 98
                legend_y = plot_top + 17 + (legend_index // 4) * 18
                ball_id = selected[0]["ball_id"]
                elements.append(f'<circle cx="{legend_x:.1f}" cy="{legend_y - 4:.1f}" r="4" fill="{colors[group]}"/>')
                elements.append(f'<text class="legend" x="{legend_x + 8:.1f}" y="{legend_y:.1f}">Ball {ball_id}</text>')

    panel(35, 28, "Nominal midpoint speed", "Edited gear", "deg/s", "target_gear", "target_nominal_speed_deg_s", legend=True)
    panel(635, 28, "Recorded-path effective speed", "Nominal target deg/s", "EEF path deg/s", "target_nominal_speed_deg_s", "effective_measured_speed_deg_s", equality=True)
    panel(35, 455, "Action continuity audit", "Edited gear", "max step: edited / source", "target_gear", "edited_to_source_max_action_step_ratio")
    panel(635, 455, "Post-impact temporal coverage", "Edited gear", "frames after edited swing", "target_gear", "post_frames_after_edited_swing")
    elements.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source_metadata)
    impact_rows = [row for row in rows if row.get("sampling_kind") == "impact"]
    groups = sorted({int(row["environment_index"]) for row in impact_rows})
    if not groups:
        raise RuntimeError("Source metadata contains no impact rows.")

    speeds_by_gear: dict[int, float] = {}
    for gear in range(1, 11):
        values = {
            round(float(row["expected_commanded_angular_speed_deg_s"]), 9)
            for row in impact_rows
            if int(row["skill_gear"]) == gear
        }
        if len(values) != 1:
            raise RuntimeError(f"Expected one nominal speed for gear {gear}, found {sorted(values)}.")
        speeds_by_gear[gear] = values.pop()

    args.synthetic_root.mkdir(parents=True, exist_ok=True)
    support_rows: list[dict] = []
    query_rows: list[dict] = []
    audit_rows: list[dict] = []
    selection_rows: list[dict] = []

    for group in groups:
        support = choose_impact_row(impact_rows, group, args.support_gear, target_ratio=1.0)
        support["valid_frames"] = int(support.get("valid_physical_frames", 60))
        support["half_gear_support"] = True
        support_rows.append(support)

        for lower_gear in range(1, 10):
            lower_speed = speeds_by_gear[lower_gear]
            upper_speed = speeds_by_gear[lower_gear + 1]
            target_speed = 0.5 * (lower_speed + upper_speed)
            speed_ratio = target_speed / lower_speed
            source = choose_impact_row(impact_rows, group, lower_gear, target_ratio=speed_ratio)
            source_path = args.dataset_root / source["action"]
            table = pq.read_table(source_path)
            right_start = int(source["right_motion_start"])
            right_end = int(source["right_motion_end"])
            source_time, target_end = build_time_map(table.num_rows, right_start, right_end, speed_ratio)

            vectors: dict[str, np.ndarray] = {}
            edited: dict[str, np.ndarray] = {}
            for name in TEMPORAL_VECTOR_COLUMNS:
                vectors[name] = np.asarray(table[name].to_pylist(), dtype=np.float64)
                edited[name] = (
                    interpolate_eef(vectors[name], source_time)
                    if name == "observation.eef_state"
                    else interpolate_vectors(vectors[name], source_time)
                )

            start_frame = int(source["start_frame"])
            end_frame = int(source["end_frame"])
            if not np.array_equal(edited["action"][start_frame], vectors["action"][start_frame]):
                raise RuntimeError(f"Edited action changed the GT conditioning frame for {source['sample_id']}.")
            if not np.array_equal(edited["observation.state"][start_frame], vectors["observation.state"][start_frame]):
                raise RuntimeError(f"Edited state changed the GT conditioning frame for {source['sample_id']}.")
            if not np.allclose(edited["action"][-1], vectors["action"][-1], atol=1e-7, rtol=0.0):
                raise RuntimeError(f"Edited action does not settle at the recorded endpoint for {source['sample_id']}.")
            if not np.isfinite(np.concatenate(tuple(edited.values()), axis=1)).all():
                raise RuntimeError(f"Non-finite value produced for {source['sample_id']}.")

            output_table = table
            for name in TEMPORAL_VECTOR_COLUMNS:
                output_table = replace_vector_column(output_table, name, edited[name])

            target_gear = lower_gear + 0.5
            ball_id = int(source["ball_id"])
            synthetic_path = args.synthetic_root / f"ball{ball_id}_gear{lower_gear}_to_{target_gear:.1f}_ep{int(source['source_episode_index']):06d}.parquet"
            pq.write_table(output_table, synthetic_path, compression="zstd")

            query = dict(source)
            query.update(
                {
                    "action": relative_or_absolute(synthetic_path, args.dataset_root),
                    "source_action": source["action"],
                    "source_video": source["video"],
                    "sample_id": f"halfgear:ball{ball_id}:source{lower_gear}:target{target_gear:.1f}:ep{int(source['source_episode_index']):06d}",
                    "action_id": target_gear,
                    "skill_gear": target_gear,
                    "skill_speed_scale": None,
                    "expected_commanded_angular_speed_deg_s": target_speed,
                    "source_skill_gear": lower_gear,
                    "counterfactual_skill_gear": target_gear,
                    "counterfactual_speed_ratio": speed_ratio,
                    "counterfactual_right_motion_end_float": target_end,
                    "right_motion_end": int(round(target_end)),
                    "right_motion_frames": int(round(target_end)) - right_start + 1,
                    "window_right_motion_frames": max(
                        0,
                        min(end_frame, int(round(target_end))) - max(start_frame, right_start) + 1,
                    ),
                    "counterfactual_action_edit": "same_recorded_path_continuous_timewarp_v1",
                    "counterfactual_gt_role": "source_lower_gear_reference_and_initial_frame",
                    "valid_frames": int(source.get("valid_physical_frames", 60)),
                }
            )
            query_index = len(query_rows)
            query_rows.append(query)

            eef_angle = quaternion_distance_degrees(
                vectors["observation.eef_state"][right_start, 3:7],
                vectors["observation.eef_state"][right_end, 3:7],
            )
            source_duration_seconds = (right_end - right_start) / args.fps
            edited_duration_seconds = (target_end - right_start) / args.fps
            measured_source_speed = eef_angle / source_duration_seconds
            effective_measured_speed = eef_angle / edited_duration_seconds
            source_action_step = np.linalg.norm(np.diff(vectors["action"][:, :7], axis=0), axis=1)
            edited_action_step = np.linalg.norm(np.diff(edited["action"][:, :7], axis=0), axis=1)
            source_max_step = float(source_action_step.max())
            edited_max_step = float(edited_action_step.max())
            if edited_max_step > source_max_step * speed_ratio * 1.30 + 1e-7:
                raise RuntimeError(
                    f"Unexpected action discontinuity for {query['sample_id']}: edited max step "
                    f"{edited_max_step:.6g} exceeds bounded time-warp expectation."
                )

            audit = {
                "query_index": query_index,
                "environment_index": group,
                "ball_id": ball_id,
                "source_sample_id": source["sample_id"],
                "source_episode_index": int(source["source_episode_index"]),
                "source_gear": lower_gear,
                "target_gear": target_gear,
                "source_nominal_speed_deg_s": lower_speed,
                "upper_nominal_speed_deg_s": upper_speed,
                "target_nominal_speed_deg_s": target_speed,
                "speed_ratio": speed_ratio,
                "eef_rightward_angle_deg": eef_angle,
                "source_measured_speed_deg_s": measured_source_speed,
                "effective_measured_speed_deg_s": effective_measured_speed,
                "source_right_motion_start": right_start,
                "source_right_motion_end": right_end,
                "edited_right_motion_end_float": target_end,
                "query_start_frame": start_frame,
                "query_end_frame": end_frame,
                "pre_swing_frames": right_start - start_frame,
                "post_frames_after_edited_swing": end_frame - target_end,
                "source_max_action_step_l2": source_max_step,
                "edited_max_action_step_l2": edited_max_step,
                "edited_to_source_max_action_step_ratio": edited_max_step / max(source_max_step, 1e-12),
                "conditioning_action_exact": True,
                "conditioning_state_exact": True,
                "settled_endpoint_preserved": True,
                "synthetic_action": query["action"],
            }
            audit_rows.append(audit)
            selection_rows.append(
                {
                    "query_index": query_index,
                    "environment_index": group,
                    "ball_id": ball_id,
                    "source_gear": lower_gear,
                    "target_gear": target_gear,
                    "source_episode_index": int(source["source_episode_index"]),
                    "source_start_frame": start_frame,
                    "source_end_frame": end_frame,
                    "support_gear": args.support_gear,
                    "support_episode_index": int(support["source_episode_index"]),
                    "support_start_frame": int(support["start_frame"]),
                    "support_end_frame": int(support["end_frame"]),
                }
            )

    write_jsonl(args.query_metadata, query_rows)
    write_jsonl(args.support_metadata, support_rows)
    args.selection_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.selection_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selection_rows)
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    make_audit_plot(audit_rows, args.audit_plot)

    manifest = {
        "action_edit": "same_recorded_path_continuous_timewarp_v1",
        "source_metadata": str(args.source_metadata.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "groups": groups,
        "support_gear": args.support_gear,
        "support_count": len(support_rows),
        "query_count": len(query_rows),
        "target_gears": [gear + 0.5 for gear in range(1, 10)],
        "nominal_speeds_deg_s": speeds_by_gear,
        "fps": args.fps,
        "invariants": {
            "same_recorded_geometric_path": True,
            "state_and_action_share_time_map": True,
            "gt_conditioning_frame_unchanged": True,
            "recorded_endpoint_preserved": True,
            "full_action_vector_not_scaled": True,
            "unrelated_episodes_not_interpolated": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"[halfgear_ready] supports={len(support_rows)} queries={len(query_rows)} "
        f"groups={groups} audit={args.audit_csv}"
    )


if __name__ == "__main__":
    main()
