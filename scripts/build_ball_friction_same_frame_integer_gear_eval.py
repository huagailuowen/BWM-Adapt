#!/usr/bin/env python3
"""Build same-initial-frame, integer-gear counterfactual ball actions.

For each ball, one real Level-6 impact chunk supplies the support video, first
frame, and geometric robot trajectory.  Ten query actions preserve that frame
and path while continuously resampling the rightward swing to Levels 1--10.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from build_ball_friction_half_gear_eval import (
    TEMPORAL_VECTOR_COLUMNS,
    build_time_map,
    interpolate_eef,
    interpolate_vectors,
    make_audit_plot,
    quaternion_distance_degrees,
    read_jsonl,
    relative_or_absolute,
    replace_vector_column,
    write_jsonl,
)


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
    parser.add_argument("--minimum-post-frames", type=float, default=4.0)
    return parser.parse_args()


def choose_shared_support(
    rows: list[dict],
    group: int,
    support_gear: int,
    slowest_ratio: float,
    minimum_post_frames: float,
) -> dict:
    candidates = [
        dict(row)
        for row in rows
        if row.get("sampling_kind") == "impact"
        and int(row["environment_index"]) == group
        and int(row["skill_gear"]) == support_gear
    ]
    if not candidates:
        raise RuntimeError(f"No Level-{support_gear} impact support for environment_index={group}.")

    def measurements(row: dict) -> tuple[float, float]:
        right_start = int(row["right_motion_start"])
        right_end = int(row["right_motion_end"])
        edited_end = right_start + (right_end - right_start) / slowest_ratio
        return right_start - int(row["start_frame"]), int(row["end_frame"]) - edited_end

    valid = []
    for row in candidates:
        pre_frames, post_frames = measurements(row)
        if pre_frames >= 4 and post_frames >= minimum_post_frames:
            valid.append(row)
    if not valid:
        best = max(candidates, key=lambda row: measurements(row)[1])
        pre_frames, post_frames = measurements(best)
        raise RuntimeError(
            f"No shared support window fits the slowest target for group={group}; "
            f"best pre={pre_frames:.2f}, post={post_frames:.2f}."
        )

    def score(row: dict) -> tuple:
        pre_frames, post_frames = measurements(row)
        return (
            abs(pre_frames - 4.0),
            -post_frames,
            int(row["source_episode_index"]),
            int(row.get("candidate_index", 0)),
        )

    return min(valid, key=score)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source_metadata)
    impact_rows = [row for row in rows if row.get("sampling_kind") == "impact"]
    groups = sorted({int(row["environment_index"]) for row in impact_rows})

    speeds_by_gear: dict[int, float] = {}
    for gear in range(1, 11):
        values = {
            round(float(row["expected_commanded_angular_speed_deg_s"]), 9)
            for row in impact_rows
            if int(row["skill_gear"]) == gear
        }
        if len(values) != 1:
            raise RuntimeError(f"Expected one nominal speed for Level {gear}, found {sorted(values)}.")
        speeds_by_gear[gear] = values.pop()

    source_speed = speeds_by_gear[args.support_gear]
    slowest_ratio = speeds_by_gear[1] / source_speed
    args.synthetic_root.mkdir(parents=True, exist_ok=True)
    support_rows: list[dict] = []
    query_rows: list[dict] = []
    audit_rows: list[dict] = []
    selection_rows: list[dict] = []

    for group in groups:
        support = choose_shared_support(
            impact_rows,
            group,
            args.support_gear,
            slowest_ratio,
            args.minimum_post_frames,
        )
        support["valid_frames"] = int(support.get("valid_physical_frames", 60))
        support["same_frame_integer_gear_support"] = True
        support_rows.append(support)

        source_path = args.dataset_root / support["action"]
        table = pq.read_table(source_path)
        right_start = int(support["right_motion_start"])
        right_end = int(support["right_motion_end"])
        start_frame = int(support["start_frame"])
        end_frame = int(support["end_frame"])
        vectors = {
            name: np.asarray(table[name].to_pylist(), dtype=np.float64)
            for name in TEMPORAL_VECTOR_COLUMNS
        }
        source_action_step = np.linalg.norm(np.diff(vectors["action"][:, :7], axis=0), axis=1)
        source_max_step = float(source_action_step.max())
        eef_angle = quaternion_distance_degrees(
            vectors["observation.eef_state"][right_start, 3:7],
            vectors["observation.eef_state"][right_end, 3:7],
        )
        source_duration_seconds = (right_end - right_start) / args.fps
        measured_source_speed = eef_angle / source_duration_seconds

        group_first_action = vectors["action"][start_frame].copy()
        group_first_state = vectors["observation.state"][start_frame].copy()
        group_video = json.dumps(support["video"], sort_keys=True)

        for target_gear in range(1, 11):
            target_speed = speeds_by_gear[target_gear]
            speed_ratio = target_speed / source_speed
            source_time, target_end = build_time_map(table.num_rows, right_start, right_end, speed_ratio)
            if target_end > end_frame - args.minimum_post_frames:
                raise RuntimeError(
                    f"Level {target_gear} does not finish inside the shared query window for "
                    f"group={group}: edited_end={target_end:.3f}, query_end={end_frame}."
                )

            edited = {}
            for name in TEMPORAL_VECTOR_COLUMNS:
                edited[name] = (
                    interpolate_eef(vectors[name], source_time)
                    if name == "observation.eef_state"
                    else interpolate_vectors(vectors[name], source_time)
                )
            if not np.array_equal(edited["action"][start_frame], group_first_action):
                raise RuntimeError(f"Initial action changed for group={group}, target Level {target_gear}.")
            if not np.array_equal(edited["observation.state"][start_frame], group_first_state):
                raise RuntimeError(f"Initial state changed for group={group}, target Level {target_gear}.")
            if not np.isfinite(np.concatenate(tuple(edited.values()), axis=1)).all():
                raise RuntimeError(f"Non-finite condition for group={group}, target Level {target_gear}.")

            output_table = table
            for name in TEMPORAL_VECTOR_COLUMNS:
                output_table = replace_vector_column(output_table, name, edited[name])
            ball_id = int(support["ball_id"])
            synthetic_path = args.synthetic_root / (
                f"ball{ball_id}_sameframe_supportL{args.support_gear}_targetL{target_gear}_"
                f"ep{int(support['source_episode_index']):06d}.parquet"
            )
            pq.write_table(output_table, synthetic_path, compression="zstd")

            query = dict(support)
            query.update(
                {
                    "action": relative_or_absolute(synthetic_path, args.dataset_root),
                    "source_action": support["action"],
                    "source_video": support["video"],
                    "sample_id": (
                        f"sameframe:ball{ball_id}:supportL{args.support_gear}:"
                        f"targetL{target_gear}:ep{int(support['source_episode_index']):06d}"
                    ),
                    "action_id": target_gear,
                    "skill_gear": target_gear,
                    "skill_speed_scale": None,
                    "expected_commanded_angular_speed_deg_s": target_speed,
                    "source_skill_gear": args.support_gear,
                    "counterfactual_skill_gear": target_gear,
                    "counterfactual_speed_ratio": speed_ratio,
                    "counterfactual_right_motion_end_float": target_end,
                    "right_motion_end": int(round(target_end)),
                    "right_motion_frames": int(round(target_end)) - right_start + 1,
                    "window_right_motion_frames": max(
                        0,
                        min(end_frame, int(round(target_end))) - max(start_frame, right_start) + 1,
                    ),
                    "counterfactual_action_edit": "shared_support_path_integer_gear_timewarp_v1",
                    "counterfactual_same_initial_frame": True,
                    "counterfactual_gt_role": "same_level6_support_reference_and_initial_frame",
                    "valid_frames": int(support.get("valid_physical_frames", 60)),
                }
            )
            if json.dumps(query["video"], sort_keys=True) != group_video or int(query["start_frame"]) != start_frame:
                raise RuntimeError(f"Visual conditioning changed inside group={group}.")
            query_index = len(query_rows)
            query_rows.append(query)

            edited_action_step = np.linalg.norm(np.diff(edited["action"][:, :7], axis=0), axis=1)
            edited_max_step = float(edited_action_step.max())
            continuity_limit = source_max_step * max(1.0, speed_ratio) * 1.35 + 1e-7
            if edited_max_step > continuity_limit:
                raise RuntimeError(
                    f"Unexpected action discontinuity for group={group}, target Level {target_gear}: "
                    f"{edited_max_step:.6g} > {continuity_limit:.6g}."
                )
            effective_measured_speed = measured_source_speed * speed_ratio
            audit_rows.append(
                {
                    "query_index": query_index,
                    "environment_index": group,
                    "ball_id": ball_id,
                    "source_sample_id": support["sample_id"],
                    "source_episode_index": int(support["source_episode_index"]),
                    "source_gear": args.support_gear,
                    "target_gear": target_gear,
                    "source_nominal_speed_deg_s": source_speed,
                    "upper_nominal_speed_deg_s": target_speed,
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
                    "same_video_and_start_frame": True,
                    "synthetic_action": query["action"],
                }
            )
            selection_rows.append(
                {
                    "query_index": query_index,
                    "environment_index": group,
                    "ball_id": ball_id,
                    "target_gear": target_gear,
                    "source_gear": args.support_gear,
                    "support_episode_index": int(support["source_episode_index"]),
                    "support_start_frame": start_frame,
                    "support_end_frame": end_frame,
                    "edited_right_motion_end": f"{target_end:.6f}",
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
        "action_edit": "shared_support_path_integer_gear_timewarp_v1",
        "groups": groups,
        "support_gear": args.support_gear,
        "support_count": len(support_rows),
        "query_count": len(query_rows),
        "target_gears": list(range(1, 11)),
        "nominal_speeds_deg_s": speeds_by_gear,
        "same_initial_frame_within_environment": True,
        "same_source_episode_within_environment": True,
        "state_and_action_share_time_map": True,
        "fps": args.fps,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    minimum_post = min(float(row["post_frames_after_edited_swing"]) for row in audit_rows)
    print(
        f"[sameframe_ready] supports={len(support_rows)} queries={len(query_rows)} "
        f"groups={groups} minimum_post_frames={minimum_post:.3f}"
    )


if __name__ == "__main__":
    main()
