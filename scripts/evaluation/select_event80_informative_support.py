#!/usr/bin/env python3
"""Select one informative, method-independent Event80 support per environment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.event80_pushbox import track_event80_block
from wan_video_action.evaluation.io import read_video_frames, write_json_atomic


DEFAULT_ID_ENVIRONMENTS = (0, 10, 20, 30, 35)
DEFAULT_OOD_ENVIRONMENTS = (1, 9, 17, 26, 34)


def _parse_indices(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _main_video_path(row: dict[str, Any], dataset_root: Path) -> Path:
    raw_videos = row.get("video", [])
    videos = [raw_videos] if isinstance(raw_videos, str) else list(raw_videos)
    matches = [
        value
        for value in videos
        if "observation.images.image" in str(value)
        and "wrist_image" not in str(value)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one main-camera video for episode {row.get('episode_index')}, "
            f"found {matches}."
        )
    path = Path(str(matches[0]))
    return path if path.is_absolute() else dataset_root / path


def _measure_candidate(task: tuple[dict[str, Any], str, int, int]) -> dict[str, Any]:
    row, dataset_root_raw, endpoint_frames, minimum_visible = task
    dataset_root = Path(dataset_root_raw)
    sample_index = int(row["episode_index"])
    output: dict[str, Any] = {
        "sample_index": sample_index,
        "action_id": int(row["action_id"]),
        "mu_index": int(row["mu_index"]),
        "friction_mu": float(row["friction_mu"]),
        "push_action_peak_x": (
            float(row["push_action_peak_x"])
            if row.get("push_action_peak_x") is not None
            else None
        ),
    }
    try:
        video_path = _main_video_path(row, dataset_root)
        start_frame = int(row["start_frame"])
        frame_count = int(
            row.get(
                "length",
                int(row["end_frame"]) - start_frame + 1,
            )
        )
        frames = read_video_frames(video_path, start_frame, frame_count)
        track = track_event80_block(frames)
        usable = track.visible & ~track.offscreen & np.all(
            np.isfinite(track.centers), axis=1
        )
        start_mask = usable[:endpoint_frames]
        end_mask = usable[-endpoint_frames:]
        start_centers = track.centers[:endpoint_frames][start_mask]
        end_centers = track.centers[-endpoint_frames:][end_mask]
        start_center = (
            np.median(start_centers, axis=0)
            if len(start_centers)
            else np.array([np.nan, np.nan])
        )
        end_center = (
            np.median(end_centers, axis=0)
            if len(end_centers)
            else np.array([np.nan, np.nan])
        )
        displacement_xy = end_center - start_center
        endpoint_valid = bool(
            len(start_centers) >= minimum_visible
            and len(end_centers) >= minimum_visible
            and not np.any(track.offscreen[-endpoint_frames:])
            and np.all(np.isfinite(displacement_xy))
        )
        output.update(
            {
                "video_path": str(video_path),
                "start_frame": start_frame,
                "end_frame": start_frame + len(frames) - 1,
                "tracked_frames": int(np.count_nonzero(usable)),
                "visible_fraction": float(np.mean(usable)),
                "start_visible_count": int(len(start_centers)),
                "end_visible_count": int(len(end_centers)),
                "first_offscreen_window_frame": track.first_offscreen_frame,
                "start_centroid_xy_px": [float(value) for value in start_center],
                "end_centroid_xy_px": [float(value) for value in end_center],
                "displacement_xy_px": [float(value) for value in displacement_xy],
                "final_displacement_px": float(np.linalg.norm(displacement_xy)),
                "valid_endpoint": endpoint_valid,
                "error": None,
            }
        )
    except Exception as error:  # Preserve every failed candidate in the audit log.
        output.update(
            {
                "valid_endpoint": False,
                "final_displacement_px": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return output


def _choose_candidate(
    candidates: list[dict[str, Any]],
    minimum_displacement: float,
    maximum_displacement: float,
) -> tuple[dict[str, Any], str]:
    valid = [
        row
        for row in candidates
        if row["valid_endpoint"] and row["final_displacement_px"] is not None
    ]
    if not valid:
        errors = "; ".join(
            f"sample {row['sample_index']}: {row.get('error') or 'invalid endpoint'}"
            for row in candidates
        )
        raise RuntimeError(f"No valid support candidate. {errors}")
    target = (minimum_displacement + maximum_displacement) / 2.0
    in_band = [
        row
        for row in valid
        if minimum_displacement
        <= float(row["final_displacement_px"])
        <= maximum_displacement
    ]
    if in_band:
        return (
            min(
                in_band,
                key=lambda row: (
                    abs(float(row["final_displacement_px"]) - target),
                    row["action_id"],
                ),
            ),
            "within_target_band_closest_to_midpoint",
        )
    displacements = [float(row["final_displacement_px"]) for row in valid]
    if max(displacements) < minimum_displacement:
        return (
            max(valid, key=lambda row: (row["final_displacement_px"], -row["action_id"])),
            "fallback_maximum_visible_displacement_below_band",
        )
    if min(displacements) > maximum_displacement:
        return (
            min(valid, key=lambda row: (row["final_displacement_px"], row["action_id"])),
            "fallback_minimum_visible_displacement_above_band",
        )
    return (
        min(
            valid,
            key=lambda row: (
                abs(float(row["final_displacement_px"]) - target),
                row["action_id"],
            ),
        ),
        "fallback_nearest_visible_displacement_outside_band",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _write_text_atomic(path, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-id",
        default="event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1",
    )
    parser.add_argument(
        "--id-environments",
        type=_parse_indices,
        default=DEFAULT_ID_ENVIRONMENTS,
    )
    parser.add_argument(
        "--ood-environments",
        type=_parse_indices,
        default=DEFAULT_OOD_ENVIRONMENTS,
    )
    parser.add_argument("--minimum-displacement", type=float, default=25.0)
    parser.add_argument("--maximum-displacement", type=float, default=60.0)
    parser.add_argument("--endpoint-frames", type=int, default=5)
    parser.add_argument("--minimum-visible", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.minimum_displacement >= args.maximum_displacement:
        raise ValueError("minimum-displacement must be below maximum-displacement.")
    rows = _read_jsonl(args.metadata_path)
    domains = {
        **{index: "id" for index in args.id_environments},
        **{index: "ood" for index in args.ood_environments},
    }
    selected_rows = [row for row in rows if int(row["mu_index"]) in domains]
    grouped: dict[int, list[dict[str, Any]]] = {index: [] for index in domains}
    for row in selected_rows:
        grouped[int(row["mu_index"])].append(row)
    for mu_index, group in grouped.items():
        if len(group) != 10:
            raise ValueError(
                f"mu_index={mu_index} has {len(group)} episodes; expected exactly 10."
            )

    tasks = [
        (row, str(args.dataset_root), args.endpoint_frames, args.minimum_visible)
        for row in selected_rows
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        measured = list(executor.map(_measure_candidate, tasks))
    measured.sort(key=lambda row: (row["mu_index"], row["action_id"]))
    measured_by_mu: dict[int, list[dict[str, Any]]] = {index: [] for index in domains}
    for row in measured:
        measured_by_mu[row["mu_index"]].append(row)

    metadata_by_index = {int(row["episode_index"]): row for row in rows}
    environment_records = []
    support_rows = []
    ordered_environments = list(args.id_environments) + list(args.ood_environments)
    for mu_index in ordered_environments:
        candidates = measured_by_mu[mu_index]
        chosen, reason = _choose_candidate(
            candidates,
            args.minimum_displacement,
            args.maximum_displacement,
        )
        support_index = int(chosen["sample_index"])
        query_indices = [
            int(row["sample_index"])
            for row in sorted(candidates, key=lambda row: row["action_id"])
            if int(row["sample_index"]) != support_index
        ]
        environment_records.append(
            {
                "environment_id": f"mu_index_{mu_index}",
                "mu_index": mu_index,
                "friction_mu": float(chosen["friction_mu"]),
                "domain": domains[mu_index],
                "support_indices": [support_index],
                "query_indices": query_indices,
                "selection": {
                    "support_action_id": int(chosen["action_id"]),
                    "final_displacement_px": float(chosen["final_displacement_px"]),
                    "reason": reason,
                },
            }
        )
        support_rows.append(metadata_by_index[support_index])

    manifest = {
        "evaluation_id": args.evaluation_id,
        "protocol": (
            "oracle informative K=1 support selected from all ten GT actions per "
            "environment; selected support is excluded from nine read-only queries"
        ),
        "support_size": 1,
        "selection_uses_ground_truth": True,
        "selection_target": {
            "metric": "Euclidean distance between median block centroids in first and last endpoint frames",
            "minimum_displacement_px": args.minimum_displacement,
            "maximum_displacement_px": args.maximum_displacement,
            "target_midpoint_px": (
                args.minimum_displacement + args.maximum_displacement
            )
            / 2.0,
            "endpoint_frames": args.endpoint_frames,
            "minimum_visible_endpoint_frames": args.minimum_visible,
            "require_visible_final_endpoint": True,
        },
        "environments": environment_records,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "support_query_manifest.json", manifest)
    write_json_atomic(
        output_dir / "support_selection_summary.json",
        {
            "evaluation_id": args.evaluation_id,
            "selected_environment_count": len(environment_records),
            "candidate_count": len(measured),
            "environments": environment_records,
        },
    )
    _write_jsonl_atomic(output_dir / "support_candidates.jsonl", measured)
    _write_jsonl_atomic(output_dir / "support_metadata.jsonl", support_rows)

    id_queries = [
        index
        for record in environment_records
        if record["domain"] == "id"
        for index in record["query_indices"]
    ]
    ood_queries = [
        index
        for record in environment_records
        if record["domain"] == "ood"
        for index in record["query_indices"]
    ]
    supports = [record["support_indices"][0] for record in environment_records]
    env_text = "\n".join(
        [
            f"EVALUATION_ID={args.evaluation_id}",
            f"SUPPORTS={','.join(map(str, supports))}",
            f"ID_QUERIES={','.join(map(str, id_queries))}",
            f"OOD_QUERIES={','.join(map(str, ood_queries))}",
            f"ALL_QUERIES={','.join(map(str, id_queries + ood_queries))}",
            "",
        ]
    )
    _write_text_atomic(output_dir / "indices.env", env_text)
    protocol_text = (
        "# Event80 oracle informative-support protocol\n\n"
        "This benchmark preserves the fixed 5-ID/5-OOD environment set but chooses "
        "one method-independent support from all ten ground-truth actions in each "
        "environment. It is therefore an oracle demonstration-selection benchmark, "
        "not a strict online K=1 benchmark. The selected support is excluded from the "
        "nine read-only query trajectories, and every method consumes the same manifest.\n"
    )
    _write_text_atomic(output_dir / "README.md", protocol_text)
    for record in environment_records:
        selection = record["selection"]
        print(
            f"[{record['domain']}] mu_index={record['mu_index']:02d} "
            f"support={record['support_indices'][0]:03d} "
            f"action={selection['support_action_id']} "
            f"displacement={selection['final_displacement_px']:.2f}px "
            f"reason={selection['reason']}",
            flush=True,
        )
    print(f"[done] output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
