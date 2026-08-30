#!/usr/bin/env python3
"""Select informative collision supports while keeping both objects on screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.evaluation.io import read_video_frames  # noqa: E402
from wan_video_action.evaluation.sim_task_extractors import extract_sim_task_state  # noqa: E402


def _metadata(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _video_path(row: dict, dataset_root: Path) -> Path:
    values = row["video"] if isinstance(row["video"], list) else [row["video"]]
    candidates = [value for value in values if "wrist" not in str(value).lower()]
    for value in [*candidates, *values]:
        path = Path(value)
        resolved = path if path.is_absolute() else dataset_root / path
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"No main-view video for {row.get('sample_id')}")


def _frames(row: dict, dataset_root: Path) -> np.ndarray:
    frames = np.asarray(read_video_frames(_video_path(row, dataset_root)))
    start = int(row.get("start_frame", 0))
    stride = int(row.get("frame_stride", 1))
    length = int(row.get("length", len(frames)))
    indices = np.clip(start + np.arange(length) * stride, 0, len(frames) - 1)
    return frames[indices]


def _median_center(centroids: np.ndarray, visible: np.ndarray, indices: np.ndarray) -> np.ndarray:
    usable = indices[visible[indices]]
    if len(usable) == 0:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return np.median(centroids[usable], axis=0)


def _candidate(
    index: int,
    row: dict,
    dataset_root: Path,
    *,
    safety_margin_px: float,
    minimum_visible_fraction: float,
    target_min_px: float,
    target_max_px: float,
) -> dict:
    frames = _frames(row, dataset_root)
    state = extract_sim_task_state(
        "mass_collision",
        frames,
        fps=20.0,
        main_view_width=224,
        edge_margin=16,
        max_tracking_jump_px=64.0,
    )
    visible = np.asarray(state.visible, dtype=bool)
    centroids = np.asarray(state.centroids, dtype=np.float64)
    first = np.arange(min(5, len(frames)))
    last = np.arange(max(0, len(frames) - 5), len(frames))
    red_initial = _median_center(centroids[:, 0], visible[:, 0], first)
    red_final = _median_center(centroids[:, 0], visible[:, 0], last)
    red_displacement = float(np.linalg.norm(red_final - red_initial))
    visible_fraction = visible.mean(axis=0)
    final_visible = visible[last].all(axis=0)
    offscreen = np.asarray(state.events.get("offscreen", np.zeros_like(visible)), dtype=bool)
    offscreen_any = offscreen.any(axis=0)

    final_centers = centroids[last]
    edge_distances = np.stack(
        [
            final_centers[..., 0],
            (state.image_width - 1) - final_centers[..., 0],
            final_centers[..., 1],
            (state.image_height - 1) - final_centers[..., 1],
        ],
        axis=-1,
    )
    final_min_margin = np.nanmin(edge_distances, axis=(0, 2))
    strict_visible = bool(
        np.all(visible_fraction >= minimum_visible_fraction)
        and np.all(final_visible)
        and not np.any(offscreen_any)
        and np.all(final_min_margin >= safety_margin_px)
    )
    target_mid = 0.5 * (target_min_px + target_max_px)
    if target_min_px <= red_displacement <= target_max_px:
        displacement_penalty = abs(red_displacement - target_mid)
        displacement_class = "inside_informative_band"
    else:
        displacement_penalty = 100.0 + min(
            abs(red_displacement - target_min_px), abs(red_displacement - target_max_px)
        )
        displacement_class = "outside_informative_band"
    return {
        "sample_index": index,
        "sample_id": row.get("sample_id"),
        "target_mass_kg": float(row["target_mass_kg"]),
        "target_mass_index": int(row["target_mass_index"]),
        "action_id": int(row["action_id"]),
        "action_amplitude": float(row["action_amplitude"]),
        "red_displacement_px": red_displacement,
        "red_visible_fraction": float(visible_fraction[0]),
        "blue_visible_fraction": float(visible_fraction[1]),
        "red_final_margin_px": float(final_min_margin[0]),
        "blue_final_margin_px": float(final_min_margin[1]),
        "red_offscreen": bool(offscreen_any[0]),
        "blue_offscreen": bool(offscreen_any[1]),
        "strict_both_visible": strict_visible,
        "displacement_class": displacement_class,
        "selection_score": displacement_penalty,
    }


def _uniform_indices(count: int, total: int) -> list[int]:
    if total < count:
        raise RuntimeError(f"Need {count} valid mass groups, found {total}")
    selected = np.rint(np.linspace(0, total - 1, count)).astype(int).tolist()
    if len(set(selected)) != count:
        raise RuntimeError(f"Uniform selection produced duplicate indices: {selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-count", type=int, default=10)
    parser.add_argument("--safety-margin-px", type=float, default=20.0)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.95)
    parser.add_argument("--target-min-px", type=float, default=25.0)
    parser.add_argument("--target-max-px", type=float, default=60.0)
    parser.add_argument("--id-masses", type=float, nargs="*", default=None)
    parser.add_argument("--ood-masses", type=float, nargs="*", default=None)
    parser.add_argument(
        "--allow-min-action-fallback",
        action="store_true",
        help="Retain a requested group with action_id=0 when no strict visible support exists.",
    )
    args = parser.parse_args()

    rows = _metadata(args.metadata_path)
    explicit_protocol = args.id_masses is not None or args.ood_masses is not None
    if explicit_protocol and (not args.id_masses or not args.ood_masses):
        raise ValueError("Explicit selection requires non-empty --id-masses and --ood-masses.")
    available_masses = sorted({float(row["target_mass_kg"]) for row in rows})
    requested: list[tuple[str, float]] = []
    if explicit_protocol:
        for domain, values in (("id", args.id_masses), ("ood", args.ood_masses)):
            for requested_mass in values:
                nearest = min(available_masses, key=lambda value: abs(value - requested_mass))
                if not math.isclose(nearest, requested_mass, rel_tol=1e-5, abs_tol=1e-9):
                    raise ValueError(
                        f"Requested {domain} mass {requested_mass:.12g} is absent; nearest={nearest:.12g}."
                    )
                requested.append((domain, nearest))
        resolved_masses = [mass for _, mass in requested]
        if len(set(resolved_masses)) != len(resolved_masses):
            raise ValueError(f"ID/OOD mass requests overlap after resolution: {resolved_masses}")
        requested_mass_set = set(resolved_masses)
        indexed_rows = [
            (index, row)
            for index, row in enumerate(rows)
            if float(row["target_mass_kg"]) in requested_mass_set
        ]
    else:
        indexed_rows = list(enumerate(rows))
    candidates = [
        _candidate(
            index,
            row,
            args.dataset_root,
            safety_margin_px=args.safety_margin_px,
            minimum_visible_fraction=args.minimum_visible_fraction,
            target_min_px=args.target_min_px,
            target_max_px=args.target_max_px,
        )
        for index, row in indexed_rows
    ]
    by_mass: dict[float, list[dict]] = {}
    for candidate in candidates:
        by_mass.setdefault(float(candidate["target_mass_kg"]), []).append(candidate)

    best_by_mass = []
    for mass in sorted(by_mass):
        valid = [item for item in by_mass[mass] if item["strict_both_visible"]]
        if not valid:
            continue
        valid.sort(key=lambda item: (item["selection_score"], item["action_id"]))
        best_by_mass.append(valid[0])

    if explicit_protocol:
        selected_with_domain: list[tuple[str, dict]] = []
        for domain, mass in requested:
            strict = [item for item in by_mass[mass] if item["strict_both_visible"]]
            if strict:
                strict.sort(key=lambda item: (item["selection_score"], item["action_id"]))
                support = dict(strict[0])
                support["selection_mode"] = "strict_visible_preferred_displacement"
            elif args.allow_min_action_fallback:
                fallback = [item for item in by_mass[mass] if int(item["action_id"]) == 0]
                if len(fallback) != 1:
                    raise RuntimeError(
                        f"Expected one minimum-action fallback for mass={mass}, found {len(fallback)}."
                    )
                support = dict(fallback[0])
                support["selection_mode"] = "minimum_action_visibility_fallback"
            else:
                raise RuntimeError(f"No strict support candidate for requested {domain} mass={mass:.12g}.")
            selected_with_domain.append((domain, support))
        selected_with_domain.sort(
            key=lambda item: (0 if item[0] == "id" else 1, -float(item[1]["target_mass_kg"]))
        )
    else:
        chosen_positions = _uniform_indices(args.environment_count, len(best_by_mass))
        selected = [best_by_mass[position] for position in chosen_positions]
        selected.sort(key=lambda item: float(item["target_mass_kg"]), reverse=True)
        selected_with_domain = [("id", item) for item in selected]

    environments = []
    for domain, support in selected_with_domain:
        mass = float(support["target_mass_kg"])
        support_index = int(support["sample_index"])
        queries = sorted(
            int(candidate["sample_index"])
            for candidate in candidates
            if math.isclose(float(candidate["target_mass_kg"]), mass, rel_tol=1e-7, abs_tol=1e-9)
            and int(candidate["sample_index"]) != support_index
        )
        environments.append({
            "domain": domain,
            "environment_id": f"{domain}_mass_index_{int(support['target_mass_index']):02d}",
            "target_mass_index": int(support["target_mass_index"]),
            "target_mass_kg": mass,
            "support_indices": [support_index],
            "query_indices": queries,
            "selection": support,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "K=1 support selected directly from no-leak GT with both-object visibility constraints",
        "support_size": 1,
        "selection_uses_ground_truth": True,
        "selection_constraints": {
            "minimum_visible_fraction_per_object": args.minimum_visible_fraction,
            "no_offscreen_event_for_either_object": True,
            "final_five_frame_minimum_centroid_margin_px": args.safety_margin_px,
            "preferred_red_displacement_px": [args.target_min_px, args.target_max_px],
        },
        "environment_selection": (
            "explicit balanced 5-ID/5-OOD masses with strict-visible preference and minimum-action fallback"
            if explicit_protocol
            else "mass groups uniformly sampled from groups with at least one valid support"
        ),
        "environments": environments,
    }
    (args.output_dir / "support_query_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    transfer_plan = [
        {
            "source_index": int(environment["support_indices"][0]),
            "source_sample_id": rows[int(environment["support_indices"][0])].get("sample_id"),
            "source_friction_mu": float(environment["target_mass_kg"]),
            "target_indices": [int(value) for value in environment["query_indices"]],
            "target_sample_ids": [rows[int(value)].get("sample_id") for value in environment["query_indices"]],
        }
        for environment in environments
    ]
    (args.output_dir / "transfer_plan.json").write_text(
        json.dumps(transfer_plan, indent=2) + "\n", encoding="utf-8"
    )
    for domain in ("id", "ood"):
        domain_plan = [
            record
            for record, environment in zip(transfer_plan, environments)
            if environment["domain"] == domain
        ]
        (args.output_dir / f"transfer_plan_{domain}.json").write_text(
            json.dumps(domain_plan, indent=2) + "\n", encoding="utf-8"
        )
    (args.output_dir / "source_indices.txt").write_text(
        ",".join(str(environment["support_indices"][0]) for environment in environments) + "\n",
        encoding="utf-8",
    )
    for domain in ("id", "ood"):
        (args.output_dir / f"source_indices_{domain}.txt").write_text(
            ",".join(
                str(environment["support_indices"][0])
                for environment in environments
                if environment["domain"] == domain
            )
            + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "target_indices_by_source.txt").write_text(
        ";".join(
            f"{environment['support_indices'][0]}:"
            + ",".join(str(value) for value in environment["query_indices"])
            for environment in environments
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "candidate_report.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "candidate_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    (args.output_dir / "source_episode_mapping.json").write_text(
        json.dumps(
            {
                str(environment["support_indices"][0]): {
                    "domain": environment["domain"],
                    "target_mass_kg": environment["target_mass_kg"],
                    "target_mass_index": environment["target_mass_index"],
                    "support_sample_id": rows[int(environment["support_indices"][0])].get("sample_id"),
                    "query_indices": environment["query_indices"],
                }
                for environment in environments
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[done] "
        f"candidates={len(candidates)} valid_mass_groups={len(best_by_mass)} "
        f"selected={len(selected_with_domain)} "
        f"fallbacks={sum(item.get('selection_mode') == 'minimum_action_visibility_fallback' for _, item in selected_with_domain)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
