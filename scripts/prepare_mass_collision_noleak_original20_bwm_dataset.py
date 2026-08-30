#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass/"
    "libero_two_box_collision_9speed_30mass_linear_theory_distance_noleak_"
    "270eps_lerobot_2026-08-27_hai-machine"
)
DEFAULT_REFERENCE = Path("data/mass_collision_linear_theory_distance_bwm_full61_20260717/train.jsonl")
DEFAULT_OUTPUT = Path("data/mass_collision_noleak_original20_mainview_bwm_full61_20260827")
MAIN_VIDEO_KEY = "observation.images.image"
NUM_FRAMES = 61
MASS_MATCH_RTOL = 1e-6
PROMPT = (
    "observe a moving box collide with a second box on a smooth table and predict "
    "both boxes' post-collision motion; the environment parameter is hidden"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def data_path(episode_index: int) -> str:
    return f"data/chunk-000/episode_{episode_index:06d}.parquet"


def video_path(episode_index: int) -> str:
    return f"videos/chunk-000/{MAIN_VIDEO_KEY}/episode_{episode_index:06d}.mp4"


def read_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    return np.asarray(table.to_pydict()["action"], dtype=np.float32)


def compute_action_stats(source_root: Path, metadata: list[dict]) -> dict:
    actions = np.concatenate(
        [read_actions(source_root / data_path(int(row["episode_index"]))) for row in metadata],
        axis=0,
    )
    stat = {
        "shape": [int(actions.shape[1])],
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "p01": np.quantile(actions, 0.01, axis=0).tolist(),
        "p99": np.quantile(actions, 0.99, axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
    }
    return {"action_pose": stat, "eef_delta": stat}


def match_reference_masses(source_masses: list[float], reference_masses: list[float]) -> list[float]:
    matched = []
    for reference in reference_masses:
        nearest = min(source_masses, key=lambda value: abs(value - reference))
        if not np.isclose(nearest, reference, rtol=MASS_MATCH_RTOL, atol=1e-10):
            raise ValueError(f"No source mass matches reference mass {reference:.17g}; nearest={nearest:.17g}.")
        matched.append(nearest)
    if len(set(matched)) != len(reference_masses):
        raise ValueError("Reference masses do not map one-to-one onto source masses.")
    return sorted(matched)


def build_row(
    source_root: Path,
    meta: dict,
    mass_rank: int,
    total_frames: int,
    source_split: str,
) -> dict:
    episode_index = int(meta["episode_index"])
    mass = float(meta["target_mass_kg"])
    action_id = int(meta["action_id"])
    action_rel = data_path(episode_index)
    video_rel = video_path(episode_index)
    for relative in (action_rel, video_rel):
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)

    metrics = meta.get("metrics", {})
    return {
        "sample_id": f"mass_collision_noleak:ep{episode_index:06d}:frames0000-0060",
        "episode_index": episode_index,
        "case_id": str(meta["case_id"]),
        "pair_id": f"m{mass_rank:02d}_a{action_id:02d}",
        "source_dataset": source_root.name,
        "source_split": source_split,
        "target_mass_index": mass_rank,
        "source_mass_index_short_to_long": int(meta["mass_index_short_to_long"]),
        "target_mass_kg": mass,
        "target_mass_g": float(meta["target_mass_g"]),
        "mass_source": str(meta.get("mass_source", "")),
        "physical_parameter_name": "target_mass_kg",
        # Compatibility alias used only as the grouped-Z table key.
        # The numerical mass is never exposed directly to the model.
        "friction_mu": mass,
        "action_id": action_id,
        "speed_index": action_id,
        "action_amplitude": float(meta["A"]),
        "preimpact_speed_mps": float(metrics["preimpact_projectile_vx_mps"]),
        "push_steps": int(meta["push_steps"]),
        "first_collision_frame": int(metrics.get("first_block_collision_frame", -1)),
        "separation_frame": int(metrics.get("separation_frames", -1)),
        "visual_leak_gate_passed": bool(meta["visual_leak_gate_passed"]),
        "start_frame": 0,
        "end_frame": NUM_FRAMES - 1,
        "length": NUM_FRAMES,
        "valid_frames": total_frames,
        "total_frames": total_frames,
        "pad_short": total_frames < NUM_FRAMES,
        "chunk_type": "full_two_box_collision_rollout_noleak",
        # Main camera only. Wrist-view data is deliberately excluded.
        "video": [video_rel],
        "action": action_rel,
        "prompt": PROMPT,
        "task": "hidden_target_mass_collision_dynamics_noleak",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-metadata", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-mode",
        choices=("reference20", "all30"),
        default="reference20",
        help="Keep the historical 20-mass split, or materialize all 30 masses for evaluation.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    source_rows = read_jsonl(source_root / "meta/push_box_episode_metadata.jsonl")
    episodes = read_jsonl(source_root / "meta/episodes.jsonl")
    episode_lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}

    if len(source_rows) != 270 or len(episode_lengths) != 270:
        raise ValueError(f"Expected 270 source rows and episodes, found {len(source_rows)} and {len(episode_lengths)}.")
    if set(episode_lengths.values()) != {60}:
        raise ValueError("All source episodes must contain exactly 60 valid frames.")

    source_masses = sorted({float(row["target_mass_kg"]) for row in source_rows})
    if len(source_masses) != 30:
        raise ValueError(f"Expected 30 source masses, found {len(source_masses)}.")
    if args.selection_mode == "reference20":
        reference_rows = read_jsonl(args.reference_metadata)
        reference_masses = sorted({float(row["target_mass_kg"]) for row in reference_rows})
        if len(reference_masses) != 20:
            raise ValueError(f"Expected 20 reference masses, found {len(reference_masses)}.")
        selected_masses = match_reference_masses(source_masses, reference_masses)
        expected_rows = 180
        source_split = "mass20_of_30_speed9_noleak_mainview_hidden"
        selection_method = f"nearest reference target_mass_kg with rtol={MASS_MATCH_RTOL}"
        reference_metadata = str(args.reference_metadata)
    else:
        selected_masses = source_masses
        expected_rows = 270
        source_split = "mass30_speed9_noleak_mainview_hidden_eval"
        selection_method = "all 30 source target_mass_kg values"
        reference_metadata = None
    selected_rows = [
        row
        for row in source_rows
        if any(np.isclose(float(row["target_mass_kg"]), mass, rtol=MASS_MATCH_RTOL, atol=1e-10) for mass in selected_masses)
    ]
    selected_rows.sort(key=lambda row: int(row["episode_index"]))
    if len(selected_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} selected rows, found {len(selected_rows)}.")
    if not all(bool(row.get("visual_leak_gate_passed")) for row in selected_rows):
        raise ValueError("At least one selected episode failed the visual leak gate.")

    selected_mass_for_row = {}
    for row in selected_rows:
        source_mass = float(row["target_mass_kg"])
        selected_mass_for_row[int(row["episode_index"])] = min(selected_masses, key=lambda value: abs(value - source_mass))
    mass_to_rank = {mass: rank for rank, mass in enumerate(selected_masses)}
    pairs = Counter(
        (selected_mass_for_row[int(row["episode_index"])], int(row["action_id"]))
        for row in selected_rows
    )
    if len(pairs) != expected_rows or set(pairs.values()) != {1}:
        raise ValueError(
            f"Selected data is not a complete, unique {len(selected_masses)}-mass x 9-speed Cartesian product."
        )

    rows = [
        build_row(
            source_root,
            row,
            mass_to_rank[selected_mass_for_row[int(row["episode_index"])]],
            episode_lengths[int(row["episode_index"])],
            source_split,
        )
        for row in selected_rows
    ]
    write_jsonl(args.output_dir / "train.jsonl", rows)
    action_stats = compute_action_stats(source_root, selected_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "action_stats.json").write_text(
        json.dumps(action_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_source_indices = sorted({int(row["mass_index_short_to_long"]) for row in selected_rows})
    summary = {
        "source_root": str(source_root),
        "reference_metadata": reference_metadata,
        "selection_mode": args.selection_mode,
        "selection_method": selection_method,
        "num_samples": len(rows),
        "mass_group_count": len(selected_masses),
        "selected_mass_values_kg": selected_masses,
        "selected_source_mass_indices_short_to_long": selected_source_indices,
        "held_out_mass_values_kg": [mass for mass in source_masses if mass not in selected_masses],
        "speed_action_count": 9,
        "samples_per_mass": 9,
        "video_keys": [MAIN_VIDEO_KEY],
        "wrist_view_excluded": True,
        "visual_leak_gate_passed": True,
        "valid_frames": 60,
        "model_frames_after_tail_padding": NUM_FRAMES,
        "group_field": "target_mass_kg",
        "trainer_compatibility_group_field": "friction_mu",
        "mass_is_model_input": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] samples={len(rows)} masses={len(selected_masses)} speeds=9 "
        f"views=main-only output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
