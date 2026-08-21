#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


SUBSET = "hidden_straight_lerobot"
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PROMPT = "observe how the object slides after a short robot push on the table; no target is shown"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def episode_path(kind: str, episode_index: int, video_key: str | None = None) -> str:
    chunk = f"chunk-{int(episode_index) // 1000:03d}"
    if kind == "action":
        return f"{SUBSET}/data/{chunk}/episode_{episode_index:06d}.parquet"
    if kind == "video" and video_key is not None:
        return f"{SUBSET}/videos/{chunk}/{video_key}/episode_{episode_index:06d}.mp4"
    raise ValueError(f"Unsupported episode path: kind={kind!r}, video_key={video_key!r}")


def validate_grid(
    episodes: list[dict],
    expected_environments: int,
    expected_frictions: int,
    expected_actions: int,
) -> tuple[list[int], list[float], list[int]]:
    environment_indices = sorted({int(row["environment_index"]) for row in episodes})
    expected_environment_indices = list(range(expected_environments))
    if environment_indices != expected_environment_indices:
        raise ValueError(
            f"Expected environment indices {expected_environment_indices}, got {environment_indices}."
        )

    expected_mu_indices = list(range(expected_frictions))
    expected_action_ids = list(range(expected_actions))
    reference_mu_values = None
    seen_cells = Counter()
    for row in episodes:
        seen_cells[(
            int(row["environment_index"]),
            int(row["mu_index"]),
            int(row["action_id"]),
        )] += 1

    for environment_index in environment_indices:
        subset = [
            row for row in episodes
            if int(row["environment_index"]) == environment_index
        ]
        mu_indices = sorted({int(row["mu_index"]) for row in subset})
        mu_values = [
            float(next(row["mu"] for row in subset if int(row["mu_index"]) == mu_index))
            for mu_index in mu_indices
        ]
        action_ids = sorted({int(row["action_id"]) for row in subset})
        if mu_indices != expected_mu_indices or action_ids != expected_action_ids:
            raise ValueError(
                f"Environment {environment_index} has mu_indices={mu_indices} "
                f"and action_ids={action_ids}."
            )
        if reference_mu_values is None:
            reference_mu_values = mu_values
        elif mu_values != reference_mu_values:
            raise ValueError("All visual backgrounds must share the same friction table.")

    expected_cells = expected_environments * expected_frictions * expected_actions
    if len(seen_cells) != expected_cells or set(seen_cells.values()) != {1}:
        raise ValueError(
            f"Expected {expected_cells} unique background/friction/action cells, "
            f"got {len(seen_cells)} with multiplicities {sorted(set(seen_cells.values()))}."
        )
    return environment_indices, reference_mu_values or [], expected_action_ids


def build_rows(
    source_root: Path,
    episodes: list[dict],
    start_frame: int,
    num_frames: int,
    friction_count: int,
) -> tuple[list[dict], list[dict]]:
    rows = []
    group_records: dict[int, dict] = {}
    environment_records: dict[int, str] = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        environment_index = int(episode["environment_index"])
        environment_id = str(episode["environment_id"])
        mu_index = int(episode["mu_index"])
        action_id = int(episode["action_id"])
        physical_mu = float(episode["mu"])
        metrics = episode["metrics"]
        phase_counts = metrics["phase_counts"]
        push_start = int(phase_counts["approach"]) + int(phase_counts["descend"])
        push_end = push_start + int(phase_counts["push"])
        total_frames = int(metrics["steps"])
        valid_frames = max(0, min(int(num_frames), total_frames - int(start_frame)))
        if valid_frames <= 0:
            raise ValueError(
                f"Episode {episode_index} ends before start_frame={start_frame}: "
                f"total_frames={total_frames}."
            )

        action_path = episode_path("action", episode_index)
        video_paths = [
            episode_path("video", episode_index, video_key)
            for video_key in VIDEO_KEYS
        ]
        for relative_path in [action_path, *video_paths]:
            if not (source_root / relative_path).is_file():
                raise FileNotFoundError(source_root / relative_path)

        source_context_group_id = environment_index * friction_count + mu_index
        environment_group = f"env{environment_index:02d}_mu{mu_index:02d}"
        rows.append(
            {
                "sample_id": (
                    f"{SUBSET}:ep{episode_index:06d}:"
                    f"frames{start_frame:04d}-{start_frame + num_frames - 1:04d}"
                ),
                "episode_index": episode_index,
                "source_dataset": SUBSET,
                "source_split": "train",
                "pair_id": environment_group,
                "case_id": str(episode["case_id"]),
                "environment_group": environment_group,
                "environment_index": environment_index,
                "environment_id": environment_id,
                "context_group_id": mu_index,
                "friction_mu": float(mu_index),
                "shared_dynamics_group": mu_index,
                "source_context_group_id": source_context_group_id,
                "source_friction_mu": float(source_context_group_id),
                "physical_friction_mu": physical_mu,
                "mu_index": mu_index,
                "action_id": action_id,
                "action_amplitude": float(episode["A"]),
                "push_action_peak_x": float(episode["A"]),
                "push_start": push_start,
                "push_end": push_end,
                "push_steps": int(phase_counts["push"]),
                "chunk_type": "fixed_65_105",
                "start_frame": int(start_frame),
                "end_frame": int(start_frame) + int(num_frames) - 1,
                "length": int(num_frames),
                "valid_frames": valid_frames,
                "total_frames": total_frames,
                "pad_short": valid_frames < int(num_frames),
                "video": video_paths,
                "action": action_path,
                "prompt": PROMPT,
                "task": "libero_plus_push_box_various_environment_physical_observation",
            }
        )
        environment_records[environment_index] = environment_id
        group_records[mu_index] = {
            "context_group_id": mu_index,
            "shared_dynamics_group": mu_index,
            "mu_index": mu_index,
            "physical_friction_mu": physical_mu,
        }

    environment_indices = sorted(environment_records)
    environment_ids = [environment_records[index] for index in environment_indices]
    for record in group_records.values():
        record["environment_indices"] = environment_indices
        record["environment_ids"] = environment_ids
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows, [group_records[index] for index in sorted(group_records)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a five-background x 30-friction x 10-action manifest in which "
            "all backgrounds of the same friction share one grouped context."
        )
    )
    parser.add_argument(
        "--source-root",
        default=(
            "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox_various_env/"
            "libero_plus_push_box_event80_matched_physics_5randombackground_30friction_"
            "10action_1500eps_adaptive_end_2026-08-19_hai-machine"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/push_box_bwm_matchedphysics5bg30fric10action_65_105_"
            "shared_friction30_20260819"
        ),
    )
    parser.add_argument(
        "--action-stats-source",
        default=(
            "data/push_box_bwm_various_env3x40_10action_65_105_20260727/"
            "action_stats.json"
        ),
    )
    parser.add_argument("--start-frame", type=int, default=65)
    parser.add_argument("--num-frames", type=int, default=41)
    parser.add_argument("--expected-environments", type=int, default=5)
    parser.add_argument("--expected-frictions", type=int, default=30)
    parser.add_argument("--expected-actions", type=int, default=10)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    metadata_path = source_root / SUBSET / "meta" / "push_box_episode_metadata.jsonl"
    episodes = read_jsonl(metadata_path)
    environment_indices, friction_values, action_ids = validate_grid(
        episodes,
        args.expected_environments,
        args.expected_frictions,
        args.expected_actions,
    )
    rows, group_records = build_rows(
        source_root,
        episodes,
        args.start_frame,
        args.num_frames,
        args.expected_frictions,
    )
    counts = Counter(int(row["context_group_id"]) for row in rows)
    expected_per_group = args.expected_environments * args.expected_actions
    if len(counts) != args.expected_frictions or set(counts.values()) != {expected_per_group}:
        raise ValueError(
            f"Expected {args.expected_frictions} groups with {expected_per_group} episodes each, "
            f"got counts={dict(counts)}."
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    with (output_dir / "context_group_map.json").open("w", encoding="utf-8") as handle:
        json.dump(group_records, handle, indent=2, sort_keys=True)
    action_stats_source = Path(args.action_stats_source)
    if not action_stats_source.is_file():
        raise FileNotFoundError(action_stats_source)
    shutil.copy2(action_stats_source, output_dir / "action_stats.json")

    summary = {
        "source_root": str(source_root),
        "source_metadata": str(metadata_path),
        "episodes": len(episodes),
        "train_samples": len(rows),
        "environment_indices": environment_indices,
        "friction_values": friction_values,
        "action_ids": action_ids,
        "context_groups": len(group_records),
        "samples_per_context_group": sorted(set(counts.values())),
        "context_semantics": "one context shared by all five backgrounds at the same friction",
        "sampling_semantics": "sample friction, sample action, then randomly choose a background",
        "context_lookup_field": "friction_mu",
        "context_lookup_encoding": "mu_index",
        "physical_friction_field": "physical_friction_mu",
        "start_frame": int(args.start_frame),
        "end_frame": int(args.start_frame) + int(args.num_frames) - 1,
        "num_frames": int(args.num_frames),
        "pad_short_chunks": sum(1 for row in rows if row["pad_short"]),
        "padding_policy": "repeat the final available video frame and action row",
    }
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
