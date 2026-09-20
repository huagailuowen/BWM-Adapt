#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SUBSET = "hidden_straight_lerobot"
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
SELECTED_MU_INDICES = (0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29)
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
    chunk = f"chunk-{episode_index // 1000:03d}"
    if kind == "action":
        return f"{SUBSET}/data/{chunk}/episode_{episode_index:06d}.parquet"
    if kind == "video" and video_key is not None:
        return f"{SUBSET}/videos/{chunk}/{video_key}/episode_{episode_index:06d}.mp4"
    raise ValueError((kind, episode_index, video_key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=(
            "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox_various_env/"
            "libero_plus_push_box_event80_matched_physics_30background_pool_30friction_"
            "10action_random5per_pair_1500eps_adaptive_end_2026-09-03_hai-machine"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/push_box_matchedphysics30bg_uniform20fric_10action_65_105_shared_20260903",
    )
    parser.add_argument(
        "--action-stats-source",
        default=(
            "data/push_box_bwm_matchedphysics5bg30fric10action_65_105_"
            "shared_friction30_20260819/action_stats.json"
        ),
    )
    parser.add_argument("--start-frame", type=int, default=65)
    parser.add_argument("--num-frames", type=int, default=41)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    meta_root = source_root / SUBSET / "meta"
    episodes = read_jsonl(meta_root / "push_box_episode_metadata.jsonl")
    episode_lengths = {
        int(row["episode_index"]): int(row["length"])
        for row in read_jsonl(meta_root / "episodes.jsonl")
    }

    pair_counts = Counter((int(row["mu_index"]), int(row["action_id"])) for row in episodes)
    if len(episodes) != 1500 or len(pair_counts) != 300 or set(pair_counts.values()) != {5}:
        raise ValueError(
            f"Expected 1500 episodes and five backgrounds for each of 300 mu/action pairs; "
            f"got episodes={len(episodes)}, pairs={len(pair_counts)}, multiplicities={set(pair_counts.values())}."
        )

    context_by_source_mu = {source_mu: context for context, source_mu in enumerate(SELECTED_MU_INDICES)}
    rows: list[dict] = []
    group_backgrounds: dict[int, dict[int, str]] = defaultdict(dict)
    for episode in episodes:
        source_mu_index = int(episode["mu_index"])
        if source_mu_index not in context_by_source_mu:
            continue
        context_group_id = context_by_source_mu[source_mu_index]
        episode_index = int(episode["episode_index"])
        background_index = int(episode["environment_index"])
        background_id = str(episode["environment_id"])
        action_id = int(episode["action_id"])
        physical_mu = float(episode["mu"])
        push_steps = int(episode["push_steps"])
        total_frames = episode_lengths[episode_index]
        start_frame = int(args.start_frame)
        num_frames = int(args.num_frames)
        valid_frames = max(0, min(num_frames, total_frames - start_frame))
        if valid_frames <= 0:
            raise ValueError(f"Episode {episode_index} ends before frame {start_frame}.")

        action_path = episode_path("action", episode_index)
        video_paths = [episode_path("video", episode_index, key) for key in VIDEO_KEYS]
        environment_group = f"bg{background_index:02d}_mu{source_mu_index:02d}"
        rows.append(
            {
                "sample_id": f"{SUBSET}:ep{episode_index:06d}:frames{start_frame:04d}-{start_frame + num_frames - 1:04d}",
                "episode_index": episode_index,
                "source_dataset": SUBSET,
                "source_split": "train",
                "pair_id": environment_group,
                "case_id": str(episode["case_id"]),
                "environment_group": environment_group,
                "environment_index": background_index,
                "environment_id": background_id,
                "background_index": background_index,
                "background_id": background_id,
                "context_group_id": context_group_id,
                "friction_mu": float(context_group_id),
                "shared_dynamics_group": context_group_id,
                "source_context_group_id": source_mu_index,
                "source_friction_mu": float(source_mu_index),
                "physical_friction_mu": physical_mu,
                "mu_index": context_group_id,
                "source_mu_index": source_mu_index,
                "action_id": action_id,
                "action_amplitude": float(episode["A"]),
                "push_action_peak_x": float(episode["A"]),
                "push_start": 70,
                "push_end": 70 + push_steps,
                "push_steps": push_steps,
                "chunk_type": "fixed_65_105",
                "start_frame": start_frame,
                "end_frame": start_frame + num_frames - 1,
                "length": num_frames,
                "valid_frames": valid_frames,
                "total_frames": total_frames,
                "pad_short": valid_frames < num_frames,
                "video": video_paths,
                "action": action_path,
                "prompt": PROMPT,
                "task": "libero_plus_push_box_multibackground_physical_observation",
            }
        )
        group_backgrounds[context_group_id][background_index] = background_id

    rows.sort(key=lambda row: int(row["episode_index"]))
    context_counts = Counter(int(row["context_group_id"]) for row in rows)
    action_counts = Counter((int(row["context_group_id"]), int(row["action_id"])) for row in rows)
    if len(rows) != 1000 or set(context_counts.values()) != {50} or set(action_counts.values()) != {5}:
        raise ValueError(
            f"Unexpected selected grid: rows={len(rows)}, context_counts={dict(context_counts)}, "
            f"action multiplicities={set(action_counts.values())}."
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "test.jsonl", [])
    group_records = []
    for context_group_id, source_mu_index in enumerate(SELECTED_MU_INDICES):
        sample = next(row for row in rows if int(row["context_group_id"]) == context_group_id)
        backgrounds = group_backgrounds[context_group_id]
        group_records.append(
            {
                "context_group_id": context_group_id,
                "shared_dynamics_group": context_group_id,
                "mu_index": context_group_id,
                "source_mu_index": source_mu_index,
                "physical_friction_mu": float(sample["physical_friction_mu"]),
                "environment_indices": sorted(backgrounds),
                "environment_ids": [backgrounds[index] for index in sorted(backgrounds)],
            }
        )
    (output_dir / "context_group_map.json").write_text(
        json.dumps(group_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(Path(args.action_stats_source), output_dir / "action_stats.json")
    summary = {
        "source_root": str(source_root),
        "train_samples": len(rows),
        "selected_source_mu_indices": list(SELECTED_MU_INDICES),
        "context_group_ids": list(range(len(SELECTED_MU_INDICES))),
        "samples_per_context": 50,
        "samples_per_context_action": 5,
        "background_pool_size": len({row["background_id"] for row in rows}),
        "context_semantics": "one latent per friction shared across all sampled backgrounds",
        "sampling_semantics": "common actions for Ours; group-local random sampling for DINO and TTT",
        "start_frame": int(args.start_frame),
        "end_frame": int(args.start_frame) + int(args.num_frames) - 1,
        "num_frames": int(args.num_frames),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
