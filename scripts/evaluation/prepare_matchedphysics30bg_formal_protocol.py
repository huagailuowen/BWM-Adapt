#!/usr/bin/env python3
"""Prepare the shared 5-ID/5-OOD K=1 protocol for matched-physics 30bg."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path


ID_MU_INDICES = (0, 8, 15, 21, 29)
OOD_MU_INDICES = (1, 7, 13, 22, 28)
SUPPORT_ANCHORS = (
    (0.002, 0),
    (0.0332, 3),
    (0.07125, 4),
    (0.12375, 6),
    (0.1972, 8),
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lerobot = args.dataset_root / "hidden_straight_lerobot"
    raw_rows = [
        json.loads(line)
        for line in (lerobot / "meta/push_box_episode_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    episodes = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (lerobot / "meta/episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(int(row["mu_index"]), int(row["action_id"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["episode_index"]))

    selected_mu = (*ID_MU_INDICES, *OOD_MU_INDICES)
    metadata = []
    environments = []
    manifest = []
    for mu_index in selected_mu:
        domain = "id" if mu_index in ID_MU_INDICES else "ood"
        friction = float(grouped[(mu_index, 0)][0]["mu"])
        support_action = min(
            SUPPORT_ANCHORS,
            key=lambda item: (abs(friction - item[0]), item[1]),
        )[1]
        indices_by_action = {}
        for action_id in range(10):
            candidates = grouped[(mu_index, action_id)]
            if len(candidates) != 5:
                raise ValueError(
                    f"mu_index={mu_index} action={action_id}: expected 5 episodes, "
                    f"got {len(candidates)}"
                )
            raw = candidates[(mu_index + action_id) % len(candidates)]
            episode_index = int(raw["episode_index"])
            chunk_index = episode_index // 1000
            total_frames = int(episodes[episode_index]["length"])
            index = len(metadata)
            indices_by_action[action_id] = index
            metadata.append(
                {
                    "action": f"hidden_straight_lerobot/data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
                    "action_amplitude": float(raw["A"]),
                    "action_id": action_id,
                    "background_id": str(raw["environment_id"]),
                    "background_index": int(raw["environment_index"]),
                    "case_id": str(raw["case_id"]),
                    "chunk_type": "fixed_65_105",
                    "context_group_id": mu_index,
                    "end_frame": 105,
                    "environment_group": f"mu_index_{mu_index}",
                    "environment_id": str(raw["environment_id"]),
                    "environment_index": int(raw["environment_index"]),
                    "episode_index": episode_index,
                    "friction_mu": friction,
                    "length": 41,
                    "mu_index": mu_index,
                    "pad_short": total_frames < 106,
                    "physical_friction_mu": friction,
                    "prompt": "observe how the object slides after a short robot push on the table; no target is shown",
                    "push_action_peak_x": float(raw["A"]),
                    "push_end": 70 + int(raw["push_steps"]),
                    "push_start": 70,
                    "push_steps": int(raw["push_steps"]),
                    "sample_id": f"hidden_straight_lerobot:ep{episode_index:06d}:frames0065-0105",
                    "shared_dynamics_group": mu_index,
                    "source_context_group_id": mu_index,
                    "source_dataset": "hidden_straight_lerobot",
                    "source_friction_mu": friction,
                    "source_mu_index": mu_index,
                    "source_split": domain,
                    "start_frame": 65,
                    "task": "libero_plus_push_box_multibackground_physical_observation",
                    "total_frames": total_frames,
                    "valid_frames": min(41, max(0, total_frames - 65)),
                    "video": [
                        f"hidden_straight_lerobot/videos/chunk-{chunk_index:03d}/observation.images.image/episode_{episode_index:06d}.mp4",
                        f"hidden_straight_lerobot/videos/chunk-{chunk_index:03d}/observation.images.wrist_image/episode_{episode_index:06d}.mp4",
                    ],
                }
            )
        support_index = indices_by_action[support_action]
        query_indices = [indices_by_action[action] for action in range(10) if action != support_action]
        record = {
            "domain": domain,
            "environment_id": f"mu_index_{mu_index}",
            "friction_mu": friction,
            "mu_index": mu_index,
            "source_index": support_index,
            "source_sample_id": metadata[support_index]["sample_id"],
            "support_indices": [support_index],
            "support_sample_ids": [metadata[support_index]["sample_id"]],
            "query_indices": query_indices,
            "target_indices": query_indices,
            "target_sample_ids": [metadata[index]["sample_id"] for index in query_indices],
            "support_selection": "matched_event80_informative_25_60_anchor",
            "selection": {
                "support_action_id": support_action,
                "reason": "nearest matched Event80 informative-support anchor",
            },
        }
        manifest.append(record)
        environments.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_dir / "evaluation.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata),
    )
    atomic_text(
        args.output_dir / "support_query_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(
        args.output_dir / "protocol.json",
        json.dumps(
            {
                "evaluation_id": "matchedphysics30bg_active20_id5_ood5_k1_informative_v1",
                "protocol": "Event80-matched K=1 informative support with nine disjoint action queries",
                "support_size": 1,
                "selection_uses_ground_truth": True,
                "selection_target": {
                    "minimum_displacement_px": 25.0,
                    "maximum_displacement_px": 60.0,
                    "mapping": "nearest matched Event80 informative-support action anchor",
                },
                "id_mu_indices": list(ID_MU_INDICES),
                "ood_mu_indices": list(OOD_MU_INDICES),
                "environments": environments,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        f"[done] metadata={len(metadata)} environments={len(environments)} "
        f"queries={sum(len(row['query_indices']) for row in manifest)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
