#!/usr/bin/env python3
"""Opt-in static Soft trainer: repeat group slots only for undersized C pools."""
from __future__ import annotations

from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_real97_soft_static_1frame_grouped_context as static

grouped = static.grouped
_legacy_sample = grouped._sample_update_indices
_reported_pools = set()


def sample_update(**kwargs):
    args = kwargs["args"]
    desired = int(args.grouped_context_friction_groups_per_update)
    episodes_per_slot = int(args.grouped_context_actions_per_update)
    if (desired, episodes_per_slot) != (5, 6):
        raise ValueError("Balanced-nine Soft requires 5 group slots x 6 episodes per GPU")
    if args.grouped_context_sampling_mode != "uniform_episode_then_window":
        raise ValueError("Preserve episode-then-annotated-window sampling")
    allowed = kwargs.get("allowed_friction_values")
    values = sorted(float(v) for v in kwargs["grouped_indices"]
                    if allowed is None or any(abs(float(v) - float(a)) <= 1e-5 for a in allowed))
    if not values:
        raise ValueError("Empty curriculum sampling pool; never fall back to another group")
    kwargs.update(friction_groups=desired, actions_per_update=episodes_per_slot,
                  microbatches_per_update=desired * episodes_per_slot)
    if len(values) >= desired:
        return _legacy_sample(**kwargs)
    accelerator = kwargs["accelerator"]
    rng = random.Random(int(args.seed) + int(kwargs["update_idx"]) * max(1, int(accelerator.num_processes))
                        + int(accelerator.process_index))
    selected_groups = rng.choices(values, k=desired)
    if tuple(values) not in _reported_pools:
        print(f"[soft_dual9] sampling WITH replacement from allowed latent pool {values}; "
              f"{desired} slots x {episodes_per_slot} episodes, rank={accelerator.process_index}", flush=True)
        _reported_pools.add(tuple(values))
    rows = kwargs["rows"]
    result = []
    for value in selected_groups:
        by_episode = {}
        for candidates in kwargs["grouped_indices"][value].values():
            for index in candidates:
                row = rows[index]
                if row["dataset_split"] != "train" or row["sampling_kind"] != "stationary_valid":
                    raise ValueError("Sampling pool contains an unapproved row")
                episode = row[str(args.grouped_context_episode_key)]
                by_episode.setdefault(episode, []).append(index)
        if len(by_episode) < episodes_per_slot:
            raise ValueError(f"Latent {value} has too few eligible training episodes; never prune it")
        for episode in rng.sample(sorted(by_episode, key=str), episodes_per_slot):
            result.append(rng.choice(by_episode[episode]))
    rng.shuffle(result)
    return result


if __name__ == "__main__":
    grouped.build_dataset = static.build_static_dataset
    grouped._sample_update_indices = sample_update
    grouped.main()
