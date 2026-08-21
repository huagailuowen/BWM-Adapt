"""Prequential Event80 sampling for the practical TTT-KVB baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import yaml


@dataclass(frozen=True)
class PrequentialEpisode:
    """One randomly ordered stream from a single physical environment."""

    environment_id: int
    indices: tuple[int, ...]
    action_ids: tuple[int, ...]


class Event80PrequentialSampler:
    """Sample disjoint action chunks and assign disjoint environments to ranks."""

    def __init__(
        self,
        *,
        metadata_path: str | Path,
        active_environment_manifest: str | Path,
        seed: int,
        sequence_length: int = 6,
        environment_key: str = "mu_index",
        action_key: str = "action_id",
    ) -> None:
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.environment_key = str(environment_key)
        self.action_key = str(action_key)
        if self.sequence_length < 2:
            raise ValueError("A prequential stream needs at least two chunks.")

        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        with Path(active_environment_manifest).open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        self.active_environment_ids = tuple(
            int(value) for value in manifest["selection"]["active_environment_ids"]
        )

        grouped: dict[int, list[int]] = {
            environment_id: [] for environment_id in self.active_environment_ids
        }
        for index, row in enumerate(self.rows):
            environment_id = int(row[self.environment_key])
            if environment_id in grouped:
                grouped[environment_id].append(index)
        for environment_id, indices in grouped.items():
            indices.sort(key=lambda index: (int(self.rows[index][self.action_key]), index))
            action_ids = [int(self.rows[index][self.action_key]) for index in indices]
            if len(indices) < self.sequence_length:
                raise ValueError(
                    f"Environment {environment_id} has {len(indices)} chunks; "
                    f"need {self.sequence_length}."
                )
            if len(action_ids) != len(set(action_ids)):
                raise ValueError(
                    f"Environment {environment_id} contains duplicate action ids: {action_ids}."
                )
        self.grouped_indices = grouped

    def sample(
        self,
        *,
        step: int,
        process_index: int,
        num_processes: int,
        environments_per_rank: int = 1,
    ) -> tuple[PrequentialEpisode, ...]:
        global_environment_count = int(num_processes) * int(environments_per_rank)
        if global_environment_count > len(self.active_environment_ids):
            raise ValueError(
                f"Requested {global_environment_count} environments from "
                f"{len(self.active_environment_ids)} active environments."
            )
        rng = random.Random(self.seed + int(step) * 104729)
        selected_environments = rng.sample(
            list(self.active_environment_ids), global_environment_count
        )
        episodes = []
        for environment_id in selected_environments:
            indices = rng.sample(
                self.grouped_indices[environment_id], self.sequence_length
            )
            episodes.append(
                PrequentialEpisode(
                    environment_id=environment_id,
                    indices=tuple(indices),
                    action_ids=tuple(
                        int(self.rows[index][self.action_key]) for index in indices
                    ),
                )
            )
        start = int(process_index) * int(environments_per_rank)
        end = start + int(environments_per_rank)
        return tuple(episodes[start:end])
