"""Deterministic Event80 environment/action indexing for TTT episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class SupportQueryEpisode:
    environment_id: int
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    support_action_ids: tuple[int, ...]
    query_action_ids: tuple[int, ...]


class Event80Index:
    """Groups metadata rows by friction environment and action identity."""

    def __init__(
        self,
        metadata_path: str | Path,
        environment_key: str = "mu_index",
        action_key: str = "action_id",
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.environment_key = environment_key
        self.action_key = action_key
        self.rows = []
        self.by_environment: dict[int, dict[int, int]] = {}
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                environment_id = int(row[environment_key])
                action_id = int(row[action_key])
                action_map = self.by_environment.setdefault(environment_id, {})
                if action_id in action_map:
                    raise ValueError(
                        f"Duplicate action={action_id} in environment={environment_id}"
                    )
                action_map[action_id] = row_index
                self.rows.append(row)

    @property
    def environment_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.by_environment))

    def sample_episode(
        self,
        environment_id: int,
        support_size: int,
        query_count: int,
        rng: random.Random,
    ) -> SupportQueryEpisode:
        actions = sorted(self.by_environment[int(environment_id)])
        required = int(support_size) + int(query_count)
        if required > len(actions):
            raise ValueError(
                f"Environment {environment_id} has {len(actions)} actions, needs {required}"
            )
        selected = rng.sample(actions, required)
        support_actions = tuple(selected[:support_size])
        query_actions = tuple(selected[support_size:])
        mapping = self.by_environment[int(environment_id)]
        return SupportQueryEpisode(
            environment_id=int(environment_id),
            support_indices=tuple(mapping[action] for action in support_actions),
            query_indices=tuple(mapping[action] for action in query_actions),
            support_action_ids=support_actions,
            query_action_ids=query_actions,
        )

    def evenly_spaced_environment_order(self) -> tuple[int, ...]:
        """Nested deterministic order for the 5-at-a-time progressive stream."""

        remaining = list(self.environment_ids)
        order = []
        while remaining:
            count = min(5, len(remaining))
            if count == 1:
                positions = [0]
            else:
                positions = sorted(
                    set(round(i * (len(remaining) - 1) / (count - 1)) for i in range(count))
                )
            chosen = [remaining[position] for position in positions]
            order.extend(chosen)
            chosen_set = set(chosen)
            remaining = [value for value in remaining if value not in chosen_set]
        return tuple(order)
