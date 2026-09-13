"""Opt-in per-episode window balancing; the legacy path consumes no extra RNG."""

from __future__ import annotations

import math


class WindowIndexSelector:
    def __init__(
        self, rows, *, mode="legacy", kind_field="sampling_kind",
        preferred_kind="precise", preferred_probability=0.5,
    ):
        if mode not in ("legacy", "uniform_episode_then_window"):
            raise ValueError(f"Unknown window sampling mode: {mode}")
        self.rows = rows
        self.enabled = mode != "legacy"
        self.kind_field = str(kind_field)
        self.preferred_kind = str(preferred_kind)
        self.probability = float(preferred_probability)
        if self.enabled and (
            not math.isfinite(self.probability) or not 0 <= self.probability <= 1
        ):
            raise ValueError("preferred_probability must be in [0, 1].")

    def choose(self, rng, indices):
        if not self.enabled:
            # This is exactly the original call, including its RNG consumption.
            return rng.choice(indices)
        if not indices:
            raise ValueError("An episode has no candidate windows; never prune it silently.")
        preferred, other = [], []
        for index in indices:
            row = self.rows[index]
            if self.kind_field not in row:
                raise ValueError(f"Window {index} is missing {self.kind_field!r}.")
            pool = preferred if row[self.kind_field] == self.preferred_kind else other
            pool.append(index)
        if preferred and other:
            pool = preferred if rng.random() < self.probability else other
        else:
            # Match the grouped real-data sampler: retain short episodes even
            # when only one of the annotated branches exists.
            pool = preferred or other
        return rng.choice(pool)
