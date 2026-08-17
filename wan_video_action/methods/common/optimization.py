"""Serializable staged learning-rate schedules for test-time state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LearningRateStage:
    steps: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("Stage steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Stage learning rate must be positive.")


@dataclass(frozen=True)
class StagedLearningRateSchedule:
    stages: tuple[LearningRateStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("At least one learning-rate stage is required.")

    @classmethod
    def from_config(cls, values: list[dict[str, Any]]) -> "StagedLearningRateSchedule":
        return cls(
            tuple(
                LearningRateStage(
                    steps=int(value["steps"]),
                    learning_rate=float(value["learning_rate"]),
                )
                for value in values
            )
        )

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.stages)

    def learning_rate_at(self, zero_based_step: int) -> float:
        if zero_based_step < 0 or zero_based_step >= self.total_steps:
            raise IndexError(f"Step outside schedule: {zero_based_step}")
        cursor = zero_based_step
        for stage in self.stages:
            if cursor < stage.steps:
                return stage.learning_rate
            cursor -= stage.steps
        raise AssertionError("Unreachable staged schedule state.")

    def expand(self) -> tuple[float, ...]:
        return tuple(
            learning_rate
            for stage in self.stages
            for learning_rate in (stage.learning_rate,) * stage.steps
        )
