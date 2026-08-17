"""Method-agnostic outer-loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import TrainingBundle


@dataclass(frozen=True)
class TrainingResult:
    completed_steps: int
    final_metrics: dict[str, Any]


class MethodTrainer:
    def __init__(self, bundle: TrainingBundle) -> None:
        self.bundle = bundle

    def run(self) -> TrainingResult:
        completed_steps = 0
        final_metrics: dict[str, Any] = {}
        for step, batch in enumerate(self.bundle.batches, start=1):
            if step > self.bundle.max_steps:
                break
            final_metrics = dict(self.bundle.runner.outer_training_step(batch))
            completed_steps = step
            for callback in self.bundle.callbacks:
                callback(step, final_metrics)
        return TrainingResult(
            completed_steps=completed_steps,
            final_metrics=final_metrics,
        )
