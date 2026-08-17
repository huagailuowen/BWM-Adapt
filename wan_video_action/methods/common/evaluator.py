"""Method-agnostic disjoint support/query inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .runtime import InferenceBundle


@dataclass(frozen=True)
class InferenceResult:
    completed_episodes: int


class SupportQueryEvaluator:
    def __init__(self, bundle: InferenceBundle) -> None:
        self.bundle = bundle

    def run(self) -> InferenceResult:
        completed = 0
        for episode in self.bundle.episodes:
            state = self.bundle.runner.initialize_adaptation_state(
                environment_id=episode.environment_id
            )
            state = self.bundle.runner.adapt_support(state, episode.supports)
            predictions = self.bundle.runner.predict_query(state, episode.queries)
            self.bundle.prediction_writer(episode, predictions)
            completed += 1
        return InferenceResult(completed_episodes=completed)
