"""Shared contracts for world-model methods.

These contracts deliberately do not import the existing training scripts. They
provide a stable boundary that those scripts can adopt after the experiment
protocols are finalized.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable


class MethodFamily(str, Enum):
    OURS = "ours"
    BASELINE = "baseline"
    ABLATION = "ablation"


class AdaptationTarget(str, Enum):
    NONE = "none"
    ENVIRONMENT_CODE = "environment_code"
    HISTORY = "history"
    LORA = "lora"
    FAST_WEIGHTS = "fast_weights"
    AMORTIZED_CODE = "amortized_code"


class QueryStatePolicy(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"


@dataclass(frozen=True)
class MethodSpec:
    """Protocol-level definition of one comparison method."""

    slug: str
    display_name: str
    family: MethodFamily
    summary: str
    training_protocol: str
    inference_protocol: str
    adaptation_target: AdaptationTarget
    query_state_policy: QueryStatePolicy
    requires_grouped_training: bool
    parent_slug: str | None = None
    invariants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.slug or self.slug != self.slug.lower() or " " in self.slug:
            raise ValueError("Method slug must be non-empty, lowercase, and space-free.")
        if not self.display_name or not self.summary:
            raise ValueError("Method display name and summary are required.")
        if self.family is MethodFamily.ABLATION and not self.parent_slug:
            raise ValueError("Every ablation must name its parent method.")
        if (
            self.adaptation_target is AdaptationTarget.NONE
            and self.query_state_policy is not QueryStatePolicy.NONE
        ):
            raise ValueError("A stateless method cannot expose query adaptation state.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "family": self.family.value,
            "summary": self.summary,
            "training_protocol": self.training_protocol,
            "inference_protocol": self.inference_protocol,
            "adaptation_target": self.adaptation_target.value,
            "query_state_policy": self.query_state_policy.value,
            "requires_grouped_training": self.requires_grouped_training,
            "parent_slug": self.parent_slug,
            "invariants": list(self.invariants),
        }


SampleT = TypeVar("SampleT")
SupportT = TypeVar("SupportT")
QueryT = TypeVar("QueryT")
TrainingBatchT = TypeVar("TrainingBatchT")
AdaptationStateT = TypeVar("AdaptationStateT")
QueryBatchT = TypeVar("QueryBatchT")
PredictionT = TypeVar("PredictionT")


@dataclass(frozen=True)
class EnvironmentGroup(Generic[SampleT]):
    """Samples that share one training-time environment identity."""

    environment_id: Hashable
    samples: Sequence[SampleT]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("An environment group must contain at least one sample.")


@dataclass(frozen=True)
class GroupedTrainingBatch(Generic[SampleT]):
    """Training-time grouped batch, independent of test-time support size K."""

    groups: Sequence[EnvironmentGroup[SampleT]]

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("A grouped training batch must contain an environment group.")
        environment_ids = [group.environment_id for group in self.groups]
        if len(set(environment_ids)) != len(environment_ids):
            raise ValueError("Environment IDs must be unique within a grouped batch.")


@dataclass(frozen=True)
class SupportQueryEpisode(Generic[SupportT, QueryT]):
    """One adaptation episode with disjoint support and query trajectories."""

    environment_id: Hashable
    supports: Sequence[SupportT]
    queries: Sequence[QueryT]
    support_ids: Sequence[str] = ()
    query_ids: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.supports or not self.queries:
            raise ValueError("An adaptation episode requires support and query samples.")
        if self.support_ids and len(self.support_ids) != len(self.supports):
            raise ValueError("support_ids must align with supports.")
        if self.query_ids and len(self.query_ids) != len(self.queries):
            raise ValueError("query_ids must align with queries.")
        if self.support_ids and self.query_ids:
            overlap = set(self.support_ids).intersection(self.query_ids)
            if overlap:
                raise ValueError(f"Support/query leakage detected for IDs: {sorted(overlap)}")


@runtime_checkable
class MethodRunner(
    Protocol[TrainingBatchT, AdaptationStateT, QueryBatchT, PredictionT]
):
    """Execution boundary for a future method implementation.

    If ``spec.query_state_policy`` is ``READ_ONLY``, ``predict_query`` must not
    mutate the supplied adaptation state. Outer training may backpropagate a
    query loss through support-time writes, but query tokens must not write.
    """

    @property
    def spec(self) -> MethodSpec:
        ...

    def outer_training_step(self, batch: TrainingBatchT) -> Mapping[str, Any]:
        ...

    def initialize_adaptation_state(
        self, *, environment_id: Hashable | None = None
    ) -> AdaptationStateT:
        ...

    def adapt_support(
        self, state: AdaptationStateT, supports: Sequence[Any]
    ) -> AdaptationStateT:
        ...

    def predict_query(
        self, state: AdaptationStateT, queries: QueryBatchT
    ) -> PredictionT:
        ...
