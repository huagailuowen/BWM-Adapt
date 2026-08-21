"""Metrics for model-based action selection.

These metrics deliberately operate on a set of candidate rollouts rather than on
one predicted/ground-truth video pair. The world model selects an action from its
predicted outcome; only after selection is the matching ground-truth outcome used
for success and regret.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TargetInterval:
    """Closed one-dimensional target interval measured in metres."""

    target_area_id: str
    low_m: float
    high_m: float

    def __post_init__(self) -> None:
        if not self.target_area_id:
            raise ValueError("target_area_id must be non-empty")
        if not math.isfinite(self.low_m) or not math.isfinite(self.high_m):
            raise ValueError("target bounds must be finite")
        if self.low_m > self.high_m:
            raise ValueError(f"target low_m={self.low_m} exceeds high_m={self.high_m}")

    @property
    def width_m(self) -> float:
        return self.high_m - self.low_m

    def distance_m(self, value_m: float) -> float:
        if not math.isfinite(value_m):
            raise ValueError(f"target outcome must be finite, got {value_m}")
        return max(self.low_m - value_m, 0.0, value_m - self.high_m)

    def contains(self, value_m: float, *, tolerance_m: float = 0.0) -> bool:
        return self.low_m - tolerance_m <= value_m <= self.high_m + tolerance_m


@dataclass(frozen=True)
class CollisionActionCandidate:
    """Predicted and observed outcome for one candidate projectile action."""

    action_id: str
    predicted_target_forward_displacement_m: float
    gt_target_forward_displacement_m: float
    clean_collision: bool
    target_on_table: bool | None = None
    target_in_workspace: bool | None = None
    target_lateral_drift_m: float | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CollisionActionCandidate":
        required = (
            "action_id",
            "predicted_target_forward_displacement_m",
            "gt_target_forward_displacement_m",
            "clean_collision",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"collision candidate is missing fields: {missing}")

        candidate = cls(
            action_id=str(row["action_id"]),
            predicted_target_forward_displacement_m=float(
                row["predicted_target_forward_displacement_m"]
            ),
            gt_target_forward_displacement_m=float(row["gt_target_forward_displacement_m"]),
            clean_collision=_as_bool(row["clean_collision"], "clean_collision"),
            target_on_table=_optional_bool(row.get("target_on_table"), "target_on_table"),
            target_in_workspace=_optional_bool(
                row.get("target_in_workspace"), "target_in_workspace"
            ),
            target_lateral_drift_m=_optional_float(row.get("target_lateral_drift_m")),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        for name, value in (
            (
                "predicted_target_forward_displacement_m",
                self.predicted_target_forward_displacement_m,
            ),
            ("gt_target_forward_displacement_m", self.gt_target_forward_displacement_m),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.target_lateral_drift_m is not None and not math.isfinite(
            self.target_lateral_drift_m
        ):
            raise ValueError("target_lateral_drift_m must be finite when provided")


@dataclass(frozen=True)
class CollisionActionSettings:
    """Ground-truth validity constraints used only after model selection."""

    require_clean_collision: bool = True
    require_on_table: bool = False
    require_in_workspace: bool = False
    lateral_tolerance_m: float | None = None
    success_tolerance_m: float = 0.0
    oracle_tie_tolerance_m: float = 1e-9
    invalid_cost_margin_m: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("lateral_tolerance_m", self.lateral_tolerance_m),
            ("invalid_cost_margin_m", self.invalid_cost_margin_m),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.success_tolerance_m < 0.0 or self.oracle_tie_tolerance_m < 0.0:
            raise ValueError("success and oracle tie tolerances must be non-negative")


COLLISION_ACTION_METRIC_NAMES: tuple[str, ...] = (
    "collision_task_success",
    "collision_oracle_reachable",
    "collision_selected_action_valid",
    "collision_selected_action_is_oracle",
    "collision_selected_predicted_target_error_m",
    "collision_selected_gt_target_error_m",
    "collision_oracle_gt_target_error_m",
    "collision_selected_gt_task_cost_m",
    "collision_regret_m",
    "collision_normalized_regret",
    "collision_candidate_count",
    "collision_valid_candidate_count",
    "collision_successful_candidate_count",
)


def evaluate_collision_action_decision(
    candidates: Sequence[CollisionActionCandidate],
    target: TargetInterval,
    settings: CollisionActionSettings | None = None,
) -> dict[str, Any]:
    """Select by predicted displacement, then score the selected GT rollout.

    Ground-truth validity is never used to choose ``selected``. It is used only
    for scoring and for defining the oracle among actions that are physically
    valid under the requested protocol.
    """

    settings = settings or CollisionActionSettings()
    if not candidates:
        raise ValueError("collision action decision has no candidates")
    for candidate in candidates:
        candidate.validate()
    action_ids = [candidate.action_id for candidate in candidates]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError(f"action_id values must be unique within a decision: {action_ids}")

    predicted_errors = [
        target.distance_m(candidate.predicted_target_forward_displacement_m)
        for candidate in candidates
    ]
    gt_errors = [
        target.distance_m(candidate.gt_target_forward_displacement_m)
        for candidate in candidates
    ]
    valid = [_collision_candidate_is_valid(candidate, settings) for candidate in candidates]
    valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
    if not valid_indices:
        raise ValueError("collision decision contains no valid ground-truth candidate")

    successful = [
        valid[index]
        and target.contains(
            candidate.gt_target_forward_displacement_m,
            tolerance_m=settings.success_tolerance_m,
        )
        for index, candidate in enumerate(candidates)
    ]

    selected_index = min(
        range(len(candidates)), key=lambda index: (predicted_errors[index], action_ids[index])
    )
    oracle_index = min(
        valid_indices, key=lambda index: (gt_errors[index], action_ids[index])
    )

    largest_valid_error = max(gt_errors[index] for index in valid_indices)
    margin = settings.invalid_cost_margin_m
    if margin is None:
        margin = max(target.width_m, 1e-6)
    invalid_cost = max(largest_valid_error, target.width_m) + margin
    task_costs = [gt_errors[index] if valid[index] else invalid_cost for index in range(len(candidates))]

    oracle_cost = task_costs[oracle_index]
    selected_cost = task_costs[selected_index]
    regret = max(0.0, selected_cost - oracle_cost)
    regret_denominator = max(max(task_costs) - oracle_cost, target.width_m, 1e-12)
    normalized_regret = min(1.0, regret / regret_denominator)
    selected_is_oracle = (
        valid[selected_index]
        and selected_cost <= oracle_cost + settings.oracle_tie_tolerance_m
    )

    selected = candidates[selected_index]
    oracle = candidates[oracle_index]
    return {
        "target_area_id": target.target_area_id,
        "target_low_m": target.low_m,
        "target_high_m": target.high_m,
        "target_width_m": target.width_m,
        "selected_action_id": selected.action_id,
        "oracle_action_id": oracle.action_id,
        "selected_predicted_target_forward_displacement_m": (
            selected.predicted_target_forward_displacement_m
        ),
        "selected_gt_target_forward_displacement_m": selected.gt_target_forward_displacement_m,
        "oracle_gt_target_forward_displacement_m": oracle.gt_target_forward_displacement_m,
        "collision_task_success": float(successful[selected_index]),
        "collision_oracle_reachable": float(any(successful)),
        "collision_selected_action_valid": float(valid[selected_index]),
        "collision_selected_action_is_oracle": float(selected_is_oracle),
        "collision_selected_predicted_target_error_m": predicted_errors[selected_index],
        "collision_selected_gt_target_error_m": gt_errors[selected_index],
        "collision_oracle_gt_target_error_m": gt_errors[oracle_index],
        "collision_selected_gt_task_cost_m": selected_cost,
        "collision_regret_m": regret,
        "collision_normalized_regret": normalized_regret,
        "collision_candidate_count": float(len(candidates)),
        "collision_valid_candidate_count": float(sum(valid)),
        "collision_successful_candidate_count": float(sum(successful)),
    }


def summarize_collision_action_decisions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build micro, per-target, per-domain, and macro-over-target summaries."""

    decisions = [dict(row) for row in rows]
    if not decisions:
        raise ValueError("cannot summarize an empty collision action result set")

    by_target: dict[str, list[dict[str, Any]]] = {}
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in decisions:
        by_target.setdefault(str(row["target_area_id"]), []).append(row)
        by_domain.setdefault(str(row.get("domain", "unspecified")), []).append(row)

    target_summaries = {
        key: _mean_metrics(values) for key, values in sorted(by_target.items())
    }
    domain_summaries = {
        key: _mean_metrics(values) for key, values in sorted(by_domain.items())
    }
    macro = {
        metric: _mean([summary[metric] for summary in target_summaries.values()])
        for metric in COLLISION_ACTION_METRIC_NAMES
    }
    return {
        "decision_count": len(decisions),
        "target_area_count": len(target_summaries),
        "micro": _mean_metrics(decisions),
        "macro_over_target_areas": macro,
        "by_target_area": target_summaries,
        "by_domain": domain_summaries,
    }


def _collision_candidate_is_valid(
    candidate: CollisionActionCandidate, settings: CollisionActionSettings
) -> bool:
    if settings.require_clean_collision and not candidate.clean_collision:
        return False
    if settings.require_on_table and candidate.target_on_table is not True:
        return False
    if settings.require_in_workspace and candidate.target_in_workspace is not True:
        return False
    if settings.lateral_tolerance_m is not None:
        if candidate.target_lateral_drift_m is None:
            return False
        if abs(candidate.target_lateral_drift_m) > settings.lateral_tolerance_m:
            return False
    return True


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        metric: _mean([float(row[metric]) for row in rows])
        for metric in COLLISION_ACTION_METRIC_NAMES
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute the mean of an empty sequence")
    return float(sum(values) / len(values))


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be boolean, got {value!r}")


def _optional_bool(value: Any, name: str) -> bool | None:
    return None if value is None else _as_bool(value, name)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class ScalarActionCandidate:
    """One candidate action for a task with a scalar physical outcome."""

    action_id: str
    predicted_outcome: float
    gt_outcome: float
    valid: bool = True

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ScalarActionCandidate":
        required = ("action_id", "predicted_outcome", "gt_outcome")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"scalar action candidate is missing fields: {missing}")
        candidate = cls(
            action_id=str(row["action_id"]),
            predicted_outcome=float(row["predicted_outcome"]),
            gt_outcome=float(row["gt_outcome"]),
            valid=_as_bool(row.get("valid", True), "valid"),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not math.isfinite(self.predicted_outcome) or not math.isfinite(self.gt_outcome):
            raise ValueError("predicted_outcome and gt_outcome must be finite")


@dataclass(frozen=True)
class ScalarActionSettings:
    success_tolerance: float = 0.0
    oracle_tie_tolerance: float = 1e-9
    invalid_cost_margin: float | None = None

    def __post_init__(self) -> None:
        if self.success_tolerance < 0.0 or self.oracle_tie_tolerance < 0.0:
            raise ValueError("scalar action tolerances must be non-negative")
        if self.invalid_cost_margin is not None and self.invalid_cost_margin < 0.0:
            raise ValueError("invalid_cost_margin must be non-negative")


SCALAR_ACTION_METRIC_NAMES: tuple[str, ...] = (
    "action_task_success",
    "action_oracle_reachable",
    "action_selected_valid",
    "action_selected_is_oracle",
    "action_selected_predicted_target_error",
    "action_selected_gt_target_error",
    "action_oracle_gt_target_error",
    "action_selected_gt_task_cost",
    "action_regret",
    "action_normalized_regret",
    "action_candidate_count",
    "action_valid_candidate_count",
    "action_successful_candidate_count",
)


def evaluate_scalar_action_decision(
    candidates: Sequence[ScalarActionCandidate],
    target: TargetInterval,
    settings: ScalarActionSettings | None = None,
) -> dict[str, Any]:
    """Select an action from predicted scalar outcomes and score it with GT."""

    settings = settings or ScalarActionSettings()
    if not candidates:
        raise ValueError("scalar action decision has no candidates")
    for candidate in candidates:
        candidate.validate()
    action_ids = [candidate.action_id for candidate in candidates]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError(f"action_id values must be unique within a decision: {action_ids}")

    predicted_errors = [target.distance_m(candidate.predicted_outcome) for candidate in candidates]
    gt_errors = [target.distance_m(candidate.gt_outcome) for candidate in candidates]
    valid_indices = [index for index, candidate in enumerate(candidates) if candidate.valid]
    if not valid_indices:
        raise ValueError("scalar action decision contains no valid GT candidate")
    successful = [
        candidate.valid
        and target.contains(candidate.gt_outcome, tolerance_m=settings.success_tolerance)
        for candidate in candidates
    ]

    selected_index = min(
        range(len(candidates)), key=lambda index: (predicted_errors[index], action_ids[index])
    )
    oracle_index = min(
        valid_indices, key=lambda index: (gt_errors[index], action_ids[index])
    )
    largest_valid_error = max(gt_errors[index] for index in valid_indices)
    margin = settings.invalid_cost_margin
    if margin is None:
        margin = max(target.width_m, 1e-6)
    invalid_cost = max(largest_valid_error, target.width_m) + margin
    task_costs = [
        gt_errors[index] if candidates[index].valid else invalid_cost
        for index in range(len(candidates))
    ]
    oracle_cost = task_costs[oracle_index]
    selected_cost = task_costs[selected_index]
    regret = max(0.0, selected_cost - oracle_cost)
    denominator = max(max(task_costs) - oracle_cost, target.width_m, 1e-12)

    return {
        "target_area_id": target.target_area_id,
        "target_low": target.low_m,
        "target_high": target.high_m,
        "selected_action_id": candidates[selected_index].action_id,
        "oracle_action_id": candidates[oracle_index].action_id,
        "selected_predicted_outcome": candidates[selected_index].predicted_outcome,
        "selected_gt_outcome": candidates[selected_index].gt_outcome,
        "oracle_gt_outcome": candidates[oracle_index].gt_outcome,
        "action_task_success": float(successful[selected_index]),
        "action_oracle_reachable": float(any(successful)),
        "action_selected_valid": float(candidates[selected_index].valid),
        "action_selected_is_oracle": float(
            candidates[selected_index].valid
            and selected_cost <= oracle_cost + settings.oracle_tie_tolerance
        ),
        "action_selected_predicted_target_error": predicted_errors[selected_index],
        "action_selected_gt_target_error": gt_errors[selected_index],
        "action_oracle_gt_target_error": gt_errors[oracle_index],
        "action_selected_gt_task_cost": selected_cost,
        "action_regret": regret,
        "action_normalized_regret": min(1.0, regret / denominator),
        "action_candidate_count": float(len(candidates)),
        "action_valid_candidate_count": float(len(valid_indices)),
        "action_successful_candidate_count": float(sum(successful)),
    }


@dataclass(frozen=True)
class LightSwitchActionCandidate:
    """One red/blue button candidate for a requested final lamp state."""

    action_id: str
    button_color: str
    predicted_final_light_on_probability: float
    gt_final_light_on: bool
    valid_press: bool = True

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "LightSwitchActionCandidate":
        required = (
            "action_id",
            "button_color",
            "predicted_final_light_on_probability",
            "gt_final_light_on",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"LightSwitch candidate is missing fields: {missing}")
        candidate = cls(
            action_id=str(row["action_id"]),
            button_color=str(row["button_color"]).strip().lower(),
            predicted_final_light_on_probability=float(
                row["predicted_final_light_on_probability"]
            ),
            gt_final_light_on=_as_bool(row["gt_final_light_on"], "gt_final_light_on"),
            valid_press=_as_bool(row.get("valid_press", True), "valid_press"),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if self.button_color not in {"red", "blue"}:
            raise ValueError(f"button_color must be red or blue, got {self.button_color!r}")
        probability = self.predicted_final_light_on_probability
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("predicted_final_light_on_probability must lie in [0, 1]")


LIGHTSWITCH_ACTION_METRIC_NAMES: tuple[str, ...] = (
    "action_task_success",
    "action_oracle_reachable",
    "action_selected_valid",
    "action_selected_is_oracle",
    "action_selected_predicted_target_error",
    "action_selected_gt_target_error",
    "action_regret",
    "action_normalized_regret",
    "action_candidate_count",
    "action_valid_candidate_count",
    "action_successful_candidate_count",
)


def evaluate_lightswitch_action_decision(
    candidates: Sequence[LightSwitchActionCandidate],
    *,
    target_area_id: str,
    desired_light_on: bool,
) -> dict[str, Any]:
    """Choose a button from predicted lamp state and score the observed state."""

    if not candidates:
        raise ValueError("LightSwitch action decision has no candidates")
    for candidate in candidates:
        candidate.validate()
    action_ids = [candidate.action_id for candidate in candidates]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError(f"action_id values must be unique within a decision: {action_ids}")

    target_value = float(desired_light_on)
    predicted_errors = [
        abs(candidate.predicted_final_light_on_probability - target_value)
        for candidate in candidates
    ]
    gt_errors = [float(candidate.gt_final_light_on != desired_light_on) for candidate in candidates]
    valid_indices = [index for index, candidate in enumerate(candidates) if candidate.valid_press]
    if not valid_indices:
        raise ValueError("LightSwitch decision contains no valid button press")
    successful = [
        candidate.valid_press and candidate.gt_final_light_on == desired_light_on
        for candidate in candidates
    ]
    selected_index = min(
        range(len(candidates)), key=lambda index: (predicted_errors[index], action_ids[index])
    )
    oracle_index = min(
        valid_indices, key=lambda index: (gt_errors[index], action_ids[index])
    )
    task_costs = [gt_errors[index] if candidates[index].valid_press else 2.0 for index in range(len(candidates))]
    regret = max(0.0, task_costs[selected_index] - task_costs[oracle_index])

    return {
        "target_area_id": target_area_id,
        "desired_light_on": desired_light_on,
        "selected_action_id": candidates[selected_index].action_id,
        "selected_button_color": candidates[selected_index].button_color,
        "oracle_action_id": candidates[oracle_index].action_id,
        "oracle_button_color": candidates[oracle_index].button_color,
        "action_task_success": float(successful[selected_index]),
        "action_oracle_reachable": float(any(successful)),
        "action_selected_valid": float(candidates[selected_index].valid_press),
        "action_selected_is_oracle": float(
            candidates[selected_index].valid_press
            and task_costs[selected_index] == task_costs[oracle_index]
        ),
        "action_selected_predicted_target_error": predicted_errors[selected_index],
        "action_selected_gt_target_error": gt_errors[selected_index],
        "action_regret": regret,
        "action_normalized_regret": min(1.0, regret),
        "action_candidate_count": float(len(candidates)),
        "action_valid_candidate_count": float(len(valid_indices)),
        "action_successful_candidate_count": float(sum(successful)),
    }


def summarize_action_decisions(
    rows: Iterable[Mapping[str, Any]], metric_names: Sequence[str]
) -> dict[str, Any]:
    """Aggregate action decisions globally, by target, and by domain."""

    decisions = [dict(row) for row in rows]
    if not decisions:
        raise ValueError("cannot summarize an empty action result set")
    by_target: dict[str, list[dict[str, Any]]] = {}
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in decisions:
        by_target.setdefault(str(row["target_area_id"]), []).append(row)
        by_domain.setdefault(str(row.get("domain", "unspecified")), []).append(row)

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {
            metric: _mean([float(row[metric]) for row in group])
            for metric in metric_names
        }

    target_summaries = {key: summarize(value) for key, value in sorted(by_target.items())}
    return {
        "decision_count": len(decisions),
        "target_area_count": len(target_summaries),
        "micro": summarize(decisions),
        "macro_over_target_areas": {
            metric: _mean([summary[metric] for summary in target_summaries.values()])
            for metric in metric_names
        },
        "by_target_area": target_summaries,
        "by_domain": {key: summarize(value) for key, value in sorted(by_domain.items())},
    }
