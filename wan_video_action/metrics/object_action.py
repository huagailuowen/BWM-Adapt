"""Prediction-only action selection from object-centric rollout outcomes.

The selector in this module never reads ground-truth outcomes. Ground truth is
used only after the selected action has been frozen, to score task success and
regret against the best action available in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, Iterable, Mapping, Sequence


Point = tuple[float, float]


def _point(value: Any) -> Point | None:
    if value is None or not isinstance(value, Sequence) or len(value) != 2:
        return None
    x, y = float(value[0]), float(value[1])
    return (x, y) if isfinite(x) and isfinite(y) else None


@dataclass(frozen=True)
class RectangleTarget:
    """Axis-aligned target region in normalized image coordinates."""

    target_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    selection_strategy: str = "nearest"
    success_policy: str = "target_reach"
    max_action_steps_above_minimum: int = 0

    def __post_init__(self) -> None:
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError(f"Invalid rectangle for target {self.target_id!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RectangleTarget":
        region = value.get("region", value)
        return cls(
            target_id=str(value["id"]),
            x_min=float(region["x_min"]),
            x_max=float(region["x_max"]),
            y_min=float(region["y_min"]),
            y_max=float(region["y_max"]),
            selection_strategy=str(value.get("selection_strategy", "nearest")),
            success_policy=str(value.get("success_policy", "target_reach")),
            max_action_steps_above_minimum=int(
                value.get("max_action_steps_above_minimum", 0)
            ),
        )

    def distance(self, value: Any) -> float:
        point = _point(value)
        if point is None:
            return float("inf")
        x, y = point
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        return hypot(dx, dy)

    def contains(self, value: Any) -> bool:
        return self.distance(value) == 0.0

    def as_dict(self) -> dict[str, Any]:
        output = {
            "id": self.target_id,
            "region": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
        }
        if self.selection_strategy != "nearest":
            output["selection_strategy"] = self.selection_strategy
        if self.success_policy != "target_reach":
            output["success_policy"] = self.success_policy
            output["max_action_steps_above_minimum"] = (
                self.max_action_steps_above_minimum
            )
        return output


def _action_sort_key(record: Mapping[str, Any]) -> tuple[int, float | str]:
    value = record.get("action_order", record.get("action_id"))
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(record["action_id"]))


def evaluate_action_choice(
    candidates: Iterable[Mapping[str, Any]],
    target: RectangleTarget,
    *,
    expected_action_count: int | None = None,
) -> dict[str, Any]:
    """Select from predictions, then score the frozen action with GT.

    Candidate records must contain ``action_id`` and may contain
    ``selection_xy`` and ``ground_truth_xy``. ``selection_xy`` is either a
    model-predicted outcome or an already observed support outcome. Ground
    truth for a query action is never used during selection.
    """

    records = [dict(record) for record in candidates]
    action_ids = [str(record["action_id"]) for record in records]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Each environment must contain at most one candidate per action_id")

    selectable = [record for record in records if _point(record.get("selection_xy")) is not None]
    ground_truth = [record for record in records if _point(record.get("ground_truth_xy")) is not None]
    selectable.sort(
        key=lambda record: (target.distance(record["selection_xy"]), _action_sort_key(record))
    )
    ground_truth.sort(
        key=lambda record: (target.distance(record["ground_truth_xy"]), _action_sort_key(record))
    )

    available = len(selectable)
    protocol_count = len(records)
    expected = protocol_count if expected_action_count is None else int(expected_action_count)
    result: dict[str, Any] = {
        "target": target.as_dict(),
        "expected_action_count": expected,
        "protocol_action_count": protocol_count,
        "candidate_outcome_count": available,
        "ground_truth_action_count": len(ground_truth),
        "action_coverage": available / protocol_count if protocol_count else 0.0,
        "candidate_set_complete": available == protocol_count,
        "missing_candidate_outcome_action_ids": sorted(
            set(action_ids) - {str(record["action_id"]) for record in selectable}
        ),
    }
    if not selectable:
        result["status"] = "no_valid_candidate_outcomes"
        return result

    selection_details: dict[str, Any] = {}
    if target.selection_strategy == "nearest":
        selected = selectable[0]
    elif target.selection_strategy == "first_reaching":
        ordered_selectable = sorted(selectable, key=_action_sort_key)
        predicted_reaching = [
            record
            for record in ordered_selectable
            if target.contains(record.get("selection_xy"))
        ]
        selected = predicted_reaching[0] if predicted_reaching else selectable[0]
        selection_details = {
            "predicted_reaching_action_ids": [
                str(record["action_id"]) for record in predicted_reaching
            ],
            "predicted_minimum_reaching_action_id": (
                str(predicted_reaching[0]["action_id"])
                if predicted_reaching
                else None
            ),
            "used_nearest_fallback": not predicted_reaching,
        }
    else:
        raise ValueError(f"Unknown selection strategy {target.selection_strategy!r}")
    selected_action = str(selected["action_id"])
    selected_gt = next(
        (record for record in records if str(record["action_id"]) == selected_action), None
    )
    selected_gt_xy = None if selected_gt is None else _point(selected_gt.get("ground_truth_xy"))
    selected_gt_distance = target.distance(selected_gt_xy)

    ordered_ground_truth = sorted(ground_truth, key=_action_sort_key)
    ground_truth_rank = {
        str(record["action_id"]): rank
        for rank, record in enumerate(ordered_ground_truth)
    }
    target_reaching = [
        record
        for record in ordered_ground_truth
        if target.contains(record.get("ground_truth_xy"))
    ]
    minimum_reaching = target_reaching[0] if target_reaching else None
    if target.max_action_steps_above_minimum < 0:
        raise ValueError("max_action_steps_above_minimum must be non-negative")
    if target.success_policy == "target_reach":
        policy_feasible = target_reaching
    elif target.success_policy == "minimum_reaching":
        if minimum_reaching is None:
            policy_feasible = []
        else:
            minimum_rank = ground_truth_rank[str(minimum_reaching["action_id"])]
            policy_feasible = [
                record
                for record in target_reaching
                if 0
                <= ground_truth_rank[str(record["action_id"])] - minimum_rank
                <= target.max_action_steps_above_minimum
            ]
    else:
        raise ValueError(f"Unknown success policy {target.success_policy!r}")
    policy_feasible_ids = {
        str(record["action_id"]) for record in policy_feasible
    }

    result.update(
        {
            "status": "ok" if selected_gt_xy is not None else "selected_action_missing_gt",
            "selected_action_id": selected_action,
            "selected_sample_index": selected.get("sample_index"),
            "selected_outcome_source": selected.get("selection_source", "model_prediction"),
            "selected_candidate_xy": list(_point(selected["selection_xy"])),
            "selected_candidate_distance": target.distance(selected["selection_xy"]),
            "selected_ground_truth_xy": None if selected_gt_xy is None else list(selected_gt_xy),
            "selected_ground_truth_distance": selected_gt_distance,
            "selection_strategy": target.selection_strategy,
            "selection_details": selection_details,
            "success_policy": target.success_policy,
            "task_success": selected_action in policy_feasible_ids,
            "target_reaching_action_ids": [
                str(record["action_id"]) for record in target_reaching
            ],
            "oracle_feasible_action_ids": [
                str(record["action_id"]) for record in policy_feasible
            ],
            "minimum_reaching_action_id": (
                str(minimum_reaching["action_id"])
                if minimum_reaching is not None
                else None
            ),
        }
    )

    if ground_truth:
        oracle = (
            minimum_reaching
            if target.success_policy == "minimum_reaching" and minimum_reaching is not None
            else ground_truth[0]
        )
        oracle_distance = target.distance(oracle["ground_truth_xy"])
        result.update(
            {
                "oracle_action_id": str(oracle["action_id"]),
                "oracle_sample_index": oracle.get("sample_index"),
                "oracle_ground_truth_xy": list(_point(oracle["ground_truth_xy"])),
                "oracle_ground_truth_distance": oracle_distance,
                "oracle_reachable": target.contains(oracle["ground_truth_xy"]),
                "selected_is_oracle": selected_action == str(oracle["action_id"]),
                "selected_action_offset_from_minimum_steps": (
                    ground_truth_rank[selected_action]
                    - ground_truth_rank[str(minimum_reaching["action_id"])]
                    if selected_action in ground_truth_rank and minimum_reaching is not None
                    else None
                ),
                "regret": (
                    selected_gt_distance - oracle_distance
                    if isfinite(selected_gt_distance)
                    else None
                ),
            }
        )
    else:
        result.update(
            {
                "oracle_action_id": None,
                "oracle_sample_index": None,
                "oracle_ground_truth_xy": None,
                "oracle_ground_truth_distance": None,
                "oracle_reachable": False,
                "selected_is_oracle": False,
                "regret": None,
                "selected_action_offset_from_minimum_steps": None,
            }
        )
    return result
