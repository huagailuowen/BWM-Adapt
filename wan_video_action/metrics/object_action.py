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
        return {
            "id": self.target_id,
            "region": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
        }


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
        key=lambda record: (target.distance(record["selection_xy"]), str(record["action_id"]))
    )
    ground_truth.sort(
        key=lambda record: (target.distance(record["ground_truth_xy"]), str(record["action_id"]))
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

    selected = selectable[0]
    selected_action = str(selected["action_id"])
    selected_gt = next(
        (record for record in records if str(record["action_id"]) == selected_action), None
    )
    selected_gt_xy = None if selected_gt is None else _point(selected_gt.get("ground_truth_xy"))
    selected_gt_distance = target.distance(selected_gt_xy)

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
            "task_success": selected_gt_xy is not None and target.contains(selected_gt_xy),
        }
    )

    if ground_truth:
        oracle = ground_truth[0]
        oracle_distance = target.distance(oracle["ground_truth_xy"])
        result.update(
            {
                "oracle_action_id": str(oracle["action_id"]),
                "oracle_sample_index": oracle.get("sample_index"),
                "oracle_ground_truth_xy": list(_point(oracle["ground_truth_xy"])),
                "oracle_ground_truth_distance": oracle_distance,
                "oracle_reachable": target.contains(oracle["ground_truth_xy"]),
                "selected_is_oracle": selected_action == str(oracle["action_id"]),
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
            }
        )
    return result
