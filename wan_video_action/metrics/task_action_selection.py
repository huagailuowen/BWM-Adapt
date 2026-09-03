"""Task-state targets and prediction-only action selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isclose, isfinite
from typing import Any, Iterable, Mapping, Sequence


def _scalar(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    x, y = _scalar(value[0]), _scalar(value[1])
    return None if x is None or y is None else (x, y)


def _action_sort_key(record: Mapping[str, Any]) -> tuple[int, float | str]:
    order = _scalar(record.get("action_order"))
    return (0, order) if order is not None else (1, str(record["action_id"]))


def _for_target_object(
    record: Mapping[str, Any], target: "TaskTarget"
) -> dict[str, Any]:
    output = dict(record)
    raw_index = target.parameters.get("object_index")
    if raw_index is None:
        return output
    index = str(int(raw_index))
    for value_key in ("selection_value", "ground_truth_value"):
        object_values = output.get(value_key.replace("_value", "_object_values"))
        if isinstance(object_values, Mapping):
            output[value_key] = object_values.get(
                index,
                object_values.get(int(index)),
            )
        else:
            output[value_key] = None
    output["evaluated_object_index"] = int(index)
    return output


def _boundary_crossing_choice(
    records: list[dict[str, Any]],
    target: "TaskTarget",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if target.kind != "scalar_interval":
        raise ValueError("boundary_crossing requires a scalar_interval target.")
    ordered = sorted(records, key=_action_sort_key)
    orders = [_scalar(record.get("action_order")) for record in ordered]
    if any(value is None for value in orders):
        raise ValueError("boundary_crossing requires numeric action_order values.")
    if len(set(orders)) != len(orders):
        raise ValueError("boundary_crossing requires unique action_order values.")

    lower = float(target.parameters["min"])
    upper = float(target.parameters["max"])
    center = 0.5 * (lower + upper)
    boundaries: list[dict[str, Any]] = []
    distance_to_boundary: dict[str, float] = {}
    for left, right, left_order, right_order in zip(
        ordered, ordered[1:], orders, orders[1:]
    ):
        left_value = _scalar(left.get("selection_value"))
        right_value = _scalar(right.get("selection_value"))
        assert left_value is not None and right_value is not None
        left_delta = left_value - center
        right_delta = right_value - center
        if not (
            isclose(left_delta, 0.0, abs_tol=1e-12)
            or isclose(right_delta, 0.0, abs_tol=1e-12)
            or left_delta * right_delta < 0.0
        ):
            continue
        denominator = right_delta - left_delta
        if isclose(denominator, 0.0, abs_tol=1e-12):
            root = 0.5 * (left_order + right_order)
        else:
            root = left_order - left_delta * (right_order - left_order) / denominator
        boundary = {
            "left_action_id": str(left["action_id"]),
            "right_action_id": str(right["action_id"]),
            "left_action_order": left_order,
            "right_action_order": right_order,
            "left_value": left_value,
            "right_value": right_value,
            "interpolated_action_order": float(root),
        }
        boundaries.append(boundary)
        for record, order in ((left, left_order), (right, right_order)):
            action_id = str(record["action_id"])
            distance = abs(order - root)
            distance_to_boundary[action_id] = min(
                distance_to_boundary.get(action_id, float("inf")), distance
            )

    def score(record: Mapping[str, Any]) -> tuple[float, int, float, float, float]:
        action_id = str(record["action_id"])
        value = _scalar(record.get("selection_value"))
        order = _scalar(record.get("action_order"))
        assert value is not None and order is not None
        boundary_distance = distance_to_boundary.get(action_id, float("inf"))
        return (
            target.distance(value),
            0 if isfinite(boundary_distance) else 1,
            boundary_distance,
            abs(value - center),
            order,
        )

    selected = min(ordered, key=score)
    details = {
        "predicted_boundary_count": len(boundaries),
        "predicted_boundaries": boundaries,
        "predicted_feasible_action_ids": [
            str(record["action_id"])
            for record in ordered
            if target.contains(record.get("selection_value"))
        ],
        "selected_boundary_distance": distance_to_boundary.get(
            str(selected["action_id"])
        ),
    }
    return selected, details


def _first_reaching_choice(
    records: list[dict[str, Any]],
    target: "TaskTarget",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the weakest ordered action predicted to reach the target."""

    ordered = sorted(records, key=_action_sort_key)
    predicted_reaching = [
        record for record in ordered if target.contains(record.get("selection_value"))
    ]
    if predicted_reaching:
        selected = predicted_reaching[0]
        fallback = False
    else:
        selected = min(
            ordered,
            key=lambda record: (
                target.distance(record.get("selection_value")),
                _action_sort_key(record),
            ),
        )
        fallback = True
    return selected, {
        "predicted_reaching_action_ids": [
            str(record["action_id"]) for record in predicted_reaching
        ],
        "predicted_minimum_reaching_action_id": (
            str(predicted_reaching[0]["action_id"]) if predicted_reaching else None
        ),
        "used_nearest_fallback": fallback,
    }


@dataclass(frozen=True)
class TaskTarget:
    target_id: str
    kind: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskTarget":
        return cls(
            target_id=str(value["id"]),
            kind=str(value["kind"]),
            parameters=dict(value),
        )

    def valid(self, value: Any) -> bool:
        if self.kind == "point_rectangle":
            return _point(value) is not None
        if self.kind == "scalar_interval":
            return _scalar(value) is not None
        if self.kind == "binary":
            return isinstance(value, (bool, int)) and int(value) in (0, 1)
        raise ValueError(f"Unknown target kind {self.kind!r}.")

    def distance(self, value: Any) -> float:
        if not self.valid(value):
            return float("inf")
        if self.kind == "point_rectangle":
            x, y = _point(value)
            region = self.parameters["region"]
            dx = max(float(region["x_min"]) - x, 0.0, x - float(region["x_max"]))
            dy = max(float(region["y_min"]) - y, 0.0, y - float(region["y_max"]))
            return hypot(dx, dy)
        if self.kind == "scalar_interval":
            scalar = _scalar(value)
            lower, upper = float(self.parameters["min"]), float(self.parameters["max"])
            return max(lower - scalar, 0.0, scalar - upper)
        return 0.0 if bool(value) == bool(self.parameters["value"]) else 1.0

    def contains(self, value: Any) -> bool:
        return self.distance(value) == 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.parameters)


def evaluate_task_action_choice(
    candidates: Iterable[Mapping[str, Any]],
    target: TaskTarget,
    *,
    selection_strategy: str = "nearest",
) -> dict[str, Any]:
    """Choose using candidate outcomes, then score that action with GT."""

    records = [_for_target_object(record, target) for record in candidates]
    action_ids = [str(record["action_id"]) for record in records]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Action candidates must be aggregated to unique action_id values.")

    selectable = [record for record in records if target.valid(record.get("selection_value"))]
    ground_truth = [record for record in records if target.valid(record.get("ground_truth_value"))]
    ground_truth.sort(
        key=lambda record: (target.distance(record["ground_truth_value"]), _action_sort_key(record))
    )
    available = len(selectable)
    result: dict[str, Any] = {
        "target": target.as_dict(),
        "protocol_action_count": len(records),
        "candidate_outcome_count": available,
        "ground_truth_action_count": len(ground_truth),
        "action_coverage": available / len(records) if records else 0.0,
        "candidate_set_complete": available == len(records),
        "missing_candidate_outcome_action_ids": sorted(
            set(action_ids) - {str(record["action_id"]) for record in selectable}
        ),
    }
    if not selectable:
        result["status"] = "no_valid_candidate_outcomes"
        return result

    effective_selection_strategy = str(
        target.parameters.get("selection_strategy", selection_strategy)
    )
    selection_details: dict[str, Any] = {}
    if effective_selection_strategy == "nearest":
        selectable.sort(
            key=lambda record: (
                target.distance(record["selection_value"]),
                str(record["action_id"]),
            )
        )
        selected = selectable[0]
    elif effective_selection_strategy == "boundary_crossing":
        selected, selection_details = _boundary_crossing_choice(selectable, target)
    elif effective_selection_strategy == "first_reaching":
        selected, selection_details = _first_reaching_choice(selectable, target)
    else:
        raise ValueError(
            f"Unknown selection strategy {effective_selection_strategy!r}."
        )
    selected_action = str(selected["action_id"])
    selected_gt = next(
        (record for record in records if str(record["action_id"]) == selected_action), None
    )
    selected_gt_value = None if selected_gt is None else selected_gt.get("ground_truth_value")
    selected_gt_distance = target.distance(selected_gt_value)
    success_policy = str(target.parameters.get("success_policy", "target_reach"))
    ordered_ground_truth = sorted(ground_truth, key=_action_sort_key)
    ground_truth_rank = {
        str(record["action_id"]): rank
        for rank, record in enumerate(ordered_ground_truth)
    }
    target_reaching = [
        record
        for record in ordered_ground_truth
        if target.contains(record.get("ground_truth_value"))
    ]
    minimum_reaching = target_reaching[0] if target_reaching else None
    maximum_steps_above_minimum = int(
        target.parameters.get("max_action_steps_above_minimum", 0)
    )
    if maximum_steps_above_minimum < 0:
        raise ValueError("max_action_steps_above_minimum must be non-negative.")
    if success_policy == "target_reach":
        policy_feasible = target_reaching
    elif success_policy == "minimum_reaching":
        if minimum_reaching is None:
            policy_feasible = []
        else:
            minimum_rank = ground_truth_rank[str(minimum_reaching["action_id"])]
            policy_feasible = [
                record
                for record in target_reaching
                if 0
                <= ground_truth_rank[str(record["action_id"])] - minimum_rank
                <= maximum_steps_above_minimum
            ]
    else:
        raise ValueError(f"Unknown success policy {success_policy!r}.")
    policy_feasible_ids = {
        str(record["action_id"]) for record in policy_feasible
    }
    result.update(
        {
            "status": "ok" if target.valid(selected_gt_value) else "selected_action_missing_gt",
            "selected_action_id": selected_action,
            "selected_action_order": selected.get("action_order"),
            "selection_strategy": effective_selection_strategy,
            "selection_details": selection_details,
            "success_policy": success_policy,
            "selected_sample_indices": selected.get("sample_indices", []),
            "selected_outcome_source": selected.get("selection_source"),
            "selected_candidate_value": selected.get("selection_value"),
            "selected_candidate_distance": target.distance(selected.get("selection_value")),
            "selected_ground_truth_value": selected_gt_value,
            "selected_ground_truth_distance": selected_gt_distance,
            "task_success": selected_action in policy_feasible_ids,
            "target_reaching_action_ids": [
                str(record["action_id"]) for record in target_reaching
            ],
            "minimum_reaching_action_id": (
                str(minimum_reaching["action_id"])
                if minimum_reaching is not None
                else None
            ),
            "max_action_steps_above_minimum": (
                maximum_steps_above_minimum
                if success_policy == "minimum_reaching"
                else None
            ),
        }
    )
    if ground_truth:
        oracle = (
            minimum_reaching
            if success_policy == "minimum_reaching" and minimum_reaching is not None
            else ground_truth[0]
        )
        oracle_distance = target.distance(oracle["ground_truth_value"])
        if success_policy == "minimum_reaching" and minimum_reaching is not None:
            oracle_set = [minimum_reaching]
        else:
            oracle_set = [
                record
                for record in ground_truth
                if isclose(
                    target.distance(record["ground_truth_value"]),
                    oracle_distance,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
        minimum_rank = (
            ground_truth_rank[str(minimum_reaching["action_id"])]
            if minimum_reaching is not None
            else None
        )
        selected_rank = ground_truth_rank.get(selected_action)
        result.update(
            {
                "oracle_action_id": str(oracle["action_id"]),
                "oracle_action_ids": [str(record["action_id"]) for record in oracle_set],
                "oracle_feasible_action_ids": [
                    str(record["action_id"]) for record in policy_feasible
                ],
                "oracle_sample_indices": oracle.get("sample_indices", []),
                "oracle_ground_truth_value": oracle.get("ground_truth_value"),
                "oracle_ground_truth_distance": oracle_distance,
                "oracle_reachable": bool(target_reaching),
                "selected_is_oracle": selected_action
                in {str(record["action_id"]) for record in oracle_set},
                "selected_matches_canonical_oracle": selected_action
                == str(oracle["action_id"]),
                "selected_action_rank": selected_rank,
                "minimum_reaching_action_rank": minimum_rank,
                "selected_action_offset_from_minimum_steps": (
                    selected_rank - minimum_rank
                    if selected_rank is not None and minimum_rank is not None
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
                "oracle_action_ids": [],
                "oracle_feasible_action_ids": [],
                "oracle_sample_indices": [],
                "oracle_ground_truth_value": None,
                "oracle_ground_truth_distance": None,
                "oracle_reachable": False,
                "selected_is_oracle": False,
                "selected_matches_canonical_oracle": False,
                "regret": None,
                "selected_action_rank": None,
                "minimum_reaching_action_rank": None,
                "selected_action_offset_from_minimum_steps": None,
            }
        )
    return result
