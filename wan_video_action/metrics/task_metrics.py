"""Task-specific physical metrics computed from frozen state trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wan_video_action.evaluation.manifest import EvaluationRecord
from wan_video_action.evaluation.task_state import TaskState, load_task_state


@dataclass(frozen=True)
class TaskMetricResult:
    values: dict[str, Any]
    metric_names: tuple[str, ...]


def _aligned(record: EvaluationRecord, gt: TaskState, pred: TaskState) -> tuple[TaskState, TaskState]:
    gt = gt.sliced(record.gt_start_frame, record.num_frames)
    pred = pred.sliced(record.pred_start_frame, record.num_frames)
    count = min(gt.frame_count, pred.frame_count)
    return gt.sliced(0, count), pred.sliced(0, count)


def _base(record: EvaluationRecord, task: str, frames: int) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "environment_id": record.environment_id,
        "method": record.method,
        "split": record.split,
        "domain": record.domain,
        "support_size": record.support_size,
        "seed": record.seed,
        "task": task,
        "evaluated_frames": frames,
    }


def _object_names(settings: dict[str, Any], count: int) -> list[str]:
    names = list(settings.get("object_names") or [])
    if not names:
        names = [f"object{index}" for index in range(count)]
    if len(names) != count:
        raise ValueError(f"object_names has {len(names)} entries for {count} objects.")
    return names


def _centroid_metrics(
    gt: TaskState,
    pred: TaskState,
    settings: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    if gt.centroids is None or pred.centroids is None:
        raise ValueError("This task requires centroid tracks in both state files.")
    if gt.centroids.shape[1] != pred.centroids.shape[1]:
        raise ValueError("GT and prediction contain different object counts.")
    height = gt.image_height or int(settings.get("image_height", 0))
    width = gt.image_width or int(settings.get("image_width", 0))
    if height <= 0 or width <= 0:
        raise ValueError("image_height and image_width are required for normalized errors.")
    diagonal = float(np.hypot(height, width))
    penalty = float(settings.get("missing_penalty_normalized", 1.0)) * diagonal
    gt_visible = gt.visible if gt.visible is not None else np.all(np.isfinite(gt.centroids), axis=-1)
    pred_visible = pred.visible if pred.visible is not None else np.all(np.isfinite(pred.centroids), axis=-1)
    names = _object_names(settings, gt.centroids.shape[1])
    output: dict[str, float] = {}
    metric_names: list[str] = []
    all_errors: list[float] = []
    all_final: list[float] = []
    for object_index, name in enumerate(names):
        valid = np.flatnonzero(gt_visible[:, object_index])
        if not len(valid):
            raise ValueError(f"GT object {name!r} is never visible.")
        errors = []
        for frame_index in valid:
            if pred_visible[frame_index, object_index] and np.all(
                np.isfinite(pred.centroids[frame_index, object_index])
            ):
                error = float(np.linalg.norm(
                    gt.centroids[frame_index, object_index]
                    - pred.centroids[frame_index, object_index]
                ))
            else:
                error = penalty
            errors.append(error)
        prefix = f"{name}_centroid"
        values = {
            f"{prefix}_ade_px": float(np.mean(errors)),
            f"{prefix}_fde_px": float(errors[-1]),
            f"{prefix}_ade_normalized": float(np.mean(errors) / diagonal),
            f"{prefix}_fde_normalized": float(errors[-1] / diagonal),
            f"{name}_missing_rate": float(
                np.mean(~pred_visible[valid, object_index])
            ),
        }
        output.update(values)
        metric_names.extend(values)
        all_errors.extend(errors)
        all_final.append(errors[-1])
    aggregate = {
        "centroid_ade_px": float(np.mean(all_errors)),
        "centroid_fde_px": float(np.mean(all_final)),
        "centroid_ade_normalized": float(np.mean(all_errors) / diagonal),
        "centroid_fde_normalized": float(np.mean(all_final) / diagonal),
    }
    output.update(aggregate)
    metric_names.extend(aggregate)
    return output, metric_names


def _kinematic_metrics(
    gt: TaskState,
    pred: TaskState,
    object_index: int,
) -> tuple[dict[str, float], list[str]]:
    assert gt.centroids is not None and pred.centroids is not None
    gt_track = gt.centroids[:, object_index]
    pred_track = pred.centroids[:, object_index]
    valid = np.all(np.isfinite(gt_track), axis=1) & np.all(np.isfinite(pred_track), axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return {}, []
    gt_track = gt_track[indices]
    pred_track = pred_track[indices]
    gt_steps = np.linalg.norm(np.diff(gt_track, axis=0), axis=1)
    pred_steps = np.linalg.norm(np.diff(pred_track, axis=0), axis=1)
    fps = float(gt.fps)
    output = {
        "travel_distance_abs_error_px": float(abs(pred_steps.sum() - gt_steps.sum())),
        "final_displacement_abs_error_px": float(abs(
            np.linalg.norm(pred_track[-1] - pred_track[0])
            - np.linalg.norm(gt_track[-1] - gt_track[0])
        )),
        "terminal_speed_abs_error_px_s": float(abs(pred_steps[-1] - gt_steps[-1]) * fps),
    }
    return output, list(output)


def _angle_metrics(
    gt: TaskState,
    pred: TaskState,
    settings: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    if gt.angles_rad is None or pred.angles_rad is None:
        raise ValueError("Mass-balance evaluation requires bar angles or bar masks.")
    index = int(settings.get("angle_object_index", 0))
    gt_angle = gt.angles_rad[:, index]
    pred_angle = pred.angles_rad[:, index]
    valid = np.isfinite(gt_angle) & np.isfinite(pred_angle)
    if not np.any(valid):
        raise ValueError("No valid GT/pred bar-angle pairs.")
    # A bar edge is unoriented: theta and theta + pi describe the same edge.
    delta = (pred_angle[valid] - gt_angle[valid] + np.pi / 2.0) % np.pi - np.pi / 2.0
    error_deg = np.abs(np.rad2deg(delta))
    output = {
        "bar_tilt_mae_deg": float(np.mean(error_deg)),
        "bar_tilt_final_error_deg": float(error_deg[-1]),
    }
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) >= 2:
        gt_velocity = np.diff(np.unwrap(gt_angle[valid_indices], period=np.pi)) * gt.fps
        pred_velocity = np.diff(np.unwrap(pred_angle[valid_indices], period=np.pi)) * pred.fps
        output["bar_angular_velocity_mae_deg_s"] = float(np.mean(
            np.abs(np.rad2deg(pred_velocity - gt_velocity))
        ))
    return output, list(output)


def _first_transition(values: np.ndarray) -> int | None:
    changes = np.flatnonzero(values[1:] != values[0])
    return int(changes[0] + 1) if len(changes) else None


def _light_metrics(gt: TaskState, pred: TaskState) -> tuple[dict[str, float], list[str]]:
    if gt.light_on is None or pred.light_on is None:
        raise ValueError("LightSwitch evaluation requires light_on state arrays.")
    if gt.light_on.shape[1] != pred.light_on.shape[1]:
        raise ValueError("GT and prediction contain different lamp counts.")
    equal = gt.light_on == pred.light_on
    output = {
        "light_state_accuracy": float(np.mean(equal)),
        "light_final_state_accuracy": float(np.mean(equal[-1])),
    }
    transition_errors = []
    transition_exact = []
    frame_count = len(gt.light_on)
    for lamp_index in range(gt.light_on.shape[1]):
        gt_transition = _first_transition(gt.light_on[:, lamp_index])
        pred_transition = _first_transition(pred.light_on[:, lamp_index])
        transition_exact.append(gt_transition == pred_transition)
        if gt_transition is None and pred_transition is None:
            transition_errors.append(0.0)
        elif gt_transition is None or pred_transition is None:
            transition_errors.append(float(frame_count))
        else:
            transition_errors.append(float(abs(gt_transition - pred_transition)))
    output.update({
        "light_transition_time_abs_error_frames": float(np.mean(transition_errors)),
        "light_transition_exact_rate": float(np.mean(transition_exact)),
    })
    return output, list(output)


def _event_time_metrics(gt: TaskState, pred: TaskState, event: str) -> tuple[dict[str, float], list[str]]:
    if not gt.events or not pred.events or event not in gt.events or event not in pred.events:
        return {}, []
    gt_indices = np.flatnonzero(gt.events[event])
    pred_indices = np.flatnonzero(pred.events[event])
    if not len(gt_indices) and not len(pred_indices):
        error = 0.0
    elif not len(gt_indices) or not len(pred_indices):
        error = float(gt.frame_count)
    else:
        error = float(abs(int(pred_indices[0]) - int(gt_indices[0])))
    name = f"{event}_time_abs_error_frames"
    return {name: error}, [name]


def evaluate_task_record(
    record: EvaluationRecord,
    settings: dict[str, Any],
) -> TaskMetricResult:
    if not record.gt_state_path or not record.pred_state_path:
        raise ValueError(f"Task record {record.sample_id!r} requires both state paths.")
    task = record.task or settings.get("task")
    if not task:
        raise ValueError("Task must be set in the manifest or evaluation config.")
    gt, pred = _aligned(
        record,
        load_task_state(record.gt_state_path, mask_key=settings.get("mask_key", "masks")),
        load_task_state(record.pred_state_path, mask_key=settings.get("mask_key", "masks")),
    )
    output = _base(record, str(task), gt.frame_count)
    metric_names: list[str] = []

    object_tasks = {
        "pushbox_friction",
        "multi_background_pushbox",
        "gravity",
        "mass_collision",
        "mass_balance",
        "joint_mass_friction",
        "pnp_payload",
        "real_slope_friction",
    }
    if task in object_tasks:
        values, names = _centroid_metrics(gt, pred, settings)
        output.update(values)
        metric_names.extend(names)
    if task in {
        "pushbox_friction",
        "multi_background_pushbox",
        "gravity",
        "mass_collision",
        "joint_mass_friction",
        "pnp_payload",
        "real_slope_friction",
    }:
        values, names = _kinematic_metrics(
            gt, pred, int(settings.get("kinematic_object_index", 0))
        )
        output.update(values)
        metric_names.extend(names)
    if task == "mass_balance":
        values, names = _angle_metrics(gt, pred, settings)
        output.update(values)
        metric_names.extend(names)
    if task == "lightswitch":
        values, names = _light_metrics(gt, pred)
        output.update(values)
        metric_names.extend(names)
    for event in settings.get("event_metrics", []):
        values, names = _event_time_metrics(gt, pred, str(event))
        output.update(values)
        metric_names.extend(names)
    return TaskMetricResult(output, tuple(dict.fromkeys(metric_names)))
