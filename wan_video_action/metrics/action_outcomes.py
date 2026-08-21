"""Frozen physical-outcome extraction from predicted task-state artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wan_video_action.evaluation.task_state import TaskState, load_task_state


def extract_predicted_outcome(
    state_path: str | Path, settings: Mapping[str, Any]
) -> float:
    """Extract one scalar outcome or LightSwitch probability from a state NPZ."""

    state = load_task_state(state_path)
    kind = str(settings["kind"])
    if kind == "centroid_forward_displacement":
        return _centroid_displacement(state, settings)
    if kind == "centroid_final_position":
        return _centroid_position(state, settings)
    if kind == "final_angle_deg":
        return _final_angle_deg(state, settings)
    if kind == "final_light_probability":
        return _final_light_probability(state, settings)
    raise ValueError(f"unsupported predicted-outcome extractor: {kind!r}")


def _centroid_track(state: TaskState, settings: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if state.centroids is None:
        raise ValueError("centroid outcome extraction requires centroids")
    object_index = int(settings.get("object_index", 0))
    if not 0 <= object_index < state.centroids.shape[1]:
        raise ValueError(f"object_index {object_index} is outside {state.centroids.shape}")
    axis_name = str(settings.get("axis", "x"))
    if axis_name not in {"x", "y"}:
        raise ValueError("centroid axis must be x or y")
    axis = 0 if axis_name == "x" else 1
    track = np.asarray(state.centroids[:, object_index, axis], dtype=np.float64)
    visible = (
        np.asarray(state.visible[:, object_index], dtype=bool)
        if state.visible is not None
        else np.isfinite(track)
    )
    visible &= np.isfinite(track)
    if not np.any(visible):
        raise ValueError("selected object is never visible in predicted task state")
    return track, visible


def _centroid_displacement(state: TaskState, settings: Mapping[str, Any]) -> float:
    track, visible = _centroid_track(state, settings)
    indices = np.flatnonzero(visible)
    start_index = int(indices[0])
    end_index = int(indices[-1])
    require_final_visible = bool(settings.get("require_final_visible", True))
    if require_final_visible and end_index != len(track) - 1:
        raise ValueError("object is not visible in the final predicted frame")
    displacement_px = float(track[end_index] - track[start_index])
    return _pixel_delta_to_physical(displacement_px, state, settings)


def _centroid_position(state: TaskState, settings: Mapping[str, Any]) -> float:
    track, visible = _centroid_track(state, settings)
    selector = str(settings.get("frame_selector", "last_visible"))
    if selector == "last_visible":
        frame_index = int(np.flatnonzero(visible)[-1])
    elif selector == "first_event":
        event_name = str(settings["event_name"])
        if not state.events or event_name not in state.events:
            raise ValueError(f"task state does not contain event_{event_name}")
        event_indices = np.flatnonzero(np.asarray(state.events[event_name], dtype=bool) & visible)
        if not len(event_indices):
            raise ValueError(f"no visible frame carries event_{event_name}")
        frame_index = int(event_indices[0])
    else:
        raise ValueError(f"unsupported centroid frame_selector: {selector!r}")
    scale = _required_float(settings, "physical_units_per_pixel")
    offset = _required_float(settings, "physical_offset", default=0.0)
    return float(track[frame_index] * scale + offset)


def _pixel_delta_to_physical(
    displacement_px: float, state: TaskState, settings: Mapping[str, Any]
) -> float:
    if settings.get("physical_units_per_pixel") is not None:
        return displacement_px * _required_float(settings, "physical_units_per_pixel")
    units_per_normalized = _required_float(settings, "physical_units_per_normalized_image")
    axis_name = str(settings.get("axis", "x"))
    extent = state.image_width if axis_name == "x" else state.image_height
    if extent is None or extent <= 0:
        raise ValueError("normalized centroid calibration requires image dimensions")
    return displacement_px / float(extent) * units_per_normalized


def _final_angle_deg(state: TaskState, settings: Mapping[str, Any]) -> float:
    if state.angles_rad is None:
        raise ValueError("angle outcome extraction requires angles_rad")
    object_index = int(settings.get("object_index", 0))
    tail_frames = int(settings.get("tail_frames", 5))
    values = np.asarray(state.angles_rad[-tail_frames:, object_index], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no finite bar angles in the requested tail window")
    mean_angle = 0.5 * math.atan2(float(np.mean(np.sin(2.0 * values))), float(np.mean(np.cos(2.0 * values))))
    reference = _required_float(settings, "reference_angle_rad")
    delta = 0.5 * math.atan2(math.sin(2.0 * (mean_angle - reference)), math.cos(2.0 * (mean_angle - reference)))
    return math.degrees(delta)


def _final_light_probability(state: TaskState, settings: Mapping[str, Any]) -> float:
    lamp_index = int(settings.get("lamp_index", 0))
    tail_frames = int(settings.get("tail_frames", 3))
    if state.light_score is not None:
        score = float(np.mean(state.light_score[-tail_frames:, lamp_index]))
        off_score = _required_float(settings, "off_score")
        on_score = _required_float(settings, "on_score")
        if on_score <= off_score:
            raise ValueError("on_score must exceed off_score")
        return float(np.clip((score - off_score) / (on_score - off_score), 0.0, 1.0))
    if state.light_on is not None:
        return float(np.mean(state.light_on[-tail_frames:, lamp_index]))
    raise ValueError("LightSwitch outcome extraction requires light_score or light_on")


def _required_float(
    settings: Mapping[str, Any], key: str, *, default: float | None = None
) -> float:
    value = settings.get(key, default)
    if value is None:
        raise ValueError(f"outcome extractor requires a frozen calibration value for {key}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"outcome calibration {key} must be finite")
    return result
