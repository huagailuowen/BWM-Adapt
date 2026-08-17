"""Frozen color/trajectory tracker for the Event80 push-box camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event80TrackerConfig:
    roi_y_min: int = 82
    blue_minus_red: float = 0.035
    blue_minus_green: float = 0.008
    minimum_saturation: float = 0.16
    maximum_value: float = 0.72
    minimum_area: int = 8
    maximum_area: int = 700
    maximum_width: int = 45
    maximum_height: int = 55
    offscreen_min_y: float = 185.0
    initial_x: float = 118.0
    initial_y: float = 103.0


@dataclass(frozen=True)
class TrackedMasks:
    masks: np.ndarray
    visible: np.ndarray
    offscreen: np.ndarray
    centers: np.ndarray
    first_offscreen_frame: int | None


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    output = np.zeros_like(mask)
    target_y = slice(max(0, dy), mask.shape[0] + min(0, dy))
    target_x = slice(max(0, dx), mask.shape[1] + min(0, dx))
    source_y = slice(max(0, -dy), mask.shape[0] - max(0, dy))
    source_x = slice(max(0, -dx), mask.shape[1] - max(0, dx))
    output[target_y, target_x] = mask[source_y, source_x]
    return output


def _dilate(mask: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce([
        _shift(mask, dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ])


def _erode(mask: np.ndarray) -> np.ndarray:
    return np.logical_and.reduce([
        _shift(mask, dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ])


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for start_y, start_x in zip(*np.where(mask)):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        ys: list[int] = []
        xs: list[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    neighbor_y, neighbor_x = y + dy, x + dx
                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and mask[neighbor_y, neighbor_x]
                        and not seen[neighbor_y, neighbor_x]
                    ):
                        seen[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        output.append((np.asarray(ys), np.asarray(xs)))
    return output


def track_event80_block(
    frames: np.ndarray,
    config: Event80TrackerConfig = Event80TrackerConfig(),
) -> TrackedMasks:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames [T,H,W,3], got {frames.shape}.")
    height, width = frames.shape[1:3]
    masks = np.zeros((len(frames), height, width), dtype=bool)
    visible = np.zeros(len(frames), dtype=bool)
    offscreen = np.zeros(len(frames), dtype=bool)
    centers = np.full((len(frames), 2), np.nan, dtype=np.float64)
    previous = np.array([config.initial_x, config.initial_y], dtype=np.float64)
    recent_centers: list[np.ndarray] = []
    exited = False

    for frame_index, frame in enumerate(frames):
        rgb = frame.astype(np.float64) / 255.0
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maximum = rgb.max(axis=-1)
        minimum = rgb.min(axis=-1)
        saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
        y_grid = np.indices(red.shape)[0]
        candidate_pixels = (
            (y_grid >= config.roi_y_min)
            & (blue - red > config.blue_minus_red)
            & (blue - green > config.blue_minus_green)
            & (saturation > config.minimum_saturation)
            & (maximum < config.maximum_value)
        )
        candidate_pixels = _erode(
            _dilate(_dilate(_erode(candidate_pixels)))
        )
        candidates = []
        if not exited:
            for ys, xs in _components(candidate_pixels):
                area = len(xs)
                if not config.minimum_area <= area <= config.maximum_area:
                    continue
                component_width = int(xs.max() - xs.min() + 1)
                component_height = int(ys.max() - ys.min() + 1)
                if (
                    component_width > config.maximum_width
                    or component_height > config.maximum_height
                ):
                    continue
                center = np.array([xs.mean(), ys.mean()], dtype=np.float64)
                continuity = float(np.linalg.norm(center - previous))
                color_score = float(np.mean((blue - red)[ys, xs]))
                score = (
                    color_score * 80.0
                    + min(area, 180) / 40.0
                    - continuity / 35.0
                    + center[1] / 180.0
                )
                candidates.append((score, center, ys, xs))

        if candidates:
            _, center, ys, xs = max(candidates, key=lambda item: item[0])
            masks[frame_index, ys, xs] = True
            visible[frame_index] = True
            centers[frame_index] = center
            previous = center
            recent_centers.append(center.copy())
            recent_centers = recent_centers[-4:]
            continue

        downward_velocity = 0.0
        if len(recent_centers) >= 2:
            downward_velocity = float(
                recent_centers[-1][1] - recent_centers[0][1]
            ) / float(len(recent_centers) - 1)
        if recent_centers and (
            previous[1] >= config.offscreen_min_y
            or (previous[1] >= 160.0 and downward_velocity > 2.0)
        ):
            exited = True
        if exited:
            offscreen[frame_index] = True
            sentinel_x, sentinel_y = width // 2, height - 1
            masks[frame_index, sentinel_y, sentinel_x] = True
            centers[frame_index] = np.array([sentinel_x, sentinel_y])

    offscreen_indices = np.flatnonzero(offscreen)
    first_offscreen = int(offscreen_indices[0]) if len(offscreen_indices) else None
    return TrackedMasks(
        masks=masks,
        visible=visible,
        offscreen=offscreen,
        centers=centers,
        first_offscreen_frame=first_offscreen,
    )
