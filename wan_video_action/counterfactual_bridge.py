from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENVIRONMENT_SLUGS = (
    "env0_neither",
    "env1_red_only",
    "env2_blue_only",
    "env3_both",
)
ENVIRONMENT_CONTROLS = (
    {"red": False, "blue": False},
    {"red": True, "blue": False},
    {"red": False, "blue": True},
    {"red": True, "blue": True},
)


def nonlinear_bridge_alpha(position: float, power: float = 5.0) -> float:
    position = float(position)
    power = float(power)
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"Bridge position must lie in [0,1], got {position}.")
    if power <= 0.0:
        raise ValueError(f"Bridge curve power must be positive, got {power}.")
    return 1.0 - (1.0 - position) ** power


def environment_mixture(target_index: int, alpha: float, count: int = 4) -> list[float]:
    if not 0 <= int(target_index) < int(count):
        raise ValueError(f"Target index {target_index} is outside [0,{count}).")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Mixture alpha must lie in [0,1], got {alpha}.")
    base = (1.0 - alpha) / float(count)
    return [
        base + (alpha if index == int(target_index) else 0.0)
        for index in range(int(count))
    ]


@dataclass(frozen=True)
class BridgeCondition:
    kind: str
    target_index: int
    position: float
    alpha: float
    weights: tuple[float, ...]


def sample_nonlinear_bridge_condition(
    rng: random.Random,
    *,
    endpoint_probability: float = 0.4,
    curve_power: float = 5.0,
    target_count: int = 4,
    forced_kind: str | None = None,
) -> BridgeCondition:
    endpoint_probability = float(endpoint_probability)
    if not 0.0 <= endpoint_probability <= 1.0:
        raise ValueError("endpoint_probability must lie in [0,1].")
    if forced_kind is None:
        draw = rng.random()
        if draw < endpoint_probability:
            kind = "endpoint"
        elif draw < endpoint_probability + (1.0 - endpoint_probability) / 2.0:
            kind = "near_global"
        else:
            kind = "interior"
    else:
        kind = str(forced_kind)
    if kind not in {"endpoint", "near_global", "interior"}:
        raise ValueError(f"Unsupported bridge condition kind: {kind}")

    target_index = rng.randrange(int(target_count))
    if kind == "endpoint":
        position = 1.0
    elif kind == "near_global":
        stratum = rng.randrange(3)
        lower = (0.1 / 3.0) * stratum
        upper = (0.1 / 3.0) * (stratum + 1)
        position = rng.uniform(max(1e-4, lower), upper)
    else:
        position = rng.uniform(0.1, 1.0 - 1e-4)
    alpha = nonlinear_bridge_alpha(position, curve_power)
    return BridgeCondition(
        kind=kind,
        target_index=target_index,
        position=position,
        alpha=alpha,
        weights=tuple(environment_mixture(target_index, alpha, target_count)),
    )


def parse_noise_bands(spec: str) -> tuple[tuple[float, float, float], ...]:
    bands = []
    for item in str(spec).split(","):
        fields = [float(value.strip()) for value in item.split(":")]
        if len(fields) != 3:
            raise ValueError(f"Noise band must be min:max:weight, got {item!r}.")
        lower, upper, weight = fields
        if not 0.0 <= lower < upper <= 1.0 or weight <= 0.0:
            raise ValueError(f"Invalid noise band {item!r}.")
        bands.append((lower, upper, weight))
    total = sum(weight for _, _, weight in bands)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Noise-band weights must sum to 1, got {total}.")
    return tuple(bands)


def sample_noise_fraction(
    rng: random.Random,
    bands: Iterable[tuple[float, float, float]],
) -> float:
    bands = tuple(bands)
    draw = rng.random()
    cumulative = 0.0
    for lower, upper, weight in bands:
        cumulative += float(weight)
        if draw <= cumulative:
            return rng.uniform(float(lower), float(upper))
    lower, upper, _ = bands[-1]
    return rng.uniform(float(lower), float(upper))


def _covered_colors(row: dict) -> tuple[str, ...]:
    colors = row.get("covered_button_colors", ())
    if isinstance(colors, str):
        colors = (colors,)
    return tuple(
        color for color in ("red", "blue")
        if color in {str(value).lower() for value in colors}
    )


class CounterfactualSourceBank:
    """Offline full-rollout Teacher videos indexed by their original real chunk."""

    def __init__(self, manifest_path: str, raw_root: str):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.raw_root = Path(raw_root).expanduser().resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            entries = payload.get(
                "samples",
                payload.get("records", payload.get("entries", payload.get("plan", ()))),
            )
        else:
            entries = payload
        self.entries: dict[int, dict] = {}
        for entry in entries:
            source_index = int(entry["source_index"])
            self.entries[source_index] = dict(entry)
        if not self.entries:
            raise ValueError(f"No Teacher samples found in {self.manifest_path}.")

    @property
    def available_indices(self) -> set[int]:
        return set(self.entries)

    def restrict_grouped_indices(self, grouped_indices: dict) -> dict:
        available = self.available_indices
        restricted = {}
        for value, by_action in grouped_indices.items():
            kept_actions = {}
            for action_id, indices in by_action.items():
                kept = [index for index in indices if int(index) in available]
                if kept:
                    kept_actions[action_id] = kept
            if kept_actions:
                restricted[value] = kept_actions
        return restricted

    def _candidate_source_groups(self, target_index: int, target_row: dict) -> list[int]:
        entry = self.entries.get(int(target_index))
        if entry is None:
            raise KeyError(f"Teacher bank has no source chunk for metadata index {target_index}.")
        available = [int(value) for value in entry.get("target_groups", range(4))]
        colors = _covered_colors(target_row)
        if not colors:
            raise ValueError(
                f"Counterfactual sample {target_index} has no covered red/blue button metadata."
            )
        target_outcome = tuple(
            bool(target_row.get(f"{color}_controls_lamp", False))
            for color in colors
        )
        causal = [
            group for group in available
            if tuple(ENVIRONMENT_CONTROLS[group][color] for color in colors) != target_outcome
        ]
        if not causal:
            raise ValueError(
                f"No causally different Teacher context exists for sample {target_index}; "
                f"colors={colors} available={available}."
            )
        return causal

    def _video_path(self, target_index: int, target_row: dict, source_group: int) -> Path:
        entry = self.entries[int(target_index)]
        for key in ("raw_paths", "output_paths", "predictions"):
            mapping = entry.get(key)
            if isinstance(mapping, dict):
                candidate = mapping.get(str(source_group), mapping.get(source_group))
                if candidate:
                    path = Path(candidate)
                    return path if path.is_absolute() else self.raw_root / path
        episode = int(target_row.get("episode_index", entry.get("episode_index", target_index)))
        start = int(target_row.get("start_frame", entry.get("start_frame", 0)))
        end = int(target_row.get("end_frame", entry.get("end_frame", start)))
        stem = (
            f"sample{int(target_index):04d}_episode{episode:06d}_"
            f"frames{start:04d}-{end:04d}.mp4"
        )
        return self.raw_root / f"context_{ENVIRONMENT_SLUGS[int(source_group)]}" / stem

    def materialize(
        self,
        *,
        dataset,
        target_data: dict,
        target_index: int,
        target_row: dict,
        rng: random.Random,
    ) -> dict:
        source_group = rng.choice(self._candidate_source_groups(target_index, target_row))
        path = self._video_path(target_index, target_row, source_group)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Teacher counterfactual video: {path}")
        target_video = target_data["video"]
        target_frames = int(target_video.shape[-3])
        video_operator = dataset.special_operator_map.get("video", dataset.main_data_operator)
        fake_video = video_operator(
            {
                "data": str(path),
                "start_frame": 0,
                "end_frame": target_frames - 1,
                "frame_stride": 1,
            }
        )
        if tuple(fake_video.shape) == tuple(target_video.shape):
            source_video = fake_video
        elif (
            fake_video.ndim == target_video.ndim == 5
            and int(fake_video.shape[0]) == 1
            and tuple(fake_video.shape[1:]) == tuple(target_video.shape[1:])
        ):
            source_video = target_video.clone()
            source_video[0:1] = fake_video
        else:
            raise ValueError(
                "Teacher/target video shape mismatch: "
                f"teacher={tuple(fake_video.shape)} target={tuple(target_video.shape)}"
            )
        source_data = target_data.copy()
        source_data["video"] = source_video
        source_data["_teacher_counterfactual_source"] = True
        source_data["_teacher_source_environment"] = int(source_group)
        source_data["_teacher_source_path"] = str(path)
        return source_data
