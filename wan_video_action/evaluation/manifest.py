"""Frozen per-query evaluation manifests shared by every method."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    environment_id: str
    method: str
    split: str
    domain: str
    support_size: int
    seed: int
    gt_video_path: str
    pred_video_path: str
    support_ids: tuple[str, ...] = ()
    gt_start_frame: int = 0
    pred_start_frame: int = 0
    num_frames: int | None = None
    gt_frame_stride: int = 1
    pred_frame_stride: int = 1
    gt_state_start_frame: int | None = None
    pred_state_start_frame: int | None = None
    gt_mask_path: str | None = None
    pred_mask_path: str | None = None
    gt_state_path: str | None = None
    pred_state_path: str | None = None
    task: str | None = None
    object_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.environment_id:
            raise ValueError("sample_id and environment_id are required.")
        if self.sample_id in self.support_ids:
            raise ValueError(
                f"Query {self.sample_id!r} is also present in its support set."
            )
        if self.support_size <= 0:
            raise ValueError("support_size must be positive.")
        if self.support_ids and len(self.support_ids) != self.support_size:
            raise ValueError("support_ids must contain exactly support_size entries.")
        if self.domain not in {"id", "ood"}:
            raise ValueError("domain must be 'id' or 'ood'.")
        if self.gt_start_frame < 0 or self.pred_start_frame < 0:
            raise ValueError("Frame offsets cannot be negative.")
        if self.gt_frame_stride <= 0 or self.pred_frame_stride <= 0:
            raise ValueError("Frame strides must be positive.")
        if self.gt_state_start_frame is not None and self.gt_state_start_frame < 0:
            raise ValueError("GT state offset cannot be negative.")
        if self.pred_state_start_frame is not None and self.pred_state_start_frame < 0:
            raise ValueError("Prediction state offset cannot be negative.")
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError("num_frames must be positive when provided.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationRecord":
        values = dict(payload)
        values["support_ids"] = tuple(values.get("support_ids", ()))
        return cls(**values)

    def resolve_paths(self, base_path: str | Path) -> "EvaluationRecord":
        base = Path(base_path)

        def resolve(value: str | None) -> str | None:
            if value is None:
                return None
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base / path
            return str(path.resolve())

        return replace(
            self,
            gt_video_path=resolve(self.gt_video_path) or "",
            pred_video_path=resolve(self.pred_video_path) or "",
            gt_mask_path=resolve(self.gt_mask_path),
            pred_mask_path=resolve(self.pred_mask_path),
            gt_state_path=resolve(self.gt_state_path),
            pred_state_path=resolve(self.pred_state_path),
        )


def load_manifest(path: str | Path) -> list[EvaluationRecord]:
    manifest_path = Path(path).expanduser().resolve()
    records: list[EvaluationRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = EvaluationRecord.from_dict(payload)
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation manifest row {line_number}: {manifest_path}"
                ) from exc
            records.append(record.resolve_paths(manifest_path.parent))
    if not records:
        raise ValueError(f"Evaluation manifest is empty: {manifest_path}")
    return records
