#!/usr/bin/env python3
"""Compose support-plus-query grids from completed context-transfer rollouts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from make_gt_pred_comparison import (  # noqa: E402
    _default_pred_name,
    _read_gt_video,
    _read_pred_video,
    _resize_rgb,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decorate(frame: np.ndarray, label: str, *, border: tuple[int, int, int] | None = None) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    text_box = draw.textbbox((0, 0), label)
    label_width = min(image.width, max(132, text_box[2] - text_box[0] + 14))
    draw.rectangle((0, 0, label_width, 25), fill=(0, 0, 0))
    draw.text((7, 7), label, fill=(255, 255, 255))
    if border is not None:
        draw.rectangle((1, 1, image.width - 2, image.height - 2), outline=border, width=5)
    return np.asarray(image, dtype=np.uint8)


def _prediction_dir(root: Path, source_index: int) -> Path:
    matches = sorted(path for path in root.glob(f"source{source_index:04d}_*") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prediction directory for source={source_index}, found {matches}")
    return matches[0]


def _safe_label(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def _action_label(row: dict) -> str:
    action = row.get("button_color", row.get("action_id", "unknown"))
    amplitude = row.get("action_amplitude")
    return str(action) if amplitude is None else f"{action} ({float(amplitude):g})"


def _compose_one(
    *,
    metadata: list[dict],
    dataset_root: Path,
    prediction_root: Path,
    output_dir: Path,
    item: dict,
    width: int,
    height: int,
    fps: int,
    quality: int,
    columns: int,
    support_size: int,
    prediction_label: str,
) -> dict:
    source_index = int(item["source_index"])
    target_indices = [int(value) for value in item["target_indices"]]
    if not target_indices:
        raise ValueError(f"Expected at least one disjoint query for source={source_index}.")

    source_row = metadata[source_index]
    source_frames = _read_gt_video(
        source_row,
        dataset_root,
        width,
        height,
        int(source_row.get("length", source_row["end_frame"] - source_row["start_frame"] + 1)),
    )
    pred_dir = _prediction_dir(prediction_root, source_index)
    query_gt: list[list[np.ndarray]] = []
    query_pred: list[list[np.ndarray]] = []
    for target_index in target_indices:
        row = metadata[target_index]
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        query_gt.append(_read_gt_video(row, dataset_root, width, height, total_frames))
        prediction_path = pred_dir / _default_pred_name(target_index, row)
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        query_pred.append(_read_pred_video(prediction_path, width * len(row["video"]), height))

    frame_count = min(
        [len(source_frames)]
        + [len(frames) for frames in query_gt]
        + [len(frames) for frames in query_pred]
    )
    environment = source_row.get(
        "causal_class",
        source_row.get("environment_id", source_row.get("target_mass_kg", source_index)),
    )
    output_path = output_dir / (
        f"source{source_index:04d}_{_safe_label(environment)}_"
        f"support{support_size}_plus_{len(target_indices)}queries_gt_"
        f"{_safe_label(prediction_label)}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=quality, macro_block_size=1
    ) as writer:
        for frame_id in range(frame_count):
            support = _resize_rgb(source_frames[frame_id], width, height)
            support_cell = np.concatenate(
                [
                    _decorate(support, "SUPPORT exemplar: observed GT", border=(255, 214, 0)),
                    _decorate(
                        support,
                        (
                            f"SUPPORT set: K={support_size} used"
                            if support_size > 0
                            else "SUPPORT: shown, not used by model"
                        ),
                        border=(255, 214, 0),
                    ),
                ],
                axis=0,
            )
            cells = [support_cell]
            for target_index, gt_frames, pred_frames in zip(target_indices, query_gt, query_pred):
                row = metadata[target_index]
                gt = _resize_rgb(gt_frames[frame_id], width, height)
                pred = _resize_rgb(pred_frames[frame_id], width, height)
                cells.append(
                    np.concatenate(
                        [
                            _decorate(gt, f"GT query: {_action_label(row)}"),
                            _decorate(pred, f"{prediction_label}: {_action_label(row)}"),
                        ],
                        axis=0,
                    )
                )
            row_count = math.ceil(len(cells) / columns)
            cells.extend([np.zeros_like(support_cell)] * (row_count * columns - len(cells)))
            rows = [
                np.concatenate(cells[offset : offset + columns], axis=1)
                for offset in range(0, len(cells), columns)
            ]
            writer.append_data(np.concatenate(rows, axis=0))

    result = {
        "source_index": source_index,
        "environment": environment,
        "support_size": support_size,
        "support_action": _action_label(source_row),
        "query_indices": target_indices,
        "output": str(output_path),
    }
    if "target_mass_kg" in source_row:
        result["target_mass_kg"] = float(source_row["target_mass_kg"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--support-size", type=int, default=1)
    parser.add_argument("--prediction-label", default="Stage2 query")
    args = parser.parse_args()

    metadata = _read_jsonl(args.metadata_path)
    plan = json.loads(args.transfer_plan.read_text(encoding="utf-8"))
    outputs = [
        _compose_one(
            metadata=metadata,
            dataset_root=args.dataset_root,
            prediction_root=args.prediction_root,
            output_dir=args.output_dir,
            item=item,
            width=args.width,
            height=args.height,
            fps=args.fps,
            quality=args.quality,
            columns=args.columns,
            support_size=args.support_size,
            prediction_label=args.prediction_label,
        )
        for item in plan
    ]
    manifest_path = args.output_dir / "support_plus_query_grid_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(f"[done] grids={len(outputs)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
