#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from make_gt_pred_comparison import (
    _default_pred_name,
    _read_gt_video,
    _read_pred_video,
    _resize_rgb,
)


def read_metadata(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_width = min(image.width, max(180, 7 * len(label) + 14))
    draw.rectangle((0, 0, label_width, 23), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def parse_indices(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 10:
        raise ValueError(f"Exactly 10 sample indices are required, got {len(values)}.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser("Write one GT/prediction grid for ten video chunks.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--environment-label", required=True)
    parser.add_argument("--prediction-label", default="Stage1")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quality", type=int, default=6)
    args = parser.parse_args()

    metadata = read_metadata(Path(args.metadata_path))
    indices = parse_indices(args.indices)
    columns = int(args.columns)
    if columns <= 0 or len(indices) % columns != 0:
        raise ValueError(
            f"columns must be a positive divisor of {len(indices)}, got {columns}."
        )
    prediction_dir = Path(args.prediction_dir)
    dataset_base = Path(args.dataset_base_path)
    gt_videos = []
    pred_videos = []

    for sample_index in indices:
        row = metadata[sample_index]
        num_views = len(row["video"])
        target_width = int(args.width) * num_views
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        prediction_path = prediction_dir / _default_pred_name(sample_index, row)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        gt_videos.append(
            _read_gt_video(row, dataset_base, int(args.width), int(args.height), total_frames)
        )
        pred_videos.append(_read_pred_video(prediction_path, target_width, int(args.height)))

    frame_count = min(len(video) for video in gt_videos + pred_videos)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path), fps=int(args.fps), codec="libx264", quality=int(args.quality)
    ) as writer:
        for frame_id in range(frame_count):
            canvas_rows = []
            for begin in range(0, len(indices), columns):
                gt_cells = []
                pred_cells = []
                for offset in range(begin, begin + columns):
                    sample_index = indices[offset]
                    row = metadata[sample_index]
                    target_width = int(args.width) * len(row["video"])
                    descriptor = (
                        f"s{sample_index} ep{int(row['episode_index'])} "
                        f"raw{int(row['start_frame'])}-{int(row['end_frame'])}"
                    )
                    gt_frame = _resize_rgb(gt_videos[offset][frame_id], target_width, int(args.height))
                    pred_frame = _resize_rgb(pred_videos[offset][frame_id], target_width, int(args.height))
                    gt_cells.append(draw_label(gt_frame, f"GT | {descriptor}"))
                    pred_cells.append(draw_label(pred_frame, f"{args.prediction_label} | {descriptor}"))
                canvas_rows.append(np.concatenate(gt_cells, axis=1))
                canvas_rows.append(np.concatenate(pred_cells, axis=1))
            writer.append_data(np.concatenate(canvas_rows, axis=0))

    print(
        f"[grid] environment={args.environment_label} cases=10 frames={frame_count} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
