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


def _read_metadata(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_indices(raw: str) -> list[int]:
    indices = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        if "-" in value:
            start, end = value.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(value))
    if len(indices) != 10:
        raise ValueError(f"Exactly 10 indices are required, got {len(indices)}")
    return indices


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = min(image.width, max(180, 7 * len(label) + 14))
    draw.rectangle((0, 0, width, 23), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser("Create a 3x10 GT/correct-action/swapped-action grid.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--prediction-dir-a", required=True)
    parser.add_argument("--prediction-label-a", default="Correct action")
    parser.add_argument("--prediction-dir-b", required=True)
    parser.add_argument("--prediction-label-b", default="Swapped action")
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quality", type=int, default=6)
    args = parser.parse_args()

    metadata = _read_metadata(Path(args.metadata_path))
    indices = _parse_indices(args.indices)
    dataset_base = Path(args.dataset_base_path)
    pred_a_dir = Path(args.prediction_dir_a)
    pred_b_dir = Path(args.prediction_dir_b)
    gt_videos = []
    pred_a_videos = []
    pred_b_videos = []

    for sample_index in indices:
        row = metadata[sample_index]
        pred_name = _default_pred_name(sample_index, row)
        pred_a_path = pred_a_dir / pred_name
        pred_b_path = pred_b_dir / pred_name
        if not pred_a_path.is_file():
            raise FileNotFoundError(pred_a_path)
        if not pred_b_path.is_file():
            raise FileNotFoundError(pred_b_path)
        target_width = int(args.width) * len(row["video"])
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        gt_videos.append(
            _read_gt_video(row, dataset_base, int(args.width), int(args.height), total_frames)
        )
        pred_a_videos.append(_read_pred_video(pred_a_path, target_width, int(args.height)))
        pred_b_videos.append(_read_pred_video(pred_b_path, target_width, int(args.height)))

    frame_count = min(
        len(video) for video in gt_videos + pred_a_videos + pred_b_videos
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path), fps=int(args.fps), codec="libx264", quality=int(args.quality)
    ) as writer:
        for frame_id in range(frame_count):
            rows = []
            for label, videos in (
                ("GT", gt_videos),
                (args.prediction_label_a, pred_a_videos),
                (args.prediction_label_b, pred_b_videos),
            ):
                cells = []
                for offset, sample_index in enumerate(indices):
                    row = metadata[sample_index]
                    target_width = int(args.width) * len(row["video"])
                    original_index = int(row.get("source_sample_index", sample_index))
                    descriptor = f"s{original_index} ep{int(row['episode_index'])}"
                    frame = _resize_rgb(
                        videos[offset][frame_id], target_width, int(args.height)
                    )
                    cells.append(_draw_label(frame, f"{label} | {descriptor}"))
                rows.append(np.concatenate(cells, axis=1))
            writer.append_data(np.concatenate(rows, axis=0))

    print(f"[grid] cases=10 frames={frame_count} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
