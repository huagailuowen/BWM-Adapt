#!/usr/bin/env python3
"""Build per-ball train/test video grids without relying on ffmpeg filters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def load_rows(selection_path: Path) -> list[dict[str, str]]:
    with selection_path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_comparison(output_dir: Path, sample_index: int) -> Path:
    matches = sorted(
        output_dir.glob(f"shard*/comparisons/sample{sample_index:04d}_*.mp4")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one comparison for sample {sample_index}, found {len(matches)}"
        )
    return matches[0]


def load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def add_label(frame: np.ndarray, label: str, font: ImageFont.ImageFont) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    draw.rounded_rectangle(
        (5, 5, 17 + text_width, 15 + text_height),
        radius=4,
        fill=(0, 0, 0, 190),
    )
    draw.text((11, 8), label, font=font, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def next_or_last(iterator, last: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        frame = next(iterator)
        return frame, frame
    except StopIteration:
        return last, last


def build_horizontal_grid(
    source_paths: list[Path],
    labels: list[str],
    destination: Path,
    fps: float,
) -> None:
    readers = [imageio.get_reader(path) for path in source_paths]
    iterators = [iter(reader) for reader in readers]
    last_frames: list[np.ndarray | None] = [None] * len(readers)
    font = load_font()
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        destination,
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=None,
    )
    try:
        while True:
            frames: list[np.ndarray] = []
            live = False
            for index, iterator in enumerate(iterators):
                frame, last_frames[index] = next_or_last(iterator, last_frames[index])
                if frame is not None:
                    frames.append(frame)
                    live = live or frame is not last_frames[index - 1] if index else True
                else:
                    frames = []
                    break

            if not frames:
                break

            # Stop once every reader is exhausted; shorter videos retain their last frame.
            exhausted = []
            for reader, iterator in zip(readers, iterators):
                del reader, iterator
                exhausted.append(False)

            labelled = [add_label(frame, label, font) for frame, label in zip(frames, labels)]
            writer.append_data(np.concatenate(labelled, axis=1))

            # Comparison clips in this evaluation are equal length. Use the first reader as
            # the authoritative end condition and avoid relying on ffmpeg frame-count metadata.
            try:
                probe = next(iterators[0])
            except StopIteration:
                break
            iterators[0] = iter([probe, *iterators[0]])
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def build_vertical_grid(top_path: Path, bottom_path: Path, destination: Path, fps: float) -> None:
    top_reader = imageio.get_reader(top_path)
    bottom_reader = imageio.get_reader(bottom_path)
    writer = imageio.get_writer(
        destination,
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=None,
    )
    try:
        for top, bottom in zip(top_reader, bottom_reader):
            writer.append_data(np.concatenate([top, bottom], axis=0))
    finally:
        writer.close()
        top_reader.close()
        bottom_reader.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    rows = load_rows(output_dir / "selection.tsv")
    grid_dir = output_dir / "ball_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    comparison_paths = sorted(
        path for path in output_dir.glob("shard*/comparisons/*.mp4") if path.is_file()
    )
    generated: list[Path] = []

    groups = sorted({int(row["context_group"]) for row in rows})
    for group in groups:
        split_outputs: dict[str, Path] = {}
        group_rows = [row for row in rows if int(row["context_group"]) == group]
        ball_ids = {int(row["ball_id"]) for row in group_rows}
        if len(ball_ids) != 1:
            raise RuntimeError(f"Context group {group} maps to ball IDs {sorted(ball_ids)}")
        ball_id = next(iter(ball_ids))

        for split_name in ("train_id", "test_heldout"):
            selected = sorted(
                (row for row in group_rows if row["eval_split"] == split_name),
                key=lambda row: int(row["slot"]),
            )
            if len(selected) != 6:
                raise RuntimeError(
                    f"Expected 6 rows for group={group} split={split_name}, got {len(selected)}"
                )
            sources = [
                find_comparison(output_dir, int(row["sample_index"])) for row in selected
            ]
            labels = [
                f"Ball {ball_id} | {split_name} | L{row['skill_level']} | ep{row['episode']}"
                for row in selected
            ]
            destination = (
                grid_dir
                / f"ball-{ball_id}_{split_name}_6chunks_gt_stage1_stage2_grid.mp4"
            )
            build_horizontal_grid(sources, labels, destination, args.fps)
            split_outputs[split_name] = destination
            generated.append(destination)

        combined = grid_dir / f"ball-{ball_id}_trainid_vs_testheldout_combined_grid.mp4"
        build_vertical_grid(
            split_outputs["train_id"], split_outputs["test_heldout"], combined, args.fps
        )
        generated.append(combined)

    if len(comparison_paths) != 84:
        raise RuntimeError(f"Expected 84 comparison videos, found {len(comparison_paths)}")
    if len(generated) != 21 or not all(path.is_file() for path in generated):
        raise RuntimeError(f"Expected 21 generated grids, got {len(generated)}")

    (output_dir / "comparison_videos.txt").write_text(
        "".join(f"{path}\n" for path in comparison_paths)
    )
    (output_dir / "grid_videos.txt").write_text(
        "".join(f"{path}\n" for path in generated)
    )
    print(f"Generated {len(generated)} grids from {len(comparison_paths)} comparison videos")
    print(f"Grid directory: {grid_dir}")


if __name__ == "__main__":
    main()
