#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from make_gt_pred_comparison import _default_pred_name, _read_gt_video, _read_pred_video, _resize_rgb


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _support_label(row: dict, support_id: int) -> str:
    return (
        f"TTT SUPPORT {support_id} GT | train ep{int(row['episode_index'])} "
        f"{row['source_split']} mu={float(row['friction_mu']):g} "
        f"frames {int(row['start_frame'])}-{int(row['end_frame'])}"
    )


def _query_label(row: dict) -> str:
    return (
        f"QUERY GT | test ep{int(row['episode_index'])} "
        f"{row['source_split']} mu={float(row['friction_mu']):g} "
        f"frames {int(row['start_frame'])}-{int(row['end_frame'])}"
    )


def _load_support_frames(
    *,
    support_rows: list[dict],
    support_indices: list[int],
    dataset_base: Path,
    width: int,
    height: int,
    total_frames: int,
) -> list[tuple[str, list[np.ndarray]]]:
    loaded = []
    for support_id, support_index in enumerate(support_indices, start=1):
        row = support_rows[int(support_index)]
        frames = _read_gt_video(row, dataset_base, width, height, total_frames)
        loaded.append((_support_label(row, support_id), frames))
    return loaded


def _make_video(
    *,
    result: dict,
    query_row: dict,
    support_rows: list[dict],
    dataset_base: Path,
    baseline_pred_dir: Path,
    stage2_pred_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> Path:
    sample_index = int(result["sample_index"])
    num_views = len(query_row["video"])
    target_width = width * num_views
    total_frames = int(query_row.get("length", query_row["end_frame"] - query_row["start_frame"] + 1))

    pred_name = _default_pred_name(sample_index, query_row)
    stage1_path = baseline_pred_dir / pred_name
    stage2_path = stage2_pred_dir / pred_name
    if not stage1_path.exists():
        raise FileNotFoundError(stage1_path)
    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)

    support_items = _load_support_frames(
        support_rows=support_rows,
        support_indices=result["support_indices"],
        dataset_base=dataset_base,
        width=width,
        height=height,
        total_frames=total_frames,
    )
    query_gt = _read_gt_video(query_row, dataset_base, width, height, total_frames)
    stage1_pred = _read_pred_video(stage1_path, target_width, height)
    stage2_pred = _read_pred_video(stage2_path, target_width, height)

    frame_count = min(
        len(query_gt),
        len(stage1_pred),
        len(stage2_pred),
        *(len(frames) for _, frames in support_items),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / pred_name.replace(".mp4", "_support_gt_query_gt_stage1_stage2ttt.mp4")
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=quality) as writer:
        for frame_id in range(frame_count):
            rows = []
            for label, frames in support_items:
                rows.append(_draw_label(_resize_rgb(frames[frame_id], target_width, height), label))
            rows.append(_draw_label(_resize_rgb(query_gt[frame_id], target_width, height), _query_label(query_row)))
            rows.append(_draw_label(_resize_rgb(stage1_pred[frame_id], target_width, height), "STAGE1 PRED | no TTT"))
            rows.append(_draw_label(_resize_rgb(stage2_pred[frame_id], target_width, height), "STAGE2+TTT PRED | adapted from support rows above"))
            writer.append_data(np.concatenate(rows, axis=0))
    return output_path


def _contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in paths:
        reader = imageio.get_reader(str(path))
        try:
            try:
                frame = reader.get_data(40)
            except Exception:
                frame = reader.get_data(0)
        finally:
            reader.close()
        image = Image.fromarray(frame[:, :, :3]).convert("RGB")
        image.thumbnail((224, 560), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (224, 584), "white")
        canvas.paste(image, ((224 - image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        label = path.name.split("_support_gt", 1)[0]
        draw.text((4, 564), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 224, rows * 584), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 224, (idx // cols) * 584))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detailed TTT comparison videos with support GT rows.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--train-metadata-path", required=True)
    parser.add_argument("--test-metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--baseline-pred-dir", required=True)
    parser.add_argument("--stage2-pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contact-sheet-path", default=None)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=5)
    args = parser.parse_args()

    results = _read_jsonl(Path(args.results_path))
    train_rows = _read_jsonl(Path(args.train_metadata_path))
    test_rows = _read_jsonl(Path(args.test_metadata_path))
    output_dir = Path(args.output_dir)

    written = []
    for result in results:
        sample_index = int(result["sample_index"])
        path = _make_video(
            result=result,
            query_row=test_rows[sample_index],
            support_rows=train_rows,
            dataset_base=Path(args.dataset_base_path),
            baseline_pred_dir=Path(args.baseline_pred_dir),
            stage2_pred_dir=Path(args.stage2_pred_dir),
            output_dir=output_dir,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            quality=int(args.quality),
        )
        written.append(path)
        print(f"[write] {path}", flush=True)

    if args.contact_sheet_path:
        _contact_sheet(written, Path(args.contact_sheet_path))
        print(f"[write] {args.contact_sheet_path}", flush=True)


if __name__ == "__main__":
    main()
