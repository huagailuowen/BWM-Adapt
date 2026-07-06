import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def _parse_indices(value: str | None):
    if not value:
        return None
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def _resize_rgb(frame, width: int, height: int):
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _read_gt_video(row, dataset_base: Path, width: int, height: int, total_frames: int):
    start_frame = int(row["start_frame"])
    valid_frames = int(row.get("valid_frames", row.get("length", total_frames)))
    readers = [imageio.get_reader(str(dataset_base / rel_path)) for rel_path in row["video"]]
    frames = []
    try:
        for offset in range(total_frames):
            source_offset = min(offset, max(valid_frames - 1, 0))
            frame_idx = start_frame + source_offset
            view_frames = [_resize_rgb(reader.get_data(frame_idx), width, height) for reader in readers]
            frames.append(np.concatenate(view_frames, axis=1))
    finally:
        for reader in readers:
            reader.close()
    return frames


def _read_pred_video(path: Path, target_width: int, target_height: int):
    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for frame in reader:
            frame = frame[:, :, :3]
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = _resize_rgb(frame, target_width, target_height)
            frames.append(frame)
    finally:
        reader.close()
    return frames


def _draw_label(frame: np.ndarray, label: str):
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(100, min(220, 10 + 7 * len(label)))
    draw.rectangle((0, 0, width, 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _default_pred_name(sample_index: int, row):
    return (
        f"sample{sample_index:04d}_episode{int(row['episode_index']):06d}_"
        f"frames{int(row['start_frame']):04d}-{int(row['end_frame']):04d}.mp4"
    )


def _parse_pred_dirs(values: list[str]):
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--pred must be LABEL=DIR, got {value!r}")
        label, path = value.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Create GT + multiple prediction comparison videos.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--pred", action="append", required=True, help="LABEL=DIR. Can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    dataset_base = Path(args.dataset_base_path)
    pred_dirs = _parse_pred_dirs(args.pred)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    indices = _parse_indices(args.indices)
    if indices is None:
        indices = []
        for sample_index, row in enumerate(rows):
            pred_name = _default_pred_name(sample_index, row)
            if all((pred_dir / pred_name).exists() for _, pred_dir in pred_dirs):
                indices.append(sample_index)

    written = []
    for sample_index in indices:
        row = rows[sample_index]
        pred_name = _default_pred_name(sample_index, row)
        missing = [(label, pred_dir / pred_name) for label, pred_dir in pred_dirs if not (pred_dir / pred_name).exists()]
        if missing:
            for label, path in missing:
                print(f"[skip] missing {label}: {path}")
            continue

        num_views = len(row["video"])
        target_width = args.width * num_views
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        videos = [("GT", _read_gt_video(row, dataset_base, args.width, args.height, total_frames))]
        for label, pred_dir in pred_dirs:
            videos.append((label, _read_pred_video(pred_dir / pred_name, target_width, args.height)))

        frame_count = min(len(frames) for _, frames in videos)
        output_frames = []
        for frame_idx in range(frame_count):
            rows_to_stack = []
            for label, frames in videos:
                frame = frames[frame_idx]
                if not args.no_labels:
                    frame = _draw_label(frame, label)
                rows_to_stack.append(frame)
            output_frames.append(np.concatenate(rows_to_stack, axis=0))

        output_path = output_dir / pred_name.replace(".mp4", "_gt_c_sweep.mp4")
        with imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", quality=args.quality) as writer:
            for frame in output_frames:
                writer.append_data(frame)
        written.append(output_path)
        print(f"[write] {output_path}")

    if not written:
        raise SystemExit("No comparison videos were written.")


if __name__ == "__main__":
    main()
