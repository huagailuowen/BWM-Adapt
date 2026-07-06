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
    draw.rectangle((0, 0, 112, 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _default_pred_name(sample_index: int, row):
    return (
        f"sample{sample_index:04d}_episode{int(row['episode_index']):06d}_"
        f"frames{int(row['start_frame']):04d}-{int(row['end_frame']):04d}.mp4"
    )


def _comparison_frames(gt_frames, w_fric_frames, wo_fric_frames, add_labels: bool):
    total = min(len(gt_frames), len(w_fric_frames), len(wo_fric_frames))
    output = []
    for idx in range(total):
        gt = gt_frames[idx]
        w_fric = w_fric_frames[idx]
        wo_fric = wo_fric_frames[idx]
        if add_labels:
            gt = _draw_label(gt, "GT")
            w_fric = _draw_label(w_fric, "w/ fric")
            wo_fric = _draw_label(wo_fric, "w/o fric")
        output.append(np.concatenate([gt, w_fric, wo_fric], axis=0))
    return output


def main():
    parser = argparse.ArgumentParser(description="Create GT / w-friction / without-friction comparison videos.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--with-fric-dir", required=True)
    parser.add_argument("--without-fric-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default=None, help="Comma/range list, e.g. 0,1,50-52. Defaults to existing shared predictions.")
    parser.add_argument("--width", type=int, default=224, help="Per-view frame width.")
    parser.add_argument("--height", type=int, default=224, help="Per-view frame height.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    dataset_base = Path(args.dataset_base_path)
    with_fric_dir = Path(args.with_fric_dir)
    without_fric_dir = Path(args.without_fric_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    indices = _parse_indices(args.indices)
    if indices is None:
        indices = []
        for sample_index, row in enumerate(rows):
            name = _default_pred_name(sample_index, row)
            if (with_fric_dir / name).exists() and (without_fric_dir / name).exists():
                indices.append(sample_index)

    written = []
    for sample_index in indices:
        row = rows[sample_index]
        pred_name = _default_pred_name(sample_index, row)
        with_fric_path = with_fric_dir / pred_name
        without_fric_path = without_fric_dir / pred_name
        if not with_fric_path.exists():
            print(f"[skip] missing w/ fric prediction: {with_fric_path}")
            continue
        if not without_fric_path.exists():
            print(f"[skip] missing w/o fric prediction: {without_fric_path}")
            continue

        num_views = len(row["video"])
        target_width = args.width * num_views
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        gt_frames = _read_gt_video(row, dataset_base, args.width, args.height, total_frames)
        with_fric_frames = _read_pred_video(with_fric_path, target_width, args.height)
        without_fric_frames = _read_pred_video(without_fric_path, target_width, args.height)
        frames = _comparison_frames(gt_frames, with_fric_frames, without_fric_frames, add_labels=not args.no_labels)

        output_path = output_dir / pred_name.replace(".mp4", "_gt_wfric_wofric.mp4")
        with imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", quality=args.quality) as writer:
            for frame in frames:
                writer.append_data(frame)
        written.append(output_path)
        print(f"[write] {output_path}")

    if not written:
        raise SystemExit("No 3-way comparison videos were written.")


if __name__ == "__main__":
    main()
