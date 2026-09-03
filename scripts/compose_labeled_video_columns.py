#!/usr/bin/env python3
"""Compose equally sized videos horizontally with a persistent label header."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--header-height", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path, label = line.split("\t", 1)
        entries.append((Path(path), label))
    if not entries:
        raise ValueError(f"No video entries in {args.manifest}.")
    for path, _ in entries:
        if not path.is_file():
            raise FileNotFoundError(path)

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    readers = [imageio.get_reader(str(path)) for path, _ in entries]
    iterators = [reader.iter_data() for reader in readers]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264", quality=args.quality, macro_block_size=16
    )
    frame_count = 0
    try:
        while True:
            frames = []
            for iterator in iterators:
                try:
                    frames.append(np.asarray(next(iterator), dtype=np.uint8))
                except StopIteration:
                    frames = []
                    break
            if not frames:
                break
            shape = frames[0].shape
            if any(frame.shape != shape for frame in frames):
                raise ValueError(f"Input frame shapes differ: {[frame.shape for frame in frames]}")
            labeled = []
            for frame, (_, label) in zip(frames, entries):
                height, width = frame.shape[:2]
                canvas = Image.new("RGB", (width, height + args.header_height), (24, 24, 22))
                canvas.paste(Image.fromarray(frame).convert("RGB"), (0, args.header_height))
                draw = ImageDraw.Draw(canvas)
                draw.text((7, 7), label, fill=(255, 255, 255), font=font)
                labeled.append(np.asarray(canvas, dtype=np.uint8))
            writer.append_data(np.concatenate(labeled, axis=1))
            frame_count += 1
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    print(f"[grid] columns={len(entries)} frames={frame_count} output={args.output}")


if __name__ == "__main__":
    main()
