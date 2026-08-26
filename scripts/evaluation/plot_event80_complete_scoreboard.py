#!/usr/bin/env python3
"""Render the frozen Event80 benchmark scoreboard as a paper-style table."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROWS = [
    ("Ours", 31.976, 0.9513, 0.0495, 3.85, 7.59, 68.0),
    ("Joint Model + Latent", 31.737, 0.9494, 0.0587, 4.35, 8.95, 64.0),
    ("No-Curriculum Joint", 31.697, 0.9507, 0.0547, 5.62, 12.04, 60.0),
    ("History-Conditioned WM", 30.936, 0.9519, 0.0658, 12.33, 27.54, 60.0),
    ("LoRA TTA", 32.031, 0.9535, 0.0582, 8.19, 19.19, 40.0),
    ("Standard Pooled WM", 31.113, 0.9513, 0.0667, 11.06, 24.41, 40.0),
    ("DINOv2 Context Encoder", 31.054, 0.9488, 0.0671, 11.25, 24.51, 40.0),
    ("TTT-KQV", 32.253, 0.9557, 0.0595, 11.80, 25.75, 24.0),
]
HEADERS = [
    "Method", "PSNR (MV) ↑", "SSIM (MV) ↑", "LPIPS (MV) ↓",
    "ADE (px) ↓", "FDE (px) ↓", "Action Success ↑",
]
FORMATTED = [
    [name, f"{psnr:.3f}", f"{ssim:.4f}", f"{lpips:.4f}",
     f"{ade:.2f}", f"{fde:.2f}", f"{success:.0f}%"]
    for name, psnr, ssim, lpips, ade, fde, success in ROWS
]
BEST = {(7, 1), (7, 2), (0, 3), (0, 4), (0, 5), (0, 6)}
WIDTH, HEIGHT = 4560, 1710
LEFT = 135
COL_WIDTHS = [1260, 500, 500, 500, 500, 500, 630]
STARTS = [LEFT]
for column_width in COL_WIDTHS[:-1]:
    STARTS.append(STARTS[-1] + column_width)
CENTERS = [start + column_width / 2 for start, column_width in zip(STARTS, COL_WIDTHS)]
RIGHT = LEFT + sum(COL_WIDTHS)
TABLE_TOP, HEADER_HEIGHT, ROW_HEIGHT = 285, 145, 128
HEADER_BOTTOM = TABLE_TOP + HEADER_HEIGHT
TABLE_BOTTOM = HEADER_BOTTOM + ROW_HEIGHT * len(ROWS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def serif_font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError("No suitable system serif font was found")


def draw_png(path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title = serif_font(True, 58)
    subtitle = serif_font(False, 34)
    header = serif_font(True, 38)
    body = serif_font(False, 40)
    body_bold = serif_font(True, 40)
    note = serif_font(False, 30)

    draw.text((WIDTH / 2, 92), "Event80 Physics Adaptation Benchmark",
              fill="#111111", font=title, anchor="mm")
    draw.text((WIDTH / 2, 177),
              "K=1 informative support; 5 ID + 5 OOD environments; 90 disjoint queries per method",
              fill="#111111", font=subtitle, anchor="mm")
    draw.line((LEFT, TABLE_TOP, RIGHT, TABLE_TOP), fill="#111111", width=7)
    draw.line((LEFT, HEADER_BOTTOM, RIGHT, HEADER_BOTTOM), fill="#111111", width=3)
    draw.line((LEFT, TABLE_BOTTOM, RIGHT, TABLE_BOTTOM), fill="#111111", width=7)

    header_y = TABLE_TOP + HEADER_HEIGHT / 2
    for column, value in enumerate(HEADERS):
        x = STARTS[column] + 24 if column == 0 else CENTERS[column]
        anchor = "lm" if column == 0 else "mm"
        draw.text((x, header_y), value, fill="#111111", font=header, anchor=anchor)

    for row, values in enumerate(FORMATTED):
        y = HEADER_BOTTOM + ROW_HEIGHT * (row + 0.5)
        for column, value in enumerate(values):
            is_bold = (row == 0 and column == 0) or (row, column) in BEST
            font = body_bold if is_bold else body
            x = STARTS[column] + 24 if column == 0 else CENTERS[column]
            anchor = "lm" if column == 0 else "mm"
            draw.text((x, y), value, fill="#111111", font=font, anchor=anchor)

    draw.text((LEFT, TABLE_BOTTOM + 72),
              "MV denotes the concatenated main and wrist camera views. ADE/FDE measure the pushed-object centroid.",
              fill="#111111", font=note, anchor="lm")
    draw.text((LEFT, TABLE_BOTTOM + 122),
              "Action success is evaluated over 25 oracle-reachable environment-target decisions; higher is better.",
              fill="#111111", font=note, anchor="lm")
    image.save(path, dpi=(300, 300), optimize=True)


def svg_text(x: float, y: float, value: str, size: int,
             anchor: str = "middle", bold: bool = False) -> str:
    weight = "700" if bold else "400"
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" '
        f'font-size="{size}" font-weight="{weight}">{html.escape(value)}</text>'
    )


def draw_svg(path: Path) -> None:
    values = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g fill="#111111" font-family="DejaVu Serif, Liberation Serif, Times New Roman, serif">',
        svg_text(WIDTH / 2, 92, "Event80 Physics Adaptation Benchmark", 58, bold=True),
        svg_text(WIDTH / 2, 177, "K=1 informative support; 5 ID + 5 OOD environments; 90 disjoint queries per method", 34),
        f'<line x1="{LEFT}" y1="{TABLE_TOP}" x2="{RIGHT}" y2="{TABLE_TOP}" stroke="#111111" stroke-width="7"/>',
        f'<line x1="{LEFT}" y1="{HEADER_BOTTOM}" x2="{RIGHT}" y2="{HEADER_BOTTOM}" stroke="#111111" stroke-width="3"/>',
        f'<line x1="{LEFT}" y1="{TABLE_BOTTOM}" x2="{RIGHT}" y2="{TABLE_BOTTOM}" stroke="#111111" stroke-width="7"/>',
    ]
    header_y = TABLE_TOP + HEADER_HEIGHT / 2
    for column, value in enumerate(HEADERS):
        values.append(svg_text(
            STARTS[column] + 24 if column == 0 else CENTERS[column],
            header_y, value, 38, "start" if column == 0 else "middle", True,
        ))
    for row, row_values in enumerate(FORMATTED):
        y = HEADER_BOTTOM + ROW_HEIGHT * (row + 0.5)
        for column, value in enumerate(row_values):
            values.append(svg_text(
                STARTS[column] + 24 if column == 0 else CENTERS[column],
                y, value, 40, "start" if column == 0 else "middle",
                (row == 0 and column == 0) or (row, column) in BEST,
            ))
    values.extend([
        svg_text(LEFT, TABLE_BOTTOM + 72,
                 "MV denotes the concatenated main and wrist camera views. ADE/FDE measure the pushed-object centroid.",
                 30, "start"),
        svg_text(LEFT, TABLE_BOTTOM + 122,
                 "Action success is evaluated over 25 oracle-reachable environment-target decisions; higher is better.",
                 30, "start"),
        "</g>", "</svg>",
    ])
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "event80_complete_benchmark_table"
    draw_png(stem.with_suffix(".png"))
    draw_svg(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
