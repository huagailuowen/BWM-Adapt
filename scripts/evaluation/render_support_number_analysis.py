#!/usr/bin/env python3
"""Render the completed support-number/action-success analysis."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/support_number_analysis"
CSV_PATH = RESULT_DIR / "support_number_summary.csv"
SVG_PATH = RESULT_DIR / "support_number_action_success.svg"
PNG_PATH = RESULT_DIR / "support_number_action_success.png"

WIDTH, HEIGHT = 2700, 1160
MARGIN_X, PLOT_TOP, PLOT_BOTTOM = 120, 235, 880
PANEL_GAP = 85
PANEL_WIDTH = (WIDTH - 2 * MARGIN_X - 2 * PANEL_GAP) / 3
TASKS = ("Event80", "Light Switch", "Mass Balance")
COLORS = {
    "Event80": "#2F6690",
    "Light Switch": "#D97706",
    "Mass Balance": "#16856B",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{('-' + suffix) if suffix else ''}.ttf",
        size,
    )


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Serif, serif" fill="#172033">',
    ]

    def text(x: float, y: float, value: str, size: int, *, bold: bool = False, anchor: str = "middle", color: str = "#172033") -> None:
        draw.text((x, y), value, font=font(size, bold), fill=color, anchor={"middle": "mm", "start": "lm", "end": "rm"}[anchor])
        svg.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>'
        )

    def line(points: list[tuple[float, float]], color: str, width: int = 5, dashed: bool = False) -> None:
        if dashed:
            for a, b in zip(points[:-1], points[1:]):
                x1, y1 = a; x2, y2 = b
                pieces = 12
                for i in range(0, pieces, 2):
                    p1 = i / pieces; p2 = min((i + 1) / pieces, 1)
                    q1 = (x1 + (x2 - x1) * p1, y1 + (y2 - y1) * p1)
                    q2 = (x1 + (x2 - x1) * p2, y1 + (y2 - y1) * p2)
                    draw.line((q1, q2), fill=color, width=width)
                svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" stroke-dasharray="14 12"/>')
        else:
            draw.line(points, fill=color, width=width, joint="curve")
            svg.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="{width}"/>')

    text(WIDTH / 2, 72, "Support Number and Support Composition", 48, bold=True)
    text(WIDTH / 2, 137, "Action success on completed Ours evaluations", 29, color="#52606D")

    for panel_index, task in enumerate(TASKS):
        x0 = MARGIN_X + panel_index * (PANEL_WIDTH + PANEL_GAP)
        x1 = x0 + PANEL_WIDTH
        plot_left, plot_right = x0 + 92, x1 - 38
        color = COLORS[task]
        task_rows = [row for row in rows if row["task"] == task]
        ks = sorted({int(row["k"]) for row in task_rows})
        x_by_k = {k: plot_left + index * (plot_right - plot_left) / max(len(ks) - 1, 1) for index, k in enumerate(ks)}
        y_of = lambda value: PLOT_BOTTOM - (float(value) - 0.4) / 0.6 * (PLOT_BOTTOM - PLOT_TOP)

        draw.rounded_rectangle((x0, 180, x1, 960), radius=22, fill="#FAFBFC", outline="#D9E1E8", width=2)
        svg.append(f'<rect x="{x0:.1f}" y="180" width="{PANEL_WIDTH:.1f}" height="780" rx="22" fill="#FAFBFC" stroke="#D9E1E8" stroke-width="2"/>')
        text((x0 + x1) / 2, 216, task, 32, bold=True)
        for score in (0.4, 0.6, 0.8, 1.0):
            y = y_of(score)
            draw.line((plot_left, y, plot_right, y), fill="#DDE3E8", width=2)
            svg.append(f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" stroke="#DDE3E8" stroke-width="2"/>')
            text(plot_left - 18, y, f"{score * 100:.0f}", 22, anchor="end", color="#52606D")
        for k in ks:
            x = x_by_k[k]
            draw.line((x, PLOT_BOTTOM, x, PLOT_BOTTOM + 10), fill="#5C6773", width=2)
            text(x, PLOT_BOTTOM + 38, f"K={k}", 23, color="#394654")

        if task == "Mass Balance":
            main_rows = [row for row in task_rows if "unbalanced+balanced" not in row["variant"]]
        else:
            main_rows = task_rows
        points = [(x_by_k[int(row["k"])], y_of(row["action_success"])) for row in main_rows]
        line(points, color, dashed=(task == "Event80"))

        for row in task_rows:
            x = x_by_k[int(row["k"])]
            y = y_of(row["action_success"])
            special = "unbalanced+balanced" in row["variant"]
            radius = 11 if not special else 13
            fill = "white" if row["status"].startswith("diagnostic") else color
            if special:
                polygon = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
                draw.polygon(polygon, fill="#B45309", outline="#172033")
                svg.append(f'<polygon points="{x:.1f},{y-radius:.1f} {x+radius:.1f},{y:.1f} {x:.1f},{y+radius:.1f} {x-radius:.1f},{y:.1f}" fill="#B45309" stroke="#172033" stroke-width="2"/>')
            else:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=color, width=4)
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="4"/>')
            label = f"{float(row['action_success']) * 100:.1f}%".replace(".0%", "%")
            text(x, y - 31, label, 22, bold=True, color=color if not special else "#9A3412")
            if special:
                text(x + 20, y + 34, "balanced K2", 18, anchor="start", color="#9A3412")

        if task == "Event80":
            text((plot_left + plot_right) / 2, 925, "K2/K4 are overlap diagnostics", 19, color="#8A4B08")
        elif task == "Mass Balance":
            text((plot_left + plot_right) / 2, 925, "Diamond: informative unbalanced+balanced K2", 18, color="#8A4B08")
        else:
            text((plot_left + plot_right) / 2, 925, "Complementary red+blue K2 is sufficient", 19, color="#52606D")

    text(WIDTH / 2, 1025, "Support composition is controlled within each labeled protocol; points across tasks are not directly comparable.", 24, color="#52606D")
    text(WIDTH / 2, 1070, "Event80 action scores use the revised 2026-09-20 selection rule.", 23, color="#52606D")
    svg.extend(["</g>", "</svg>"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text("\n".join(svg) + "\n")
    image.save(PNG_PATH, dpi=(220, 220))
    print(SVG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
