#!/usr/bin/env python3
"""Render the completed support-number/action-success analysis."""

from __future__ import annotations

import csv
import html
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/support_number_analysis"
CSV_PATH = RESULT_DIR / "support_number_summary.csv"
SVG_PATH = RESULT_DIR / "support_number_action_success.svg"
PNG_PATH = RESULT_DIR / "support_number_action_success.png"
TABLE_SVG_PATH = RESULT_DIR / "support_number_summary_table.svg"
TABLE_PNG_PATH = RESULT_DIR / "support_number_summary_table.png"
METRICS_CSV_PATH = RESULT_DIR / "support_number_metrics.csv"
METRICS_SVG_PATH = RESULT_DIR / "support_number_metrics_table.svg"
METRICS_PNG_PATH = RESULT_DIR / "support_number_metrics_table.png"
METRICS_MD_PATH = RESULT_DIR / "support_number_metrics_table.md"

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


def render_table(rows: list[dict[str, str]]) -> None:
    columns = [
        ("task", "Task", 360),
        ("variant", "Setting", 580),
        ("k", "K", 130),
        ("support_construction", "Support construction", 930),
        ("action_success", "Overall", 260),
        ("action_success_id", "ID", 230),
        ("action_success_ood", "OOD", 230),
        ("status", "Protocol status", 470),
    ]
    left, right, top = 95, 95, 250
    header_height, row_height = 100, 108
    width = left + sum(column[2] for column in columns) + right
    bottom = top + header_height + row_height * len(rows)
    height = bottom + 225
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Serif, serif" fill="#172033">',
    ]

    def put_text(x: float, y: float, value: str, size: int, *, bold: bool = False,
                 anchor: str = "middle", color: str = "#172033") -> None:
        draw.text((x, y), value, font=font(size, bold), fill=color,
                  anchor={"middle": "mm", "start": "lm", "end": "rm"}[anchor])
        svg.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>'
        )

    def rule(y: float, width_px: int = 2, color: str = "#263442") -> None:
        draw.line((left, y, width - right, y), fill=color, width=width_px)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{color}" stroke-width="{width_px}"/>')

    def percent(value: str) -> str:
        return "--" if value == "" else f"{float(value) * 100:.1f}%".replace(".0%", "%")

    def status_label(value: str) -> str:
        return {
            "formal": "Formal",
            "complete": "Complete",
            "diagnostic_support_query_overlap": "Overlap diagnostic (mean)",
            "diagnostic_support_query_overlap_sum_loss": "Overlap diagnostic (sum)",
        }.get(value, value.replace("_", " "))

    put_text(width / 2, 68, "Support-Number Ablation Summary", 50, bold=True)
    put_text(width / 2, 137, "Action-selection results under each recorded support protocol", 29, color="#52606D")
    put_text(width / 2, 185, "Event80 K=2/K=4 rows are overlap diagnostics and are not formal disjoint-query evaluations.", 25, color="#8A4B08")
    rule(top, 5)
    rule(top + header_height, 2)

    x = left
    for _, label, column_width in columns:
        put_text(x + column_width / 2, top + header_height / 2, label, 28, bold=True)
        x += column_width

    previous_task = None
    for index, row in enumerate(rows):
        y0 = top + header_height + index * row_height
        y1 = y0 + row_height
        if index % 2:
            draw.rectangle((left, y0, width - right, y1), fill="#F7F9FB")
            svg.append(f'<rect x="{left}" y="{y0}" width="{width-left-right}" height="{row_height}" fill="#F7F9FB"/>')
        if previous_task is not None and row["task"] != previous_task:
            rule(y0, 3, "#7A8793")
        previous_task = row["task"]

        values = [
            row["task"], row["variant"], row["k"], row["support_construction"],
            percent(row["action_success"]), percent(row["action_success_id"]),
            percent(row["action_success_ood"]), status_label(row["status"]),
        ]
        wrap_widths = [18, 31, 4, 52, 12, 10, 10, 29]
        x = left
        for column_index, ((_, _, column_width), value) in enumerate(zip(columns, values)):
            lines = textwrap.wrap(value, width=wrap_widths[column_index]) or [""]
            line_gap = 29
            start_y = (y0 + y1) / 2 - (len(lines) - 1) * line_gap / 2
            align = "start" if column_index in (0, 1, 3, 7) else "middle"
            text_x = x + 18 if align == "start" else x + column_width / 2
            for line_index, line_value in enumerate(lines[:3]):
                put_text(text_x, start_y + line_index * line_gap, line_value, 23,
                         bold=(column_index == 4), anchor=align)
            x += column_width
    rule(bottom, 5)
    put_text(left, bottom + 60,
             "Overall/ID/OOD report action success. Missing domain splits are shown as --, not zero.",
             24, anchor="start", color="#52606D")
    put_text(left, bottom + 108,
             "Mean-loss averages support losses; sum-loss preserves their sum and therefore increases effective update magnitude with K.",
             24, anchor="start", color="#52606D")
    put_text(left, bottom + 156,
             "Source values and result roots are recorded in support_number_summary.csv.",
             24, anchor="start", color="#52606D")
    svg.extend(["</g>", "</svg>"])
    TABLE_SVG_PATH.write_text("\n".join(svg) + "\n")
    image.save(TABLE_PNG_PATH, dpi=(220, 220))


def render_metrics_table() -> None:
    with METRICS_CSV_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = [
        ("task", "Task", 330), ("variant", "Setting", 510), ("k", "K", 105),
        ("psnr", "PSNR up", 225), ("ssim", "SSIM up", 225), ("lpips", "LPIPS down", 225),
        ("object_mean", "Object mean down", 300), ("object_final", "Object final down", 300),
        ("physical_error", "Physical error down", 310),
        ("action_success", "Action success up", 300),
    ]
    left, right, top = 90, 90, 235
    header_height, row_height = 105, 90
    width = left + sum(column[2] for column in columns) + right
    bottom = top + header_height + len(rows) * row_height
    height = bottom + 260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Serif, serif" fill="#172033">',
    ]

    def put_text(x: float, y: float, value: str, size: int, *, bold: bool = False,
                 anchor: str = "middle", color: str = "#172033") -> None:
        draw.text((x, y), value, font=font(size, bold), fill=color,
                  anchor={"middle": "mm", "start": "lm", "end": "rm"}[anchor])
        svg.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>'
        )

    def rule(y: float, width_px: int = 2, color: str = "#263442") -> None:
        draw.line((left, y, width - right, y), fill=color, width=width_px)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{color}" stroke-width="{width_px}"/>')

    def metric_value(key: str, value: str) -> str:
        if value == "":
            return "--"
        number = float(value)
        if key == "psnr":
            return f"{number:.3f}"
        if key in ("ssim", "lpips"):
            return f"{number:.4f}"
        if key == "action_success":
            return f"{number * 100:.1f}%".replace(".0%", "%")
        return f"{number:.3f}"

    put_text(width / 2, 66, "Support-Number Ablation: Detailed Metrics", 50, bold=True)
    put_text(width / 2, 132, "Global appearance, object-centric accuracy, physical consistency, and action selection", 28, color="#52606D")
    put_text(width / 2, 181, "Arrows indicate metric direction; values are aggregated under each recorded protocol.", 24, color="#52606D")
    rule(top, 5)
    rule(top + header_height, 2)
    x = left
    for _, label, column_width in columns:
        label = label.replace(" up", "\n(higher)").replace(" down", "\n(lower)")
        lines = label.splitlines()
        for line_index, line_value in enumerate(lines):
            put_text(x + column_width / 2,
                     top + header_height / 2 + (line_index - (len(lines)-1)/2) * 29,
                     line_value, 25 if line_index == 0 else 21, bold=(line_index == 0))
        x += column_width

    previous_task = None
    for index, row in enumerate(rows):
        y0 = top + header_height + index * row_height
        y1 = y0 + row_height
        if index % 2:
            draw.rectangle((left, y0, width - right, y1), fill="#F7F9FB")
            svg.append(f'<rect x="{left}" y="{y0}" width="{width-left-right}" height="{row_height}" fill="#F7F9FB"/>')
        if previous_task is not None and row["task"] != previous_task:
            rule(y0, 3, "#7A8793")
        previous_task = row["task"]
        x = left
        for key, _, column_width in columns:
            raw = row[key]
            value = raw if key in ("task", "variant", "k") else metric_value(key, raw)
            align = "start" if key in ("task", "variant") else "middle"
            put_text(x + (16 if align == "start" else column_width / 2), (y0 + y1) / 2,
                     value, 23, bold=(key == "action_success"), anchor=align)
            x += column_width
    rule(bottom, 5)
    notes = [
        "Event80 object: pushed-block centroid ADE/FDE (px); no separate physical-error column.",
        "Light Switch object: yellow-light score MAE/final absolute error; physical: transition-time absolute error (frames).",
        "Mass Balance object: bar-centroid ADE/FDE (px); physical: beam-tilt MAE (deg).",
        "Event80 K=2/K=4 mean/sum rows retain support-query overlap and are diagnostic, not formal disjoint evaluations.",
    ]
    for index, note in enumerate(notes):
        put_text(left, bottom + 48 + index * 45, note, 23, anchor="start",
                 color="#8A4B08" if index == 3 else "#52606D")
    svg.extend(["</g>", "</svg>"])
    METRICS_SVG_PATH.write_text("\n".join(svg) + "\n")
    image.save(METRICS_PNG_PATH, dpi=(220, 220))

    headers = [column[1].replace(" up", " higher").replace(" down", " lower") for column in columns]
    lines = ["# Support-number ablation: detailed metrics", "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [row[key] if key in ("task", "variant", "k") else metric_value(key, row[key])
                  for key, _, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", *["- " + note for note in notes], ""]
    METRICS_MD_PATH.write_text("\n".join(lines))


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
        elif task == "Event80":
            main_rows = [row for row in task_rows if "sum-loss" not in row["variant"]]
        else:
            main_rows = task_rows
        points = [(x_by_k[int(row["k"])], y_of(row["action_success"])) for row in main_rows]
        line(points, color, dashed=(task == "Event80"))
        if task == "Event80":
            sum_rows = [row for row in task_rows if int(row["k"]) == 1 or "sum-loss" in row["variant"]]
            sum_points = [(x_by_k[int(row["k"])], y_of(row["action_success"])) for row in sum_rows]
            line(sum_points, color, width=4)

        for row in task_rows:
            x = x_by_k[int(row["k"])]
            y = y_of(row["action_success"])
            special = "unbalanced+balanced" in row["variant"]
            sum_loss = "sum-loss" in row["variant"]
            radius = 11 if not special else 13
            fill = "white" if row["status"].startswith("diagnostic") else color
            if special:
                polygon = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
                draw.polygon(polygon, fill="#B45309", outline="#172033")
                svg.append(f'<polygon points="{x:.1f},{y-radius:.1f} {x+radius:.1f},{y:.1f} {x:.1f},{y+radius:.1f} {x-radius:.1f},{y:.1f}" fill="#B45309" stroke="#172033" stroke-width="2"/>')
            elif sum_loss:
                draw.rectangle((x-radius, y-radius, x+radius, y+radius), fill=color, outline="#172033", width=3)
                svg.append(f'<rect x="{x-radius:.1f}" y="{y-radius:.1f}" width="{2*radius}" height="{2*radius}" fill="{color}" stroke="#172033" stroke-width="3"/>')
            else:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=color, width=4)
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="4"/>')
            label = f"{float(row['action_success']) * 100:.1f}%".replace(".0%", "%")
            label_y = y + 32 if sum_loss else y - 31
            text(x, label_y, label, 22, bold=True, color=color if not special else "#9A3412")
            if special:
                text(x + 20, y + 34, "balanced K2", 18, anchor="start", color="#9A3412")

        if task == "Event80":
            text((plot_left + plot_right) / 2, 912, "Hollow/dashed: mean loss; square/solid: sum loss", 17, color="#8A4B08")
            text((plot_left + plot_right) / 2, 938, "K2/K4 are overlap diagnostics", 17, color="#8A4B08")
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
    render_table(rows)
    render_metrics_table()
    print(SVG_PATH)
    print(PNG_PATH)
    print(TABLE_SVG_PATH)
    print(TABLE_PNG_PATH)
    print(METRICS_SVG_PATH)
    print(METRICS_PNG_PATH)
    print(METRICS_MD_PATH)


if __name__ == "__main__":
    main()
