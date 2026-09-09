#!/usr/bin/env python3
"""Render the compact six-task simulation benchmark table."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


METRICS = (
    ("lpips", "LPIPS", "min"),
    ("object", "Object / Physical", "min"),
    ("action_success", "Action Success", "max"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/evaluation/sim_all_methods_main_table_v1.yaml",
    )
    return parser.parse_args()


def display_value(metric: str, value: float | None) -> str:
    if value is None:
        return "--"
    if metric == "lpips":
        return f"{value:.4f}"
    if metric == "action_success":
        return f"{100.0 * value:.1f}"
    return f"{value:.2f}"


def marker_suffix(marker: str | None) -> str:
    if marker == "id_placeholder":
        return "*"
    if marker == "prior_protocol":
        return "\u2020"
    return ""


def ranking(tasks: list[dict], methods: list[dict]) -> dict[tuple[str, str, str], int]:
    ranks: dict[tuple[str, str, str], int] = {}
    for task in tasks:
        for metric, _, direction in METRICS:
            eligible = []
            for method in methods:
                item = task["values"][method["id"]]
                value = item.get(metric)
                if value is not None and item.get("rank", True):
                    eligible.append((float(value), method["id"]))
            eligible.sort(reverse=(direction == "max"))
            distinct = []
            for value, _ in eligible:
                if not distinct or abs(value - distinct[-1]) > 1e-12:
                    distinct.append(value)
            for value, method_id in eligible:
                if distinct and abs(value - distinct[0]) <= 1e-12:
                    ranks[(task["id"], method_id, metric)] = 1
                elif len(distinct) > 1 and abs(value - distinct[1]) <= 1e-12:
                    ranks[(task["id"], method_id, metric)] = 2
    return ranks


def write_csv(config: dict, output_dir: Path) -> None:
    fields = ["method"]
    for task in config["tasks"]:
        fields.extend(
            [
                f"{task['id']}_lpips",
                f"{task['id']}_{task['object_metric'].lower().replace(' ', '_')}",
                f"{task['id']}_action_success",
            ]
        )
    with (output_dir / "sim_all_methods_main_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in config["methods"]:
            row = {"method": method["label"]}
            for task in config["tasks"]:
                values = task["values"][method["id"]]
                row[f"{task['id']}_lpips"] = values.get("lpips")
                row[f"{task['id']}_{task['object_metric'].lower().replace(' ', '_')}"] = values.get("object")
                row[f"{task['id']}_action_success"] = values.get("action_success")
            writer.writerow(row)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def render(config: dict, output_dir: Path) -> None:
    methods = config["methods"]
    tasks = config["tasks"]
    ranks = ranking(tasks, methods)
    width, height = 5000, 1280
    left, right = 40, 4960
    method_width = 600
    metric_width = (right - left - method_width) / (len(tasks) * len(METRICS))
    method_right = left + method_width
    table_top, group_bottom, header_bottom = 155, 275, 420
    row_height = 110
    table_bottom = header_bottom + len(methods) * row_height
    palette = ["#e8f0f8", "#eef4e7", "#f8eee5", "#f7f2df", "#e9f3f1", "#f0edf7"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"DejaVu Serif",serif;fill:#111827}.bold{font-weight:700}.muted{fill:#8a919e}.note{fill:#3f4858}</style>',
    ]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": load_font(52, True),
        "task": load_font(29, True),
        "header": load_font(23),
        "header_bold": load_font(27, True),
        "body": load_font(27),
        "body_bold": load_font(27, True),
        "note": load_font(20),
    }

    def svg_text(x: float, y: float, value: str, size: int, anchor: str = "middle", css: str = "") -> None:
        svg.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="{size}" class="{css}">{html.escape(value)}</text>'
        )

    svg_text(width / 2, 72, config["title"], 52, css="bold")
    draw.text((width / 2, 72), config["title"], font=fonts["title"], fill="#172033", anchor="mm")
    svg_text((left + method_right) / 2, (table_top + header_bottom) / 2, "Method", 29, css="bold")
    draw.text(((left + method_right) / 2, (table_top + header_bottom) / 2), "Method", font=fonts["header_bold"], fill="#111827", anchor="mm")

    for task_index, task in enumerate(tasks):
        group_x0 = method_right + task_index * len(METRICS) * metric_width
        group_x1 = group_x0 + len(METRICS) * metric_width
        color = palette[task_index]
        svg.append(f'<rect x="{group_x0:.1f}" y="{table_top}" width="{group_x1-group_x0:.1f}" height="{group_bottom-table_top}" fill="{color}"/>')
        draw.rectangle((group_x0, table_top, group_x1, group_bottom), fill=color)
        svg_text((group_x0 + group_x1) / 2, (table_top + group_bottom) / 2, task["label"], 29, css="bold")
        draw.text(((group_x0 + group_x1) / 2, (table_top + group_bottom) / 2), task["label"], font=fonts["task"], fill="#172033", anchor="mm")
        labels = (
            ("LPIPS", "down"),
            (task["object_metric"], f"({task['object_unit']}) down"),
            ("Success (%)", "up"),
        )
        for metric_index, (line1, line2) in enumerate(labels):
            x = group_x0 + (metric_index + 0.5) * metric_width
            y = (group_bottom + header_bottom) / 2
            svg_text(x, y - 16, line1, 22)
            svg_text(x, y + 17, line2, 20, css="note")
            draw.text((x, y - 16), line1, font=fonts["header"], fill="#111827", anchor="mm")
            draw.text((x, y + 17), line2, font=fonts["note"], fill="#3f4858", anchor="mm")

    for row_index, method in enumerate(methods):
        y_top = header_bottom + row_index * row_height
        y_bottom = y_top + row_height
        y = (y_top + y_bottom) / 2
        fill = "#f8f9fb" if row_index % 2 == 0 else "white"
        if method["id"] == "ours":
            fill = "#eaf4ef"
        svg.append(f'<rect x="{left}" y="{y_top}" width="{right-left}" height="{row_height}" fill="{fill}"/>')
        draw.rectangle((left, y_top, right, y_bottom), fill=fill)
        method_bold = method["id"] == "ours"
        svg_text(left + 28, y, method["label"], 28, anchor="start", css="bold" if method_bold else "")
        draw.text((left + 28, y), method["label"], font=fonts["body_bold" if method_bold else "body"], fill="#111827", anchor="lm")

        for task_index, task in enumerate(tasks):
            item = task["values"][method["id"]]
            suffix = marker_suffix(item.get("marker"))
            for metric_index, (metric, _, _) in enumerate(METRICS):
                value = item.get(metric)
                text_value = display_value(metric, value)
                if suffix and value is not None:
                    text_value += suffix
                x = method_right + (task_index * len(METRICS) + metric_index + 0.5) * metric_width
                rank = ranks.get((task["id"], method["id"], metric))
                css = "bold" if rank == 1 else ("muted" if value is None else "")
                svg_text(x, y, text_value, 27, css=css)
                font = fonts["body_bold" if rank == 1 else "body"]
                color = "#111827" if value is not None else "#8a919e"
                draw.text((x, y), text_value, font=font, fill=color, anchor="mm")
                if rank == 2:
                    bbox = draw.textbbox((x, y), text_value, font=font, anchor="mm")
                    underline_y = bbox[3] + 3
                    svg.append(f'<line x1="{bbox[0]}" y1="{underline_y}" x2="{bbox[2]}" y2="{underline_y}" stroke="#111827" stroke-width="2"/>')
                    draw.line((bbox[0], underline_y, bbox[2], underline_y), fill="#111827", width=2)

    rules = (
        (left, table_top, right, table_top, 5),
        (left, group_bottom, right, group_bottom, 2),
        (left, header_bottom, right, header_bottom, 4),
        (left, table_bottom, right, table_bottom, 5),
        (method_right, table_top, method_right, table_bottom, 2),
    )
    for x1, y1, x2, y2, line_width in rules:
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#25324a" stroke-width="{line_width}"/>')
        draw.line((x1, y1, x2, y2), fill="#25324a", width=line_width)
    for task_index in range(1, len(tasks)):
        x = method_right + task_index * len(METRICS) * metric_width
        svg.append(f'<line x1="{x:.1f}" y1="{table_top}" x2="{x:.1f}" y2="{table_bottom}" stroke="#9aa2af" stroke-width="2"/>')
        draw.line((x, table_top, x, table_bottom), fill="#9aa2af", width=2)

    notes = [
        "Bold: best; underline: second-best. Object/physical metrics are task-specific and listed in each header.",
        "* Mass Balance Ours is an ID-only fixed-pose placeholder; available values participate in column ranking.",
        "\u2020 Mass Collision LoRA uses the earlier compatible no-leak balanced-support protocol. -- indicates pending/unavailable.",
        "DINOv2 fusion: Transformer for Friction/Gravity/Collision; concat-MLP for Light/Balance. Mass x Friction DINO and TTT-KQV are pending.",
    ]
    note_y = table_bottom + 48
    for index, note in enumerate(notes):
        y = note_y + index * 43
        svg_text(left, y, note, 20, anchor="start", css="note")
        draw.text((left, y), note, font=fonts["note"], fill="#3f4858", anchor="lm")

    svg.append("</svg>")
    (output_dir / "sim_all_methods_main_table.svg").write_text("\n".join(svg) + "\n")
    image.save(output_dir / "sim_all_methods_main_table.png")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config, output_dir)
    with (output_dir / "protocol.json").open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    render(config, output_dir)


if __name__ == "__main__":
    main()
