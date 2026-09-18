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
    ("action_success", "Action Score", "max"),
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
    if marker in {"id_placeholder", "cross_dataset"}:
        return "*"
    if marker == "prior_protocol":
        return "\u2020"
    return ""


def ranking(tasks: list[dict], methods: list[dict]) -> dict[tuple[str, str, str], int]:
    ranks: dict[tuple[str, str, str], int] = {}
    for task in tasks:
        for metric, _, direction in METRICS:
            if metric in task.get("unranked_metrics", []):
                continue
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
                f"{task['id']}_psnr",
                f"{task['id']}_ssim",
                f"{task['id']}_lpips",
                f"{task['id']}_object_metric_value",
                f"{task['id']}_object_metric_name",
                f"{task['id']}_object_metric_unit",
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
                row[f"{task['id']}_psnr"] = values.get("psnr")
                row[f"{task['id']}_ssim"] = values.get("ssim")
                row[f"{task['id']}_lpips"] = values.get("lpips")
                row[f"{task['id']}_object_metric_value"] = values.get("object")
                row[f"{task['id']}_object_metric_name"] = task["object_metric"]
                row[f"{task['id']}_object_metric_unit"] = task["object_unit"]
                row[f"{task['id']}_action_success"] = values.get("action_success")
            writer.writerow(row)

    long_fields = [
        "task_id",
        "task",
        "method_id",
        "method",
        "psnr",
        "ssim",
        "lpips",
        "object_metric_name",
        "object_metric_unit",
        "object_metric_value",
        "action_success",
    ]
    with (output_dir / "sim_all_methods_main_table_detailed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for task in config["tasks"]:
            for method in config["methods"]:
                values = task["values"][method["id"]]
                writer.writerow(
                    {
                        "task_id": task["id"],
                        "task": task["label"],
                        "method_id": method["id"],
                        "method": method["label"],
                        "psnr": values.get("psnr"),
                        "ssim": values.get("ssim"),
                        "lpips": values.get("lpips"),
                        "object_metric_name": task["object_metric"],
                        "object_metric_unit": task["object_unit"],
                        "object_metric_value": values.get("object"),
                        "action_success": values.get("action_success"),
                    }
                )


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
        "Bold: best. Object/physical metrics are task-specific and listed in each header.",
        "* Mass Balance Ours uses the completed fixed-pose 5-ID/5-OOD test; baseline rows use workspace-random data.",
        "\u2020 Collision LoRA: prior protocol. Mass Friction action: +/-1 level at y=0.62/0.75/0.92 (post-hoc revision); --: unavailable.",
        "DINOv2: Transformer for Friction; concat-MLP for Gravity/Collision/Light/Balance. Mass x Friction is pending.",
    ]
    note_y = table_bottom + 48
    for index, note in enumerate(notes):
        y = note_y + index * 43
        svg_text(left, y, note, 20, anchor="start", css="note")
        draw.text((left, y), note, font=fonts["note"], fill="#3f4858", anchor="lm")

    svg.append("</svg>")
    (output_dir / "sim_all_methods_main_table.svg").write_text("\n".join(svg) + "\n")
    image.save(output_dir / "sim_all_methods_main_table.png")


def render_detailed(config: dict, output_dir: Path) -> None:
    """Detailed companion, sourced from the same values as the compact table."""
    widths = [310, 420, 180, 180, 180, 370, 240]
    left, top, header_height, row_height = 60, 170, 75, 64
    tasks, methods = config['tasks'], config['methods']
    count = len(tasks) * len(methods)
    bottom = top + header_height + count * row_height
    notes = [
        'Object metric: centroid ADE (px); Light Switch: lamp MAE; Mass Balance: bar-tilt MAE (deg).',
        'Action Score uses each task\'s recorded protocol; Mass x Friction uses first-crossing +/-1 level match.',
        '* Mass Balance Ours: fixed-pose; baselines: workspace-random. Not a matched-dataset comparison.',
        'Dagger: Mass Collision LoRA uses the earlier recorded protocol. Missing results are --, not zero.',
        'Historical runs and their recorded settings are retained. See protocol.json for provenance and caveats.',
    ]
    width, height = sum(widths) + 2 * left, bottom + 235
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g fill="#111" font-family="DejaVu Serif,serif">']
    def text(x, y, value, size=25, bold=False, center=False):
        draw.text((x, y), value, font=load_font(size, bold), fill='#111', anchor='mm' if center else 'lm')
        anchor = 'middle' if center else 'start'
        weight = 700 if bold else 400
        svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" font-size="{size}" font-weight="{weight}">{html.escape(value)}</text>')
    def line(y, weight):
        draw.line((left, y, width - left, y), fill='#111', width=weight)
        svg.append(f'<line x1="{left}" x2="{width-left}" y1="{y}" y2="{y}" stroke="#111" stroke-width="{weight}"/>')
    headers = ['Task', 'Method', 'PSNR (up)', 'SSIM (up)', 'LPIPS (down)', 'Object metric (down)', 'Action Score (up)']
    text(width / 2, 60, 'Simulation benchmark: detailed metrics', 40, True, True)
    text(width / 2, 115, 'Recorded formal evaluations; task-specific protocols and caveats retained', 25, center=True)
    line(top, 4)
    x = left
    for label, cell_width in zip(headers, widths):
        text(x + 12, top + header_height / 2, label, 23, True)
        x += cell_width
    line(top + header_height, 2)
    markdown = ['# Simulation benchmark: detailed metrics', '',
                '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    index = 0
    for task in tasks:
        for method in methods:
            value = task['values'][method['id']]
            def fmt(key, precision, scale=1):
                number = value.get(key)
                return '--' if number is None else f'{number * scale:.{precision}f}'
            action = fmt('action_success', 1, 100)
            if action != '--':
                action += '%'
            cells = [task['label'], method['label'] + marker_suffix(value.get('marker')),
                     fmt('psnr', 3), fmt('ssim', 4), fmt('lpips', 4),
                     fmt('object', 4) + ' ' + task['object_unit'], action]
            y = top + header_height + row_height * (index + 0.5)
            x = left
            for column, (cell, cell_width) in enumerate(zip(cells, widths)):
                text(x + 12, y, cell if column or method == methods[0] else '', 24,
                     bold=column == 1 and method['id'] == 'ours')
                x += cell_width
            markdown.append('| ' + ' | '.join(cells) + ' |')
            index += 1
        line(top + header_height + index * row_height, 1)
    line(bottom, 4)
    for index, note in enumerate(notes):
        text(left, bottom + 40 + index * 37, note, 21)
    svg.append('</g></svg>')
    stem = output_dir / 'sim_all_methods_main_table_detailed'
    stem.with_suffix('.svg').write_text('\n'.join(svg) + '\n')
    image.save(stem.with_suffix('.png'))
    markdown.extend(['', '## Scope and provenance', '', *['- ' + note for note in config.get('notes', [])],
                     '', *['- ' + note for note in notes], '',
                     'Detailed numeric data: `sim_all_methods_main_table_detailed.csv`.',
                     'Full recorded configuration and sources: `protocol.json`.', ''])
    stem.with_suffix('.md').write_text('\n'.join(markdown))


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
    render_detailed(config, output_dir)


if __name__ == "__main__":
    main()
