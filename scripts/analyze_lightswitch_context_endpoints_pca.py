#!/usr/bin/env python3
"""Plot LightSwitch Stage1 contexts as circles and Stage2 endpoints as triangles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_ORDER = ("neither", "red_only", "blue_only", "both")
CLASS_COLORS = {
    "neither": "#6B7280",
    "red_only": "#D1495B",
    "blue_only": "#2878B5",
    "both": "#2A9D8F",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def load_font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def triangle_points(x: float, y: float, radius: float = 13.0):
    return [
        (x, y - radius),
        (x - radius * 0.9, y + radius * 0.72),
        (x + radius * 0.9, y + radius * 0.72),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--active-groups", required=True)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    metadata = read_jsonl(Path(args.metadata_path))
    group_classes: dict[int, str] = {}
    for row in metadata:
        group = int(round(float(row["friction_mu"])))
        causal_class = str(row["causal_class"])
        previous = group_classes.setdefault(group, causal_class)
        if previous != causal_class:
            raise ValueError(f"Group {group} maps to both {previous} and {causal_class}.")

    records = json.loads(Path(args.table_path).read_text())["records"]
    table_groups = np.asarray(
        [int(round(float(row["friction_mu"]))) for row in records], dtype=np.int64
    )
    table_contexts = np.stack(
        [np.asarray(row["context"], dtype=np.float64).reshape(-1) for row in records]
    )
    active_set = set(parse_ints(args.active_groups))
    active_mask = np.asarray([int(group) in active_set for group in table_groups], dtype=bool)
    if int(active_mask.sum()) < 3:
        raise ValueError("At least three active groups are required for PCA.")

    center = table_contexts[active_mask].mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(
        table_contexts[active_mask] - center, full_matrices=False
    )
    variance = singular**2
    explained = variance / max(float(variance.sum()), 1e-12)
    table_scores = (table_contexts - center) @ components[:2].T

    source_indices = parse_ints(args.source_indices)
    trajectory_by_source: dict[int, list[dict]] = {source: [] for source in source_indices}
    for row in read_jsonl(Path(args.trajectory_path)):
        source = int(row["sample_index"])
        if source in trajectory_by_source:
            trajectory_by_source[source].append(row)

    endpoints = []
    for source in source_indices:
        rows = sorted(trajectory_by_source[source], key=lambda row: int(row["inner_step"]))
        if not rows:
            raise ValueError(f"No Stage2 trajectory found for source {source}.")
        initial = np.asarray(rows[0]["context_flat"], dtype=np.float64).reshape(-1)
        final = np.asarray(rows[-1]["context_flat"], dtype=np.float64).reshape(-1)
        causal_class = str(metadata[source]["causal_class"])
        endpoints.append(
            {
                "source_index": source,
                "causal_class": causal_class,
                "group": int(round(float(metadata[source]["friction_mu"]))),
                "initial_context": initial,
                "final_context": final,
                "initial_score": ((initial - center[0]) @ components[:2].T),
                "final_score": ((final - center[0]) @ components[:2].T),
                "inner_step": int(rows[-1]["inner_step"]),
                "support_indices": rows[-1].get("support_indices", []),
            }
        )

    displayed = np.concatenate(
        [table_scores[active_mask]]
        + [np.stack([item["initial_score"], item["final_score"]]) for item in endpoints],
        axis=0,
    )
    x_min, x_max = float(displayed[:, 0].min()), float(displayed[:, 0].max())
    y_min, y_max = float(displayed[:, 1].min()), float(displayed[:, 1].max())
    x_pad = max((x_max - x_min) * 0.16, 1e-6)
    y_pad = max((y_max - y_min) * 0.18, 1e-6)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    width, height = 1500, 960
    left, top, right, bottom = 125, 170, 1110, 820

    def project(score):
        x = left + (float(score[0]) - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (float(score[1]) - y_min) / (y_max - y_min) * (bottom - top)
        return float(x), float(y)

    image = Image.new("RGB", (width, height), "#F7F3EA")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (38, 32, width - 38, height - 32),
        22,
        fill="#FFFCF5",
        outline="#CBC4B7",
        width=2,
    )
    draw.text((80, 64), args.title, font=load_font(29, True), fill="#17212B")
    subtitle = (
        f"{int(active_mask.sum())} active Stage1 groups (circles) | "
        f"{len(endpoints)} Stage2 endpoints (triangles)"
    )
    draw.text((82, 111), subtitle, font=load_font(17), fill="#5B6470")
    for fraction in np.linspace(0.0, 1.0, 6):
        x = left + fraction * (right - left)
        y = top + fraction * (bottom - top)
        draw.line((x, top, x, bottom), fill="#DED8CC", width=1)
        draw.line((left, y, right, y), fill="#DED8CC", width=1)
    draw.rectangle((left, top, right, bottom), outline="#C8C1B5", width=2)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#F7F3EA"/>',
        f'<rect x="38" y="32" width="{width-76}" height="{height-64}" rx="22" fill="#FFFCF5" stroke="#CBC4B7" stroke-width="2"/>',
        f'<text x="80" y="98" font-family="DejaVu Sans" font-size="29" font-weight="700" fill="#17212B">{args.title}</text>',
        f'<text x="82" y="136" font-family="DejaVu Sans" font-size="17" fill="#5B6470">{subtitle}</text>',
    ]
    for fraction in np.linspace(0.0, 1.0, 6):
        x = left + fraction * (right - left)
        y = top + fraction * (bottom - top)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#DED8CC"/>'
        )
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#DED8CC"/>'
        )
    svg.append(
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#C8C1B5" stroke-width="2"/>'
    )

    for group, score, is_active in zip(table_groups, table_scores, active_mask):
        if not is_active:
            continue
        causal_class = group_classes[int(group)]
        color = CLASS_COLORS[causal_class]
        x, y = project(score)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline="white", width=2)
        draw.text((x + 11, y - 16), f"g{int(group)}", font=load_font(13, True), fill="#343A40")
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#FFFDF8" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{x+12:.1f}" y="{y-6:.1f}" font-family="DejaVu Sans" font-size="13" font-weight="700" fill="#343A40">g{int(group)}</text>'
        )

    output_rows = []
    for item in endpoints:
        color = CLASS_COLORS[item["causal_class"]]
        start_x, start_y = project(item["initial_score"])
        end_x, end_y = project(item["final_score"])
        draw.line((start_x, start_y, end_x, end_y), fill=color, width=3)
        points = triangle_points(end_x, end_y)
        draw.polygon(points, fill=color, outline="#17212B")
        draw.text(
            (end_x + 12, end_y + 5),
            item["causal_class"],
            font=load_font(13, True),
            fill="#28313A",
        )
        svg.append(
            f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="{color}" stroke-width="3" opacity=".72"/>'
        )
        svg_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        svg.append(
            f'<polygon points="{svg_points}" fill="{color}" stroke="#17212B" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{end_x+13:.1f}" y="{end_y+15:.1f}" font-family="DejaVu Sans" font-size="13" font-weight="700" fill="#28313A">{item["causal_class"]}</text>'
        )
        active_indices = np.where(active_mask)[0]
        distances = np.linalg.norm(table_contexts[active_mask] - item["final_context"], axis=1)
        nearest_index = active_indices[int(np.argmin(distances))]
        output_rows.append(
            {
                "source_index": item["source_index"],
                "causal_class": item["causal_class"],
                "context_group_id": item["group"],
                "inner_step": item["inner_step"],
                "pc1_final": float(item["final_score"][0]),
                "pc2_final": float(item["final_score"][1]),
                "context_delta_l2": float(
                    np.linalg.norm(item["final_context"] - item["initial_context"])
                ),
                "nearest_stage1_group": int(table_groups[nearest_index]),
                "nearest_stage1_l2": float(distances.min()),
                "support_indices": ",".join(str(value) for value in item["support_indices"]),
            }
        )

    legend_x = 1170
    draw.text((legend_x, 190), "ENVIRONMENT", font=load_font(17, True), fill="#25313A")
    svg.append(
        f'<text x="{legend_x}" y="210" font-family="DejaVu Sans" font-size="17" font-weight="700" fill="#25313A">ENVIRONMENT</text>'
    )
    for index, causal_class in enumerate(CLASS_ORDER):
        y = 240 + index * 60
        color = CLASS_COLORS[causal_class]
        draw.ellipse((legend_x, y, legend_x + 22, y + 22), fill=color)
        draw.text((legend_x + 34, y - 1), causal_class, font=load_font(15, True), fill="#2D3338")
        svg.append(f'<circle cx="{legend_x+11}" cy="{y+11}" r="11" fill="{color}"/>')
        svg.append(
            f'<text x="{legend_x+34}" y="{y+16}" font-family="DejaVu Sans" font-size="15" font-weight="700" fill="#2D3338">{causal_class}</text>'
        )

    marker_y = 530
    draw.ellipse((legend_x, marker_y, legend_x + 20, marker_y + 20), fill="#64748B")
    draw.text((legend_x + 33, marker_y - 1), "Stage1 table", font=load_font(14), fill="#4B5563")
    triangle = triangle_points(legend_x + 10, marker_y + 65, 11)
    draw.polygon(triangle, fill="#64748B", outline="#17212B")
    draw.text((legend_x + 33, marker_y + 53), "Stage2 adapted", font=load_font(14), fill="#4B5563")
    svg.append(
        f'<circle cx="{legend_x+10}" cy="{marker_y+10}" r="10" fill="#64748B"/>'
    )
    svg.append(
        f'<text x="{legend_x+33}" y="{marker_y+15}" font-family="DejaVu Sans" font-size="14" fill="#4B5563">Stage1 table</text>'
    )
    triangle_svg = " ".join(f"{x:.1f},{y:.1f}" for x, y in triangle)
    svg.append(
        f'<polygon points="{triangle_svg}" fill="#64748B" stroke="#17212B" stroke-width="2"/>'
    )
    svg.append(
        f'<text x="{legend_x+33}" y="{marker_y+70}" font-family="DejaVu Sans" font-size="14" fill="#4B5563">Stage2 adapted</text>'
    )

    x_label = f"PC1 ({explained[0] * 100:.1f}% active-table variance)"
    y_label = f"PC2 ({explained[1] * 100:.1f}% active-table variance)"
    draw.text((450, 865), x_label, font=load_font(17, True), fill="#3B4147")
    draw.text((1165, 700), y_label, font=load_font(15, True), fill="#3B4147")
    svg.append(
        f'<text x="{(left+right)/2:.1f}" y="890" text-anchor="middle" font-family="DejaVu Sans" font-size="17" font-weight="700" fill="#3B4147">{x_label}</text>'
    )
    svg.append(
        f'<text x="1165" y="720" font-family="DejaVu Sans" font-size="15" font-weight="700" fill="#3B4147">{y_label}</text>'
    )
    svg.append("</svg>")

    png_path = Path(args.output_png)
    svg_path = Path(args.output_svg)
    csv_path = Path(args.output_csv)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    svg_path.write_text("\n".join(svg))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[done] png={png_path} svg={svg_path} csv={csv_path}")


if __name__ == "__main__":
    main()
