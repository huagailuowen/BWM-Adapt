#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_ORDER = ("neither", "red_only", "blue_only", "both")
CLASS_COLORS = {
    "neither": "#6B7280",
    "red_only": "#D1495B",
    "blue_only": "#2878B5",
    "both": "#2A9D8F",
}
DISPLAY_STEPS = (0, 2, 5, 8, 10, 20, 30, 40, 60, 80)


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


def star_points(x: float, y: float, outer: float = 13, inner: float = 5.5):
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser("Plot categorical LightSwitch C-table PCA and Stage2 trajectories.")
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
    group_classes = {}
    for row in metadata:
        group = int(round(float(row["friction_mu"])))
        causal_class = str(row["causal_class"])
        previous = group_classes.setdefault(group, causal_class)
        if previous != causal_class:
            raise ValueError(f"Group {group} maps to both {previous} and {causal_class}.")

    records = json.loads(Path(args.table_path).read_text())["records"]
    groups = np.asarray([int(round(float(row["friction_mu"]))) for row in records], dtype=np.int64)
    contexts = np.stack([np.asarray(row["context"], dtype=np.float64).reshape(-1) for row in records])
    active_groups = set(parse_ints(args.active_groups))
    active_mask = np.asarray([int(group) in active_groups for group in groups], dtype=bool)
    if int(active_mask.sum()) < 3:
        raise ValueError("At least three active groups are required for a two-dimensional PCA plot.")

    center = contexts[active_mask].mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(contexts[active_mask] - center, full_matrices=False)
    variance = singular ** 2
    explained = variance / max(float(variance.sum()), 1e-12)
    table_scores = (contexts - center) @ components[:2].T

    grouped_trajectory = defaultdict(list)
    for row in read_jsonl(Path(args.trajectory_path)):
        grouped_trajectory[int(row["sample_index"])].append(row)
    source_indices = parse_ints(args.source_indices)

    plotted_trajectories = []
    output_rows = []
    for source_index in source_indices:
        rows = sorted(grouped_trajectory[source_index], key=lambda row: int(row["inner_step"]))
        if not rows:
            raise ValueError(f"No trajectory found for sample {source_index}.")
        trajectory_contexts = np.stack([
            np.asarray(row["context_flat"], dtype=np.float64).reshape(-1) for row in rows
        ])
        trajectory_scores = (trajectory_contexts - center) @ components[:2].T
        steps = np.asarray([int(row["inner_step"]) for row in rows], dtype=np.int64)
        final_step = int(steps[-1])
        targets = sorted(set(DISPLAY_STEPS + (final_step,)))
        selected = []
        for target in targets:
            if target > final_step:
                continue
            index = int(np.argmin(np.abs(steps - target)))
            if index not in selected:
                selected.append(index)
        selected_scores = trajectory_scores[selected]
        selected_steps = steps[selected]
        sample = metadata[source_index]
        group = int(round(float(sample["friction_mu"])))
        causal_class = str(sample["causal_class"])
        plotted_trajectories.append({
            "source_index": source_index,
            "group": group,
            "class": causal_class,
            "scores": selected_scores,
            "steps": selected_steps,
            "final_step": final_step,
        })

        path_length = float(np.linalg.norm(np.diff(trajectory_contexts, axis=0), axis=1).sum())
        total_delta = float(np.linalg.norm(trajectory_contexts[-1] - trajectory_contexts[0]))
        active_indices = np.where(active_mask)[0]
        final_distances = np.linalg.norm(contexts[active_mask] - trajectory_contexts[-1], axis=1)
        nearest_index = active_indices[int(np.argmin(final_distances))]
        output_rows.append({
            "sample_index": source_index,
            "context_group_id": group,
            "causal_class": causal_class,
            "episode_index": sample.get("episode_index"),
            "inner_steps": final_step,
            "context_total_delta_l2": total_delta,
            "context_path_l2": path_length,
            "pc1_initial": float(trajectory_scores[0, 0]),
            "pc2_initial": float(trajectory_scores[0, 1]),
            "pc1_final": float(trajectory_scores[-1, 0]),
            "pc2_final": float(trajectory_scores[-1, 1]),
            "nearest_active_group_final": int(groups[nearest_index]),
            "nearest_active_class_final": group_classes[int(groups[nearest_index])],
            "nearest_active_l2_final": float(final_distances.min()),
        })

    displayed = [table_scores[active_mask]] + [item["scores"] for item in plotted_trajectories]
    displayed = np.concatenate(displayed, axis=0)
    x_min, x_max = float(displayed[:, 0].min()), float(displayed[:, 0].max())
    y_min, y_max = float(displayed[:, 1].min()), float(displayed[:, 1].max())
    x_pad = max((x_max - x_min) * 0.12, 1e-6)
    y_pad = max((y_max - y_min) * 0.14, 1e-6)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    width, height = 1800, 1200
    left, top, right, bottom = 145, 190, 1320, 1030

    def project(point):
        x, y = point
        px = left + (x - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return float(px), float(py)

    image = Image.new("RGB", (width, height), "#F7F3EA")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((50, 40, width - 50, height - 40), 24, fill="#FFFCF5", outline="#CBC4B7", width=2)
    draw.text((95, 72), args.title, font=load_font(34, True), fill="#17212B")
    subtitle = f"{int(active_mask.sum())} active Stage1 groups | Stage2 trajectories end at step {max(item['final_step'] for item in plotted_trajectories)}"
    draw.text((98, 125), subtitle, font=load_font(18), fill="#5B6470")
    for fraction in np.linspace(0, 1, 6):
        px = left + fraction * (right - left)
        py = top + fraction * (bottom - top)
        draw.line((px, top, px, bottom), fill="#DED8CC", width=1)
        draw.line((left, py, right, py), fill="#DED8CC", width=1)
    draw.rectangle((left, top, right, bottom), outline="#C8C1B5", width=2)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#F7F3EA"/>',
        f'<rect x="50" y="40" width="{width-100}" height="{height-80}" rx="24" fill="#FFFCF5" stroke="#CBC4B7" stroke-width="2"/>',
        f'<text x="95" y="108" font-family="DejaVu Sans" font-size="34" font-weight="700" fill="#17212B">{escape(args.title)}</text>',
        f'<text x="98" y="150" font-family="DejaVu Sans" font-size="18" fill="#5B6470">{escape(subtitle)}</text>',
    ]
    for fraction in np.linspace(0, 1, 6):
        px = left + fraction * (right - left)
        py = top + fraction * (bottom - top)
        svg.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}" stroke="#DED8CC"/>')
        svg.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="#DED8CC"/>')
    svg.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#C8C1B5" stroke-width="2"/>')

    for group, score, is_active in zip(groups, table_scores, active_mask):
        if not is_active:
            continue
        causal_class = group_classes[int(group)]
        color = CLASS_COLORS[causal_class]
        x, y = project(score)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline="white", width=2)
        draw.text((x + 10, y - 17), f"g{int(group)}", font=load_font(13, True), fill="#333842")
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#FFFDF8" stroke-width="2"/>')
        svg.append(f'<text x="{x+11:.1f}" y="{y-7:.1f}" font-family="DejaVu Sans" font-size="13" font-weight="700" fill="#333842">g{int(group)}</text>')

    for item in plotted_trajectories:
        color = CLASS_COLORS[item["class"]]
        projected = [project(score) for score in item["scores"]]
        for start, end in zip(projected, projected[1:]):
            draw.line((*start, *end), fill=color, width=4)
            svg.append(f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="{color}" stroke-width="4" opacity=".85"/>')
        for index, ((x, y), step) in enumerate(zip(projected, item["steps"])):
            step = int(step)
            if index == 0:
                diamond = [(x, y - 10), (x + 10, y), (x, y + 10), (x - 10, y)]
                draw.polygon(diamond, fill="#111827", outline="white")
                svg.append(f'<polygon points="{x:.1f},{y-10:.1f} {x+10:.1f},{y:.1f} {x:.1f},{y+10:.1f} {x-10:.1f},{y:.1f}" fill="#111827" stroke="white"/>')
            elif step == item["final_step"]:
                draw.polygon(star_points(x, y), fill=color, outline="#17212B")
                points = " ".join(f"{px:.1f},{py:.1f}" for px, py in star_points(x, y))
                svg.append(f'<polygon points="{points}" fill="{color}" stroke="#17212B" stroke-width="1.5"/>')
            else:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white")
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white"/>')
            if step in (0, 10, 20, 40, 60, 80, item["final_step"]):
                draw.text((x + 7, y + 6), str(step), font=load_font(12, True), fill="#374151")
                svg.append(f'<text x="{x+7:.1f}" y="{y+18:.1f}" font-family="DejaVu Sans" font-size="12" font-weight="700" fill="#374151">{step}</text>')

    legend_x, legend_y = 1380, 255
    draw.text((legend_x, 205), "ENVIRONMENT", font=load_font(18, True), fill="#25313A")
    svg.append(f'<text x="{legend_x}" y="225" font-family="DejaVu Sans" font-size="18" font-weight="700" fill="#25313A">ENVIRONMENT</text>')
    for index, causal_class in enumerate(CLASS_ORDER):
        y = legend_y + index * 72
        color = CLASS_COLORS[causal_class]
        draw.ellipse((legend_x, y, legend_x + 24, y + 24), fill=color)
        draw.text((legend_x + 38, y - 2), causal_class, font=load_font(17, True), fill="#2D3338")
        svg.append(f'<circle cx="{legend_x+12}" cy="{y+12}" r="12" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+38}" y="{y+17}" font-family="DejaVu Sans" font-size="17" font-weight="700" fill="#2D3338">{causal_class}</text>')
    draw.text((legend_x, 600), "circle: Stage1 table", font=load_font(15), fill="#5B6470")
    draw.text((legend_x, 630), "diamond: Stage2 start", font=load_font(15), fill="#5B6470")
    draw.text((legend_x, 660), "star: Stage2 final", font=load_font(15), fill="#5B6470")
    x_label = f"PC1 ({explained[0] * 100:.1f}% active-table variance)"
    y_label = f"PC2 ({explained[1] * 100:.1f}% active-table variance)"
    draw.text(((left + right) / 2 - 130, 1080), x_label, font=load_font(18, True), fill="#3B4147")
    draw.text((legend_x, 760), y_label, font=load_font(16, True), fill="#3B4147")
    svg.append(f'<text x="{(left+right)/2:.1f}" y="1085" text-anchor="middle" font-family="DejaVu Sans" font-size="18" font-weight="700" fill="#3B4147">{escape(x_label)}</text>')
    svg.append(f'<text x="{legend_x}" y="780" font-family="DejaVu Sans" font-size="16" font-weight="700" fill="#3B4147">{escape(y_label)}</text>')
    svg.append("</svg>")

    png_path = Path(args.output_png)
    svg_path = Path(args.output_svg)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    svg_path.write_text("\n".join(svg), encoding="utf-8")

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[done] png={png_path} svg={svg_path} csv={csv_path}")


if __name__ == "__main__":
    main()
