#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import json
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in str(raw).split(",") if value.strip()]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def context_vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid context vector shape={vector.shape}.")
    return vector


def padded_limits(values: np.ndarray, ratio: float = 0.1) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    return low - ratio * span, high + ratio * span


def joint_color(mass_rank: float, friction_rank: float) -> str:
    # Friction controls hue from blue (230 degrees) to red (0 degrees).
    # Mass controls HLS lightness from dark to bright.
    hue = (230.0 * (1.0 - friction_rank)) / 360.0
    lightness = 0.22 + 0.52 * mass_rank
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.82)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def triangle(cx: float, cy: float, radius: float) -> str:
    return " ".join(
        (
            f"{cx:.2f},{cy-radius:.2f}",
            f"{cx-radius*0.9:.2f},{cy+radius*0.72:.2f}",
            f"{cx+radius*0.9:.2f},{cy+radius*0.72:.2f}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--environment-table-path", required=True)
    parser.add_argument("--active-environment-ids", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    active_ids = parse_ints(args.active_environment_ids)
    table_data = json.loads(Path(args.table_path).read_text())
    table = {
        int(round(float(record["friction_mu"]))): context_vector(record["context"])
        for record in table_data["records"]
    }
    environment_data = json.loads(Path(args.environment_table_path).read_text())
    environments = {
        int(record["environment_group_id"]): record
        for record in environment_data["records"]
    }
    missing = [group_id for group_id in active_ids if group_id not in table or group_id not in environments]
    if missing:
        raise KeyError(f"Missing active joint environments: {missing}")

    active_contexts = np.stack([table[group_id] for group_id in active_ids])
    center = active_contexts.mean(axis=0)
    centered = active_contexts - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    active_scores = centered @ components.T
    explained = singular[:2] ** 2 / np.sum(singular ** 2)

    all_masses = sorted({float(record["target_mass_kg"]) for record in environments.values()})
    all_frictions = sorted({float(record["target_table_friction_mu"]) for record in environments.values()})
    mass_rank = {value: index / max(len(all_masses) - 1, 1) for index, value in enumerate(all_masses)}
    friction_rank = {
        value: index / max(len(all_frictions) - 1, 1)
        for index, value in enumerate(all_frictions)
    }

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(Path(args.trajectory_path)):
        grouped[int(row["sample_index"])].append(row)
    trajectories = []
    csv_rows = []
    for sample_index in sorted(grouped):
        rows = sorted(grouped[sample_index], key=lambda row: int(row["inner_step"]))
        group_id = int(round(float(rows[-1]["friction_mu"])))
        if group_id not in active_ids:
            continue
        contexts = np.stack([context_vector(row["context_flat"]) for row in rows])
        scores = (contexts - center) @ components.T
        env = environments[group_id]
        mass = float(env["target_mass_kg"])
        friction = float(env["target_table_friction_mu"])
        color = joint_color(mass_rank[mass], friction_rank[friction])
        target = table[group_id]
        target_initial = float(np.linalg.norm(contexts[0] - target))
        target_final = float(np.linalg.norm(contexts[-1] - target))
        path_l2 = float(np.linalg.norm(np.diff(contexts, axis=0), axis=1).sum())
        total_delta = float(np.linalg.norm(contexts[-1] - contexts[0]))
        trajectories.append(
            {
                "sample_index": sample_index,
                "group_id": group_id,
                "mass": mass,
                "friction": friction,
                "color": color,
                "scores": scores,
                "steps": np.asarray([int(row["inner_step"]) for row in rows]),
            }
        )
        csv_rows.append(
            {
                "sample_index": sample_index,
                "environment_group_id": group_id,
                "target_mass_kg": mass,
                "target_table_friction_mu": friction,
                "target_l2_initial": target_initial,
                "target_l2_final": target_final,
                "target_progress": (target_initial - target_final) / max(target_initial, 1e-12),
                "context_total_delta_l2": total_delta,
                "context_path_l2": path_l2,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    width, height = 1420, 900
    left, top, plot_width, plot_height = 95, 125, 1050, 610
    right, bottom = left + plot_width, top + plot_height
    sampled_steps = (0, 2, 5, 8, 10, 15, 20, 30, 40)
    displayed = []
    for item in trajectories:
        indices = []
        for target_step in sampled_steps:
            index = int(np.argmin(np.abs(item["steps"] - target_step)))
            if index not in indices:
                indices.append(index)
        displayed.append((item, item["scores"][indices]))
    bounds = np.concatenate([active_scores, *[scores for _, scores in displayed]], axis=0)
    xmin, xmax = padded_limits(bounds[:, 0])
    ymin, ymax = padded_limits(bounds[:, 1])
    sx = lambda value: left + (float(value) - xmin) / (xmax - xmin) * plot_width
    sy = lambda value: bottom - (float(value) - ymin) / (ymax - ymin) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="6" fill="#ffffff"/>',
        f'<text x="{left}" y="48" font-family="DejaVu Sans,sans-serif" font-size="25" font-weight="700" fill="#0f172a">{escape(args.title)}</text>',
        f'<text x="{left}" y="79" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#64748b">PCA fitted on {len(active_ids)} active joint-environment Z entries | friction: blue to red hue | mass: dark to bright</text>',
    ]
    for index in range(6):
        x = left + index * plot_width / 5
        xvalue = xmin + index * (xmax - xmin) / 5
        y = bottom - index * plot_height / 5
        yvalue = ymin + index * (ymax - ymin) / 5
        svg.extend(
            (
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#e2e8f0"/>',
                f'<text x="{x:.2f}" y="{bottom+25}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#64748b">{xvalue:.2f}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#64748b">{yvalue:.2f}</text>',
            )
        )
    svg.extend(
        (
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#94a3b8"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#94a3b8"/>',
            f'<text x="{left+plot_width/2}" y="{bottom+59}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#334155">PC1 ({explained[0]*100:.1f}% active variance)</text>',
            f'<text x="31" y="{top+plot_height/2}" transform="rotate(-90 31 {top+plot_height/2})" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#334155">PC2 ({explained[1]*100:.1f}% active variance)</text>',
        )
    )

    for group_id, score in zip(active_ids, active_scores):
        env = environments[group_id]
        mass = float(env["target_mass_kg"])
        friction = float(env["target_table_friction_mu"])
        color = joint_color(mass_rank[mass], friction_rank[friction])
        svg.append(
            f'<circle cx="{sx(score[0]):.2f}" cy="{sy(score[1]):.2f}" r="6.5" fill="{color}" stroke="#111827" stroke-width="0.9">'
            f'<title>training Z env={group_id} mass={mass:.6g}kg friction={friction:.6g}</title></circle>'
        )
    for item, scores in displayed:
        points = " ".join(f"{sx(score[0]):.2f},{sy(score[1]):.2f}" for score in scores)
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{item["color"]}" stroke-width="2" opacity="0.58"/>'
        )
        endpoint = scores[-1]
        svg.append(
            f'<polygon points="{triangle(sx(endpoint[0]), sy(endpoint[1]), 9)}" fill="{item["color"]}" stroke="#111827" stroke-width="1.8">'
            f'<title>inference Z sample={item["sample_index"]} env={item["group_id"]} mass={item["mass"]:.6g}kg friction={item["friction"]:.6g}</title></polygon>'
        )
    if displayed:
        start = displayed[0][1][0]
        x, y = sx(start[0]), sy(start[1])
        svg.append(
            f'<polygon points="{x:.2f},{y-8:.2f} {x+8:.2f},{y:.2f} {x:.2f},{y+8:.2f} {x-8:.2f},{y:.2f}" fill="#111827" stroke="#ffffff" stroke-width="1.2"><title>shared inference-time mean Z start</title></polygon>'
        )

    # Two-dimensional color key: columns are friction hue, rows are mass brightness.
    key_x, key_y, cell = 1200, 190, 34
    levels = np.linspace(0.0, 1.0, 5)
    svg.append(f'<text x="{key_x+2*cell}" y="{key_y-45}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" font-weight="650" fill="#334155">Joint color key</text>')
    svg.append(f'<text x="{key_x+2*cell}" y="{key_y-23}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#64748b">friction: blue to red</text>')
    for mass_index, mass_level in enumerate(levels):
        for friction_index, friction_level in enumerate(levels):
            x = key_x + friction_index * cell
            y = key_y + (len(levels) - 1 - mass_index) * cell
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{joint_color(float(mass_level), float(friction_level))}" stroke="#ffffff" stroke-width="1"/>'
            )
    svg.extend(
        (
            f'<text x="{key_x}" y="{key_y+5*cell+20}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">low friction</text>',
            f'<text x="{key_x+5*cell}" y="{key_y+5*cell+20}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">high friction</text>',
            f'<text x="{key_x-12}" y="{key_y+5*cell-4}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">low mass</text>',
            f'<text x="{key_x-12}" y="{key_y+10}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">high mass</text>',
            f'<circle cx="{key_x+15}" cy="{key_y+5*cell+72}" r="6.5" fill="#64748b" stroke="#111827"/><text x="{key_x+30}" y="{key_y+5*cell+77}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">training-time Z</text>',
            f'<polygon points="{triangle(key_x+15, key_y+5*cell+108, 8)}" fill="#64748b" stroke="#111827" stroke-width="1.6"/><text x="{key_x+30}" y="{key_y+5*cell+113}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">inference-time Z</text>',
            '</svg>',
        )
    )
    output_svg = Path(args.output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(svg))
    print(
        f"[done] active={len(active_ids)} samples={len(trajectories)} "
        f"variance={explained[0]:.6f},{explained[1]:.6f} svg={output_svg} csv={output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
