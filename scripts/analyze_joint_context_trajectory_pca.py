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
    # Keep fill lightness fixed: mass is encoded independently by black outline width.
    hue = (230.0 * (1.0 - friction_rank)) / 360.0
    lightness = 0.52
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.82)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def mass_stroke_width(mass_rank: float) -> float:
    # A wide but bounded range keeps high-mass outlines obvious without hiding the fill.
    return 1.0 + 3.8 * float(mass_rank)


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
    parser.add_argument("--id-indices", default="")
    parser.add_argument("--ood-indices", default="")
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    active_ids = parse_ints(args.active_environment_ids)
    id_indices = set(parse_ints(args.id_indices))
    ood_indices = set(parse_ints(args.ood_indices))
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
        if group_id not in table or group_id not in environments:
            raise KeyError(f"Trajectory sample {sample_index} has unknown environment {group_id}.")
        if sample_index in ood_indices:
            domain = "ood"
        elif sample_index in id_indices:
            domain = "id"
        else:
            domain = "id" if group_id in active_ids else "ood"
        contexts = np.stack([context_vector(row["context_flat"]) for row in rows])
        scores = (contexts - center) @ components.T
        env = environments[group_id]
        mass = float(env["target_mass_kg"])
        friction = float(env["target_table_friction_mu"])
        color = joint_color(mass_rank[mass], friction_rank[friction])
        if domain == "id":
            target = table[group_id]
            target_initial = float(np.linalg.norm(contexts[0] - target))
            target_final = float(np.linalg.norm(contexts[-1] - target))
            target_progress = (target_initial - target_final) / max(target_initial, 1e-12)
        else:
            target_initial = float("nan")
            target_final = float("nan")
            target_progress = float("nan")
        path_l2 = float(np.linalg.norm(np.diff(contexts, axis=0), axis=1).sum())
        total_delta = float(np.linalg.norm(contexts[-1] - contexts[0]))
        trajectories.append(
            {
                "sample_index": sample_index,
                "group_id": group_id,
                "mass": mass,
                "friction": friction,
                "domain": domain,
                "color": color,
                "stroke_width": mass_stroke_width(mass_rank[mass]),
                "scores": scores,
                "steps": np.asarray([int(row["inner_step"]) for row in rows]),
            }
        )
        csv_rows.append(
            {
                "sample_index": sample_index,
                "domain": domain,
                "environment_group_id": group_id,
                "target_mass_kg": mass,
                "target_table_friction_mu": friction,
                "target_l2_initial": target_initial,
                "target_l2_final": target_final,
                "target_progress": target_progress,
                "context_total_delta_l2": total_delta,
                "context_path_l2": path_l2,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fieldnames = [
        "sample_index",
        "domain",
        "environment_group_id",
        "target_mass_kg",
        "target_table_friction_mu",
        "target_l2_initial",
        "target_l2_final",
        "target_progress",
        "context_total_delta_l2",
        "context_path_l2",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
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
        f'<text x="{left}" y="79" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#64748b">PCA fitted on {len(active_ids)} active joint-environment Z entries | friction: blue to red fill | mass: thin to thick black outline</text>',
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
        stroke_width = mass_stroke_width(mass_rank[mass])
        svg.append(
            f'<circle cx="{sx(score[0]):.2f}" cy="{sy(score[1]):.2f}" r="8.5" fill="{color}" stroke="#111827" stroke-width="{stroke_width:.2f}">'
            f'<title>training Z env={group_id} mass={mass:.6g}kg friction={friction:.6g}</title></circle>'
        )
    for item, scores in displayed:
        points = " ".join(f"{sx(score[0]):.2f},{sy(score[1]):.2f}" for score in scores)
        dash = ' stroke-dasharray="8 6"' if item["domain"] == "ood" else ""
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{item["color"]}" stroke-width="2" opacity="0.65"{dash}/>'
        )
        endpoint = scores[-1]
        svg.append(
            f'<polygon points="{triangle(sx(endpoint[0]), sy(endpoint[1]), 10)}" fill="{item["color"]}" stroke="#111827" stroke-width="{item["stroke_width"]:.2f}">'
            f'<title>inference Z domain={item["domain"]} sample={item["sample_index"]} env={item["group_id"]} mass={item["mass"]:.6g}kg friction={item["friction"]:.6g}</title></polygon>'
        )
    if displayed:
        start = displayed[0][1][0]
        x, y = sx(start[0]), sy(start[1])
        svg.append(
            f'<polygon points="{x:.2f},{y-8:.2f} {x+8:.2f},{y:.2f} {x:.2f},{y+8:.2f} {x-8:.2f},{y:.2f}" fill="#111827" stroke="#ffffff" stroke-width="1.2"><title>shared inference-time mean Z start</title></polygon>'
        )

    # Independent encodings: friction uses fill hue and mass uses black outline width.
    key_x, key_y, cell = 1200, 190, 34
    levels = np.linspace(0.0, 1.0, 5)
    svg.append(f'<text x="{key_x+2*cell}" y="{key_y-45}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" font-weight="650" fill="#334155">Visual encoding</text>')
    svg.append(f'<text x="{key_x+2*cell}" y="{key_y-18}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#64748b">friction: fill hue</text>')
    for friction_index, friction_level in enumerate(levels):
        x = key_x + friction_index * cell
        svg.append(
            f'<rect x="{x}" y="{key_y}" width="{cell}" height="{cell}" fill="{joint_color(0.5, float(friction_level))}" stroke="#ffffff" stroke-width="1"/>'
        )
    svg.extend(
        (
            f'<text x="{key_x}" y="{key_y+cell+20}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">low friction</text>',
            f'<text x="{key_x+5*cell}" y="{key_y+cell+20}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">high friction</text>',
            f'<text x="{key_x+2*cell}" y="{key_y+cell+58}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#64748b">mass: black outline width</text>',
        )
    )
    mass_key_y = key_y + cell + 90
    for mass_index, mass_level in enumerate(levels):
        x = key_x + mass_index * cell + cell / 2
        svg.append(
            f'<circle cx="{x:.2f}" cy="{mass_key_y}" r="9" fill="{joint_color(float(mass_level), 0.5)}" stroke="#111827" stroke-width="{mass_stroke_width(float(mass_level)):.2f}"/>'
        )
    svg.extend(
        (
            f'<text x="{key_x}" y="{mass_key_y+29}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">low mass</text>',
            f'<text x="{key_x+5*cell}" y="{mass_key_y+29}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">high mass</text>',
            f'<circle cx="{key_x+15}" cy="{mass_key_y+72}" r="8.5" fill="#64748b" stroke="#111827" stroke-width="2"/><text x="{key_x+32}" y="{mass_key_y+77}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">training-time Z</text>',
        )
    )
    if displayed:
        svg.append(
            f'<polygon points="{triangle(key_x+15, mass_key_y+108, 9)}" fill="#64748b" stroke="#111827" stroke-width="2"/><text x="{key_x+32}" y="{mass_key_y+113}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">inference-time Z</text>'
        )
        svg.append(
            f'<line x1="{key_x+2}" y1="{mass_key_y+143}" x2="{key_x+28}" y2="{mass_key_y+143}" stroke="#64748b" stroke-width="2"/><text x="{key_x+36}" y="{mass_key_y+147}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">ID path: solid</text>'
        )
        if any(item["domain"] == "ood" for item, _ in displayed):
            svg.append(
                f'<line x1="{key_x+2}" y1="{mass_key_y+174}" x2="{key_x+28}" y2="{mass_key_y+174}" stroke="#64748b" stroke-width="2" stroke-dasharray="8 6"/><text x="{key_x+36}" y="{mass_key_y+178}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#475569">OOD path: dashed</text>'
            )
    svg.append('</svg>')
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
