#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np


COLORS = ("#2774AE", "#D7543D")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def padded_limits(values: np.ndarray, ratio: float = 0.12) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    return low - ratio * span, high + ratio * span


def triangle_points(x: float, y: float, radius: float = 9.5) -> str:
    return (
        f"{x:.2f},{y-radius:.2f} "
        f"{x-radius*0.9:.2f},{y+radius*0.78:.2f} "
        f"{x+radius*0.9:.2f},{y+radius*0.78:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training-time Z entries and Stage2 endpoint Z values without trajectories."
    )
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--trajectory-path", action="append", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--environment-label", default="grit")
    args = parser.parse_args()

    with Path(args.table_path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    if len(records) != 2:
        raise ValueError(f"Expected exactly two training-time Z records, got {len(records)}.")

    training = []
    for record in sorted(records, key=lambda item: float(item["friction_mu"])):
        training.append(
            {
                "mu": float(record["friction_mu"]),
                "context": np.asarray(record["context"], dtype=np.float64).reshape(-1),
            }
        )

    grouped: dict[int, list[dict]] = defaultdict(list)
    for raw_path in args.trajectory_path:
        for row in read_jsonl(Path(raw_path)):
            grouped[int(row["sample_index"])].append(row)
    if not grouped:
        raise ValueError("No Stage2 trajectories were provided.")

    endpoints = []
    for sample_index, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["inner_step"]))
        endpoints.append(
            {
                "sample_index": sample_index,
                "mu": float(rows[-1]["friction_mu"]),
                "inner_step": int(rows[-1]["inner_step"]),
                "initial": np.asarray(rows[0]["context_flat"], dtype=np.float64),
                "context": np.asarray(rows[-1]["context_flat"], dtype=np.float64),
            }
        )

    fit_contexts = np.stack(
        [item["context"] for item in training] + [item["context"] for item in endpoints]
    )
    center = fit_contexts.mean(axis=0)
    _, singular_values, components = np.linalg.svd(
        fit_contexts - center,
        full_matrices=False,
    )
    components = components[:2]
    explained = singular_values[:2] ** 2 / max(float(np.sum(singular_values**2)), 1e-12)

    training_scores = np.stack([(item["context"] - center) @ components.T for item in training])
    if training_scores[1, 0] < training_scores[0, 0]:
        components[0] *= -1
        training_scores[:, 0] *= -1
    endpoint_scores = np.stack([(item["context"] - center) @ components.T for item in endpoints])

    mus = [item["mu"] for item in training]
    color_by_mu = {mu: COLORS[index] for index, mu in enumerate(mus)}
    for item, score in zip(endpoints, endpoint_scores):
        item["score"] = score
    for item, score in zip(training, training_scores):
        item["score"] = score

    output_rows = []
    for item in training:
        output_rows.append(
            {
                "kind": "training_time",
                "sample_index": "",
                "environment_value": item["mu"],
                "inner_step": "",
                "pc1": float(item["score"][0]),
                "pc2": float(item["score"][1]),
                "context_delta_l2": "",
            }
        )
    for item in endpoints:
        output_rows.append(
            {
                "kind": "inference_time_stage2",
                "sample_index": item["sample_index"],
                "environment_value": item["mu"],
                "inner_step": item["inner_step"],
                "pc1": float(item["score"][0]),
                "pc2": float(item["score"][1]),
                "context_delta_l2": float(np.linalg.norm(item["context"] - item["initial"])),
            }
        )

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    all_scores = np.concatenate([training_scores, endpoint_scores], axis=0)
    xmin, xmax = padded_limits(all_scores[:, 0])
    ymin, ymax = padded_limits(all_scores[:, 1])
    width, height = 1120, 790
    plot_x, plot_y, plot_width, plot_height = 105, 115, 930, 535
    sx = lambda value: plot_x + (float(value) - xmin) / (xmax - xmin) * plot_width
    sy = lambda value: plot_y + plot_height - (float(value) - ymin) / (ymax - ymin) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="24" font-weight="650" fill="#17212b">{escape(args.title)}</text>',
        f'<text x="{width/2}" y="69" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#6c675f">PCA fitted jointly on 2 training-time Z values and {len(endpoints)} final Stage2 Z values | update paths omitted</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" rx="8" fill="#fffdf8" stroke="#d8d2c7"/>',
    ]

    for tick in np.linspace(xmin, xmax, 6):
        x = sx(tick)
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y+plot_height}" stroke="#ded9ce" opacity=".72"/>',
                f'<text x="{x:.2f}" y="674" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">{tick:.3g}</text>',
            ]
        )
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        svg.extend(
            [
                f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x+plot_width}" y2="{y:.2f}" stroke="#ded9ce" opacity=".72"/>',
                f'<text x="94" y="{y+4:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">{tick:.3g}</text>',
            ]
        )

    for item in endpoints:
        x, y = sx(item["score"][0]), sy(item["score"][1])
        color = color_by_mu[item["mu"]]
        svg.append(
            f'<polygon points="{triangle_points(x, y)}" fill="{color}" fill-opacity=".82" stroke="#111827" stroke-width="1.5" stroke-linejoin="round">'
            f'<title>Stage2 sample={item["sample_index"]} {escape(args.environment_label)}={item["mu"]:g}</title></polygon>'
        )

    for item in training:
        x, y = sx(item["score"][0]), sy(item["score"][1])
        color = color_by_mu[item["mu"]]
        svg.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="{color}" stroke="#111827" stroke-width="1.8">'
                f'<title>training-time {escape(args.environment_label)}={item["mu"]:g}</title></circle>',
                f'<text x="{x+13:.2f}" y="{y-11:.2f}" font-family="DejaVu Sans,sans-serif" font-size="12" font-weight="650" fill="{color}">{escape(args.environment_label)} {item["mu"]:g}</text>',
            ]
        )

    svg.extend(
        [
            f'<text x="{plot_x+plot_width/2}" y="710" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#25313a">PC1 ({explained[0]*100:.1f}% combined variance)</text>',
            f'<text x="29" y="{plot_y+plot_height/2}" transform="rotate(-90 29 {plot_y+plot_height/2})" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#25313a">PC2 ({explained[1]*100:.1f}% combined variance)</text>',
            '<circle cx="125" cy="756" r="7" fill="#7a7a7a" stroke="#111827" stroke-width="1.5"/>',
            '<text x="142" y="761" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">training-time Z</text>',
            f'<polygon points="{triangle_points(310, 756, 8)}" fill="#7a7a7a" stroke="#111827" stroke-width="1.5"/>',
            '<text x="328" y="761" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">Stage2 inference-time Z</text>',
            f'<circle cx="555" cy="756" r="7" fill="{COLORS[0]}" stroke="#111827" stroke-width="1"/>',
            f'<text x="572" y="761" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">{escape(args.environment_label)} {mus[0]:g}</text>',
            f'<circle cx="715" cy="756" r="7" fill="{COLORS[1]}" stroke="#111827" stroke-width="1"/>',
            f'<text x="732" y="761" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">{escape(args.environment_label)} {mus[1]:g}</text>',
            '</svg>',
        ]
    )

    svg_path = Path(args.output_svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(
        f"[done] svg={svg_path} csv={csv_path} "
        f"training={len(training)} stage2_endpoints={len(endpoints)}"
    )


if __name__ == "__main__":
    main()
