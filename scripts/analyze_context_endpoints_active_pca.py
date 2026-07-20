#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np


PALETTE = [
    "#2463eb", "#2368df", "#226bd7", "#216fcc", "#2173c4", "#2077b8", "#1f7ab0",
    "#1e7fa5", "#1d829d", "#1c8791", "#1c8a89", "#1a8e7e", "#1a9276", "#189964",
    "#179e57", "#1da348", "#25a346", "#2ea243", "#36a241", "#3fa23f", "#4fa23a",
    "#60a135", "#7aa12e", "#8ba029", "#a4a022", "#b59f1d", "#ce9f16", "#df9e11",
    "#f59c0b", "#f3930d", "#f08511", "#ed7714", "#e7591b", "#e2441f", "#dc2626",
]


def parse_values(raw: str, cast=float) -> list:
    return [cast(value.strip()) for value in str(raw).split(",") if value.strip()]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def friction_color(mu: float, low: float, high: float) -> str:
    position = (float(mu) - low) / max(high - low, 1e-12)
    index = round(min(1.0, max(0.0, position)) * (len(PALETTE) - 1))
    return PALETTE[index]


def padded_limits(values: np.ndarray, ratio: float = 0.08) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    return low - ratio * span, high + ratio * span


def write_svg(
    path: Path,
    *,
    title: str,
    active_mus: np.ndarray,
    active_scores: np.ndarray,
    endpoints: list[dict],
    explained: np.ndarray,
) -> None:
    width, height = 1240, 820
    plot_x, plot_y, plot_width, plot_height = 105, 120, 1055, 555
    endpoint_scores = np.stack([item["score"] for item in endpoints])
    all_scores = np.concatenate([active_scores, endpoint_scores], axis=0)
    xmin, xmax = padded_limits(all_scores[:, 0])
    ymin, ymax = padded_limits(all_scores[:, 1])
    sx = lambda value: plot_x + (float(value) - xmin) / (xmax - xmin) * plot_width
    sy = lambda value: plot_y + plot_height - (float(value) - ymin) / (ymax - ymin) * plot_height
    mu_low = min(float(active_mus.min()), min(float(item["mu"]) for item in endpoints))
    mu_high = max(float(active_mus.max()), max(float(item["mu"]) for item in endpoints))

    stops = [(0, PALETTE[0]), (24, PALETTE[8]), (44, PALETTE[15]), (68, PALETTE[23]), (83, PALETTE[28]), (100, PALETTE[-1])]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><linearGradient id="fric">' + "".join(
            f'<stop offset="{offset}%" stop-color="{color}" />' for offset, color in stops
        ) + '</linearGradient></defs>',
        '<rect width="100%" height="100%" fill="#fbfaf6" />',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" rx="7" fill="#fffdf8" stroke="#d8d2c7" />',
        f'<text x="{width / 2}" y="39" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="24" font-weight="650" fill="#17212b">{escape(title)}</text>',
        f'<text x="{width / 2}" y="72" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#6c675f">PCA fitted on {len(active_mus)} active training-time Z-table entries | {len(endpoints)} final inference-time Z values | update paths omitted</text>',
    ]

    for tick in np.linspace(xmin, xmax, 6):
        x = sx(tick)
        svg.extend([
            f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}" stroke="#ded9ce" opacity=".72" />',
            f'<text x="{x:.2f}" y="700" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">{tick:.3g}</text>',
        ])
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        svg.extend([
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" stroke="#ded9ce" opacity=".72" />',
            f'<text x="94" y="{y + 4:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">{tick:.3g}</text>',
        ])

    for mu, score in sorted(zip(active_mus, active_scores), key=lambda item: float(item[0])):
        color = friction_color(float(mu), mu_low, mu_high)
        svg.append(
            f'<circle cx="{sx(score[0]):.2f}" cy="{sy(score[1]):.2f}" r="5.5" fill="{color}" fill-opacity=".86" stroke="#fffdf8" stroke-width="1.3">'
            f'<title>training-time table mu={float(mu):.6g}</title></circle>'
        )

    for item in endpoints:
        x, y = sx(item["score"][0]), sy(item["score"][1])
        color = friction_color(float(item["mu"]), mu_low, mu_high)
        border = "#111827" if item["split"] == "OOD" else color
        border_width = 2.0 if item["split"] == "OOD" else 1.5
        points = f"{x:.2f},{y - 9:.2f} {x - 8.2:.2f},{y + 7.2:.2f} {x + 8.2:.2f},{y + 7.2:.2f}"
        svg.append(
            f'<polygon points="{points}" fill="{color}" stroke="{border}" stroke-width="{border_width}" stroke-linejoin="round">'
            f'<title>inference-time {item["split"]} sample={item["sample_index"]} mu={item["mu"]:.6g} step={item["inner_step"]}</title></polygon>'
        )

    svg.extend([
        f'<text x="{plot_x + plot_width / 2}" y="737" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="15" fill="#25313a">PC1 ({explained[0] * 100:.1f}% active variance)</text>',
        f'<text x="28" y="{plot_y + plot_height / 2}" transform="rotate(-90 28 {plot_y + plot_height / 2})" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="15" fill="#25313a">PC2 ({explained[1] * 100:.1f}% active variance)</text>',
        '<circle cx="120" cy="778" r="6" fill="#4fa23a" stroke="#fffdf8" stroke-width="1.2" />',
        '<text x="136" y="783" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">training time active Z-table</text>',
        '<polygon points="385,769 377,785 393,785" fill="#e2441f" stroke="#111827" stroke-width="2.0" stroke-linejoin="round" />',
        '<text x="403" y="783" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">inference time final Z (OOD)</text>',
        '<rect x="910" y="772" width="210" height="14" rx="2" fill="url(#fric)" stroke="#8b857b" stroke-width=".6" />',
        '<text x="900" y="783" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">low mu</text>',
        '<text x="1130" y="783" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">high mu</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--active-frictions", required=True)
    parser.add_argument("--id-indices", default="")
    parser.add_argument("--ood-indices", default="")
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    with Path(args.table_path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    active_values = np.asarray(parse_values(args.active_frictions), dtype=np.float64)
    table_mus = np.asarray([float(record["friction_mu"]) for record in records], dtype=np.float64)
    table_contexts = [np.asarray(record["context"], dtype=np.float64).reshape(-1) for record in records]
    active_contexts = []
    for mu in active_values:
        index = int(np.argmin(np.abs(table_mus - mu)))
        if abs(float(table_mus[index]) - float(mu)) > 6e-6:
            raise KeyError(f"No table context found for active friction {mu:g}.")
        active_contexts.append(table_contexts[index])
    active_contexts = np.stack(active_contexts)

    center = active_contexts.mean(axis=0)
    _, singular_values, components = np.linalg.svd(active_contexts - center, full_matrices=False)
    explained = singular_values ** 2 / max(float(np.sum(singular_values ** 2)), 1e-12)
    active_scores = (active_contexts - center) @ components[:2].T
    correlation = np.corrcoef(active_values, active_scores[:, 0])[0, 1]
    if np.isfinite(correlation) and correlation < 0:
        components[0] *= -1
        active_scores[:, 0] *= -1

    id_indices = set(parse_values(args.id_indices, int))
    ood_indices = set(parse_values(args.ood_indices, int))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(Path(args.trajectory_path)):
        grouped[int(row["sample_index"])].append(row)
    if not grouped:
        raise ValueError(f"No trajectories found in {args.trajectory_path}.")

    endpoints = []
    output_rows = []
    for sample_index, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["inner_step"]))
        initial = np.asarray(rows[0]["context_flat"], dtype=np.float64).reshape(-1)
        final = np.asarray(rows[-1]["context_flat"], dtype=np.float64).reshape(-1)
        score = (final - center) @ components[:2].T
        mu = float(rows[-1]["friction_mu"])
        split = "ID" if sample_index in id_indices else "OOD" if sample_index in ood_indices else "other"
        distances = np.linalg.norm(active_contexts - final, axis=1)
        nearest_index = int(np.argmin(distances))
        item = {
            "sample_index": sample_index,
            "split": split,
            "mu": mu,
            "inner_step": int(rows[-1]["inner_step"]),
            "score": score,
        }
        endpoints.append(item)
        output_rows.append({
            "sample_index": sample_index,
            "split": split,
            "friction_mu": mu,
            "inner_step": item["inner_step"],
            "pc1": float(score[0]),
            "pc2": float(score[1]),
            "context_delta_l2": float(np.linalg.norm(final - initial)),
            "nearest_active_mu": float(active_values[nearest_index]),
            "nearest_active_l2": float(distances[nearest_index]),
        })

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    write_svg(
        Path(args.output_svg),
        title=args.title,
        active_mus=active_values,
        active_scores=active_scores,
        endpoints=endpoints,
        explained=explained,
    )
    print(f"[done] svg={args.output_svg} csv={csv_path} active={len(active_values)} endpoints={len(endpoints)}")


if __name__ == "__main__":
    main()
