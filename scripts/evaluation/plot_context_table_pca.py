#!/usr/bin/env python3
"""Plot active entries from a learned context table in a two-dimensional PCA space."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--parameter-label", default="environment value")
    parser.add_argument("--title", default="Final active context-table PCA")
    return parser.parse_args()


def blend(low: tuple[int, int, int], high: tuple[int, int, int], weight: float) -> str:
    rgb = tuple(round(a + (b - a) * weight) for a, b in zip(low, high))
    return "#" + "".join(f"{value:02x}" for value in rgb)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.table_path.read_text())
    records = sorted(payload["records"], key=lambda row: float(row["friction_mu"]))
    values = np.asarray([float(row["friction_mu"]) for row in records], dtype=np.float64)
    contexts = np.stack([
        np.asarray(row["context"], dtype=np.float64).reshape(-1) for row in records
    ])
    centered = contexts - contexts.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ components[:2].T
    explained = singular_values ** 2 / max(float(np.sum(singular_values ** 2)), 1e-12)
    correlation = float(np.corrcoef(values, scores[:, 0])[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        scores[:, 0] *= -1
        components[0] *= -1
        correlation *= -1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            args.parameter_label, "pc1", "pc2", "context_l2_norm"
        ])
        writer.writeheader()
        for value, score, context in zip(values, scores, contexts):
            writer.writerow({
                args.parameter_label: float(value),
                "pc1": float(score[0]),
                "pc2": float(score[1]),
                "context_l2_norm": float(np.linalg.norm(context)),
            })

    width, height = 1120, 820
    left, right, top, bottom = 110, 1010, 125, 700
    x_min, x_max = float(scores[:, 0].min()), float(scores[:, 0].max())
    y_min, y_max = float(scores[:, 1].min()), float(scores[:, 1].max())
    x_pad = max((x_max - x_min) * 0.13, 0.05)
    y_pad = max((y_max - y_min) * 0.16, 0.05)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    denominator = max(float(values.max() - values.min()), 1e-12)
    colors = [blend((37, 99, 235), (220, 38, 38), float((v - values.min()) / denominator)) for v in values]
    points = [(sx(float(score[0])), sy(float(score[1]))) for score in scores]
    escaped_title = html.escape(args.title)
    escaped_parameter = html.escape(args.parameter_label)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"DejaVu Sans",sans-serif;fill:#17212b}.muted{fill:#64748b}</style>',
        f'<text x="{width/2}" y="48" text-anchor="middle" font-size="28" font-weight="700">{escaped_title}</text>',
        f'<text x="{width/2}" y="82" text-anchor="middle" font-size="15" class="muted">20 active training environments | PC1 {explained[0]*100:.1f}% | PC2 {explained[1]*100:.1f}% | corr({escaped_parameter}, PC1)={correlation:.3f}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#263445" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#263445" stroke-width="2"/>',
        f'<text x="{(left+right)/2}" y="770" text-anchor="middle" font-size="18">PC1</text>',
        f'<text x="34" y="{(top+bottom)/2}" text-anchor="middle" font-size="18" transform="rotate(-90 34 {(top+bottom)/2})">PC2</text>',
        '<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points) + '" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6 6" opacity="0.8"/>',
    ]
    for index, ((x, y), value, color) in enumerate(zip(points, values, colors)):
        dy = -13 if index % 2 == 0 else 20
        svg.extend([
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{color}" stroke="#111827" stroke-width="1.5"><title>{escaped_parameter}={value:g}</title></circle>',
            f'<text x="{x:.2f}" y="{y+dy:.2f}" text-anchor="middle" font-size="12" font-weight="600">{value:g}</text>',
        ])
    gradient_id = "parameter-gradient"
    svg.extend([
        '<defs>',
        f'<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#2563eb"/><stop offset="100%" stop-color="#dc2626"/></linearGradient>',
        '</defs>',
        f'<rect x="760" y="735" width="210" height="14" rx="7" fill="url(#{gradient_id})"/>',
        f'<text x="750" y="747" text-anchor="end" font-size="12" class="muted">low {escaped_parameter}</text>',
        f'<text x="980" y="747" font-size="12" class="muted">high {escaped_parameter}</text>',
        '</svg>',
    ])
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("\n".join(svg) + "\n")
    print(f"[done] svg={args.output_svg} csv={args.output_csv}")


if __name__ == "__main__":
    main()
