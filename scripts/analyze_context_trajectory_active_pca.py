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
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def padded_limits(values: np.ndarray, ratio: float = 0.08) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    return low - ratio * span, high + ratio * span


def friction_color(mu: float, low: float, high: float) -> str:
    position = (float(mu) - low) / max(high - low, 1e-12)
    index = round(min(1.0, max(0.0, position)) * (len(PALETTE) - 1))
    return PALETTE[index]


def write_svg(
    path: Path,
    *,
    title: str,
    active_mus: np.ndarray,
    active_scores: np.ndarray,
    trajectories: list[dict],
    explained: np.ndarray,
    parameter_label: str,
) -> None:
    width, height = 1680, 1040
    detail_panels = [(840, 140), (1260, 140), (840, 520), (1260, 520)]
    show_detail_panels = len(trajectories) <= len(detail_panels)
    overview = (70, 140, 720, 750) if show_detail_panels else (70, 140, 1540, 750)
    detail_width, detail_height = 380, 330
    shown_steps = (0, 2, 5, 8, 10, 15, 20, 30, 40)
    phase_colors = ("#dc2626", "#f59e0b", "#168a80", "#2563eb")
    phase_labels = ("steps 1-10 | LR 3.0", "steps 11-20 | LR 1.5", "steps 21-30 | LR 0.5", "steps 31-40 | LR 0.15")

    def sampled_indices(item: dict) -> list[int]:
        steps = np.asarray(item["steps"], dtype=np.int64)
        result = []
        for target in shown_steps:
            index = int(np.argmin(np.abs(steps - target)))
            if index not in result:
                result.append(index)
        return result

    def stable_limits(values: np.ndarray, ratio: float = 0.15, minimum_span: float = 0.035) -> tuple[float, float]:
        low, high = float(values.min()), float(values.max())
        center = 0.5 * (low + high)
        span = max(high - low, minimum_span)
        half = 0.5 * span * (1.0 + 2.0 * ratio)
        return center - half, center + half

    def mapper(panel: tuple[float, float, float, float], limits: tuple[float, float, float, float]):
        x, y, panel_width, panel_height = panel
        xmin, xmax, ymin, ymax = limits
        sx = lambda value: x + (float(value) - xmin) / (xmax - xmin) * panel_width
        sy = lambda value: y + panel_height - (float(value) - ymin) / (ymax - ymin) * panel_height
        return sx, sy

    def phase_for_step(step: int) -> int:
        if step <= 10:
            return 0
        if step <= 20:
            return 1
        if step <= 30:
            return 2
        return 3

    sampled = []
    for item in trajectories:
        indices = sampled_indices(item)
        sampled.append((item, indices, item["scores"][indices]))

    all_scores = [active_scores]
    all_scores.extend(scores for _, _, scores in sampled)
    bounds = np.concatenate(all_scores, axis=0)
    xmin, xmax = padded_limits(bounds[:, 0], ratio=0.1)
    ymin, ymax = padded_limits(bounds[:, 1], ratio=0.1)
    ox, oy, ow, oh = overview
    sx, sy = mapper((ox + 58, oy + 58, ow - 88, oh - 126), (xmin, xmax, ymin, ymax))
    mu_low = min(float(active_mus.min()), min(float(item["mu"]) for item in trajectories))
    mu_high = max(float(active_mus.max()), max(float(item["mu"]) for item in trajectories))

    stops = [(0, PALETTE[0]), (24, PALETTE[8]), (44, PALETTE[15]), (68, PALETTE[23]), (83, PALETTE[28]), (100, PALETTE[-1])]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<linearGradient id="fric">' + "".join(f'<stop offset="{offset}%" stop-color="{color}"/>' for offset, color in stops) + '</linearGradient>',
        *[
            f'<marker id="phase-arrow-{index}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L8,4 L0,8 z" fill="{color}"/></marker>'
            for index, color in enumerate(phase_colors)
        ],
        '</defs>',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        f'<text x="{width/2}" y="42" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="25" font-weight="650" fill="#17212b">{escape(title)}</text>',
        f'<text x="{width/2}" y="75" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="15" fill="#6c675f">PCA basis is fitted on active C-table entries | full 40-step metrics retained | displayed steps: 0, 2, 5, 8, 10, 15, 20, 30, 40</text>',
        f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="10" fill="#fffdf8" stroke="#d9d3c7" stroke-width="1.2"/>',
        f'<text x="{ox+24}" y="{oy+31}" font-family="DejaVu Sans,sans-serif" font-size="18" font-weight="600" fill="#25313a">Global view: C table, shared start and endpoints</text>',
    ]
    plot_left, plot_right = ox + 58, ox + ow - 30
    plot_top, plot_bottom = oy + 58, oy + oh - 68
    for tick in np.linspace(xmin, xmax, 5):
        x = sx(tick)
        svg.extend([
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#ded9ce" opacity=".7"/>',
            f'<text x="{x:.2f}" y="{plot_bottom+23}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">{tick:.3g}</text>',
        ])
    for tick in np.linspace(ymin, ymax, 5):
        y = sy(tick)
        svg.extend([
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#ded9ce" opacity=".7"/>',
            f'<text x="{plot_left-10}" y="{y+4:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">{tick:.3g}</text>',
        ])
    svg.extend([
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#27323a" stroke-width="1.25"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#27323a" stroke-width="1.25"/>',
    ])

    order = np.argsort(active_mus)
    table_points = " ".join(f"{sx(active_scores[index, 0]):.2f},{sy(active_scores[index, 1]):.2f}" for index in order)
    svg.append(f'<polyline points="{table_points}" fill="none" stroke="#8b857b" stroke-width="1.05" stroke-dasharray="4 5" opacity=".42"/>')
    for index in order:
        color = friction_color(float(active_mus[index]), mu_low, mu_high)
        svg.append(
            f'<circle cx="{sx(active_scores[index,0]):.2f}" cy="{sy(active_scores[index,1]):.2f}" r="4.5" fill="{color}" stroke="#fffdf8" stroke-width="1">'
            f'<title>table {escape(parameter_label)}={active_mus[index]:.6g}</title></circle>'
        )

    for item, indices, scores in sampled:
        color = friction_color(float(item["mu"]), mu_low, mu_high)
        dash = "" if item["split"] == "ID" else ' stroke-dasharray="7 5"'
        points = " ".join(f"{sx(row[0]):.2f},{sy(row[1]):.2f}" for row in scores)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.1" opacity=".62"{dash}/>')
        svg.append(
            f'<circle cx="{sx(scores[-1,0]):.2f}" cy="{sy(scores[-1,1]):.2f}" r="6.2" fill="{color}" stroke="#17212b" stroke-width="1.2">'
            f'<title>{item["split"]} sample={item["sample_index"]} mu={item["mu"]:.6g}</title></circle>'
        )
        if item["target_score"] is not None:
            tx, ty = sx(item["target_score"][0]), sy(item["target_score"][1])
            svg.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="9" fill="none" stroke="{color}" stroke-width="2.2" opacity=".85"/>')

    common_start = sampled[0][2][0]
    start_x, start_y = sx(common_start[0]), sy(common_start[1])
    svg.extend([
        f'<polygon points="{start_x:.2f},{start_y-8:.2f} {start_x+8:.2f},{start_y:.2f} {start_x:.2f},{start_y+8:.2f} {start_x-8:.2f},{start_y:.2f}" fill="#111827" stroke="#fffdf8" stroke-width="1.3"/>',
        f'<text x="{ox+ow/2}" y="{oy+oh-20}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#25313a">PC1 ({explained[0]*100:.1f}% active variance)</text>',
        f'<text x="{ox+17}" y="{oy+oh/2}" transform="rotate(-90 {ox+17} {oy+oh/2})" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#25313a">PC2 ({explained[1]*100:.1f}% active variance)</text>',
    ])

    panel_items = sampled if show_detail_panels else []
    for panel_index, (item, indices, selected_scores) in enumerate(panel_items):
        panel_x, panel_y = detail_panels[panel_index]
        border_dash = "" if item["split"] == "ID" else ' stroke-dasharray="8 5"'
        friction = friction_color(float(item["mu"]), mu_low, mu_high)
        svg.extend([
            f'<rect x="{panel_x}" y="{panel_y}" width="{detail_width}" height="{detail_height}" rx="10" fill="#fffdf8" stroke="{friction}" stroke-width="1.8"{border_dash}/>',
            f'<text x="{panel_x+18}" y="{panel_y+27}" font-family="DejaVu Sans,sans-serif" font-size="17" font-weight="650" fill="#17212b">{item["split"]} | sample {item["sample_index"]} | {escape(parameter_label)}={item["mu"]:.6g}</text>',
            f'<text x="{panel_x+18}" y="{panel_y+48}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#6c675f">local PCA zoom | sampled updates only</text>',
        ])
        local_xmin, local_xmax = stable_limits(selected_scores[:, 0])
        local_ymin, local_ymax = stable_limits(selected_scores[:, 1])
        local_plot = (panel_x + 48, panel_y + 65, detail_width - 70, 185)
        lsx, lsy = mapper(local_plot, (local_xmin, local_xmax, local_ymin, local_ymax))
        local_left, local_top, local_width, local_height = local_plot
        local_right, local_bottom = local_left + local_width, local_top + local_height
        for tick in np.linspace(local_xmin, local_xmax, 3):
            x = lsx(tick)
            svg.extend([
                f'<line x1="{x:.2f}" y1="{local_top}" x2="{x:.2f}" y2="{local_bottom}" stroke="#e4dfd5" stroke-width=".8"/>',
                f'<text x="{x:.2f}" y="{local_bottom+15}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="9" fill="#817b72">{tick:.3g}</text>',
            ])
        for tick in np.linspace(local_ymin, local_ymax, 3):
            y = lsy(tick)
            svg.extend([
                f'<line x1="{local_left}" y1="{y:.2f}" x2="{local_right}" y2="{y:.2f}" stroke="#e4dfd5" stroke-width=".8"/>',
                f'<text x="{local_left-6}" y="{y+3:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="9" fill="#817b72">{tick:.3g}</text>',
            ])
        selected_steps = [int(item["steps"][index]) for index in indices]
        for segment in range(len(selected_scores) - 1):
            step = selected_steps[segment + 1]
            phase = phase_for_step(step)
            marker = f' marker-end="url(#phase-arrow-{phase})"' if step in (10, 20, 30, 40) else ""
            svg.append(
                f'<line x1="{lsx(selected_scores[segment,0]):.2f}" y1="{lsy(selected_scores[segment,1]):.2f}" '
                f'x2="{lsx(selected_scores[segment+1,0]):.2f}" y2="{lsy(selected_scores[segment+1,1]):.2f}" '
                f'stroke="{phase_colors[phase]}" stroke-width="2.5" opacity=".88"{marker}/>'
            )
        for point_index, (score, step) in enumerate(zip(selected_scores, selected_steps)):
            x, y = lsx(score[0]), lsy(score[1])
            if step == 0:
                svg.append(f'<polygon points="{x:.2f},{y-7:.2f} {x+7:.2f},{y:.2f} {x:.2f},{y+7:.2f} {x-7:.2f},{y:.2f}" fill="#111827" stroke="#fffdf8" stroke-width="1"/>')
            elif step == 40:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="#fffdf8" stroke="{phase_colors[-1]}" stroke-width="3"/>')
            else:
                phase = phase_for_step(step)
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{phase_colors[phase]}" stroke="#fffdf8" stroke-width="1"/>')
            dx = 7 if point_index % 2 == 0 else -7
            anchor = "start" if dx > 0 else "end"
            dy = -8 if point_index % 3 else 13
            label = "START 0" if step == 0 else f"{step}"
            svg.append(f'<text x="{x+dx:.2f}" y="{y+dy:.2f}" text-anchor="{anchor}" font-family="DejaVu Sans,sans-serif" font-size="10" font-weight="600" fill="#27323a">{label}</text>')

        metrics = item["metrics"]
        efficiency = float(metrics["context_total_delta_l2"]) / max(float(metrics["context_path_l2"]), 1e-12)
        svg.append(
            f'<text x="{panel_x+18}" y="{panel_y+288}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#3d464d">'
            f'net Δ={float(metrics["context_total_delta_l2"]):.3f} | full path={float(metrics["context_path_l2"]):.3f} | net/path={efficiency:.1%}</text>'
        )
        if metrics["true_table_progress"] != "":
            target_line = (
                f'target L2 {float(metrics["true_table_l2_initial"]):.3f} → {float(metrics["true_table_l2_final"]):.3f} '
                f'| closer by {float(metrics["true_table_progress"]):.1%}'
            )
        else:
            target_line = (
                f'held-out: no exact table target | nearest final {escape(parameter_label)}={float(metrics["nearest_active_mu_final"]):.4g} '
                f'| L2={float(metrics["nearest_active_l2_final"]):.3f}'
            )
        svg.append(f'<text x="{panel_x+18}" y="{panel_y+310}" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#3d464d">{target_line}</text>')

    svg.extend([
        '<text x="70" y="936" font-family="DejaVu Sans,sans-serif" font-size="13" font-weight="650" fill="#25313a">Update schedule and direction:</text>',
        *[
            f'<line x1="{285 + index*260}" y1="932" x2="{335 + index*260}" y2="932" stroke="{color}" stroke-width="4" marker-end="url(#phase-arrow-{index})"/>'
            f'<text x="{345 + index*260}" y="937" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">{phase_labels[index]}</text>'
            for index, color in enumerate(phase_colors)
        ],
        '<polygon points="76,982 84,990 76,998 68,990" fill="#111827"/><text x="94" y="995" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">shared START (step 0)</text>',
        '<circle cx="310" cy="990" r="6.5" fill="#fffdf8" stroke="#2563eb" stroke-width="3"/><text x="324" y="995" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">END (step 40)</text>',
        '<line x1="500" y1="990" x2="542" y2="990" stroke="#27323a" stroke-width="2"/><text x="552" y="995" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">ID</text>',
        '<line x1="620" y1="990" x2="662" y2="990" stroke="#27323a" stroke-width="2" stroke-dasharray="7 5"/><text x="672" y="995" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#4b5359">OOD / held-out</text>',
        '<rect x="1260" y="982" width="280" height="15" rx="2" fill="url(#fric)" stroke="#8b857b" stroke-width=".6"/>',
        f'<text x="1248" y="995" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">low {escape(parameter_label)}</text>',
        f'<text x="1550" y="995" font-family="DejaVu Sans,sans-serif" font-size="11" fill="#746f66">high {escape(parameter_label)}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--active-frictions", required=True)
    parser.add_argument("--id-indices", required=True)
    parser.add_argument("--ood-indices", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--parameter-label", default="mu")
    args = parser.parse_args()

    table_records = json.loads(Path(args.table_path).read_text())["records"]
    table_mus = np.asarray([float(row["friction_mu"]) for row in table_records])
    table_contexts = np.stack([np.asarray(row["context"], dtype=np.float64).reshape(-1) for row in table_records])
    active_values = np.asarray(parse_values(args.active_frictions), dtype=np.float64)
    active = np.min(np.abs(table_mus[:, None] - active_values[None, :]), axis=1) <= 6e-6
    active_contexts = table_contexts[active]
    active_mus = table_mus[active]
    center = active_contexts.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(active_contexts - center, full_matrices=False)
    explained = singular ** 2 / np.sum(singular ** 2)
    active_scores = (active_contexts - center) @ components[:2].T

    id_indices = set(parse_values(args.id_indices, int))
    ood_indices = set(parse_values(args.ood_indices, int))
    grouped = defaultdict(list)
    for row in read_jsonl(Path(args.trajectory_path)):
        grouped[int(row["sample_index"])].append(row)

    output_rows = []
    plot_trajectories = []
    for sample_index in sorted(grouped):
        rows = sorted(grouped[sample_index], key=lambda row: int(row["inner_step"]))
        contexts = np.stack([np.asarray(row["context_flat"], dtype=np.float64) for row in rows])
        scores = (contexts - center) @ components[:2].T
        mu = float(rows[-1]["friction_mu"])
        split = "ID" if sample_index in id_indices else "OOD" if sample_index in ood_indices else "other"
        step_deltas = np.linalg.norm(np.diff(contexts, axis=0), axis=1)
        start_distances = np.linalg.norm(active_contexts - contexts[0], axis=1)
        final_distances = np.linalg.norm(active_contexts - contexts[-1], axis=1)
        start_nearest = int(np.argmin(start_distances))
        final_nearest = int(np.argmin(final_distances))
        true_matches = np.where(np.abs(active_mus - mu) <= 6e-6)[0]
        target_initial = target_final = progress = ""
        if len(true_matches):
            target = active_contexts[int(true_matches[0])]
            target_initial_value = float(np.linalg.norm(contexts[0] - target))
            target_final_value = float(np.linalg.norm(contexts[-1] - target))
            target_initial = target_initial_value
            target_final = target_final_value
            progress = (target_initial_value - target_final_value) / max(target_initial_value, 1e-12)
        output_rows.append({
            "sample_index": sample_index,
            "split": split,
            "friction_mu": mu,
            "inner_steps": int(rows[-1]["inner_step"]),
            "context_total_delta_l2": float(np.linalg.norm(contexts[-1] - contexts[0])),
            "context_path_l2": float(step_deltas.sum()),
            "nonzero_steps": int(np.sum(step_deltas > 1e-8)),
            "pc1_initial": float(scores[0, 0]),
            "pc2_initial": float(scores[0, 1]),
            "pc1_final": float(scores[-1, 0]),
            "pc2_final": float(scores[-1, 1]),
            "nearest_active_mu_initial": float(active_mus[start_nearest]),
            "nearest_active_mu_final": float(active_mus[final_nearest]),
            "nearest_active_l2_initial": float(start_distances[start_nearest]),
            "nearest_active_l2_final": float(final_distances[final_nearest]),
            "true_table_l2_initial": target_initial,
            "true_table_l2_final": target_final,
            "true_table_progress": progress,
        })
        plot_trajectories.append({
            "sample_index": sample_index,
            "split": split,
            "mu": mu,
            "scores": scores,
            "steps": np.asarray([int(row["inner_step"]) for row in rows]),
            "initial_source": str(rows[0].get("initial_context_source", "shared initialization")),
            "target_score": active_scores[int(true_matches[0])].copy() if len(true_matches) else None,
            "metrics": output_rows[-1],
        })

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    write_svg(
        Path(args.output_svg),
        title=args.title,
        active_mus=active_mus,
        active_scores=active_scores,
        trajectories=plot_trajectories,
        explained=explained,
        parameter_label=str(args.parameter_label),
    )
    start_sources = sorted({item["initial_source"] for item in plot_trajectories})
    print(
        f"[done] csv={csv_path} svg={args.output_svg} samples={len(output_rows)} "
        f"active={active.sum()} start_sources={start_sources}"
    )


if __name__ == "__main__":
    main()
