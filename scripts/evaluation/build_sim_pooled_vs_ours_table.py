#!/usr/bin/env python3
"""Build the cross-task Ours versus standard pooled-WM benchmark table."""

from __future__ import annotations

import csv
import html
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/sim_standard_pooled_vs_ours_v1"


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def weighted_action(paths: list[str]) -> float | None:
    weighted = 0.0
    count = 0
    for path in paths:
        headline = load_json(path).get("headline_oracle_reachable_complete_decisions", {})
        item_count = int(headline.get("count", 0))
        rate = headline.get("task_success_rate")
        if item_count and rate is not None:
            weighted += item_count * float(rate)
            count += item_count
    return weighted / count if count else None


def generic_record(
    *,
    task: str,
    method: str,
    queries: int,
    roots: list[str],
    action_paths: list[str],
    task_metric_key: str,
    task_metric_label: str,
    task_metric_higher_is_better: bool,
) -> dict[str, Any]:
    global_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for root in roots:
        base = ROOT / root
        global_rows.extend(load_json(base / "video_metrics/global/global_summary.json")["summary"])
        object_path = base / "video_metrics/object_centric/object_summary.json"
        if object_path.exists():
            object_rows.extend(load_json(object_path).get("aggregation", {}).get("summary", []))
        task_rows.extend(load_json(base / "video_metrics/task_specific/task_summary.json")["aggregation"]["summary"])
    if task == "Light switch":
        object_mean_error = mean_key(task_rows, "light_yellow_score_mae")
        object_final_error = mean_key(task_rows, "light_yellow_score_final_abs_error")
        object_metric_label = "Lamp yellow-score error"
        object_metric_unit = "fraction"
    else:
        object_mean_error = mean_key(object_rows, "centroid_ade_px")
        object_final_error = mean_key(object_rows, "centroid_fde_px")
        object_metric_label = "Object-centroid error"
        object_metric_unit = "px"
    record = {
        "task": task,
        "method": method,
        "queries": queries,
        "status": "complete",
        "psnr": mean_key(global_rows, "psnr"),
        "ssim": mean_key(global_rows, "ssim"),
        "lpips": mean_key(global_rows, "lpips"),
        "object_mean_error": object_mean_error,
        "object_final_error": object_final_error,
        "object_metric_label": object_metric_label,
        "object_metric_unit": object_metric_unit,
        "action_success": weighted_action(action_paths),
        "task_metric": mean_key(task_rows, task_metric_key),
        "task_metric_label": task_metric_label,
        "task_metric_higher_is_better": task_metric_higher_is_better,
    }
    required = [
        "psnr", "ssim", "lpips", "object_mean_error", "object_final_error",
        "action_success", "task_metric",
    ]
    missing = [key for key in required if record[key] is None]
    if missing:
        record["status"] = f"partial: missing {', '.join(missing)}"
    return record


def generic_record_if_available(
    *,
    task: str,
    method: str,
    queries: int,
    roots: list[str],
    action_paths: list[str],
    task_metric_key: str,
    task_metric_label: str,
    task_metric_higher_is_better: bool,
) -> dict[str, Any]:
    required_paths = []
    for root in roots:
        base = ROOT / root
        required_paths.extend([
            base / "video_metrics/global/global_summary.json",
            base / "video_metrics/task_specific/task_summary.json",
        ])
    required_paths.extend(ROOT / path for path in action_paths)
    if not all(path.exists() for path in required_paths):
        return {
            "task": task,
            "method": method,
            "queries": queries,
            "status": "pending metrics",
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "object_mean_error": None,
            "object_final_error": None,
            "object_metric_label": "Lamp yellow-score error" if task == "Light switch" else "Object-centroid error",
            "object_metric_unit": "fraction" if task == "Light switch" else "px",
            "action_success": None,
            "task_metric": None,
            "task_metric_label": task_metric_label,
            "task_metric_higher_is_better": task_metric_higher_is_better,
        }
    return generic_record(
        task=task,
        method=method,
        queries=queries,
        roots=roots,
        action_paths=action_paths,
        task_metric_key=task_metric_key,
        task_metric_label=task_metric_label,
        task_metric_higher_is_better=task_metric_higher_is_better,
    )


def event80_records() -> list[dict[str, Any]]:
    scoreboard = load_json(
        "results/pushbox_friction_event80/"
        "event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1/"
        "metrics/complete_v1/scoreboard.json"
    )
    by_method = {row["method"]: row for row in scoreboard}
    output = []
    for key, label in (("standard_pooled_wm", "Standard pooled WM"), ("ours", "Ours")):
        source = by_method[key]
        output.append({
            "task": "Push-box friction (Event80)",
            "method": label,
            "queries": int(source["query_count"]),
            "status": "complete",
            "psnr": float(source["psnr_multiview"]),
            "ssim": float(source["ssim_multiview"]),
            "lpips": float(source["lpips_multiview"]),
            "object_mean_error": float(source["centroid_ade_px"]),
            "object_final_error": float(source["centroid_fde_px"]),
            "object_metric_label": "Object-centroid error",
            "object_metric_unit": "px",
            "action_success": float(source["action_success_all"]),
            "task_metric": float(source["centroid_fde_px"]),
            "task_metric_label": "Final centroid error (px) ↓",
            "task_metric_higher_is_better": False,
        })
    return output


def build_records() -> list[dict[str, Any]]:
    records = event80_records()
    definitions = [
        {
            "task": "Gravity",
            "queries": 90,
            "ours": "results/gravity/gravity80_uniform5id5ood_strict_v1/methods/ours/step_3837/seed_20260712",
            "pooled": "results/gravity/gravity80_uniform5id5ood_strict_v1/methods/standard_pooled_wm/step_5218/seed_20260712",
            "metric": "final_displacement_abs_error_px",
            "label": "Final displacement error (px) ↓",
            "higher": False,
        },
        {
            "task": "Mass balance (fixed pose)",
            "queries": 140,
            "ours": "results/mass_balance/fixed_pose_dense15_boundary_104381/methods/ours",
            "pooled": "results/mass_balance/fixed_pose_dense15_boundary_104381/methods/standard_pooled_wm/step_8000/seed_20260722",
            "metric": "bar_tilt_mae_deg",
            "label": "Bar tilt MAE (deg) ↓",
            "higher": False,
        },
        {
            "task": "Light switch",
            "queries": 60,
            "ours": "results/lightswitch/physicalpress33_all4env_support8_query15_v1/methods/ours/step_3100/seed_20260828",
            "pooled": "results/lightswitch/physicalpress33_all4env_support8_query15_v1/methods/standard_pooled_wm/step_3289/seed_20260828",
            "metric": "light_state_accuracy",
            "label": "Lamp-state frame accuracy ↑",
            "higher": True,
        },
    ]
    for definition in definitions:
        for method, root in (("Standard pooled WM", definition["pooled"]), ("Ours", definition["ours"])):
            records.append(generic_record_if_available(
                task=definition["task"],
                method=method,
                queries=definition["queries"],
                roots=[root],
                action_paths=[f"{root}/action_evaluation/summary.json"],
                task_metric_key=definition["metric"],
                task_metric_label=definition["label"],
                task_metric_higher_is_better=definition["higher"],
            ))

    # Leakage-free replacement for the superseded linear-theory-distance run.
    # Ours is the action-batch-8 Stage1 checkpoint from job 107912.
    collision_roots = {
        "Standard pooled WM": (
            "results/mass_collision/noleak_grid_id5_ood5_k1_balanced_visible_or_min_action_v4/"
            "methods/standard_pooled_wm/step_8000/seed_20260827"
        ),
        "Ours": (
            "results/mass_collision/noleak_grid_id5_ood5_k1_balanced_visible_or_min_action_v4/"
            "methods/ours_action8/step_4300/seed_20260827"
        ),
    }
    collision_records = [
        generic_record_if_available(
            task="Mass collision",
            method=method,
            queries=80,
            roots=[root],
            action_paths=[f"{root}/action_evaluation/summary.json"],
            task_metric_key="final_displacement_abs_error_px",
            task_metric_label="Final displacement error (px) ↓",
            task_metric_higher_is_better=False,
        )
        for method, root in collision_roots.items()
    ]
    insertion = next(index for index, row in enumerate(records) if row["task"] == "Mass balance (fixed pose)")
    records[insertion:insertion] = collision_records

    mass_friction_ours = "results/mass_friction/joint100_same_environment_actions_91479/methods/ours"
    mass_friction_pool = "results/mass_friction/joint100_same_environment_actions_91479/methods/standard_pooled_wm/step_3277/seed_20260719"
    records.extend([
        generic_record(
            task="Mass × friction",
            method="Standard pooled WM",
            queries=48,
            roots=[f"{mass_friction_pool}/id", f"{mass_friction_pool}/ood"],
            action_paths=[f"{mass_friction_pool}/id/action_evaluation/summary.json", f"{mass_friction_pool}/ood/action_evaluation/summary.json"],
            task_metric_key="final_displacement_abs_error_px",
            task_metric_label="Final displacement error (px) ↓",
            task_metric_higher_is_better=False,
        ),
        generic_record(
            task="Mass × friction",
            method="Ours",
            queries=48,
            roots=[mass_friction_ours],
            action_paths=[f"{mass_friction_ours}/action_evaluation/summary.json"],
            task_metric_key="final_displacement_abs_error_px",
            task_metric_label="Final displacement error (px) ↓",
            task_metric_higher_is_better=False,
        ),
    ])

    return records


def display(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "psnr":
        return f"{value:.2f}"
    if kind in {"ssim", "lpips"}:
        return f"{value:.3f}"
    if kind == "success":
        return f"{100.0 * value:.1f}%"
    return f"{value:.2f}"


def write_outputs(records: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT / "sim_standard_pooled_vs_ours.json"
    csv_path = OUTPUT / "sim_standard_pooled_vs_ours.csv"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    protocol = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "Ours versus standard pooled world model",
        "global_metrics": "PSNR/SSIM/LPIPS are averaged over the available ID/OOD environment summaries.",
        "object_metrics": "Mean/final object-state error: centroid ADE/FDE in pixels for moving objects; yellow-lamp score MAE/final absolute error for LightSwitch.",
        "action_metric": "Task success on complete candidate sets whose targets are reachable in GT.",
        "query_protocol": "Each pooled baseline reuses the exact Ours transfer plan; pooled WM ignores support by design.",
        "superseded_for_leakage": [
            "Mass collision: job 90515 / evaluation 90735",
            "Light switch: job 95498 / evaluation 95707",
        ],
        "replacement_provenance": {
            "mass_collision_ours": {
                "train_config": "configs/train/train_mass20of30_collision_noleak_mainview_full61_curriculum_c32_old_random_8action_stage1_4300.yaml",
                "checkpoint": "outputs/mass20of30_collision_noleak_mainview_c32_oldrandom_8action_s1_107912/step-4300.safetensors",
                "actions_per_environment_per_update": 8,
            },
            "light_switch_ours": {
                "train_config": "configs/train/train_lightswitch_physicalpress33jitter11to22_maincam_group20_c32_3wave1400_actions8_stage1_4500.yaml",
                "checkpoint": "outputs/lightswitch_physicalpress33jitter11to22_maincam_group20_c32_3wave1400_actions8_high2gpu_107549/step-3100.safetensors",
                "support_size": 8,
            },
        },
        "pending": [
            f"{row['task']} / {row['method']}: {row['status']}"
            for row in records
            if row["status"] != "complete"
        ],
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    columns = [
        "Task", "Method", "N", "PSNR ↑", "SSIM ↑", "LPIPS ↓",
        "Object mean error ↓", "Object final error ↓", "Action success ↑", "Task-specific metric",
    ]
    body = []
    for row in records:
        task_metric = "—" if row["task_metric"] is None else f"{row['task_metric']:.2f}"
        if row["task"] == "Light switch" and row["task_metric"] is not None:
            task_metric = f"{row['task_metric']:.3f}"
        if row["object_mean_error"] is None:
            object_mean = "—"
            object_final = "—"
        elif row["object_metric_unit"] == "px":
            object_mean = f"{row['object_mean_error']:.2f} px"
            object_final = f"{row['object_final_error']:.2f} px"
        else:
            object_mean = f"{row['object_mean_error']:.3f}"
            object_final = f"{row['object_final_error']:.3f}"
        body.append([
            row["task"], row["method"], str(row["queries"]),
            display(row["psnr"], "psnr"), display(row["ssim"], "ssim"), display(row["lpips"], "lpips"),
            object_mean, object_final,
            display(row["action_success"], "success"), task_metric,
        ])

    task_order = list(dict.fromkeys(row["task"] for row in records))
    group_colors = ["#f6f8fa", "#eef4f7"]
    metric_columns = {
        3: ("psnr", True), 4: ("ssim", True), 5: ("lpips", False),
        6: ("object_mean_error", False), 7: ("object_final_error", False), 8: ("action_success", True),
    }
    best_cells: set[tuple[int, int]] = set()
    for task in task_order:
        indices = [index for index, row in enumerate(records) if row["task"] == task]
        comparable = indices
        if len(comparable) != 2:
            continue
        for column, (key, higher) in metric_columns.items():
            values = [(index, records[index][key]) for index in comparable if records[index][key] is not None]
            if len(values) != 2:
                continue
            best = (max if higher else min)(value for _, value in values)
            for index, value in values:
                if abs(value - best) <= 1e-12:
                    best_cells.add((index, column))
        values = [(index, records[index]["task_metric"]) for index in comparable if records[index]["task_metric"] is not None]
        if len(values) == 2:
            higher = records[comparable[0]]["task_metric_higher_is_better"]
            best = (max if higher else min)(value for _, value in values)
            for index, value in values:
                if abs(value - best) <= 1e-12:
                    best_cells.add((index, 9))

    column_widths = [520, 330, 100, 180, 180, 180, 220, 220, 220, 330]
    margin = 60
    title_height = 165
    header_height = 92
    row_height = 76
    footer_height = 145
    width = sum(column_widths) + 2 * margin
    height = title_height + header_height + row_height * len(body) + footer_height

    regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

    def load_font(path: str, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    fonts = {
        "title": load_font(bold_path, 42),
        "subtitle": load_font(regular_path, 23),
        "header": load_font(bold_path, 20),
        "body": load_font(regular_path, 20),
        "body_bold": load_font(bold_path, 20),
        "foot": load_font(regular_path, 17),
    }

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 36), "Simulation Benchmarks: Ours vs. Standard Pooled World Model", font=fonts["title"], fill="#17324d")
    draw.text((margin, 98), "Shared frozen support/query protocols; the pooled baseline receives no test-time adaptation.", font=fonts["subtitle"], fill="#52606d")

    x_edges = [margin]
    for item in column_widths:
        x_edges.append(x_edges[-1] + item)
    table_top = title_height

    def centered_text(text: str, box: tuple[int, int, int, int], font: ImageFont.ImageFont, fill: str) -> None:
        left, top, right, bottom = box
        lines = text.split("\n")
        line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 4, 18)
        y = (top + bottom - line_height * len(lines)) / 2
        for line in lines:
            bounds = draw.textbbox((0, 0), line, font=font)
            x = (left + right - (bounds[2] - bounds[0])) / 2
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height

    for column, label in enumerate(columns):
        box = (x_edges[column], table_top, x_edges[column + 1], table_top + header_height)
        draw.rectangle(box, fill="#17324d", outline="#17324d", width=2)
        centered_text(label, box, fonts["header"], "white")

    for row_index, (record, values) in enumerate(zip(records, body)):
        top = table_top + header_height + row_index * row_height
        bottom = top + row_height
        group_index = task_order.index(record["task"])
        fill = "#fff4cf" if record["status"] != "complete" else group_colors[group_index % 2]
        for column, value in enumerate(values):
            box = (x_edges[column], top, x_edges[column + 1], bottom)
            draw.rectangle(box, fill=fill, outline="#c8d2dc", width=2)
            bold = column == 0 or (row_index, column) in best_cells
            font = fonts["body_bold"] if bold else fonts["body"]
            color = "#0b5a75" if column == 1 and record["method"] == "Ours" else "#27313a"
            if column in (0, 1):
                wrapped = "\n".join(textwrap.wrap(value, width=28 if column == 0 else 24))
                line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 3
                lines = wrapped.split("\n")
                y = (top + bottom - line_height * len(lines)) / 2
                for line in lines:
                    draw.text((x_edges[column] + 12, y), line, font=font, fill=color)
                    y += line_height
            else:
                centered_text(value, box, font, color)

    footer_y = table_top + header_height + len(body) * row_height + 25
    draw.text((margin, footer_y), "Task-specific metric: endpoint error (Event80), final displacement error (gravity/collision/mass×friction), bar tilt MAE (balance), and yellow-light score MAE.", font=fonts["foot"], fill="#52606d")
    draw.text((margin, footer_y + 40), "Blank cells indicate unavailable or withheld metrics; LightSwitch pooled metrics await leakage-free retraining. LPIPS uses official AlexNet.", font=fonts["foot"], fill="#7a4b00")
    image.save(OUTPUT / "sim_standard_pooled_vs_ours.png", dpi=(220, 220))

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="68" font-family="Georgia,serif" font-size="42" font-weight="700" fill="#17324d">Simulation Benchmarks: Ours vs. Standard Pooled World Model</text>',
        f'<text x="{margin}" y="120" font-family="Georgia,serif" font-size="23" fill="#52606d">Shared K=1 query protocols; the pooled baseline receives no test-time adaptation.</text>',
    ]

    def svg_centered(text: str, left: int, top: int, right: int, bottom: int, *, size: int, fill: str, weight: str = "400") -> None:
        lines = text.split("\n")
        x = (left + right) / 2
        line_height = size + 5
        first_y = (top + bottom - line_height * (len(lines) - 1)) / 2 + size * 0.35
        for offset, line in enumerate(lines):
            svg.append(f'<text x="{x:.1f}" y="{first_y + offset * line_height:.1f}" text-anchor="middle" font-family="Georgia,serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(line)}</text>')

    for column, label in enumerate(columns):
        left, right = x_edges[column], x_edges[column + 1]
        svg.append(f'<rect x="{left}" y="{table_top}" width="{right-left}" height="{header_height}" fill="#17324d" stroke="#17324d"/>')
        svg_centered(label, left, table_top, right, table_top + header_height, size=20, fill="white", weight="700")
    for row_index, (record, values) in enumerate(zip(records, body)):
        top = table_top + header_height + row_index * row_height
        group_index = task_order.index(record["task"])
        fill = "#fff4cf" if record["status"] != "complete" else group_colors[group_index % 2]
        for column, value in enumerate(values):
            left, right = x_edges[column], x_edges[column + 1]
            svg.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{row_height}" fill="{fill}" stroke="#c8d2dc"/>')
            weight = "700" if column == 0 or (row_index, column) in best_cells else "400"
            color = "#0b5a75" if column == 1 and record["method"] == "Ours" else "#27313a"
            if column in (0, 1):
                svg.append(f'<text x="{left+12}" y="{top + row_height/2 + 7:.1f}" font-family="Georgia,serif" font-size="20" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>')
            else:
                svg_centered(value, left, top, right, top + row_height, size=20, fill=color, weight=weight)
    svg.append(f'<text x="{margin}" y="{footer_y+18}" font-family="Georgia,serif" font-size="17" fill="#52606d">Task-specific metric: endpoint error (Event80), final displacement error (gravity/collision/mass×friction), bar tilt MAE (balance), and yellow-light score MAE.</text>')
    svg.append(f'<text x="{margin}" y="{footer_y+58}" font-family="Georgia,serif" font-size="17" fill="#7a4b00">Blank cells indicate unavailable or withheld metrics; LightSwitch pooled metrics await leakage-free retraining. LPIPS uses official AlexNet.</text>')
    svg.append("</svg>")
    (OUTPUT / "sim_standard_pooled_vs_ours.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    records = build_records()
    write_outputs(records)
    print(f"[done] rows={len(records)} output={OUTPUT}")


if __name__ == "__main__":
    main()
