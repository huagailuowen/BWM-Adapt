#!/usr/bin/env python3
"""Build the formal Event80 inference-time schedule ablation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[2]
METRICS = [
    ("psnr_main", "PSNR\nmain", True, ".2f"),
    ("ssim_main", "SSIM\nmain", True, ".4f"),
    ("lpips_multiview", "LPIPS\nmultiview", False, ".4f"),
    ("centroid_ade_px", "Object ADE\n(px)", False, ".2f"),
    ("centroid_fde_px", "Object FDE\n(px)", False, ".2f"),
    ("action_success_all", "Action success\n(%)", True, ".1%"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/evaluation/event80_inner_schedule_ablation.yaml",
    )
    return parser.parse_args()


def read_scoreboard(path: Path, method: str | None) -> dict:
    rows = list(csv.DictReader(path.open()))
    if method is None:
        if len(rows) != 1:
            raise ValueError(f"Expected one row in {path}, found {len(rows)}")
        return rows[0]
    matches = [row for row in rows if row["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"Expected method={method} once in {path}, found {len(matches)}")
    return matches[0]


def main() -> None:
    config = yaml.safe_load(parse_args().config.read_text())
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in config["rows"]:
        source = ROOT / spec["scoreboard"]
        row = read_scoreboard(source, spec.get("method"))
        record = {
            "key": spec["key"],
            "label": spec["label"],
            "steps": int(spec["steps"]),
            "schedule": spec["schedule"],
            "source_scoreboard": spec["scoreboard"],
        }
        record.update({metric: float(row[metric]) for metric, _, _, _ in METRICS})
        records.append(record)

    csv_path = output / "event80_inner_schedule_ablation_table.csv"
    fieldnames = ["key", "label", "steps", "schedule", *[metric for metric, _, _, _ in METRICS], "source_scoreboard"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    best = {}
    for metric, _, maximize, _ in METRICS:
        values = [record[metric] for record in records]
        best[metric] = max(values) if maximize else min(values)

    headers = ["Method", "Z LR schedule", "Steps", *[label for _, label, _, _ in METRICS]]
    cells = []
    for record in records:
        cells.append(
            [
                record["label"],
                record["schedule"],
                str(record["steps"]),
                *[
                    format(record[metric], formatter)
                    for metric, _, _, formatter in METRICS
                ],
            ]
        )

    fig, ax = plt.subplots(figsize=(17.2, 5.6), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.145, 0.235, 0.055, 0.075, 0.075, 0.085, 0.085, 0.085, 0.09, 0.09],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    table.scale(1.0, 2.0)
    for column in range(len(headers)):
        cell = table[(0, column)]
        cell.set_facecolor("#17324D")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")
        cell.set_linewidth(1.0)
    for row_index, record in enumerate(records, start=1):
        base_color = "#E8F1F8" if record["key"] == "ours" else ("#F6F8FA" if row_index % 2 == 0 else "white")
        for column in range(len(headers)):
            cell = table[(row_index, column)]
            cell.set_facecolor(base_color)
            cell.set_edgecolor("#D3DAE1")
            cell.set_linewidth(0.65)
        table[(row_index, 0)].set_text_props(weight="bold" if record["key"] == "ours" else "normal")
        table[(row_index, 0)].get_text().set_ha("left")
        table[(row_index, 1)].get_text().set_ha("left")
        for metric_index, (metric, _, _, _) in enumerate(METRICS):
            if abs(record[metric] - best[metric]) < 1e-12:
                table[(row_index, 3 + metric_index)].set_text_props(weight="bold", color="#0B6B3A")

    ax.set_title(
        "Event80 inference-time environment-code optimization ablation",
        fontsize=16,
        weight="bold",
        pad=18,
        color="#17212B",
    )
    ax.text(
        0.5,
        0.055,
        "Same step-7272 checkpoint, active-35 initialization, ID5/OOD5 environments, K=1 informative support, and 90 disjoint queries.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(output / "event80_inner_schedule_ablation_table.svg", bbox_inches="tight")
    fig.savefig(output / "event80_inner_schedule_ablation_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    protocol = {
        "checkpoint": "outputs/curc32r65ib2-nc015_88823/step-7272.safetensors",
        "fixed_protocol": "Event80 ID5/OOD5, K=1 informative support (25-60 px), nine disjoint queries per environment",
        "query_count_per_method": 90,
        "rows": records,
        "best_by_metric": best,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
