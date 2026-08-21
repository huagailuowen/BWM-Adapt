#!/usr/bin/env python3
"""Plot the learned PnP machine context table in a two-dimensional PCA space."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default="PnP machine latent Z")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    table_payload = load_json(args.table_path)
    records = table_payload["records"] if isinstance(table_payload, dict) else table_payload
    plan = load_json(args.collection_plan)
    machines = {int(item["index"]): item for item in plan["target_machines"]}

    rows = []
    for record in records:
        machine_index = int(round(float(record["friction_mu"])))
        if machine_index not in machines:
            raise ValueError(f"Machine index {machine_index} is absent from the collection plan")
        machine = machines[machine_index]
        rows.append(
            {
                "machine_index": machine_index,
                "machine_id": machine["machine_id"],
                "offset": float(machine["joint6_initial_offset_deg"]),
                "gain": float(machine["joint6_command_response_gain"]),
                "context": np.asarray(record["context"], dtype=np.float64).reshape(-1),
            }
        )
    rows.sort(key=lambda item: item["machine_index"])

    contexts = np.stack([item["context"] for item in rows])
    offsets = np.asarray([item["offset"] for item in rows])
    gains = np.asarray([item["gain"] for item in rows])
    centered = contexts - contexts.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ components[:2].T
    explained = singular_values**2
    explained = explained / explained.sum()

    if correlation(scores[:, 0], offsets) < 0.0:
        scores[:, 0] *= -1.0
        components[0] *= -1.0
    if correlation(scores[:, 1], gains) < 0.0:
        scores[:, 1] *= -1.0
        components[1] *= -1.0

    corr_pc1_offset = correlation(scores[:, 0], offsets)
    corr_pc2_offset = correlation(scores[:, 1], offsets)
    corr_pc1_gain = correlation(scores[:, 0], gains)
    corr_pc2_gain = correlation(scores[:, 1], gains)

    step_match = re.search(r"step[-_](\d+)", args.table_path.name)
    step_label = step_match.group(1) if step_match else "unknown"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "machine_index",
                "machine_id",
                "joint6_initial_offset_deg",
                "joint6_command_response_gain",
                "pc1",
                "pc2",
                "context_l2",
            ]
        )
        for item, score in zip(rows, scores):
            writer.writerow(
                [
                    item["machine_index"],
                    item["machine_id"],
                    item["offset"],
                    item["gain"],
                    float(score[0]),
                    float(score[1]),
                    float(np.linalg.norm(item["context"])),
                ]
            )

    cmap = LinearSegmentedColormap.from_list(
        "joint_offset", ["#174a8b", "#77a9cf", "#eee6d7", "#df8068", "#a51f2d"]
    )
    norm = Normalize(vmin=float(offsets.min()), vmax=float(offsets.max()))
    marker_by_gain = {1.0: "o", 0.5: "^"}

    fig, ax = plt.subplots(figsize=(12.2, 7.5), facecolor="#f2eee5")
    fig.subplots_adjust(left=0.09, right=0.78, top=0.84, bottom=0.13)
    ax.set_facecolor("#fffdf8")

    for offset in sorted(set(offsets.tolist())):
        pair = np.flatnonzero(np.isclose(offsets, offset))
        if len(pair) > 1:
            ax.plot(
                scores[pair, 0],
                scores[pair, 1],
                color="#9b958a",
                linewidth=1.0,
                linestyle=(0, (3, 3)),
                alpha=0.65,
                zorder=1,
            )

    for gain in sorted(set(gains.tolist()), reverse=True):
        selected = np.flatnonzero(np.isclose(gains, gain))
        ax.scatter(
            scores[selected, 0],
            scores[selected, 1],
            c=offsets[selected],
            cmap=cmap,
            norm=norm,
            marker=marker_by_gain.get(gain, "s"),
            s=110,
            edgecolors="#171717",
            linewidths=1.0,
            zorder=3,
        )

    for item, score in zip(rows, scores):
        ax.annotate(
            f"m{item['machine_index']:02d}",
            xy=(score[0], score[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#34312d",
            zorder=4,
        )

    ax.axhline(0.0, color="#d7d0c4", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#d7d0c4", linewidth=0.8, zorder=0)
    ax.grid(color="#ded8cd", linewidth=0.65, alpha=0.55)
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}% variance)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{args.title} | step {step_label}", x=0.09, y=0.955, ha="left", fontsize=20, weight="bold")
    ax.set_title(
        "Color: joint-6 initial offset   Shape: command response gain   Dashed: matched offset",
        loc="left",
        fontsize=10,
        color="#5d574f",
        pad=12,
    )

    color_ax = fig.add_axes([0.82, 0.53, 0.025, 0.28])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
    colorbar.set_label("Joint-6 initial offset (deg)", fontsize=10)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#d7d0c4", markeredgecolor="#171717", markersize=9, label="Gain 1.0"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#d7d0c4", markeredgecolor="#171717", markersize=9, label="Gain 0.5"),
    ]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.805, 0.45), frameon=False)

    metrics = (
        f"Physical alignment\n"
        f"corr(PC1, offset)  {corr_pc1_offset:+.3f}\n"
        f"corr(PC2, offset)  {corr_pc2_offset:+.3f}\n"
        f"corr(PC1, gain)    {corr_pc1_gain:+.3f}\n"
        f"corr(PC2, gain)    {corr_pc2_gain:+.3f}\n\n"
        f"PC1 + PC2 variance\n"
        f"{(explained[0] + explained[1]) * 100.0:.1f}%"
    )
    fig.text(
        0.81,
        0.34,
        metrics,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
        color="#34312d",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#fffaf0", "edgecolor": "#d4ccbe"},
    )

    svg_path = args.output_prefix.with_suffix(".svg")
    png_path = args.output_prefix.with_suffix(".png")
    fig.savefig(svg_path, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"step={step_label} records={len(rows)} context_dim={contexts.shape[1]}")
    print(f"explained_pc1={explained[0]:.6f} explained_pc2={explained[1]:.6f}")
    print(
        "correlations "
        f"pc1_offset={corr_pc1_offset:.6f} pc2_offset={corr_pc2_offset:.6f} "
        f"pc1_gain={corr_pc1_gain:.6f} pc2_gain={corr_pc2_gain:.6f}"
    )
    print(f"svg={svg_path}")
    print(f"png={png_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
