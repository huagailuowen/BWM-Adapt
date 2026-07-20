#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import build_pipeline  # noqa: E402
from infer_stage2_ttt import (  # noqa: E402
    _build_support_dataset,
    _flow_match_loss,
    _prepare_loss_inputs,
    prepare_runtime_config,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


def _read_points(manifest_path: Path, metrics_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = {row["point_id"]: row for row in csv.DictReader(handle)}
    points = []
    for row in manifest:
        point_id = row["point_id"]
        if point_id not in metrics:
            continue
        metric = metrics[point_id]
        points.append(
            {
                **row,
                "automatic_state": metric["automatic_state"],
                "on_score": float(metric["on_score"]),
            }
        )
    if not points:
        raise ValueError("No shared points between the sweep manifest and boundary metrics.")
    return points


def _rankdata(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _lower_loss_auc(on_losses: list[float], off_losses: list[float]) -> float:
    outcomes = []
    for on_loss in on_losses:
        for off_loss in off_losses:
            outcomes.append(1.0 if on_loss < off_loss else 0.5 if on_loss == off_loss else 0.0)
    return float(np.mean(outcomes))


def _best_balanced_accuracy(on_losses: list[float], off_losses: list[float]) -> dict:
    candidates = sorted(set(on_losses + off_losses))
    best = {"threshold": None, "balanced_accuracy": -1.0, "on_recall": 0.0, "off_recall": 0.0}
    for threshold in candidates:
        on_recall = sum(value <= threshold for value in on_losses) / len(on_losses)
        off_recall = sum(value > threshold for value in off_losses) / len(off_losses)
        balanced = 0.5 * (on_recall + off_recall)
        if balanced > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced),
                "on_recall": float(on_recall),
                "off_recall": float(off_recall),
            }
    return best


def _cohen_d(on_losses: list[float], off_losses: list[float]) -> float:
    on = np.asarray(on_losses, dtype=np.float64)
    off = np.asarray(off_losses, dtype=np.float64)
    pooled_numerator = (len(on) - 1) * on.var(ddof=1) + (len(off) - 1) * off.var(ddof=1)
    pooled_denominator = max(len(on) + len(off) - 2, 1)
    pooled_std = math.sqrt(max(float(pooled_numerator / pooled_denominator), 0.0))
    return float((off.mean() - on.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def _write_scatter_svg(rows: list[dict], protocol: str, output_path: Path) -> None:
    selected = [row for row in rows if row["protocol"] == protocol]
    width, height = 960, 620
    margin_left, margin_right, margin_top, margin_bottom = 90, 35, 60, 75
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_values = [float(row["oracle_distance"]) for row in selected]
    y_values = [float(row["loss_mean"]) for row in selected]
    x_min, x_max = 0.0, max(x_values) * 1.04
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max((y_max - y_min) * 0.08, 1e-8)
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / max(x_max - x_min, 1e-12) * plot_width

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / max(y_max - y_min, 1e-12) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f5ee"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{protocol}: flow-matching loss vs oracle latent distance</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
    ]
    for tick in range(6):
        x_value = x_min + tick / 5.0 * (x_max - x_min)
        x = sx(x_value)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#ddd"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.2f}</text>')
    for tick in range(6):
        y_value = y_min + tick / 5.0 * (y_max - y_min)
        y = sy(y_value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.5f}</text>')
    for row in selected:
        color = "#e3b629" if row["automatic_state"] == "on" else "#222222"
        stroke = "#8a6a00" if row["automatic_state"] == "on" else "#b94b3f"
        parts.append(
            f'<circle cx="{sx(float(row["oracle_distance"])):.2f}" cy="{sy(float(row["loss_mean"])):.2f}" r="5" fill="{color}" stroke="{stroke}" stroke-width="1.5"><title>{row["point_id"]}: {row["automatic_state"]}, loss={float(row["loss_mean"]):.7f}</title></circle>'
        )
    parts.extend(
        [
            f'<text x="{margin_left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">L2 distance from Stage1 oracle C2</text>',
            f'<text x="22" y="{margin_top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {margin_top + plot_height / 2})" font-family="sans-serif" font-size="15">Mean weighted flow-matching MSE</text>',
            f'<circle cx="{width - 205}" cy="45" r="6" fill="#e3b629"/><text x="{width - 193}" y="50" font-family="sans-serif" font-size="13">lamp on</text>',
            f'<circle cx="{width - 105}" cy="45" r="6" fill="#222"/><text x="{width - 93}" y="50" font-family="sans-serif" font-size="13">lamp off</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _summarize(per_draw_rows: list[dict], points: list[dict], output_root: Path) -> list[dict]:
    point_lookup = {point["point_id"]: point for point in points}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_draw_rows:
        grouped[(row["protocol"], row["point_id"])].append(float(row["loss"]))

    summary_rows = []
    for (protocol, point_id), losses in grouped.items():
        point = point_lookup[point_id]
        summary_rows.append(
            {
                "protocol": protocol,
                "point_id": point_id,
                "family": point["family"],
                "automatic_state": point["automatic_state"],
                "on_score": point["on_score"],
                "oracle_distance": point["oracle_distance"],
                "nearest_group": point["nearest_group"],
                "loss_mean": statistics.fmean(losses),
                "loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "loss_median": statistics.median(losses),
                "loss_min": min(losses),
                "loss_max": max(losses),
                "draws": len(losses),
            }
        )
    summary_rows.sort(key=lambda row: (row["protocol"], row["family"], float(row["oracle_distance"])))
    with (output_root / "point_loss_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    matched_pairs = [
        ("env0_boundary", "toward_env0_a_0p270", "toward_env0_a_0p280"),
        ("env1_boundary", "toward_env1_a_0p350", "toward_env1_a_0p360"),
        ("failed_stage2_boundary", "toward_failed_stage2_a_0p430", "toward_failed_stage2_a_0p440"),
    ]
    statistics_payload = {"protocols": {}, "matched_boundary_pairs": {}}
    for protocol in sorted({row["protocol"] for row in summary_rows}):
        selected = [row for row in summary_rows if row["protocol"] == protocol]
        on_losses = [float(row["loss_mean"]) for row in selected if row["automatic_state"] == "on"]
        off_losses = [float(row["loss_mean"]) for row in selected if row["automatic_state"] == "off"]
        statistics_payload["protocols"][protocol] = {
            "num_on_points": len(on_losses),
            "num_off_points": len(off_losses),
            "on_loss_mean": statistics.fmean(on_losses),
            "off_loss_mean": statistics.fmean(off_losses),
            "off_minus_on": statistics.fmean(off_losses) - statistics.fmean(on_losses),
            "off_over_on": statistics.fmean(off_losses) / statistics.fmean(on_losses),
            "lower_loss_predicts_on_auc": _lower_loss_auc(on_losses, off_losses),
            "cohen_d_off_minus_on": _cohen_d(on_losses, off_losses),
            "spearman_on_score_vs_negative_loss": _spearman(
                [float(row["on_score"]) for row in selected],
                [-float(row["loss_mean"]) for row in selected],
            ),
            "best_loss_threshold": _best_balanced_accuracy(on_losses, off_losses),
        }

        protocol_draws = [row for row in per_draw_rows if row["protocol"] == protocol]
        draw_lookup = {(row["point_id"], int(row["draw_index"])): float(row["loss"]) for row in protocol_draws}
        for pair_name, on_point, off_point in matched_pairs:
            differences = []
            for draw_index in range(max(int(row["draw_index"]) for row in protocol_draws) + 1):
                on_key = (on_point, draw_index)
                off_key = (off_point, draw_index)
                if on_key in draw_lookup and off_key in draw_lookup:
                    differences.append(draw_lookup[off_key] - draw_lookup[on_key])
            statistics_payload["matched_boundary_pairs"].setdefault(pair_name, {})[protocol] = {
                "on_point": on_point,
                "off_point": off_point,
                "mean_off_minus_on": statistics.fmean(differences),
                "median_off_minus_on": statistics.median(differences),
                "fraction_off_loss_greater": sum(value > 0 for value in differences) / len(differences),
                "differences": differences,
            }

    (output_root / "flow_loss_statistics.json").write_text(
        json.dumps(statistics_payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    for protocol in statistics_payload["protocols"]:
        _write_scatter_svg(summary_rows, protocol, output_root / f"{protocol}_loss_scatter.svg")
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser("Compare flow-matching loss for lamp-on and lamp-off latent contexts.")
    parser = add_general_config(parser)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--analysis_manifest_path", required=True)
    parser.add_argument("--analysis_metrics_path", required=True)
    parser.add_argument("--analysis_support_metadata_path", required=True)
    parser.add_argument("--analysis_output_path", required=True)
    parser.add_argument("--analysis_sample_index", type=int, default=675)
    parser.add_argument("--analysis_draws", type=int, default=16)
    parser.add_argument("--analysis_fixed_timestep", type=int, default=500)
    parser.add_argument("--analysis_seed", type=int, default=20260721)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.analysis_output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    points = _read_points(Path(args.analysis_manifest_path), Path(args.analysis_metrics_path))
    args.support_metadata_path = args.analysis_support_metadata_path
    runtime_config = prepare_runtime_config(args)
    dataset = _build_support_dataset(args, runtime_config)
    pipe = build_pipeline(args)
    pipe.scheduler.set_timesteps(1000, training=True)
    for model_name in pipe.in_iteration_models:
        model = getattr(pipe, model_name)
        if model is not None:
            model.eval()
    data = dataset[int(args.analysis_sample_index)]

    fields = [
        "protocol",
        "point_id",
        "family",
        "automatic_state",
        "on_score",
        "oracle_distance",
        "draw_index",
        "draw_seed",
        "fixed_timestep_index",
        "loss",
    ]
    per_draw_rows = []
    per_draw_path = output_root / "per_draw_flow_loss.csv"
    with per_draw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for point_index, point in enumerate(points, start=1):
                context = torch.tensor(point["context"], dtype=pipe.torch_dtype, device=pipe.device)
                set_global_seed(int(args.analysis_seed))
                inputs = _prepare_loss_inputs(pipe, data, context, args)
                for protocol, fixed_timestep in (
                    ("random_t", None),
                    (f"fixed_t{int(args.analysis_fixed_timestep)}", int(args.analysis_fixed_timestep)),
                ):
                    args.stage2_fixed_timestep_index = fixed_timestep
                    for draw_index in range(int(args.analysis_draws)):
                        draw_seed = int(args.analysis_seed) + draw_index
                        set_global_seed(draw_seed)
                        loss = float(_flow_match_loss(pipe, inputs, args).detach().float().cpu())
                        row = {
                            "protocol": protocol,
                            "point_id": point["point_id"],
                            "family": point["family"],
                            "automatic_state": point["automatic_state"],
                            "on_score": point["on_score"],
                            "oracle_distance": point["oracle_distance"],
                            "draw_index": draw_index,
                            "draw_seed": draw_seed,
                            "fixed_timestep_index": "" if fixed_timestep is None else fixed_timestep,
                            "loss": loss,
                        }
                        writer.writerow(row)
                        per_draw_rows.append(row)
                handle.flush()
                print(
                    f"[flow_loss] {point_index}/{len(points)} id={point['point_id']} "
                    f"state={point['automatic_state']}",
                    flush=True,
                )
                del inputs, context
                if point_index % 8 == 0:
                    torch.cuda.empty_cache()

    _summarize(per_draw_rows, points, output_root)
    print(f"[done] points={len(points)} draws={args.analysis_draws} output={output_root}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
