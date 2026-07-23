#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from infer import _run_autoregressive, build_infer_dataset, build_pipeline, prepare_sample_for_rollout
from infer_stage2_ttt import (
    _build_support_dataset,
    _compose_known_physical_context,
    _freeze_pipe,
    _load_grouped_context_table,
    _read_jsonl,
    parse_args,
)
from make_gt_pred_comparison import _read_pred_video, _resize_rgb
from wan_video_action.parsers import prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
from wan_video_action.utils import set_global_seed


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_width = min(image.width, max(170, 8 * len(label) + 14))
    draw.rectangle((0, 0, label_width, 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _render(
    *,
    pipe,
    query_dataset,
    metadata_row: dict,
    sample_index: int,
    action: torch.Tensor,
    context: torch.Tensor,
    shared_seed: int,
    output_path: Path,
    args,
) -> Path:
    sample = prepare_sample_for_rollout(query_dataset[sample_index], sample_index, pipe, args)
    reference_action = sample["action"]
    if tuple(action.shape) != tuple(reference_action.shape):
        raise ValueError(
            f"Action shape mismatch: midpoint={tuple(action.shape)} query={tuple(reference_action.shape)}."
        )
    sample["action"] = action.detach().to(
        device=reference_action.device,
        dtype=reference_action.dtype,
    )
    sample["physical_context"] = _compose_known_physical_context(sample, context, args)
    sample["seed"] = int(shared_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample["output_path"] = str(output_path)
    _run_autoregressive(pipe=pipe, sample=sample, args=args)
    return output_path


def _blue_object_centroid(frame: np.ndarray, view_width: int) -> np.ndarray | None:
    image = frame[:, :view_width, :3].astype(np.int16)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    yy, xx = np.indices(red.shape)
    mask = (
        (yy >= int(0.39 * image.shape[0]))
        & (blue > red + 5)
        & (blue > green + 2)
        & (blue > 25)
        & (red < 120)
    )
    if int(mask.sum()) < 35:
        return None
    return np.asarray([float(xx[mask].mean()), float(yy[mask].mean())])


def _predicted_displacement(frames: list[np.ndarray], view_width: int) -> float:
    initial = _blue_object_centroid(frames[0], view_width)
    final = _blue_object_centroid(frames[-1], view_width)
    if initial is None or final is None:
        return float("nan")
    return float(np.linalg.norm(final - initial))


def _write_grid(
    *,
    columns: list[dict],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> None:
    videos = []
    for column in columns:
        row_videos = {}
        for kind in ("lower", "midpoint", "upper"):
            row_videos[kind] = _read_pred_video(
                Path(column[f"{kind}_path"]),
                int(width) * 2,
                int(height),
            )
        videos.append(row_videos)
    frame_count = min(
        len(videos[column][kind])
        for column in range(len(videos))
        for kind in ("lower", "midpoint", "upper")
    )
    labels = {
        "lower": "seen lower",
        "midpoint": "unseen midpoint",
        "upper": "seen upper",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=int(fps),
        codec="libx264",
        quality=int(quality),
    ) as writer:
        for frame_id in range(frame_count):
            grid_rows = []
            for kind in ("lower", "midpoint", "upper"):
                cells = []
                for column, row_videos in zip(columns, videos):
                    amplitude = float(column[f"{kind}_amplitude"])
                    frame = _resize_rgb(
                        row_videos[kind][frame_id],
                        int(width) * 2,
                        int(height),
                    )
                    cells.append(_draw_label(frame, f"{labels[kind]} A={amplitude:.4f}"))
                grid_rows.append(np.concatenate(cells, axis=1))
            writer.append_data(np.concatenate(grid_rows, axis=0))


def _write_curve(rows: list[dict], output_path: Path) -> None:
    styles = {
        "lower": ("circle", "#2463a5", "seen lower"),
        "midpoint": ("triangle", "#d1493f", "unseen midpoint"),
        "upper": ("square", "#3c8c59", "seen upper"),
    }
    valid_rows = [
        row for row in rows if np.isfinite(float(row["predicted_displacement_px"]))
    ]
    if not valid_rows:
        raise ValueError("No finite predicted displacements are available for the curve.")
    width, height = 900, 520
    left, right, top, bottom = 82, 28, 52, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row["amplitude"]) for row in valid_rows]
    y_values = [float(row["predicted_displacement_px"]) for row in valid_rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.0, max(y_values) * 1.1
    if y_max <= 0.0:
        y_max = 1.0

    def x_pos(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-12) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">Board-touch action continuity at friction mu=0.0682</text>',
    ]
    for step in range(6):
        fraction = step / 5.0
        x_value = x_min + fraction * (x_max - x_min)
        x = x_pos(x_value)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#ddd"/>')
        svg.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle">{x_value:.3f}</text>')
        y_value = y_min + fraction * (y_max - y_min)
        y = y_pos(y_value)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{y_value:.1f}</text>')
    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle">Action peak amplitude A</text>',
            f'<text class="label" x="20" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2})">Predicted object displacement (px)</text>',
        ]
    )
    legend_x = left + 18
    for legend_index, (kind, (marker, color, label)) in enumerate(styles.items()):
        selected = sorted(
            (row for row in valid_rows if row["kind"] == kind),
            key=lambda row: row["amplitude"],
        )
        points = [
            (x_pos(float(row["amplitude"])), y_pos(float(row["predicted_displacement_px"])))
            for row in selected
        ]
        svg.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in points:
            if marker == "circle":
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
            elif marker == "triangle":
                svg.append(
                    f'<polygon points="{x:.2f},{y - 6:.2f} {x - 5.5:.2f},{y + 4.5:.2f} {x + 5.5:.2f},{y + 4.5:.2f}" fill="{color}"/>'
                )
            else:
                svg.append(f'<rect x="{x - 4.5:.2f}" y="{y - 4.5:.2f}" width="9" height="9" fill="{color}"/>')
        lx = legend_x + legend_index * 170
        svg.append(f'<circle cx="{lx}" cy="{top + 18}" r="5" fill="{color}"/>')
        svg.append(f'<text class="tick" x="{lx + 10}" y="{top + 22}">{label}</text>')
    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    group_start = int(os.environ.get("ACTION_MIDPOINT_GROUP_START", "300"))
    gaps = [
        int(value.strip())
        for value in os.environ.get(
            "ACTION_MIDPOINT_GAPS",
            "0,3,6,9,12,15,18,21,24,28",
        ).split(",")
        if value.strip()
    ]
    base_action_id = int(os.environ.get("ACTION_MIDPOINT_BASE_ACTION_ID", "15"))
    table_path = Path(os.environ["ACTION_MIDPOINT_CONTEXT_TABLE"])
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    base_index = group_start + base_action_id
    base_row = metadata_rows[base_index]
    friction_mu = float(base_row["friction_mu"])
    for offset in range(30):
        row = metadata_rows[group_start + offset]
        if int(row["action_id"]) != offset:
            raise ValueError(f"Expected action_id={offset} at sample={group_start + offset}.")
        if abs(float(row["friction_mu"]) - friction_mu) > 1e-8:
            raise ValueError("All sweep actions must use the same friction group.")
    for gap in gaps:
        if not 0 <= gap < 29:
            raise ValueError(f"Gap index must be in [0, 28], got {gap}.")

    runtime_config = prepare_runtime_config(args)
    action_dataset = _build_support_dataset(args, runtime_config)
    query_dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    load_checkpoint_weights(pipe, args.stage2_ckpt_path)
    _freeze_pipe(pipe)

    context_table = _load_grouped_context_table(table_path)
    friction_values = np.asarray(context_table["friction_values"], dtype=np.float64)
    context_index = int(np.argmin(np.abs(friction_values - friction_mu)))
    if abs(float(friction_values[context_index]) - friction_mu) > 1e-6:
        raise ValueError(
            f"No training-time context for mu={friction_mu}; nearest={friction_values[context_index]}."
        )
    context = torch.tensor(
        context_table["contexts"][context_index],
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    shared_seed = int(args.seed) + 6802

    columns = []
    metric_rows = []
    for column_id, gap in enumerate(gaps):
        lower_index = group_start + gap
        upper_index = lower_index + 1
        lower_row = metadata_rows[lower_index]
        upper_row = metadata_rows[upper_index]
        lower_action = torch.as_tensor(
            action_dataset[lower_index]["action"],
            device=pipe.device,
            dtype=torch.float32,
        )
        upper_action = torch.as_tensor(
            action_dataset[upper_index]["action"],
            device=pipe.device,
            dtype=torch.float32,
        )
        if tuple(lower_action.shape) != tuple(upper_action.shape):
            raise ValueError(f"Cannot interpolate action shapes {lower_action.shape} and {upper_action.shape}.")
        midpoint_action = 0.5 * (lower_action + upper_action)
        lower_amplitude = float(lower_row["action_amplitude"])
        upper_amplitude = float(upper_row["action_amplitude"])
        midpoint_amplitude = 0.5 * (lower_amplitude + upper_amplitude)
        column = {
            "column": column_id,
            "lower_action_id": gap,
            "upper_action_id": gap + 1,
            "lower_amplitude": lower_amplitude,
            "midpoint_amplitude": midpoint_amplitude,
            "upper_amplitude": upper_amplitude,
        }
        for kind, action, amplitude in (
            ("lower", lower_action, lower_amplitude),
            ("midpoint", midpoint_action, midpoint_amplitude),
            ("upper", upper_action, upper_amplitude),
        ):
            output_path = output_root / "raw" / f"column{column_id:02d}_{kind}_A{amplitude:.6f}.mp4"
            _render(
                pipe=pipe,
                query_dataset=query_dataset,
                metadata_row=base_row,
                sample_index=base_index,
                action=action,
                context=context,
                shared_seed=shared_seed,
                output_path=output_path,
                args=args,
            )
            torch.cuda.empty_cache()
            frames = _read_pred_video(output_path, int(args.width) * 2, int(args.height))
            displacement = _predicted_displacement(frames, int(args.width))
            column[f"{kind}_path"] = str(output_path)
            metric_rows.append(
                {
                    "column": column_id,
                    "kind": kind,
                    "action_id_lower": gap,
                    "action_id_upper": gap + 1,
                    "amplitude": amplitude,
                    "predicted_displacement_px": displacement,
                    "prediction_path": str(output_path),
                }
            )
            print(
                f"[rollout] column={column_id} kind={kind} A={amplitude:.6f} "
                f"displacement={displacement:.3f} output={output_path}",
                flush=True,
            )
        columns.append(column)

    with (output_root / "rollout_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.stage2_ckpt_path),
                "context_table": str(table_path),
                "context_table_index": context_index,
                "friction_mu": friction_mu,
                "fixed_query_sample": base_index,
                "fixed_query_action_id": base_action_id,
                "gap_indices": gaps,
                "midpoint_amplitudes": [column["midpoint_amplitude"] for column in columns],
                "same_initial_observation": True,
                "same_training_time_context": True,
                "same_diffusion_seed": shared_seed,
                "midpoint_definition": "elementwise mean of adjacent normalized 7D action trajectories",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_grid(
        columns=columns,
        output_path=output_root / "action_midpoint_continuity_3x10.mp4",
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        quality=int(args.quality),
    )
    _write_curve(metric_rows, output_root / "action_displacement_curve.svg")
    print(f"[done] output={output_root}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
