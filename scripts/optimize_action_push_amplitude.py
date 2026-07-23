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
    _prepare_loss_inputs,
    _read_jsonl,
    _shared_inputs,
    parse_args,
)
from make_gt_pred_comparison import _read_gt_video, _read_pred_video, _resize_rgb
from optimize_action_token_final_frame_fm import (
    _fixed_timestep,
    _terminal_flow_loss,
    _terminal_only_item,
)
from wan_video_action.parsers import prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
from wan_video_action.utils import set_global_seed


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_width = min(image.width, max(180, 8 * len(label) + 14))
    draw.rectangle((0, 0, label_width, 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _render(
    *,
    pipe,
    query_dataset,
    row: dict,
    sample_index: int,
    action: torch.Tensor,
    context: torch.Tensor,
    seed: int,
    output_path: Path,
    args,
) -> Path:
    sample = prepare_sample_for_rollout(query_dataset[sample_index], sample_index, pipe, args)
    reference_action = sample["action"]
    sample["action"] = action.detach().to(reference_action.device, reference_action.dtype)
    sample["physical_context"] = _compose_known_physical_context(sample, context, args)
    sample["seed"] = int(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample["output_path"] = str(output_path)
    _run_autoregressive(pipe=pipe, sample=sample, args=args)
    return output_path


def _write_comparison(
    *,
    row: dict,
    dataset_base_path: Path,
    predictions: list[tuple[str, Path]],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> None:
    total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
    target_width = int(width) * len(row["video"])
    videos = [
        (
            "Dataset GT | A=%.6f" % float(row["action_amplitude"]),
            _read_gt_video(row, dataset_base_path, int(width), int(height), total_frames),
        )
    ]
    videos.extend(
        (label, _read_pred_video(path, target_width, int(height)))
        for label, path in predictions
    )
    frame_count = min(len(frames) for _, frames in videos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path), fps=int(fps), codec="libx264", quality=int(quality)
    ) as writer:
        for frame_id in range(frame_count):
            cells = [
                _draw_label(
                    _resize_rgb(frames[frame_id], target_width, int(height)),
                    label,
                )
                for label, frames in videos
            ]
            writer.append_data(np.concatenate(cells, axis=0))


def _write_trajectory_svg(rows: list[dict], target_amplitude: float, output_path: Path) -> None:
    width, height = 850, 480
    left, right, top, bottom = 76, 26, 48, 64
    plot_width, plot_height = width - left - right, height - top - bottom
    x_max = max(int(row["step"]) for row in rows)
    y_min, y_max = 0.0, 0.5

    def x_pos(value: float) -> float:
        return left + value / max(x_max, 1) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="27" text-anchor="middle">Learnable push amplitude with fixed Z</text>',
    ]
    for index in range(6):
        fraction = index / 5.0
        x_value = fraction * x_max
        y_value = fraction * y_max
        x, y = x_pos(x_value), y_pos(y_value)
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#ddd"/>',
                f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle">{x_value:.0f}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>',
                f'<text class="tick" x="{left - 9}" y="{y + 4:.2f}" text-anchor="end">{y_value:.2f}</text>',
            ]
        )
    target_y = y_pos(target_amplitude)
    svg.extend(
        [
            f'<line x1="{left}" y1="{target_y:.2f}" x2="{left + plot_width}" y2="{target_y:.2f}" stroke="#222" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<text class="tick" x="{left + plot_width - 4}" y="{target_y - 7:.2f}" text-anchor="end">target A={target_amplitude:.6f}</text>',
        ]
    )
    points = " ".join(
        f'{x_pos(float(row["step"])):.2f},{y_pos(float(row["amplitude"])):.2f}'
        for row in rows
    )
    svg.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#d1493f" stroke-width="2.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 17}" text-anchor="middle">Inner optimization step</text>',
            f'<text class="label" x="18" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})">Physical push amplitude A</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_loss_landscape_svg(
    rows: list[dict], target_amplitude: float, output_path: Path
) -> None:
    width, height = 850, 480
    left, right, top, bottom = 76, 26, 48, 64
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 0.5
    losses = [float(row["loss"]) for row in rows]
    y_min, y_max = min(losses), max(losses)
    padding = max((y_max - y_min) * 0.12, 1e-6)
    y_min -= padding
    y_max += padding

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    best = min(rows, key=lambda row: float(row["loss"]))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="27" text-anchor="middle">Action-amplitude flow-matching loss landscape</text>',
    ]
    for index in range(6):
        fraction = index / 5.0
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x, y = x_pos(x_value), y_pos(y_value)
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#ddd"/>',
                f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle">{x_value:.2f}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>',
                f'<text class="tick" x="{left - 9}" y="{y + 4:.2f}" text-anchor="end">{y_value:.5f}</text>',
            ]
        )
    target_x = x_pos(target_amplitude)
    points = " ".join(
        f'{x_pos(float(row["amplitude"])):.2f},{y_pos(float(row["loss"])):.2f}'
        for row in rows
    )
    svg.extend(
        [
            f'<line x1="{target_x:.2f}" y1="{top}" x2="{target_x:.2f}" y2="{top + plot_height}" stroke="#222" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<polyline points="{points}" fill="none" stroke="#2463a5" stroke-width="2.5"/>',
        ]
    )
    for row in rows:
        svg.append(
            f'<circle cx="{x_pos(float(row["amplitude"])):.2f}" cy="{y_pos(float(row["loss"])):.2f}" r="4" fill="#2463a5"/>'
        )
    best_x, best_y = x_pos(float(best["amplitude"])), y_pos(float(best["loss"]))
    svg.extend(
        [
            f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="7" fill="#d1493f" stroke="white" stroke-width="2"/>',
            f'<text class="tick" x="{best_x + 10:.2f}" y="{best_y - 8:.2f}">scan min A={float(best["amplitude"]):.3f}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 17}" text-anchor="middle">Physical push amplitude A</text>',
            f'<text class="label" x="18" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})">Flow-matching loss</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    target_index = int(os.environ.get("ACTION_AMP_TARGET_INDEX", "313"))
    initial_amplitude = float(os.environ.get("ACTION_AMP_INITIAL", "0.05"))
    action_x_dim = int(os.environ.get("ACTION_AMP_X_DIM", "7"))
    table_path = Path(os.environ["ACTION_AMP_CONTEXT_TABLE"])
    timestep_fractions = [
        float(value)
        for value in os.environ.get("ACTION_AMP_TIMESTEPS", "0.25,0.5,0.75").split(",")
    ]
    loss_frames = os.environ.get("ACTION_AMP_LOSS_FRAMES", "all").strip().lower()
    if loss_frames not in {"all", "final"}:
        raise ValueError(
            f"ACTION_AMP_LOSS_FRAMES must be all or final, got {loss_frames!r}."
        )
    hide_intermediate = os.environ.get("ACTION_AMP_HIDE_INTERMEDIATE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    scan_points = int(os.environ.get("ACTION_AMP_SCAN_POINTS", "0"))
    learning_rates = [0.02] * 30 + [0.01] * 30 + [0.003] * 30
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_config = prepare_runtime_config(args)
    dataset = _build_support_dataset(args, runtime_config)
    query_dataset = build_infer_dataset(args)
    rows = _read_jsonl(args.dataset_metadata_path)
    row = rows[target_index]
    target_amplitude = float(row["action_amplitude"])
    target_item = dataset[target_index]
    loss_item = (
        _terminal_only_item(target_item)
        if loss_frames == "final" and hide_intermediate
        else target_item
    )
    target_action = torch.as_tensor(target_item["action"], dtype=torch.float32)
    if target_action.shape[-1] != int(args.action_dim):
        raise ValueError(
            f"Expected action_dim={args.action_dim}, got action shape={tuple(target_action.shape)}."
        )
    if not 0 <= action_x_dim < target_action.shape[-1]:
        raise ValueError(f"Invalid x-action dimension: {action_x_dim}.")

    stats = json.loads(Path(args.action_stat_path).read_text(encoding="utf-8"))
    stats_entry = stats.get(args.action_type, stats.get("eef_delta"))
    stat_min = float(stats_entry.get("p01", stats_entry["min"])[0])
    stat_max = float(stats_entry.get("p99", stats_entry["max"])[0])

    pipe = build_pipeline(args)
    load_checkpoint_weights(pipe, args.stage2_ckpt_path)
    _freeze_pipe(pipe)
    pipe.scheduler.set_timesteps(1000, training=True)
    target_action = target_action.to(device=pipe.device)

    context_table = _load_grouped_context_table(table_path)
    friction_values = np.asarray(context_table["friction_values"], dtype=np.float64)
    friction_mu = float(row["friction_mu"])
    context_index = int(np.argmin(np.abs(friction_values - friction_mu)))
    if abs(float(friction_values[context_index]) - friction_mu) > 1e-6:
        raise ValueError(
            f"No fixed context for mu={friction_mu}; nearest={friction_values[context_index]}."
        )
    context = torch.tensor(
        context_table["contexts"][context_index],
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )

    target_x_normalized = target_action[..., action_x_dim]
    target_x_raw = (target_x_normalized + 1.0) * 0.5 * (stat_max - stat_min) + stat_min
    push_mask = target_x_raw > 1e-6
    push_count = int(push_mask.sum().item())
    if push_count < 1:
        raise ValueError("No positive x-action push entries were found.")
    push_profile = torch.zeros_like(target_x_raw)
    push_profile[push_mask] = target_x_raw[push_mask] / target_amplitude

    def build_action(amplitude: torch.Tensor) -> torch.Tensor:
        raw_x = amplitude * push_profile
        normalized_x = 2.0 * (raw_x - stat_min) / (stat_max - stat_min) - 1.0
        action = target_action.clone()
        action[..., action_x_dim] = torch.where(push_mask, normalized_x, target_x_normalized)
        return action

    timesteps = [_fixed_timestep(pipe, fraction) for fraction in timestep_fractions]
    probes = []
    with torch.no_grad():
        for probe_index, timestep in enumerate(timesteps):
            probe_action = build_action(torch.tensor(initial_amplitude, device=pipe.device))
            inputs = dict(
                _shared_inputs(
                    _prepare_loss_inputs(
                        pipe,
                        {**loss_item, "action": probe_action.to(dtype=pipe.torch_dtype)},
                        context,
                        args,
                    )
                )
            )
            generator = torch.Generator(device=pipe.device).manual_seed(
                int(args.seed) + target_index * 10 + probe_index
            )
            noise = torch.randn(
                inputs["input_latents"].shape,
                generator=generator,
                device=inputs["input_latents"].device,
                dtype=inputs["input_latents"].dtype,
            )
            probes.append((timestep, noise))

    def average_loss(action: torch.Tensor) -> torch.Tensor:
        losses = [
            _terminal_flow_loss(
                pipe,
                loss_item,
                action.to(dtype=pipe.torch_dtype),
                context,
                args,
                timestep=timestep,
                noise=noise,
                loss_frames=loss_frames,
            )
            for timestep, noise in probes
        ]
        return torch.stack(losses).mean()

    scan_rows = []
    if scan_points > 1:
        with torch.no_grad():
            for value in torch.linspace(0.0, 0.5, scan_points).tolist():
                scan_loss = float(
                    average_loss(
                        build_action(torch.tensor(value, device=pipe.device))
                    ).cpu()
                )
                scan_rows.append({"amplitude": float(value), "loss": scan_loss})
                print(f"[action_scan] A={value:.6f} loss={scan_loss:.8f}", flush=True)
        with (output_root / "loss_landscape.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scan_rows[0]))
            writer.writeheader()
            writer.writerows(scan_rows)
        _write_loss_landscape_svg(
            scan_rows,
            target_amplitude,
            output_root / "loss_landscape.svg",
        )

    with torch.no_grad():
        oracle_loss = float(average_loss(target_action).cpu())
        initial_action = build_action(torch.tensor(initial_amplitude, device=pipe.device))
        initial_loss = float(average_loss(initial_action).cpu())

    amplitude = torch.tensor(
        initial_amplitude, device=pipe.device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([amplitude], lr=learning_rates[0])
    trajectory = []
    for step, learning_rate in enumerate(learning_rates, start=1):
        optimizer.param_groups[0]["lr"] = float(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        loss = average_loss(build_action(amplitude))
        loss.backward()
        gradient = float(amplitude.grad.detach().cpu())
        optimizer.step()
        with torch.no_grad():
            amplitude.clamp_(0.0, 0.5)
        record = {
            "step": step,
            "lr": float(learning_rate),
            "amplitude": float(amplitude.detach().cpu()),
            "target_amplitude": target_amplitude,
            "amplitude_error": float(amplitude.detach().cpu()) - target_amplitude,
            "loss": float(loss.detach().cpu()),
            "gradient": gradient,
        }
        trajectory.append(record)
        print(
            f"[action_amp] step={step:03d}/090 lr={learning_rate:g} "
            f"A={record['amplitude']:.6f} target={target_amplitude:.6f} "
            f"loss={record['loss']:.8f} grad={gradient:.6g}",
            flush=True,
        )
        with (output_root / "amplitude_trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
            writer.writeheader()
            writer.writerows(trajectory)

    learned_amplitude = float(amplitude.detach().cpu())
    learned_action = build_action(amplitude.detach())
    with torch.no_grad():
        learned_loss = float(average_loss(learned_action).cpu())

    rollout_seed = int(args.seed) + 6802
    initial_path = _render(
        pipe=pipe,
        query_dataset=query_dataset,
        row=row,
        sample_index=target_index,
        action=initial_action,
        context=context,
        seed=rollout_seed,
        output_path=output_root / "raw" / "initial_A005.mp4",
        args=args,
    )
    torch.cuda.empty_cache()
    learned_path = _render(
        pipe=pipe,
        query_dataset=query_dataset,
        row=row,
        sample_index=target_index,
        action=learned_action,
        context=context,
        seed=rollout_seed,
        output_path=output_root / "raw" / "learned_A.mp4",
        args=args,
    )
    torch.cuda.empty_cache()
    oracle_path = _render(
        pipe=pipe,
        query_dataset=query_dataset,
        row=row,
        sample_index=target_index,
        action=target_action,
        context=context,
        seed=rollout_seed,
        output_path=output_root / "raw" / "oracle_A0251724.mp4",
        args=args,
    )
    _write_comparison(
        row=row,
        dataset_base_path=Path(args.dataset_base_path),
        predictions=[
            (f"Model initial | A={initial_amplitude:.6f}", initial_path),
            (f"Model learned | A={learned_amplitude:.6f}", learned_path),
            (f"Model oracle | A={target_amplitude:.6f}", oracle_path),
        ],
        output_path=output_root / "gt_initial_learned_oracle_4x1.mp4",
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        quality=int(args.quality),
    )
    _write_trajectory_svg(
        trajectory,
        target_amplitude,
        output_root / "amplitude_trajectory.svg",
    )
    result = {
        "checkpoint": str(args.stage2_ckpt_path),
        "context_table": str(table_path),
        "context_table_index": context_index,
        "context_fixed": True,
        "friction_mu": friction_mu,
        "target_index": target_index,
        "target_action_id": int(row["action_id"]),
        "action_x_dimension": action_x_dim,
        "positive_push_entries": push_count,
        "initial_amplitude": initial_amplitude,
        "target_amplitude": target_amplitude,
        "learned_amplitude": learned_amplitude,
        "learned_amplitude_error": learned_amplitude - target_amplitude,
        "initial_loss": initial_loss,
        "learned_loss": learned_loss,
        "oracle_loss": oracle_loss,
        "loss_frames": loss_frames,
        "hide_intermediate_frames": hide_intermediate,
        "target_information": (
            "observed first frame plus target final frame only"
            if loss_frames == "final" and hide_intermediate
            else (
                "complete target video clip encoded; loss on final latent slice only"
                if loss_frames == "final"
                else "complete target video clip"
            )
        ),
        "loss_scan_points": scan_points,
        "loss_scan_min_amplitude": (
            float(min(scan_rows, key=lambda row: row["loss"])["amplitude"])
            if scan_rows
            else None
        ),
        "loss_scan_min_loss": (
            float(min(scan_rows, key=lambda row: row["loss"])["loss"])
            if scan_rows
            else None
        ),
        "fixed_timestep_fractions": timestep_fractions,
        "lr_schedule": "0.02:30,0.01:30,0.003:30",
        "same_rollout_seed": rollout_seed,
    }
    (output_root / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[done] {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
