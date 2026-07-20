#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from infer import build_infer_dataset, build_pipeline
from infer_stage2_ttt import _build_support_dataset, _default_pred_name, _freeze_pipe, _read_jsonl, parse_args
from make_gt_pred_comparison import _read_gt_video, _read_pred_video, _resize_rgb
from optimize_action_token_final_frame_fm import _final_context, _render
from wan_video_action.parsers import prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
from wan_video_action.utils import set_global_seed


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(125, min(image.width, 9 * len(label) + 14))
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _ratio_tag(ratio: float) -> str:
    return f"ratio_{int(round(100 * ratio)):03d}"


def _write_3x3_grid(
    *,
    metadata_row: dict,
    sample_index: int,
    dataset_base_path: Path,
    ratio_predictions: list[tuple[float, float, Path]],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> Path:
    num_views = len(metadata_row["video"])
    target_width = int(width) * num_views
    total_frames = int(metadata_row.get("length", metadata_row["end_frame"] - metadata_row["start_frame"] + 1))
    gt_frames = _read_gt_video(metadata_row, dataset_base_path, int(width), int(height), total_frames)
    predictions = [
        _read_pred_video(path, target_width, int(height))
        for _, _, path in ratio_predictions
    ]
    frame_count = min(len(frames) for frames in [gt_frames, *predictions])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=int(quality)) as writer:
        for frame_id in range(frame_count):
            cells = [
                _draw_label(
                    _resize_rgb(gt_frames[frame_id], target_width, int(height)),
                    f"GT action-5 | s{sample_index}",
                )
            ]
            for (ratio, peak_speed, _), frames in zip(ratio_predictions, predictions):
                cells.append(
                    _draw_label(
                        _resize_rgb(frames[frame_id], target_width, int(height)),
                        f"{ratio:g}x | peak={peak_speed:.3f}",
                    )
                )
            while len(cells) < 9:
                cells.append(np.zeros_like(cells[0]))
            grid_rows = [np.concatenate(cells[index:index + 3], axis=1) for index in range(0, 9, 3)]
            writer.append_data(np.concatenate(grid_rows, axis=0))
    return output_path


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    source_indices = [
        int(value.strip())
        for value in os.environ.get("ACTION_SWEEP_SOURCE_INDICES", "205,175").split(",")
        if value.strip()
    ]
    ratios = [
        float(value.strip())
        for value in os.environ.get(
            "ACTION_SWEEP_RATIOS",
            "0,0.25,0.5,0.75,1,1.25,1.5,1.75",
        ).split(",")
        if value.strip()
    ]
    trajectory_path = Path(os.environ["ACTION_SWEEP_CONTEXT_TRAJECTORY"])
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.action_stat_path).read_text(encoding="utf-8"))
    stats_entry = stats.get(args.action_type, stats.get("eef_delta"))
    stat_min = float(stats_entry.get("p01", stats_entry["min"])[0])
    stat_max = float(stats_entry.get("p99", stats_entry["max"])[0])
    zero_normalized = 2.0 * (0.0 - stat_min) / (stat_max - stat_min) - 1.0

    runtime_config = prepare_runtime_config(args)
    action_dataset = _build_support_dataset(args, runtime_config)
    query_dataset = build_infer_dataset(args)
    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    trajectory_rows = _read_jsonl(trajectory_path)

    pipe = build_pipeline(args)
    load_checkpoint_weights(pipe, args.stage2_ckpt_path)
    _freeze_pipe(pipe)
    context_reference = pipe.physical_context_encoder.default_context.detach()

    manifest_rows = []
    grid_paths = []
    for sample_index in source_indices:
        row = metadata_rows[sample_index]
        if int(row.get("action_id", -1)) != 5:
            raise ValueError(f"Sweep source {sample_index} must be action-5, got {row.get('action_id')}.")
        context, context_step = _final_context(
            trajectory_rows,
            sample_index,
            context_reference,
            device=pipe.device,
        )
        item = action_dataset[sample_index]
        base_action = torch.as_tensor(item["action"], device=pipe.device, dtype=torch.float32)
        if base_action.shape[-1] != 14:
            raise ValueError(f"Expected action dim 14, got {tuple(base_action.shape)}.")
        base_raw_speed = (base_action[..., 7] + 1.0) * 0.5 * (stat_max - stat_min) + stat_min
        metadata_peak = float(row["push_action_peak_x"])
        if abs(float(base_raw_speed.max().cpu()) - metadata_peak) > 1e-4:
            raise ValueError(
                f"a7 physical peak {float(base_raw_speed.max().cpu())} does not match metadata {metadata_peak}."
            )

        ratio_predictions = []
        for ratio in ratios:
            scaled_raw_speed = base_raw_speed * float(ratio)
            scaled_normalized = 2.0 * (scaled_raw_speed - stat_min) / (stat_max - stat_min) - 1.0
            if float(scaled_normalized.min().cpu()) < -1.00001 or float(scaled_normalized.max().cpu()) > 1.00001:
                raise ValueError(
                    f"ratio={ratio:g} leaves normalized training range: "
                    f"[{float(scaled_normalized.min().cpu()):.6f}, {float(scaled_normalized.max().cpu()):.6f}]."
                )
            scaled_action = base_action.clone()
            scaled_action[..., 7] = scaled_normalized
            ratio_dir = output_root / "raw" / _ratio_tag(ratio)
            pred_path = _render(
                pipe=pipe,
                query_dataset=query_dataset,
                metadata_row=row,
                sample_index=sample_index,
                action=scaled_action,
                context=context,
                output_dir=ratio_dir,
                args=args,
            )
            torch.cuda.empty_cache()
            peak_speed = float(scaled_raw_speed.max().cpu())
            ratio_predictions.append((ratio, peak_speed, pred_path))
            manifest_rows.append(
                {
                    "sample_index": sample_index,
                    "episode_index": int(row["episode_index"]),
                    "friction_mu": float(row["friction_mu"]),
                    "action_id": int(row["action_id"]),
                    "stage2_context_inner_step": context_step,
                    "speed_ratio": float(ratio),
                    "physical_peak_speed": peak_speed,
                    "normalized_zero_speed": zero_normalized,
                    "normalized_a7_min": float(scaled_normalized.min().cpu()),
                    "normalized_a7_max": float(scaled_normalized.max().cpu()),
                    "prediction_path": str(pred_path),
                }
            )
            with (output_root / "sweep_manifest.jsonl").open("w", encoding="utf-8") as handle:
                for manifest_row in manifest_rows:
                    handle.write(json.dumps(manifest_row, sort_keys=True) + "\n")
            print(
                f"[ratio_done] sample={sample_index} mu={float(row['friction_mu']):g} "
                f"ratio={ratio:g} peak={peak_speed:.4f} output={pred_path}",
                flush=True,
            )

        grid_path = _write_3x3_grid(
            metadata_row=row,
            sample_index=sample_index,
            dataset_base_path=Path(args.dataset_base_path),
            ratio_predictions=ratio_predictions,
            output_path=output_root / "grids" / f"sample{sample_index:04d}_action5_speed_sweep_3x3.mp4",
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            quality=int(args.quality),
        )
        grid_paths.append(grid_path)
        print(f"[grid_done] {grid_path}", flush=True)

    (output_root / "grid_videos.txt").write_text(
        "".join(f"{path}\n" for path in grid_paths),
        encoding="utf-8",
    )
    summary = {
        "checkpoint": str(args.stage2_ckpt_path),
        "context_trajectory": str(trajectory_path),
        "source_indices": source_indices,
        "speed_ratios": ratios,
        "scaled_action_dimension": 7,
        "physical_speed_stat_min": stat_min,
        "physical_speed_stat_max": stat_max,
        "normalized_zero_speed": zero_normalized,
        "same_diffusion_seed_across_ratios": True,
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] output={output_root}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
