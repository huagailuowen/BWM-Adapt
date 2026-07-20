#!/usr/bin/env python3
from __future__ import annotations

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
    _default_pred_name,
    _freeze_pipe,
    _read_jsonl,
    parse_args,
)
from make_gt_pred_comparison import _read_gt_video, _read_pred_video, _resize_rgb
from optimize_action_token_final_frame_fm import _final_context
from wan_video_action.parsers import prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
from wan_video_action.utils import set_global_seed


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(130, min(image.width, 9 * len(label) + 14))
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
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
    output_dir: Path,
    args,
) -> Path:
    sample = prepare_sample_for_rollout(query_dataset[sample_index], sample_index, pipe, args)
    reference_action = sample["action"]
    sample["action"] = action.detach().to(device=reference_action.device, dtype=reference_action.dtype)
    sample["physical_context"] = _compose_known_physical_context(sample, context, args)
    sample["seed"] = int(shared_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _default_pred_name(sample_index, metadata_row)
    sample["output_path"] = str(output_path)
    _run_autoregressive(pipe=pipe, sample=sample, args=args)
    return output_path


def _write_2x4_grid(
    *,
    metadata_rows: list[dict],
    target_indices: list[int],
    dataset_base_path: Path,
    prediction_dir: Path,
    source_index: int,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> Path:
    gt_videos = []
    pred_videos = []
    for target_index in target_indices:
        row = metadata_rows[target_index]
        pred_path = prediction_dir / _default_pred_name(target_index, row)
        num_views = len(row["video"])
        target_width = int(width) * num_views
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        gt_videos.append(_read_gt_video(row, dataset_base_path, int(width), int(height), total_frames))
        pred_videos.append(_read_pred_video(pred_path, target_width, int(height)))

    frame_count = min(len(frames) for frames in [*gt_videos, *pred_videos])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=int(quality)) as writer:
        for frame_id in range(frame_count):
            top = []
            bottom = []
            for target_index, gt_frames, pred_frames in zip(target_indices, gt_videos, pred_videos):
                row = metadata_rows[target_index]
                target_width = int(width) * len(row["video"])
                action_id = int(row["action_id"])
                gt = _resize_rgb(gt_frames[frame_id], target_width, int(height))
                pred = _resize_rgb(pred_frames[frame_id], target_width, int(height))
                top.append(_draw_label(gt, f"GT action-{action_id}"))
                bottom.append(_draw_label(pred, f"Z* s{source_index} + action-{action_id}"))
            writer.append_data(
                np.concatenate(
                    [np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)],
                    axis=0,
                )
            )
    return output_path


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    source_indices = [
        int(value.strip())
        for value in os.environ.get("SAME_Z_SOURCE_INDICES", "205,175").split(",")
        if value.strip()
    ]
    trajectory_path = Path(os.environ["SAME_Z_CONTEXT_TRAJECTORY"])
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

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
    for source_index in source_indices:
        source_row = metadata_rows[source_index]
        if int(source_row.get("action_id", -1)) != 5:
            raise ValueError(f"Source {source_index} must be action-5, got {source_row.get('action_id')}.")
        target_indices = [source_index + offset for offset in range(1, 5)]
        for expected_action, target_index in enumerate(target_indices, start=6):
            target_row = metadata_rows[target_index]
            if int(target_row.get("action_id", -1)) != expected_action:
                raise ValueError(
                    f"Expected action-{expected_action} at index {target_index}, "
                    f"got {target_row.get('action_id')}."
                )
            if float(target_row["friction_mu"]) != float(source_row["friction_mu"]):
                raise ValueError(f"Friction mismatch for source={source_index}, target={target_index}.")

        context, context_step = _final_context(
            trajectory_rows,
            source_index,
            context_reference,
            device=pipe.device,
        )
        prediction_dir = output_root / f"source{source_index:04d}_raw"
        shared_seed = int(args.seed) + source_index
        for target_index in target_indices:
            target_row = metadata_rows[target_index]
            action = torch.as_tensor(
                action_dataset[target_index]["action"],
                device=pipe.device,
                dtype=torch.float32,
            )
            pred_path = _render(
                pipe=pipe,
                query_dataset=query_dataset,
                metadata_row=target_row,
                sample_index=target_index,
                action=action,
                context=context,
                shared_seed=shared_seed,
                output_dir=prediction_dir,
                args=args,
            )
            torch.cuda.empty_cache()
            manifest_rows.append(
                {
                    "source_index": source_index,
                    "source_action_id": 5,
                    "target_index": target_index,
                    "target_action_id": int(target_row["action_id"]),
                    "friction_mu": float(target_row["friction_mu"]),
                    "stage2_context_inner_step": context_step,
                    "shared_diffusion_seed": shared_seed,
                    "prediction_path": str(pred_path),
                }
            )
            with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
                for result in manifest_rows:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"[action_done] source={source_index} target={target_index} "
                f"action={target_row['action_id']} output={pred_path}",
                flush=True,
            )

        grid_path = _write_2x4_grid(
            metadata_rows=metadata_rows,
            target_indices=target_indices,
            dataset_base_path=Path(args.dataset_base_path),
            prediction_dir=prediction_dir,
            source_index=source_index,
            output_path=output_root / "grids" / f"source{source_index:04d}_same_z_actions6_9_2x4.mp4",
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
    manifest = {
        "checkpoint": str(args.stage2_ckpt_path),
        "context_trajectory": str(trajectory_path),
        "source_indices": source_indices,
        "target_action_ids": [6, 7, 8, 9],
        "same_stage2_z_per_environment": True,
        "same_diffusion_seed_per_environment": True,
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] output={output_root}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
