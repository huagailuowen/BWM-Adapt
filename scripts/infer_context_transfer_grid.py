#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import (  # noqa: E402
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from make_gt_pred_comparison import (  # noqa: E402
    _default_pred_name,
    _read_gt_video,
    _read_pred_video,
    _resize_rgb,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_contexts_by_source(trajectory_path: str | Path) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for row in _read_jsonl(trajectory_path):
        sample_index = int(row["sample_index"])
        step = int(row["inner_step"])
        if sample_index not in latest or step > int(latest[sample_index]["inner_step"]):
            latest[sample_index] = row
    return latest


def _select_transfer_targets(rows: list[dict], source_index: int, count: int) -> list[int]:
    source = rows[int(source_index)]
    source_mu = float(source["friction_mu"])
    candidates = [
        idx
        for idx, row in enumerate(rows)
        if idx != int(source_index) and abs(float(row["friction_mu"]) - source_mu) <= 1e-8
    ]
    if len(candidates) < int(count):
        raise ValueError(
            f"Need {count} transfer targets for source={source_index}, mu={source_mu}; got {len(candidates)}."
        )
    return candidates[: int(count)]


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(110, min(image.width, 9 * len(label) + 14))
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _write_2x5_grid_video(
    *,
    metadata_rows: list[dict],
    dataset_base_path: Path,
    source_index: int,
    target_indices: list[int],
    pred_dir: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> Path:
    gt_videos = []
    pred_videos = []
    for target_index in target_indices:
        row = metadata_rows[int(target_index)]
        pred_name = _default_pred_name(int(target_index), row)
        pred_path = pred_dir / pred_name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for target={target_index}: {pred_path}")
        num_views = len(row["video"])
        target_width = int(width) * num_views
        total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
        gt_videos.append(_read_gt_video(row, dataset_base_path, int(width), int(height), total_frames))
        pred_videos.append(_read_pred_video(pred_path, target_width, int(height)))

    frame_count = min(len(frames) for frames in gt_videos + pred_videos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=int(quality)) as writer:
        for frame_id in range(frame_count):
            top = []
            bottom = []
            for target_index, gt_frames, pred_frames in zip(target_indices, gt_videos, pred_videos):
                row = metadata_rows[int(target_index)]
                num_views = len(row["video"])
                target_width = int(width) * num_views
                mu = float(row["friction_mu"])
                gt = _resize_rgb(gt_frames[frame_id], target_width, int(height))
                pred = _resize_rgb(pred_frames[frame_id], target_width, int(height))
                top.append(_draw_label(gt, f"GT s{target_index} mu={mu:g}"))
                bottom.append(_draw_label(pred, f"C from s{source_index} -> s{target_index}"))
            writer.append_data(np.concatenate([np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)], axis=0))
    return output_path


def parse_args():
    parser = argparse.ArgumentParser("Transfer a learned latent C to other same-friction chunks and write 2x5 videos.")
    parser = add_general_config(parser)
    parser.add_argument("--trajectory_path", required=True)
    parser.add_argument("--source_indices", required=True)
    parser.add_argument("--targets_per_source", type=int, default=5)
    parser.add_argument("--grid_output_path", required=True)
    parser.add_argument("--raw_output_path", required=True)
    parser.add_argument("--plan_output_path", default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    latest_contexts = _latest_contexts_by_source(args.trajectory_path)
    source_indices = _parse_sample_indices(args.source_indices)
    if not source_indices:
        raise ValueError("--source_indices is empty.")

    plan = []
    for source_index in source_indices:
        source_index = int(source_index)
        if source_index not in latest_contexts:
            raise KeyError(f"No learned final C found for source_index={source_index} in {args.trajectory_path}.")
        targets = _select_transfer_targets(metadata_rows, source_index, int(args.targets_per_source))
        plan.append(
            {
                "source_index": source_index,
                "source_sample_id": metadata_rows[source_index].get("sample_id"),
                "source_friction_mu": metadata_rows[source_index].get("friction_mu"),
                "source_inner_step": latest_contexts[source_index].get("inner_step"),
                "target_indices": targets,
                "target_sample_ids": [metadata_rows[index].get("sample_id") for index in targets],
            }
        )
        print(f"[plan] source={source_index} mu={metadata_rows[source_index].get('friction_mu')} targets={targets}", flush=True)

    plan_path = Path(args.plan_output_path) if args.plan_output_path else Path(args.raw_output_path).parent / "transfer_plan.json"
    _write_json(plan_path, plan)

    raw_root = Path(args.raw_output_path)
    grid_root = Path(args.grid_output_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    grid_root.mkdir(parents=True, exist_ok=True)

    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)

    for item in plan:
        source_index = int(item["source_index"])
        source_context = latest_contexts[source_index]
        context_values = torch.tensor(
            source_context["context_flat"],
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        source_raw_dir = raw_root / f"source{source_index:04d}_mu{float(item['source_friction_mu']):.6f}"
        source_raw_dir.mkdir(parents=True, exist_ok=True)
        for target_index in item["target_indices"]:
            target_index = int(target_index)
            row = metadata_rows[target_index]
            pred_name = _default_pred_name(target_index, row)
            pred_path = source_raw_dir / pred_name
            if pred_path.exists() and args.skip_existing:
                print(f"[skip] {pred_path}", flush=True)
                continue
            sample = dataset[target_index]
            sample = prepare_sample_for_rollout(sample, target_index, pipe, args)
            sample["physical_context"] = context_values
            sample["output_path"] = str(pred_path)
            print(f"[infer] source={source_index} target={target_index} output={pred_path}", flush=True)
            _run_autoregressive(pipe=pipe, sample=sample, args=args)
            torch.cuda.empty_cache()

        grid_path = grid_root / f"source{source_index:04d}_mu{float(item['source_friction_mu']):.6f}_2x5_gt_transfer.mp4"
        _write_2x5_grid_video(
            metadata_rows=metadata_rows,
            dataset_base_path=Path(args.dataset_base_path),
            source_index=source_index,
            target_indices=[int(x) for x in item["target_indices"]],
            pred_dir=source_raw_dir,
            output_path=grid_path,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            quality=int(args.quality),
        )
        print(f"[grid] {grid_path}", flush=True)

    print(f"[done] raw={raw_root} grids={grid_root} plan={plan_path}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
