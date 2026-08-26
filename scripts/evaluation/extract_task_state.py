#!/usr/bin/env python3
"""Create a frozen task-state NPZ from masks or a calibrated lamp ROI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.io import read_mask_array, read_video_frames
from wan_video_action.evaluation.sim_task_extractors import (
    DEFAULT_LIGHT_ROI,
    extract_sim_task_state,
    render_task_state_audit,
)
from wan_video_action.evaluation.task_state import TaskState, save_task_state, states_from_masks


def parse_roi(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1.")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("ROI must have positive area.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extractor", choices=("legacy", "sim_rgb_v1"), default="legacy")
    parser.add_argument("--mask-path")
    parser.add_argument("--mask-key", default="masks")
    parser.add_argument("--video-path")
    parser.add_argument("--roi", type=parse_roi)
    parser.add_argument("--light-threshold", type=float)
    parser.add_argument("--yellow-threshold", type=float, default=0.35)
    parser.add_argument("--main-view-width", type=int, default=224)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-area", type=int, default=3000)
    parser.add_argument("--edge-margin", type=int, default=16)
    parser.add_argument("--audit-video")
    parser.add_argument("--fps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extractor == "sim_rgb_v1":
        if not args.video_path:
            raise SystemExit("sim_rgb_v1 requires --video-path.")
        frames = read_video_frames(args.video_path)
        light_roi = args.roi or DEFAULT_LIGHT_ROI
        state = extract_sim_task_state(
            args.task,
            frames,
            fps=args.fps,
            main_view_width=args.main_view_width,
            min_area=args.min_area,
            max_area=args.max_area,
            edge_margin=args.edge_margin,
            light_roi=light_roi,
            yellow_threshold=args.yellow_threshold,
        )
        if args.audit_video:
            render_task_state_audit(
                frames,
                state,
                args.audit_video,
                task=args.task,
                main_view_width=args.main_view_width,
                light_roi=light_roi,
            )
    elif args.task == "lightswitch":
        if not args.video_path or args.roi is None or args.light_threshold is None:
            raise SystemExit("LightSwitch requires --video-path, --roi and --light-threshold.")
        frames = read_video_frames(args.video_path)
        x0, y0, x1, y1 = args.roi
        if y1 > frames.shape[1] or x1 > frames.shape[2]:
            raise ValueError("Lamp ROI lies outside the video frame.")
        rgb = frames[:, y0:y1, x0:x1].astype(np.float64) / 255.0
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        score = np.mean(luma, axis=(1, 2))
        state = TaskState(
            light_on=(score >= args.light_threshold)[:, None],
            light_score=score[:, None],
            image_height=frames.shape[1],
            image_width=frames.shape[2],
            fps=args.fps,
        )
    else:
        if not args.mask_path:
            raise SystemExit("Object tasks require --mask-path.")
        state = states_from_masks(
            read_mask_array(args.mask_path, key=args.mask_key), fps=args.fps
        )
    save_task_state(args.output, state)


if __name__ == "__main__":
    main()
