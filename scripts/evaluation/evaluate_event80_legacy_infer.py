#!/usr/bin/env python3
"""Smoke-test global and object metrics on a legacy Event80 inference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.event80_pushbox import (
    Event80TrackerConfig,
    track_event80_block,
)
from wan_video_action.evaluation.io import (
    read_video_frames,
    write_json_atomic,
    write_jsonl_atomic,
)
from wan_video_action.metrics.aggregation import aggregate_query_metrics
from wan_video_action.metrics.global_video import (
    LPIPSEvaluator,
    frame_psnr,
    frame_ssim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-index", type=int, action="append")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    count = min(len(gt), len(pred))
    gt_float = gt[:count].astype(np.float32) / 255.0
    pred_float = pred[:count].astype(np.float32) / 255.0
    return float(np.mean(frame_psnr(gt_float, pred_float)))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    count = min(len(gt), len(pred))
    gt_float = gt[:count].astype(np.float32) / 255.0
    pred_float = pred[:count].astype(np.float32) / 255.0
    return float(np.mean(frame_ssim(gt_float, pred_float)))


def sample_indices_from_video_list(path: Path) -> set[int]:
    pattern = re.compile(r"sample(\d+)_")
    output = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            output.add(int(match.group(1)))
    return output


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = config["source"]
    run_dir = Path(source["inference_dir"]).expanduser().resolve()
    dataset_dir = Path(source["dataset_dir"]).expanduser().resolve()
    metadata_path = Path(source["metadata_path"])
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    output_root = Path(config["output_dir"]).expanduser().resolve()
    method = str(config["method"])
    seed = int(config["seed"])
    future_start = int(config.get("future_start_frame", 1))
    lpips_config = config.get("lpips", {})
    lpips_enabled = bool(lpips_config.get("enabled", True))
    lpips_evaluator = (
        LPIPSEvaluator(
            net=str(lpips_config.get("net", "alex")),
            device=str(lpips_config.get("device", "auto")),
        )
        if lpips_enabled else None
    )
    lpips_batch_size = int(lpips_config.get("batch_size", 8))

    metadata_rows = read_jsonl(metadata_path)
    metadata_by_index = {
        int(row["episode_index"]): row for row in metadata_rows
    }
    result_rows = read_jsonl(run_dir / "results.jsonl")
    selected = set(args.sample_index or [])
    if selected:
        result_rows = [
            row for row in result_rows if int(row["sample_index"]) in selected
        ]
    if not result_rows:
        raise ValueError("No inference records match the requested samples.")
    id_indices = sample_indices_from_video_list(run_dir / "id_videos.txt")

    tracker_config = Event80TrackerConfig(**config.get("tracker", {}))
    global_dir = output_root / "methods" / method / "legacy_89097" / f"seed_{seed}" / "metrics" / "global"
    centroid_dir = output_root / "methods" / method / "legacy_89097" / f"seed_{seed}" / "metrics" / "centroid_trajectory"
    protocol_dir = output_root / "protocol"
    for path in (global_dir, centroid_dir, protocol_dir):
        path.mkdir(parents=True, exist_ok=True)

    global_rows = []
    centroid_rows = []
    centroid_frame_rows = []
    manifest_rows = []
    for result in result_rows:
        sample_index = int(result["sample_index"])
        metadata = metadata_by_index[sample_index]
        start_frame = int(metadata["start_frame"])
        frame_count = int(metadata["length"])
        video_paths = [dataset_dir / value for value in metadata["video"]]
        gt_main = read_video_frames(video_paths[0], start_frame, frame_count)
        gt_wrist = read_video_frames(video_paths[1], start_frame, frame_count)
        pred_path = Path(result["prediction_path"])
        if not pred_path.is_absolute():
            pred_path = ROOT / pred_path
        pred_full = read_video_frames(pred_path, 0, frame_count)
        main_width = gt_main.shape[2]
        pred_main = pred_full[:, :, :main_width]
        pred_wrist = pred_full[:, :, main_width:main_width + gt_wrist.shape[2]]
        gt_multiview = np.concatenate((gt_main, gt_wrist), axis=2)
        count = min(len(gt_multiview), len(pred_full))
        evaluation_slice = slice(future_start, count)
        domain = "id" if sample_index in id_indices else "ood"
        common = {
            "sample_id": str(metadata["sample_id"]),
            "environment_id": str(metadata["mu_tag"]),
            "method": method,
            "split": "legacy_same_episode_smoke",
            "domain": domain,
            "support_size": 1,
            "seed": seed,
        }
        global_row = {
            **common,
            "evaluated_frames": count - future_start,
            "psnr_main": psnr(gt_main[evaluation_slice], pred_main[evaluation_slice]),
            "psnr_wrist": psnr(gt_wrist[evaluation_slice], pred_wrist[evaluation_slice]),
            "psnr_multiview": psnr(gt_multiview[evaluation_slice], pred_full[evaluation_slice]),
            "ssim_main": ssim(gt_main[evaluation_slice], pred_main[evaluation_slice]),
            "ssim_wrist": ssim(gt_wrist[evaluation_slice], pred_wrist[evaluation_slice]),
            "ssim_multiview": float(np.mean([
                ssim(gt_main[evaluation_slice], pred_main[evaluation_slice]),
                ssim(gt_wrist[evaluation_slice], pred_wrist[evaluation_slice]),
            ])),
        }
        if lpips_evaluator is not None:
            gt_main_future = (
                gt_main[evaluation_slice].astype(np.float32) / 255.0
            )
            pred_main_future = (
                pred_main[evaluation_slice].astype(np.float32) / 255.0
            )
            gt_wrist_future = (
                gt_wrist[evaluation_slice].astype(np.float32) / 255.0
            )
            pred_wrist_future = (
                pred_wrist[evaluation_slice].astype(np.float32) / 255.0
            )
            lpips_values = lpips_evaluator(
                np.concatenate((gt_main_future, gt_wrist_future), axis=0),
                np.concatenate((pred_main_future, pred_wrist_future), axis=0),
                batch_size=lpips_batch_size,
            )
            future_count = len(gt_main_future)
            global_row["lpips_main"] = float(np.mean(
                lpips_values[:future_count]
            ))
            global_row["lpips_wrist"] = float(np.mean(
                lpips_values[future_count:]
            ))
            global_row["lpips_multiview"] = float(np.mean([
                global_row["lpips_main"], global_row["lpips_wrist"]
            ]))
        global_rows.append(global_row)

        gt_track = track_event80_block(gt_main, tracker_config)
        pred_track = track_event80_block(pred_main, tracker_config)
        diagonal = float(np.hypot(gt_main.shape[1], gt_main.shape[2]))
        distances_px = []
        normalized_distances = []
        evaluated_indices = list(range(future_start, count))
        for frame_index in evaluated_indices:
            gt_center = gt_track.centers[frame_index]
            pred_center = pred_track.centers[frame_index]
            if not np.all(np.isfinite(gt_center)):
                raise ValueError(
                    f"GT tracker lost sample {sample_index} at frame {frame_index}."
                )
            if np.all(np.isfinite(pred_center)):
                distance_px = float(np.linalg.norm(gt_center - pred_center))
            else:
                distance_px = diagonal
            normalized_distance = distance_px / diagonal
            distances_px.append(distance_px)
            normalized_distances.append(normalized_distance)
            centroid_frame_rows.append({
                **common,
                "window_frame": frame_index,
                "source_frame": start_frame + frame_index,
                "gt_x_px": float(gt_center[0]),
                "gt_y_px": float(gt_center[1]),
                "pred_x_px": (
                    float(pred_center[0])
                    if np.isfinite(pred_center[0]) else None
                ),
                "pred_y_px": (
                    float(pred_center[1])
                    if np.isfinite(pred_center[1]) else None
                ),
                "centroid_distance_px": distance_px,
                "centroid_distance_normalized": normalized_distance,
            })

        centroid_row = {
            **common,
            "evaluated_frames": len(evaluated_indices),
            "centroid_mean_distance_px": float(np.mean(distances_px)),
            "centroid_mean_distance_normalized": float(
                np.mean(normalized_distances)
            ),
        }
        centroid_rows.append(centroid_row)
        manifest_rows.append({
            **common,
            "gt_video_path": str(video_paths[0]),
            "pred_video_path": str(pred_path),
            "gt_start_frame": start_frame,
            "pred_start_frame": 0,
            "num_frames": frame_count,
            "metadata": {
                "legacy_support_equals_query": True,
                "main_camera_pred_crop": [0, 0, main_width, gt_main.shape[1]],
                "event80_primary_metric": "per_frame_centroid_distance",
            },
        })

    global_metrics = [
        "psnr_main", "psnr_wrist", "psnr_multiview",
        "ssim_main", "ssim_wrist", "ssim_multiview",
    ]
    if lpips_evaluator is not None:
        global_metrics.extend([
            "lpips_main", "lpips_wrist", "lpips_multiview",
        ])
    centroid_metrics = (
        "centroid_mean_distance_px",
        "centroid_mean_distance_normalized",
    )
    global_summary = aggregate_query_metrics(global_rows, global_metrics)
    centroid_summary = aggregate_query_metrics(centroid_rows, centroid_metrics)
    write_jsonl_atomic(global_dir / "global_per_query.jsonl", global_rows)
    write_json_atomic(global_dir / "global_summary.json", global_summary)
    write_jsonl_atomic(centroid_dir / "centroid_per_query.jsonl", centroid_rows)
    write_jsonl_atomic(
        centroid_dir / "centroid_per_frame.jsonl",
        centroid_frame_rows,
    )
    write_json_atomic(centroid_dir / "centroid_summary.json", centroid_summary)
    write_jsonl_atomic(protocol_dir / "manifest.jsonl", manifest_rows)
    write_json_atomic(output_root / "smoke_summary.json", {
        "status": "legacy_metric_smoke_not_final_benchmark",
        "lpips": (
            "official_learned_calibrated_alexnet"
            if lpips_evaluator is not None else "skipped"
        ),
        "future_frames_only": True,
        "future_start_frame": future_start,
        "global": global_summary,
        "centroid_trajectory": centroid_summary,
    })
    print(f"[done] samples={len(result_rows)} output={output_root}")


if __name__ == "__main__":
    main()
