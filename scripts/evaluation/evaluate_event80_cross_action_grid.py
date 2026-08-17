#!/usr/bin/env python3
"""Evaluate one-support-to-many-query Event80 context transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from wan_video_action.metrics.global_video import frame_psnr, frame_ssim


METRICS = (
    "centroid_mean_distance_px",
    "centroid_mean_distance_normalized",
    "psnr_main",
    "psnr_wrist",
    "psnr_multiview",
    "ssim_main",
    "ssim_wrist",
    "ssim_multiview",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_pair(
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(gt), len(pred))
    return (
        gt[:count].astype(np.float32) / 255.0,
        pred[:count].astype(np.float32) / 255.0,
    )


def appearance_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    gt_float, pred_float = normalized_pair(gt, pred)
    return (
        float(np.mean(frame_psnr(gt_float, pred_float))),
        float(np.mean(frame_ssim(gt_float, pred_float))),
    )


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = config["source"]
    run_dir = Path(source["inference_dir"]).expanduser().resolve()
    dataset_dir = Path(source["dataset_dir"]).expanduser().resolve()
    metadata_path = Path(source["metadata_path"])
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    transfer_plan_path = run_dir / source.get(
        "transfer_plan", "same_friction_other_actions/transfer_plan.json"
    )
    transfer_plan = json.loads(transfer_plan_path.read_text(encoding="utf-8"))
    metadata_by_index = {
        int(row["episode_index"]): row for row in read_jsonl(metadata_path)
    }

    output_root = Path(config["output_dir"]).expanduser().resolve()
    method = str(config["method"])
    seed = int(config["seed"])
    future_start = int(config.get("future_start_frame", 1))
    id_sources = {int(value) for value in config["id_source_indices"]}
    tracker_config = Event80TrackerConfig(**config.get("tracker", {}))

    metric_dir = (
        output_root / "methods" / method / "legacy_89098"
        / f"seed_{seed}" / "metrics" / "cross_action_transfer"
    )
    protocol_dir = output_root / "protocol"
    metric_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    query_rows: list[dict] = []
    frame_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for environment in transfer_plan:
        source_index = int(environment["source_index"])
        friction_mu = float(environment["source_friction_mu"])
        domain = "id" if source_index in id_sources else "ood"
        raw_dir = (
            run_dir / "same_friction_other_actions" / "raw"
            / f"source{source_index:04d}_mu{friction_mu:.6f}"
        )
        for target_index in environment["target_indices"]:
            target_index = int(target_index)
            metadata = metadata_by_index[target_index]
            matches = sorted(raw_dir.glob(f"sample{target_index:04d}_*.mp4"))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one prediction for query {target_index}, got {matches}."
                )
            pred_path = matches[0]
            start_frame = int(metadata["start_frame"])
            frame_count = int(metadata["length"])
            video_paths = [dataset_dir / value for value in metadata["video"]]
            gt_main = read_video_frames(video_paths[0], start_frame, frame_count)
            gt_wrist = read_video_frames(video_paths[1], start_frame, frame_count)
            pred_full = read_video_frames(pred_path, 0, frame_count)
            main_width = gt_main.shape[2]
            wrist_width = gt_wrist.shape[2]
            pred_main = pred_full[:, :, :main_width]
            pred_wrist = pred_full[:, :, main_width:main_width + wrist_width]
            count = min(len(gt_main), len(gt_wrist), len(pred_full))
            evaluation_slice = slice(future_start, count)

            gt_main_future = gt_main[evaluation_slice]
            pred_main_future = pred_main[evaluation_slice]
            gt_wrist_future = gt_wrist[evaluation_slice]
            pred_wrist_future = pred_wrist[evaluation_slice]
            gt_multiview = np.concatenate(
                (gt_main_future, gt_wrist_future), axis=2
            )
            pred_multiview = np.concatenate(
                (pred_main_future, pred_wrist_future), axis=2
            )
            psnr_main, ssim_main = appearance_metrics(
                gt_main_future, pred_main_future
            )
            psnr_wrist, ssim_wrist = appearance_metrics(
                gt_wrist_future, pred_wrist_future
            )
            psnr_multiview, _ = appearance_metrics(
                gt_multiview, pred_multiview
            )

            gt_track = track_event80_block(gt_main, tracker_config)
            pred_track = track_event80_block(pred_main, tracker_config)
            diagonal = float(np.hypot(gt_main.shape[1], gt_main.shape[2]))
            distances: list[float] = []
            for frame_index in range(future_start, count):
                gt_center = gt_track.centers[frame_index]
                pred_center = pred_track.centers[frame_index]
                if not np.all(np.isfinite(gt_center)):
                    raise ValueError(
                        f"GT tracker lost query {target_index}, frame {frame_index}."
                    )
                distance = (
                    float(np.linalg.norm(gt_center - pred_center))
                    if np.all(np.isfinite(pred_center)) else diagonal
                )
                distances.append(distance)
                frame_rows.append({
                    "domain": domain,
                    "environment_id": f"mu{friction_mu:.6f}",
                    "source_support_index": source_index,
                    "query_index": target_index,
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
                    "centroid_distance_px": distance,
                    "centroid_distance_normalized": distance / diagonal,
                })

            row = {
                "sample_id": str(metadata["sample_id"]),
                "environment_id": f"mu{friction_mu:.6f}",
                "friction_mu": friction_mu,
                "source_support_index": source_index,
                "query_index": target_index,
                "domain": domain,
                "method": method,
                "seed": seed,
                "support_size": 1,
                "evaluated_frames": count - future_start,
                "centroid_mean_distance_px": float(np.mean(distances)),
                "centroid_mean_distance_normalized": float(
                    np.mean(distances) / diagonal
                ),
                "psnr_main": psnr_main,
                "psnr_wrist": psnr_wrist,
                "psnr_multiview": psnr_multiview,
                "ssim_main": ssim_main,
                "ssim_wrist": ssim_wrist,
                "ssim_multiview": float(np.mean([ssim_main, ssim_wrist])),
            }
            query_rows.append(row)
            manifest_rows.append({
                "sample_id": row["sample_id"],
                "environment_id": row["environment_id"],
                "source_support_index": source_index,
                "query_index": target_index,
                "domain": domain,
                "gt_main_video_path": str(video_paths[0]),
                "gt_wrist_video_path": str(video_paths[1]),
                "prediction_video_path": str(pred_path),
                "gt_start_frame": start_frame,
                "pred_start_frame": 0,
                "num_frames": frame_count,
                "support_excluded_from_queries": True,
            })

    environment_rows: list[dict] = []
    for source_index in sorted({row["source_support_index"] for row in query_rows}):
        rows = [
            row for row in query_rows
            if row["source_support_index"] == source_index
        ]
        environment_rows.append({
            "source_support_index": source_index,
            "environment_id": rows[0]["environment_id"],
            "friction_mu": rows[0]["friction_mu"],
            "domain": rows[0]["domain"],
            "query_count": len(rows),
            **mean_metrics(rows),
        })

    domain_rows: list[dict] = []
    for domain in ("id", "ood"):
        rows = [row for row in environment_rows if row["domain"] == domain]
        domain_rows.append({
            "domain": domain,
            "environment_count": len(rows),
            "query_count": sum(row["query_count"] for row in rows),
            **mean_metrics(rows),
        })

    write_jsonl_atomic(metric_dir / "per_query.jsonl", query_rows)
    write_jsonl_atomic(metric_dir / "per_frame_centroid.jsonl", frame_rows)
    write_jsonl_atomic(metric_dir / "per_environment.jsonl", environment_rows)
    write_json_atomic(metric_dir / "domain_summary.json", domain_rows)
    write_jsonl_atomic(protocol_dir / "manifest.jsonl", manifest_rows)
    write_json_atomic(output_root / "summary.json", {
        "protocol": "one_support_adapted_z_to_disjoint_same_environment_actions",
        "support_size": 1,
        "support_included_in_metrics": False,
        "lpips": "skipped",
        "query_count": len(query_rows),
        "environment_count": len(environment_rows),
        "domains": domain_rows,
    })
    print(
        f"[done] queries={len(query_rows)} environments={len(environment_rows)} "
        f"output={output_root}"
    )


if __name__ == "__main__":
    main()
