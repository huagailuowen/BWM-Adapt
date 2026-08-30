#!/usr/bin/env python3
"""Compute train-only normalization statistics for EEF pose conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("eef_observed_state", "eef_state_action"),
        default="eef_observed_state",
    )
    return parser.parse_args()


def canonicalize_eef(values: np.ndarray, path: Path) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Expected observation.eef_state [T,7], got {values.shape} in {path}")
    result = values.astype(np.float64, copy=True)
    quaternions = result[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"Zero-norm EEF quaternion in {path}")
    quaternions /= norms
    dominant = int(np.argmax(np.abs(quaternions[0])))
    if quaternions[0, dominant] < 0:
        quaternions[0] *= -1
    for frame_index in range(1, len(quaternions)):
        if np.dot(quaternions[frame_index - 1], quaternions[frame_index]) < 0:
            quaternions[frame_index] *= -1
    result[:, 3:7] = quaternions
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite EEF state in {path}")
    return result


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.metadata_path.read_text().splitlines()
        if line.strip()
    ]
    action_paths = sorted({str(row["action"]) for row in rows})
    arrays = []
    total_frames = 0
    for relative_path in action_paths:
        path = args.dataset_root / relative_path
        values = np.asarray(
            pq.read_table(path, columns=["observation.eef_state"])["observation.eef_state"].to_pylist(),
            dtype=np.float64,
        )
        values = canonicalize_eef(values, path)
        if args.representation == "eef_state_action":
            next_values = np.concatenate([values[1:], values[-1:]], axis=0)
            values = np.concatenate([values, next_values], axis=1)
        arrays.append(values)
        total_frames += values.shape[0]

    all_values = np.concatenate(arrays, axis=0)
    stats = {
        "shape": [int(all_values.shape[1])],
        "min": all_values.min(axis=0).tolist(),
        "max": all_values.max(axis=0).tolist(),
        "p01": np.percentile(all_values, 1.0, axis=0).tolist(),
        "p99": np.percentile(all_values, 99.0, axis=0).tolist(),
        "mean": all_values.mean(axis=0).tolist(),
        "std": all_values.std(axis=0).tolist(),
    }
    if args.representation == "eef_state_action":
        layout = [
            "x_t", "y_t", "z_t", "qw_t", "qx_t", "qy_t", "qz_t",
            "x_t+1", "y_t+1", "z_t+1", "qw_t+1", "qx_t+1", "qy_t+1", "qz_t+1",
        ]
        next_pose_policy = (
            "read the next sampled observation.eef_state row; repeat the final pose "
            "only at the physical end of an episode"
        )
    else:
        layout = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        next_pose_policy = None

    payload = {
        args.representation: stats,
        "metadata": {
            "source_metadata": str(args.metadata_path),
            "dataset_root": str(args.dataset_root),
            "unique_episodes": len(action_paths),
            "total_frames": total_frames,
            "layout": layout,
            "quaternion_processing": "unit-normalize, first dominant component positive, temporal sign continuity",
            "statistics_scope": "train episodes only",
            "next_pose_policy": next_pose_policy,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"[eef_stats] representation={args.representation} "
        f"episodes={len(action_paths)} frames={total_frames} "
        f"output={args.output_path}"
    )


if __name__ == "__main__":
    main()
