#!/usr/bin/env python3
"""Compute train-only normalization statistics for the 1-D EEF swing angle."""

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
    return parser.parse_args()


def load_roll_degrees(path: Path) -> np.ndarray:
    states = np.asarray(
        pq.read_table(path, columns=["observation.eef_state"])[
            "observation.eef_state"
        ].to_pylist(),
        dtype=np.float64,
    )
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"Expected observation.eef_state [T,7], got {states.shape} in {path}")
    quaternions = states[:, 3:7]
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12
    )
    for frame_index in range(1, len(quaternions)):
        if np.dot(quaternions[frame_index - 1], quaternions[frame_index]) < 0:
            quaternions[frame_index] *= -1
    qw, qx, qy, qz = (quaternions[:, index] for index in range(4))
    roll_x = np.unwrap(
        np.arctan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
    )
    roll_x += 2.0 * np.pi * np.rint(
        (np.pi - np.median(roll_x)) / (2.0 * np.pi)
    )
    return np.rad2deg(roll_x)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.metadata_path.read_text().splitlines()
        if line.strip()
    ]
    action_paths = sorted({str(row["action"]) for row in rows})
    roll = np.concatenate(
        [load_roll_degrees(args.dataset_root / relative) for relative in action_paths]
    )
    values = np.zeros((roll.shape[0], 14), dtype=np.float64)
    values[:, 0] = roll

    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "p01": np.percentile(values, 1.0, axis=0),
        "p99": np.percentile(values, 99.0, axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
    }
    # The loader normalizes all 14 channels to [-1, 1]. Symmetric dummy bounds
    # ensure the thirteen padding channels remain exactly zero after normalization.
    for key in ("min", "p01"):
        stats[key][1:] = -1.0
    for key in ("max", "p99", "std"):
        stats[key][1:] = 1.0
    stats["mean"][1:] = 0.0

    payload = {
        "eef_swing_angle": {
            "shape": [14],
            **{key: value.tolist() for key, value in stats.items()},
        },
        "metadata": {
            "dataset_root": str(args.dataset_root.resolve()),
            "source_metadata": str(args.metadata_path),
            "statistics_scope": "train episodes only",
            "source_column": "observation.eef_state=[x,y,z,qw,qx,qy,qz]",
            "selected_scalar": "unwrapped Euler XYZ roll_x in degrees",
            "layout": ["normalized_roll_x"] + ["zero_padding"] * 13,
            "unique_episodes": len(action_paths),
            "total_frames": int(roll.shape[0]),
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output_path}: episodes={len(action_paths)} "
        f"frames={roll.shape[0]} p01={stats['p01'][0]:.6f} "
        f"p99={stats['p99'][0]:.6f}"
    )


if __name__ == "__main__":
    main()
