#!/usr/bin/env python3
"""Audit cached sim_rgb_v1 states for identity jumps and exit discontinuities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", action="append", type=Path, required=True)
    parser.add_argument("--jump-threshold-px", type=float, default=64.0)
    parser.add_argument("--angle-jump-threshold-deg", type=float, default=35.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_root(root: Path, jump_threshold: float, angle_threshold: float) -> dict:
    anomalies = []
    file_count = 0
    centroid_tracks = 0
    angle_tracks = 0
    light_tracks = 0
    for path in sorted(root.glob("**/*.npz")):
        file_count += 1
        state = np.load(path)
        if "centroids" in state.files:
            centroids = np.asarray(state["centroids"], dtype=np.float64)
            centroid_tracks += centroids.shape[1]
            for object_index in range(centroids.shape[1]):
                track = centroids[:, object_index]
                finite = np.all(np.isfinite(track), axis=1)
                steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
                for frame in np.flatnonzero(finite[1:] & finite[:-1] & (steps > jump_threshold)) + 1:
                    anomalies.append({
                        "kind": "identity_jump",
                        "path": str(path),
                        "object_index": object_index,
                        "frame": int(frame),
                        "distance_px": float(steps[frame - 1]),
                    })
                key = "event_offscreen"
                if key in state.files:
                    offscreen = np.asarray(state[key][:, object_index], dtype=bool)
                    indices = np.flatnonzero(offscreen)
                    if len(indices):
                        start = int(indices[0])
                        reference_index = max(0, start - 1)
                        reference = track[reference_index]
                        held = track[start:]
                        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(held)):
                            anomalies.append({
                                "kind": "offscreen_nonfinite",
                                "path": str(path),
                                "object_index": object_index,
                                "frame": start,
                            })
                        else:
                            deviation = np.linalg.norm(held - reference, axis=1)
                            maximum = float(np.max(deviation)) if len(deviation) else 0.0
                            if maximum > 1e-6:
                                anomalies.append({
                                    "kind": "offscreen_not_held",
                                    "path": str(path),
                                    "object_index": object_index,
                                    "frame": start,
                                    "maximum_deviation_px": maximum,
                                })
        if "angles_rad" in state.files:
            angles = np.asarray(state["angles_rad"], dtype=np.float64)
            angle_tracks += angles.shape[1]
            delta = np.abs((np.diff(angles, axis=0) + np.pi / 2.0) % np.pi - np.pi / 2.0)
            for frame, object_index in np.argwhere(
                np.isfinite(delta) & (delta > np.deg2rad(angle_threshold))
            ):
                anomalies.append({
                    "kind": "angle_jump",
                    "path": str(path),
                    "object_index": int(object_index),
                    "frame": int(frame + 1),
                    "degrees": float(np.rad2deg(delta[frame, object_index])),
                })
        if "light_score" in state.files:
            light_tracks += int(np.asarray(state["light_score"]).shape[1])

    return {
        "state_root": str(root),
        "file_count": file_count,
        "centroid_track_count": centroid_tracks,
        "angle_track_count": angle_tracks,
        "light_track_count": light_tracks,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def main() -> None:
    args = parse_args()
    reports = [
        audit_root(root, args.jump_threshold_px, args.angle_jump_threshold_deg)
        for root in args.state_root
    ]
    payload = {
        "protocol": {
            "extractor": "sim_rgb_v1",
            "jump_threshold_px": args.jump_threshold_px,
            "angle_jump_threshold_deg": args.angle_jump_threshold_deg,
            "offscreen_rule": "hold_last_observed_position",
        },
        "state_root_count": len(reports),
        "anomaly_count": sum(report["anomaly_count"] for report in reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "state_roots": len(reports),
        "anomalies": payload["anomaly_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
