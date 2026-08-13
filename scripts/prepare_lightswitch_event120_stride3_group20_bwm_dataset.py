#!/usr/bin/env python3
"""Convert event-centered LightSwitch metadata to 120-frame-span windows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


NUM_FRAMES = 41
FRAME_STRIDE = 3
RAW_SPAN = (NUM_FRAMES - 1) * FRAME_STRIDE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--source-metadata-path", type=Path, required=True)
    parser.add_argument("--source-group-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def aligned_window(event_frame: int, frame_count: int) -> tuple[int, int, int]:
    max_start = frame_count - 1 - RAW_SPAN
    if max_start < 0:
        raise ValueError(f"Episode length {frame_count} is shorter than raw span {RAW_SPAN + 1}")
    residue = event_frame % FRAME_STRIDE
    first = residue
    if first > max_start:
        raise ValueError(f"No stride-aligned window for event={event_frame}, length={frame_count}")
    last = max_start - ((max_start - residue) % FRAME_STRIDE)
    desired = event_frame - RAW_SPAN // 2
    step = round((desired - first) / FRAME_STRIDE)
    start = min(max(first + step * FRAME_STRIDE, first), last)
    end = start + RAW_SPAN
    event_index = (event_frame - start) // FRAME_STRIDE
    if start + event_index * FRAME_STRIDE != event_frame:
        raise AssertionError("Event frame is not on the sampled frame grid")
    return start, end, event_index


def action_stats(dataset_root: Path, episode_indices: list[int]) -> dict:
    arrays = []
    for episode_index in episode_indices:
        path = dataset_root / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
        values = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float64)
        arrays.append(values[::FRAME_STRIDE])
    merged = np.concatenate(arrays, axis=0)
    return {
        "mean": merged.mean(axis=0).tolist(),
        "std": np.maximum(merged.std(axis=0), 1e-6).tolist(),
        "min": merged.min(axis=0).tolist(),
        "max": merged.max(axis=0).tolist(),
        "count": int(merged.shape[0]),
        "frame_stride": FRAME_STRIDE,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_base_path.resolve()
    source_rows = read_jsonl(args.source_metadata_path)
    episodes = {
        int(row["episode_index"]): int(row["length"])
        for row in read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    report = []
    for source in source_rows:
        row = dict(source)
        episode_index = int(row["episode_index"])
        event_frame = int(row["event_frame"])
        start, end, event_window_index = aligned_window(
            event_frame, episodes[episode_index]
        )
        group_id = int(row["context_group_id"])
        press_number = int(row["press_number"])
        row.update(
            {
                "start_frame": start,
                "end_frame": end,
                "length": NUM_FRAMES,
                "frame_stride": FRAME_STRIDE,
                "raw_frame_span": RAW_SPAN,
                "target_fps": 10,
                "event_window_index": event_window_index,
                "grouping_name": "event_centered120_stride3_random_episode_group20",
                "sample_id": (
                    f"lightswitch:event120s3:g{group_id:02d}:ep{episode_index:06d}:"
                    f"press{press_number}:raw{start:04d}-{end:04d}"
                ),
            }
        )
        rows.append(row)
        report.append(
            {
                "episode_index": episode_index,
                "causal_class": row["causal_class"],
                "context_group_id": group_id,
                "press_number": press_number,
                "button_color": row["button_color"],
                "lamp_toggled": row["lamp_toggled"],
                "lamp_before": row["lamp_before"],
                "lamp_after": row["lamp_after"],
                "event_frame": event_frame,
                "start_frame": start,
                "end_frame": end,
                "event_window_index": event_window_index,
                "frames_before_event": event_frame - start,
                "frames_after_event": end - event_frame,
            }
        )

    write_jsonl(output_dir / "event120_stride3_group20_train.jsonl", rows)
    write_jsonl(output_dir / "event120_stride3_report.jsonl", report)
    with (output_dir / "event120_stride3_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    stats = action_stats(dataset_root, sorted(episodes))
    (output_dir / "action_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n"
    )
    group_manifest = json.loads(args.source_group_manifest.read_text())
    group_manifest["derived_from_metadata"] = str(args.source_metadata_path)
    group_manifest["window"] = {
        "num_frames": NUM_FRAMES,
        "frame_stride": FRAME_STRIDE,
        "raw_frame_span": RAW_SPAN,
        "source_fps": 30,
        "target_fps": 10,
        "alignment": "event frame lies exactly on sampled stride-3 grid",
    }
    (output_dir / "group20_manifest.json").write_text(
        json.dumps(group_manifest, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "source_metadata": str(args.source_metadata_path),
        "sample_count": len(rows),
        "episode_count": len(episodes),
        "event_type_counts": dict(Counter(row["event_type"] for row in rows)),
        "causal_class_counts": dict(Counter(row["causal_class"] for row in rows)),
        "group_counts": {
            str(key): value
            for key, value in sorted(Counter(int(row["context_group_id"]) for row in rows).items())
        },
        "window": group_manifest["window"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
