#!/usr/bin/env python3
"""Jitter the event position in the existing 33-frame LightSwitch dataset.

The source metadata fixes every causal event at window index 16.  This tool
keeps exactly the same 1,600 events, episode-to-group assignments, actions,
and 33-frame/30-FPS representation, but deterministically samples one event
index per event from an inclusive configurable range.  It therefore changes
temporal alignment without changing event or environment sampling weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


NUM_FRAMES = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-index-min", type=int, default=11)
    parser.add_argument("--event-index-max", type=int, default=22)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def deterministic_index(sample_id: str, low: int, high: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return low + int.from_bytes(digest[:8], "big") % (high - low + 1)


def main() -> None:
    args = parse_args()
    if not 0 <= args.event_index_min <= args.event_index_max < NUM_FRAMES:
        raise ValueError(
            f"Expected event indices inside [0, {NUM_FRAMES - 1}], got "
            f"[{args.event_index_min}, {args.event_index_max}]."
        )

    dataset_root = args.dataset_base_path.resolve()
    source_dir = args.source_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(source_dir / "event_centered_group20_train.jsonl")

    episode_lengths: dict[int, int] = {}
    output_rows: list[dict] = []
    for source in source_rows:
        episode_index = int(source["episode_index"])
        if episode_index not in episode_lengths:
            episode_lengths[episode_index] = pq.ParquetFile(
                dataset_root / source["action"]
            ).metadata.num_rows
        frame_count = episode_lengths[episode_index]
        event_frame = int(source["event_frame"])
        target_index = deterministic_index(
            str(source["sample_id"]),
            args.event_index_min,
            args.event_index_max,
            args.seed,
        )
        start_frame = event_frame - target_index
        end_frame = start_frame + NUM_FRAMES - 1
        if start_frame < 0 or end_frame >= frame_count:
            feasible = [
                index
                for index in range(args.event_index_min, args.event_index_max + 1)
                if 0 <= event_frame - index
                and event_frame - index + NUM_FRAMES <= frame_count
            ]
            if not feasible:
                raise ValueError(
                    f"No valid event index in [{args.event_index_min}, "
                    f"{args.event_index_max}] for episode {episode_index}, "
                    f"event frame {event_frame}, length {frame_count}."
                )
            target_index = feasible[
                deterministic_index(
                    str(source["sample_id"]), 0, len(feasible) - 1, args.seed + 1
                )
            ]
            start_frame = event_frame - target_index
            end_frame = start_frame + NUM_FRAMES - 1

        row = dict(source)
        row.update(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "event_window_index": target_index,
                "event_window_index_min": args.event_index_min,
                "event_window_index_max": args.event_index_max,
                "requested_core_start_frame": start_frame,
                "requested_core_end_frame": end_frame,
                "grouping_name": "event33_jitter11to22_random_episode_group20",
                "window_alignment_seed": args.seed,
                "sample_id": (
                    f"lightswitch:event33jitter11to22:"
                    f"g{int(source['context_group_id']):02d}:"
                    f"ep{episode_index:06d}:press{int(source['press_number'])}:"
                    f"idx{target_index:02d}:raw{start_frame:04d}-{end_frame:04d}"
                ),
            }
        )
        output_rows.append(row)

    write_jsonl(output_dir / "event_jitter_group20_train.jsonl", output_rows)
    write_jsonl(output_dir / "event_jitter_report.jsonl", output_rows)
    csv_fields = [
        "episode_index",
        "causal_class",
        "context_group_id",
        "press_number",
        "button_color",
        "event_type",
        "lamp_toggled",
        "event_frame",
        "event_window_index",
        "start_frame",
        "end_frame",
    ]
    with (output_dir / "event_jitter_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    for filename in ("action_stats.json", "group20_manifest.json"):
        shutil.copy2(source_dir / filename, output_dir / filename)

    summary = {
        "dataset_base_path": str(dataset_root),
        "source_metadata": str(source_dir / "event_centered_group20_train.jsonl"),
        "sample_count": len(output_rows),
        "episode_count": len(episode_lengths),
        "seed": args.seed,
        "window": {
            "num_frames": NUM_FRAMES,
            "frame_stride": 1,
            "source_fps": 30,
            "target_fps": 30,
            "event_window_index_min": args.event_index_min,
            "event_window_index_max": args.event_index_max,
            "sampling": "one deterministic uniform draw per source event",
        },
        "event_window_index_counts": dict(
            sorted(Counter(row["event_window_index"] for row in output_rows).items())
        ),
        "sample_counts_by_causal_class": dict(
            sorted(Counter(row["causal_class"] for row in output_rows).items())
        ),
        "sample_counts_by_group": dict(
            sorted(Counter(str(row["context_group_id"]) for row in output_rows).items())
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
