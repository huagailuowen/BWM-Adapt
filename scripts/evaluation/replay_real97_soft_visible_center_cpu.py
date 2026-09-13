#!/usr/bin/env python3
"""Recompute visible-object centers and overlays from cached dense detections."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from track_real97_soft_dino_cpu import ROOT, aggregate, audit_episode, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm allocation required for video replay")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    override = json.loads(args.config.read_text())
    if override["center_definition"] != "visible_box":
        raise ValueError("This replay is specifically for visible-box centers")
    snapshot = json.loads((args.source_run / "shard00/config_snapshot.json").read_text())
    config = dict(snapshot["tracking"], center_definition=override["center_definition"])
    filter_config = snapshot["filtering"]
    samples = json.loads((args.source_run / "shard00/sample_manifest.json").read_text())
    output = args.source_run / "visible_center"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_snapshot.json", dict(tracking=config, filtering=filter_config,
               source_run=str(args.source_run), override=override, detector_execution="reuse_saved_proposals_no_model"))
    write_json(output / "sample_manifest.json", samples)
    missing = []
    completed = 0
    for sample in samples:
        name = f'{sample["environment"]}_ep{sample["episode_index"]:06d}'
        source = args.source_run / "episodes" / name
        destination = output / "episodes" / name
        if (destination / "summary.json").is_file():
            completed += 1
            continue
        if not (source / "summary.json").is_file():
            missing.append(name)
            continue
        source_summary = json.loads((source / "summary.json").read_text())
        rows = [json.loads(line) for line in (source / "tracks.jsonl").read_text().splitlines() if line.strip()]
        cursor = 0

        def cached_detect(image):
            nonlocal cursor
            if cursor >= len(rows) or rows[cursor]["frame"] != cursor:
                raise RuntimeError("Cached detections do not align with decoded video frames")
            proposals = rows[cursor]["all_proposals"]
            cursor += 1
            return proposals

        started = time.monotonic()
        result = audit_episode(sample, destination, cached_detect, config, filter_config)
        if cursor != len(rows):
            raise RuntimeError("Decoded video ended before all cached detections were replayed")
        result.update(detector_execution="reuse_saved_proposals_no_model",
                      source_summary=str(source / "summary.json"),
                      replay_elapsed_seconds=time.monotonic()-started,
                      mean_original_detector_seconds=source_summary["mean_detector_seconds"])
        write_json(destination / "summary.json", result)
        completed += 1
    aggregate(output, samples, center_definition="visible_box")
    print("[visible_center_replay] " + json.dumps(dict(completed=completed, expected=len(samples),
          unfinished_sources=missing, allow_partial=args.allow_partial, output=str(output))), flush=True)
    if missing and not args.allow_partial:
        raise RuntimeError(f"{len(missing)} source episodes unfinished; no missing episodes were silently discarded")


if __name__ == "__main__":
    main()
