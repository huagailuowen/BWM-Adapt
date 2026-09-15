#!/usr/bin/env python3
"""Add missing Standard queries without changing the historical inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import infer_real97_standard_reference as standard

original_prepare = standard.prepare


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(row):
    return row["environment"], row["dataset_split"], int(row["episode_index"])


def link(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination) and digest(source) != digest(destination):
            raise RuntimeError(f"Existing artifact differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(own, protocol):
    source = standard.resolve(protocol["matched_ours"])
    old = standard.resolve(protocol["reuse_standard"])
    prepared = standard.resolve(protocol["tasks"]["soft"]["prepared"])
    if prepared != own.output / "prepared":
        raise RuntimeError("Prepared manifest must belong to this independent output")
    for name in ("plan.json", "query.jsonl", "support.jsonl"):
        standard.snapshot(source / name, prepared / name)
    standard.snapshot(source / "input_manifest/stage_files.txt", prepared / "stage_files.txt")
    plan, rows, runtime, flat = original_prepare(own, protocol)
    training = standard.resolve(protocol["tasks"]["soft"]["training_run"])
    training_flat = standard.yaml.safe_load((training / "submitted_config.yaml").read_text())["training"]
    seen = set()
    for row in standard.read_jsonl(Path(training_flat["dataset_metadata_path"])):
        seen.add((row["environment"], int(row["episode_index"])))
    seen_envs = {env for env, _ in seen}
    old_rows = standard.read_jsonl(old / "query.jsonl")
    old_index = {identity(row): (i, row) for i, row in enumerate(old_rows)}
    if len(old_index) != len(old_rows):
        raise RuntimeError("Duplicate historical Standard query identity")
    reused, pending, cohort = [], [], []
    old_data = Path(training_flat["dataset_base_path"])
    new_data = Path(plan["source_dataset"])
    for index, row in enumerate(rows):
        row["prediction_padding_frames"] = 28 - len(row["evaluation_frame_indices"])
        key = f'q{index:04d}_{row["environment"]}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'
        cohort.append(dict(index=index, key=key, environment=row["environment"], split=row["dataset_split"],
                           standard_environment_seen=row["environment"] in seen_envs,
                           standard_episode_seen=(row["environment"], int(row["episode_index"])) in seen,
                           ours_environment_seen=True))
        if (own.output / "completed" / (key + ".json")).exists():
            continue
        match = old_index.get(identity(row))
        if match is None:
            pending.append(key)
            continue
        previous_index, previous = match
        fields = ("video", "action", "total_frames", "native_frame_indices",
                  "evaluation_frame_indices", "observed_native_frame_indices", "frame_stride", "length")
        if any(row.get(field) != previous.get(field) for field in fields):
            pending.append(key)
            continue
        previous_key = f'q{previous_index:04d}_{previous["environment"]}_{previous["dataset_split"]}_ep{previous["episode_index"]:06d}'
        marker_path = old / "completed" / (previous_key + ".json")
        if not marker_path.is_file():
            pending.append(key)
            continue
        marker = read(marker_path)
        if marker["row"] != previous or marker["observed_frames"] != 5 or not marker["paired_gt"]:
            raise RuntimeError("Historical Standard completion identity mismatch")
        fingerprints = {}
        for relative in list(row["video"]) + [row["action"]]:
            before, now = old_data / relative, new_data / relative
            a, b = digest(before), digest(now)
            if a != b:
                raise RuntimeError(f"Dataset assets changed; cannot reuse {relative}")
            fingerprints[relative] = a
        paths = {kind: own.output / "raw" / kind / (key + ".mp4") for kind in ("gt", "standard")}
        for kind, path in paths.items():
            link(marker["paths"][kind], path)
        standard.write_json(own.output / "completed" / (key + ".json"), dict(
            index=index, row=row, paths={k: str(v) for k, v in paths.items()},
            observed_frames=5, paired_gt=True, reused_from=str(marker_path),
            source_asset_sha256=fingerprints, seed=int(protocol["seed"]) + previous_index))
        reused.append(key)
    standard.write_jsonl(own.output / "query.jsonl", rows)
    standard.write_json(own.output / "matched_cohort.json", {
        "cases": cohort, "standard_training_environments": sorted(seen_envs),
        "reused_this_attempt": reused, "pending_this_attempt": pending,
        "paired_noise_realizations": False,
        "comparison_note": "Match physical timestamps, not raw frame positions. Standard has five identical initial frames; Ours has one.",
        "common_native_timestamps": list(range(0, 85, 3)),
        "unseen_environment_note": "Report shared-training environments separately from Ours-ID / Standard-OOD environments.",
    })
    print("[matched_plan] " + json.dumps({"reused": len(reused), "pending": len(pending)}), flush=True)
    return plan, rows, runtime, flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    own = parser.parse_args()
    standard.prepare = prepare
    sys.argv = ["standard", "--task", "soft", "--config", str(own.config),
                "--output", str(own.output), "--episode-start-history-padding"]
    standard.main()


if __name__ == "__main__":
    main()
