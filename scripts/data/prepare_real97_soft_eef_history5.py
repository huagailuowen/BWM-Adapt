#!/usr/bin/env python3
"""CPU-only preparation: preserve episode split, enumerate causal anchors."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wan_video_action.data.history_prefix import canonicalize_target_eef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Prepare all episodes on a Slurm CPU node, not the login node")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest {args.output}")
    temporary = args.output.with_name(args.output.name + ".preparing_" + os.environ["SLURM_JOB_ID"])
    temporary.mkdir(parents=True, exist_ok=False)
    old_summary = json.loads((args.source_manifest / "manifest_summary.json").read_text())
    episodes = {}
    with (args.source_manifest / "train.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dataset_split"] != "train":
                raise ValueError("Non-training episode in source training manifest")
            episodes.setdefault((int(row["environment_index"]), int(row["episode_index"])), row)
    arrays, reference = [], None
    records = []
    counts = {}
    stage_files = set()
    for key, template in sorted(episodes.items()):
        native = np.asarray(pq.read_table(
            args.dataset / template["action"], columns=["action"],
        ).to_pydict()["action"], dtype=np.float32)
        if reference is None:
            reference = native[0, 3:7].astype(np.float64)
            reference /= np.linalg.norm(reference)
        targets = canonicalize_target_eef(native, reference)
        arrays.append(targets)
        length = len(targets)
        if length != int(template["total_frames"]):
            raise ValueError(f"Episode length changed since approved split: {template['action']}")
        stage_files.add(template["action"])
        videos = template["video"] if isinstance(template["video"], list) else [template["video"]]
        stage_files.update(videos)
        count = 0
        for anchor in range(max(0, length - 85) + 1):
            logical_start = anchor - 12
            padding = max(0, (-logical_start + 2) // 3)
            row = dict(template)
            row.update(
                sample_id=f"{template['environment']}/ep{key[1]:06d}/history5/anchor{anchor:04d}",
                start_frame=logical_start + padding * 3,
                end_frame=min(anchor + 84, length - 1),
                length=33, frame_stride=3, sampling_kind="general",
                action_semantics="eef_target", history_anchor_frame=anchor,
                history_padding_frames=padding, num_history_frames=5,
                native_frame_indices=np.clip(logical_start + np.arange(33) * 3, 0, length - 1).tolist(),
            )
            records.append(row)
            count += 1
        counts[f"{template['environment']}/ep{key[1]:06d}"] = count
    if len(episodes) != int(old_summary["episode_counts"]["train"]):
        raise ValueError("Training episode membership changed")
    values = np.concatenate(arrays, axis=0)
    stats = {"eef_target": {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "source": "all native frames of approved ID training episodes only",
        "source_field": "action", "layout": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width"],
        "quaternion_reference_wxyz": reference.tolist(),
        "quaternion_policy": "unit norm; first sign against fixed training reference; causal temporal sign continuity",
        "constant_channels": "zero", "packing": "8 target channels first, then 6 zeros",
    }}
    (temporary / "action_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    with (temporary / "train.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    digest = hashlib.sha256((temporary / "train.jsonl").read_bytes()).hexdigest()
    summary = dict(old_summary)
    summary.update(
        version="20260910_eef_history5_v1", train_manifest_sha256=digest,
        action_type="eef_target", action_source_fields=["action"],
        model_frames=33, history_frames=5, predicted_frames=28,
        anchor_native_span=85, frame_stride=3,
        padding="clip(anchor-12+3*k,0,episode_length-1), identical video/action indices",
        history_condition_noise=False, chunk_counts={"train": len(records)},
        source_manifest=str(args.source_manifest),
        train_episode_membership="unchanged; all 243 ID train episodes retained; test/OOD not used",
    )
    (temporary / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (temporary / "episode_window_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    (temporary / "stage_files.txt").write_text("\n".join(sorted(stage_files)) + "\n")
    shutil.copy2(args.dataset / "TRAIN_TEST_SPLIT.json", temporary / "source_TRAIN_TEST_SPLIT.json")
    shutil.copy2(args.dataset / "ENVIRONMENT_HOLDOUT_SPLIT.json", temporary / "source_ENVIRONMENT_HOLDOUT_SPLIT.json")
    os.rename(temporary, args.output)
    print(json.dumps({"prepared": str(args.output), "train_episodes": len(episodes),
                      "windows": len(records), "history_frames": 5, "future_frames": 28}), flush=True)


if __name__ == "__main__":
    main()
