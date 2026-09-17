#!/usr/bin/env python3
"""Create immutable metadata only; no video decoding or source-data writes."""
import fcntl
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/stick_balance_2026-09-16")
DEST = ROOT / "data/real916_stick_lift15_driftstable_v1"


def load(path):
    return json.loads(path.read_text())


def build():
    policy_path = SOURCE / "CHUNK_SAMPLING_POLICY.json"
    stats_path = SOURCE / "action_normalization_shared.json"
    policy = load(policy_path)
    normalization = load(stats_path)
    split_paths = sorted(SOURCE.glob("*_lerobot/meta/train_test_split.json"))
    if len(split_paths) != 9:
        raise ValueError("Expected exactly nine explicit environment split files")
    source_paths = [policy_path, stats_path] + split_paths
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(SOURCE)).encode())
        digest.update(path.read_bytes())
    source_sha = digest.hexdigest()
    if DEST.exists():
        summary = load(DEST / "manifest_summary.json")
        if summary["source_metadata_sha256"] != source_sha:
            raise RuntimeError("Dataset metadata changed: use a new version, never overwrite a manifest")
        print(json.dumps(summary), flush=True)
        return

    windows = {(r["environment"], int(r["episode_index"])): r for r in policy["episode_windows"]}
    if len(windows) != 466:
        raise ValueError("Expected 466 distinct annotated episodes")
    env_ids = {}
    rows = {"train": [], "test": []}
    inventory, staged = [], set()
    counts, drift_counts = Counter(), Counter()
    for env_id, split_path in enumerate(split_paths):
        env_root = split_path.parent.parent
        env = env_root.name.removesuffix("_lerobot")
        env_ids[env] = env_id
        split = load(split_path)
        info = load(env_root / "meta/info.json")
        if float(info["fps"]) != 20 or info["features"]["action"]["shape"] != [8]:
            raise ValueError("Expected 20Hz target EEF data")
        if info["features"]["action"]["names"][:7] != [
            "target_x_m", "target_y_m", "target_z_m",
            "target_qw", "target_qx", "target_qy", "target_qz"
        ]:
            raise ValueError("Action must be the explicitly exported tool-tip target")
        train = set(map(int, split["train_episode_indices"]))
        test = set(map(int, split["test_episode_indices"]))
        if train & test or len(train | test) != info["total_episodes"] or len(test) != 5:
            raise ValueError("Invalid frozen episode split")
        if len(train) < 8:
            raise ValueError("Insufficient distinct episodes for a 3x8 batch")
        for ep in sorted(train | test):
            w = windows[(env, ep)]
            subset = "train" if ep in train else "test"
            if subset != w["split"]:
                raise ValueError("Policy and frozen train/test list disagree")
            n = int(w["frame_count"])
            drift = bool(w["initial_slide_override"])
            if drift:
                e = int(w["stable_start_frame"])
                bounds = {"lift": (e, min(e + 10, n - 1))}
                if bounds["lift"] != (int(w["start_low"]), int(w["start_high"])):
                    raise ValueError("Stable candidate interval differs from official policy")
                drift_counts[subset] += 1
            else:
                a = int(w["lift_start_frame"])
                bounds = {"general": (0, max(0, n - 97)), "lift": (max(0, a - 15), a)}
            fmt = dict(episode_index=ep, episode_chunk=ep // int(info.get("chunks_size", 1000)),
                       video_key="observation.images.image")
            video = str(Path(env_root.name) / info["video_path"].format(**fmt))
            action = str(Path(env_root.name) / info["data_path"].format(**fmt))
            for kind, (lo, hi) in bounds.items():
                if not 0 <= lo <= hi < n:
                    raise ValueError(f"Invalid range, no silent pruning: {env}/{ep}: {bounds}")
                for start in range(lo, hi + 1):
                    valid = min(33, (n - 1 - start) // 3 + 1)
                    if valid < 5:
                        raise ValueError(f"No complete future VAE block: {env}/{ep}/{start}")
                    rows[subset].append({
                        "sample_id": f"{env}/ep{ep:06d}/{kind}/{start:04d}",
                        "video": [video], "action": action,
                        "start_frame": start, "end_frame": min(n - 1, start + 96),
                        "frame_stride": 3, "length": 33, "total_frames": n,
                        "valid_video_frames": valid, "valid_future_latent_frames": (valid - 1) // 4,
                        "sampling_kind": kind, "initial_slide_override": drift,
                        "environment": env, "environment_index": env_id, "friction_mu": float(env_id),
                        "action_id": ep, "episode_index": ep,
                        "source_episode_index": w["source_episode_index"],
                        "dataset_split": subset, "action_semantics": "eef_target",
                    })
            inventory.append({"environment": env, "episode_index": ep, "split": subset,
                              "initial_slide_override": drift, "frame_count": n, "start_ranges": bounds})
            counts[subset] += 1
            if subset == "train":
                staged.update([video, action, str(Path(env_root.name) / "meta/info.json")])
    if dict(counts) != {"test": 45, "train": 421}:
        raise ValueError(f"Unexpected episode counts: {dict(counts)}")
    if sum(drift_counts.values()) != 113 or drift_counts["train"] != 106:
        raise ValueError("Unexpected drift cohort")
    if int(normalization["train_episode_count"]) != 421:
        raise ValueError("Normalization must use this frozen training split")

    temporary = Path(tempfile.mkdtemp(prefix=DEST.name + ".partial-", dir=DEST.parent))
    try:
        for subset, records in rows.items():
            with (temporary / f"{subset}.jsonl").open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        with (temporary / "episode_split.jsonl").open("w") as handle:
            for record in inventory:
                handle.write(json.dumps(record) + "\n")
        stats = {"eef_target": {**normalization["statistics"], "source": "dataset shared train-only statistics",
                                "constant_channels": "zero", "packing": "8 targets followed by 6 zeros"}}
        (temporary / "action_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        (temporary / "stage_files.txt").write_text("\n".join(sorted(staged)) + "\n")
        originals = temporary / "source_metadata"
        originals.mkdir()
        for path in source_paths:
            target = originals / path.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        summary = {
            "task": "stick916", "version": "lift15_driftstable_v1",
            "dataset_base_path": str(SOURCE), "source_metadata_sha256": source_sha,
            "train_manifest_sha256": hashlib.sha256((temporary / "train.jsonl").read_bytes()).hexdigest(),
            "environment_ids": env_ids, "episode_counts": dict(counts), "drift_episode_counts": dict(drift_counts),
            "chunk_counts": {s: len(r) for s, r in rows.items()},
            "model_size_wh": [256, 192], "model_frames": 33, "frame_stride": 3,
            "action_type": "eef_target", "batch_per_gpu": [3, 8], "curriculum_groups": [5, 4],
            "drift_sampling": "stable Lift only; no General",
            "other_sampling": "30% General, 70% Lift [max(0,a-15),a]",
            "padding_loss": "omit condition and any future VAE temporal block containing padded frames",
            "light_augmentation_probability": 0.7, "light_gradient_abs_max": 0.08,
        }
        (temporary / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.rename(temporary, DEST)
        print(json.dumps(summary), flush=True)
    except BaseException:
        # Leave a distinct partial directory for diagnosis; never remove published data.
        raise


if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with (DEST.parent / (DEST.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build()

