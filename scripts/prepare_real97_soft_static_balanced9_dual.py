#!/usr/bin/env python3
"""Freeze approved nine-environment static windows and independent latent aliases.

No videos or recorded robot targets are modified. Unassigned environments and
held-out episodes never enter train manifests or normalization statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def nested_order(count, first):
    order = [round(i * (count - 1) / (first - 1)) for i in range(first)]
    while len(order) < count:
        remaining = [i for i in range(count) if i not in order]
        order.append(max(remaining, key=lambda i: (
            min(abs(i - old) for old in order), -abs(i - (count - 1) / 2), -i)))
    return order


def prepare(contract_path):
    raw_contract = contract_path.read_bytes()
    contract = json.loads(raw_contract)
    digest = hashlib.sha256(raw_contract).hexdigest()
    source = Path(contract["dataset_root"])
    destination = ROOT / contract["output_manifest_directory"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / (destination.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            summary = json.loads((destination / "manifest_summary.json").read_text())
            if summary["data_contract_sha256"] != digest:
                raise ValueError("Frozen manifest contract differs; refusing overwrite")
            print(json.dumps(summary, indent=2), flush=True)
            return
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Fit action statistics on a compute allocation, not the login node")
        documents = {}
        metadata_hash = hashlib.sha256()
        for key in ("selection_file", "environment_split_file", "episode_split_file"):
            name = contract[key]
            content = (source / name).read_bytes()
            documents[name] = content
            metadata_hash.update(name.encode() + content)
        selection = json.loads(documents[contract["selection_file"]])
        env_split = json.loads(documents[contract["environment_split_file"]])
        splits = json.loads(documents[contract["episode_split_file"]])["environment_splits"]
        names = contract["selected_environments"]
        if (len(names) != 9 or len(set(names)) != 9 or
                selection["selected_environments"] != names or
                env_split["train_environments"] != names or env_split["status"] != "approved"):
            raise ValueError("Dataset environment selection differs from the approved balanced-nine contract")
        if selection["episode_exclusions"] != contract["quality_exclusions"]:
            raise ValueError("Episode quality exclusions changed")
        for name, ids in contract["quality_exclusions"].items():
            if env_split["quality_exclusions"][name]["lerobot_episode_indices"] != ids:
                raise ValueError(f"Conflicting quality exclusion: {name}")
        if (set(names) & set(env_split["unassigned_environments"]) or env_split["test_environments"]):
            raise ValueError("Unexpected environment assignment; do not infer an OOD split")
        physical_ids = {name: i for i, name in enumerate(names)}
        aliases = [{"virtual_group_id": 2 * physical_ids[name] + replica,
                    "virtual_environment": f"{name}__z{replica}",
                    "physical_environment": name, "physical_environment_index": physical_ids[name],
                    "physical_directory": name + "_lerobot", "latent_replica": replica,
                    "chunk_pool": "all_eligible_train_episodes_and_approved_starts"}
                   for name in names for replica in range(2)]
        physical_rows, test_rows, train_rows = [], [], []
        train_episodes, test_episodes = Counter(), Counter()
        inventory, exclusions, action_files = [], [], []
        stage_files = set()
        for name in names:
            directory = name + "_lerobot"
            metadata = source / directory / "meta"
            info_bytes = (metadata / "info.json").read_bytes()
            episode_bytes = (metadata / "episodes.jsonl").read_bytes()
            metadata_hash.update(directory.encode() + info_bytes + episode_bytes)
            documents[directory + "/meta/info.json"] = info_bytes
            documents[directory + "/meta/episodes.jsonl"] = episode_bytes
            info = json.loads(info_bytes)
            rows = [json.loads(line) for line in episode_bytes.splitlines() if line.strip()]
            expected_names = ["target_x_m", "target_y_m", "target_z_m", "target_qw", "target_qx",
                              "target_qy", "target_qz", "target_gripper_width_m"]
            if info["features"]["action"].get("names") != expected_names or info["fps"] != 20:
                raise ValueError(f"Expected recorded eight-channel target EEF at 20 Hz: {name}")
            views = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
            if len(views) != 1:
                raise ValueError(f"Expected one video view: {name}")
            train_ids = {int(i) for i in splits[name]["train_episode_indices"]}
            test_ids = {int(r["episode_index"]) for r in splits[name]["test"]}
            if train_ids & test_ids or train_ids | test_ids != {int(r["episode_index"]) for r in rows}:
                raise ValueError(f"Conflicting or incomplete episode split: {name}")
            disabled = set(contract["quality_exclusions"].get(name, []))
            for episode in rows:
                index, length = int(episode["episode_index"]), int(episode["length"])
                split = "train" if index in train_ids else "test"
                annotation = episode.get("stationary_start_annotation", {})
                starts = episode.get(contract["start_field"])
                if (not isinstance(starts, list) or len(starts) != len(set(starts)) or
                        any(type(s) is not int or s < 0 or s + 96 >= length for s in starts)):
                    raise ValueError(f"Invalid complete static window list: {name}/{index}")
                if index in disabled and starts:
                    raise ValueError(f"Disabled episode still has enabled starts: {name}/{index}")
                fmt = {"episode_index": index, "episode_chunk": index // int(info.get("chunks_size", 1000)),
                       "video_key": views[0]}
                parquet = directory + "/" + info["data_path"].format(**fmt)
                video = directory + "/" + info["video_path"].format(**fmt)
                inventory.append({"environment": name, "episode_index": index, "split": split,
                                  "eligible": bool(starts) and index not in disabled,
                                  "valid_33x3_starts": starts, "action": parquet, "video": video})
                if index in disabled or not starts:
                    exclusions.append({"environment": name, "episode_index": index,
                                       "reason": "approved_quality_exclusion" if index in disabled else "empty_approved_start_list"})
                    continue
                # An empty legal-start list is an explicit sampling exclusion,
                # including annotations marked disabled_dynamics_review. Only
                # episodes that can actually be sampled require approved status.
                if (annotation.get("version") != contract["annotation_version"] or
                        annotation.get("status") != "approved" or
                        annotation.get("model_frames") != 33 or annotation.get("stride") != 3):
                    raise ValueError(f"Missing approved static-start annotation: {name}/{index}")
                if split == "train":
                    train_episodes[name] += 1
                    action_files.append(parquet)
                    stage_files.update([parquet, video, directory + "/meta/info.json"])
                else:
                    test_episodes[name] += 1
                for start in starts:
                    row = {"sample_id": f"{directory}/ep{index:06d}/stationary_valid/{start:04d}",
                           "video": [video], "action": parquet, "start_frame": start, "end_frame": start + 96,
                           "frame_stride": 3, "length": 33, "total_frames": length,
                           "native_frame_indices": [start + 3 * k for k in range(33)], "num_history_frames": 1,
                           "sampling_kind": "stationary_valid", "stationary_start_annotation_version": annotation["version"],
                           "environment": name, "environment_index": physical_ids[name],
                           "friction_mu": float(physical_ids[name]), "action_id": index, "episode_index": index,
                           "source_episode_index": episode.get("source_episode_index", index),
                           "dataset_split": split, "source_dataset_split": split, "action_semantics": "eef_target",
                           "prompt": "Predict the video conditioned on the recorded robot targets."}
                    if split == "test":
                        test_rows.append(row)
                        continue
                    physical_rows.append(row)
                    for replica in range(2):
                        alias = aliases[2 * physical_ids[name] + replica]
                        virtual = dict(row)
                        virtual.update(source_sample_id=row["sample_id"], sample_id=row["sample_id"] + f"__z{replica}",
                                       physical_environment=name, physical_environment_index=physical_ids[name],
                                       source_friction_mu=row["friction_mu"], latent_replica=replica,
                                       virtual_environment_id=alias["virtual_group_id"],
                                       environment=alias["virtual_environment"], environment_index=alias["virtual_group_id"],
                                       friction_mu=float(alias["virtual_group_id"]))
                        train_rows.append(virtual)
        if any(train_episodes[name] < 6 for name in names):
            raise ValueError(f"Insufficient eligible episodes; never prune a group: {train_episodes}")
        actual = {"physical_episodes": len(inventory), "eligible_train_episodes": sum(train_episodes.values()),
                  "eligible_test_episodes": sum(test_episodes.values()), "physical_train_starts": len(physical_rows),
                  "physical_test_starts": len(test_rows), "virtual_train_starts": len(train_rows)}
        if actual != contract["expected_counts"]:
            raise ValueError(f"Dataset differs from the approved metadata snapshot: {actual}")
        import numpy as np
        from wan_video_action.data.recorded_targets import read_recorded_targets
        from wan_video_action.data.history_prefix import canonicalize_target_eef
        low = np.full(8, np.inf)
        high = np.full(8, -np.inf)
        reference = None
        for path in action_files:
            targets = read_recorded_targets(source / path, "eef_target")
            if reference is None:
                reference = np.asarray(targets[0, 3:7], dtype=np.float32)
                reference = reference / np.linalg.norm(reference)
            canonical = canonicalize_target_eef(targets, reference)
            low = np.minimum(low, canonical.min(axis=0))
            high = np.maximum(high, canonical.max(axis=0))
        stats = {"eef_target": {"min": low.tolist(), "max": high.tolist(),
                 "quaternion_reference_wxyz": reference.tolist(),
                 "source": "all native frames of 146 selected, eligible training episodes only",
                 "source_field": "action", "layout": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width"],
                 "quaternion_policy": "unit norm; first sign against fixed training reference; causal temporal sign continuity",
                 "constant_channels": "zero", "packing": "8 target channels first, then 6 zeros"}}
        temporary = destination.with_name(destination.name + ".partial-" + str(os.getpid()))
        temporary.mkdir()
        hashes = {}
        for split, records in (("train", train_rows), ("physical_train", physical_rows), ("test", test_rows), ("episode_split", inventory)):
            hasher = hashlib.sha256()
            with (temporary / (split + ".jsonl")).open("xb") as handle:
                for row in records:
                    encoded = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                    handle.write(encoded)
                    hasher.update(encoded)
            hashes[split] = hasher.hexdigest()
        for name, content in documents.items():
            path = temporary / "source_metadata" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (temporary / "data_contract.json").write_bytes(raw_contract)
        (temporary / "stage_files.txt").write_text("\n".join(sorted(stage_files)) + "\n")
        write_json(temporary / "action_stats.json", stats)
        write_json(temporary / "excluded_episodes.json", exclusions)
        write_json(temporary / "latent_aliases.json", {"records": aliases,
                   "independent_initialization": contract["latent_init"], "data_partition_between_replicas": False,
                   "heldout_evaluation": "test rows retain physical IDs; select a replica explicitly using this mapping"})
        order = nested_order(18, 5)
        rounds, cursor = [], 0
        for i, count in enumerate(contract["curriculum_group_counts"]):
            ids = order[cursor:cursor + count]
            cursor += count
            rounds.append({"round": i + 1, "start_step": 301 + 1000 * i, "end_step": 1300 + 1000 * i,
                           "new_virtual_groups": ids, "new_groups": [aliases[j] for j in ids],
                           "active_virtual_groups": order[:cursor],
                           "phases": ["new_C_200_lr003", "all_active_C_200_lr003", "model_200_lr1e-5",
                                      "all_active_C_200_lr003", "model_200_lr1e-5"]})
        write_json(temporary / "dual_latent_curriculum.json", {"initial_model_only_steps": [1, 300],
                   "initial_model_warmup_steps": 100, "group_order": order, "rounds": rounds,
                   "iterative_start_step": 4301, "last_step": 5500,
                   "small_pool_sampling": contract["small_pool_sampling"]})
        summary = {"task": "soft", "version": contract["version"], "dataset_base_path": str(source),
                   "data_contract_sha256": digest, "source_metadata_sha256": metadata_hash.hexdigest(),
                   "train_manifest_sha256": hashes["train"], "manifest_hashes": hashes,
                   "environment_ids": {a["virtual_environment"]: a["virtual_group_id"] for a in aliases},
                   "physical_environment_ids": {name + "_lerobot": i for name, i in physical_ids.items()},
                   "selected_physical_environments": names, "unassigned_environments": env_split["unassigned_environments"],
                   "heldout_environments": [], "physical_environment_count": 9, "virtual_environment_count": 18,
                   "latent_replicas_per_physical_environment": 2,
                   "train_episodes_per_environment": {a["virtual_environment"]: train_episodes[a["physical_environment"]] for a in aliases},
                   "physical_train_episodes_per_environment": dict(train_episodes),
                   "episode_counts": {"train": sum(train_episodes.values()), "test": sum(test_episodes.values())},
                   "virtual_training_episode_count": 2 * sum(train_episodes.values()),
                   "chunk_counts": {"train": len(train_rows), "physical_train": len(physical_rows), "test": len(test_rows)},
                   "excluded_episodes": exclusions, "source_dataset_cache_identity": {
                       "task": "soft", "version": contract["version"] + "_physical", "train_manifest_sha256": hashes["physical_train"]},
                   "action_type": "eef_target", "action_statistics_scope": "selected eligible training episodes only",
                   "model_size_wh": [320, 160], "model_frames": 33, "frame_stride": 3, "num_history_frames": 1,
                   "batch_per_gpu": [5, 6], "curriculum_groups": [5, 5, 5, 3], "counts": actual}
        write_json(temporary / "manifest_summary.json", summary)
        os.replace(temporary, destination)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/train/real97_soft_static_balanced9_dual_data_contract_20260913_v1.json")
    prepare(parser.parse_args().contract.resolve())
