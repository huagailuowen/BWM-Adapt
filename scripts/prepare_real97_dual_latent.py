#!/usr/bin/env python3
"""Expand train-only metadata into two independent latent groups per environment.

No videos, actions, windows, normalizers, or held-out episode assignments change.
The legacy friction_mu field is only a categorical virtual-group identifier here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def nested_order(count, first):
    """The existing trainer's nested-uniform order for these 4/5-group curricula."""
    selected = [round(i * (count - 1) / (first - 1)) for i in range(first)]
    while len(selected) < count:
        remaining = [i for i in range(count) if i not in selected]
        selected.append(max(remaining, key=lambda i: (
            min(abs(i - old) for old in selected),
            -abs(i - (count - 1) / 2), -i,
        )))
    return selected


def prepare(task):
    source_config = ROOT / f"configs/train/train_real97_{task}_target_c32_2gpu_20260909_v2.yaml"
    config = yaml.safe_load(source_config.read_text())["training"]
    source = Path(config["dataset_metadata_path"]).parent
    source_summary = json.loads((source / "manifest_summary.json").read_text())
    version = "20260912_dual_latent_lr003_v1"
    destination = ROOT / f"data/real97_{task}_{version}"
    target_config = ROOT / f"configs/train/train_real97_{task}_target_c32_dual_latent_lr003_2gpu_20260912_v1.yaml"
    if destination.exists() or target_config.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment manifest/config.")
    physical_names = {int(index): name for name, index in source_summary["environment_ids"].items()}
    count = len(physical_names)
    expected = 10 if task == "door" else 8
    if sorted(physical_names) != list(range(expected)):
        raise ValueError(f"Unexpected source environment IDs for {task}.")
    with (source / "train.jsonl").open() as handle:
        original_rows = [json.loads(line) for line in handle if line.strip()]
    if any(row["dataset_split"] != "train" for row in original_rows):
        raise ValueError("Non-training data in the source training manifest.")
    first = int(config["grouped_context_curriculum_initial_groups"])
    aliases = []
    for physical_id in range(count):
        for replica in range(2):
            aliases.append({
                "virtual_group_id": 2 * physical_id + replica,
                "virtual_environment": f"{physical_names[physical_id]}__z{replica}",
                "physical_environment_index": physical_id,
                "physical_environment": physical_names[physical_id],
                "latent_replica": replica,
                "chunk_pool": "all_train_chunks_of_physical_environment",
            })
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("train.jsonl", "manifest_summary.json"))
    train_hash = hashlib.sha256()
    physical_row_counts = Counter()
    with (destination / "train.jsonl").open("xb") as handle:
        for row in original_rows:
            physical_id = int(row["environment_index"])
            physical_row_counts[physical_id] += 1
            for replica in range(2):
                alias = aliases[2 * physical_id + replica]
                item = dict(row)
                item.update(
                    source_sample_id=row["sample_id"],
                    sample_id=f"{row['sample_id']}__z{replica}",
                    physical_environment=row["environment"],
                    physical_environment_index=physical_id,
                    source_friction_mu=row["friction_mu"],
                    latent_replica=replica,
                    virtual_environment_id=alias["virtual_group_id"],
                    environment_index=alias["virtual_group_id"],
                    environment=alias["virtual_environment"],
                    friction_mu=float(alias["virtual_group_id"]),
                )
                encoded = (json.dumps(item, separators=(",", ":")) + "\n").encode()
                handle.write(encoded)
                train_hash.update(encoded)
    summary = dict(source_summary)
    summary.update(
        version=version, train_manifest_sha256=train_hash.hexdigest(),
        source_manifest=str(source), source_chunk_counts=source_summary["chunk_counts"],
        source_dataset_cache_identity={key: source_summary[key] for key in ("task", "version", "train_manifest_sha256")},
        environment_ids={a["virtual_environment"]: a["virtual_group_id"] for a in aliases},
        physical_environment_ids=source_summary["environment_ids"],
        train_episodes_per_environment={
            a["virtual_environment"]: source_summary["train_episodes_per_environment"][a["physical_environment"]]
            for a in aliases
        },
        chunk_counts={**source_summary["chunk_counts"], "train": 2 * len(original_rows)},
        physical_train_chunk_count=len(original_rows),
        physical_environment_count=count, virtual_environment_count=2 * count,
        latent_replicas_per_physical_environment=2, curriculum_groups=[first] * 4,
        physical_episode_counts=source_summary["episode_counts"],
        virtual_training_episode_count=2 * int(source_summary["episode_counts"]["train"]),
        heldout_manifests="unchanged_physical_environment_ids_use_alias_mapping_for_evaluation",
        group_key_semantics="friction_mu_is_virtual_group_id_not_physical_friction",
    )
    write_json(destination / "manifest_summary.json", summary)
    order = nested_order(2 * count, first)
    rounds = []
    for index in range(4):
        new_ids = order[index * first:(index + 1) * first]
        rounds.append({
            "round": index + 1, "start_step": 301 + 1000 * index,
            "end_step": 1300 + 1000 * index,
            "new_virtual_groups": new_ids, "new_groups": [aliases[i] for i in new_ids],
            "active_virtual_groups": order[:(index + 1) * first],
            "phases": [
                {"phase": "new_context", "steps": 200, "c_lr": 0.03, "model_lr": 0.0},
                {"phase": "all_context", "steps": 200, "c_lr": 0.03, "model_lr": 0.0},
                {"phase": "model", "steps": 200, "c_lr": 0.0, "model_lr": config["learning_rate"]},
                {"phase": "all_context", "steps": 200, "c_lr": 0.03, "model_lr": 0.0},
                {"phase": "model", "steps": 200, "c_lr": 0.0, "model_lr": config["learning_rate"]},
            ],
        })
    write_json(destination / "latent_aliases.json", {
        "task": task, "records": aliases,
        "independent_initialization": "each_coordinate_iid_uniform_minus1_plus1",
        "independent_trainable_table_rows": True,
        "same_physical_environment_replicas_share_all_train_data": True,
        "data_partition_between_replicas": False,
        "physical_train_chunks_by_environment": dict(physical_row_counts),
        "source_manifest": str(source),
    })
    write_json(destination / "dual_latent_curriculum.json", {
        "initial_model_only_steps": [1, 300], "initial_model_lr_warmup_steps": 100,
        "group_order": order, "group_order_rule": "legacy_nested_uniform_on_virtual_ids",
        "rounds": rounds, "iterative_start_step": 4301,
        "iterative_phases": ["all_context_200_steps_lr003", "model_200_steps_lr1e-5"],
        "last_step": 5500, "initial_c_dim": 32,
        "batch_per_gpu": "3 virtual groups x 8 episodes/chunks",
        "physical_environment_replicas_can_cooccur_in_batch": True,
    })
    config.update(
        dataset_metadata_path=str(destination / "train.jsonl"),
        action_stat_path=str(destination / "action_stats.json"),
        grouped_context_init_mode="uniform", grouped_context_init_min=-1.0,
        grouped_context_init_max=1.0, physical_context_dim=32,
        grouped_context_new_context_lr=0.03, grouped_context_lr=0.03,
        grouped_context_curriculum_total_groups=2 * count,
        grouped_context_structured_updates=5500,
        output_path=str(ROOT / f"outputs/real97_{task}_target_c32_dual_latent_lr003_20260912_v1"),
    )
    with target_config.open("x") as handle:
        yaml.safe_dump({"training": config}, handle, sort_keys=False)
    print(json.dumps({"task": task, "config": str(target_config),
                      "physical_environments": count, "latent_groups": 2 * count,
                      "train_metadata_rows": 2 * len(original_rows), "iterative_start": 4301}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("door", "ball", "both"), default="both")
    args = parser.parse_args()
    for task in (("door", "ball") if args.task == "both" else (args.task,)):
        prepare(task)
