#!/usr/bin/env python3
"""Freeze dual-latent evaluation references without changing previous experiments."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def prepare(config_path, task):
    raw = config_path.read_bytes()
    config = json.loads(raw)
    setting = config["tasks"][task]
    source = ROOT / setting["source_prepared"]
    run = ROOT / setting["training_run"]
    prepared = ROOT / setting["prepared"]
    fingerprint = hashlib.sha256(raw).hexdigest()
    if (prepared / "prepared_complete.json").is_file():
        previous = read(prepared / "prepared_complete.json")
        if previous["config_sha256"] != fingerprint:
            raise RuntimeError("Existing prepared reference belongs to another configuration")
        print(json.dumps({"task": task, "prepared": str(prepared), "reused": True}), flush=True)
        return
    prepared.mkdir(parents=True, exist_ok=False)
    reference = prepared / "reference"
    reference.mkdir()
    old_plan = read(source / "plan.json")
    plan = copy.deepcopy(old_plan)
    summary = read(run / "input_manifest/manifest_summary.json")
    table_path = run / f"step-{setting['table_step']}.context_table.json"
    table = read(table_path)
    records = {int(row["friction_mu"]): row for row in table["records"]}
    count = int(setting["expected_latents"])
    physical_count = int(summary["physical_environment_count"])
    if len(records) != count or count != 2 * physical_count:
        raise RuntimeError("The final table must contain exactly two latents per physical environment")
    effective = []
    aliases = []
    seen = set()
    for environment in old_plan["environments"]:
        name = environment["environment"]
        physical = int(environment["environment_index"])
        if summary["physical_environment_ids"][name + "_lerobot"] != physical:
            raise RuntimeError(f"Physical environment identity changed: {name}")
        for replica in (0, 1):
            alias = name + f"_lerobot__z{replica}"
            original_id = int(summary["environment_ids"][alias])
            # The legacy reference evaluator looks up physical IDs for Stage1.
            # Retain the other replicas under noncolliding IDs so its mean uses
            # every original row exactly once, not only the selected Stage1 rows.
            selected = replica == config["stage1_replica"]
            evaluation_id = physical if selected else physical_count + physical
            item = copy.deepcopy(records[original_id])
            item.update(friction_mu=evaluation_id, source_virtual_group_id=original_id,
                        environment=name, physical_environment_index=physical,
                        replica=replica, stage1_selected=selected)
            effective.append(item)
            aliases.append({k: item[k] for k in ["friction_mu", "source_virtual_group_id", "environment",
                                                "physical_environment_index", "replica", "stage1_selected"]})
            if original_id in seen:
                raise RuntimeError("An original latent would be counted twice")
            seen.add(original_id)
    if seen != set(records):
        raise RuntimeError("The evaluation did not retain every trained latent")
    effective.sort(key=lambda row: row["friction_mu"])
    remapped = {**table, "records": effective,
                "evaluation_only_id_remapping": True,
                "original_table": "full_training_context_table.json"}
    write(reference / "context_table.json", remapped)
    shutil.copyfile(table_path, reference / "full_training_context_table.json")
    write(reference / "latent_aliases.json", aliases)
    os.link(run / f"step-{setting['model_step']}.safetensors", reference / "model.safetensors")
    shutil.copyfile(run / "submitted_config.yaml", reference / "training_config.yaml")
    shutil.copyfile(ROOT / setting["action_stats"], reference / "action_stats.json")
    manifest_hashes = {}
    for name in ["query.jsonl", "support.jsonl", "extension_action_templates.jsonl", "stage_files.txt"]:
        content = (source / name).read_bytes()
        (prepared / name).write_bytes(content)
        manifest_hashes[name] = hashlib.sha256(content).hexdigest()
    plan["setting"].update(training_run=setting["training_run"], training_config=setting["training_config"],
                           model_file=f"step-{setting['model_step']}.safetensors",
                           table_file=f"step-{setting['table_step']}.context_table.json",
                           model_step=setting["model_step"], table_step=setting["table_step"])
    plan["protocol"]["ttt"].update(
        stage2_inner_lr_schedule=config["stage2"]["inner_lr_schedule"],
        stage2_inner_steps=config["stage2"]["inner_steps"],
        stage2_inner_grad_clip=0, stage2_context_reg_weight=0,
        ttt_context_fp32=True, ttt_support_gradient_accumulation=True,
        ttt_adapt_scope="context", spatial_loss_mode="none",
        ttt_disable_context_clamp=True,
    )
    plan["protocol"].setdefault("tasks", {})[task] = copy.deepcopy(plan["setting"])
    plan["experiment"].setdefault("tasks", {})[task] = copy.deepcopy(plan["setting"])
    plan.update(prepared=str(prepared), source_reference=str(source),
                stage1_reference=f"physical_environment_replica_z{config['stage1_replica']}",
                stage2_initial="mean_of_all_trained_latents",
                stage2_mean_latent_count=count, effective_context_bounds=None,
                original_source_outputs_modified=False,
                source_manifest_sha256=manifest_hashes,
                latent_alias_mapping="reference/latent_aliases.json")
    historical = plan.pop("support_override", None)
    plan["historical_support_selection_provenance"] = historical
    write(prepared / "source_plan.json", old_plan)
    write(prepared / "plan.json", plan)
    write(prepared / "experiment.json", plan["experiment"])
    support = [json.loads(line) for line in (source / "support.jsonl").read_text().splitlines() if line.strip()]
    active_support = {env["environment"]: [{"level": support[i]["action_level"],
                                            "episode_index": support[i]["episode_index"],
                                            "start_frame": support[i]["start_frame"]}
                                           for i in env["support_indices"]]
                      for env in plan["environments"]}
    record = {"task": task, "config_sha256": fingerprint,
              "model_step": setting["model_step"], "table_step": setting["table_step"],
              "stage1_replica": config["stage1_replica"], "stage2_mean_latents": count,
              "query_count": plan["query_count"], "active_support": active_support,
              "manifest_sha256": manifest_hashes,
              "source_context_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
              "model_reference_is_hard_link": True}
    write(prepared / "prepared_complete.json", record)
    print(json.dumps({"prepared": str(prepared), **record}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["door", "ball"], required=True)
    args = parser.parse_args()
    prepare(args.config, args.task)
