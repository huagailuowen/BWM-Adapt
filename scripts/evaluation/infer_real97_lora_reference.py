#!/usr/bin/env python3
"""Opt-in Real97 LoRA TTA with unchanged Standard weights and frozen query plans."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.evaluation.infer_real97_ball_door_baselines import (
    frame_array, json_write, key, read_video, rows, snapshot, titled, video_write,
)
from scripts.evaluation.infer_real97_standard_reference import copy_cached
from scripts.infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
from scripts.infer_stage2_ttt import _freeze_pipe
from scripts.methods.infer_lora_tta_event80 import adapt_lora, install_lora
from scripts.methods.infer_lora_tta_transfer import _load_adapter, _save_adapter
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["door", "ball"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("LoRA adaptation and video work require a one-GPU compute allocation")
    protocol = json.loads(cli.config.read_text())
    setting, lora = protocol["tasks"][cli.task], protocol["lora"]
    prepared = ROOT / setting["prepared"]
    training = ROOT / setting["standard_training"]
    standard_output = ROOT / setting["standard_inference"]
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = output / "input_manifest"
    for name in ("plan.json", "query.jsonl", "support.jsonl", "extension_action_templates.jsonl", "stage_files.txt"):
        snapshot(prepared / name, frozen / name)
    snapshot(training / "submitted_config.yaml", frozen / "training_config.yaml")
    snapshot(training / "input_manifest/action_stats.json", frozen / "action_stats.json")
    snapshot(cli.config, frozen / "evaluation_config.json")
    plan = json.loads((frozen / "plan.json").read_text())
    queries, supports, templates = [rows(frozen / name) for name in
                                   ("query.jsonl", "support.jsonl", "extension_action_templates.jsonl")]
    config = yaml.safe_load((frozen / "training_config.yaml").read_text())["training"]
    if len(queries) != setting["expected_queries"]:
        raise RuntimeError("Unexpected factual query count")
    if (config["num_frames"], config["height"], config["width"]) != tuple(setting["geometry"]):
        raise RuntimeError("Training/inference video geometry differs")
    if config.get("physical_context_enabled") or config.get("physical_adapter_enabled"):
        raise RuntimeError("LoRA must start from Standard, not an Ours context/adapter model")
    if any(row["action_semantics"] != config["action_type"] for row in queries + supports + templates):
        raise RuntimeError("Recorded action semantics differ from Standard training")
    for environment in plan["environments"]:
        selected = [supports[i] for i in environment["support_indices"]]
        if not selected or any(row["dataset_split"] != "train" or row["environment"] != environment["environment"] for row in selected):
            raise RuntimeError("Supports must be from this environment's training split")
        if {row["episode_index"] for row in selected} & {queries[i]["episode_index"] for i in environment["query_indices"]}:
            raise RuntimeError("Support/query episode leakage")

    source_ckpt = training / setting["checkpoint"]
    stat = source_ckpt.stat()
    identity = {"source": str(source_ckpt.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "checkpoint_step": 5500}
    identity_file = frozen / "model_identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise RuntimeError("The source Standard checkpoint changed")
    else:
        json_write(identity_file, identity)
    protected = frozen / "model.safetensors"
    if not protected.exists():
        os.link(source_ckpt, protected)
    elif not os.path.samefile(source_ckpt, protected):
        raise RuntimeError("Protected checkpoint link points to another model")

    cache = Path("/tmp") / os.environ["USER"] / "bwm_shared_cache"
    wan = cache / "Wan2.2-TI2V-5B"
    copy_cached(ROOT / "models/Wan2.2-TI2V-5B", wan, directory=True)
    data_tag = hashlib.sha256(str(config["dataset_base_path"]).encode() + (frozen / "stage_files.txt").read_bytes()).hexdigest()[:16]
    local_data = cache / "datasets_real" / (cli.task + "_eval_" + data_tag)
    copy_cached(Path(config["dataset_base_path"]), local_data, directory=True, files=frozen / "stage_files.txt")
    ckpt_tag = hashlib.sha256(f"{source_ckpt.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    local_ckpt = cache / "real97_baseline_eval" / ckpt_tag / "model.safetensors"
    copy_cached(source_ckpt, local_ckpt)
    config.update(
        dataset_base_path=str(local_data), dataset_metadata_path=str(frozen / "query.jsonl"),
        action_stat_path=str(frozen / "action_stats.json"), model_paths=str(wan),
        ckpt_path=str(local_ckpt), output_path=str(output / "raw/lora"),
        num_inference_steps=protocol["num_inference_steps"], cfg_scale=1.0,
        seed=protocol["seed"], frame_stride=1, num_history_frames=1, action_dim=14,
        video_light_augmentation_enabled=False, spatial_loss_mode="none", fps=20, quality=8,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
    )
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"inference": config}, sort_keys=False))
    inference_parser = add_general_config(argparse.ArgumentParser())
    if "--frame_stride" not in inference_parser._option_string_actions:
        inference_parser.add_argument("--frame_stride", type=int, default=1)
    args = inference_parser.parse_args(["--config", str(runtime)])
    args = merge_yaml_and_args(str(runtime), inference_parser, args)
    args.stage2_fixed_timestep_index = None
    args.lora_steps = int(lora["steps"])
    args.lora_learning_rate = float(lora["learning_rate"])
    args.lora_gradient_clip = float(lora["gradient_clip_norm"])
    args.lora_alpha = float(lora["alpha"])
    def dataset_for(name):
        local_args = copy.copy(args)
        local_args.dataset_metadata_path = str(frozen / name)
        return build_infer_dataset(local_args)
    dataset, support_dataset, template_dataset = [dataset_for(name) for name in
                                                  ("query.jsonl", "support.jsonl", "extension_action_templates.jsonl")]
    set_global_seed(int(protocol["seed"]))
    pipe = build_pipeline(args)
    _freeze_pipe(pipe)
    adapters = install_lora(pipe.dit, targets=set(lora["target_modules"]), rank=int(lora["rank"]),
                            alpha=float(lora["alpha"]), seed=int(lora["adapter_seed"]))
    count = sum(adapter.lora_a.numel() + adapter.lora_b.numel() for _, adapter in adapters)
    query_sha = hashlib.sha256((frozen / "query.jsonl").read_bytes()).hexdigest()
    support_sha = hashlib.sha256((frozen / "support.jsonl").read_bytes()).hexdigest()
    json_write(output / "provenance.json", {
        "method": "lora_tta", "task": cli.task, "source": identity, "source_model": "standard_pooled_wm",
        "lora": lora, "adapter_parameter_count": count, "adapted_linear_modules": [name for name, _ in adapters],
        "query_manifest_sha256": query_sha, "support_manifest_sha256": support_sha,
        "reset_per_environment": True, "query_updates_lora": False,
        "support_loss": "mean of all selected support flow-matching losses at each AdamW update",
        "base_model_and_action_encoder_frozen": True, "roi": False, "inference_augmentation": False,
        "source_results_modified": False, "formal_metric_approved": False,
    })
    print(f"[lora] modules={len(adapters)} trainable_parameters={count}", flush=True)

    def generate(sample, index, destination, seed):
        args.seed = int(seed)
        set_global_seed(args.seed)
        rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
        rollout["output_path"] = str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            _run_autoregressive(pipe=pipe, sample=rollout, args=args)
        video = read_video(destination)
        if len(video) != args.num_frames:
            raise RuntimeError("Generated video has an unexpected frame count")
        video_write(destination, video, 20)
        return video

    completed, extension_count, grid_order = 0, 0, {}
    for environment in plan["environments"]:
        name = environment["environment"]
        env_seed = int(lora["adapter_seed"]) + int(environment["environment_index"]) * 1009
        expected = {**identity, "environment": name, "support_indices": environment["support_indices"],
                    "support_manifest_sha256": support_sha, "lora": lora, "support_noise_seed": env_seed}
        adapter_file = output / "adapters" / (name + ".pt")
        if adapter_file.exists():
            losses = _load_adapter(adapter_file, adapters, expected)
            print("[reuse_adapter]", name, flush=True)
        else:
            set_global_seed(env_seed)
            selected_supports = [support_dataset[i] for i in environment["support_indices"]]
            losses = adapt_lora(pipe, selected_supports, adapters, args)
            if len(losses) != int(lora["steps"]) or not np.isfinite(losses).all():
                raise FloatingPointError("LoRA support adaptation did not finish with finite losses")
            if not all(torch.isfinite(parameter).all().item() for _, adapter in adapters
                       for parameter in (adapter.lora_a, adapter.lora_b)):
                raise FloatingPointError("LoRA weights became nonfinite")
            _save_adapter(adapter_file, adapters, {**expected, "inner_losses": losses})
            del selected_supports
        for _, adapter in adapters:
            adapter.lora_a.grad = None
            adapter.lora_b.grad = None
        pipe.eval()
        json_write(output / "adaptation" / (name + ".json"), {
            **expected, "inner_losses": losses, "adapter_state": str(adapter_file),
            "support_levels": [supports[i]["action_level"] for i in environment["support_indices"]],
            "query_state_policy": "read_only", "parameter_count": count,
        })
        for index in environment["query_indices"]:
            row = queries[index]
            sample_key = key(index, row)
            prediction_path = output / "raw/lora" / (sample_key + ".mp4")
            gt_path = output / "raw/gt" / (sample_key + ".mp4")
            comparison = output / "comparisons" / (sample_key + ".mp4")
            marker = output / "completed" / (sample_key + ".json")
            if marker.exists() and prediction_path.exists() and gt_path.exists() and comparison.exists():
                completed += 1
                continue
            standard_record = json.loads((standard_output / "completed" / (sample_key + ".json")).read_text())
            for field in ("sample_id", "environment", "episode_index", "dataset_split", "action_level",
                          "action", "video", "native_frame_indices", "evaluation_frame_indices"):
                if row.get(field) != standard_record["query"].get(field):
                    raise RuntimeError(f"Standard reference query mismatch: {sample_key}/{field}")
            sample = dataset[index]
            gt = frame_array(sample["video"])
            prediction = generate(sample, index, prediction_path, int(protocol["seed"]) + index)
            standard = read_video(standard_record["prediction"])
            if gt.shape != prediction.shape or gt.shape != standard.shape:
                raise RuntimeError(f"GT/Standard/LoRA frame geometry differs: {sample_key}")
            video_write(gt_path, gt, 20)
            video_write(comparison, np.concatenate([
                titled(gt, f"GT L{row['action_level']} {row['dataset_split']}"),
                titled(standard, f"Standard L{row['action_level']}"),
                titled(prediction, f"LoRA L{row['action_level']}"),
            ], axis=1), 20)
            json_write(marker, {"query": row, "prediction": str(prediction_path), "gt": str(gt_path),
                                "standard_prediction": standard_record["prediction"], "checkpoint_step": 5500,
                                "adapter_state": str(adapter_file), "paired_gt": True, "query_updates_lora": False})
            completed += 1
            json_write(output / "progress.json", {"completed_queries": completed, "total_queries": len(queries),
                                                 "current_environment": name, "completed_extensions": extension_count})
            print(f"[query_done] {sample_key} {completed}/{len(queries)}", flush=True)
        grid_order[name] = {}
        for split in ("train", "test"):
            indices = sorted([i for i in environment["query_indices"] if queries[i]["dataset_split"] == split],
                             key=lambda i: (int(queries[i]["action_level"]), int(queries[i]["episode_index"]), i))
            grid_order[name][split] = indices
            if indices:
                columns = [read_video(output / "comparisons" / (key(i, queries[i]) + ".mp4")) for i in indices]
                video_write(output / "grids" / f"{name}_gt_standard_lora_{split}_levels.mp4",
                            np.concatenate(columns, axis=2), 20)
        anchor_index = int(environment["extension_anchor_query_index"])
        extension_columns = []
        for template_index in sorted(environment["extension_template_indices"], key=lambda i: int(templates[i]["action_level"])):
            row = templates[template_index]
            if row["environment"] != name or row["dataset_split"] != "train":
                raise RuntimeError("Action template must come from this environment's train split")
            level = int(row["action_level"])
            destination = output / "raw/action_sweeps" / f"{name}_testinitial_L{level:02d}_lora.mp4"
            marker = output / "completed" / f"extension_{name}_L{level:02d}.json"
            if destination.exists() and marker.exists():
                prediction = read_video(destination)
            else:
                sample = dataset[anchor_index]
                sample["action"] = template_dataset[template_index]["action"]
                seed = int(protocol["seed"]) + anchor_index
                prediction = generate(sample, anchor_index, destination, seed)
                json_write(marker, {"anchor_query_index": anchor_index, "action_template": row,
                                    "paired_gt": False, "seed": seed, "adapter_state": str(adapter_file)})
            extension_columns.append(titled(prediction, f"LoRA L{level} fixed initial"))
            extension_count += 1
        if extension_columns:
            video_write(output / "action_sweeps" / f"{name}_lora_levels.mp4",
                        np.concatenate(extension_columns, axis=2), 20)
        json_write(output / "progress.json", {"completed_queries": completed, "total_queries": len(queries),
                                             "current_environment": name, "completed_extensions": extension_count})
    json_write(output / "grid_order.json", grid_order)
    json_write(output / "inference_complete.json", {
        "task": cli.task, "method": "lora_tta", "checkpoint_step": 5500,
        "queries": completed, "environments": len(plan["environments"]), "extensions": extension_count,
        "lora": lora, "adapter_parameter_count": count, "query_manifest_sha256": query_sha,
        "support_manifest_sha256": support_sha, "query_updates_lora": False,
        "grid_order": "ascending_numeric_levels_separate_train_test",
        "metric_status": "scoring_pending", "source_results_modified": False,
    })


if __name__ == "__main__":
    main()
