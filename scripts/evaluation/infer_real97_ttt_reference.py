#!/usr/bin/env python3
"""Real97 paired-query TTT inference with same-timestep multi-support replay."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import yaml

from scripts.evaluation.infer_real97_ball_door_baselines import (
    frame_array, json_write, key, read_video, rows, snapshot, titled, video_write,
)
from scripts.evaluation.infer_real97_standard_reference import copy_cached
from scripts.infer import build_infer_dataset, prepare_sample_for_rollout, _run_autoregressive
from scripts.methods.infer_ttt_kvb_event80 import build_training_faithful_model
from scripts.methods.infer_ttt_kvb_oneminute_event80 import (
    SameTimestepSupportQueryModelFn, _prepare_support_model_inputs, _update_counts,
)
from scripts.methods.train_ttt_kvb_event80 import add_ttt_kvb_config
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


class MultiSupportReplay:
    """Extend the existing K=1 replay to an ordered support prefix, without GT queries."""

    def __init__(self, pipe, controller, original, support_replays):
        self.pipe = pipe
        self.controller = controller
        self.original = original
        self.support_replays = support_replays
        self.trace = []

    def __call__(self, *args, **query_inputs):
        timestep = query_inputs.get("timestep")
        if not isinstance(timestep, torch.Tensor):
            raise RuntimeError("Query call is missing its diffusion timestep")
        self.controller.reset(batch_size=1)
        try:
            support_counts = []
            for replay in self.support_replays:
                with self.controller.causal_scan(differentiable=False):
                    self.original(**replay._support_call_inputs(timestep))
                support_counts.append(_update_counts(self.controller.write_statistics()))
            with self.controller.causal_scan(differentiable=False):
                prediction = self.original(*args, **query_inputs)
            complete = _update_counts(self.controller.write_statistics())
            before = support_counts[-1] if support_counts else {}
            self.trace.append({
                "denoising_call": len(self.trace),
                "timestep": float(timestep.detach().float().reshape(-1)[0].cpu()),
                "cumulative_updates_after_each_support": support_counts,
                "query_updates_per_layer": {name: count - before.get(name, 0) for name, count in complete.items()},
                "state_rebuilt_from_learned_initial": True,
            })
            return prediction
        finally:
            self.controller.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["door", "ball"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TTT inference requires exactly one GPU in a compute allocation")
    os.chdir(ROOT)
    protocol = json.loads(cli.config.read_text())
    setting = protocol["tasks"][cli.task]
    prepared = ROOT / setting["prepared"]
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = output / "input_manifest"
    for name in ["plan.json", "query.jsonl", "support.jsonl", "extension_action_templates.jsonl", "stage_files.txt", "evaluation_config.json"]:
        snapshot(prepared / name, frozen / name)
    snapshot(prepared / "reference/training_config.yaml", frozen / "training_config.yaml")
    snapshot(prepared / "reference/action_stats.json", frozen / "action_stats.json")
    snapshot(cli.config, frozen / "launch_config.json")
    plan = json.loads((frozen / "plan.json").read_text())
    queries, supports, templates = [rows(frozen / name) for name in ["query.jsonl", "support.jsonl", "extension_action_templates.jsonl"]]
    config = yaml.safe_load((frozen / "training_config.yaml").read_text())["training"]
    if config["ttt_protocol"] != "oneminute_write_then_predict":
        raise RuntimeError("Checkpoint training protocol is not write-then-predict")
    if (config["num_frames"], config["height"], config["width"]) != tuple(setting["geometry"]):
        raise RuntimeError("Training/inference video geometry mismatch")
    if len(queries) != setting["expected_queries"]:
        raise RuntimeError("The frozen query count changed")
    if any(row["action_semantics"] != config["action_type"] for row in queries + supports + templates):
        raise RuntimeError("Training/inference action semantics mismatch")
    for env in plan["environments"]:
        support = [supports[i] for i in env["support_indices"]]
        if not support or any(row["dataset_split"] != "train" or row["environment"] != env["environment"] for row in support):
            raise RuntimeError("Invalid training-only support prefix")
        if {row["episode_index"] for row in support} & {queries[i]["episode_index"] for i in env["query_indices"]}:
            raise RuntimeError("Support/query episode overlap")

    cache = Path("/tmp") / os.environ["USER"] / "bwm_shared_cache"
    cache.mkdir(parents=True, exist_ok=True)
    wan = cache / "Wan2.2-TI2V-5B"
    copy_cached(ROOT / "models/Wan2.2-TI2V-5B", wan, directory=True)
    base = cache / "ckpt/BLM/step-12000.safetensors"
    copy_cached(ROOT / "ckpt/BLM/step-12000.safetensors", base)
    data_tag = hashlib.sha256(str(config["dataset_base_path"]).encode() + (frozen / "stage_files.txt").read_bytes()).hexdigest()[:16]
    local_data = cache / "datasets_real" / (cli.task + "_eval_" + data_tag)
    copy_cached(Path(config["dataset_base_path"]), local_data, directory=True, files=frozen / "stage_files.txt")
    checkpoint = prepared / "reference/model.safetensors"
    stat = checkpoint.stat()
    checkpoint_tag = hashlib.sha256(f"{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    local_checkpoint = cache / "real97_ttt_eval" / checkpoint_tag / "model.safetensors"
    copy_cached(checkpoint, local_checkpoint)
    config.update(dataset_base_path=str(local_data), dataset_metadata_path=str(frozen / "query.jsonl"),
                  action_stat_path=str(frozen / "action_stats.json"), model_paths=str(wan),
                  ckpt_path=str(base), output_path=str(output / "raw/ttt"),
                  num_inference_steps=protocol["num_inference_steps"], cfg_scale=1.0,
                  seed=protocol["seed"], frame_stride=1, num_history_frames=1,
                  video_light_augmentation_enabled=False, spatial_loss_mode="none",
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                  fps=20, quality=8)
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"inference": config}, sort_keys=False))
    inference_parser = add_ttt_kvb_config(add_general_config(argparse.ArgumentParser()))
    if "--frame_stride" not in inference_parser._option_string_actions:
        inference_parser.add_argument("--frame_stride", type=int, default=1)
    args = inference_parser.parse_args(["--config", str(runtime)])
    args = merge_yaml_and_args(str(runtime), inference_parser, args)
    args.ttt_checkpoint_path = str(local_checkpoint)
    set_global_seed(int(args.seed))
    dataset = build_infer_dataset(args)
    support_args = copy.copy(args)
    support_args.dataset_metadata_path = str(frozen / "support.jsonl")
    support_dataset = build_infer_dataset(support_args)
    template_args = copy.copy(args)
    template_args.dataset_metadata_path = str(frozen / "extension_action_templates.jsonl")
    template_dataset = build_infer_dataset(template_args)
    model, installation = build_training_faithful_model(args)
    model.requires_grad_(False)
    pipe, controller = model.pipe, installation.controller
    original = pipe.model_fn
    json_write(output / "provenance.json", {
        "task": cli.task, "method": "ttt", "checkpoint_step": setting["checkpoint_step"],
        "training_run": setting["training_run"], "training_checkpoint": setting["checkpoint"],
        "local_checkpoint": str(local_checkpoint), "layers": list(installation.layer_indices),
        "protocol": protocol, "query_future_gt_used": False,
        "normalization": str(frozen / "action_stats.json"),
        "reference_model_is_not_an_ours_context_table": True,
    })

    def generate(sample, sample_index, destination, replays, seed, trace_path):
        args.seed = int(seed)
        set_global_seed(args.seed)
        rollout = prepare_sample_for_rollout(copy.copy(sample), sample_index, pipe, args)
        rollout["output_path"] = str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        wrapper = MultiSupportReplay(pipe, controller, original, replays)
        pipe.model_fn = wrapper
        try:
            with torch.no_grad():
                _run_autoregressive(pipe=pipe, sample=rollout, args=args)
        finally:
            pipe.model_fn = original
            controller.clear()
        prediction = read_video(destination)
        video_write(destination, prediction, 20)
        json_write(trace_path, {"seed": args.seed, "support_count": len(replays),
                               "denoising_calls": len(wrapper.trace), "trace": wrapper.trace})
        return prediction

    grid_order = {}
    completed_count = 0
    extensions = 0
    for env in plan["environments"]:
        name = env["environment"]
        replays, support_records = [], []
        for position, index in enumerate(env["support_indices"]):
            seed = int(protocol["seed"]) + int(protocol["support_noise_seed_offset"]) + int(env["environment_index"]) * 1009 + position
            inputs, latents, noise = _prepare_support_model_inputs(model, support_dataset[index], noise_seed=seed)
            replays.append(SameTimestepSupportQueryModelFn(
                pipe=pipe, controller=controller, original_model_fn=original,
                support_model_inputs=inputs, support_input_latents=latents, support_noise=noise))
            support_records.append({"support_index": index, "level": supports[index]["action_level"],
                                    "episode_index": supports[index]["episode_index"], "noise_seed": seed})
        json_write(output / "support_memory" / f"{name}.json", {
            "environment": name, "supports": support_records,
            "write_policy": "all_supports_at_current_query_denoising_timestep",
            "same_clean_support_latents_encoded_once": True, "fast_state_reset_per_timestep": True})
        for index in env["query_indices"]:
            row = queries[index]
            query_key = key(index, row)
            gt_path = output / "raw/gt" / f"{query_key}.mp4"
            prediction_path = output / "raw/ttt" / f"{query_key}.mp4"
            comparison = output / "comparisons" / f"{query_key}.mp4"
            marker = output / "completed" / f"{query_key}.json"
            if not (marker.exists() and gt_path.exists() and prediction_path.exists() and comparison.exists()):
                sample = dataset[index]
                gt = frame_array(sample["video"])
                prediction = generate(sample, index, prediction_path, replays, int(protocol["seed"]) + index,
                                      output / "memory_trace" / f"{query_key}.json")
                if prediction.shape != gt.shape:
                    raise RuntimeError(f"GT/prediction geometry mismatch: {query_key}")
                video_write(gt_path, gt, 20)
                video_write(comparison, np.concatenate([titled(gt, f"GT L{row['action_level']} {row['dataset_split']}"),
                                                        titled(prediction, f"TTT L{row['action_level']}")], axis=1), 20)
                json_write(marker, {"query": row, "prediction": str(prediction_path), "gt": str(gt_path),
                                    "checkpoint_step": setting["checkpoint_step"], "support_indices": env["support_indices"],
                                    "paired_gt": True, "query_future_gt_used": False})
                del sample, gt, prediction
            completed_count += 1
            json_write(output / "progress.json", {"completed_queries": completed_count, "total_queries": len(queries),
                                                  "current_environment": name, "completed_extensions": extensions})
            print(f"[query_done] {query_key} {completed_count}/{len(queries)}", flush=True)
        grid_order[name] = {}
        for split in ["train", "test"]:
            indices = sorted([i for i in env["query_indices"] if queries[i]["dataset_split"] == split],
                             key=lambda i: (int(queries[i]["action_level"]), int(queries[i]["episode_index"]), i))
            grid_order[name][split] = indices
            columns = [read_video(output / "comparisons" / (key(i, queries[i]) + ".mp4")) for i in indices]
            if columns:
                video_write(output / "grids" / f"{name}_gt_ttt_{split}_levels.mp4", np.concatenate(columns, axis=2), 20)
            del columns
        anchor = int(env["extension_anchor_query_index"])
        columns = []
        for index in sorted(env["extension_template_indices"], key=lambda i: int(templates[i]["action_level"])):
            row = templates[index]
            if row["dataset_split"] != "train" or row["environment"] != name:
                raise RuntimeError("Action template must be a same-environment training sample")
            level = int(row["action_level"])
            identifier = f"{name}_testinitial_L{level:02d}_ttt"
            destination = output / "raw/action_sweeps" / f"{identifier}.mp4"
            marker = output / "completed" / f"extension_{name}_L{level:02d}.json"
            if marker.exists() and destination.exists():
                prediction = read_video(destination)
            else:
                sample = dataset[anchor]
                sample["action"] = template_dataset[index]["action"]
                seed = int(protocol["seed"]) + 100000 + anchor
                prediction = generate(sample, anchor, destination, replays, seed,
                                      output / "memory_trace" / f"{identifier}.json")
                json_write(marker, {"anchor_query_index": anchor, "action_template": row, "seed": seed,
                                    "paired_gt": False, "query_future_gt_used": False})
                del sample
            columns.append(titled(prediction, f"TTT L{level} fixed initial; no paired GT"))
            extensions += 1
            print(f"[extension_done] {identifier}", flush=True)
        if columns:
            video_write(output / "action_sweeps" / f"{name}_ttt_levels.mp4", np.concatenate(columns, axis=2), 20)
        del columns, replays
        controller.clear()
        torch.cuda.empty_cache()
        json_write(output / "grid_order.json", grid_order)
    json_write(output / "inference_complete.json", {
        "task": cli.task, "method": "ttt", "checkpoint_step": setting["checkpoint_step"],
        "queries": completed_count, "environments": len(plan["environments"]), "extensions": extensions,
        "protocol": "same_timestep_multi_support_write_then_query_write_predict",
        "query_manifest_sha256": hashlib.sha256((frozen / "query.jsonl").read_bytes()).hexdigest(),
        "support_manifest_sha256": hashlib.sha256((frozen / "support.jsonl").read_bytes()).hexdigest(),
        "grid_order": "ascending_numeric_levels_separate_train_test", "metric_status": "scoring_pending",
    })
    print("[inference_complete]", output, flush=True)


if __name__ == "__main__":
    main()
