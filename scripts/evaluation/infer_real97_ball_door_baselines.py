#!/usr/bin/env python3
"""Frozen factual queries and action sweeps for Standard and DINO Real97 models.

All image/model work runs on a compute node. DINO consumes only the explicitly
selected training supports; factual query futures are never encoder inputs.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from scripts.infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
from scripts.methods.infer_dinov2_event80 import materialize_wan_checkpoint, load_support_encoder
from scripts.methods.train_dinov2_event80 import add_dinov2_config
from scripts.evaluation.infer_real97_standard_reference import copy_cached
from wan_video_action.parsers import add_general_config, merge_yaml_and_args


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temp, path)


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def snapshot(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError(f"Frozen input changed: {source}")
    else:
        shutil.copy2(source, target)


def frame_array(video):
    tensor = torch.as_tensor(video).detach().cpu().float()
    if tensor.ndim == 5:
        tensor = tensor[0]
    return ((tensor.permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).round().byte().numpy()


def read_video(path):
    reader = imageio.get_reader(str(path))
    try:
        return np.stack([frame for frame in reader])
    finally:
        reader.close()


def video_write(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.partial.mp4")
    with imageio.get_writer(str(temp), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    os.replace(temp, path)


def titled(frames, title):
    canvas = frames.copy()
    banner_height = min(23, canvas.shape[1])
    banner = Image.new("RGB", (canvas.shape[2], banner_height), (15, 15, 15))
    ImageDraw.Draw(banner).text((5, 4), title, fill=(255, 255, 255))
    canvas[:, :banner_height] = np.asarray(banner)
    return canvas


def key(index, row):
    return (f"q{index:04d}_{row['environment']}_{row['dataset_split']}"
            f"_L{int(row['action_level']):02d}_ep{int(row['episode_index']):06d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "ball"))
    parser.add_argument("--method", required=True, choices=("standard", "dino"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    own = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This inference must run on a single-GPU compute allocation.")
    protocol = json.loads(Path(own.config).read_text())
    setting = protocol["tasks"][own.task]
    prepared = Path(setting["prepared"])
    training = Path(setting[own.method + "_training"])
    output = Path(own.output)
    output.mkdir(parents=True, exist_ok=True)
    run_lock = (output / ".run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = output / "input_manifest"
    for name in ("plan.json", "query.jsonl", "support.jsonl", "extension_action_templates.jsonl", "stage_files.txt"):
        snapshot(prepared / name, frozen / name)
    snapshot(training / "submitted_config.yaml", frozen / "training_config.yaml")
    snapshot(training / "input_manifest/action_stats.json", frozen / "action_stats.json")
    snapshot(Path(own.config), frozen / "evaluation_config.json")
    plan = json.loads((frozen / "plan.json").read_text())
    queries = rows(frozen / "query.jsonl")
    supports = rows(frozen / "support.jsonl")
    templates = rows(frozen / "extension_action_templates.jsonl")
    config = yaml.safe_load((frozen / "training_config.yaml").read_text())["training"]
    if any(row["action_semantics"] != config["action_type"] for row in queries + supports + templates):
        raise ValueError("Training/evaluation action semantics mismatch.")
    if (config["num_frames"], config["height"], config["width"]) != tuple(setting["geometry"]):
        raise ValueError("Training/evaluation frame geometry mismatch.")
    for env in plan["environments"]:
        support_rows = [supports[i] for i in env["support_indices"]]
        if any(row["dataset_split"] != "train" or row["environment"] != env["environment"] for row in support_rows):
            raise ValueError("Support must be from the same environment's train split.")
        if {r["episode_index"] for r in support_rows} & {queries[i]["episode_index"] for i in env["query_indices"]}:
            raise ValueError("Support/query episode leakage.")
    cache = Path("/tmp") / os.environ["USER"] / "bwm_shared_cache"
    cache.mkdir(parents=True, exist_ok=True)
    wan = cache / "Wan2.2-TI2V-5B"
    copy_cached(ROOT / "models/Wan2.2-TI2V-5B", wan, directory=True)
    data_tag = hashlib.sha256(str(config["dataset_base_path"]).encode() + (frozen / "stage_files.txt").read_bytes()).hexdigest()[:16]
    local_data = cache / "datasets_real" / (own.task + "_eval_" + data_tag)
    copy_cached(Path(config["dataset_base_path"]), local_data, directory=True, files=frozen / "stage_files.txt")
    source_ckpt = training / "protected/step-5500.safetensors"
    stat = source_ckpt.stat()
    ckpt_tag = hashlib.sha256(f"{source_ckpt.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    local_ckpt = cache / "real97_baseline_eval" / ckpt_tag / "model.safetensors"
    copy_cached(source_ckpt, local_ckpt)
    if own.method == "dino":
        dino_local = cache / "dinov2-base"
        copy_cached(Path(config["dinov2_model_path"]), dino_local, directory=True)
        config["dinov2_model_path"] = str(dino_local)
        wan_ckpt = local_ckpt.with_name("wan.safetensors")
        with local_ckpt.with_name("wan.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            materialize_wan_checkpoint(str(local_ckpt), str(wan_ckpt))
    else:
        wan_ckpt = local_ckpt
    config.update(
        dataset_base_path=str(local_data), dataset_metadata_path=str(frozen / "query.jsonl"),
        action_stat_path=str(frozen / "action_stats.json"), model_paths=str(wan),
        ckpt_path=str(wan_ckpt), output_path=str(output / "raw" / own.method),
        num_inference_steps=protocol["num_inference_steps"], cfg_scale=1.0,
        seed=protocol["seed"], frame_stride=1, num_history_frames=1,
        video_light_augmentation_enabled=False, dinov2_support_light_augmentation_enabled=False,
        spatial_loss_mode="none", fps=20, quality=8,
    )
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"inference": config}, sort_keys=False))
    inference_parser = add_dinov2_config(add_general_config(argparse.ArgumentParser()))
    if "--frame_stride" not in inference_parser._option_string_actions:
        inference_parser.add_argument("--frame_stride", type=int, default=1)
    args = inference_parser.parse_args(["--config", str(runtime)])
    args = merge_yaml_and_args(str(runtime), inference_parser, args)
    args.dinov2_checkpoint_path = str(local_ckpt)
    dataset = build_infer_dataset(args)
    support_args = copy.copy(args)
    support_args.dataset_metadata_path = str(frozen / "support.jsonl")
    support_dataset = build_infer_dataset(support_args)
    template_args = copy.copy(args)
    template_args.dataset_metadata_path = str(frozen / "extension_action_templates.jsonl")
    template_dataset = build_infer_dataset(template_args)
    pipe = build_pipeline(args)
    contexts = {}
    if own.method == "dino":
        encoder = load_support_encoder(args, pipe.device)
        with torch.no_grad():
            for env in plan["environments"]:
                data = [support_dataset[i] for i in env["support_indices"]]
                features = [encoder.extract_visual_features(row["video"]) for row in data]
                code = encoder.project_supports(
                    visual_features=tuple(item[0] for item in features),
                    actions=tuple(row["action"] for row in data),
                    frame_indices=tuple(item[1] for item in features),
                    frame_counts=tuple(item[2] for item in features),
                )[0].detach().float().cpu()
                contexts[env["environment"]] = code
                json_write(output / "contexts" / (env["environment"] + ".json"), {
                    "environment": env["environment"], "context": code.tolist(),
                    "support_indices": env["support_indices"],
                    "support_levels": [int(supports[i]["action_level"]) for i in env["support_indices"]],
                    "aggregation": "mean_support_summary_before_output_head",
                    "uses_query_future": False,
                })
        del encoder
        torch.cuda.empty_cache()

    def generate(sample, sample_index, destination, environment):
        args.seed = int(protocol["seed"]) + int(sample_index)
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        sample["output_path"] = str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if own.method == "dino":
            sample["physical_context"] = contexts[environment].to(device=pipe.device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            _run_autoregressive(pipe=pipe, sample=sample, args=args)
        frames = read_video(destination)
        video_write(destination, frames, 20)
        return frames

    grid_order = {}
    for env in plan["environments"]:
        environment = env["environment"]
        for index in env["query_indices"]:
            row = queries[index]
            name = key(index, row)
            gt_path = output / "raw/gt" / (name + ".mp4")
            prediction_path = output / "raw" / own.method / (name + ".mp4")
            comparison = output / "comparisons" / (name + ".mp4")
            marker = output / "completed" / (name + ".json")
            if marker.exists() and gt_path.exists() and prediction_path.exists() and comparison.exists():
                continue
            sample = dataset[index]
            gt = frame_array(sample["video"])
            prediction = generate(sample, index, prediction_path, environment)
            if prediction.shape != gt.shape:
                raise RuntimeError(f"Prediction/GT shape mismatch for {name}: {prediction.shape}, {gt.shape}")
            video_write(gt_path, gt, 20)
            video_write(comparison, np.concatenate([
                titled(gt, f"GT L{int(row['action_level'])} {row['dataset_split']}"),
                titled(prediction, f"{own.method} L{int(row['action_level'])}"),
            ], axis=1), 20)
            json_write(marker, {"query": row, "prediction": str(prediction_path),
                                "gt": str(gt_path), "checkpoint_step": 5500})
            print(f"[query] method={own.method} task={own.task} index={index}/{len(queries)}", flush=True)
        grid_order[environment] = {}
        for split in ("train", "test"):
            indices = sorted((i for i in env["query_indices"] if queries[i]["dataset_split"] == split),
                             key=lambda i: (int(queries[i]["action_level"]), int(queries[i]["episode_index"]), i))
            grid_order[environment][split] = indices
            columns = [read_video(output / "comparisons" / (key(i, queries[i]) + ".mp4")) for i in indices]
            if columns:
                video_write(output / "grids" / f"{environment}_gt_{own.method}_{split}_levels.mp4",
                            np.concatenate(columns, axis=2), 20)

        anchor_index = int(env["extension_anchor_query_index"])
        extension_columns = []
        for template_index in sorted(env["extension_template_indices"], key=lambda i: int(templates[i]["action_level"])):
            row = templates[template_index]
            level = int(row["action_level"])
            destination = output / "raw/action_sweeps" / f"{environment}_testinitial_L{level:02d}_{own.method}.mp4"
            marker = output / "completed" / f"extension_{environment}_L{level:02d}.json"
            if destination.exists() and marker.exists():
                prediction = read_video(destination)
            else:
                sample = dataset[anchor_index]
                sample["action"] = template_dataset[template_index]["action"]
                prediction = generate(sample, anchor_index, destination, environment)
                json_write(marker, {"anchor_query_index": anchor_index, "action_template": row,
                                    "paired_gt": False, "seed": int(protocol["seed"]) + anchor_index})
            extension_columns.append(titled(prediction, f"{own.method} L{level} fixed initial"))
        if extension_columns:
            video_write(output / "action_sweeps" / f"{environment}_{own.method}_levels.mp4",
                        np.concatenate(extension_columns, axis=2), 20)
    json_write(output / "grid_order.json", grid_order)
    json_write(output / "inference_complete.json", {
        "task": own.task, "method": own.method, "checkpoint": str(source_ckpt),
        "checkpoint_step": 5500, "queries": len(queries), "environments": len(plan["environments"]),
        "query_manifest_sha256": hashlib.sha256((frozen / "query.jsonl").read_bytes()).hexdigest(),
        "support_policy": "current_ours_selected_train_supports" if own.method == "dino" else "none",
        "grid_order": "ascending_action_level_separate_train_test", "inference_augmentation": False,
        "extensions_have_paired_gt": False, "metric_status": "tracking_scoring_pending",
    })


if __name__ == "__main__":
    main()
