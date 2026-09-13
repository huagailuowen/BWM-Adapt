#!/usr/bin/env python3
"""Opt-in pooled baseline on the frozen real97 query windows.

Soft preserves the previous 33 timestamps: the first five are observed, not
autoregressively generated. Its last observed frame is the training anchor.
No old inference, training, or evaluation entry point is modified.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from real97_trial_common import ROOT, read_jsonl, require_compute, write_json, write_jsonl
from infer_real97_reference_trial import labeled, write_video


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def copy_cached(source, destination, *, directory=False, files=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(destination) + ".copy_complete")
    with Path(str(destination) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory:
            complete = destination.is_dir() and (marker.is_file() or (destination / ".copy_complete").is_file())
        else:
            complete = marker.is_file() and destination.is_file() and destination.stat().st_size == source.stat().st_size
        if complete:
            print(f"[cache_reuse] {destination}", flush=True)
            return
        print(f"[cache_stage] {source} -> {destination}", flush=True)
        if directory:
            destination.mkdir(parents=True, exist_ok=True)
            command = ["rsync", "-a"]
            if files is not None:
                command.append("--files-from=" + str(files))
            subprocess.run(command + [str(source) + "/", str(destination) + "/"], check=True)
        else:
            temporary = Path(str(destination) + ".partial")
            subprocess.run(["rsync", "-a", str(source), str(temporary)], check=True)
            if temporary.stat().st_size != source.stat().st_size:
                raise RuntimeError("Incomplete staged checkpoint")
            os.replace(temporary, destination)
        marker.write_text("complete\n")


def snapshot(source, destination):
    """Freeze small inputs; requeue must not silently change its protocol."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError(f"Frozen inference input changed: {source}")
        return
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(content)
    os.replace(temporary, destination)


def prepare(own, protocol):
    setting = protocol["tasks"][own.task]
    prepared = resolve(setting["prepared"])
    training = resolve(setting["training_run"])
    reference = own.output / "reference"
    for source, name in [
        (prepared / "plan.json", "source_plan.json"),
        (prepared / "query.jsonl", "source_query.jsonl"),
        (prepared / "support.jsonl", "source_support.jsonl"),
        (prepared / "stage_files.txt", "stage_files.txt"),
        (training / "submitted_config.yaml", "training_config.yaml"),
        (training / "input_manifest/action_stats.json", "action_stats.json"),
        (own.config, "inference_config.json"),
    ]:
        snapshot(source, reference / name)
    plan = json.loads((reference / "source_plan.json").read_text())
    flat = yaml.safe_load((reference / "training_config.yaml").read_text())["training"]
    rows = read_jsonl(reference / "source_query.jsonl")
    history = int(setting["history_frames"])
    frames = int(setting["model_frames"])
    stride = int(flat["frame_stride"])
    if len(rows) != setting["expected_queries"] or len(plan["environments"]) != setting["expected_environments"]:
        raise ValueError("Frozen query population differs from the approved previous experiment")
    if (int(flat["num_frames"]), int(flat["num_history_frames"]), stride) != (frames, history, 3):
        raise ValueError("Inference temporal contract differs from the trained checkpoint")
    if flat["physical_context_mode"] != "none" or flat["physical_adapter_mode"] != "none":
        raise ValueError("This entry point is exclusively the no-adaptation pooled baseline")
    if flat["action_type"] != "eef_target":
        raise ValueError("The selected Stick and new Soft checkpoints require target EEF")
    if bool(flat.get("standard_eef_canonicalize", False)) != bool(setting["canonical_target_eef"]):
        raise ValueError("Training/inference EEF canonicalization mismatch")
    covered = sorted(i for env in plan["environments"] for i in env["query_indices"])
    if covered != list(range(len(rows))):
        raise ValueError("Frozen plan must cover each query exactly once")
    converted = []
    for index, old in enumerate(rows):
        row = copy.deepcopy(old)
        if own.episode_start_history_padding:
            row.update(
                source_sample_id=old["sample_id"],
                sample_id=f'{old["environment"]}/ep{old["episode_index"]:06d}/episode_start_history5',
                start_frame=0, end_frame=min(84, int(old["total_frames"])-1),
            )
        if int(row["length"]) != frames or int(row["frame_stride"]) != stride:
            raise ValueError(f"Query {index} has a different trained window length/stride")
        if row["dataset_split"] not in ("train", "test"):
            raise ValueError("Unexpected query split")
        start = int(row["start_frame"])
        last = min(int(row["end_frame"]), int(row["total_frames"]) - 1)
        if not 0 <= start <= last:
            raise ValueError("Empty or invalid frozen query window")
        indices = [min(start + stride * k, last) for k in range(frames)]
        row.update(
            action_semantics=flat["action_type"],
            native_frame_indices=indices,
            num_history_frames=history,
            observed_native_frame_indices=indices[:history],
            prediction_native_frame_indices=indices[history:],
            evaluation_frame_indices=[k for k in range(history, frames) if start + stride * k <= last],
            source_query_index=index,
        )
        if history == 5:
            # Preserve the OLD chunk start/end, not a newly selected anchor.
            # For [27,30,...,123], history is 27..39 and prediction is 42..123.
            row.update(history_anchor_frame=start + 12, history_padding_frames=0)
            if own.episode_start_history_padding:
                indices = [0] * 5 + [min(3*k, last) for k in range(1, 29)]
                eligible = [k for k in range(5, 33) if 3*(k-4) <= last]
                row.update(
                    history_anchor_frame=0, history_padding_frames=4,
                    native_frame_indices=indices,
                    observed_native_frame_indices=indices[:5],
                    prediction_native_frame_indices=indices[5:],
                    evaluation_frame_indices=eligible,
                    comparison_evaluation_frame_indices=[k-4 for k in eligible],
                    initial_stationarity="dataset-start property supplied by user; no detector selection",
                )
        converted.append(row)
    manifest = own.output / "query.jsonl"
    write_jsonl(manifest, converted)
    cache = Path("/tmp") / os.environ["USER"] / "bwm_shared_cache"
    model = cache / "Wan2.2-TI2V-5B"
    copy_cached(ROOT / "models/Wan2.2-TI2V-5B", model, directory=True)
    source_ckpt = training / setting["checkpoint"]
    stat = source_ckpt.stat()
    identity = f"{source_ckpt.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    tag = hashlib.sha256(identity.encode()).hexdigest()[:20]
    checkpoint = cache / "eval_checkpoints" / ("standard_" + tag + ".safetensors")
    copy_cached(source_ckpt, checkpoint)
    dataset_tag = hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
    local_data = cache / "eval_datasets" / dataset_tag
    copy_cached(Path(plan["source_dataset"]), local_data, directory=True, files=reference / "stage_files.txt")
    flat.update(
        dataset_base_path=str(local_data), dataset_metadata_path=str(manifest),
        action_stat_path=str(reference / "action_stats.json"), model_paths=str(model),
        ckpt_path=str(checkpoint), output_path=str(own.output / "raw/standard"),
        seed=int(protocol["seed"]), num_inference_steps=int(protocol["num_inference_steps"]),
        cfg_scale=float(protocol["cfg_scale"]), video_light_augmentation_enabled=False,
        spatial_loss_mode="none", max_samples=0, fps=7, quality=8,
    )
    runtime = own.output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"inference": flat}, sort_keys=False))
    write_json(own.output / "provenance.json", dict(
        method="standard_pooled", task=own.task, training_run=str(training),
        checkpoint=str(source_ckpt), checkpoint_identity=identity, local_checkpoint=str(checkpoint),
        prepared=str(prepared), queries=len(rows), adaptation="none", support_used=False,
        source_query_sha256=hashlib.sha256((reference / "source_query.jsonl").read_bytes()).hexdigest(),
        source_episode_and_window_selection=("same episodes and splits; all anchors reset to native frame zero"
                                            if own.episode_start_history_padding else "unchanged"), model_frames=frames,
        observed_frames=history, predicted_frames=frames-history,
        soft_history_policy=("native indices [0,0,0,0,0,3,6,...,84]; identical video/action padding"
                             if own.episode_start_history_padding else
                             "first five frames of unchanged previous chunk; fifth frame is causal anchor"),
        episode_start_history_padding=own.episode_start_history_padding,
        comparison_frames=29 if own.episode_start_history_padding else frames,
        action_policy="training target EEF and frozen training-only normalization",
        prefix_excluded_from_metrics=True, padded_tail_excluded_from_metrics=True,
        comparison_warning="Soft history and action representation changed versus old one-frame joint-target ours; score only common future timestamps and report conditioning differences",
        formal_metric_approved=False, metric_status="awaiting_independent_calibration",
        fps=float(protocol["source_fps"]) / stride,
    ))
    return plan, converted, runtime, flat


def clean_history_prediction(pipe, video, action, args):
    """Keep observed VAE latents clean at EVERY denoising step, as in training.

This instance-local guard does not alter the legacy pipeline implementation.
It also restores the observed prefix after the final scheduler update.
"""
    expected = (int(args.num_history_frames) - 1) // 4 + 1
    model_fn, scheduler_step = pipe.model_fn, pipe.scheduler.step
    fixed = {}

    def model_with_observed_prefix(*models, **inputs):
        if int(inputs.get("fused_condition_latent_frames", -1)) != expected:
            raise RuntimeError("VAE observed prefix does not match the training contract")
        if "latents" not in fixed:
            fixed["latents"] = inputs["latents"][:, :, :expected].detach().clone()
        inputs["latents"][:, :, :expected].copy_(fixed["latents"])
        return model_fn(*models, **inputs)

    def step_with_observed_prefix(*values, **kwargs):
        result = scheduler_step(*values, **kwargs)
        result[:, :, :expected].copy_(fixed["latents"])
        return result

    pipe.model_fn, pipe.scheduler.step = model_with_observed_prefix, step_with_observed_prefix
    try:
        return pipe(
            input_video=video[:, :, :int(args.num_history_frames)], action=action,
            physical_context=None, seed=int(args.seed), rand_device="cpu", tiled=False,
            height=int(args.height), width=int(args.width), num_frames=int(args.num_frames),
            num_history_frames=int(args.num_history_frames), cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.num_inference_steps),
            use_history_condition_noise_in_inference=False,
            progress_bar_cmd=lambda iterable, *unused, **kwargs: iterable,
            output_type="floatpoint",
        )
    finally:
        pipe.model_fn, pipe.scheduler.step = model_fn, scheduler_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stick", "soft"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-start-history-padding", action="store_true", default=False,
                        help="Soft only: native frame 0 anchor with four missing-history repeats and 28 real future frames")
    own = parser.parse_args()
    if own.episode_start_history_padding and own.task != "soft":
        raise ValueError("Episode-start history padding is opt-in for the five-frame Soft model only")
    require_compute()
    own.config, own.output = own.config.resolve(), own.output.resolve()
    own.output.mkdir(parents=True, exist_ok=True)
    # Prevent a requeue/manual duplicate from writing the same videos together.
    run_lock = (own.output / ".run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one usable allocated GPU is required")
    protocol = json.loads(own.config.read_text())
    plan, rows, runtime, flat = prepare(own, protocol)
    sys.path.insert(0, str(ROOT / "scripts"))
    from infer import parse_args, build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from wan_video_action.utils import save_video, set_global_seed
    sys.argv = ["infer", "--config", str(runtime)]
    args = parse_args()
    dataset = build_infer_dataset(args)
    if own.task == "soft":
        from wan_video_action.data.history_prefix import CanonicalTargetEEF, HistoryPrefixDataset
        stats = json.loads(Path(args.action_stat_path).read_text())["eef_target"]
        dataset.special_operator_map["action"] = CanonicalTargetEEF(args, stats)
        dataset = HistoryPrefixDataset(dataset, args)
    set_global_seed(int(protocol["seed"]))
    pipe = build_pipeline(args)
    fps = float(protocol["source_fps"]) / int(args.frame_stride)
    results = []
    for environment in plan["environments"]:
        name = environment["environment"]
        columns = []
        for index in environment["query_indices"]:
            row = rows[index]
            if row["environment"] != name:
                raise ValueError("Frozen environment/query plan mismatch")
            key = f'q{index:04d}_{name}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'
            paths = {kind: own.output / "raw" / kind / (key + ".mp4") for kind in ("gt", "standard")}
            complete = own.output / "completed" / (key + ".json")
            if not complete.is_file():
                sample = dataset[index]
                expected_video = (1, 3, int(args.num_frames), int(args.height), int(args.width))
                if tuple(sample["video"].shape) != expected_video or tuple(sample["action"].shape) != (1, int(args.num_frames), 14):
                    raise ValueError(f"Training/inference shape mismatch in query {index}")
                args.seed = int(protocol["seed"]) + index
                set_global_seed(args.seed)
                paths["gt"].parent.mkdir(parents=True, exist_ok=True)
                save_video(sample["video"], output_path=str(paths["gt"]), fps=7, quality=8)
                with torch.inference_mode():
                    if own.task == "stick":
                        rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                        rollout["output_path"] = str(paths["standard"])
                        _run_autoregressive(pipe, rollout, args)
                    else:
                        action = torch.as_tensor(sample["action"], device=pipe.device, dtype=pipe.torch_dtype)
                        generated = clean_history_prediction(pipe, sample["video"], action, args).detach().cpu()
                        if tuple(generated.shape) != expected_video or not torch.isfinite(generated).all():
                            raise RuntimeError("Invalid generated history-conditioned video")
                        # Observed frames are presented as observed, not scored as predictions.
                        generated[:, :, :5] = sample["video"][:, :, :5].to(generated.dtype)
                        paths["standard"].parent.mkdir(parents=True, exist_ok=True)
                        temporary = paths["standard"].with_name("." + key + ".partial.mp4")
                        save_video(generated, output_path=str(temporary), fps=7, quality=8)
                        os.replace(temporary, paths["standard"])
                        del generated, action
                for path in paths.values():
                    write_video(path, imageio.mimread(str(path)), fps)
                write_json(complete, dict(index=index, row=row, paths={k: str(v) for k, v in paths.items()},
                                         observed_frames=int(args.num_history_frames), paired_gt=True))
                del sample
            videos = {kind: imageio.mimread(str(path)) for kind, path in paths.items()}
            if any(len(video) != int(args.num_frames) for video in videos.values()):
                raise RuntimeError("GT and Standard output frame counts differ")
            column = []
            display_start = 4 if own.episode_start_history_padding else 0
            for frame in range(display_start, int(args.num_frames)):
                phase = "observed" if frame < int(args.num_history_frames) else "predicted"
                column.append(np.vstack([
                    labeled(videos[kind][frame], f'{kind} | {row["dataset_split"]} ep{row["episode_index"]} | {phase}')
                    for kind in ("gt", "standard")
                ]))
            write_video(own.output / "comparisons" / (key + "_gt_standard.mp4"), column, fps)
            columns.append(column)
            results.append(dict(index=index, environment=name, split=row["dataset_split"],
                                paths={k: str(v) for k, v in paths.items()},
                                evaluation_frame_indices=row["evaluation_frame_indices"]))
            write_json(own.output / "progress.json", dict(completed_queries=len(results), expected_queries=len(rows), results=results))
            print(f"[query_done] {key} {len(results)}/{len(rows)}", flush=True)
        write_video(own.output / "grids" / (name + "_gt_standard_train_test.mp4"),
                    [np.hstack([column[frame] for column in columns]) for frame in range(len(columns[0]))], fps)
    write_json(own.output / "inference_complete.json", dict(
        task=own.task, method="standard_pooled", queries=len(results), environments=len(plan["environments"]),
        model_step=5500, metric_status="awaiting_independent_calibration", formal_metric_approved=False,
        episode_start_history_padding=own.episode_start_history_padding,
    ))
    print(f"[inference_complete] {own.output}", flush=True)


if __name__ == "__main__":
    main()
