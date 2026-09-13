#!/usr/bin/env python3
"""Ours step5300 history5 inference; static default, real history is opt-in."""

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
from infer_real97_reference_trial import write_video, labeled, pca_plot
from infer_real97_standard_reference import clean_history_prediction


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def frozen_copy(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise RuntimeError(f"Frozen input changed: {source}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(data)
        os.replace(temporary, destination)


def cached_copy(source, destination, directory=False, files=None):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(destination) + ".copy_complete")
    with Path(str(destination) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists() or (directory and (destination / ".copy_complete").exists()):
            if not directory and destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"Cached checkpoint size mismatch: {destination}")
            return
        if directory:
            destination.mkdir(exist_ok=True)
            command = ["rsync", "-a"] + (["--files-from=" + str(files)] if files else [])
            subprocess.run(command + [str(source) + "/", str(destination) + "/"], check=True)
        else:
            temporary = Path(str(destination) + ".partial")
            subprocess.run(["rsync", "-a", str(source), str(temporary)], check=True)
            os.replace(temporary, destination)
        marker.write_text("complete\n")


def static_support(row):
    result = dict(row)
    end = min(84, int(row["total_frames"]) - 1)
    native = [0] * 5 + [min(frame, end) for frame in range(3, 85, 3)]
    result.update(source_sample_id=row["sample_id"],
                  sample_id=f'{row["environment"]}/ep{row["episode_index"]:06d}/static_history5_support',
                  source_start_frame=row["start_frame"], source_end_frame=row["end_frame"],
                  start_frame=0, end_frame=end, frame_stride=3, length=33,
                  action_semantics="eef_target", num_history_frames=5,
                  history_anchor_frame=0, history_padding_frames=4,
                  native_frame_indices=native, observed_native_frame_indices=[0] * 5,
                  prediction_native_frame_indices=native[5:],
                  evaluation_frame_indices=[index for index in range(5, 33) if (index - 4) * 3 <= end],
                  original_direction_annotation_scope="old selected window, not re-certified for the static prefix")
    return result


def real_history_support(row):
    """Keep the original informative window, with its first five frames observed."""
    result = dict(row)
    start = int(row["start_frame"])
    end = min(int(row["end_frame"]), int(row["total_frames"]) - 1)
    if start < 0 or end < start or int(row["frame_stride"]) != 3 or int(row["length"]) != 33:
        raise RuntimeError("Original support does not match the 33-frame stride3 contract")
    native = [min(start + 3 * index, end) for index in range(33)]
    result.update(
        action_semantics="eef_target", num_history_frames=5,
        history_anchor_frame=start + 12, history_padding_frames=0,
        native_frame_indices=native, observed_native_frame_indices=native[:5],
        prediction_native_frame_indices=native[5:],
        evaluation_frame_indices=[index for index in range(5, 33) if start + 3 * index <= end],
        support_window_policy="unchanged_original_window_real_history5",
    )
    return result


class ContextPipe:
    """Inject Z into the existing Standard clean-history sampler, without edits."""

    def __init__(self, pipe, context):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_context", context)

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def __setattr__(self, name, value):
        setattr(self._pipe, name, value)

    def __call__(self, *args, **kwargs):
        kwargs["physical_context"] = self._context
        return self._pipe(*args, **kwargs)


def history_support_loss(pipe, inputs, args):
    from wan_video_action.methods.baselines.history_loss import history_prefix_flow_loss
    if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
        shared, positive, negative = inputs
    elif isinstance(inputs, dict) and all(key in inputs for key in ("inputs_shared", "inputs_posi", "inputs_nega")):
        shared, positive, negative = (inputs[key] for key in ("inputs_shared", "inputs_posi", "inputs_nega"))
    elif isinstance(inputs, dict):
        shared, positive, negative = inputs, {}, {}
    else:
        raise TypeError(f"Unsupported prepared support input: {type(inputs)}")
    shared = dict(shared)
    shared["num_history_frames"] = 5
    shared.setdefault("max_timestep_boundary", float(args.max_timestep_boundary))
    shared.setdefault("min_timestep_boundary", float(args.min_timestep_boundary))
    return history_prefix_flow_loss(pipe, shared, positive, negative)


def rgb_video(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.ndim == 5:
            value = value[0].permute(1, 2, 3, 0)
        elif value.ndim == 4 and value.shape[1] == 3:
            value = value.permute(0, 2, 3, 1)
        return np.clip((value.detach().float().cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    result = np.asarray([np.asarray(frame) for frame in value])
    if result.dtype != np.uint8:
        raise ValueError("Unexpected non-uint8 pipeline frames")
    return result


def prepare(config, output):
    snapshot = resolve(config["snapshot"])
    old = resolve(config["original_prepared"])
    static_mode = bool(config.get("support_and_query_static", True))
    standard = resolve(config["standard_static_inference"] if static_mode else config["standard_reference_inference"])
    frozen_copy(snapshot / "training_config.yaml", output / "input_manifest/training_config.yaml")
    frozen_copy(snapshot / "action_stats.json", output / "input_manifest/action_stats.json")
    frozen_copy(snapshot / "context_table.json", output / "input_manifest/context_table.json")
    frozen_copy(old / "plan.json", output / "input_manifest/original_plan.json")
    frozen_copy(old / "support.jsonl", output / "input_manifest/original_support.jsonl")
    frozen_copy(standard / "query.jsonl", output / "query.jsonl")
    original = json.loads((output / "input_manifest/original_plan.json").read_text())
    support_converter = static_support if static_mode else real_history_support
    supports = [support_converter(row) for row in read_jsonl(output / "input_manifest/original_support.jsonl")]
    support_path = output / "support.jsonl"
    if support_path.exists():
        if read_jsonl(support_path) != supports:
            raise RuntimeError("Frozen support manifest mismatch")
    else:
        write_jsonl(support_path, supports)
    queries = read_jsonl(output / "query.jsonl")
    if len(queries) != config["query_count"] or len(original["environments"]) != config["environment_count"]:
        raise RuntimeError("Unexpected query or environment count")
    if static_mode:
        if any(row["observed_native_frame_indices"] != [0] * 5 or row["history_padding_frames"] != 4 for row in queries + supports):
            raise RuntimeError("Non-static input entered this experiment")
    else:
        for row in queries + supports:
            start = int(row["start_frame"])
            end = min(int(row["end_frame"]), int(row["total_frames"]) - 1)
            expected = [min(start + 3 * index, end) for index in range(33)]
            if (row["native_frame_indices"] != expected
                    or row["observed_native_frame_indices"] != expected[:5]
                    or row["prediction_native_frame_indices"] != expected[5:]
                    or row["history_anchor_frame"] != start + 12
                    or row["history_padding_frames"] != 0):
                raise RuntimeError("Real-history input differs from the original window")
    for env in original["environments"]:
        chosen = [supports[index] for index in env["support_indices"]]
        if len(chosen) != 4 or any(row["dataset_split"] != "train" or row["environment"] != env["environment"] for row in chosen):
            raise RuntimeError("Support split or identity mismatch")
        support_episodes = {row["episode_index"] for row in chosen}
        if any(queries[index]["episode_index"] in support_episodes for index in env["query_indices"]):
            raise RuntimeError("Query/support episode overlap")
    cache = Path("/tmp") / os.environ["USER"] / "bwm_shared_cache"
    wan = cache / "Wan2.2-TI2V-5B"
    cached_copy(ROOT / "models/Wan2.2-TI2V-5B", wan, directory=True)
    data_tag = hashlib.sha256(str(old.resolve()).encode()).hexdigest()[:16]
    local_data = cache / "eval_datasets" / data_tag
    cached_copy(Path(original["source_dataset"]), local_data, directory=True, files=old / "stage_files.txt")
    model_source = snapshot / "model.safetensors"
    identity = f"{model_source.resolve()}:{model_source.stat().st_size}:{model_source.stat().st_mtime_ns}"
    checkpoint = cache / "eval_checkpoints" / ("ours_history5_" + hashlib.sha256(identity.encode()).hexdigest()[:20] + ".safetensors")
    cached_copy(model_source, checkpoint)
    training = yaml.safe_load((output / "input_manifest/training_config.yaml").read_text())
    flat = {key: value for section in training.values() if isinstance(section, dict) for key, value in section.items()}
    flat.update(original["protocol"]["ttt"])
    flat.update(dataset_base_path=str(local_data), dataset_metadata_path=str(output / "query.jsonl"),
                action_stat_path=str(output / "input_manifest/action_stats.json"), ckpt_path=str(checkpoint),
                model_paths=str(wan), output_path=str(output / "raw/stage2"),
                seed=config["seed"], max_samples=0, fps=7, quality=8,
                video_light_augmentation_enabled=False, use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=False, stage2_group_keys="environment",
                stage2_inner_lr_schedule=config["stage2_inner_lr_schedule"], stage2_inner_steps=config["stage2_inner_steps"],
                num_inference_steps=config["num_inference_steps"])
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"inference": flat}, sort_keys=False))
    write_json(output / "provenance.json", {
        "method": "ours", "training_run": config["training_run"], "model_step": 5300, "table_step": 5300,
        "protected_snapshot": str(snapshot), "local_checkpoint": str(checkpoint), "checkpoint_identity": identity,
        "standard_reference": str(standard), "standard_model_step": 5500,
        "query_and_support_static": static_mode, "support_episode_ids_reused": True,
        "support_windows_changed_to_episode_start": static_mode,
        "old_window_direction_labels_not_recertified": static_mode,
        "support_window_policy": "episode_start_padding" if static_mode else "unchanged_original_informative_window",
        "observed_frames_policy": "five_repeated_frame_zero" if static_mode else "first_five_real_sampled_frames_of_original_window",
        "action": "training canonical target EEF, eight channels plus six zero channels",
        "history_loss": "training history_prefix_flow_loss; two clean observed latent frames, seven future targets",
        "raw_frames": 33, "comparison_frames": 29 if static_mode else 33, "prediction_frames": 28,
        "stage2_initial": "mean step5300 table", "stage2_trainable": "Z only", "query_GT_used_for_adaptation": False,
        "formal_metric_approved": False,
    })
    return original, runtime, checkpoint, queries, supports, standard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    own = parser.parse_args()
    require_compute()
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one usable allocated GPU required")
    output = own.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen_copy(own.config, output / "input_manifest/experiment.json")
    config = json.loads(own.config.read_text())
    static_mode = bool(config.get("support_and_query_static", True))
    comparison_start = 4 if static_mode else 0
    comparison_frames = 33 - comparison_start
    plan, runtime, checkpoint, query_rows, support_rows, standard = prepare(config, output)
    sys.path.insert(0, str(ROOT / "scripts"))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset, build_pipeline
    from wan_video_action.data.history_prefix import CanonicalTargetEEF, HistoryPrefixDataset
    from wan_video_action.utils import set_global_seed
    sys.argv = ["infer_stage2_ttt", "--config", str(runtime), "--stage2_ckpt_path", str(checkpoint),
                "--support_metadata_path", str(output / "support.jsonl")]
    args = ttt.parse_args()
    if (args.num_frames, args.num_history_frames, args.frame_stride, args.action_type, args.action_dim,
            args.height, args.width, args.physical_context_dim) != (33, 5, 3, "eef_target", 14, 160, 320, 32):
        raise RuntimeError("Ours history model contract mismatch")
    stats = json.loads((output / "input_manifest/action_stats.json").read_text())["eef_target"]
    def dataset_for(metadata):
        dataset_args = copy.copy(args)
        dataset_args.dataset_metadata_path = str(metadata)
        dataset = build_infer_dataset(dataset_args)
        dataset.special_operator_map["action"] = CanonicalTargetEEF(dataset_args, stats)
        return HistoryPrefixDataset(dataset, dataset_args)
    query_dataset = dataset_for(output / "query.jsonl")
    support_dataset = dataset_for(output / "support.jsonl")
    set_global_seed(int(args.seed))
    pipe = build_pipeline(args)
    ttt._freeze_pipe(pipe)
    table = json.loads((output / "input_manifest/context_table.json").read_text())
    lookup = {float(row["friction_mu"]): torch.as_tensor(row["context"], device=pipe.device, dtype=torch.float32).reshape(1, 32)
              for row in table["records"]}
    if len(lookup) != 15:
        raise RuntimeError("Expected fifteen step5300 table entries")
    initial = torch.stack(list(lookup.values())).mean(0)
    contexts, results = {}, []
    for env in plan["environments"]:
        name = env["environment"]
        context_path = output / "contexts" / (name + ".json")
        if context_path.exists():
            state = json.loads(context_path.read_text())
            if (state["model_step"] != 5300 or state["table_step"] != 5300
                    or state.get("static_support") != static_mode):
                raise RuntimeError("Cannot reuse context from a different conditioning experiment")
            adapted = torch.as_tensor(state["context"], device=pipe.device, dtype=torch.float32)
        else:
            support = [support_dataset[index] for index in env["support_indices"]]
            set_global_seed(int(config["seed"]) + int(env["environment_index"]) * 1009)
            legacy_loss = ttt._flow_match_loss
            ttt._flow_match_loss = history_support_loss
            try:
                adapted, losses, _, metrics, trajectory = ttt._adapt_ttt_state(
                    pipe, support, args, adapter_named_params=[], base_adapter_state={}, initial_context=initial,
                    target_context=None, trajectory_meta={"environment": name, "sample_index": env["query_indices"][0],
                                                         "friction_mu": float(env["environment_index"])})
            finally:
                ttt._flow_match_loss = legacy_loss
            state = {"context": adapted.detach().cpu().tolist(), "initial": initial.cpu().tolist(),
                     "losses": losses, "metrics": metrics, "trajectory": trajectory,
                     "support_indices": env["support_indices"], "model_step": 5300, "table_step": 5300,
                     "static_support": static_mode}
            write_json(context_path, state)
            del support
        contexts[name] = state
        columns, baseline_columns = [], []
        for index in env["query_indices"]:
            row = query_rows[index]
            key = f'q{index:04d}_{name}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'
            baseline_marker = json.loads((standard / "completed" / (key + ".json")).read_text())
            if baseline_marker["row"] != row:
                raise RuntimeError("Standard/Ours query identity mismatch")
            paths = {kind: output / "raw" / kind / (key + ".mp4") for kind in ("gt", "stage1", "stage2")}
            marker = output / "completed" / (key + ".json")
            if not marker.exists():
                sample = query_dataset[index]
                if tuple(sample["video"].shape) != (1, 3, 33, 160, 320) or tuple(np.shape(sample["action"])) != (1, 33, 14):
                    raise RuntimeError("History5 data/video/action shape mismatch")
                # The shared history sampler expects [B,C,T,H,W] in [-1,1],
                # not uint8 PIL frames. It selects the five observed frames.
                observed_video = torch.as_tensor(sample["video"], dtype=torch.float32)
                gt_rgb = rgb_video(observed_video)
                paths["gt"].parent.mkdir(parents=True, exist_ok=True)
                if not paths["gt"].exists():
                    try:
                        os.link(baseline_marker["paths"]["gt"], paths["gt"])
                    except OSError:
                        shutil.copy2(baseline_marker["paths"]["gt"], paths["gt"])
                for kind, context in (("stage1", lookup[float(env["environment_index"])]), ("stage2", adapted)):
                    variant_marker = output / "completed_variants" / f"{key}_{kind}.json"
                    if not variant_marker.exists():
                        args.seed = int(config["seed"]) + index
                        set_global_seed(args.seed)
                        predicted = rgb_video(clean_history_prediction(ContextPipe(pipe, context), observed_video, sample["action"], args))
                        if predicted.shape != (33, 160, 320, 3):
                            raise RuntimeError(f"Unexpected prediction shape: {predicted.shape}")
                        predicted[:5] = gt_rgb[:5]
                        write_video(paths[kind], predicted, config["fps"])
                        write_json(variant_marker, {"query": key, "method": kind, "seed": args.seed, "frames": 33})
                write_json(marker, {"index": index, "row": row, "paths": {kind: str(path) for kind, path in paths.items()},
                                    "observed_frames": 5, "paired_gt": True})
                del sample
                torch.cuda.empty_cache()
            videos = {kind: imageio.mimread(str(path)) for kind, path in paths.items()}
            videos["standard"] = imageio.mimread(baseline_marker["paths"]["standard"])
            if any(len(video) != 33 for video in videos.values()):
                raise RuntimeError("Prediction/GT/Standard frame count mismatch")
            def comparison(kinds):
                return [np.vstack([labeled(np.asarray(videos[kind][frame]), f'{kind} | {row["dataset_split"]} ep{row["episode_index"]}')
                                   for kind in kinds]) for frame in range(comparison_start, 33)]
            column = comparison(("gt", "stage1", "stage2"))
            baseline_column = comparison(("gt", "standard", "stage1", "stage2"))
            write_video(output / "comparisons" / (key + ".mp4"), column, config["fps"])
            write_video(output / "comparisons_with_standard" / (key + ".mp4"), baseline_column, config["fps"])
            columns.append(column)
            baseline_columns.append(baseline_column)
            results.append({"index": index, "environment": name, "split": row["dataset_split"], "paths": {kind: str(path) for kind, path in paths.items()}})
            write_json(output / "progress.json", {"completed_environments": list(contexts), "query_results": results})
            print(f"[query_done] {key}", flush=True)
        for directory, values in (("grids", columns), ("grids_with_standard", baseline_columns)):
            suffix = "_static_train_test.mp4" if static_mode else "_history5_train_test.mp4"
            write_video(output / directory / (name + suffix),
                        (np.hstack([column[frame] for column in values]) for frame in range(comparison_frames)), config["fps"])
        pca_plot(output / "training_inference_Z_pca.svg", table, contexts,
                 [item for item in plan["environments"] if item["environment"] in contexts])
    write_json(output / "inference_complete.json", {
               "method": "ours_history5_static" if static_mode else "ours_history5_real_history",
               "model_step": 5300, "table_step": 5300,
               "queries": len(results), "environments": len(contexts), "support_and_query_static": static_mode,
               "raw_frames": 33, "comparison_frames": comparison_frames, "formal_metric_approved": False})
    print("[inference_complete] " + str(output), flush=True)


if __name__ == "__main__":
    main()
