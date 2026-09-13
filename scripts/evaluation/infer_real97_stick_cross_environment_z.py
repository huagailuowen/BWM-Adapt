#!/usr/bin/env python3
"""Hold each factual query fixed and swap all eight training/adapted Z values."""

import argparse
import colorsys
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from real97_trial_common import ROOT, read_jsonl, require_compute, write_json
from infer_real97_reference_trial import stage, write_video


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def freeze_file(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != data:
            raise RuntimeError(f"Input changed during resumable inference: {source}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return hashlib.sha256(data).hexdigest()


def query_key(index, row):
    return f'q{index:04d}_{row["environment"]}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'


def choose_queries(rows, environments, source, config):
    candidates = []
    for index, row in enumerate(rows):
        key = query_key(index, row)
        path = source / "metrics" / (key + "_gt_tracks.jsonl")
        track_rows = read_jsonl(path)
        prefix = track_rows[:config["grasp_fraction_reference_frames"]]
        fractions = [float(item["support_fraction"]) for item in prefix
                     if item.get("support_fraction") is not None and item.get("confidence", 0) >= 0.5
                     and np.isfinite(item["support_fraction"])]
        if not fractions:
            continue
        fraction = float(np.median(fractions))
        candidates.append(dict(index=index, key=key, row=row, grasp_fraction=fraction,
                               grasp_fraction_source=str(path), grasp_fraction_samples=fractions))
    names = [env["environment"] for env in environments]
    if len(names) != 8 or config["query_splits"] != {"train": 4, "test": 4}:
        raise RuntimeError("This experiment requires eight source environments and a 4/4 split")
    targets = config["grasp_fraction_targets_per_split"]
    best, best_score = None, float("inf")
    for train_names in itertools.combinations(names, 4):
        train_names = set(train_names)
        options = [[item for item in candidates if item["row"]["environment"] == name and
                    item["row"]["dataset_split"] == ("train" if name in train_names else "test")] for name in names]
        if any(not items for items in options):
            continue
        for choice in itertools.product(*options):
            score = 0.0
            for split in ("train", "test"):
                positions = sorted(item["grasp_fraction"] for item in choice if item["row"]["dataset_split"] == split)
                score += sum((position - target) ** 2 for position, target in zip(positions, targets))
            if score < best_score:
                best, best_score = choice, score
    if best is None:
        raise RuntimeError("Cannot obtain eight distinct source environments with four train/four test queries")
    selected = sorted(best, key=lambda item: (item["row"]["dataset_split"] != "train", item["grasp_fraction"]))
    return {"selected": selected, "selection_score": best_score, "candidates": candidates,
            "position_targets_per_split": targets, "uses_prediction_quality_for_selection": False,
            "position_definition": "median initial-frame grasp/support fraction along the visible rod, left=0 right=1"}


def cell(frame, text, color, matched=False):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 24), "white")
    image.paste(Image.fromarray(np.asarray(frame, dtype=np.uint8)), (0, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, 23), fill=color)
    draw.text((4, 5), text, fill="white")
    if matched:
        draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="black", width=3)
    return np.asarray(image)


def headed(frame, text):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 40), "white")
    image.paste(Image.fromarray(frame), (0, 40))
    ImageDraw.Draw(image).text((8, 12), text, fill="black")
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    own = parser.parse_args()
    require_compute()
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated usable GPU is required")
    config = json.loads(own.config.read_text())
    prepared, source = resolve(config["prepared"]), resolve(config["source_inference"])
    output = own.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_file(own.config, output / "input_manifest" / "experiment.json")
    hashes = {}
    for relative in ("plan.json", "query.jsonl", "support.jsonl", "stage_files.txt", "reference/training_config.yaml",
                     "reference/action_stats.json", "reference/context_table.json"):
        hashes[relative] = freeze_file(prepared / relative, output / "input_manifest" / relative)
    plan = json.loads((output / "input_manifest/plan.json").read_text())
    if plan["task"] != "stick" or not plan["no_query_GT_in_adaptation"]:
        raise RuntimeError("An existing support-only Stick experiment is required")
    environments = plan["environments"]
    query_rows = read_jsonl(output / "input_manifest/query.jsonl")
    support_rows = read_jsonl(output / "input_manifest/support.jsonl")
    states = {}
    for env in environments:
        name = env["environment"]
        relative = f"contexts/{name}.json"
        hashes[relative] = freeze_file(source / relative, output / "input_manifest" / relative)
        state = json.loads((output / "input_manifest" / relative).read_text())
        if state["support_indices"] != env["support_indices"]:
            raise RuntimeError(f"Support identity mismatch for {name}")
        if any(support_rows[index]["dataset_split"] != "train" or support_rows[index]["environment"] != name
               for index in env["support_indices"]):
            raise RuntimeError(f"Support leakage for {name}")
        if len(state["trajectory"]) != 41 or state["trajectory"][-1]["inner_step"] != 40:
            raise RuntimeError(f"Incomplete original adaptation for {name}")
        states[name] = state
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        selection = choose_queries(query_rows, environments, source, config)
        write_json(selection_path, selection)
    print("[selection] " + json.dumps([{k: item[k] for k in ("index", "key", "grasp_fraction")} for item in selection["selected"]]), flush=True)
    plan, runtime, checkpoint = stage(prepared, output)
    sys.path.insert(0, str(ROOT / "scripts"))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from wan_video_action.utils import save_video, set_global_seed
    sys.argv = ["infer_stage2_ttt", "--config", str(runtime), "--stage2_ckpt_path", str(checkpoint),
                "--support_metadata_path", str(output / "input_manifest/support.jsonl")]
    args = ttt.parse_args()
    args.dataset_metadata_path = str(output / "input_manifest/query.jsonl")
    args.action_stat_path = str(output / "input_manifest/reference/action_stats.json")
    if (args.num_frames, args.frame_stride, args.height, args.width, args.action_type, args.physical_context_dim) != (41, 3, 192, 256, "eef_target", 32):
        raise RuntimeError("Unexpected training/inference geometry or conditioning")
    if args.num_inference_steps != config["num_inference_steps"] or args.video_light_augmentation_enabled:
        raise RuntimeError("Inference settings differ from the original experiment")
    set_global_seed(int(args.seed))
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    ttt._freeze_pipe(pipe)
    table = json.loads((output / "input_manifest/reference/context_table.json").read_text())
    training_values = {float(row["friction_mu"]): row["context"] for row in table["records"]}
    latents = {}
    for env in environments:
        name = env["environment"]
        latents[name] = {}
        for kind, values in (("stage1", training_values[float(env["environment_index"])]), ("stage2", states[name]["context"])):
            value = torch.as_tensor(values, device=pipe.device, dtype=torch.float32).reshape(1, 32)
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite {kind} Z for {name}")
            latents[name][kind] = value
    columns = sorted(environments, key=lambda env: env["train_only_balance_fraction"])
    colors = {env["environment"]: tuple(round(255 * value) for value in colorsys.hsv_to_rgb(index / len(environments), 0.7, 0.65))
              for index, env in enumerate(environments)}
    write_json(output / "latent_provenance.json", {"source": str(source), "input_sha256": hashes,
               "checkpoint_source": str(prepared / "reference/model.safetensors"), "column_order": [env["environment"] for env in columns],
               "stage2_refitted": False, "query_GT_used_for_adaptation": False,
               "latents": {name: {kind: value.cpu().tolist() for kind, value in values.items()} for name, values in latents.items()},
               "same_frame_actions_seed_all_columns": True, "formal_metrics": False})
    completed_grids = []
    base_seed = int(plan["protocol"]["seed"])
    fps = float(config["fps"])
    for selected in selection["selected"]:
        index, row, key = selected["index"], selected["row"], selected["key"]
        source_env = next(env for env in environments if env["environment"] == row["environment"])
        if row["episode_index"] in {support_rows[i]["episode_index"] for i in source_env["support_indices"]}:
            raise RuntimeError("Query episode overlaps original support")
        sample = dataset[index]
        if tuple(sample["video"].shape) != (1, 3, 41, 192, 256) or tuple(np.shape(sample["action"])) != (1, 41, 14):
            raise RuntimeError("Training loader sample shape mismatch")
        gt_path = output / "raw/gt" / (key + ".mp4")
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        if not gt_path.exists():
            save_video(sample["video"], output_path=str(gt_path), fps=7, quality=8)
            write_video(gt_path, imageio.mimread(str(gt_path)), fps)
        gt = imageio.mimread(str(gt_path))
        paths = {}
        for env in columns:
            name = env["environment"]
            for kind in ("stage1", "stage2"):
                path = output / "raw" / kind / name / (key + ".mp4")
                marker = output / "completed" / key / f"{kind}_{name}.json"
                if not marker.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                    rollout.update(physical_context=latents[name][kind], output_path=str(path))
                    args.seed = base_seed + index
                    set_global_seed(args.seed)
                    _run_autoregressive(pipe, rollout, args)
                    frames = imageio.mimread(str(path))
                    if len(frames) != 41 or any(np.asarray(frame).shape != (192, 256, 3) for frame in frames):
                        raise RuntimeError(f"Malformed generated video: {path}")
                    write_video(path, frames, fps)
                    write_json(marker, {"source_query": key, "latent_environment": name, "method": kind,
                               "seed": args.seed, "path": str(path), "same_environment": name == row["environment"]})
                    print(f"[rollout_done] {key} {kind} Z={name}", flush=True)
                paths[(kind, name)] = path
        videos = {identifier: imageio.mimread(str(path)) for identifier, path in paths.items()}
        if len(gt) != 41 or any(len(frames) != 41 for frames in videos.values()):
            raise RuntimeError("Resumed GT/prediction frame-count mismatch")
        title = (f"{row['dataset_split'].upper()} | source {row['environment']} | episode {row['episode_index']} | "
                 f"native {row['start_frame']}-{row['end_frame']} stride 3 | grasp fraction {selected['grasp_fraction']:.3f} | "
                 "same initial frame / action / seed; black border = matching environment Z")
        def grid_frames():
            for frame_index in range(41):
                strips = []
                strips.append(np.hstack([cell(gt[frame_index], "GT | same factual query", (80, 80, 80)) for _ in columns]))
                for kind in ("stage1", "stage2"):
                    strips.append(np.hstack([cell(videos[(kind, env["environment"])][frame_index],
                                                f"{kind} Z | {env['environment'].removeprefix('stick-')}",
                                                colors[env["environment"]], env["environment"] == row["environment"]) for env in columns]))
                yield headed(np.vstack(strips), title)
        grid_path = output / "grids" / (key + "_gt_stage1_stage2_cross_environment_Z.mp4")
        write_video(grid_path, grid_frames(), fps)
        completed_grids.append(str(grid_path))
        write_json(output / "progress.json", {"completed_grids": completed_grids, "planned_grids": 8,
                   "generated_rollouts_per_grid": 16, "current_query": key})
        print(f"[grid_done] {grid_path}", flush=True)
        del sample, videos
        torch.cuda.empty_cache()
    write_json(output / "inference_complete.json", {"queries": 8, "train_queries": 4, "test_queries": 4,
               "latent_environments": 8, "predicted_videos": 128, "grid_layout": "3 rows x 8 columns",
               "grids": completed_grids, "new_stage2_adaptations": 0, "formal_metrics": False})
    print("[inference_complete] " + str(output), flush=True)


if __name__ == "__main__":
    main()
