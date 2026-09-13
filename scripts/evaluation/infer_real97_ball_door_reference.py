#!/usr/bin/env python3
"""Ball/Door factual GT/S1/S2 and clearly separated fixed-frame action sweeps."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
from real97_trial_common import ROOT, read_jsonl, require_compute, write_json
from infer_real97_reference_trial import stage, write_video, labeled, pca_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    own = parser.parse_args()
    require_compute()
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one usable GPU required")
    own.prepared, own.output = own.prepared.resolve(), own.output.resolve()
    own.output.mkdir(parents=True, exist_ok=True)
    lock = (own.output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan, runtime, checkpoint = stage(own.prepared, own.output)
    setting = plan["setting"]
    fps = float(plan["frames_per_second"])
    write_json(own.output / "provenance.json", {
        "prepared": str(own.prepared), "plan": plan, "checkpoint": str(checkpoint),
        "model_step": setting["model_step"], "table_step": setting["table_step"],
        "frames_per_second": fps, "formal_metric_approved": False,
        "counterfactuals_have_paired_GT": False,
    })
    sys.path.insert(0, str(ROOT / "scripts"))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from wan_video_action.utils import save_video, set_global_seed
    sys.argv = ["infer_stage2_ttt", "--config", str(runtime),
                "--stage2_ckpt_path", str(checkpoint),
                "--support_metadata_path", str(own.prepared / "support.jsonl")]
    args = ttt.parse_args()
    args.fps = fps
    expected = (setting["num_frames"], setting["height"], setting["width"], setting["action_type"])
    if (args.num_frames, args.height, args.width, args.action_type) != expected:
        raise RuntimeError("Training and inference video/action contract differs")
    if int(args.num_history_frames) != 1 or int(args.frame_stride) != 1 or int(args.action_dim) != 14:
        raise RuntimeError("Ball/Door requires one observed frame and stride1 aligned 14D actions")
    def dataset_for(name):
        configured = copy.copy(args)
        configured.dataset_metadata_path = str(own.prepared / name)
        return build_infer_dataset(configured)
    query_rows = read_jsonl(own.prepared / "query.jsonl")
    support_rows = read_jsonl(own.prepared / "support.jsonl")
    template_rows = read_jsonl(own.prepared / "extension_action_templates.jsonl")
    dataset = dataset_for("query.jsonl")
    supports_dataset = dataset_for("support.jsonl")
    templates_dataset = dataset_for("extension_action_templates.jsonl")
    set_global_seed(int(args.seed))
    pipe = build_pipeline(args)
    ttt._freeze_pipe(pipe)
    table = json.loads((own.prepared / "reference/context_table.json").read_text())
    lookup = {
        float(row["friction_mu"]): torch.tensor(row["context"], device=pipe.device, dtype=torch.float32)
        for row in table["records"]
    }
    initial = torch.stack(list(lookup.values())).mean(0)
    contexts, results, extension_results = {}, [], []

    def prediction(sample, sample_index, z, destination, seed):
        rollout = prepare_sample_for_rollout(copy.copy(sample), sample_index, pipe, args)
        rollout.update(physical_context=z, output_path=str(destination))
        old_seed = args.seed
        args.seed = int(seed)
        set_global_seed(args.seed)
        try:
            _run_autoregressive(pipe, rollout, args)
        finally:
            args.seed = old_seed
        frames = imageio.mimread(str(destination))
        if len(frames) != int(args.num_frames):
            raise RuntimeError("Unexpected generated frame count")
        write_video(destination, frames, fps)

    for env in plan["environments"]:
        name = env["environment"]
        context_path = own.output / "contexts" / (name + ".json")
        if context_path.is_file():
            state = json.loads(context_path.read_text())
            if (state["model_step"] != setting["model_step"] or state["table_step"] != setting["table_step"]
                    or state["support_indices"] != env["support_indices"]):
                raise RuntimeError("Saved context does not match this checkpoint/support set")
            adapted = torch.tensor(state["context"], device=pipe.device, dtype=torch.float32)
        else:
            if any(support_rows[index]["dataset_split"] != "train"
                   or support_rows[index]["environment"] != name for index in env["support_indices"]):
                raise RuntimeError("Support split/environment leakage")
            support = [supports_dataset[index] for index in env["support_indices"]]
            set_global_seed(int(plan["protocol"]["seed"]) + int(env["environment_index"]) * 1009)
            adapted, losses, _, metrics, trajectory = ttt._adapt_ttt_state(
                pipe, support, args, adapter_named_params=[], base_adapter_state={},
                initial_context=initial, target_context=None,
                trajectory_meta={"environment": name, "sample_index": env["query_indices"][0],
                                 "friction_mu": float(env["environment_index"])},
            )
            state = {
                "context": adapted.detach().cpu().tolist(), "initial": initial.cpu().tolist(),
                "losses": losses, "metrics": metrics, "trajectory": trajectory,
                "support_indices": env["support_indices"],
                "model_step": setting["model_step"], "table_step": setting["table_step"],
            }
            write_json(context_path, state)
            del support
        contexts[name] = state
        columns = []
        for index in env["query_indices"]:
            row = query_rows[index]
            if row["episode_index"] in {support_rows[i]["episode_index"] for i in env["support_indices"]}:
                raise RuntimeError("Query episode overlaps support")
            key = f'q{index:04d}_{name}_{row["dataset_split"]}_L{int(row["action_level"]):02d}_ep{row["episode_index"]:06d}'
            paths = {kind: own.output / "raw" / kind / (key + ".mp4") for kind in ("gt", "stage1", "stage2")}
            marker = own.output / "completed" / (key + ".json")
            if not marker.is_file():
                sample = dataset[index]
                if (tuple(sample["video"].shape) != (1, 3, args.num_frames, args.height, args.width)
                        or tuple(np.shape(sample["action"])) != (1, args.num_frames, 14)):
                    raise RuntimeError("Training/inference tensor shape mismatch")
                paths["gt"].parent.mkdir(parents=True, exist_ok=True)
                save_video(sample["video"], output_path=str(paths["gt"]), fps=fps, quality=8)
                for kind, z in (("stage1", lookup[float(env["environment_index"])]), ("stage2", adapted)):
                    variant = own.output / "completed_variants" / (key + "_" + kind + ".json")
                    if not variant.exists():
                        seed = int(plan["protocol"]["seed"]) + index
                        prediction(sample, index, z, paths[kind], seed)
                        write_json(variant, {"key": key, "method": kind, "seed": seed})
                write_json(marker, {"index": index, "row": row,
                                    "paths": {kind: str(path) for kind, path in paths.items()}, "paired_gt": True})
                del sample
                torch.cuda.empty_cache()
            videos = {kind: imageio.mimread(str(path)) for kind, path in paths.items()}
            if any(len(frames) != args.num_frames for frames in videos.values()):
                raise RuntimeError("Comparison videos have unequal lengths")
            column = [
                np.vstack([labeled(videos[kind][frame],
                           f'{kind} | {row["dataset_split"]} L{row["action_level"]} ep{row["episode_index"]}')
                           for kind in ("gt", "stage1", "stage2")])
                for frame in range(args.num_frames)
            ]
            write_video(own.output / "comparisons" / (key + ".mp4"), column, fps)
            columns.append(column)
            results.append({"index": index, "environment": name, "split": row["dataset_split"],
                            "level": row["action_level"], "paths": {kind: str(path) for kind, path in paths.items()}})
            write_json(own.output / "progress.json", {
                "completed_contexts": list(contexts), "query_results": results,
                "extension_results": extension_results,
            })
            print("[query_done] " + key, flush=True)
        write_video(own.output / "grids" / (name + "_gt_stage1_stage2_train_test.mp4"),
                    (np.hstack([column[frame] for column in columns]) for frame in range(args.num_frames)), fps)
        pca_plot(own.output / "training_inference_Z_pca.svg", table, contexts,
                 [item for item in plan["environments"] if item["environment"] in contexts])

        # Separate extension: identical test initial frame, observed target templates.
        # There is NO matched future GT and no formal action-success score here.
        anchor_index = env["extension_anchor_query_index"]
        anchor = dataset[anchor_index]
        anchor_row = query_rows[anchor_index]
        extension_columns = []
        for template_index in env["extension_template_indices"]:
            row = template_rows[template_index]
            template = templates_dataset[template_index]
            if row["dataset_split"] != "train" or row["environment"] != name:
                raise RuntimeError("Counterfactual action template must come from this environment's train set")
            sample = copy.copy(anchor)
            sample["action"] = template["action"]
            key = f'{name}_anchorq{anchor_index:04d}_L{int(row["action_level"]):02d}'
            paths = {kind: own.output / "extensions/raw" / kind / (key + ".mp4")
                     for kind in ("stage1", "stage2")}
            marker = own.output / "extensions/completed" / (key + ".json")
            if not marker.exists():
                delta = torch.as_tensor(template["action"]).float().reshape(-1, 14)[0] - torch.as_tensor(
                    anchor["action"]).float().reshape(-1, 14)[0]
                for kind, z in (("stage1", lookup[float(env["environment_index"])]), ("stage2", adapted)):
                    variant = own.output / "extensions/completed_variants" / (key + "_" + kind + ".json")
                    if not variant.exists():
                        seed = int(plan["protocol"]["seed"]) + 100000 + anchor_index
                        prediction(sample, anchor_index, z, paths[kind], seed)
                        write_json(variant, {"key": key, "method": kind, "seed": seed})
                write_json(marker, {
                    "anchor_query": anchor_row, "action_template": row,
                    "initial_normalized_target_delta_l2": float(delta.norm().item()),
                    "paired_gt": False, "action_retiming": False, "action_interpolation": False,
                    "initial_frame_fixed": True, "formal_action_selection_metric": False,
                    "paths": {kind: str(path) for kind, path in paths.items()},
                })
            videos = {kind: imageio.mimread(str(path)) for kind, path in paths.items()}
            column = [
                np.vstack([labeled(videos[kind][frame], f'{kind} L{row["action_level"]} | counterfactual NO paired GT')
                           for kind in ("stage1", "stage2")])
                for frame in range(args.num_frames)
            ]
            write_video(own.output / "extensions/comparisons" / (key + ".mp4"), column, fps)
            extension_columns.append(column)
            extension_results.append({"environment": name, "level": row["action_level"], "paired_gt": False})
            print("[extension_done] " + key, flush=True)
        if extension_columns:
            write_video(own.output / "extensions/grids" / (name + "_fixed_test_initial_action_levels.mp4"),
                        (np.hstack([column[frame] for column in extension_columns])
                         for frame in range(args.num_frames)), fps)
        del anchor
        torch.cuda.empty_cache()
        write_json(own.output / "progress.json", {
            "completed_contexts": list(contexts), "query_results": results,
            "extension_results": extension_results,
        })
    write_json(own.output / "inference_complete.json", {
        "task": plan["task"], "model_step": setting["model_step"], "table_step": setting["table_step"],
        "queries": len(results), "environments": len(contexts), "extensions": len(extension_results),
        "unavailable_environments": plan["unavailable_environments"],
        "metric_status": "CPU_object_tracking_and_action_calibration_pending",
        "formal_metric_approved": False, "frames_per_second": fps,
    })
    print("[inference_complete]", own.output, flush=True)


if __name__ == "__main__":
    main()
