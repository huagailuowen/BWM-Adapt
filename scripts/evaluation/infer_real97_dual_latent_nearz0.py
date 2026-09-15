#!/usr/bin/env python3
"""Independent near-z0 ablation; retain all existing reference implementations."""

import argparse
import colorsys
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def copy_once(source, destination):
    if destination.exists():
        return
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Required completed Stage1 artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def prepare(config_path, config, task):
    setting = config["tasks"][task]
    source = ROOT / setting["source_prepared"]
    target = ROOT / setting["prepared"]
    fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    ready = target / "prepared_complete.json"
    if ready.exists():
        if read(ready)["config_sha256"] != fingerprint:
            raise RuntimeError("Prepared reference belongs to a different configuration")
        print(json.dumps({"task": task, "prepared": str(target), "reused": True}), flush=True)
        return
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite an incomplete prepared directory: {target}")
    stage1_source = ROOT / setting["source_inference"]
    completion = read(stage1_source / "dual_latent_inference_complete.json")
    if (completion["model_step"], completion["table_step"], completion["stage1_replica"]) != (5500, 5500, 0):
        raise RuntimeError("Stage1 reuse requires the completed step5500 z0 reference")
    temporary = target.with_name(target.name + f".partial-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    (temporary / "reference").mkdir()
    manifest_hashes = {}
    for name in ("query.jsonl", "support.jsonl", "extension_action_templates.jsonl", "stage_files.txt"):
        content = (source / name).read_bytes()
        (temporary / name).write_bytes(content)
        manifest_hashes[name] = hashlib.sha256(content).hexdigest()
    for name in ("context_table.json", "full_training_context_table.json", "latent_aliases.json",
                 "training_config.yaml", "action_stats.json"):
        shutil.copy2(source / "reference" / name, temporary / "reference" / name)
    os.link(source / "reference/model.safetensors", temporary / "reference/model.safetensors")
    table = read(temporary / "reference/context_table.json")
    if len(table["records"]) != setting["expected_latents"]:
        raise RuntimeError("Unexpected number of dual-latent table rows")
    plan = read(source / "plan.json")
    for environment in plan["environments"]:
        selected = [row for row in table["records"]
                    if float(row["friction_mu"]) == float(environment["environment_index"])]
        if len(selected) != 1 or selected[0]["replica"] != 0 or not selected[0]["stage1_selected"]:
            raise RuntimeError("Physical environment lookup must resolve to its selected z0")
    if plan["protocol"]["ttt"]["stage2_inner_lr_schedule"] != config["stage2"]["inner_lr_schedule"]:
        raise RuntimeError("Support adaptation schedule differs from the frozen comparison")
    plan.pop("stage2_mean_latent_count", None)
    plan.update(prepared=str(target), source_reference=str(source),
                stage2_initial=config["stage2_initialization"],
                stage2_initialization_policy=setting["initial_context"],
                stage2_initialization_noise={"min": config["noise_min"], "max": config["noise_max"],
                                             "seed": config["noise_seed"], "per_coordinate": True},
                known_environment_initialization=True, effective_context_bounds=None,
                original_source_outputs_modified=False, source_manifest_sha256=manifest_hashes,
                reused_stage1_source=str(stage1_source), node_cache_reference=str(source))
    write(temporary / "plan.json", plan)
    write(temporary / "source_plan.json", read(source / "plan.json"))
    write(temporary / "experiment.json", plan["experiment"])
    shutil.copy2(config_path, temporary / "evaluation_config.json")
    record = {"task": task, "config_sha256": fingerprint, "model_step": 5500, "table_step": 5500,
              "initial_context": setting["initial_context"], "manifest_sha256": manifest_hashes,
              "source_prepared": str(source), "source_inference": str(stage1_source),
              "model_reference_is_hard_link": True, "source_outputs_modified": False}
    write(temporary / "prepared_complete.json", record)
    os.rename(temporary, target)
    print(json.dumps({"prepared": str(target), **record}), flush=True)


def reuse_stage1(output, source, plan, queries, templates, config_sha):
    from scripts.evaluation.prepare_real97_support_override import query_key
    marker = output / "stage1_reuse_complete.json"
    if marker.exists():
        previous = read(marker)
        if previous["config_sha256"] != config_sha or previous["source"] != str(source):
            raise RuntimeError("Output belongs to another initialization experiment")
        return
    keys = [query_key(index, row) for index, row in enumerate(queries)]
    for key in keys:
        for relative in (f"raw/stage1/{key}.mp4", f"completed_variants/{key}_stage1.json"):
            copy_once(source / relative, output / relative)
    extension_count = 0
    for environment in plan["environments"]:
        for index in environment["extension_template_indices"]:
            level = int(templates[index]["action_level"])
            key = f'{environment["environment"]}_anchorq{environment["extension_anchor_query_index"]:04d}_L{level:02d}'
            for relative in (f"extensions/raw/stage1/{key}.mp4",
                             f"extensions/completed_variants/{key}_stage1.json"):
                copy_once(source / relative, output / relative)
            extension_count += 1
    write(marker, {"source": str(source), "config_sha256": config_sha, "replica": 0,
                   "factual_stage1_videos": len(keys), "extension_stage1_videos": extension_count,
                   "stage2_contexts_or_videos_copied": False, "source_outputs_modified": False})


def pca_plotter(output, environments):
    import numpy as np

    def plot(path, table, contexts, completed):
        records = table["records"]
        values = np.asarray([row["context"] for row in records], dtype=float).reshape(len(records), -1)
        center = values.mean(0)
        _, singular, axes = np.linalg.svd(values - center, full_matrices=False)
        training = (values - center) @ axes[:2].T
        starts, ends = {}, {}
        for environment in completed:
            name = environment["environment"]
            initial = read(output / "initializations" / (name + ".json"))
            starts[name] = (np.asarray(initial["initial_context"]).reshape(-1) - center) @ axes[:2].T
            ends[name] = (np.asarray(contexts[name]["context"]).reshape(-1) - center) @ axes[:2].T
        points = np.vstack([training] + [point.reshape(1, 2) for point in [*starts.values(), *ends.values()]])
        lower = points.min(0)
        span = np.maximum(points.max(0) - lower, 1e-6)
        lower -= .06 * span
        span *= 1.12
        def xy(point):
            return 60 + 690 * (point[0] - lower[0]) / span[0], 535 - 450 * (point[1] - lower[1]) / span[1]
        colors = {environment["environment"]: "#" + "".join(f"{int(v * 255):02x}" for v in
                  colorsys.hsv_to_rgb(i / len(environments), .7, .8))
                  for i, environment in enumerate(environments)}
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="630">',
                 '<rect width="1120" height="630" fill="white"/>',
                 '<text x="40" y="30" font-family="sans-serif" font-size="18">Dual Z: inference initialized near the corresponding training-time z0</text>']
        for row, point in zip(records, training):
            x, y = xy(point)
            radius = 6 if row["stage1_selected"] else 4
            title = html.escape(f'{row["environment"]} z{row["replica"]}')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{colors[row["environment"]]}" stroke="black" stroke-width="0.7"><title>{title}</title></circle>')
        for name, point in starts.items():
            x, y = xy(point)
            parts.append(f'<path d="M{x-5},{y}H{x+5} M{x},{y-5}V{y+5}" stroke="{colors[name]}" stroke-width="2"><title>{html.escape(name)} actual noisy initial Z</title></path>')
        for name, point in ends.items():
            x, y = xy(point)
            parts.append(f'<polygon points="{x},{y-7} {x-6},{y+5} {x+6},{y+5}" fill="{colors[name]}" stroke="black"><title>{html.escape(name)} inference-time Z</title></polygon>')
        for i, environment in enumerate(environments):
            name = environment["environment"]
            parts.append(f'<text x="785" y="{65+i*27}" fill="{colors[name]}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
        total = max(float(np.sum(singular ** 2)), 1e-12)
        parts.append(f'<text x="55" y="581" font-family="sans-serif" font-size="13">PC1 {100*singular[0]**2/total:.1f}%; PC2 {100*singular[1]**2/total:.1f}%; same all-table PCA plane as the mean-initialization run.</text>')
        parts.append('<text x="55" y="603" font-family="sans-serif" font-size="12">Circles: training time (large = z0). Crosses: actual noisy starts. Black-bordered triangles: inference time.</text></svg>')
        path.write_text("\n".join(parts))
    return plot


def infer(config_path, config, task, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference and video processing require a compute allocation")
    import fcntl
    setting = config["tasks"][task]
    prepared = ROOT / setting["prepared"]
    source = ROOT / setting["source_prepared"]
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if read(prepared / "prepared_complete.json")["config_sha256"] != config_sha:
        raise RuntimeError("The prepared configuration fingerprint differs")
    output.mkdir(parents=True, exist_ok=True)
    run_lock = (output / ".near_z0.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(prepared / "plan.json")
    queries = rows(prepared / "query.jsonl")
    templates = rows(prepared / "extension_action_templates.jsonl")
    reuse_stage1(output, ROOT / setting["source_inference"], plan, queries, templates, config_sha)
    shutil.copy2(config_path, output / "evaluation_config.json")
    from scripts.evaluation import infer_real97_ball_door_reference as reference
    from scripts.evaluation import infer_real97_unbounded_context as unbounded
    from scripts.evaluation.prepare_real97_support_override import render_grid, query_key
    original_stage = reference.stage
    def stage_with_shared_cache(requested, destination):
        if requested.resolve() != prepared.resolve():
            raise RuntimeError("Unexpected prepared reference")
        # Data, checkpoint, normalization, and update schedule are identical.
        # The only change is the initialization policy installed by unbounded.
        _, runtime, checkpoint = original_stage(source, destination)
        return plan, runtime, checkpoint
    reference.stage = stage_with_shared_cache
    reference.pca_plot = pca_plotter(output, plan["environments"])
    sys.argv = ["infer_real97_unbounded_context", "--config", str(config_path), "--task", task,
                "--prepared", str(prepared), "--output", str(output)]
    unbounded.main()
    groups = {}
    for index, row in enumerate(queries):
        groups.setdefault((row["environment"], row["dataset_split"]), []).append((index, row))
    grid_order = []
    for (environment, split), items in sorted(groups.items()):
        items.sort(key=lambda item: (int(item[1]["action_level"]), int(item[1]["episode_index"]), item[0]))
        folder = output / "grids" if split == "test" else output / "grids/train"
        destination = folder / f"{environment}_gt_stage1_stage2_{split}_levels.mp4"
        render_grid([output / "comparisons" / (query_key(index, row) + ".mp4") for index, row in items], destination)
        grid_order.append({"environment": environment, "split": split, "video": str(destination),
                           "columns": [{"level": row["action_level"], "episode_index": row["episode_index"],
                                        "query_index": index} for index, row in items]})
    for path in (output / "grids").glob("*_gt_stage1_stage2_train_test.mp4"):
        destination = output / "grids/previous_mixed" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    write(output / "grids/level_order.json", {"row_order": ["GT", "Stage1", "Stage2"], "grids": grid_order})
    write(output / "dual_latent_nearz0_inference_complete.json", {
        "task": task, "model_step": 5500, "table_step": 5500, "stage1_replica": 0,
        "stage1_reused": True, "stage2_initialization": setting["initial_context"],
        "noise_min": config["noise_min"], "noise_max": config["noise_max"], "noise_seed": config["noise_seed"],
        "effective_context_bounds": None, "exact_original_support_and_queries": True,
        "level_sorted_split_grids_complete": True, "known_environment_initialization": True,
        "formal_metric_approved": False, "source_outputs_modified": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["door", "ball"], required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read(config_path)
    if args.prepare_only:
        prepare(config_path, config, args.task)
    else:
        if args.output is None:
            parser.error("--output is required for inference")
        infer(config_path, config, args.task, args.output.resolve())
