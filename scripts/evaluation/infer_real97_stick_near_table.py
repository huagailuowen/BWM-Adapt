#!/usr/bin/env python3
"""Opt-in Stick reference rerun from each training Z plus reproducible noise."""

import argparse
import colorsys
import copy
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
POLICY = "corresponding_environment_table_plus_uniform_noise"


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def publish_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_file(source, destination):
    source, destination = Path(source), Path(destination)
    content = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError(f"Frozen input changed: {source}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return hashlib.sha256(content).hexdigest()


def copy_reference_videos(source, output):
    copied = 0
    for kind in ("gt", "stage1"):
        parent = source / "raw" / kind
        if not parent.is_dir():
            raise RuntimeError(f"Original {kind} video directory is missing: {parent}")
        for original in sorted(parent.glob("*.mp4")):
            destination = output / "raw" / kind / original.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_name(destination.name + ".partial")
                shutil.copy2(original, temporary)
                temporary.replace(destination)
                copied += 1
    return copied


def plot_contexts(output, environments, table, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    names = [env["environment"] for env in environments]
    training = np.stack([np.asarray(table[name], dtype=np.float64).reshape(-1) for name in names])
    center = training.mean(axis=0)
    _, singular, basis = np.linalg.svd(training - center, full_matrices=False)
    axes = basis[:2].T
    variance = singular ** 2 / max(float(np.sum(singular ** 2)), 1e-30)
    training_xy = (training - center) @ axes
    initial = np.stack([np.asarray(records[name]["initial"], dtype=np.float64).reshape(-1) for name in names])
    adapted = np.stack([np.asarray(records[name]["final"], dtype=np.float64).reshape(-1) for name in names])
    initial_xy, adapted_xy = (initial - center) @ axes, (adapted - center) @ axes
    colors = [colorsys.hsv_to_rgb(index / len(names), 0.7, 0.65) for index in range(len(names))]
    for zoom, filename in ((False, "training_inference_Z_pca.svg"),
                           (True, "training_inference_Z_pca_inference_zoom.svg")):
        fig, ax = plt.subplots(figsize=(10, 7))
        for index, name in enumerate(names):
            if not zoom:
                ax.scatter(*training_xy[index], c=[colors[index]], marker="o", s=80,
                           edgecolors="white", linewidths=0.7, zorder=3)
                ax.scatter(*initial_xy[index], c=[colors[index]], marker="x", s=50,
                           linewidths=1.4, zorder=3)
            ax.scatter(*adapted_xy[index], c=[colors[index]], marker="^", s=100,
                       edgecolors="black", linewidths=0.9, zorder=4)
        handles = [Line2D([], [], marker="^", color="none", markerfacecolor=colors[i],
                          markeredgecolor="black", label=name.removeprefix("stick-"))
                   for i, name in enumerate(names)]
        legend = ax.legend(handles=handles, title="Environment", loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.add_artist(legend)
        kinds = [Line2D([], [], marker="^", color="none", markerfacecolor="gray", markeredgecolor="black", label="Inference time")]
        if not zoom:
            kinds = [Line2D([], [], marker="o", color="none", markerfacecolor="gray", label="Training time"),
                     Line2D([], [], marker="x", color="gray", linestyle="none", label="Initialization")] + kinds
        ax.legend(handles=kinds, loc="lower left", bbox_to_anchor=(1.01, 0))
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
        ax.set_title("Stick: inference region, unchanged training PCA plane" if zoom else "Stick: environment Z + noise initialization")
        ax.grid(alpha=0.2)
        ax.margins(0.15)
        fig.tight_layout()
        temporary = output / (filename + ".partial")
        fig.savefig(temporary, format="svg", bbox_inches="tight")
        plt.close(fig)
        temporary.replace(output / filename)
    publish_json(output / "training_inference_Z_pca_projection.json", {
        "fit_on": "training_table_only", "center": center.tolist(), "components": axes.T.tolist(),
        "explained_variance_ratio": variance[:2].tolist(), "environment_order": names,
        "training_xy": training_xy.tolist(), "initial_xy": initial_xy.tolist(), "adapted_xy": adapted_xy.tolist(),
        "same_basis_for_global_and_zoom": True,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    own = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference must run in a compute allocation")
    config = json.loads(own.config.read_text())
    prepared, source, output = resolve(config["prepared"]), resolve(config["source_inference"]), own.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise RuntimeError("The new output must be separate from the original inference")
    if config["initialization"] != POLICY or not config["disable_context_clamp"]:
        raise RuntimeError("This opt-in entrypoint requires near-table unbounded initialization")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {"experiment": freeze_file(own.config, output / "input_manifest/experiment.json")}
    for relative in ("plan.json", "query.jsonl", "support.jsonl", "stage_files.txt",
                     "reference/training_config.yaml", "reference/action_stats.json", "reference/context_table.json",
                     "reference/provenance.json"):
        hashes[relative] = freeze_file(prepared / relative, output / "input_manifest" / relative)
    plan = json.loads((prepared / "plan.json").read_text())
    support = [json.loads(line) for line in (prepared / "support.jsonl").read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in (prepared / "query.jsonl").read_text().splitlines() if line.strip()]
    environments = plan["environments"]
    if plan["task"] != "stick" or len(environments) != config["expected_environments"] or len(queries) != config["expected_queries"]:
        raise RuntimeError("Source protocol is not the expected 8-environment, 40-query Stick reference")
    for env in environments:
        name = env["environment"]
        selected = [support[index] for index in env["support_indices"]]
        query_episodes = {row["episode_index"] for row in queries if row["environment"] == name}
        if len(selected) != 2 or any(row["environment"] != name or row["dataset_split"] != "train" or
                                    row["episode_index"] in query_episodes for row in selected):
            raise RuntimeError(f"Source Support identity/leakage error: {name}")
    print(f"[reuse] copied {copy_reference_videos(source, output)} original GT/Stage1 videos", flush=True)
    publish_json(output / "near_table_protocol.json", {
        "config": config, "input_sha256": hashes, "source_output": str(source),
        "query_and_support_unchanged": True, "effective_context_bounds": None,
        "known_environment_initialization": True, "source_output_modified": False,
    })

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one usable allocated GPU is required")
    sys.path.insert(0, str(ROOT / "scripts"))
    ttt = importlib.import_module("infer_stage2_ttt")
    sys.modules.setdefault("scripts.infer_stage2_ttt", ttt)
    table_data = json.loads((prepared / "reference/context_table.json").read_text())
    indexed = {float(record["friction_mu"]): record["context"] for record in table_data["records"]}
    table = {env["environment"]: indexed[float(env["environment_index"])] for env in environments}
    records = {}
    for name in table:
        path = output / "initializations" / (name + ".json")
        if path.exists():
            records[name] = json.loads(path.read_text())
    original_adapt = ttt._adapt_ttt_state

    @functools.wraps(original_adapt)
    def adapt(pipe, support_items, args, **kwargs):
        meta = dict(kwargs.get("trajectory_meta") or {})
        name = meta.get("environment")
        if name not in table and kwargs.get("target_context") is not None:
            target = torch.as_tensor(kwargs["target_context"]).detach().float().cpu().reshape(-1)
            matches = [key for key, values in table.items()
                       if torch.allclose(target, torch.as_tensor(values, dtype=torch.float32).reshape(-1), atol=1e-6, rtol=0)]
            if len(matches) == 1:
                name = matches[0]
        if name not in table:
            raise RuntimeError("Cannot unambiguously identify the adaptation environment")
        if (args.num_frames, args.frame_stride, args.height, args.width, args.action_type, args.physical_context_dim) != (41, 3, 192, 256, "eef_target", 32):
            raise RuntimeError("Stick geometry or target-EEF conditioning differs from the reference")
        local = copy.copy(args)
        if float(local.stage2_context_reg_weight) != 0:
            raise RuntimeError("The reference must use Support loss without Z regularization")
        local.stage2_context_clamp_min = -math.inf
        local.stage2_context_clamp_max = math.inf
        local.ttt_context_fp32 = True
        target = torch.as_tensor(table[name], dtype=torch.float32).reshape(1, 32)
        stable_id = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        seed = (int(config["noise_seed"]) + stable_id) % (2 ** 63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.rand(target.shape, dtype=torch.float32, generator=generator)
        noise = noise * (float(config["noise_max"]) - float(config["noise_min"])) + float(config["noise_min"])
        initial = target + noise
        meta.update(environment=name, initialization_policy=POLICY, initialization_seed=seed,
                    known_environment_initialization=True, effective_context_bounds=None)
        kwargs.update(initial_context=initial.to(pipe.device), target_context=target.to(pipe.device), trajectory_meta=meta)
        record = {"environment": name, "policy": POLICY, "seed": seed, "target": target.tolist(),
                  "noise": noise.tolist(), "initial": initial.tolist(), "final": None,
                  "initial_to_table_l2": float(torch.linalg.vector_norm(noise)),
                  "initial_coordinates_outside_old_bounds": int((initial.abs() > 1).sum()),
                  "effective_context_bounds": None}
        records[name] = record
        publish_json(output / "initializations" / (name + ".json"), record)
        print(f"[adapt_start] {name} initial_to_table_l2={record['initial_to_table_l2']:.6f} seed={seed}", flush=True)
        result = original_adapt(pipe, support_items, local, **kwargs)
        final = result[0].detach().float().cpu().reshape(1, 32)
        if not torch.isfinite(final).all():
            raise RuntimeError(f"Nonfinite adapted Z: {name}")
        record.update(final=final.tolist(), final_to_table_l2=float(torch.linalg.vector_norm(final - target)),
                      displacement_from_initial_l2=float(torch.linalg.vector_norm(final - initial)))
        publish_json(output / "initializations" / (name + ".json"), record)
        print(f"[adapt_done] {name} displacement={record['displacement_from_initial_l2']:.6f}", flush=True)
        return result

    ttt._adapt_ttt_state = adapt
    infer_module = importlib.import_module("infer")
    original_rollout = infer_module._run_autoregressive

    @functools.wraps(original_rollout)
    def reuse_stage1(pipe, rollout, args, *extra, **kwargs):
        destination = Path(rollout["output_path"]).resolve()
        if destination.parent == output / "raw/stage1" and destination.is_file() and destination.stat().st_size > 0:
            print(f"[reuse_stage1] {destination.name}", flush=True)
            return None
        return original_rollout(pipe, rollout, args, *extra, **kwargs)

    infer_module._run_autoregressive = reuse_stage1
    reference = importlib.import_module("infer_real97_reference_trial")
    if getattr(reference, "_run_autoregressive", None) is original_rollout:
        reference._run_autoregressive = reuse_stage1
    original_write_json = reference.write_json

    def record_actual_initial(path, data):
        path = Path(path)
        if path.parent == output / "contexts" and path.stem in records and isinstance(data, dict):
            data.update(initial=records[path.stem]["initial"], initialization_policy=POLICY,
                        initialization_seed=records[path.stem]["seed"], effective_context_bounds=None)
        if path == output / "provenance.json" and isinstance(data, dict):
            data.update(initialization_policy=POLICY, known_environment_initialization=True, effective_context_bounds=None)
        return original_write_json(path, data)

    reference.write_json = record_actual_initial
    sys.argv = ["infer_real97_reference_trial", "--prepared", str(prepared), "--output", str(output)]
    reference.main()
    if any(name not in records or records[name].get("final") is None for name in table):
        raise RuntimeError("An environment is missing its completed near-table adaptation record")
    plot_contexts(output, environments, table, records)
    publish_json(output / "initialization_distances.json", records)
    publish_json(output / "near_table_inference_complete.json", {
        "task": "stick", "actual_slurm_job_id": os.environ["SLURM_JOB_ID"],
        "environments": len(environments), "queries": len(queries), "initialization": POLICY,
        "noise_range": [config["noise_min"], config["noise_max"]], "effective_context_bounds": None,
        "source_output": str(source), "query_and_support_unchanged": True,
        "known_environment_initialization": True, "formal_metric_approved": False,
    })
    print(f"[near_table_complete] {output}", flush=True)


if __name__ == "__main__":
    main()
