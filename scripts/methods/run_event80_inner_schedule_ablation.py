#!/usr/bin/env python3
"""Run one resumable Event80 inference-time LR schedule ablation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(*command) -> None:
    command = [str(value) for value in command]
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def stage_cache(config: dict) -> tuple[Path, Path]:
    required = ["diffusion_pytorch_model.safetensors.index.json", "Wan2.2_VAE.pth"]
    required += [
        f"diffusion_pytorch_model-{index:05d}-of-00003.safetensors"
        for index in range(1, 4)
    ]
    candidates = [
        Path("/tmp") / os.environ["USER"] / name
        for name in ("bwm_shared", "bwm_shared_cache")
    ]
    cache = next(
        (
            base
            for base in candidates
            if all((base / "Wan2.2-TI2V-5B" / name).is_file() for name in required)
        ),
        candidates[0],
    )
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "staging.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wan = cache / "Wan2.2-TI2V-5B"
        if not all((wan / name).is_file() and (wan / name).stat().st_size for name in required):
            wan.mkdir(parents=True, exist_ok=True)
            run("rsync", "-aL", "--partial", str(ROOT / "models/Wan2.2-TI2V-5B") + "/", str(wan) + "/")
        source = ROOT / config["checkpoint"]
        checkpoint = cache / "trained_ckpts/event80-ours-step7272.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file() or checkpoint.stat().st_size != source.stat().st_size:
            temporary = checkpoint.with_suffix(".partial")
            shutil.copy2(source, temporary)
            temporary.replace(checkpoint)
    print(f"[cache] Wan={wan} checkpoint={checkpoint}", flush=True)
    return wan, checkpoint


def quarantine_partial_videos(output: Path) -> None:
    for directory in (output / "support/raw", output / "transfer/raw"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.mp4"):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            try:
                streams = json.loads(probe.stdout).get("streams", [])
                valid = (
                    probe.returncode == 0
                    and streams
                    and int(streams[0].get("nb_frames", 0)) >= 41
                )
            except (ValueError, TypeError):
                valid = False
            if not valid:
                destination = output / "interrupted_videos" / path.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)


def build_active_table(config: dict, output: Path) -> tuple[Path, list[float], list[int]]:
    training = yaml.safe_load((ROOT / config["training_config"]).read_text())
    metadata = ROOT / training["dataset"]["dataset_metadata_path"]
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    manifest = yaml.safe_load((ROOT / config["method_manifest"]).read_text())
    active_ids = [int(value) for value in manifest["selection"]["active_environment_ids"]]
    mu_by_id = {int(row["mu_index"]): float(row["friction_mu"]) for row in rows}
    active_mus = [mu_by_id[index] for index in active_ids]
    table = json.loads((ROOT / config["context_table"]).read_text())
    selected = []
    for mu in active_mus:
        record = min(table["records"], key=lambda item: abs(float(item["friction_mu"]) - mu))
        if abs(float(record["friction_mu"]) - mu) > 6e-6:
            raise ValueError(f"Missing active friction {mu}")
        selected.append(record)
    active_table = output / "active_context_table.json"
    write_json(active_table, {**table, "num_groups": len(selected), "records": selected})
    return active_table, active_mus, active_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant-index", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "yejin-lo":
        raise RuntimeError("Run this evaluation on a yejin-lo compute node.")

    os.chdir(ROOT)
    config = yaml.safe_load(args.config.read_text())
    variants = config["variants"]
    if not 0 <= args.variant_index < len(variants):
        raise ValueError(f"variant-index must be in [0, {len(variants) - 1}]")
    variant = variants[args.variant_index]
    if sum(int(item.split(":")[1]) for item in variant["inner_lr_schedule"].split(",")) != int(variant["inner_steps"]):
        raise ValueError(f"Schedule length mismatch for {variant['name']}")

    benchmark = ROOT / config["benchmark_root"]
    output = (
        benchmark
        / "methods/ours_inner_schedule_ablation"
        / variant["name"]
        / f"step_{config['checkpoint_step']}"
        / f"seed_{config['seed']}"
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / config["protocol_path"]).read_text())
    environments = protocol["environments"]
    if len(environments) != 10 or sum(env["domain"] == "id" for env in environments) != 5:
        raise ValueError("Expected the frozen 5-ID/5-OOD Event80 protocol.")
    for env in environments:
        if len(env["support_indices"]) != 1 or len(env["query_indices"]) != 9:
            raise ValueError("Expected K=1 support and nine disjoint queries per environment.")
        if set(env["support_indices"]) & set(env["query_indices"]):
            raise ValueError("Support/query leakage in the frozen manifest.")

    active_table, active_mus, active_ids = build_active_table(config, output)
    training = yaml.safe_load((ROOT / config["training_config"]).read_text())
    metadata = ROOT / training["dataset"]["dataset_metadata_path"]
    dataset_root = Path(training["dataset"]["dataset_base_path"])
    supports = [int(env["support_indices"][0]) for env in environments]
    target_map = ";".join(
        f"{env['support_indices'][0]}:" + ",".join(map(str, env["query_indices"]))
        for env in environments
    )
    active_values = ",".join(f"{mu:.10g}" for mu in active_mus)
    write_json(
        output / "protocol.json",
        {
            "method": config["method"],
            "variant": variant,
            "checkpoint": config["checkpoint"],
            "checkpoint_step": config["checkpoint_step"],
            "context_table": config["context_table"],
            "active_environment_ids": active_ids,
            "active_environment_count": len(active_ids),
            "initial_context": "active-table mean",
            "context_dtype": "float32",
            "support_query_manifest": protocol,
            "model_loads_per_process": 1,
        },
    )

    if not (output / "rollouts.complete").is_file():
        wan, checkpoint = stage_cache(config)
        training["model"]["model_paths"] = str(wan)
        training["output"]["ckpt_path"] = None
        training["output"]["resume_from"] = None
        training["stage2"] = {
            "stage2_context_clamp_min": config["context_clamp_min"],
            "stage2_context_clamp_max": config["context_clamp_max"],
        }
        runtime = output / "inference_config.yaml"
        runtime.write_text(yaml.safe_dump(training, sort_keys=False))
        quarantine_partial_videos(output)
        run(
            sys.executable,
            "scripts/infer_stage2_ttt.py",
            "--config",
            runtime,
            "--stage2_ckpt_path",
            checkpoint,
            "--support_metadata_path",
            metadata,
            "--ttt_support_same_as_query",
            "--ttt_adapt_scope",
            "context",
            "--ttt_context_fp32",
            "--ttt_initial_context_table_path",
            active_table,
            "--ttt_context_table_path",
            active_table,
            "--ttt_context_active_values",
            active_values,
            "--ttt_context_trajectory_path",
            output / "context_trajectory.jsonl",
            "--ttt_context_pca_output_path",
            output / "context_trajectory_pca.svg",
            "--sample_indices",
            ",".join(map(str, supports)),
            "--output_path",
            output / "support/raw",
            "--comparison_output_path",
            output / "support/comparisons",
            "--ttt_transfer_targets_by_source",
            target_map,
            "--ttt_transfer_output_path",
            output / "transfer/raw",
            "--ttt_transfer_plan_output_path",
            output / "transfer/transfer_plan.json",
            "--ttt_transfer_skip_existing",
            "--resume_context_trajectory",
            "--skip_existing",
            "--stage2_inner_steps",
            variant["inner_steps"],
            "--stage2_inner_lr",
            float(variant["inner_lr_schedule"].split(":", 1)[0]),
            "--stage2_inner_lr_schedule",
            variant["inner_lr_schedule"],
            "--stage2_inner_grad_clip",
            config["inner_grad_clip"],
            "--stage2_context_reg_weight",
            config["context_reg_weight"],
            "--num_inference_steps",
            config["num_inference_steps"],
            "--cfg_scale",
            1,
            "--fps",
            config["fps"],
            "--quality",
            6,
            "--seed",
            config["seed"],
        )
        (output / "rollouts.complete").touch()

    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    wanted = {int(index) for env in environments for index in env["query_indices"]}
    found = set()
    for path in (output / "transfer/raw").rglob("sample*.mp4"):
        match = re.match(r"sample(\d+)_", path.name)
        if match and int(match.group(1)) in wanted:
            index = int(match.group(1))
            if index in found:
                raise ValueError(f"Duplicate query prediction {index}")
            found.add(index)
            destination = predictions / path.name
            if not destination.exists():
                destination.symlink_to(os.path.relpath(path, predictions))
    if found != wanted:
        raise RuntimeError(f"Missing query predictions: {sorted(wanted - found)}")

    if not (output / "grids.complete").is_file():
        run(
            sys.executable,
            "scripts/evaluation/compose_context_transfer_support_grids.py",
            "--metadata-path",
            metadata,
            "--dataset-root",
            dataset_root,
            "--transfer-plan",
            output / "transfer/transfer_plan.json",
            "--prediction-root",
            output / "transfer/raw",
            "--output-dir",
            output / "visualizations/grids",
            "--columns",
            5,
            "--support-size",
            1,
            "--width",
            224,
            "--height",
            224,
            "--fps",
            config["fps"],
            "--quality",
            6,
            "--prediction-label",
            variant["name"],
        )
        run(
            sys.executable,
            "scripts/analyze_context_endpoints_active_pca.py",
            "--table-path",
            active_table,
            "--trajectory-path",
            output / "context_trajectory.jsonl",
            "--active-frictions",
            active_values,
            "--id-indices",
            ",".join(str(env["support_indices"][0]) for env in environments if env["domain"] == "id"),
            "--ood-indices",
            ",".join(str(env["support_indices"][0]) for env in environments if env["domain"] == "ood"),
            "--output-svg",
            output / "context_endpoints_active_pca.svg",
            "--output-csv",
            output / "context_endpoints_active_pca.csv",
            "--title",
            f"Event80 {variant['name']}: active Z only",
        )
        (output / "grids.complete").touch()

    if not (output / "metrics.complete").is_file():
        metrics_root = output / "metrics/complete_v1"
        metrics = yaml.safe_load((ROOT / config["metric_template"]).read_text())
        metrics["output_root"] = str(metrics_root)
        metrics["methods"] = {variant["name"]: str(output.relative_to(benchmark))}
        metric_config = output / "metric_config.yaml"
        metric_config.write_text(yaml.safe_dump(metrics, sort_keys=False))
        run(sys.executable, "scripts/evaluation/evaluate_event80_benchmark.py", "--config", metric_config)
        (output / "metrics.complete").touch()

    (output / "inference.complete").touch()
    print(f"[done] {variant['name']} output={output}", flush=True)


if __name__ == "__main__":
    main()
