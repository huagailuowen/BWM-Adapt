#!/usr/bin/env python3
"""Resumable formal Event80 evaluation after an iterative representation run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
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


def complete_checkpoint(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_size < 256 * 1024 * 1024:
                return False
            header = json.loads(handle.read(header_size))
        end = max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")
        return path.stat().st_size == 8 + header_size + end
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def immutable_copy(source: Path, destination: Path, *, hardlink=False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    if hardlink:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
    else:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def pin_checkpoint(config: dict, job_root: Path) -> dict:
    pointer = job_root / "pinned_checkpoint.json"
    if pointer.is_file():
        pinned = json.loads(pointer.read_text())
        if not complete_checkpoint(Path(pinned["checkpoint"])):
            raise RuntimeError("Pinned checkpoint is missing/incomplete; do not silently change the evaluated step.")
        return pinned
    checkpoint_dir = ROOT / config["checkpoint_dir"]
    candidates = []
    for path in checkpoint_dir.glob("step-*.safetensors"):
        match = re.fullmatch(r"step-(\d+)\.safetensors", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for step, path in sorted(candidates, reverse=True):
        table_path = path.with_suffix(".context_table.json")
        if not table_path.is_file() or not complete_checkpoint(path):
            continue
        try:
            table = json.loads(table_path.read_text())
        except (OSError, ValueError):
            continue
        records = table.get("records", [])
        if not records:
            continue
        archive = ROOT / "outputs/evaluation_checkpoints" / f"event80_{config['method']}_train{config['training_job_id']}"
        checkpoint = archive / path.name
        archived_table = archive / table_path.name
        immutable_copy(path, checkpoint, hardlink=True)
        immutable_copy(table_path, archived_table)
        pinned = {"checkpoint_step": step, "checkpoint": str(checkpoint), "context_table": str(archived_table),
                  "source_checkpoint": str(path), "source_context_table": str(table_path)}
        write_json(pointer, pinned)
        return pinned
    raise RuntimeError(f"No complete checkpoint with a matching context table in {checkpoint_dir}")


def stage_cache(pinned: dict, config: dict) -> tuple[Path, Path]:
    required = ["diffusion_pytorch_model.safetensors.index.json", "Wan2.2_VAE.pth"]
    required += [f"diffusion_pytorch_model-{index:05d}-of-00003.safetensors" for index in range(1, 4)]
    candidates = [Path("/tmp") / os.environ["USER"] / name for name in ("bwm_shared", "bwm_shared_cache")]
    cache = next((base for base in candidates if all((base / "Wan2.2-TI2V-5B" / name).is_file() for name in required)), candidates[0])
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "staging.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wan = cache / "Wan2.2-TI2V-5B"
        if not all((wan / name).is_file() and (wan / name).stat().st_size for name in required):
            wan.mkdir(parents=True, exist_ok=True)
            run("rsync", "-aL", "--partial", str(ROOT / "models/Wan2.2-TI2V-5B") + "/", str(wan) + "/")
        checkpoint = cache / "trained_ckpts" / f"event80-{config['method']}-train{config['training_job_id']}-step{pinned['checkpoint_step']}.safetensors"
        if not complete_checkpoint(checkpoint):
            immutable_copy(Path(pinned["checkpoint"]), checkpoint)
        print(f"[cache] Wan={wan} checkpoint={checkpoint}", flush=True)
    return wan, checkpoint


def quarantine_partial_videos(output: Path) -> None:
    # Only our generated predictions are inspected; the dataset is never touched.
    for directory in (output / "support/raw", output / "transfer/raw"):
        for path in directory.rglob("*.mp4"):
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=nb_frames", "-of", "json", str(path)], capture_output=True, text=True)
            try:
                streams = json.loads(probe.stdout).get("streams", [])
                valid = probe.returncode == 0 and streams and int(streams[0].get("nb_frames", 0)) >= 41
            except (ValueError, TypeError):
                valid = False
            if not valid:
                destination = output / "interrupted_videos" / path.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "yejin-lo":
        raise RuntimeError("Run inference, plots, and metrics on a yejin-lo compute node.")
    os.chdir(ROOT)
    config = yaml.safe_load(args.config.read_text())
    benchmark = ROOT / config["benchmark_root"]
    job_root = benchmark / "methods" / config["method"] / f"train_job_{config['training_job_id']}"
    job_root.mkdir(parents=True, exist_ok=True)
    with (job_root / "evaluation.lock").open("a") as evaluation_lock:
        fcntl.flock(evaluation_lock, fcntl.LOCK_EX)
        pinned = pin_checkpoint(config, job_root)
        output = job_root / f"step_{pinned['checkpoint_step']}" / f"seed_{config['seed']}"
        output.mkdir(parents=True, exist_ok=True)
        training = yaml.safe_load((ROOT / config["training_config"]).read_text())
        physical = training["physical_context"]
        if int(physical["physical_context_dim"]) != int(config["context_dim"]) or physical["physical_context_projection"] != config["context_projection"]:
            raise ValueError("Inference representation differs from the actual training configuration.")
        metadata = ROOT / training["dataset"]["dataset_metadata_path"]
        dataset_root = Path(training["dataset"]["dataset_base_path"])
        rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
        manifest = json.loads((ROOT / config["protocol_path"]).read_text())
        environments = manifest["environments"]
        if len(environments) != 10 or sum(env["domain"] == "id" for env in environments) != 5:
            raise ValueError("Expected the frozen 5-ID/5-OOD Event80 protocol.")
        for env in environments:
            if len(env["support_indices"]) != 1 or len(env["query_indices"]) != 9 or set(env["support_indices"]) & set(env["query_indices"]):
                raise ValueError("Expected one support and nine disjoint query trajectories per environment.")
        pool = json.loads((ROOT / config["checkpoint_dir"] / "active_pool_manifest.json").read_text())
        group = training["grouped_context_stage1"]
        initial = int(group["grouped_context_curriculum_initial_groups"])
        initial_steps = int(group["grouped_context_curriculum_initial_model_steps"])
        cycle = int(group["grouped_context_curriculum_new_context_steps"]) + 2 * (
            int(group["grouped_context_curriculum_all_context_steps"]) + int(group["grouped_context_curriculum_model_steps"]))
        wave = max(0, (int(pinned["checkpoint_step"]) - initial_steps - 1) // cycle)
        count = min(len(pool["curriculum_group_order"]), initial + wave * int(group["grouped_context_curriculum_add_groups"]))
        active_ids = pool["curriculum_group_order"][:count]
        mu_by_id = {int(row["mu_index"]): float(row["friction_mu"]) for row in rows}
        active_mus = [mu_by_id[index] for index in active_ids]
        table = json.loads(Path(pinned["context_table"]).read_text())
        selected = []
        for mu in active_mus:
            record = min(table["records"], key=lambda item: abs(float(item["friction_mu"]) - mu))
            if abs(float(record["friction_mu"]) - mu) > 6e-6:
                raise ValueError(f"Missing active friction {mu}")
            selected.append(record)
        active_table = output / "active_context_table.json"
        write_json(active_table, {**table, "num_groups": len(selected), "records": selected})
        supports = [int(env["support_indices"][0]) for env in environments]
        target_map = ";".join(f"{env['support_indices'][0]}:" + ",".join(map(str, env["query_indices"])) for env in environments)
        active_values = ",".join(f"{mu:.10g}" for mu in active_mus)
        provenance = {**config, **pinned, "actual_active_environment_ids": active_ids,
                      "actual_active_environment_count": count, "intended_active_environment_ids": pool["active_environment_ids"],
                      "query_count": 90, "support_size": 1, "initial_context": "active-table mean",
                      "context_dtype": "float32", "model_loads_per_inference_process": 1,
                      "support_query_manifest": manifest}
        write_json(output / "protocol.json", provenance)
        if not (output / "rollouts.complete").is_file():
            wan, checkpoint = stage_cache(pinned, config)
            # Load the trained checkpoint once in infer_stage2_ttt, after building Wan.
            training["model"]["model_paths"] = str(wan)
            training["output"]["ckpt_path"] = None
            training["output"]["resume_from"] = None
            training["stage2"] = {"stage2_context_clamp_min": config["context_clamp_min"],
                                  "stage2_context_clamp_max": config["context_clamp_max"]}
            runtime = output / "inference_config.yaml"
            runtime.write_text(yaml.safe_dump(training, sort_keys=False))
            quarantine_partial_videos(output)
            run(sys.executable, "scripts/infer_stage2_ttt.py", "--config", runtime,
                "--stage2_ckpt_path", checkpoint, "--support_metadata_path", metadata,
                "--ttt_support_same_as_query", "--ttt_adapt_scope", "context", "--ttt_context_fp32",
                "--ttt_initial_context_table_path", active_table, "--ttt_context_table_path", active_table,
                "--ttt_context_active_values", active_values,
                "--ttt_context_trajectory_path", output / "context_trajectory.jsonl",
                "--ttt_context_pca_output_path", output / "context_trajectory_pca.svg",
                "--sample_indices", ",".join(map(str, supports)), "--output_path", output / "support/raw",
                "--comparison_output_path", output / "support/comparisons",
                "--ttt_transfer_targets_by_source", target_map,
                "--ttt_transfer_output_path", output / "transfer/raw",
                "--ttt_transfer_plan_output_path", output / "transfer/transfer_plan.json",
                "--ttt_transfer_skip_existing", "--resume_context_trajectory", "--skip_existing",
                "--stage2_inner_steps", config["inner_steps"], "--stage2_inner_lr", 3.0,
                "--stage2_inner_lr_schedule", config["inner_lr_schedule"],
                "--stage2_inner_grad_clip", config["inner_grad_clip"], "--stage2_context_reg_weight", config["context_reg_weight"],
                "--num_inference_steps", config["num_inference_steps"], "--cfg_scale", 1,
                "--fps", config["fps"], "--quality", 6, "--seed", config["seed"])
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
            run(sys.executable, "scripts/evaluation/compose_context_transfer_support_grids.py",
                "--metadata-path", metadata, "--dataset-root", dataset_root,
                "--transfer-plan", output / "transfer/transfer_plan.json", "--prediction-root", output / "transfer/raw",
                "--output-dir", output / "grids_support_plus_queries", "--columns", 5, "--support-size", 1,
                "--width", 224, "--height", 224, "--fps", config["fps"], "--quality", 6,
                "--prediction-label", config["method"])
            run(sys.executable, "scripts/analyze_context_endpoints_active_pca.py", "--table-path", active_table,
                "--trajectory-path", output / "context_trajectory.jsonl", "--active-frictions", active_values,
                "--id-indices", ",".join(str(env["support_indices"][0]) for env in environments if env["domain"] == "id"),
                "--ood-indices", ",".join(str(env["support_indices"][0]) for env in environments if env["domain"] == "ood"),
                "--output-svg", output / "context_endpoints_active_pca.svg",
                "--output-csv", output / "context_endpoints_active_pca.csv", "--title", config["method"] + ": active Z only")
            (output / "grids.complete").touch()
        metrics_root = output / "metrics/complete_v1"
        if not (output / "metrics.complete").is_file():
            metrics = yaml.safe_load((ROOT / config["metric_template"]).read_text())
            metrics["output_root"] = str(metrics_root)
            metrics["methods"] = {config["method"]: str(output.relative_to(benchmark))}
            metric_config = output / "metric_config.yaml"
            metric_config.write_text(yaml.safe_dump(metrics, sort_keys=False))
            run(sys.executable, "scripts/evaluation/evaluate_event80_benchmark.py", "--config", metric_config)
            (output / "metrics.complete").touch()
        write_json(job_root / "evaluation_result.json", {
            "method": config["method"], "training_job_id": config["training_job_id"],
            "checkpoint_step": pinned["checkpoint_step"], "actual_active_environment_count": count,
            "source_scoreboard": str((metrics_root / "scoreboard.csv").relative_to(ROOT)),
            "output_root": str(output.relative_to(ROOT)),
        })
        # The two jobs may finish together: serialize writes to the shared final table.
        with (benchmark / "formal_ablation_render.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run(sys.executable, "scripts/evaluation/build_event80_formal_ablation_table.py")
        (output / "inference.complete").touch()
        print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
