#!/usr/bin/env python3
"""Rerun the original real global-mean support selections with summed support loss."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/support_number_analysis/real/globalmean_20260922"
DEST = ROOT / "results/support_number_analysis/real/globalmean_sumloss_original_support_20260922"
CASES = (("ball", 2), ("ball", 4), ("door", 4))


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(task, k, source, output):
    source_prepared = source / "prepared"
    prepared = output / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    if not prepared.exists():
        temporary = output / f".prepared.{os.getpid()}.partial"
        shutil.copytree(source_prepared, temporary, symlinks=True)
        reference = temporary / "reference"
        if not reference.is_symlink():
            raise RuntimeError("Expected the original prepared reference symlink")
        reference.unlink()
        reference.symlink_to((source_prepared / "reference").resolve(), target_is_directory=True)
        plan = read_json(temporary / "plan.json")
        if len(plan["environments"]) not in (8, 10):
            raise RuntimeError("Unexpected original environment cohort")
        if any(len(env["support_indices"]) != k for env in plan["environments"]):
            raise RuntimeError("Original support count differs from requested K")
        if plan["protocol"]["ttt"].get("ttt_support_loss_reduction") != "mean":
            raise RuntimeError("Expected original mean-loss protocol")
        plan["prepared"] = str(prepared)
        plan["protocol"]["ttt"]["ttt_support_loss_reduction"] = "sum"
        write_json(temporary / "plan.json", plan)
        temporary.rename(prepared)
    plan = read_json(prepared / "plan.json")
    if plan["protocol"]["ttt"].get("ttt_support_loss_reduction") != "sum":
        raise RuntimeError("Prepared protocol does not use summed support loss")

    config = read_json(source / "config.json")
    if config["tasks"][task]["initial_context"] != "mean_training_table":
        raise RuntimeError("Original run is not global-mean initialized")
    config["output"] = str(output)
    setting = config["tasks"][task]
    setting.update(source_prepared=str(source_prepared), source_inference=str(source),
                   prepared=str(prepared), support_levels={},
                   partial_stage2_replacement=False, require_exact_original_support=True)
    config["support_count_experiment"] = {
        "K": k, "initialization": "full training table mean",
        "support_selection": "unchanged from original globalmean_20260922",
        "support_loss_reduction": "sum", "source": str(source),
    }
    config["experiment_note"] = "Original global-mean support/query selection; only support loss mean-to-sum changes."
    config_path = output / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise RuntimeError("Existing output configuration differs")
    write_json(config_path, config)

    expected = sum(len(env["query_indices"]) for env in plan["environments"])
    stage1 = source / "raw/stage1"
    markers = sorted((source / "completed_variants").glob("*_stage1.json"))
    if len(list(stage1.glob("*.mp4"))) != expected or len(markers) != expected:
        raise RuntimeError("Original Stage1 artifacts are incomplete")
    link = output / "raw/stage1"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        if link.exists():
            raise RuntimeError("Refusing to replace an existing Stage1 directory")
        link.symlink_to(stage1.resolve(), target_is_directory=True)
    if link.resolve() != stage1.resolve():
        raise RuntimeError("Stage1 reuse points to a different experiment")
    variants = output / "completed_variants"
    variants.mkdir(exist_ok=True)
    for source_marker in markers:
        target = variants / source_marker.name
        if not target.exists():
            target.symlink_to(source_marker.resolve())

    support = [json.loads(line) for line in (prepared / "support.jsonl").read_text().splitlines() if line]
    selection = {
        env["environment"]: [
            {"episode_index": support[i]["episode_index"],
             "action_level": support[i]["action_level"], "support_index": i}
            for i in env["support_indices"]
        ] for env in plan["environments"]
    }
    write_json(output / "support_protocol.json", {
        "task": task, "K": k, "initialization": "global mean",
        "support_loss_reduction": "sum", "source": str(source),
        "source_support_sha256": digest(source_prepared / "support.jsonl"),
        "source_query_sha256": digest(source_prepared / "query.jsonl"),
        "selection": selection, "query_cohort_unchanged": True,
        "stage1_reused": True,
    })
    return prepared, config_path


def reuse_compute_cache(task, source_prepared, prepared):
    cache = Path("/tmp") / os.environ["USER"] / "bwm_shared_cache"
    checkpoint_cache = cache / "eval_checkpoints"
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    from infer_real97_standard_reference import copy_cached
    shared = checkpoint_cache / f"real_support_{task}_dual6500.safetensors"
    copy_cached(prepared / "reference/model.safetensors", shared)
    tag = hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
    alias = checkpoint_cache / f"{tag}.safetensors"
    if not alias.exists():
        alias.symlink_to(shared)
    Path(str(alias) + ".copy_complete").write_text("complete\n")

    original_tag = hashlib.sha256(str(source_prepared.resolve()).encode()).hexdigest()[:16]
    original_data = cache / "eval_datasets" / original_tag
    data = cache / "eval_datasets" / tag
    original_marker = Path(str(original_data) + ".copy_complete")
    if not data.exists() and original_data.is_dir() and original_marker.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        data.symlink_to(original_data, target_is_directory=True)
        Path(str(data) + ".copy_complete").write_text("complete\n")


def summarize(task, k, output):
    report = read_json(output / "metrics/summary_provisional.json")
    image = next(row for row in report["image_object"]
                 if row["split"] == "test" and row["method"] == "ours_stage2")
    action = next(row for row in report["action"] if row["method"] == "ours_stage2")
    if task == "door":
        protocol = read_json(ROOT / "results/real97_all_methods_main_table_v1/metrics/door_first_close_level_tolerance1_20260919.json")
        decisions = []
        indexed = {row["environment"]: row for row in action["decisions"]}
        for environment, truth in protocol["ground_truth_lowest_level"].items():
            states = indexed[environment]["closed_by_level"]
            if {int(level) for level in states} != set(range(1, 11)):
                raise RuntimeError(f"Incomplete Door action levels: {environment}")
            predicted = next((level for level in range(1, 11) if states[str(level)] is True), None)
            score = 1.0 if predicted == truth else 0.5 if predicted is not None and abs(predicted - truth) == 1 else 0.0
            decisions.append({"environment": environment, "gt_first_close": truth,
                              "predicted_first_close": predicted, "score": score})
        action_score = sum(row["score"] for row in decisions) / len(decisions)
    else:
        decisions = action["decisions"]
        action_score = action["macro_score"]
    write_json(output / "formal_summary.json", {
        "task": task, "K": k, "queries": image["queries"],
        "action_pct": 100 * action_score,
        "action_denominator": len(decisions), "action_decisions": decisions,
        "image_object": image["query_mean"],
        "initialization": "global mean", "support_loss_reduction": "sum",
        "support_selection": "original globalmean_20260922",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(len(CASES)))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compute allocation required")
    os.chdir(ROOT)
    task, k = CASES[args.index]
    source = SOURCE / task / f"k{k}"
    output = DEST / task / f"k{k}"
    prepared, config = prepare(task, k, source, output)
    if not (output / "unbounded_inference_complete.json").exists():
        reuse_compute_cache(task, source / "prepared", prepared)
        subprocess.run([
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/evaluation/infer_real97_unbounded_context.py"),
            "--config", str(config), "--task", task,
            "--prepared", str(prepared), "--output", str(output),
        ], cwd=ROOT, check=True)
    if not (output / "globalmean_metric_pipeline_complete.json").exists():
        from run_real_globalmean_support_number import score
        score(task, output)
    summarize(task, k, output)


if __name__ == "__main__":
    main()
