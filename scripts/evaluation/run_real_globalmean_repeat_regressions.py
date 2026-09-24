#!/usr/bin/env python3
"""Re-evaluate only environments that regressed as real support K increased."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/support_number_analysis/real/globalmean_20260922"
RESULTS = ROOT / "results/support_number_analysis/real/globalmean_repeat_regressions_20260922"
TRAIN_RUNS = {
    "ball": ROOT / "outputs/real97_train_real97_ball_target_c32_2gpu_20260909_v2_job113231",
    "door": ROOT / "outputs/real97_train_real97_door_target_c32_2gpu_20260909_v2_job113230",
}
CASES = (
    ("ball", 2, "ball-1"),
    ("ball", 2, "ball-2"),
    ("ball", 4, "ball-2"),
    ("door", 4, "door-3u"),
    ("door", 4, "door-3u-4d"),
    ("door", 4, "door-6u-8d"),
)


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


def choose_extra(task, environment, level, template, used, query_episodes, manifest, outcomes):
    options = []
    for row in manifest:
        if row["environment"] != environment or row.get("dataset_split") != "train":
            continue
        episode = row["episode_index"]
        if episode in used or episode in query_episodes:
            continue
        outcome = outcomes.get((environment, episode))
        if not outcome or outcome.get("split") != "train" or not outcome.get("training_eligible"):
            continue
        if outcome.get("level") != level:
            continue
        if task == "ball" and outcome.get("peak_frame") is not None:
            peak = int(outcome["peak_frame"])
            contains_peak = row["start_frame"] <= peak <= row["end_frame"]
            distance = min(abs(peak - row["start_frame"]), abs(peak - row["end_frame"]))
            rank = (not contains_peak, distance, episode, row["start_frame"])
        else:
            rank = (abs(row["start_frame"] - template["start_frame"]), episode, row["start_frame"])
        options.append((rank, row, outcome))
    if not options:
        raise ValueError(f"No disjoint training support: {task} {environment} level {level}")
    _, original, outcome = min(options, key=lambda option: option[0])
    extra = copy.deepcopy(template)
    extra.update(original)
    extra["action_level"] = level
    extra["evaluation_frame_indices"] = list(template["evaluation_frame_indices"])
    extra["native_frame_indices"] = [
        min(original["total_frames"] - 1, original["start_frame"] + i * original["frame_stride"])
        for i in range(original["length"])
    ]
    extra.pop("support_measurement", None)
    extra.pop("source_event_annotation", None)
    if outcome.get("event_annotation"):
        extra["source_event_annotation"] = outcome["event_annotation"]
    return extra


def prepare(task, k, environment, output):
    source = BASE / task / f"k{k}"
    old_prepared = source / "prepared"
    prepared = output / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_prepared, prepared, symlinks=True, dirs_exist_ok=True)
    reference = prepared / "reference"
    reference.unlink()
    reference.symlink_to((old_prepared / "reference").resolve(), target_is_directory=True)

    plan = read_json(prepared / "plan.json")
    support = read_jsonl(prepared / "support.jsonl")
    query = read_jsonl(prepared / "query.jsonl")
    config = read_json(source / "config.json")
    old_group = next(group for group in plan["environments"] if group["environment"] == environment)
    query_episodes = {query[i]["episode_index"] for i in old_group["query_indices"]}
    old_indices = old_group["support_indices"]

    if task == "ball":
        original = old_indices[0]
        level = support[original]["action_level"]
        selected = [original]
        levels = [level] * k
        if k == 4:
            second = old_indices[1]
            levels[-1] = support[second]["action_level"]
    else:
        selected = old_indices[:2]
        levels = [support[i]["action_level"] for i in selected] * 2

    used = {support[i]["episode_index"] for i in selected}
    if task == "ball" and k == 4:
        used.add(support[second]["episode_index"])
    manifest = read_jsonl(TRAIN_RUNS[task] / "input_manifest/train.jsonl")
    outcomes = {
        (row["environment"], row["episode_index"]): row
        for row in read_jsonl(ROOT / plan["experiment"]["calibrated_outcomes"])
        if row.get("task") == task
    }
    if task == "ball":
        template = support[original]
        for _ in range(k - 1 - (k == 4)):
            extra = choose_extra(task, environment, level, template, used, query_episodes, manifest, outcomes)
            selected.append(len(support))
            support.append(extra)
            used.add(extra["episode_index"])
        if k == 4:
            selected.append(second)
    else:
        for initial in old_indices[:2]:
            template = support[initial]
            extra = choose_extra(task, environment, template["action_level"], template, used,
                                 query_episodes, manifest, outcomes)
            selected.append(len(support))
            support.append(extra)
            used.add(extra["episode_index"])

    assert len(selected) == k
    assert len({support[i]["episode_index"] for i in selected}) == k
    assert all(support[i]["episode_index"] not in query_episodes for i in selected)
    group = copy.deepcopy(old_group)
    group["support_indices"] = selected
    group["requested_support_levels"] = levels
    group["actual_support_levels"] = levels
    plan["environments"] = [group]
    plan["prepared"] = str(prepared)
    plan["support_count"] = len(support)
    plan["active_support_count"] = k
    plan["expected_environments"] = 1
    plan["support_manifest_retains_unused_original_rows"] = True
    plan["support_override"]["source_prepared"] = str(old_prepared)
    plan["support_override"]["source_inference"] = str(source)
    plan["support_override"]["requested_levels"] = {environment: levels}
    plan["support_override"]["selected"] = [
        {
            "environment": environment,
            "level": support[i]["action_level"],
            "support_index": i,
            "episode_index": support[i]["episode_index"],
            "start_frame": support[i]["start_frame"],
            "end_frame": support[i]["end_frame"],
            "dataset_split": "train",
        }
        for i in selected
    ]
    if task == "ball":
        plan["protocol"]["ball"].setdefault("explicit_support_levels_by_environment", {})[environment] = levels
    task_config = config["tasks"][task]
    task_config["source_prepared"] = str(old_prepared)
    task_config["source_inference"] = str(source)
    task_config["prepared"] = str(prepared)
    task_config["support_levels"] = {environment: levels}
    task_config["partial_stage2_replacement"] = True
    config["output"] = str(output)
    config["experiment_note"] = "Only lower-K-success-to-higher-K-regression cases; repeated levels, distinct training episodes."

    write_json(prepared / "plan.json", plan)
    write_jsonl(prepared / "support.jsonl", support)
    write_json(output / "config.json", config)
    paths = set((prepared / "stage_files.txt").read_text().splitlines())
    for i in selected:
        paths.update(support[i]["video"])
        paths.add(support[i]["action"])
    (prepared / "stage_files.txt").write_text("\n".join(sorted(paths)) + "\n")
    write_json(output / "support_protocol.json", {
        "task": task, "environment": environment, "K": k,
        "initialization": "global mean", "support_loss_reduction": "mean",
        "selection": [{"episode_index": support[i]["episode_index"],
                       "level": support[i]["action_level"], "support_index": i} for i in selected],
        "source": str(source), "query_indices": group["query_indices"],
        "query_episode_disjoint": True,
    })
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(len(CASES)))
    args = parser.parse_args()
    task, k, environment = CASES[args.index]
    output = RESULTS / task / f"k{k}" / environment
    if (output / "unbounded_inference_complete.json").exists():
        return
    prepared = prepare(task, k, environment, output)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "scripts/evaluation") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/evaluation/infer_real97_unbounded_context.py"),
        "--config", str(output / "config.json"),
        "--task", task, "--prepared", str(prepared), "--output", str(output),
    ], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
