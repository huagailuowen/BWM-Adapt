#!/usr/bin/env python3
"""Add only 15 train-negative candidates; all 45 test videos remain unchanged."""
import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from real916_stick_final_common import ROOT, read_rows, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compute allocation required")
    config = read(args.config)
    dest = ROOT / config["prepared"]
    digest = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    with dest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / "prepared_complete.json").exists():
            if read(dest / "prepared_complete.json")["config_sha256"] != digest:
                raise RuntimeError("Different extension config")
            return
        if dest.exists():
            raise RuntimeError("Refusing incomplete published preparation")
        temp = Path(tempfile.mkdtemp(prefix=dest.name+".partial-", dir=dest.parent))
        base = ROOT / config["original_prepared"]
        oldplan = read(base / "plan.json")
        oldqueries = read_rows(base / "query.jsonl")
        supports = read_rows(base / "support.jsonl")
        annotations = {(r["environment"], int(r["episode_index"])): r for r in
                       read_rows(Path(oldplan["source_dataset"]) / "episode_split_manifest.jsonl")}
        oldnegative = [r for r in oldqueries if r["dataset_split"] == "train" and
                       annotations[(r["environment"], r["episode_index"])]["outcome"] in ("left_down", "right_down")]
        if len(oldnegative) != 15:
            raise RuntimeError("Expected 15 existing train negative candidates")
        forbidden = {(r["environment"], r["episode_index"]) for r in oldqueries + supports}
        legal = defaultdict(list)
        for row in read_rows(ROOT / config["manifest"] / "train.jsonl"):
            key = (row["environment"], row["episode_index"])
            ann = annotations[key]
            if (key not in forbidden and row["sampling_kind"] == "lift" and
                ann["split"] == "train" and ann.get("vision_ok") and
                ann["outcome"] in ("left_down", "right_down")):
                legal[key].append(row)
        rng = random.Random(config["seed"])
        names = [e["environment"] for e in oldplan["environments"]]
        shuffled = names.copy()
        rng.shuffle(shuffled)
        targets = {name: 3 + int(name in shuffled[:3]) for name in names}
        queries = []
        envs = []
        for oldenv in oldplan["environments"]:
            name = oldenv["environment"]
            retained = [r for r in oldnegative if r["environment"] == name]
            need = targets[name] - len(retained)
            candidates = sorted(k for k in legal if k[0] == name)
            rng.shuffle(candidates)
            if len(candidates) < need or need < 0:
                raise RuntimeError("Insufficient candidates, never silently omit an environment")
            indices = []
            for key in candidates[:need]:
                windows = sorted(legal[key], key=lambda r: int(r["start_frame"]))
                row = copy.deepcopy(windows[(len(windows)-1)//2])
                row["sample_index"] = len(queries)
                row["native_frame_indices"] = [min(row["start_frame"]+3*k,row["total_frames"]-1) for k in range(33)]
                row["evaluation_frame_indices"] = [k for k in range(1,33) if row["start_frame"]+3*k < row["total_frames"]]
                indices.append(len(queries))
                queries.append(row)
            env = copy.deepcopy(oldenv)
            env["query_indices"] = indices
            envs.append(env)
            print("[selected]", name, "retained", [r["episode_index"] for r in retained],
                  "added", [queries[i]["episode_index"] for i in indices], "target", targets[name], flush=True)
        if len(queries) != 15 or sum(targets.values()) != 30:
            raise RuntimeError("Train cap must be exactly 30; only 15 new rollouts")
        for filename, rows in (("query.jsonl", queries), ("support.jsonl", supports)):
            (temp / filename).write_text("".join(json.dumps(r)+"\n" for r in rows))
        staged = {p for r in queries+supports for p in r["video"]+[r["action"],r["action"].split("/")[0]+"/meta/info.json"]}
        (temp / "stage_files.txt").write_text("\n".join(sorted(staged))+"\n")
        shutil.copy2(base / "action_stats.json", temp / "action_stats.json")
        evaluation = read(base / "evaluation_config.json")
        evaluation["prepared"] = config["prepared"]
        evaluation["extension_selection_seed"] = config["seed"]
        evaluation["query_policy"] = "15 additional train negatives only; no test re-inference"
        evaluation["checkpoint_cache_prepared"] = config["original_prepared"]
        evaluation["reuse_contexts_from"] = config["original_ours_inference"]
        evaluation["context_reuse_reason"] = "Identical support videos, checkpoint, table and Stage2 settings"
        for method in ("ours", "standard"):
            src = base / method
            dst = temp / method
            dst.mkdir()
            for filename in ("training_config.yaml", "checkpoint_selection.json"):
                shutil.copy2(src / filename, dst / filename)
            os.link(src / "model.safetensors", dst / "model.safetensors")
            if method == "ours":
                shutil.copy2(src / "context_table.json", dst / "context_table.json")
            evaluation["methods"][method]["output"] = config[method+"_output"]
        write_json(temp / "evaluation_config.json", evaluation)
        plan = {**oldplan, "environments": envs, "query_count": 15,
                "query_splits": {"train": 15, "test": 0},
                "selection": "Uniform random eligible train episodes within each environment; midpoint legal Lift start",
                "test_outcomes_used_for_selection": False, "model_predictions_used_for_selection": False,
                "added_train_total": 30, "new_train_rollouts": 15, "targets_by_environment": targets}
        write_json(temp / "plan.json", plan)
        oldkey = lambda r: f"q{r['sample_index']:04d}_{r['environment']}_{r['dataset_split']}_ep{r['episode_index']:06d}"
        write_json(temp / "mixed_pool_selection.json", {
            "all_test_queries": 45, "total_train_queries": 30, "total_candidates": 75,
            "retained_train_keys": [oldkey(r) for r in oldnegative],
            "new_query_keys": [oldkey(r) for r in queries],
            "selection_config": config, "targets_by_environment": targets,
            "gt_annotation_used_only_for_candidate_selection": True,
            "score_uses_actual_visible_chunk_tail": True,
        })
        write_json(temp / "prepared_complete.json", {"config_sha256": digest,"queries":15,"supports":18,"environments":9})
        temp.rename(dest)


if __name__ == "__main__":
    main()
