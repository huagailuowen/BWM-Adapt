#!/usr/bin/env python3
"""Freeze small TTT inference inputs and hard-link the completed checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["door", "ball"], required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    setting = config["tasks"][args.task]
    target = ROOT / setting["prepared"]
    fingerprint = hashlib.sha256(raw).hexdigest()
    if (target / "prepared_complete.json").exists():
        prior = json.loads((target / "prepared_complete.json").read_text())
        if prior["config_sha256"] != fingerprint:
            raise RuntimeError("Existing preparation has a different configuration")
        print(json.dumps({"task": args.task, "reused": str(target)}), flush=True)
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete preparation: {target}")
    temporary = target.with_name(target.name + f".partial-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    source = ROOT / setting["source_prepared"]
    training = ROOT / setting["training_run"]
    hashes = {}
    for name in ["plan.json", "query.jsonl", "support.jsonl", "extension_action_templates.jsonl", "stage_files.txt"]:
        content = (source / name).read_bytes()
        (temporary / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    reference = temporary / "reference"
    reference.mkdir()
    shutil.copyfile(training / "submitted_config.yaml", reference / "training_config.yaml")
    shutil.copyfile(training / "input_manifest/action_stats.json", reference / "action_stats.json")
    os.link(training / setting["checkpoint"], reference / "model.safetensors")
    (temporary / "evaluation_config.json").write_bytes(raw)
    record = {"task": args.task, "config_sha256": fingerprint, "manifest_sha256": hashes,
              "training_run": setting["training_run"], "checkpoint": setting["checkpoint"],
              "checkpoint_step": setting["checkpoint_step"], "model_is_hard_link": True,
              "old_results_modified": False}
    (temporary / "prepared_complete.json").write_text(json.dumps(record, indent=2) + "\n")
    os.rename(temporary, target)
    print(json.dumps({"prepared": str(target), **record}), flush=True)


if __name__ == "__main__":
    main()
