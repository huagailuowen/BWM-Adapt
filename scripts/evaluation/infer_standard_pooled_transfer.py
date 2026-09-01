#!/usr/bin/env python3
"""Run a no-adaptation pooled world model on a frozen transfer protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)
SOURCE_RE = re.compile(r"source[_-]?(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-paths", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def prediction_index(root: Path) -> dict[tuple[int, int], Path]:
    output: dict[tuple[int, int], Path] = {}
    if not root.is_dir():
        return output
    for path in root.rglob("*.mp4"):
        relative = path.relative_to(root).as_posix()
        source_match = SOURCE_RE.search(relative)
        sample_match = SAMPLE_RE.search(path.name)
        if source_match is None or sample_match is None:
            continue
        key = (int(source_match.group(1)), int(sample_match.group(1)))
        if key in output:
            raise ValueError(f"Duplicate pooled prediction for {key}: {output[key]} and {path}")
        output[key] = path.resolve()
    return output


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Pooled transfer inference must run in a Slurm allocation")

    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inference = config["inference"]
    train_config = resolve(inference["train_config"])
    transfer_entries = list(config["transfer_plans"])
    source_plan_values = inference.get("source_transfer_plans")
    if source_plan_values is None:
        source_plan_values = [inference["source_transfer_plan"]]
        if len(transfer_entries) != 1:
            raise ValueError(
                "Legacy inference.source_transfer_plan requires exactly one transfer_plans entry. "
                "Use inference.source_transfer_plans for a joint ID/OOD run."
            )
    if len(source_plan_values) != len(transfer_entries):
        raise ValueError(
            "inference.source_transfer_plans and transfer_plans must have the same length: "
            f"{len(source_plan_values)} != {len(transfer_entries)}"
        )

    method_root = resolve(inference["method_output_root"])
    plan_pairs: list[tuple[Path, Path, Path]] = []
    target_records: dict[int, tuple[int, Path]] = {}
    for source_value, transfer_entry in zip(source_plan_values, transfer_entries):
        source_plan_path = resolve(source_value)
        destination_plan_path = resolve(transfer_entry["path"])
        raw_root = destination_plan_path.parent / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)

        source_plan = read_json(source_plan_path)
        if destination_plan_path.is_file():
            if read_json(destination_plan_path) != source_plan:
                raise ValueError(f"Existing transfer plan differs from {source_plan_path}")
        else:
            write_json_atomic(destination_plan_path, source_plan)
        plan_pairs.append((source_plan_path, destination_plan_path, raw_root))

        for environment in source_plan:
            source_index = int(environment["source_index"])
            for value in environment["target_indices"]:
                target_index = int(value)
                record = (source_index, raw_root)
                previous = target_records.setdefault(target_index, record)
                if previous != record:
                    raise ValueError(
                        f"Target {target_index} belongs to both {previous} and {record}"
                    )

    if len(plan_pairs) == 1:
        flat_root = plan_pairs[0][1].parent / str(inference.get("flat_dir_name", "flat"))
    else:
        flat_root = method_root / str(inference.get("flat_dir_name", "flat"))
    flat_root.mkdir(parents=True, exist_ok=True)

    existing_by_root = {
        raw_root: prediction_index(raw_root)
        for raw_root in {record[1] for record in target_records.values()}
    }
    missing = [
        target
        for target, (source, raw_root) in sorted(target_records.items())
        if (source, target) not in existing_by_root[raw_root]
    ]
    if missing:
        command = [
            sys.executable,
            str(ROOT / "scripts/infer.py"),
            "--config",
            str(train_config),
            "--dataset_metadata_path",
            str(resolve(config["metadata_jsonl"])),
            "--model_paths",
            str(args.model_paths.resolve()),
            "--ckpt_path",
            str(args.checkpoint.resolve()),
            "--output_path",
            str(flat_root),
            "--sample_indices",
            ",".join(map(str, missing)),
            "--num_inference_steps",
            str(int(inference.get("num_inference_steps", 25))),
            "--cfg_scale",
            str(float(inference.get("cfg_scale", 1.0))),
            "--fps",
            str(int(inference.get("fps", 20))),
            "--quality",
            str(int(inference.get("quality", 6))),
            "--seed",
            str(int(inference["seed"])),
            "--skip_existing",
        ]
        subprocess.run(command, cwd=ROOT, check=True)

    for target_index, (source_index, raw_root) in sorted(target_records.items()):
        destination_dir = raw_root / f"source{source_index:04d}_pooled"
        destination_dir.mkdir(parents=True, exist_ok=True)
        existing_matches = sorted(destination_dir.glob(f"sample{target_index:04d}_*.mp4"))
        if len(existing_matches) == 1:
            continue
        if len(existing_matches) > 1:
            raise ValueError(f"Multiple destination videos for target {target_index}")
        generated = sorted(flat_root.glob(f"sample{target_index:04d}_*.mp4"))
        if len(generated) != 1:
            raise FileNotFoundError(
                f"Expected one generated video for target {target_index}, got {generated}"
            )
        shutil.move(str(generated[0]), destination_dir / generated[0].name)

    final_predictions_by_root = {
        raw_root: prediction_index(raw_root)
        for raw_root in {record[1] for record in target_records.values()}
    }
    missing_final = sorted(
        (source, target)
        for target, (source, raw_root) in target_records.items()
        if (source, target) not in final_predictions_by_root[raw_root]
    )
    if missing_final:
        raise RuntimeError(f"Missing organized predictions: {missing_final}")

    write_json_atomic(method_root / "inference_protocol.json", {
        "adaptation": "none",
        "checkpoint": str(args.checkpoint.resolve()),
        "method": "standard_pooled_wm",
        "query_count": len(target_records),
        "query_state_policy": "none",
        "source_transfer_plans": [str(pair[0]) for pair in plan_pairs],
        "support_is_ignored_by_generator": True,
        "train_config": str(train_config),
        "transfer_plans": [str(pair[1]) for pair in plan_pairs],
    })

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_sim_action_selection.py"),
            "--config",
            str(config_path),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_sim_transfer_metrics.py"),
            "--config",
            str(config_path),
            "--lpips",
            "--lpips-net",
            "alex",
            "--lpips-device",
            "cuda",
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        f"[done] task={config['task']} queries={len(target_records)} output={method_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
