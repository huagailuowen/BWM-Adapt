#!/usr/bin/env python3
"""Run support-time LoRA adaptation on a frozen simulation transfer protocol."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer import (  # noqa: E402
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from scripts.infer_stage2_ttt import _freeze_pipe  # noqa: E402
from scripts.methods.infer_lora_tta_event80 import (  # noqa: E402
    adapt_lora,
    install_lora,
    reset_lora,
)
from wan_video_action.parsers import (  # noqa: E402
    add_general_config,
    merge_yaml_and_args,
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = add_general_config(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--evaluation-config", type=Path, required=True)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    args.stage2_fixed_timestep_index = None
    return args


def _copy_frozen_plan(source: Path, destination: Path) -> list[dict[str, Any]]:
    value = read_json(source)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Transfer plan must be a non-empty list: {source}")
    if destination.is_file() and read_json(destination) != value:
        raise ValueError(f"Existing destination plan differs from {source}: {destination}")
    if not destination.is_file():
        write_json_atomic(destination, value)
    return value


def _support_manifest(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    value = read_json(path)
    if isinstance(value, Mapping):
        value = value.get("environments", value.get("groups", []))
    if not isinstance(value, list):
        raise ValueError(f"Unsupported support/query manifest: {path}")
    output = {}
    for row in value:
        source = int(row.get("source_index", row.get("support_indices", [None])[0]))
        if source in output:
            raise ValueError(f"Duplicate source_index={source} in {path}")
        output[source] = dict(row)
    return output


def _domain(entry: Mapping[str, Any], row: Mapping[str, Any], source: int) -> str:
    if row.get("domain") is not None:
        return str(row["domain"])
    if entry.get("domain") is not None:
        return str(entry["domain"])
    mapping = entry.get("domain_by_source", {})
    value = mapping.get(source, mapping.get(str(source)))
    return "unknown" if value is None else str(value)


def load_groups(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[Path]]:
    inference = config["inference"]
    destinations = list(config["transfer_plans"])
    source_values = inference.get("source_transfer_plans")
    if source_values is None:
        source_values = [inference["source_transfer_plan"]]
    if len(source_values) != len(destinations):
        raise ValueError("source and destination transfer-plan counts differ")

    manifest_path = inference.get("support_query_manifest")
    support_by_source = _support_manifest(
        None if manifest_path is None else resolve(manifest_path)
    )
    groups: list[dict[str, Any]] = []
    destination_paths: list[Path] = []
    seen_sources: set[int] = set()
    for source_value, destination_entry in zip(source_values, destinations):
        source_path = resolve(source_value)
        destination_path = resolve(destination_entry["path"])
        rows = _copy_frozen_plan(source_path, destination_path)
        destination_paths.append(destination_path)
        for row in rows:
            source = int(row["source_index"])
            if source in seen_sources:
                raise ValueError(f"source_index={source} occurs in multiple transfer plans")
            seen_sources.add(source)
            queries = [
                int(value)
                for value in row.get("query_indices", row.get("target_indices", []))
            ]
            manifest_row = support_by_source.get(source, {})
            supports = [
                int(value)
                for value in manifest_row.get(
                    "support_indices", row.get("support_indices", [source])
                )
            ]
            expected_queries = manifest_row.get("query_indices")
            if expected_queries is not None and queries != [int(v) for v in expected_queries]:
                raise ValueError(f"Query mismatch for source_index={source}")
            overlap = sorted(set(supports) & set(queries))
            if overlap:
                raise ValueError(f"Support/query leakage for source_index={source}: {overlap}")
            if not supports or not queries:
                raise ValueError(f"Empty support/query set for source_index={source}")
            groups.append(
                {
                    "source_index": source,
                    "support_indices": supports,
                    "query_indices": queries,
                    "domain": _domain(destination_entry, row, source),
                    "prediction_root": destination_path.parent / "raw",
                    "destination_plan": destination_path,
                }
            )
    return groups, destination_paths


def _adapter_payload(adapters, metadata: Mapping[str, Any]) -> dict[str, Any]:
    state = {}
    for name, adapter in adapters:
        state[f"{name}.lora_a"] = adapter.lora_a.detach().float().cpu()
        state[f"{name}.lora_b"] = adapter.lora_b.detach().float().cpu()
    return {"metadata": dict(metadata), "state": state}


def _save_adapter(path: Path, adapters, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_adapter_payload(adapters, metadata), temporary)
    temporary.replace(path)


def _load_adapter(path: Path, adapters, expected: Mapping[str, Any]) -> list[float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Adapter cache mismatch at {path}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    reset_lora(adapters)
    state = payload["state"]
    with torch.no_grad():
        for name, adapter in adapters:
            adapter.lora_a.copy_(state[f"{name}.lora_a"].to(adapter.lora_a.device))
            adapter.lora_b.copy_(state[f"{name}.lora_b"].to(adapter.lora_b.device))
    return [float(value) for value in metadata.get("inner_losses", [])]


def _prediction_name(index: int, row: Mapping[str, Any]) -> str:
    return (
        f"sample{index:04d}_episode{int(row['episode_index']):06d}_"
        f"frames{int(row['start_frame']):04d}-{int(row['end_frame']):04d}.mp4"
    )


def _metadata_subset(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sample_id",
        "episode_index",
        "environment_id",
        "gravity_index",
        "gravity_mps2",
        "target_mass_index",
        "target_mass_kg",
        "mass_index",
        "friction_index",
        "target_table_friction_mu",
        "causal_class",
        "button_color",
        "action_id",
        "action_amplitude",
    )
    return {key: row[key] for key in keys if key in row}


def _run_postprocessing(
    evaluation_config: Path,
    config: Mapping[str, Any],
    destination_paths: list[Path],
) -> None:
    inference = config["inference"]
    method_root = resolve(inference["method_output_root"])
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_sim_action_selection.py"),
            "--config",
            str(evaluation_config),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executableable if False else sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_sim_transfer_metrics.py"),
            "--config",
            str(evaluation_config),
            "--lpips",
            "--lpips-net",
            "alex",
            "--lpips-device",
            "cuda",
        ],
        cwd=ROOT,
        check=True,
    )
    for index, plan_path in enumerate(destination_paths):
        suffix = "" if len(destination_paths) == 1 else f"/{plan_path.stem}"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluation/compose_context_transfer_support_grids.py"),
                "--metadata-path",
                str(resolve(config["metadata_jsonl"])),
                "--dataset-root",
                str(resolve(config["dataset_root"])),
                "--transfer-plan",
                str(plan_path),
                "--prediction-root",
                str(plan_path.parent / "raw"),
                "--output-dir",
                str(method_root / f"transfer/grids_support_plus_queries{suffix}"),
                "--fps",
                str(int(inference.get("fps", 20))),
                "--quality",
                str(int(inference.get("quality", 6))),
                "--columns",
                "5",
                "--support-size",
                str(int(config.get("support_size", 1))),
                "--prediction-label",
                "LoRA-TTT query",
            ],
            cwd=ROOT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    evaluation_config = resolve(args.evaluation_config)
    config = yaml.safe_load(evaluation_config.read_text(encoding="utf-8"))
    lora = config["lora_tta"]
    inference = config["inference"]
    method_root = resolve(inference["method_output_root"])
    method_root.mkdir(parents=True, exist_ok=True)
    groups, destination_paths = load_groups(config)

    args.dataset_metadata_path = str(resolve(config["metadata_jsonl"]))
    args.output_path = str(method_root / "transfer/raw")
    args.num_inference_steps = int(inference.get("num_inference_steps", 25))
    args.cfg_scale = float(inference.get("cfg_scale", 1.0))
    args.fps = int(inference.get("fps", 20))
    args.quality = int(inference.get("quality", 6))
    args.seed = int(inference["seed"])
    args.lora_steps = int(lora["steps"])
    args.lora_learning_rate = float(lora["learning_rate"])
    args.lora_alpha = float(lora["alpha"])
    args.lora_gradient_clip = float(lora["gradient_clip_norm"])
    args.lora_rank = int(lora["rank"])
    args.lora_target_modules = ",".join(map(str, lora["target_modules"]))
    args.adapter_seed = int(lora["adapter_seed"])

    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    _freeze_pipe(pipe)
    targets = {str(value) for value in lora["target_modules"]}
    adapters = install_lora(
        pipe.dit,
        targets=targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        seed=args.adapter_seed,
    )
    adapter_count = sum(
        adapter.lora_a.numel() + adapter.lora_b.numel() for _, adapter in adapters
    )
    print(
        f"[lora] task={config['task']} modules={len(adapters)} params={adapter_count} "
        f"rank={args.lora_rank} steps={args.lora_steps}",
        flush=True,
    )

    result_path = method_root / "lora_adaptation/results.jsonl"
    existing_rows = read_jsonl(result_path)
    results = {
        (int(row["source_index"]), int(row["sample_index"])): row
        for row in existing_rows
    }
    adaptation_rows: list[dict[str, Any]] = []
    expected_predictions = 0
    completed_predictions = 0
    for group in groups:
        source = int(group["source_index"])
        supports = list(group["support_indices"])
        queries = list(group["query_indices"])
        expected_predictions += len(queries)
        prediction_dir = Path(group["prediction_root"]) / f"source{source:04d}_lora_tta"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for query_index in queries:
            row = dataset.data[query_index]
            prediction = prediction_dir / _prediction_name(query_index, row)
            marker = prediction.with_suffix(prediction.suffix + ".complete")
            if prediction.is_file() and marker.is_file():
                completed_predictions += 1
            else:
                if prediction.exists():
                    prediction.unlink()
                pending.append((query_index, row, prediction, marker))
        if not pending:
            print(f"[resume] source={source} all_queries_complete", flush=True)
            continue

        cache_metadata = {
            "source_index": source,
            "support_indices": supports,
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "steps": args.lora_steps,
            "learning_rate": args.lora_learning_rate,
            "gradient_clip_norm": args.lora_gradient_clip,
            "adapter_seed": args.adapter_seed,
        }
        adapter_path = method_root / "lora_adaptation/adapter_states" / f"source{source:04d}.pt"
        if adapter_path.is_file():
            losses = _load_adapter(adapter_path, adapters, cache_metadata)
            print(f"[resume_adapter] source={source} path={adapter_path}", flush=True)
        else:
            environment_seed = args.adapter_seed + source * 1009
            random.seed(environment_seed)
            np.random.seed(environment_seed % (2**32))
            torch.manual_seed(environment_seed)
            torch.cuda.manual_seed_all(environment_seed)
            support_items = [dataset[index] for index in supports]
            print(
                f"[adapt] source={source} domain={group['domain']} supports={supports} "
                f"queries={len(queries)}",
                flush=True,
            )
            losses = adapt_lora(pipe, support_items, adapters, args)
            saved_metadata = dict(cache_metadata)
            saved_metadata["inner_losses"] = losses
            _save_adapter(adapter_path, adapters, saved_metadata)
        adaptation_rows.append({**cache_metadata, "domain": group["domain"], "inner_losses": losses})

        pipe.eval()
        for query_index, query_row, prediction, marker in pending:
            sample = prepare_sample_for_rollout(dataset[query_index], query_index, pipe, args)
            sample["output_path"] = str(prediction)
            with torch.no_grad():
                _run_autoregressive(pipe=pipe, sample=sample, args=args)
            marker.touch()
            completed_predictions += 1
            results[(source, query_index)] = {
                "source_index": source,
                "support_indices": supports,
                "sample_index": query_index,
                "domain": group["domain"],
                "support_query_disjoint": True,
                "query_updates_lora": False,
                "adaptation_reused_across_queries": True,
                "prediction_path": str(prediction),
                **_metadata_subset(query_row),
            }
            write_jsonl_atomic(
                result_path,
                [results[key] for key in sorted(results)],
            )
            torch.cuda.empty_cache()

    write_json_atomic(
        method_root / "inference_protocol.json",
        {
            "method": "lora_tta",
            "source_checkpoint": str(Path(args.ckpt_path).resolve()),
            "source_model": "standard_pooled_wm",
            "train_config": str(resolve(inference["train_config"])),
            "evaluation_config": str(evaluation_config),
            "source_transfer_plans": [str(resolve(value)) for value in inference.get(
                "source_transfer_plans", [inference.get("source_transfer_plan")]
            )],
            "destination_transfer_plans": [str(path) for path in destination_paths],
            "support_query_disjoint": True,
            "query_state_policy": "read_only",
            "reset_adapter_per_environment": True,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "adapter_parameter_count": adapter_count,
            "lora": dict(lora),
            "environment_count": len(groups),
            "query_count": expected_predictions,
        },
    )
    write_jsonl_atomic(
        method_root / "lora_adaptation/latest_run.jsonl", adaptation_rows
    )
    if completed_predictions != expected_predictions:
        raise RuntimeError(
            f"Incomplete rollout set: {completed_predictions}/{expected_predictions}"
        )
    _run_postprocessing(evaluation_config, config, destination_paths)
    print(
        f"[done] task={config['task']} environments={len(groups)} "
        f"queries={expected_predictions} output={method_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
