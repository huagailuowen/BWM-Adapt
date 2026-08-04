#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import (  # noqa: E402
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from make_gt_pred_comparison import _default_pred_name  # noqa: E402
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _read_context_table(path: str | Path) -> dict[int, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = {}
    for record in payload.get("records", []):
        group = int(round(float(record["friction_mu"])))
        table[group] = np.asarray(record["context"], dtype=np.float32).reshape(-1)
    if sorted(table) != [0, 1, 2, 3]:
        raise ValueError(f"Expected context groups 0,1,2,3; got {sorted(table)}")
    return table


def _parse_indices_by_group(raw: str) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        group_text, indices_text = entry.split(":", 1)
        group = int(group_text)
        indices = _parse_sample_indices(indices_text)
        if not indices:
            raise ValueError(f"No indices supplied for group {group}")
        if group in groups:
            raise ValueError(f"Duplicate source group {group}")
        groups[group] = [int(index) for index in indices]
    if sorted(groups) != [0, 1, 2, 3]:
        raise ValueError(f"Expected source groups 0,1,2,3; got {sorted(groups)}")
    return groups


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        "Run each LightSwitch chunk under the three non-ground-truth endpoint contexts."
    )
    parser = add_general_config(parser)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--physical_context_table_path", required=True)
    parser.add_argument(
        "--sample_indices_by_group",
        required=True,
        help="Semicolon mapping such as 0:1,2;1:3,4;2:5,6;3:7,8.",
    )
    parser.add_argument("--raw_output_path", required=True)
    parser.add_argument("--manifest_output_path", default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    contexts = _read_context_table(args.physical_context_table_path)
    indices_by_group = _parse_indices_by_group(args.sample_indices_by_group)
    labels = {}
    plan = []

    for source_group, source_indices in sorted(indices_by_group.items()):
        if len(source_indices) != 16:
            raise ValueError(
                f"Expected 16 source chunks for group {source_group}; got {len(source_indices)}"
            )
        for source_index in source_indices:
            if not 0 <= source_index < len(metadata_rows):
                raise IndexError(f"Sample index out of range: {source_index}")
            row = metadata_rows[source_index]
            actual_group = int(row["context_group_id"])
            if actual_group != source_group:
                raise ValueError(
                    f"Sample {source_index} belongs to group {actual_group}, "
                    f"not requested group {source_group}"
                )
            labels[source_group] = str(row["causal_class"])
            plan.append(
                {
                    "source_index": source_index,
                    "source_group": source_group,
                    "source_label": str(row["causal_class"]),
                    "episode_index": int(row["episode_index"]),
                    "button_colors": row.get("covered_button_colors", []),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "target_groups": [group for group in sorted(contexts) if group != source_group],
                    "generation_seed": int(args.seed) + source_index * 1009,
                }
            )

    raw_root = Path(args.raw_output_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest_output_path)
        if args.manifest_output_path
        else raw_root.parent / "counterfactual_manifest.json"
    )
    _write_json(
        manifest_path,
        {
            "checkpoint": str(args.ckpt_path),
            "context_table": str(args.physical_context_table_path),
            "dataset_metadata": str(args.dataset_metadata_path),
            "frame_stride": int(args.frame_stride),
            "num_source_chunks": len(plan),
            "num_counterfactual_predictions": sum(len(item["target_groups"]) for item in plan),
            "comparison": "GT plus the three non-ground-truth endpoint contexts",
            "same_seed_across_contexts_for_each_source": True,
            "plan": plan,
        },
    )

    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)

    for item in plan:
        source_index = int(item["source_index"])
        for target_group in item["target_groups"]:
            target_group = int(target_group)
            target_label = labels.get(target_group)
            if target_label is None:
                target_label = next(
                    str(row["causal_class"])
                    for row in metadata_rows
                    if int(row["context_group_id"]) == target_group
                )
                labels[target_group] = target_label

            output_dir = raw_root / f"context_env{target_group}_{target_label}"
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_name = _default_pred_name(source_index, metadata_rows[source_index])
            pred_path = output_dir / pred_name
            if pred_path.exists() and args.skip_existing:
                print(
                    f"[skip] source={source_index} target_env={target_group} output={pred_path}",
                    flush=True,
                )
                continue

            partial_path = pred_path.with_suffix(".partial.mp4")
            partial_path.unlink(missing_ok=True)
            set_global_seed(int(item["generation_seed"]))
            sample = dataset[source_index]
            sample = prepare_sample_for_rollout(sample, source_index, pipe, args)
            sample["physical_context"] = torch.tensor(
                contexts[target_group],
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            sample["output_path"] = str(partial_path)
            print(
                f"[infer] source={source_index} true_env={item['source_group']} "
                f"target_env={target_group} seed={item['generation_seed']} output={pred_path}",
                flush=True,
            )
            generated_path = Path(_run_autoregressive(pipe=pipe, sample=sample, args=args))
            if not generated_path.exists():
                raise FileNotFoundError(f"Inference did not create {generated_path}")
            generated_path.replace(pred_path)
            torch.cuda.empty_cache()

    _write_json(
        raw_root.parent / "counterfactual_complete.json",
        {
            "complete": True,
            "num_source_chunks": len(plan),
            "num_counterfactual_predictions": sum(len(item["target_groups"]) for item in plan),
            "raw_output_path": str(raw_root),
        },
    )
    print(f"[done] raw={raw_root} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
