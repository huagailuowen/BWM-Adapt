#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _latest_contexts(path: str | Path) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for row in _read_jsonl(path):
        source_index = int(row["sample_index"])
        inner_step = int(row["inner_step"])
        if source_index not in latest or inner_step > int(latest[source_index]["inner_step"]):
            latest[source_index] = row
    return latest


def parse_args():
    parser = argparse.ArgumentParser(
        "Roll out every window of planned episodes using one Stage2-learned context per environment."
    )
    parser = add_general_config(parser)
    parser.add_argument("--plan_path", required=True)
    parser.add_argument("--trajectory_path", required=True)
    parser.add_argument("--source_indices", required=True)
    parser.add_argument("--raw_output_path", required=True)
    parser.add_argument("--episode_roles", default="support,transfer")
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    plan = json.loads(Path(args.plan_path).read_text(encoding="utf-8"))
    latest = _latest_contexts(args.trajectory_path)
    selected_sources = set(_parse_sample_indices(args.source_indices))
    roles = tuple(role.strip() for role in str(args.episode_roles).split(",") if role.strip())
    if not selected_sources:
        raise ValueError("--source_indices is empty.")
    if not roles:
        raise ValueError("--episode_roles is empty.")

    environments = [
        item for item in plan["environments"] if int(item["source_index"]) in selected_sources
    ]
    planned_sources = {int(item["source_index"]) for item in environments}
    missing = selected_sources - planned_sources
    if missing:
        raise KeyError(f"Sources are absent from the episode plan: {sorted(missing)}")
    missing = selected_sources - set(latest)
    if missing:
        raise KeyError(f"No final Stage2 context for sources: {sorted(missing)}")

    raw_root = Path(args.raw_output_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)

    for environment in environments:
        source_index = int(environment["source_index"])
        context_row = latest[source_index]
        context = torch.tensor(
            context_row["context_flat"],
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        print(
            f"[environment] source={source_index} mu={environment['friction_mu']} "
            f"inner_step={context_row['inner_step']} roles={roles}",
            flush=True,
        )
        seen: set[int] = set()
        for role in roles:
            key = f"{role}_indices"
            if key not in environment:
                raise KeyError(f"Episode plan has no {key!r} for source={source_index}.")
            for target_index in environment[key]:
                target_index = int(target_index)
                if target_index in seen:
                    continue
                seen.add(target_index)
                row = dataset.metadata[target_index] if hasattr(dataset, "metadata") else None
                if row is None:
                    # build_infer_dataset indexes in exactly the metadata JSONL order.
                    metadata_rows = _read_jsonl(args.dataset_metadata_path)
                    row = metadata_rows[target_index]
                pred_path = raw_root / _default_pred_name(target_index, row)
                if pred_path.exists() and args.skip_existing:
                    print(f"[skip] {pred_path}", flush=True)
                    continue
                sample = dataset[target_index]
                sample = prepare_sample_for_rollout(sample, target_index, pipe, args)
                sample["physical_context"] = context
                sample["output_path"] = str(pred_path)
                print(
                    f"[infer] source={source_index} role={role} target={target_index} "
                    f"output={pred_path}",
                    flush=True,
                )
                _run_autoregressive(pipe=pipe, sample=sample, args=args)
                torch.cuda.empty_cache()

    print(f"[done] raw={raw_root} sources={sorted(selected_sources)}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
