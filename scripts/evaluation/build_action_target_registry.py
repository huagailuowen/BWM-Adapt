#!/usr/bin/env python3
"""Build eligible environment/action registries from frozen GT metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.evaluation.io import write_json_atomic, write_jsonl_atomic  # noqa: E402
from wan_video_action.metrics.action_targets import (  # noqa: E402
    build_target_registry,
    flatten_eligible_environments,
    flatten_gt_action_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"target config must be a mapping: {args.config}")
    registry = build_target_registry(config)
    eligible = flatten_eligible_environments(registry)
    gt_actions = flatten_gt_action_outcomes(registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "target_registry.json", registry)
    write_jsonl_atomic(args.output_dir / "eligible_environments.jsonl", eligible)
    write_jsonl_atomic(args.output_dir / "gt_action_outcomes.scorer_only.jsonl", gt_actions)
    print(
        f"wrote {len(eligible)} target/environment pairs across "
        f"{len(registry['target_areas'])} targets and {len(gt_actions)} GT actions "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
