#!/usr/bin/env python3
"""Inspect the locked Event80 TTT-KVB protocol before submitting jobs."""

from __future__ import annotations

import argparse
import json

from omegaconf import OmegaConf

from wan_video_action.methods.baselines.ttt_kvb.config import load_event80_config
from wan_video_action.methods.baselines.ttt_kvb.event80 import Event80Index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-seed", type=int, default=20260815)
    parser.add_argument("--print-resolved", action="store_true")
    args = parser.parse_args()

    config = load_event80_config(args.config)
    index = Event80Index(
        config.benchmark.metadata_path,
        environment_key=config.benchmark.environment_key,
        action_key=config.benchmark.action_key,
    )
    summary = {
        "benchmark": config.benchmark.id,
        "method": config.method.id,
        "environment_count": len(index.environment_ids),
        "actions_per_environment": sorted(
            {len(index.by_environment[value]) for value in index.environment_ids}
        ),
        "window": [config.benchmark.window.start, config.benchmark.window.end],
        "gpus": config.resources.gpus,
        "runner": config.runtime.runner_class,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.print_resolved:
        print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
