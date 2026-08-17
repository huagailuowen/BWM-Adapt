#!/usr/bin/env python3
"""Additive support/query inference entry point for registered methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.methods.common import (
    CommandApplication,
    InferenceBundle,
    IntegrationRequiredError,
    SupportQueryEvaluator,
    load_method_config,
    resolve_factory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer with a registered method and disjoint support/query data."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the config without requiring a Wan adapter.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Resolve the configured application and print its command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_method_config(args.config)
    print(json.dumps(loaded.summary(), indent=2, sort_keys=True))
    if args.dry_run and not args.plan:
        return
    try:
        factory = resolve_factory(loaded.data, "infer")
    except IntegrationRequiredError as exc:
        raise SystemExit(str(exc)) from exc
    bundle = factory(loaded)
    if args.plan:
        if not isinstance(bundle, CommandApplication):
            raise SystemExit("--plan requires a CommandApplication factory.")
        print(json.dumps(bundle.describe(), indent=2, sort_keys=True))
        return
    if isinstance(bundle, InferenceBundle):
        result = SupportQueryEvaluator(bundle).run()
        payload = {
            "completed_episodes": result.completed_episodes,
        }
    elif isinstance(bundle, CommandApplication):
        print(json.dumps(bundle.describe(), indent=2, sort_keys=True))
        result = bundle.run()
        payload = {
            "phase": result.phase,
            "method_slug": result.method_slug,
            "return_code": result.return_code,
            "command": list(result.command),
            "elapsed_seconds": result.elapsed_seconds,
            "resource_summary_path": result.resource_summary_path,
        }
    else:
        raise TypeError(
            "Infer factory must return InferenceBundle or CommandApplication."
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
