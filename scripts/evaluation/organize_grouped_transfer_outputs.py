#!/usr/bin/env python3
"""Organize flat predictions using a grouped support/query manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--flat-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-slug", required=True)
    return parser.parse_args()


def _safe_label(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "env"


def _load_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("environments", value.get("records"))
    if not isinstance(value, list):
        raise TypeError(f"Expected a list-like grouped manifest in {path}.")
    return value


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.manifest)
    prediction_pattern = re.compile(r"sample(\d+)_")
    predictions: dict[int, Path] = {}
    for path in args.flat_root.rglob("*.mp4"):
        match = prediction_pattern.search(path.name)
        if match:
            predictions[int(match.group(1))] = path

    transfer_root = args.output_root / "transfer"
    raw_root = transfer_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    plan = []
    for row in rows:
        supports = [
            int(value)
            for value in row.get(
                "support_indices", [row.get("source_index")]
            )
            if value is not None
        ]
        queries = [
            int(value)
            for value in row.get("query_indices", row.get("target_indices", ()))
        ]
        if not supports or not queries:
            raise ValueError(f"Manifest row lacks support/query indices: {row}")
        display_source = supports[0]
        environment = row.get(
            "causal_class",
            row.get(
                "source_mass_ratio",
                row.get("environment_id", row.get("context_group_id", display_source)),
            ),
        )
        destination = raw_root / (
            f"source{display_source:04d}_{_safe_label(environment)}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        for query in queries:
            if query not in predictions:
                raise FileNotFoundError(
                    f"Missing flat prediction for sample {query} under {args.flat_root}"
                )
            target = destination / predictions[query].name
            if target.exists():
                continue
            try:
                os.link(predictions[query], target)
            except OSError:
                shutil.copy2(predictions[query], target)
        plan.append(
            {
                **row,
                "source_index": display_source,
                "protocol_source_index": row.get("source_index"),
                "support_indices": supports,
                "target_indices": queries,
                "method": args.method_slug,
            }
        )

    plan_path = transfer_root / "transfer_plan.json"
    temporary = plan_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(plan_path)
    print(f"[done] environments={len(plan)} transfer_plan={plan_path}", flush=True)


if __name__ == "__main__":
    main()
