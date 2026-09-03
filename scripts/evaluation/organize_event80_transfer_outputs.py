#!/usr/bin/env python3
"""Organize flat Event80 predictions into the formal source/query grid layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil


SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-query-manifest", type=Path, required=True)
    parser.add_argument("--flat-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-slug", required=True)
    args = parser.parse_args()

    payload = json.loads(args.support_query_manifest.read_text(encoding="utf-8"))
    environments = payload.get("environments", payload)
    if not isinstance(environments, list) or not environments:
        raise ValueError("Support/query manifest has no environments")
    predictions = {}
    for path in args.flat_root.glob("*.mp4"):
        match = SAMPLE_RE.search(path.name)
        if match:
            predictions[int(match.group(1))] = path

    raw_root = args.output_root / "transfer/raw"
    plan = []
    for environment in environments:
        supports = [int(value) for value in environment["support_indices"]]
        queries = [int(value) for value in environment["query_indices"]]
        if len(supports) != 1:
            raise ValueError("Event80 repaired-baseline protocol requires K=1")
        source = supports[0]
        destination = raw_root / f"source{source:04d}_{args.method_slug}"
        destination.mkdir(parents=True, exist_ok=True)
        for query in queries:
            if query not in predictions:
                existing = list(destination.glob(f"sample{query:04d}_*.mp4"))
                if len(existing) == 1:
                    continue
                raise FileNotFoundError(f"Missing flat prediction for sample {query}")
            target = destination / predictions[query].name
            if not target.exists():
                try:
                    os.link(predictions[query], target)
                except OSError:
                    shutil.copy2(predictions[query], target)
        plan.append(
            {
                "source_index": source,
                "source_sample_id": environment.get("support_sample_ids", [None])[0],
                "target_indices": queries,
                "target_sample_ids": environment.get("query_sample_ids", []),
                "domain": environment.get("domain", "unknown"),
            }
        )

    transfer_plan = args.output_root / "transfer/transfer_plan.json"
    transfer_plan.parent.mkdir(parents=True, exist_ok=True)
    temporary = transfer_plan.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(transfer_plan)
    shutil.copy2(
        args.support_query_manifest,
        args.output_root / "support_query_manifest.json",
    )
    print(f"[done] environments={len(plan)} transfer_plan={transfer_plan}", flush=True)


if __name__ == "__main__":
    main()
