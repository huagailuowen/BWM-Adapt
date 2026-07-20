#!/usr/bin/env python3
"""Build balanced support rows for a six-friction random-start C32 pool."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-mus", default="0.005,0.01,0.02,0.05,0.1,0.15")
    parser.add_argument("--contexts-per-group", type=int, default=50)
    parser.add_argument("--min-displacement-m", type=float, default=0.2)
    parser.add_argument("--max-displacement-m", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_manifest)
    episode_rows = read_jsonl(args.episode_metadata)
    episode_metadata = {int(row["episode_index"]): row for row in episode_rows}
    rows_by_mu = defaultdict(list)
    for row in source_rows:
        rows_by_mu[float(row["friction_mu"])].append(row)

    source_mus = sorted(rows_by_mu)
    target_mus = [float(value) for value in args.target_mus.split(",") if value.strip()]
    output_rows = []
    summary = []

    for target_index, target_mu in enumerate(target_mus):
        source_mu = min(source_mus, key=lambda value: abs(value - target_mu))
        eligible = []
        for row in rows_by_mu[source_mu]:
            metadata = episode_metadata[int(row["episode_index"])]
            displacement = float(metadata["metrics"]["final_displacement_m"])
            if args.min_displacement_m < displacement < args.max_displacement_m:
                eligible.append((row, displacement))
        if not eligible:
            raise ValueError(f"No eligible support episodes for target_mu={target_mu} source_mu={source_mu}.")

        rng = random.Random(args.seed + target_index * 10_007)
        balanced = []
        while len(balanced) < args.contexts_per_group:
            cycle = list(eligible)
            rng.shuffle(cycle)
            balanced.extend(cycle)

        for pool_index, (source_row, displacement) in enumerate(balanced[: args.contexts_per_group]):
            sample_index = len(output_rows)
            row = dict(source_row)
            row.update(
                {
                    "sample_id": f"c32pool:mu{target_mu:.6f}:sample{pool_index:02d}",
                    "pool_target_mu": target_mu,
                    "pool_source_mu": source_mu,
                    "pool_index": pool_index,
                    "support_displacement_m": displacement,
                    "initial_context_seed": args.seed + sample_index * 1_000_003,
                }
            )
            output_rows.append(row)

        summary.append(
            {
                "target_mu": target_mu,
                "source_mu": source_mu,
                "eligible_support_episodes": len(eligible),
                "contexts": args.contexts_per_group,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(args.output)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(output_rows), "groups": summary}, indent=2))


if __name__ == "__main__":
    main()
