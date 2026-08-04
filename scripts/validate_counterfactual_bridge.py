#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from wan_video_action.counterfactual_bridge import (
    CounterfactualSourceBank,
    environment_mixture,
    nonlinear_bridge_alpha,
    parse_noise_bands,
    sample_nonlinear_bridge_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    args = parser.parse_args()

    with Path(args.metadata).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    bank = CounterfactualSourceBank(args.manifest, args.raw_root)
    missing = []
    causal_coverage = 0
    for index in sorted(bank.available_indices):
        row = rows[index]
        groups = bank._candidate_source_groups(index, row)
        causal_coverage += bool(groups)
        for group in groups:
            path = bank._video_path(index, row, group)
            if not path.is_file():
                missing.append(str(path))

    q_half = nonlinear_bridge_alpha(0.5, 5.0)
    if q_half <= 0.95:
        raise AssertionError(f"Expected q(0.5)>0.95, got {q_half}.")
    if abs(sum(environment_mixture(2, q_half)) - 1.0) > 1e-7:
        raise AssertionError("Environment mixture does not sum to one.")
    bands = parse_noise_bands("0.90:1.00:0.20,0.70:0.90:0.60,0.55:0.70:0.20")
    counts = {"endpoint": 0, "near_global": 0, "interior": 0}
    rng = random.Random(17)
    for _ in range(100000):
        condition = sample_nonlinear_bridge_condition(rng)
        counts[condition.kind] += 1
    fractions = {key: value / 100000.0 for key, value in counts.items()}
    expected = {"endpoint": 0.4, "near_global": 0.3, "interior": 0.3}
    for key, target in expected.items():
        if abs(fractions[key] - target) > 0.01:
            raise AssertionError(f"Sampling quota drift for {key}: {fractions[key]}.")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} Teacher videos are missing; first={missing[0]}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "teacher_chunks": len(bank.available_indices),
                "causal_source_coverage": causal_coverage,
                "q_at_half": q_half,
                "sampling_fractions": fractions,
                "noise_bands": bands,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
