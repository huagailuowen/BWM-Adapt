#!/usr/bin/env python3
"""Re-score cached Event80 action candidates without decoding videos again."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_event80_benchmark import aggregate_action_rows
from wan_video_action.metrics.object_action import RectangleTarget, evaluate_action_choice


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def _rate(summary: dict[str, Any], section: str, key: str = "headline_task_success_rate") -> Any:
    return summary.get("by_target", {}).get(section, {}).get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    args = parser.parse_args()

    root = args.metrics_root.resolve()
    targets = [
        RectangleTarget.from_mapping(value)
        for value in yaml.safe_load(args.target_config.read_text())["targets"]
    ]
    comparison: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for method_dir in sorted((root / "methods").iterdir()):
        candidates_path = method_dir / "action_candidates.jsonl"
        summary_path = method_dir / "action_summary.json"
        decisions_path = method_dir / "action_decisions.jsonl"
        if not candidates_path.is_file() or not summary_path.is_file():
            continue
        backup_summary = method_dir / "action_summary.pre_minimum_long.json"
        backup_decisions = method_dir / "action_decisions.pre_minimum_long.jsonl"
        if not backup_summary.exists():
            shutil.copy2(summary_path, backup_summary)
        if decisions_path.exists() and not backup_decisions.exists():
            shutil.copy2(decisions_path, backup_decisions)
        old_summary = json.loads(backup_summary.read_text())
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _jsonl(candidates_path):
            grouped[str(row["environment_id"])].append(row)
        decisions: list[dict[str, Any]] = []
        for environment_id, candidates in sorted(grouped.items()):
            first = candidates[0]
            for target in targets:
                decision = evaluate_action_choice(
                    candidates, target, expected_action_count=len(candidates)
                )
                decision.update(
                    {
                        "method": method_dir.name,
                        "environment_id": environment_id,
                        "domain": first["domain"],
                        "mu_index": first["mu_index"],
                        "friction_mu": first["friction_mu"],
                    }
                )
                decisions.append(decision)
        new_summary = aggregate_action_rows(decisions)
        summaries[method_dir.name] = new_summary
        _write_jsonl(decisions_path, decisions)
        _write_json(summary_path, new_summary)
        comparison.append(
            {
                "method": method_dir.name,
                "old_overall": old_summary["overall"]["headline_task_success_rate"],
                "new_overall": new_summary["overall"]["headline_task_success_rate"],
                "old_long": _rate(old_summary, "long_push"),
                "new_long": _rate(new_summary, "long_push"),
                "new_random_overall": new_summary["overall"]["uniform_random_success_rate"],
                "new_random_long": _rate(
                    new_summary, "long_push", "uniform_random_success_rate"
                ),
            }
        )
    scoreboard_path = root / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text())
    for row in scoreboard:
        summary = summaries[str(row["method"])]
        overall = summary["overall"]
        row.update(
            {
                "action_success_all": overall["headline_task_success_rate"],
                "action_success_id": summary["by_domain"]["id"][
                    "headline_task_success_rate"
                ],
                "action_success_ood": summary["by_domain"]["ood"][
                    "headline_task_success_rate"
                ],
                "action_selected_oracle_all": overall["selected_is_oracle_rate"],
                "action_mean_regret_all": overall["mean_regret"],
                "action_mean_coverage_all": overall["mean_action_coverage"],
                "action_eligible_decisions_all": overall[
                    "oracle_reachable_complete_count"
                ],
                "action_uniform_random_success_all": overall[
                    "uniform_random_success_rate"
                ],
            }
        )
    _write_json(scoreboard_path, scoreboard)
    scoreboard_csv = root / "scoreboard.csv"
    with scoreboard_csv.with_suffix(".csv.tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scoreboard[0]))
        writer.writeheader()
        writer.writerows(scoreboard)
    scoreboard_csv.with_suffix(".csv.tmp").replace(scoreboard_csv)

    benchmark_path = root / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text())
    for method, summary in summaries.items():
        benchmark[method]["action_selection"] = summary
    _write_json(benchmark_path, benchmark)

    _write_json(
        root / "minimum_reaching_long_rescore.json",
        {
            "target_config": str(args.target_config.resolve()),
            "method_count": len(comparison),
            "methods": comparison,
        },
    )
    csv_path = root / "minimum_reaching_long_rescore.csv"
    with csv_path.with_suffix(".csv.tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    csv_path.with_suffix(".csv.tmp").replace(csv_path)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
