#!/usr/bin/env python3
"""Build an oracle informative-support protocol for shared-friction datasets."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from select_event80_informative_support import _choose_candidate, _measure_candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["sample_index"] = sample_index
            rows.append(row)
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_atomic(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0 or count > len(values):
        raise ValueError(f"Cannot choose {count} values from {len(values)} candidates.")
    if count == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    selected = [values[position] for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError(f"Even spacing produced duplicate selections: {selected}")
    return selected


def _physical_mu(row: dict[str, Any]) -> float:
    return float(row.get("physical_friction_mu", row["friction_mu"]))


def _query_rows_for_support(
    group_rows: list[dict[str, Any]],
    support: dict[str, Any],
    environment_slot: int,
) -> list[dict[str, Any]]:
    support_action = int(support["action_id"])
    support_background = int(support["environment_index"])
    action_ids = sorted({int(row["action_id"]) for row in group_rows})
    query_actions = [action_id for action_id in action_ids if action_id != support_action]
    if len(query_actions) != 9:
        raise ValueError(
            f"Expected nine non-support actions for group={support['context_group_id']}, "
            f"got {query_actions}."
        )
    backgrounds = sorted({int(row["environment_index"]) for row in group_rows})
    cross_backgrounds = [value for value in backgrounds if value != support_background]
    if not cross_backgrounds:
        raise ValueError(f"No cross-background queries for support={support['sample_index']}.")

    selected = []
    for offset, action_id in enumerate(query_actions):
        target_background = cross_backgrounds[(environment_slot + offset) % len(cross_backgrounds)]
        candidates = [
            row
            for row in group_rows
            if int(row["action_id"]) == action_id
            and int(row["environment_index"]) == target_background
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one query for group={support['context_group_id']} action={action_id} "
                f"background={target_background}; got {len(candidates)}."
            )
        selected.append(candidates[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--curriculum-order-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-count", type=int, default=15)
    parser.add_argument("--id-count", type=int, default=5)
    parser.add_argument("--ood-count", type=int, default=5)
    parser.add_argument("--minimum-displacement", type=float, default=25.0)
    parser.add_argument("--maximum-displacement", type=float, default=60.0)
    parser.add_argument("--endpoint-frames", type=int, default=5)
    parser.add_argument("--minimum-visible", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--evaluation-id",
        default="shared_friction_id5_ood5_k1_oracle_informative_support25_60_cross_background_v1",
    )
    args = parser.parse_args()

    if args.minimum_displacement >= args.maximum_displacement:
        raise ValueError("minimum-displacement must be below maximum-displacement.")

    rows = _read_jsonl(args.metadata_path)
    rows_by_group: dict[int, list[dict[str, Any]]] = {}
    physical_mu_by_group: dict[int, float] = {}
    for row in rows:
        group_id = int(row["context_group_id"])
        rows_by_group.setdefault(group_id, []).append(row)
        physical_mu_by_group.setdefault(group_id, _physical_mu(row))
    for group_id, group_rows in rows_by_group.items():
        if len(group_rows) != 50:
            raise ValueError(f"group={group_id} has {len(group_rows)} rows; expected 5x10=50.")

    order_payload = json.loads(args.curriculum_order_path.read_text(encoding="utf-8"))
    group_order = [int(value) for value in order_payload["group_order"]]
    active_ids = group_order[: int(args.active_count)]
    inactive_ids = [group_id for group_id in group_order if group_id not in set(active_ids)]
    active_sorted = sorted(active_ids, key=lambda group_id: (physical_mu_by_group[group_id], group_id))
    inactive_sorted = sorted(inactive_ids, key=lambda group_id: (physical_mu_by_group[group_id], group_id))
    id_groups = _evenly_spaced(active_sorted, int(args.id_count))
    ood_groups = _evenly_spaced(inactive_sorted, int(args.ood_count))
    domains = {**{group_id: "id" for group_id in id_groups}, **{group_id: "ood" for group_id in ood_groups}}

    candidate_rows = [row for group_id in id_groups + ood_groups for row in rows_by_group[group_id]]
    tasks = [
        (row, str(args.dataset_root), int(args.endpoint_frames), int(args.minimum_visible))
        for row in candidate_rows
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        measured = list(executor.map(_measure_candidate, tasks))
    for source, result in zip(candidate_rows, measured):
        for key in (
            "sample_index",
            "episode_index",
            "context_group_id",
            "environment_index",
            "environment_id",
            "action_id",
            "action_amplitude",
            "physical_friction_mu",
        ):
            if key in source:
                result[key] = source[key]
    measured.sort(
        key=lambda row: (
            int(row["context_group_id"]),
            int(row["environment_index"]),
            int(row["action_id"]),
        )
    )
    measured_by_group: dict[int, list[dict[str, Any]]] = {}
    for row in measured:
        measured_by_group.setdefault(int(row["context_group_id"]), []).append(row)

    records = []
    support_metadata = []
    id_queries: list[int] = []
    ood_queries: list[int] = []
    support_indices: list[int] = []
    midpoint = (float(args.minimum_displacement) + float(args.maximum_displacement)) / 2.0
    for slot, group_id in enumerate(id_groups + ood_groups):
        chosen, reason = _choose_candidate(
            measured_by_group[group_id],
            float(args.minimum_displacement),
            float(args.maximum_displacement),
        )
        support_index = int(chosen["sample_index"])
        support_row = rows[support_index]
        query_rows = _query_rows_for_support(rows_by_group[group_id], support_row, slot)
        query_indices = [int(row["sample_index"]) for row in query_rows]
        domain = domains[group_id]
        if domain == "id":
            id_queries.extend(query_indices)
        else:
            ood_queries.extend(query_indices)
        support_indices.append(support_index)
        support_metadata.append({key: value for key, value in support_row.items() if key != "sample_index"})
        records.append(
            {
                "domain": domain,
                "environment_id": f"shared_friction_group_{group_id}",
                "context_group_id": group_id,
                "friction_mu": physical_mu_by_group[group_id],
                "support_indices": [support_index],
                "query_indices": query_indices,
                "selection": {
                    "support_action_id": int(support_row["action_id"]),
                    "support_background_index": int(support_row["environment_index"]),
                    "final_displacement_px": float(chosen["final_displacement_px"]),
                    "distance_to_target_midpoint_px": abs(
                        float(chosen["final_displacement_px"]) - midpoint
                    ),
                    "reason": reason,
                },
                "queries": [
                    {
                        "sample_index": int(row["sample_index"]),
                        "action_id": int(row["action_id"]),
                        "background_index": int(row["environment_index"]),
                    }
                    for row in query_rows
                ],
            }
        )

    manifest = {
        "evaluation_id": args.evaluation_id,
        "protocol": (
            "oracle informative K=1 support selected from all five backgrounds and ten actions; "
            "support action is excluded from nine read-only cross-background action queries"
        ),
        "support_size": 1,
        "selection_uses_ground_truth": True,
        "id_definition": f"first {args.active_count} groups in curriculum order",
        "ood_definition": "groups outside the active curriculum prefix",
        "active_context_group_ids": active_ids,
        "selection_target": {
            "metric": "Euclidean distance between median block centroids in first and last endpoint frames",
            "minimum_displacement_px": float(args.minimum_displacement),
            "maximum_displacement_px": float(args.maximum_displacement),
            "target_midpoint_px": midpoint,
            "endpoint_frames": int(args.endpoint_frames),
            "minimum_visible_endpoint_frames": int(args.minimum_visible),
            "require_visible_final_endpoint": True,
        },
        "environments": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "support_query_manifest.json", manifest)
    _write_json(
        args.output_dir / "support_selection_summary.json",
        {
            "evaluation_id": args.evaluation_id,
            "candidate_count": len(candidate_rows),
            "selected_environment_count": len(records),
            "environments": records,
        },
    )
    _write_jsonl(args.output_dir / "support_candidates.jsonl", measured)
    _write_jsonl(args.output_dir / "support_metadata.jsonl", support_metadata)
    all_queries = id_queries + ood_queries
    env_text = "\n".join(
        [
            f"EVALUATION_ID={args.evaluation_id}",
            f"ACTIVE_CONTEXT_IDS={','.join(map(str, active_ids))}",
            f"ID_ENVIRONMENTS={','.join(map(str, id_groups))}",
            f"OOD_ENVIRONMENTS={','.join(map(str, ood_groups))}",
            f"SUPPORTS={','.join(map(str, support_indices))}",
            f"ID_QUERIES={','.join(map(str, id_queries))}",
            f"OOD_QUERIES={','.join(map(str, ood_queries))}",
            f"ALL_QUERIES={','.join(map(str, all_queries))}",
            "",
        ]
    )
    _write_atomic(args.output_dir / "indices.env", env_text)
    _write_atomic(
        args.output_dir / "README.md",
        "# Shared-friction oracle informative-support protocol\n\n"
        "Five ID and five OOD friction groups are selected uniformly in physical-friction order. "
        "For each group, one support is selected from all 5 backgrounds x 10 actions by GT endpoint "
        "displacement in the 25-60 px band. The support action is excluded, and the remaining nine "
        "actions are evaluated on backgrounds other than the support background.\n",
    )
    print(
        f"[protocol] id={id_groups} ood={ood_groups} supports={support_indices} "
        f"queries={len(all_queries)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
