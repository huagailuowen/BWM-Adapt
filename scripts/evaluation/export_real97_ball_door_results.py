#!/usr/bin/env python3
"""Publish text-only real97 metric snapshots without decoding videos or running models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


EVALUATION = "real97_train_support_train_test_query_v1"
RELEASES = {
    "ball": "first_reach_pair_overlap_20260913_v1",
    "door": "consecutive_closure_pair_overlap_20260913_v1",
}
BENCHMARKS = {"ball": "ball_friction", "door": "door_close"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write("\n")


def save_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=True) + "\n")


def save_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def export(source, results):
    loaded = {}
    provenance = {}

    def read(relative):
        if relative not in loaded:
            raw = (source / relative).read_bytes()
            provenance[relative] = {
                "source": str(source / relative),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            loaded[relative] = json.loads(raw)
        return loaded[relative]

    summary = read("summary_provisional.json")
    manifest = read("manifest.json")
    base_protocol = read("protocol.json")
    ball_pairs = read("ball_action_first_reach_successor_pair_overlap.json")
    ball_reach = read("ball_action_first_reach_blue_minus30.json")
    ball_top2 = read("ball_action_top2_overlap.json")
    door_audit = read("door_calibration_audit.json")
    completion = {name: read(name) for name in ["cpu_complete.json", "lpips_complete.json"]}
    csv_rows = list(csv.DictReader(io.StringIO((source / "image_object_metrics.csv").read_text())))
    destinations = {
        task: results / BENCHMARKS[task] / EVALUATION
        for task in RELEASES
    }
    for task, destination in destinations.items():
        target = destination / "comparisons" / "metrics" / RELEASES[task]
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing release: {target}")

    written = []
    for task, destination in destinations.items():
        release = RELEASES[task]
        target = destination / "comparisons" / "metrics" / release
        target.mkdir(parents=True)
        image_object = [row for row in summary["image_object"] if row["task"] == task]
        old_action = [row for row in summary["action"] if row["task"] == task]
        if task == "ball":
            action = ball_pairs
            action_scores = {row["method"]: row["mean_overlap_score"] for row in action["methods"]}
            action_counts = {row["method"]: row["environments"] for row in action["methods"]}
            action_protocol = ball_pairs["protocol"]
            action_note = (
                "Reach means farthest ball-center x >= right blue marker x - 30 original pixels. "
                "GT and model independently scan numerical levels in ascending order. Each selects "
                "{first reaching level, first reaching level + 1}; score is intersection size / 2. "
                "No reaching level scores zero. At level 10, {10,11} is a formal scoring pair only: "
                "level 11 has no recorded or generated rollout. Reaching at the successor level is "
                "not independently required. This is distinct from the older closest-two metric."
            )
            save_json(target / "first_reach_candidates.json", ball_reach)
            save_json(target / "legacy" / "closest_two_overlap.json", ball_top2)
            save_json(target / "legacy" / "closest_one_action.json", old_action)
        else:
            action = {"metric": "first_consecutive_closed_pair_overlap", "methods": old_action}
            action_scores = {row["method"]: row["macro_score"] for row in old_action}
            action_counts = {row["method"]: row["environment_count"] for row in old_action}
            action_protocol = {
                "prediction": "First ascending adjacent pair where both predicted levels close the door",
                "score": "Intersection with the user-specified GT preferred pair, divided by two",
                "no_pair": "score zero",
                "gt_preferred_pairs": base_protocol["door_prefer"],
                "action_excluded_environments": base_protocol["door_action_excluded"],
                "closure_classifier": base_protocol["door_closure"],
            }
            action_note = (
                "Scan levels upward and choose the first adjacent pair for which both predictions "
                "close the door. Score the intersection with the fixed user-specified GT pair / 2. "
                "door-12u-Half-11d-Half is excluded from action selection, but remains in image and "
                "object metrics. The held-out door-4d level-3 video is visibly open; its measured "
                "GT pair is {4,5}, while the agreed nominal scoring pair remains {3,4}. This known "
                "sample/protocol discrepancy is retained, not silently relabeled."
            )
            save_json(target / "calibration_audit.json", door_audit)

        protocol = {
            "evaluation_id": EVALUATION,
            "release": release,
            "task": task,
            "source_scoring_config": "configs/evaluation/real97_ball_door_scores_20260912_v2.json",
            "source_tracking_status": summary["status"],
            "publication_status": "archived_computed_metrics_not_a_claim_of_completed_heldout_tracking_audit",
            "action_selection": action_protocol,
            "image_metrics": ["PSNR", "SSIM", "LPIPS-AlexNet"],
            "object_metrics": ["ADE_px", "FDE_px", "missing_penalized_ADE_px", "tracking_coverage"]
            + (["farthest_x_error_px"] if task == "ball" else []),
            "task_geometry": base_protocol["tasks"][task],
            "frame_policy": "Recorded evaluation_frame_indices; exclude conditioning frame and repeated tail padding",
            "image_geometry": "Exclude letterbox bars; compare corresponding content crops",
            "object_geometry": "Inverse-letterboxed native image pixels; Door uses tracked door edge, not a physical angle",
            "ball_exit_policy": "Retain last-visible position only after confirmed right-side exit",
            "aggregation": "Report train and test separately; scoreboard uses test environment macro means",
            "paired_gt_only": True,
            "counterfactual_action_sweeps_scored": False,
            "checkpoint_steps": summary["checkpoint_steps"][task],
            "fairness_note": summary["fairness_note"],
            "media_included": False,
            "new_inference_or_metric_recomputation": False,
        }
        save_json(target / "protocol.json", protocol)
        save_json(target / "action_decisions.json", action)
        save_json(target / "image_object_summary.json", image_object)
        save_csv(target / "image_object_metrics.csv", [row for row in csv_rows if row["task"] == task])

        scoreboard = []
        for row in image_object:
            if row["split"] != "test":
                continue
            method = row["method"]
            values = row["environment_macro"]
            scoreboard.append({
                "method": method,
                "checkpoint_step": summary["checkpoint_steps"][task]["ours" if method.startswith("ours_") else method],
                "test_queries": row["queries"],
                "test_environments": row["environments"],
                **{name: values.get(name) for name in ["psnr", "ssim", "lpips", "ade_px", "fde_px", "missing_penalized_ade_px", "prediction_coverage", "peak_x_error_px"]},
                "action_environments": action_counts[method],
                "action_pair_overlap": action_scores[method],
            })
        save_json(target / "scoreboard.json", scoreboard)
        save_csv(target / "scoreboard.csv", scoreboard)

        frozen = [row for row in manifest["queries"] if row["task"] == task]
        save_jsonl(target / "frozen_query_manifest.jsonl", frozen)
        observations = []
        per_method = {name: [] for name in manifest["methods"]}
        for entry in frozen:
            query = read(f"queries/{task}/{entry['key']}.json")
            lpips = read(f"lpips/{task}/{entry['key']}.json")
            identity = {key: query[key] for key in ["key", "environment", "split", "level", "sample_id", "frame_ids"]}
            observations.append({
                **identity,
                "marker": query["marker"],
                "trajectory_summary": query["trajectory_summary"],
                "full_episode_peak_in_window": query.get("full_episode_peak_in_window"),
                "full_episode_end_in_window": query.get("full_episode_end_in_window"),
            })
            for method in per_method:
                per_method[method].append({
                    **identity,
                    **query["methods"][method],
                    "lpips": lpips[method],
                })
        save_jsonl(target / "object_observations.jsonl", observations)

        for method, rows in per_method.items():
            step = summary["checkpoint_steps"][task]["ours" if method.startswith("ours_") else method]
            method_target = destination / "methods" / method / f"step_{step}" / "metrics" / release
            save_jsonl(method_target / "per_query.jsonl", rows)
            save_json(method_target / "summary.json", [row for row in image_object if row["method"] == method])
            if task == "ball":
                decisions = [row for row in ball_pairs["rows"] if row["method"] == method]
            else:
                decisions = next(row for row in old_action if row["method"] == method)
            save_json(method_target / "action_decisions.json", decisions)

        task_inputs = {
            name: metadata for name, metadata in provenance.items()
            if not name.startswith(("queries/", "lpips/"))
            or name.startswith((f"queries/{task}/", f"lpips/{task}/"))
        }
        save_json(target / "provenance.json", {
            "source_directory": str(source),
            "source_files": task_inputs,
            "completion_records": completion,
            "frozen_inference_outputs": base_protocol["tasks"][task],
            "all_videos_remain_in_original_output_directories": True,
            "training_weights_dataset_files_and_logs_included": False,
        })
        readme = (
            f"# {BENCHMARKS[task]}: {release}\n\n"
            "This is a text-only snapshot of the existing predictions and computed metrics. "
            "No inference, video decoding, or model training is run by the exporter.\n\n"
            "## Score definitions\n\n"
            f"{action_note}\n\n"
            "Image/object metrics keep train and held-out test queries separate. The main "
            "scoreboard uses equal environment weights on held-out queries. Conditioning "
            "frames, repeated tail padding, and letterbox bars are excluded as recorded in "
            "the frozen evaluation manifest. Counterfactual action sweeps have no paired GT "
            "and are not included.\n\n"
            "## Files\n\n"
            "- `scoreboard.csv` / `scoreboard.json`: held-out image, object, and action results.\n"
            "- `image_object_summary.json`: query means, environment means, and valid counts.\n"
            "- `action_decisions.json`: per-environment action-selection decisions.\n"
            "- `object_observations.jsonl`: observed targets, peak positions, and closure labels.\n"
            "- `frozen_query_manifest.jsonl`: query identities, frame masks, and source references.\n"
            "- `protocol.json` / `provenance.json`: conventions, caveats, source paths, hashes.\n"
            "- Per-method per-query metrics are under the evaluation's `methods/` directory.\n\n"
            "## Limitations\n\n"
            f"Source audit status: `{summary['status']}`. Archiving does not upgrade this "
            "status or imply an exhaustive manual audit of every held-out tracking result.\n\n"
            f"{summary['fairness_note']} Door Ours uses step 3715, while Door baselines use "
            "step 5500; all Ball methods use step 5500.\n\n"
            "Only text metrics and their reproducibility records are published. Videos, "
            "images, model weights, latent checkpoints, logs, and dataset files are excluded.\n"
        )
        with (target / "README.md").open("x") as stream:
            stream.write(readme)
        written.append({"task": task, "directory": str(target), "queries": len(frozen), "methods": len(per_method)})
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/eval_real97_ball_door_scores_20260912_v2"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    export(args.source, args.results_root)
