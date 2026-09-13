#!/usr/bin/env python3
"""Freeze Ball/Door support selection and matched query manifests on a CPU node."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
from real97_trial_common import ROOT, read_jsonl, require_compute, write_json, write_jsonl, native_frame


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("ball", "door"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    import cv2
    cv2.setNumThreads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_text = args.config.read_text()
    frozen_config = output / "experiment.json"
    if frozen_config.exists() and frozen_config.read_text() != config_text:
        raise RuntimeError("Refusing to change an existing prepared experiment")
    if (output / "plan.json").is_file():
        print("[prepared] Existing completed preparation retained", output, flush=True)
        return
    frozen_config.write_text(config_text)
    config = json.loads(config_text)
    setting = config["tasks"][args.task]
    protocol = json.loads(resolve(config["source_protocol"]).read_text())
    protocol["tasks"][args.task] = setting
    training = resolve(setting["training_run"])
    manifest = training / "input_manifest"
    summary = json.loads((manifest / "manifest_summary.json").read_text())
    dataset_root = Path(summary["dataset_base_path"])
    reference = output / "reference"
    reference.mkdir(exist_ok=True)
    source_model = training / setting["model_file"]
    if not (training / ("." + setting["model_file"] + ".complete")).is_file():
        raise RuntimeError("Refusing an incomplete source checkpoint")
    pinned_model = reference / "model.safetensors"
    if not pinned_model.exists():
        os.link(source_model, pinned_model)
    elif not os.path.samefile(source_model, pinned_model):
        raise RuntimeError("Pinned model identity differs from the requested checkpoint")
    for source, name in (
        (training / setting["table_file"], "context_table.json"),
        (manifest / "action_stats.json", "action_stats.json"),
        (training / "submitted_config.yaml" if (training / "submitted_config.yaml").is_file()
         else resolve(setting["training_config"]), "training_config.yaml"),
        (manifest / "manifest_summary.json", "manifest_summary.json"),
    ):
        data = source.read_bytes()
        target = reference / name
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f"Frozen reference changed: {name}")
        if not target.exists():
            target.write_bytes(data)
    table = json.loads((reference / "context_table.json").read_text())
    if len(table["records"]) != setting["expected_environments"]:
        raise RuntimeError("Checkpoint C table does not contain the expected environments")
    write_json(reference / "provenance.json", {
        "setting": setting, "manifest": summary, "protocol": protocol,
        "model_and_table_same_step": True, "source_checkpoint": str(source_model),
    })
    by_episode = defaultdict(list)
    for split in ("train", "test"):
        for row in read_jsonl(manifest / (split + ".jsonl")):
            if row["dataset_split"] != split or row["action_semantics"] != setting["action_type"]:
                raise RuntimeError("Frozen training split/action semantics mismatch")
            by_episode[(row["environment"], int(row["episode_index"]), split)].append(row)
    outcomes = {}
    for row in read_jsonl(resolve(config["calibrated_outcomes"])):
        if row["task"] == args.task:
            outcomes[(row["environment"], int(row["episode_index"]), row["split"])] = row
    events = {}
    for row in read_jsonl(manifest / "chunk_events.jsonl"):
        env = row["environment"].removesuffix("_lerobot")
        events[(env, int(row["episode_index"]), row["split"])] = row
    tracks = {}
    rng = random.Random(config["seed"])

    def selected_window(key, support=False):
        rows = by_episode[key]
        precise = [row for row in rows if row["sampling_kind"] == "precise"]
        if not precise:
            raise RuntimeError(f"No training-compatible precise window: {key}")
        ordered = sorted(precise, key=lambda row: int(row["start_frame"]))
        # Query selection is event-based only, never optimized on query tracking.
        selected = ordered[(len(ordered) - 1) // 2]
        if args.task == "ball" and support:
            peak = outcomes[key].get("peak_frame")
            if peak is not None:
                containing_peak = [row for row in ordered
                                   if int(row["start_frame"]) <= int(peak) <= int(row["end_frame"])]
                if containing_peak:
                    selected = containing_peak[0]
        row = dict(selected)
        row["action_level"] = outcomes[key]["level"]
        row["evaluation_frame_indices"] = [
            index for index in range(1, int(row["length"]))
            if int(row["start_frame"]) + index * int(row["frame_stride"]) < int(row["total_frames"])
            and int(row["start_frame"]) + index * int(row["frame_stride"]) <= int(row["end_frame"])
        ]
        row["native_frame_indices"] = [
            min(int(row["start_frame"]) + index * int(row["frame_stride"]),
                int(row["end_frame"]), int(row["total_frames"]) - 1)
            for index in range(int(row["length"]))
        ]
        row["source_event_annotation"] = events.get(key, {}).get("events", {})
        return row

    def ball_measure(key, row):
        measured = outcomes[key]
        if key not in tracks:
            tracks[key] = read_jsonl(resolve(measured["track_path"]))
        start, end = int(row["start_frame"]), min(int(row["end_frame"]), int(row["total_frames"]) - 1)
        points = [point for point in tracks[key] if start <= int(point["frame"]) <= end]
        valid = [point for point in points if point.get("center") is not None and not point.get("ambiguous", False)]
        coverage = len(valid) / max(1, end - start + 1)
        if not valid:
            return None
        x = np.asarray([point["center"][0] for point in valid], dtype=float)
        offset = float(measured["initial_x_original_px"]) - float(measured["initial_center"][0])
        return {
            "coverage": coverage, "window_roll_px": float(x.max() - x.min()),
            "window_peak_original_x": float(x.max() + offset),
            "full_episode_peak_original_x": measured["peak_x_original_px"],
            "full_episode_peak_frame": measured["peak_frame"],
            "full_episode_peak_in_window": start <= int(measured["peak_frame"]) <= end,
            "boundary_seen": any(point.get("touches_right_edge", False) for point in valid),
            "original_coordinate_x_offset": offset,
            "track_path": measured["track_path"], "tracker_formally_certified": False,
        }

    support_rows, query_rows, templates, environments, unavailable = [], [], [], [], []
    env_names = sorted({key[0] for key in by_episode})
    for env in env_names:
        train_keys = sorted(key for key in by_episode if key[0] == env and key[2] == "train")
        rng.shuffle(train_keys)
        test_keys = sorted(key for key in by_episode if key[0] == env and key[2] == "test")
        missing_metadata = [key for key in train_keys + test_keys if key not in outcomes]
        if missing_metadata:
            raise RuntimeError(f"Missing level/audit metadata: {missing_metadata[:3]}")
        chosen, notes = [], {}
        if args.task == "ball":
            options = []
            lower, upper = config["ball"]["preferred_support_peak_original_x"]
            target_lo, target_hi = config["ball"]["target_original_peak_x"]
            for key in train_keys:
                outcome = outcomes[key]
                peak = outcome.get("peak_x_original_px")
                if (peak is None or outcome.get("boundary_seen", True)
                        or outcome.get("missing_rate", 1) > config["ball"]["maximum_missing_fraction"]
                        or target_lo <= float(peak) <= target_hi
                        or float(outcome.get("post_action_roll_px") or 0) < config["ball"]["minimum_support_roll_px"]):
                    continue
                row = selected_window(key, support=True)
                measured = ball_measure(key, row)
                if (measured is None or measured["coverage"] < .95 or measured["boundary_seen"]
                        or measured["window_roll_px"] < config["ball"]["minimum_support_roll_px"]
                        or target_lo <= measured["window_peak_original_x"] <= target_hi):
                    continue
                preferred = lower <= float(peak) <= upper
                rank = (not preferred, not measured["full_episode_peak_in_window"],
                        abs(float(peak) - (lower + upper) / 2))
                options.append((rank, row, measured))
            if options:
                _, row, measured = min(options, key=lambda item: item[0])
                chosen = [dict(row, support_measurement=measured)]
                notes = {
                    "support_measurement": measured,
                    "preferred_support_interval_satisfied": lower <= measured["full_episode_peak_original_x"] <= upper,
                    "support_selection_status": "provisional_tracking_based_not_formal_gold",
                }
        else:
            minimum = config["door"]["minimum_success_level"][env]
            levels = config["door"]["unreachable_support_levels"] if minimum is None else [minimum - 1, minimum + 2]
            missing_levels = []
            for level in levels:
                candidates = [key for key in train_keys if int(outcomes[key]["level"]) == int(level)]
                if candidates:
                    chosen.append(selected_window(candidates[0], support=True))
                else:
                    missing_levels.append(level)
            notes = {
                "requested_support_levels": levels, "missing_support_levels": missing_levels,
                "actual_support_levels": [row["action_level"] for row in chosen],
                "accepted_action_levels": [] if minimum is None else [minimum, minimum + 1],
                "action_selection_eligible": minimum is not None,
                "support_exception": "prediction_only_levels_9_10" if minimum is None else
                                     ("missing_requested_level_not_fabricated" if missing_levels else None),
            }
        if not chosen:
            record = {"environment": env, "reason": "no_eligible_train_support", **notes}
            unavailable.append(record)
            print("[unavailable] " + json.dumps(record), flush=True)
            continue
        used = {int(row["episode_index"]) for row in chosen}
        remaining = [key for key in train_keys if key[1] not in used]
        if len(remaining) < config["query_train_episodes_per_environment"] or not test_keys:
            raise RuntimeError(f"Insufficient support-disjoint queries for {env}")
        support_indices = list(range(len(support_rows), len(support_rows) + len(chosen)))
        support_rows.extend(chosen)
        query_indices = []
        keys = remaining[:config["query_train_episodes_per_environment"]] + test_keys
        for key in keys:
            row = selected_window(key)
            outcome = outcomes[key]
            row["gt_episode_outcome"] = {
                field: outcome.get(field) for field in (
                    "level", "peak_x_original_px", "peak_frame", "peak_in_target",
                    "boundary_seen", "final_closure_gap_px", "calibration_v2",
                )
            }
            if args.task == "ball":
                peak_frame = int(outcome["peak_frame"])
                row["full_episode_peak_in_window"] = row["start_frame"] <= peak_frame <= min(
                    row["end_frame"], row["total_frames"] - 1)
            else:
                row["full_episode_end_in_window"] = row["end_frame"] >= row["total_frames"] - 1
            row["sample_index"] = len(query_rows)
            query_indices.append(len(query_rows))
            query_rows.append(row)
        template_indices = []
        for level in sorted({int(outcomes[key]["level"]) for key in remaining}):
            key = next(key for key in remaining if int(outcomes[key]["level"]) == level)
            row = selected_window(key)
            row["template_index"] = len(templates)
            template_indices.append(len(templates))
            templates.append(row)
        anchor = next(index for index in query_indices if query_rows[index]["dataset_split"] == "test")
        record = {
            "environment": env, "environment_index": chosen[0]["environment_index"],
            "support_indices": support_indices, "query_indices": query_indices,
            "extension_anchor_query_index": anchor, "extension_template_indices": template_indices,
            **notes,
        }
        environments.append(record)
        panels = []
        for row in chosen:
            frames = row["native_frame_indices"]
            strip = []
            for frame_index in (frames[0], frames[len(frames) // 2], frames[-1]):
                frame = native_frame(dataset_root / row["video"][0], frame_index)
                cv2.putText(frame, f'{env} L{row["action_level"]} ep{row["episode_index"]} f{frame_index}',
                            (4, 18), 0, .4, (0, 0, 255), 1)
                strip.append(frame)
            panels.append(np.hstack(strip))
        review = output / "support_review"
        review.mkdir(exist_ok=True)
        cv2.imwrite(str(review / (env + ".jpg")), np.vstack(panels))
        print("[environment_ready] " + json.dumps(record), flush=True)
    if not environments:
        raise RuntimeError("No environment has usable supports; no empty evaluation is allowed")
    write_jsonl(output / "support.jsonl", support_rows)
    write_jsonl(output / "query.jsonl", query_rows)
    write_jsonl(output / "extension_action_templates.jsonl", templates)
    stage_files = set()
    for row in support_rows + query_rows + templates:
        stage_files.update(row["video"] + [row["action"], row["action"].split("/")[0] + "/meta/info.json"])
    (output / "stage_files.txt").write_text("\n".join(sorted(stage_files)) + "\n")
    plan = {
        "task": args.task, "setting": setting, "protocol": protocol, "experiment": config,
        "source_dataset": str(dataset_root), "prepared": str(output),
        "environments": environments, "unavailable_environments": unavailable,
        "expected_environments": len(env_names), "support_count": len(support_rows),
        "query_count": len(query_rows), "query_splits": dict(Counter(row["dataset_split"] for row in query_rows)),
        "stage1_reference": "exact_training_environment_table_entry", "stage2_initial": "mean_training_table",
        "no_query_GT_in_adaptation": True, "factual_actions": "original_recorded_target_actions",
        "extension_actions": "original_recorded_train_templates_no_interpolation",
        "extension_has_paired_GT": False, "frames_per_second": 20.0,
        "metric_status": "prediction_first_tracking_requires_review",
        "formal_metric_approved": False,
    }
    write_json(output / "plan.json", plan)
    print("[prepared] " + json.dumps({
        "task": args.task, "environments": len(environments), "queries": len(query_rows),
        "unavailable": unavailable, "prepared": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
