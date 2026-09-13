#!/usr/bin/env python3
"""Reuse frozen image/track measurements, with a new opt-in Door classifier.

Read-only input scores remain intact. Reclassification, refreshed review grids
and a GT detector self-check go to a separate versioned output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

from score_real97_ball_door import (
    ROOT, METHODS, read_json, save_json, freeze_manifest, apply_door_closure_policy,
    trajectory_summary, object_errors, aggregate, read_video, native_frames,
)


def final_panel(query, method, state, summary):
    import cv2
    import numpy as np
    frames = read_video(query["paths"][method])
    end = query["row"]["evaluation_frame_indices"][-1]
    native = native_frames(frames[end:end + 1], query["native_size"])[0]
    # Fixed geometry for all methods; no outcome-dependent crops.
    crop = native[70:410, 220:460]
    header = np.full((52, 240, 3), 255, dtype=np.uint8)
    cv2.putText(header, f"{method} L{query['row']['action_level']:02d}",
                (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    gap = state.get("raw_edge_seam_gap_px")
    text = f"{summary['tail_state']} gap={gap}"
    cv2.putText(header, text, (3, 33), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 0), 1)
    cv2.putText(header, query["row"]["environment"], (3, 47),
                cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)
    return np.concatenate([header, crop], axis=0)


def main():
    import cv2
    import numpy as np
    from wan_video_action.real97_eval.object_measurements import usable
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/real97_ball_door_scores_20260912_v2.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run calibration audit video decoding on a compute node")
    cv2.setNumThreads(1)
    config = read_json(ROOT / args.config)
    source = ROOT / config["reuse_cpu_output"]
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    previous = read_json(source / "manifest.json")
    manifest = freeze_manifest(config, output)
    # Threshold-only rescoring must not silently reuse another experiment.
    old_queries = {(q["task"], q["key"]): q for q in previous["queries"]}
    reviewed_queries = []
    training_check = []
    by_env = defaultdict(list)
    for q in manifest["queries"]:
        old_q = old_queries[(q["task"], q["key"])]
        if q["paths"] != old_q["paths"] or q["row"] != old_q["row"]:
            raise ValueError(f"Source query changed: {q['key']}")
        rel = Path(q["task"]) / (q["key"] + ".json")
        result = read_json(source / "queries" / rel)
        tracks = read_json(source / "tracks" / rel)
        ids = result["frame_ids"]
        if q["task"] == "door":
            for method, states in tracks.items():
                states = [apply_door_closure_policy(s, config) for s in states]
                for s in states:
                    s["usable"] = bool(usable(s))
                tracks[method] = states
                result["trajectory_summary"][method] = trajectory_summary("door", states, ids, None, config)
            for method in METHODS:
                obj = object_errors("door", tracks["gt"], tracks[method], ids, q["native_size"])
                a = result["trajectory_summary"]["gt"]["closed"]
                b = result["trajectory_summary"][method]["closed"]
                obj["closure_agreement_with_tracked_gt"] = float(a == b) if a is not None and b is not None else None
                result["methods"][method]["object"] = obj
            result.pop("review_sheet", None)
            result["calibration_version"] = config["version"]
            by_env[result["environment"]].append((q, result))
            if result["split"] == "train":
                index = str(int(q["key"].split("_")[0][1:]))
                expected = config["door_closure"]["training_visual_labels_by_query_index"][index]
                actual = result["trajectory_summary"]["gt"]["closed"]
                training_check.append({"key": q["key"], "visual_closed": expected,
                                       "detected_closed": actual, "matches": expected == actual})
        result["measurement_source"] = str(source / "queries" / rel)
        save_json(output / "queries" / rel, result)
        save_json(output / "tracks" / rel, tracks)
        reviewed_queries.append((q, result))
    audit = {"threshold_selected_using_train_only": True, "training_calibration": training_check,
             "test_gt_prefer_self_check": [], "test_gt_levels": [], "final_frame_grids": []}
    # This is a held-out detector check, not an input to threshold selection.
    for env, items in sorted(by_env.items()):
        test = sorted([(q, r) for q, r in items if r["split"] == "test"], key=lambda x: x[1]["level"])
        states = {r["level"]: r["trajectory_summary"]["gt"]["closed"] for q, r in test}
        pair = next(([l, l + 1] for l in range(1, 10) if states.get(l) is True and states.get(l + 1) is True), [])
        expected = config["door_prefer"].get(env)
        audit["test_gt_prefer_self_check"].append({"environment": env, "detected_pair": pair,
            "reference_pair": expected, "overlap": len(set(pair).intersection(expected)) / 2 if expected else None,
            "closed_by_level": states})
        for q, result in test:
            audit["test_gt_levels"].append({"key": q["key"], "environment": env, "level": result["level"],
                "state": result["trajectory_summary"]["gt"]["tail_state"],
                "tail_excess_px": result["trajectory_summary"]["gt"]["tail_excess_px"]})
        for offset in (0, 5):
            selected = test[offset:offset + 5]
            if not selected:
                continue
            rows = []
            cached_tracks = {q["key"]: read_json(output / "tracks/door" / (q["key"] + ".json"))
                             for q, r in selected}
            for method in ("gt",) + METHODS:
                panels = []
                for q, r in selected:
                    end = r["frame_ids"][-1]
                    panels.append(final_panel(q, method, cached_tracks[q["key"]][method][end],
                                              r["trajectory_summary"][method]))
                rows.append(np.concatenate(panels, axis=1))
            path = output / "audit/door" / f"{env}_test_levels_{offset + 1:02d}_{offset + len(selected):02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), np.concatenate(rows, axis=0)):
                raise RuntimeError(f"Could not write {path}")
            audit["final_frame_grids"].append(str(path))
            print(f"AUDIT_GRID {path}", flush=True)
    save_json(output / "door_calibration_audit.json", audit)
    summary = aggregate(output, manifest)
    save_json(output / "cpu_complete.json", {"queries": len(manifest["queries"]),
              "reused_image_metrics": True, "door_closure": config["door_closure"],
              "training_visual_label_matches": sum(r["matches"] for r in training_check),
              "training_visual_label_count": len(training_check)})
    print(f"RECALIBRATED {summary['queries_scored']} queries", flush=True)


if __name__ == "__main__":
    main()
