#!/usr/bin/env python3
"""Deterministic smoke checks for centroid, bar-angle, and lamp-state metrics."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.manifest import EvaluationRecord
from wan_video_action.evaluation.task_state import TaskState, save_task_state
from wan_video_action.metrics.task_metrics import evaluate_task_record


def record(task: str, gt: Path, pred: Path) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=f"smoke-{task}", environment_id="smoke-env", method="smoke",
        split="test", domain="id", support_size=1, seed=0,
        gt_video_path="unused", pred_video_path="unused", task=task,
        gt_state_path=str(gt), pred_state_path=str(pred),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        common = dict(image_height=3, image_width=4, fps=1.0)
        gt_centers = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)[:, None]
        pred_centers = np.array([[0, 0], [2, 0], [4, 0]], dtype=float)[:, None]
        save_task_state(root / "push_gt.npz", TaskState(centroids=gt_centers, **common))
        save_task_state(root / "push_pred.npz", TaskState(centroids=pred_centers, **common))
        push = evaluate_task_record(
            record("pushbox_friction", root / "push_gt.npz", root / "push_pred.npz"),
            {"object_names": ["block"]},
        ).values
        assert np.isclose(push["centroid_ade_px"], 1.0)
        assert np.isclose(push["centroid_fde_px"], 2.0)

        save_task_state(root / "bar_gt.npz", TaskState(
            centroids=gt_centers, angles_rad=np.zeros((3, 1)), **common
        ))
        save_task_state(root / "bar_pred.npz", TaskState(
            centroids=gt_centers, angles_rad=np.full((3, 1), np.pi / 6), **common
        ))
        balance = evaluate_task_record(
            record("mass_balance", root / "bar_gt.npz", root / "bar_pred.npz"),
            {"object_names": ["bar"]},
        ).values
        assert np.isclose(balance["bar_tilt_mae_deg"], 30.0)

        save_task_state(root / "light_gt.npz", TaskState(
            light_on=np.array([0, 0, 1, 1], dtype=bool)[:, None]
        ))
        save_task_state(root / "light_pred.npz", TaskState(
            light_on=np.array([0, 0, 0, 1], dtype=bool)[:, None]
        ))
        light = evaluate_task_record(
            record("lightswitch", root / "light_gt.npz", root / "light_pred.npz"), {}
        ).values
        assert np.isclose(light["light_state_accuracy"], 0.75)
        assert np.isclose(light["light_final_state_accuracy"], 1.0)
        assert np.isclose(light["light_transition_time_abs_error_frames"], 1.0)
    print("task metric smoke test passed")


if __name__ == "__main__":
    main()
