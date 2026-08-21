#!/usr/bin/env python3
"""Deterministic smoke case for collision action-selection metrics."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.metrics.action_metrics import (  # noqa: E402
    CollisionActionCandidate,
    CollisionActionSettings,
    TargetInterval,
    evaluate_collision_action_decision,
)


def main() -> None:
    result = evaluate_collision_action_decision(
        candidates=[
            CollisionActionCandidate("slow", 0.30, 0.31, True, True, True, 0.01),
            CollisionActionCandidate("medium", 0.42, 0.43, True, True, True, 0.01),
            CollisionActionCandidate("fast", 0.60, 0.61, True, True, True, 0.02),
        ],
        target=TargetInterval("short", 0.40, 0.45),
        settings=CollisionActionSettings(
            require_on_table=True,
            require_in_workspace=True,
            lateral_tolerance_m=0.05,
        ),
    )
    if result["selected_action_id"] != "medium":
        raise RuntimeError(f"unexpected collision action selection: {result}")
    if result["collision_task_success"] != 1.0:
        raise RuntimeError(f"collision action metric should succeed: {result}")
    print(result)


if __name__ == "__main__":
    main()
