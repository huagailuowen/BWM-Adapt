# Mass friction action-selection precision

This is an exploratory, outcome-informed sensitivity analysis, not a replacement
for the previously reported formal action-success rate. Better video prediction
does not by itself imply better discrete action selection. Do not select a rule,
target, environment subset, or threshold because it improves one method's ranking.

| Component | Rule |
|---|---|
| Data | Existing formal 5 ID + 5 OOD environments and frozen candidate caches |
| Targets | Original short, medium, and long target rectangles; no resizing |
| Eligibility | Same 24 GT-reachable environment-target combinations |
| Short/medium selection | Minimize predicted distance to the rectangle, then predicted distance to its center, then action order |
| Long selection | Keep original first-reaching action selection |
| Success | Keep original target-reach and minimum-reaching policies, including the existing long-target one-step tolerance |
| Continuous precision | Selected action's GT terminal centroid distance to target center, in pixels; short/medium only |
| Center regret | Selected GT center distance minus the smallest GT center distance among the same candidate actions |
| Missing/mismatched data | Fail explicitly; never drop a method's failures or silently change the cohort |
| Reporting | Publish original and alternative rows together for every method; keep formal tables unchanged |

The original rectangle distance is zero everywhere inside a target. Its nearest
selection can therefore break ties by action ID, even when one predicted landing
point is much more central. The opt-in `nearest_center_tiebreak` strategy removes
this arbitrary tie without trading predicted target feasibility for proximity to
the center. The long target remains a minimum-force reaching task, not a precision
stopping task, so its strategy is not changed.

Action selection uses model-predicted outcomes and the requested target only.
Observed support outcomes remain available exactly as in the original formal
candidate caches; they are not added or removed for this analysis. Ground truth
is used for cohort reachability, post-selection scoring, and the regret oracle,
never to select an action. All methods must have identical environments, candidate
trajectory indices, action IDs, and GT outcomes before comparison is allowed.

Terminal centroids, last-five-frame aggregation, detectors, and offscreen handling
are reused unchanged. No rollout, video decoding, or LPIPS computation is needed.
Run the evaluator on a compute node, not on the login node:

```bash
.venv/bin/python scripts/evaluation/rescore_mass_friction_target_precision.py \
  --config configs/evaluation/action_tasks/mass_friction_target_precision_v1.yaml
```

Outputs go to the benchmark's `metrics/action_target_precision_v1/` directory:
`protocol.json`, `summary.json`, `scoreboard.csv`, `decisions.jsonl`, and `report.md`.
The protocol includes source hashes and the fixed scored/excluded combinations.
Default action-selection behavior and all other tasks remain unchanged.
