# Mass friction: first-crossing force-level calibration

This outcome-informed supplementary analysis measures whether predicted motion
identifies the minimum action level needed to exceed a target line. It is not a
retroactive replacement for the original formal action-success metric.

| Component | Fixed definition |
|---|---|
| Environments and actions | Same formal 5 ID + 5 OOD cases and candidate trajectories for every method |
| Outcome | Struck object's cached terminal normalized y, using the existing detector and last-five-frame aggregation |
| Original target lines | 0.62, 0.75, 0.92, taken from the existing rectangles' lower y boundaries |
| Sensitivity sweep | All 15 lines from 0.600 to 0.950 at increments of 0.025 |
| Prediction | Sort by increasing action order; choose the first predicted terminal y strictly greater than the target |
| GT reference | First actual action whose cached GT terminal y is strictly greater than the same target |
| Score | Exact action rank: 1; absolute rank difference of one: 0.5; otherwise: 0 |
| No predicted crossing | Score 0; penalized rank error equals the maximum possible candidate rank gap |
| GT cannot reach target | Exclude this environment for this target identically for all methods |
| Other metrics | Exact match, within-one-level rate, penalized rank error, missing-crossing rate, and actual reach rate of the chosen action |
| Main aggregation | Equal weight to reachable environments within a threshold, then equal weight to thresholds |
| Auxiliary selector | Increasing isotonic projection of the predicted action-response curve; no GT used for fitting or selection |

The raw selector and the isotonic selector must both be reported. Isotonic fitting
assumes stronger actions should not reduce terminal progress; detector noise or
real nonmonotonic behavior may violate this assumption. GT decreasing-pair counts
and both predicted curves are recorded for inspection. The GT oracle is never
smoothed to agree with a model.

This measures force-rank calibration, not video fidelity, physical force error,
or binary task success. Adjacent force levels need not be equally spaced in
physical units. Overpowered and underpowered choices one level from GT both
receive 0.5 as requested; actual reach is reported separately. Correlated target
lines are not independent trials or additional dataset samples.

Cached screen coordinates cannot establish absolute world displacement without
an initial-position reference and calibration. This version explicitly evaluates
crossing a screen-y target line, not time of first passage, maximum displacement,
or metric-world distance. Existing support-candidate handling, trajectories,
detectors, and offscreen rules are preserved unchanged across methods.

Do not choose thresholds, metrics, or the raw/projected variant based on which
makes Ours win. Publish the entire sweep, including unfavorable results and each
threshold's reachable-environment count. Future primary evaluation must freeze
the protocol before evaluation on fresh cases.

Run on a compute node; no GPU, video decoding, or rollout is required:

```bash
.venv/bin/python scripts/evaluation/rescore_mass_friction_first_crossing.py \
  --config configs/evaluation/action_tasks/mass_friction_first_crossing_v1.yaml
```

Output: the formal benchmark's `metrics/action_first_crossing_v1/` directory,
containing `protocol.json`, `scoreboard.csv`, `summary.json`, `per_threshold.csv`,
`decisions.jsonl`, `action_curves.jsonl`, and `report.md`. Original formal and
previous supplementary results are untouched.
