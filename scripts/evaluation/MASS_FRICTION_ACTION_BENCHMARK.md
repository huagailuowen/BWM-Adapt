# Selected Mass Friction action benchmark

The current summary tables use the three original target lines and binary
within-one-action-level matching. This is a user-selected **post-hoc protocol
revision**, chosen after the exploratory results were observed. It must not be
described as a preregistered evaluation or as strict physical task success.

| Component | Definition |
|---|---|
| Target lines | Normalized terminal y > 0.62, 0.75, and 0.92 |
| Model decision | First force level whose predicted terminal y exceeds the target; no isotonic fitting |
| GT reference | First force level whose GT terminal y exceeds the same target |
| Score | Absolute rank difference <= 1: 1; otherwise: 0 |
| Prediction never crosses | 0, including when GT can reach the target |
| Unreachable environment | Exclude only for that target, using GT and identically for all methods |
| Aggregation | Mean over reachable environments per target, then equal mean over the three targets |
| Coverage | Same 5 ID + 5 OOD environments; 24 reachable environment-target combinations |
| Caches | Same candidates, support information, terminal centroids, and offscreen handling as before |
| Display | Action Score; explicitly not strict goal-reaching success |

| Method | Three-target within-one-level score |
|---|---:|
| Ours | 69.72% |
| DINO Concat MLP | 60.83% |
| Standard Pooled WM | 35.00% |
| LoRA TTA | 26.67% |

The legacy CSV key `action_success` is retained for compatibility with existing
renderers. Its task-specific meaning is recorded in each table's `protocol.json`
and in `configs/evaluation/action_tasks/mass_friction_first_crossing_within_one_v1.yaml`.
Do not interpret this value as the fraction of selected actions that truly reach
the goal: choosing one level below the GT minimum also earns one point here.

Original strict-success summaries remain untouched under each method's
`action_evaluation/`. The 0/0.5/1 grading, full threshold sweep, and isotonic
variants also remain available in `metrics/action_first_crossing_v1/`, so the
protocol change and sensitivity results remain auditable. Other simulation
tasks' action definitions and metrics are unchanged.
