# Ball friction: real97 reference evaluation

The text-only metric snapshot is in
[`comparisons/metrics/first_reach_pair_overlap_20260913_v1/`](comparisons/metrics/first_reach_pair_overlap_20260913_v1/README.md).

It includes Standard, DINO, Ours Stage1, and Ours Stage2, each at step 5500.
There are eight environments, with two train queries and ten held-out test
queries per environment. Support is drawn from train episodes; query train/test
metrics are reported separately.

The current Ball action metric selects the first numerical level whose predicted
farthest ball-center x reaches the right blue marker minus 30 original pixels.
GT selects its first reaching level independently. Each first level and its
successor form a pair; their intersection size divided by two is the score.
The formal pair at level 10 includes an unobserved label 11, not a tested rollout.
Earlier closest-target metrics are retained as explicitly labeled legacy results.

Ours Stage2 uses the user-selected support pairs and near-environment-table
initializations. This is an environment-informed reference, not an unknown-
environment or equal-information baseline comparison. The snapshot preserves
the source tracking audit status and coverage statistics.

Original videos stay under `outputs/infer_real97_ball_ours_reference_job114807`,
`outputs/infer_real97_ball_standard_step5500_reference_job115189`, and
`outputs/infer_real97_ball_dino_step5500_reference_job115190`. They are not
committed. Exact per-query paths and scoring frame masks are in the frozen
manifest, with per-method numerical metrics under `methods/`.
