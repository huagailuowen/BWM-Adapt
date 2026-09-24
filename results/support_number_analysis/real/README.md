# Real tasks: support-count completion

Only missing K settings are scheduled. Original formal results remain untouched.

| Array index | Task | K | Initialization | Reuse |
|---|---|---|---|---|
| 0 | Ball | 1 | Formal cluster mean | Rerun only ball-0/1, both Level 2 |
| 1 | Ball | 2 | Formal cluster mean | Reuse ball-0 Levels 1/2 and ball-1 Levels 2/3 |
| 2 | Ball | 4 | Formal cluster mean | All environments |
| 3 | Door | 1 | Formal cluster mean | Original K=2 retained |
| 4 | Door | 4 | Formal cluster mean | Original K=2 retained |
| 5 | Stick | 1 | Formal full-table global mean | Original K=2 retained |
| 6 | Stick | 4 | Formal full-table global mean | Original K=2 retained |
| 7 | Soft | 1 | Formal family mean | Original K=4 retained |
| 8 | Soft | 2 | Formal family mean | Original K=4 retained |

Support loss is averaged. The model/table pair, initialization policy, 40-step
schedule, FP32 Z, frame windows, original query indices and generation seeds stay
unchanged. Supports are distinct training trajectories, disjoint from the fixed
query trajectories. All action decisions use predictions; no observed support
video replaces a candidate prediction. Primary scores use held-out test queries.

Ball retains the formal Level-7 anchor for ball-7/9. Additional Ball/Door supports
use nearest available training action levels around the original anchor, excluding
all query episodes. Door K=1 uses the first historical support. Stick K=1 uses
the first historical support; K=4 retains its historical pair and adds distinct
training lift trajectories nearest the training-derived balance position. Soft
K=1 uses the first historical support, K=2 adds an opposite-direction historical
support. These rules do not optimize against test-query outcomes.

Each directory records the exact selected episodes and levels in
`support_protocol.json`. Old inference artifacts and old defaults are not changed.
Jobs run on low-priority compute GPUs, load a model once, reuse complete query
artifacts on restart, and run tracking/image metrics on the same compute node.

The Soft final action aggregation must use the published shared6/test12 cohort
and 6-degree onset rule. Stick must use the published seeded36 cohort and
5-degree/0.3-second predicted-balanced precision rule. Intermediate full-cohort
diagnostics must not be substituted for these formal scores.

Launch: `sbatch jobs/run_real_support_number_low.sh`.

## Global-mean initialization lineage

This is a separate initialization branch from the selected formal cluster/family-mean results above. The selected full-cohort rows below use fixed factual test queries: 80 Ball queries across eight environments and 100 Door queries across nine action-scored environments. Ball K2/K4 replace the listed environments with their repeat-support rescue predictions; every other Ball environment remains from the original global-mean run. Door K4 instead uses the complete original-support cohort with summed, not averaged, support loss. The six rescue rows are single-environment diagnostics, not additional full-cohort results. Image/object columns use held-out test queries only. Lower LPIPS and centroid ADE/FDE are better.

| Task | K | Run | Scope | Action success | PSNR | SSIM | LPIPS | Centroid ADE px | Centroid FDE px |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ball | 1 | Global mean | 8 env, 80 queries | 68.75% | 36.144 | 0.9603 | 0.0244 | 14.943 | 25.549 |
| Ball | 2 | Global mean + rescue | 8 env, 80 queries | 68.75% | 36.161 | 0.9605 | 0.0243 | 14.880 | 25.762 |
| Ball | 4 | Global mean + rescue | 8 env, 80 queries | 62.50% | 36.240 | 0.9606 | 0.0235 | 13.890 | 23.028 |
| Door | 1 | Global mean | 9 env, 100 queries | 22.22% | 31.732 | 0.9234 | 0.0378 | 11.337 | 20.230 |
| Door | 2 | Global mean | 9 env, 100 queries | 55.56% | 32.152 | 0.9267 | 0.0321 | 7.954 | 13.470 |
| Door | 4 | Global mean + sum loss, original supports | 9 env, 100 queries | 55.56% | 32.494 | 0.9292 | 0.0296 | 6.889 | 10.770 |
| Ball | 2 | Repeat-support rescue | ball-1, 10 test queries | 100% | 36.479 | 0.9608 | 0.0225 | 13.840 | 11.391 |
| Ball | 2 | Repeat-support rescue | ball-2, 10 test queries | 50% | 35.699 | 0.9596 | 0.0299 | 22.090 | 36.625 |
| Ball | 4 | Repeat-support rescue | ball-2, 10 test queries | 50% | 35.517 | 0.9593 | 0.0321 | 23.746 | 39.682 |
| Door | 4 | Repeat-support rescue | door-3u, 10 test queries | 100% | 34.265 | 0.9408 | 0.0217 | 2.977 | 5.200 |
| Door | 4 | Repeat-support rescue | door-3u-4d, 10 test queries | 0% | 30.641 | 0.9172 | 0.0582 | 24.752 | 44.100 |
| Door | 4 | Repeat-support rescue | door-6u-8d, 10 test queries | 100% | 31.421 | 0.9192 | 0.0310 | 6.466 | 8.500 |

Selected full-cohort CSV: `results/support_number_analysis/real/globalmean_20260922/selected_composite_summary.csv`. The untouched original mean-loss rows remain in `results/support_number_analysis/real/globalmean_20260922/formal_summary.csv`. Ball rescue source: `results/support_number_analysis/real/globalmean_repeat_regressions_20260922/ball/k{2,4}/<environment>/metrics/`. Ball replacement matches the original factual test-query IDs exactly, swaps ten test queries per rescued environment, and recomputes image/object means over all 80 Ball queries. Door K4 source: `results/support_number_analysis/real/globalmean_sumloss_original_support_20260922/door/k4/formal_summary.json`. Door uses the fixed ground-truth first-close level with exact first-close action-level matching across nine environments. The Door K4 row changes support-loss reduction as well as K, so it is not a pure support-count comparison with Door K1/K2. Ball K4's rescue does not improve action success and slightly worsens its visual/object means, but remains selected here as requested. Ball mean-versus-sum failure cases are recorded in `results/support_number_analysis/real/globalmean_sumloss_original_support_20260922/ball_failure_analysis.md`.

## Exploratory Stick K=8 (excluded from selected tables)

The selected seeded36 test-query cohort yields balanced-action precision 5/6 = 83.33% at the published 5-degree/0.3-second rule. The action denominator is predicted-balanced cases within 36 fixed queries, not all 36 queries. Image/object metrics remain on 45 held-out test queries: PSNR 32.010, SSIM 0.9423, LPIPS 0.0422, ADE 2.435 px, FDE 5.628 px. This is the K=4-prefix, eight-train-support result; it does not use the 45-query diagnostic action precision as the formal score. The numeric metric is complete; visual audit remains a separate publication check.

## Exploratory Door K=8 (excluded from selected tables)

The exploratory nine-environment, 100-test-query Door cohort has action score 66.67% (denominator 9), PSNR 32.407, SSIM 0.9287, LPIPS 0.0300, ADE 7.160 px, and FDE 11.150 px. It retains the original selected K=4 supports, adds four distinct train-only action levels, uses global-mean initialization and summed support loss, and leaves the query cohort unchanged. See `door/k8/formal_summary.json` for the nine action decisions.

Selected Door action scores now use exact first-close level only; adjacent levels receive zero. The per-environment decisions and source provenance for selected K=1/2/4 are in `real/door_selected_exact_action_20260923.json`.

Real K=8 runs are retained as exploratory artifacts but excluded from the selected simulation/real figures and completed selected-summary rows. Simulation K=8 remains selected.
