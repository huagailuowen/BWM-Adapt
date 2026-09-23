# Formal simulation support-count ablation

Snapshot: 2026-09-22. This is the controlled-study register for Mass Collision, Mass Balance, and Light Switch. It does not replace earlier exploratory tables. Compare K only within the same task, checkpoint, query cohort, action rule, and metric definition. A blank cell is not a zero.

## Main action-selection table

| Task | K=1 | K=2 | K=4 | K=8 | Primary decision cohort | Protocol status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Mass Collision | 61.1% (11/18) * | 66.7% (12/18) | 72.2% (13/18) | 77.8% (14/18), sum-loss | 18 reachable environment-target decisions | K8 sums support losses rather than averaging them; this changes the inner objective as well as K. The recorded action-decision source for support candidates also differs at K1. |
| Mass Balance, workspace-random no-leak | 70.0% (7/10) | 80.0% (8/10) | 90.0% (9/10) | 90.0% (9/10) | 5 ID + 5 OOD environments; balanced target | All 15 candidate actions, including support actions, use model predictions. K8 succeeds on 5/5 ID and 4/5 OOD environments. K2 is the bidirectional near-balance version. |
| Light Switch | 50.0% | 87.5% | 87.5% | 87.5% | Red-only and blue-only button-decision cases | K8 is the original formal eight-support result. |

* Collision uses an observed support set and predicts the remaining candidates at each K. The stored action-scoring metadata labels the selected K1 support candidate as observed GT and K2/4/8 support candidates as model predictions. This is an action-decision convention to document separately from support/query disjointness. A prediction-only K1 diagnostic scores 55.6% (10/18).

The selected Light K8 is the older original formal result at 87.5%. A separate controlled K8 color/state-coverage extension reports 62.5% (5/8); do not substitute or average these two distinct support constructions.

## Support construction

| Task | K=1 | K=2 | K=4 | K=8 |
| --- | --- | --- | --- | --- |
| Mass Collision | One informative impact support | Two supports | Four supports | Eight supports; sum of support losses rather than mean |
| Mass Balance | Nearest informative but unbalanced action | Two near-balance unbalanced actions, on opposite sides where available | K2 plus one random-unbalanced and one balanced action | K4 plus four action-bin-farthest actions; complete |
| Light Switch | One red-button support | One red and one blue support | Red/blue crossed with lamp-before off/on | Original formal eight-support pool |

Mass Balance has a separate K2 experiment using one unbalanced plus one balanced support. It also scores 80%, but its regret is 2.176 versus 0.592 for the selected bidirectional K2; it is not the K2 row above. Only six of ten environments admit a strict two-sided unbalanced bracket; the other four use the two nearest available unbalanced actions on one side. Consequently K2-to-K4 also changes information composition, not only count.

## Readiness and work remaining

| Priority | Task | Required work | Reason |
| --- | --- | --- | --- |
| 1 | Mass Balance | Consolidate the completed K8 full-15-action video/object metrics with the other K rows under the same cohort definition. | K8 action selection and full-action video metrics are complete; the earlier pending jobs were superseded. |
| 2 | Mass Collision | Document which support candidates enter action selection as observed outcomes versus predictions while keeping the selected K1 61.1% headline. | The support/query construction is consistent, but the stored action-decision metadata uses different support-candidate sources. |
| 3 | Light Switch | Keep the selected original K8 87.5% distinct from the controlled K8 62.5%; audit query IDs and support/query disjointness if reporting the controlled extension separately. | The two K8 results use different support constructions. |
| 4 | All three | Consolidate PSNR, SSIM, LPIPS, object mean/final error, and task-specific physical error using each task's same query cohort. Keep task-specific object units separate. | This first register reports comparable action outcomes; visual/object results remain scattered or use differing query coverage. |
| 5 | All three | Record per-environment decisions and uncertainty intervals or paired changes, not just aggregate percentages. | Ten Balance environments and eight Light decisions make percentages sensitive to a few cases. |

For Mass Balance, use the **prediction-for-every-action** rule for action selection and the full 15-action prediction cohort for visual/object evaluation. Do not import older query-only video metrics into the same numeric column without relabeling the cohort. K8 action and full-action video metrics are complete; K4 still needs the same video-metric consolidation before a detailed cross-K visual table is final.

## Result provenance

- Collision selection and the mixed-policy warning: `results/support_number_analysis/mass_collision_support_number_selected_results.json`.
- Collision prediction-only K1/K2/K4 recount: `results/support_number_analysis/mass_collision_prediction_only_recount/summary.json`.
- Collision K8 sum-loss: `results/mass_collision/noleak_highmass2x_support_k8_sumloss_v1/methods/ours_action8_highmass2x/k8/step_4300/seed_20260827/action_evaluation/summary.json`.
- Balance K1: `results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/ours/step_4300/seed_20260902/all_actions_model_predictions/action_evaluation/summary.json`.
- Balance selected K2: `results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k2_bidirectional_near_balance_dense15_v1/methods/ours/step_4300/seed_20260902/all_actions_model_predictions/action_evaluation/summary.json`.
- Balance K4: `results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k4_k2plus_random_unbalanced_balanced_dense15_v1/methods/ours/step_4300/seed_20260902/action_evaluation_predict_all_actions/summary.json`.
- Balance K8: `results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k8_k4plus_farthest4_dense15_v1/methods/ours/step_4300/seed_20260902/action_evaluation_predict_all_actions/summary.json`.
- Light selected original K8: `results/support_number_analysis/support_number_metrics.csv`.
- Light controlled K8 diagnostic: `results/lightswitch/physicalpress33_all4env_support8_query15_v1/methods/ours_redblue_oppositestate_k8_inner40/step_3100/seed_20260828/action_evaluation/summary.json`.
- Light query/support manifests: `results/lightswitch/physicalpress33_all4env_support8_query15_v1/protocol/`.
