# World-model evaluation

This directory contains additive evaluation entry points. It does not modify
generation, training, or legacy inference code.

Both evaluators consume one frozen JSONL manifest. Global evaluation reads GT
and predicted RGB videos. Object-centric evaluation reads precomputed GT and
prediction masks; segmentation and tracking are deliberately separate so every
method is measured with the same frozen evaluator.

Action-level evaluation is implemented for scalar target-interval tasks and the
discrete LightSwitch button-selection task. PnP action evaluation remains out
of scope. `build_action_target_registry.py` reads a frozen task definition from
`configs/evaluation/action_targets/`, audits every GT action, and emits the
eligible environment/action set for each target. `evaluate_action_selection.py`
then selects actions using model-predicted outcomes and scores only the selected
GT rollout. This prevents GT reachability or validity from leaking into model
selection while excluding physically unreachable environment/target pairs from
the headline success rate.

Scalar candidate manifests use `predicted_outcome`, `gt_outcome`, and `valid`.
LightSwitch manifests use `button_color`,
`predicted_final_light_on_probability`, `gt_final_light_on`, and `valid_press`.
All candidates in one decision share `method`, `decision_id`, `seed`, and
`target_area_id`. Outputs contain per-decision success/regret and micro,
per-target, per-domain, and macro-over-target summaries.

The formal action benchmark uses a strict two-process boundary. First,
`select_actions_from_rollouts.py` reads prediction-only candidate rows and
rejects every GT field. It extracts the predicted outcome from a frozen task
state, selects an action, records the source-manifest SHA-256, and writes
`selected_actions.prediction_only.jsonl`. Only then may
`score_selected_actions.py` read the separately generated
`gt_action_outcomes.scorer_only.jsonl`. This second process looks up the frozen
action, computes success and regret, and verifies that its prediction-only
recomputation matches the saved decision.

Centroid-based displacement and landing targets require a frozen pixel-to-world
calibration. Mass Balance requires a frozen zero-angle calibration, and
LightSwitch requires frozen off/on lamp scores. Target configs deliberately use
`null` for these values until calibration is fitted on training/calibration
videos; formal evaluation fails rather than silently guessing a conversion.

## Task-specific physical metrics

`evaluate_task_metrics.py` consumes the same frozen query manifest with three
additional fields: `task`, `gt_state_path`, and `pred_state_path`. State files
are compressed NPZ files. Object tasks use `centroids [T,N,2]`, optional
`visible [T,N]`, image size, and FPS. A `masks [T,N,H,W]` array can be supplied
instead; centroids and unoriented principal-axis angles are then extracted
deterministically. Event arrays use the `event_<name>` naming convention.

Mass Balance requires the bar mask or `angles_rad [T,N]`. Its tilt is the
principal axis of one fixed longitudinal bar edge. Angle error is computed
modulo pi because an undirected edge at theta and theta + pi is identical.

LightSwitch uses only `light_on [T,L]` as its primary task state. The supplied
extractor can classify a calibrated, frozen lamp ROI by mean luminance. The ROI
and threshold must be fitted on training/calibration videos and then frozen for
all methods. It reports frame accuracy, final-state accuracy, transition-time
error, and exact-transition rate; EEF motion is not a LightSwitch target metric.

Example evaluation:

```bash
python scripts/evaluation/evaluate_task_metrics.py \
  --config configs/evaluation/tasks/mass_balance.yaml \
  --manifest results/<benchmark>/<evaluation>/protocol/manifest.jsonl
```

Deterministic arithmetic smoke checks are available at
`scripts/evaluation/smoke_test_task_metrics.py`.

`evaluate_event80_legacy_infer.py` is a smoke adapter for the historical
Event80 run 89097. It evaluates the main camera, wrist camera, combined RGB,
and a frozen color/trajectory block tracker. The segmentation mask is used only
to extract the object centroid and is not itself scored. The primary object
metric is the per-frame Euclidean distance between GT and predicted centroids.
Off-screen blocks are represented by the fixed bottom-center sentinel
`(x=0.5, y=1.0)`. The historical run reused
the adapted episode for generation, so its outputs are explicitly labeled
`legacy_same_episode_smoke` and must not be reported as the final disjoint
support/query benchmark.
