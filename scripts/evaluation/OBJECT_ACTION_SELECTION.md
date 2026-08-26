# Object-Centric Action Selection

This evaluation is a post-processing layer over the existing ID/OOD rollout
protocol. It does not adapt the model again and does not require regenerating
rollouts that already exist.

## Protocol

1. Adapt the method once with the fixed support set.
2. Treat the observed support outcome as a selectable candidate, and use the
   generated query rollouts for the remaining candidate actions.
3. Extract the predicted object's terminal state from every generated video.
4. Select the action whose available terminal outcome is closest to the
   target. Query outcomes are model predictions; the support outcome is the
   observation already available to the agent during adaptation.
5. Freeze the selected `action_id`.
6. Look up that action's ground-truth rollout and test whether its terminal
   state reaches the target.
7. Report task success, the ground-truth oracle action, regret, and action-set
   coverage. Headline success includes only targets reachable by at least one
   ground-truth candidate action.

The Event80 adapter uses a normalized main-camera `(x, y)` centroid and a 2-D
target rectangle. The same selection core can be reused with task-specific
state extractors: displacement/centroid for gravity and collision, bar angle
for mass balance, and lamp state for LightSwitch.

## Per-target object selection

Collision and joint mass-friction rollouts retain terminal states for both
objects. A target may select either object independently with `object_role` or
`object_index`; object extraction and video decoding are shared across all
targets. If neither field is present, the evaluator falls back to the legacy
global `outcome.object_index`, so existing evaluations are unchanged.

```yaml
outcome:
  object_index: 0
  object_roles: {struck_object: 0, striker: 1}
targets:
  - id: struck_object_medium
    kind: point_rectangle
    object_role: struck_object
    region: {x_min: 0.43, x_max: 0.57, y_min: 0.75, y_max: 0.86}
  - id: striker_terminal_region
    kind: point_rectangle
    object_role: striker
    region: {x_min: 0.25, x_max: 0.40, y_min: 0.55, y_max: 0.70}
```

The example striker region is schematic and must be calibrated from GT before
it is enabled in a benchmark. Adding such a target does not require new model
rollouts when the same action candidates have already been generated.

## Candidate-set semantics

The evaluator does not require a hard-coded number of actions. The candidate
set is exactly the union of the support and query actions declared by each
environment's protocol. In the existing K=1 Event80 grids this normally gives
one observed support candidate plus nine model-predicted query candidates.
Reports retain `selection_source`, candidate count, and coverage so these two
sources cannot be confused. Missing outcomes are reported but do not abort the
evaluation.

## Event80 command

```bash
uv run python scripts/evaluation/evaluate_event80_object_action_selection.py \
  --benchmark-root results/pushbox_friction_event80/<benchmark> \
  --protocol results/pushbox_friction_event80/<benchmark>/protocol/support_query_grid.json \
  --metadata-jsonl data/push_box_bwm_event_tap_segmented80_10action_A500_offset160_stop_65_105_20260705/train.jsonl \
  --dataset-root /afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox/libero_push_box_event_tap_segmented80_10action_hidden_lerobot_A500_offset160_stop_2026-07-05_hai-machine \
  --target-config configs/evaluation/action_targets/event80_pushbox_image_space.yaml \
  --method ours
```

Outputs are written below
`<benchmark>/metrics/object_action_selection/methods/<method>/`:

- `candidate_outcomes.jsonl`: predicted and GT endpoint table;
- `decisions.jsonl`: prediction-only choices and post-selection GT scores;
- `summary.json`: success, oracle agreement, regret, and coverage;
- benchmark-level `protocol.json`: an auditable no-GT-selection declaration.
