# Per-Task Simulation Action Evaluation

This runner intentionally evaluates one task and one method at a time. It
matches the current Event80 evaluation depth without adding a cross-baseline
aggregation layer.

For every entry in an existing `transfer_plan.json` it:

1. uses the source rollout as the observed support candidate;
2. reads all same-environment generated target-action rollouts;
3. extracts the task state with `sim_rgb_v1`;
4. chooses the action using generated/observed outcomes only;
5. reads the selected action's GT outcome and computes success and oracle
   regret;
6. writes candidate, decision, summary, protocol, skip, and state-cache files
   below the task's configured `results/.../methods/<method>/action_evaluation`
   directory.

```bash
uv run --no-sync python scripts/evaluation/evaluate_sim_action_selection.py \
  --config configs/evaluation/action_tasks/gravity80_ours_89519.yaml
```

Equivalent configs are provided for mass collision, mass-balance random and
fixed pose, joint mass-friction, and LightSwitch.

Candidate counts come from each task's transfer plan rather than a global
constant. A target contributes to the headline metric only when at least one
GT action reaches it.

LightSwitch is partitioned by `lamp_before`, actions are aggregated by
`button_color`, and only `red_only` and `blue_only` causal environments are
evaluated. This prevents sequential chunks with different initial lamp states
from being treated as interchangeable actions.

The canonical LightSwitch manifest contains 15 query chunks for each of all
four causal environments. Action selection filters that shared manifest to
`red_only` and `blue_only`, retaining all 15 chunks in each environment rather
than evaluating the historical five-chunk subset.
