# Counterfactual bridge implementation

This file records the code-level realization of
`docs/nonlinear_bridge_teacher_distillation.md`.

## Compatibility

- The new path is disabled unless `grouped_context_counterfactual_enabled` is set.
- Legacy grouped-context, curriculum, structured-update, bridge, and
  `self_correction` behavior remains selected by the old arguments.
- A nested video payload can now override its own frame range and frame stride.
  String payloads retain the previous dataset-level behavior.

## Implemented path

- Bridge geometry uses `q(s) = 1 - (1 - s)^power`.
- The nonlinear sampler uses 40% endpoints, 30% near-Global points, and 30%
  interior points.
- The numeric Global context can be frozen after warmup while endpoint contexts
  remain frozen throughout bridge training.
- A configurable number of timestep buckets replaces the single-timestep batch
  when the new path is enabled.
- A paired high-noise update can allocate half of its examples to offline
  Teacher counterfactual sources. With paired-update probability 0.5, this is
  25% of post-Global examples.
- Counterfactual noise follows the configured 20/60/20 high-noise bands.
- A Teacher source is accepted only when the pressed-button causal outcome
  differs from the real target environment.
- The correction input is formed from the complete Teacher rollout and fresh
  Gaussian noise. Its target velocity points from that input to the real target.
- No source-token branch, editing branch, task token, or lamp-region weighting
  is introduced.

## Validation modes

- `scripts/validate_counterfactual_bridge.py` checks geometry, quotas, causal
  Teacher coverage, and source-file completeness without loading the model.
- `scripts/smoke_counterfactual_bridge.py` is a self-terminating GPU smoke test.
  It executes Global-real, endpoint-real, near-real, near-counterfactual,
  interior-real, and interior-counterfactual branches once and verifies the
  freeze policy.
- `grouped_context_bridge_smoke_sequence` applies the same six-stage sequence to
  the full training entry and skips the final checkpoint. It requires one
  warmup update and five post-warmup updates.
