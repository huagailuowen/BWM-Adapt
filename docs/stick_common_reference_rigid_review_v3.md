# Stick balance common-reference diagnostic v3

This is an opt-in diagnostic, not an approved replacement for published metrics.
Legacy evaluation code, configurations, checkpoints and inference videos are unchanged.

## User-approved semantics

- Judge only the final continuous 0.3 seconds on the source-video clock.
- Both boxes must actually clear the tabletop, with a small rod inclination.
- Unequal rise distances are allowed. There is no mandatory 1:2 rise ratio.
- A rotated box with a corner still touching the table is not a success.
- Keep success candidates, failure candidates and unresolved measurements separate.
- The 3/5-degree alternatives remain sensitivity diagnostics pending calibration.

## Changes

- All methods for an episode use the same original GT last pre-lift reference.
  This is an evaluator baseline, not extra model conditioning or TTT support.
- Native GT controls are decoded at exactly the source indices represented by
  the query GT video. Compare these controls before interpreting model differences.
- Track the full reference box hull with optical flow and a robust rigid/similarity
  transform. Measure the lowest projected corner, not just the center displacement.
- Independently segment the current body. Expand its search halo when necessary;
  distinguish artificial search truncation from actual image cropping.
- Disagreement between current appearance and rigid contact geometry remains
  unresolved. Reference truncation is not silently accepted as complete geometry.
- Estimate inclination change from fixed attachment locations on both tracked
  boxes. Sweep plausible attachment positions to retain geometric uncertainty.
  When visible rod edges are available, require agreement with this estimate.
- Save every terminal measurement and native-resolution endpoint review images.

## Remaining limitations

Tabletop perspective is approximated locally from the resting boxes. Reference
silhouette shadows, occluded corners and nonrigid generated objects can still
invalidate clearance estimates. Coverage alone is not proof of correctness.
The full GT corpus and old 40 GT/Stage1/Stage2 triplets require stratified visual
review before reporting physical success rates. All artifacts stay in outputs.

Run only on a CPU compute node with
`sbatch scripts/evaluation/run_reevaluate_stick_common_reference_rigid_cpu.sh`.
