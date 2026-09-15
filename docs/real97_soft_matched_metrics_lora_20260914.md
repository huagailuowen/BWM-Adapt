# Real97 matched Soft evaluation and LoRA metrics

This opt-in experiment leaves historical inference, training, and score artifacts unchanged.

## LoRA
Reuse all 216 completed factual Door/Ball queries and the exact existing GT, frame masks,
tracking thresholds and latest first-reaching/successor-pair Ball action rule.
Append LoRA and the completed TTT scores to the four historical methods.
Unpaired fixed-initial-frame action sweeps do not enter paired-GT metrics.

## Soft
Source: infer_real97_soft_balanced9_dual_static1_train115613_job116188, step5500 model and Z table.
36 queries: two train and two test episodes in each of nine environments.
Standard: pooled target-EEF history5 checkpoint at step5500.
24 previous episode identities overlap; reuse only after video/action asset SHA256 and
temporal-contract equality. Infer the remaining 12 queries on one low-priority GPU.
Checkpoint/data staging reuses the node-local Wan cache.

Standard has five identical native-frame-zero observations followed by 28 future frames.
Ours has one frame-zero observation followed by 32 future frames.
Comparisons align Standard raw indices 4..32 with Ours indices 0..28.
Score only native timestamps 3..84, stride3, excluding any repeated tail.
Do not pretend repeated initial frames supply real history.
Do not extend Standard's trained horizon merely to obtain 32 predictions.

Rows in comparison grids: GT, Standard, Stage1, Stage2.
The old Standard training pool differs from the new nine-environment Ours pool.
Record actual Standard train-episode membership. Report shared-training environments
separately from Ours-ID/Standard-OOD cases; this is not an equal-training-budget claim.
Reused outputs do not guarantee paired random noise realizations.

## Tracking and audit
Run GroundingDINO Tiny and color/temporal filtering on CPU compute allocations.
Track each GT/generated video independently with identical settings and no GT-center
initialization or query-GT correction. Measure the center of the visible box intersected
with the native 512x256 image, including objects clipped by the bottom border.
Missing detections stay missing: report coverage alongside ADE and true-final-timestamp
FDE. Also compare methods on the same per-frame valid intersection.
Report an initial-center-static baseline to distinguish useful motion prediction from
small errors on nearly stationary episodes. Soft has no action-selection metric.

Each query exports detector proposals, centers, coverage and jump/boundary flags,
two contact sheets, a full tracking-overlay video, and an x/y trajectory SVG.
All nine environments and both splits must be visually sampled, with additional review
of missing detections, boundary clips, large errors and jumps.
Automatic scoring completion is NOT formal metric approval.
