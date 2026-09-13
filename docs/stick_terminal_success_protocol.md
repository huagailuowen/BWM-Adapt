# Stick balance: confirmed terminal success protocol

The user confirmed that success requires sustained balance at the END of the
commanded lift, not any transient balanced interval earlier in the video.
The current duration requirement is at least 0.3 seconds. Both endpoints must
have sufficient observed lift and the agreed rise ratio throughout that window.
The detector and its numerical geometry thresholds remain under visual audit.

## Native source clock

Timing is computed from the original dataset frame indices and original FPS.
For a 20 FPS source sampled every three frames, two intervals are exactly 0.3
seconds. The encoded video's rounded 6.67 FPS must not turn them into 0.29985
seconds and manufacture a failure of the duration check.

The window includes the latest sample at or before the 0.3-second start boundary,
so the measured interval cannot be shorter merely because a boundary falls
between samples. This can conservatively include up to one additional sampling
interval. Conditioning frames and padded frames are excluded before selecting it.

The commanded lift ending comes from the existing action-derived event metadata,
not the query GT outcome. Its last observed sample must reach the lift ending to
within the sampling cadence; the exact source-frame gap is recorded. A shorter
prediction horizon is `incomplete_lift_horizon`, not an action failure or success.
All such videos remain in the output manifest and coverage accounting.

## Decisions

Success: every sampled point in the terminal interval has reliable bilateral
lift evidence, complete required geometry, and the accepted rise ratio.
Failure evidence: every sampled point in that terminal interval has reliable
below-absolute or asymmetric-rise evidence. Mixed or uncertain intervals remain
unresolved. Partially visible feet still cannot certify a full-box positive lift.

The output records source FPS, stride, excluded padding, the terminal frame
interval, actual duration, lift-end gap, and reasons for every decision. Earlier
anytime statistics are kept in their original experiment directories, not mixed
with this terminal protocol. Legacy configurations keep the earlier behavior;
the new configuration explicitly opts into the terminal branch.

Launch: `sbatch scripts/evaluation/run_stick_terminal_partial_witness_cpu.sh`.
No training changes, no GPU, no formal `results` publication until audit passes.
