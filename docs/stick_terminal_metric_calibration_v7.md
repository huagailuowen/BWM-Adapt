# Stick terminal metric: v7 candidate, not a finalized benchmark

The success definition stays: both ends clear of the table, with small tilt,
throughout the final 0.3 source-clock seconds. There is no endpoint-rise-ratio
requirement. A visible resting corner invalidates full clearance. Earlier
motion is ignored. Angle means image-plane change relative to the native
pre-lift reference, not a calibrated 3D gravity angle.

This version replaces only contact evidence with the v6 local bottom-edge
measurements. It retains v4's conservative angular intervals and both 3/5
degree sensitivity variants. Those degree limits are still diagnostic.

Candidate contact thresholds were frozen using the existing 88 provisional
contact-reviewed GT clips before drawing new review samples:

- Visible footpoint gap -6..1.5 pixels: contact evidence.
- Gap at least 2.5 pixels with a fully visible reference: clearance evidence.
- Fewer than six footpoints, crop ambiguity, excessive negative gap, or the
  interval between thresholds: unknown, never silently physical failure.
- A positive clip requires bilateral clearance and small angular upper bound
  at EVERY evaluated terminal frame.
- A negative clip needs contact or excessive-tilt evidence in at least half
  the terminal samples (minimum two). An isolated noisy witness remains
  unknown. Such a conservative negative rule does not make a partially
  failed window positive.

Review uses four new native clips per environment sampled without looking
at decisions, eight query IDs across GT/Stage1/Stage2, and all remaining
native positives/unknowns as targeted diagnostics. Images show the reference,
last frame, and every terminal endpoint crop without automatic class labels.
The mapping and provisional decisions are retained separately for audit.
Independent random review must not be mixed with targeted samples to estimate
population accuracy. The 88 development labels certify visible contact only,
not the full success criterion.

All jobs run on CPU compute nodes. Old evaluators, outputs and threshold
definitions remain unchanged. Candidate coverage and model predictions must
not be published as formal success rates before independent visual review.
