# Real97 Ball / Door scoring, 2026-09-12

This is an additive evaluation branch. It does not modify inference videos,
training code, previous configuration files, or checkpoint directories.

## Frozen comparison

Use `configs/evaluation/real97_ball_door_scores_20260912_v1.json`.
Score Standard, DINO, Ours Stage1, and Ours Stage2. Only factual `completed/q*.json`
records with paired GT are eligible. Query identities, actions, native frame
indices and evaluation masks must match across methods; mismatches fail rather
than silently shrinking the benchmark. Counterfactual action-template videos
are excluded from paired image and object errors.

Ball has 96 query clips and Door has 120. Report train and test separately;
the primary held-out action selection uses the ten factual test levels per
environment. Existing Ours support selection and near-table initialization are
retained and disclosed. Door Ours is step 3715, versus baseline step 5500. These
results are not an equal-training-step or equal-information comparison.

## Image metrics

Only `evaluation_frame_indices` are scored. Exclude the conditioning frame,
repeated last-frame padding, and letterbox margins. Do not squeeze aspect ratios.
PSNR is averaged per frame with a numerical floor of MSE=1e-12. SSIM uses the
standard 7x7 uniform window, sample covariance, data range 1, and RGB-channel
average. LPIPS uses pretrained AlexNet and the same unpadded image region.
Report query means and environment-macro means, with available counts.

## Object measurements and quality control

Undo letterboxing and restore native pixels: Ball 560x248 and Door 640x480.
Ball's original-camera x coordinate is the native crop x plus 80; translations
do not change distance errors. Existing trackers and Door calibration are used
identically for GT and every method, initialized from the shared GT first frame.
Door tracks a door-panel outer edge, not a true object centroid or angle.

Record every frame, confidence, missing detection, and exit flag. Report ADE,
FDE, paired coverage, observed-only ADE, and missing-prediction-penalized ADE.
The missing penalty is the native image diagonal for Ball and image width for
Door, applied only when GT is measurable. Never hide low coverage behind ADE.

Ball may hold its last actually visible center only after a confirmed right exit:
the preceding detection touches the right edge, or positive velocity very near
the edge predicts crossing. Arbitrary missing detections are not exits. This
is a visible-position/censored metric, not a claim to know an offscreen position.

Review overlays are generated for test levels 2, 5, 9 and tracking failures.
Blue target candidates and door-edge/seam references require visual review.
Until that review, all object/action summaries are explicitly provisional.

## Ball action selection

Identify the right-hand blue marker on the common GT initial frame. For each
candidate level, take the ball center at its largest observed x, with the exit
policy above, and measure distance to that fixed marker. Do not minimize distance
over the trajectory: an overshooting action must not win simply by crossing the
marker. Select the level of minimum predicted distance, and independently find
the GT optimum the same way. Report exact-level accuracy and GT-distance regret.
Keep incomplete GT environments visible and report the fixed-denominator lower
bound in addition to available-data scores. Marker selection currently uses the
rightmost qualifying blue component and remains subject to visual audit.

## Door action selection

On the final nonpadding 0.3 seconds, all frames must be confidently closed.
Scan levels 1 through 10. The FIRST pair of consecutive levels both predicted
closed is the model's prefer pair. No pair means score zero. Never choose the
pair by comparing it against GT first.

Score `size(predicted_prefer intersect gt_prefer) / 2`, giving 0, 0.5 or 1.
GT pairs are frozen in the config. Exclude only `door-12u-Half-11d-Half` from
action selection; retain it in image/object evaluation. Ambiguous closure and
missing levels are reported explicitly. Full-episode-end coverage is also
recorded because some query windows end before the original recording.

## Execution

CPU scoring uses `.venv-real97-eval-20260909/bin/python` with `--mode cpu`.
LPIPS uses `.venv/bin/python` with `--mode lpips` on one low-priority GPU, after
the CPU job succeeds. Both take the script `scripts/evaluation/score_real97_ball_door.py`.
Outputs are resumable per query. Final summaries remain under `outputs/` during
tracking calibration; do not publish provisional numbers as audited results.

## Version 2: train-only Door closure recalibration

Use `configs/evaluation/real97_ball_door_scores_20260912_v2.json`. The legacy
configuration and version-1 outputs remain unchanged. The old two-sided test
accepted only raw edge/seam gaps of 9 through 19 pixels, incorrectly rejecting
doors that were more tightly closed. Assistant visual review of all 20 available
training GT query endpoints found closed gaps of 1 through 16 pixels and open
gaps starting at 23 pixels. No predicted videos or held-out labels selected the
new thresholds.

The new opt-in policy accepts plausible gaps from -5 through 18 pixels as
closed, calls gaps of at least 22 pixels open, and leaves the intervening band
uncertain. Gaps below -5 indicate implausible tracking and invalidate the
measurement. The final 0.3-second temporal requirement is unchanged. The cyan
reference now denotes the closure acceptance boundary rather than a nominal
panel offset; `calibrated_closure_excess_px` is raw gap minus 18 under this policy.

`recalibrate_real97_ball_door_scores.py` reuses the version-1 per-frame detections
and image errors without overwriting them, applies this same classifier to GT
and every method, regenerates all 100 test Door final-frame review panels, and
records a held-out GT prefer self-check separately. Differences from the provided
environment-level prefer table must be audited, not tuned away using test scores.
Run LPIPS with the same version-2 config after the CPU rescoring job succeeds.
