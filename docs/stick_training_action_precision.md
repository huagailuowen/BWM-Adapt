# Stick balance: prediction-selected training action precision

The requested main metric is the fraction of predicted-success actions whose
matched recorded GT also achieves a balanced two-ended lift. It is not angle
reconstruction MAE, nor success for a single handpicked action per environment.

## Candidate protocol

Freeze one earliest annotated full_lift window per approved training episode,
excluding the 16 episodes already used for support adaptation. This yields up
to 198 independent episode candidates from the 214-episode training split.
Every candidate uses its own initial image and recorded target EEF sequence.
Choose windows before reading their outcome videos. Do not repeat dense
overlapping chunks as if they were independent action trials.

Reuse the existing one adapted Stage2 Z per environment across all candidates.
Do not optimize Z using candidate GT. Save direct Stage1-table predictions as
the same-pool comparison. This is explicitly a training-candidate evaluation;
it does not establish held-out-action generalization.

## Success measurement proposal

Use each endpoint box's visible lower silhouette, including dark body corners,
not just the blue tape center. Small blue table markers are excluded before
endpoint grouping. Derive a camera-plane rest-contact reference from the common
initial image. Measure the weakest lower-edge clearance at both ends and the
rod angle change relative to its rest orientation. GT and prediction use the
same measurement code and common initial reference, without environment labels
or predicted outcome access inside the detector.

The initial, deliberately provisional thresholds are both lower edges at least
3 native pixels clear, at most 5 degrees rest-relative tilt, sustained for at
least 0.3 seconds. Also save a 2/3/4-pixel by 3/5/7-degree sensitivity table.
Slight lift is sufficient: no large center-height requirement. A rocking box
whose lowest corner remains grounded must fail the two-ended clearance test.
Horizontal but grounded rods fail. Clip/occlusion/unstable-geometry cases are
unknown, never silently successful. This is a monocular visible-contact proxy;
it cannot prove that an invisible back corner is off the table.

Before formal scoring, visually audit overlays spanning balanced lifts,
one-ended lifts, rocking contact, barely lifted cases, and unknowns. Freeze
the audited thresholds rather than tuning them to maximize model precision.
Do not convert provisional labels into certified physical success rates.

## Reporting

Report prediction-selected count and fraction, matched GT success/failure/
unknown counts, precision, the original candidate-pool success prevalence,
and all per-environment counts. Environments with no selected actions have
undefined precision and must remain visible in coverage reporting. Unknown
GT among selected actions remains in lower/upper precision bounds rather than
being quietly removed. Separate the small existing 16-train-query pilot from
the later full-candidate result.

## First CPU audit

`scripts/evaluation/run_stick_training_action_precision_cpu.sh` processes all
214 native training videos and the 48 existing matched GT/Stage1/Stage2 videos
for 16 training queries on a CPU compute node. It saves every measured frame,
per-video decisions, sensitivity labels, review sheets, and an HTML index.
It also freezes the full candidate manifest. It does not launch bulk GPU
prediction until the contact criterion has been audited. Existing trackers,
inference products, and training configs are not modified.
