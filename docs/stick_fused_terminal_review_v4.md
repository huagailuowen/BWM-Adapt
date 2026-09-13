# Stick balance fusion diagnostic v4

This opt-in branch evaluates the existing corpus, not new model generations.
It retains all v1/v2/v3 code and results. Outputs are diagnostic, not approved
physical-success labels or formal model-comparison results.

## Tracking evidence

CPU job 114319 tested every v3 tracking failure plus positive controls and
preselected visual cases. Of 47 direct-matching failures, sequential tracking
recovered 32. Two previously trackable cases failed sequentially. This was a
targeted diagnostic, not a random-sample accuracy estimate.

V4 follows intermediate frames while retaining original reference coordinates.
It combines this evidence with v3 anchored matching instead of replacing it.
The corresponding original GT pre-lift frame remains the reference for all
three methods. No additional GT frame is supplied to model generation or TTT.

## Independent checks

- Check tracked full-body geometry against the current visible silhouette.
- Distinguish image cropping from an artificial segmentation search boundary.
- Retain the union of plausible clearance estimates rather than rejecting every
  fixed-size disagreement, even when all estimates establish the same outcome.
- Detect visible rod edges from independently located current boxes, including
  cases where a temporal tracker has lost the box.
- Combine inclination estimates conservatively. Contradictory estimates widen
  the uncertainty interval rather than selecting the convenient prediction.
- Keep the left object on the left and the right object on the right.

## Terminal criterion

Only the final continuous 0.3 seconds on the source-video clock are evaluated.
Both ends must visibly clear the tabletop with a small rod inclination. There
is no mandatory ratio of left/right rise distances. A rotated grounded corner
must not become a success merely because its box center moves upward.

Success requires reliable success evidence throughout this required interval.
A confident violation inside the interval contradicts continuous success;
therefore v4 also reports failure with its exact frame-level witness. For
comparison, every result retains the previous rule that required all terminal
frames to fail. This logical change is separate from tracking changes and must
be checked during review, particularly for isolated noisy detections.

The 3-degree and 5-degree alternatives are sensitivity settings, not newly
user-certified cutoffs. Unknown remains a separate state. Native GT controls
are evaluated at precisely the query GT source timestamps. Review their
agreement before interpreting Stage1/Stage2 differences.

All videos have individual JSON evidence and raw/overlaid endpoint review
images. Validate false positives, false negatives, small clearances, cropped
boxes and rotation/contact cases before promoting anything into results.
