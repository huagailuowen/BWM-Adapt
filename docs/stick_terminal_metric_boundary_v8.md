# Stick terminal-contact boundary diagnostic v8

This is an independent, opt-in reevaluation. V6/v7 sources, configurations,
measurements, and outputs are preserved. It is not an approved formal metric.

## Defect and measurement-only changes

V6 can accept a vertical-gradient maximum at a search-window boundary even
when the physical edge peak is outside that window. In particular, the
Stage2 q0017 left-side measurement included such a clipped peak. A small
positive apparent displacement is not sufficient evidence of lift in this
situation. Q0003 is a separate, unresolved small-gap/blur ambiguity, not a
confirmed false positive.

The new search requires two pixels of space inside both search boundaries
and a locally bracketed, unmasked gradient peak. Two policies are evaluated:

- `boundary_reject`: retain the original 10-pixel radius; reject clipped peaks.
- `bounded_expand`: only after a boundary peak, try radii 16 and 24 pixels.

Both retain the original contrast threshold, distance penalty, own-side
constraints, floor-tape masking, footpoint sampling, and minimum adjacent
triple selection. Expansion is bounded and is not driven by the desired
success label or the sign of the estimated gap. Every attempt, peak, search
radius, rejection, and accepted displacement is saved. Unresolved edges
remain missing evidence rather than becoming physical failures.

Reference/current contour discontinuities are recorded for the selected
triple but do not alter decisions in this version. They remain a separate
association risk. The native-reference versus resized-prediction resolution
difference is also unchanged and remains important for gaps of a few pixels.

## Frozen physical decision rules

- Only the terminal 0.3 seconds on the source-video clock are evaluated.
- Contact gap band: -6 through 1.5 native pixels; off-table gap: at least 2.5.
- At least six visible footpoints are required for a side decision.
- A cropped reference cannot establish complete off-table clearance.
- Both sides must be off table and within the angle bound for the whole window.
- Persistent visible contact or excessive tilt can establish failure.
- Otherwise the classification is unknown, not failure.
- The existing 3-degree and 5-degree diagnostics remain separate.
- No rise-ratio rule is introduced.

The driver reproduces the V7 decision counts from old measurements before
interpreting changes. Any mismatch is reported explicitly. These thresholds
are diagnostic settings, not newly user-approved physical ground truth.

## Scope and review

Run on CPU compute nodes: 238 complete native GT videos, 40 query-native
controls, and the same 40 GT/Stage1/Stage2 query triplets, 398 videos total.
The prior 52-case manual-review cohort is retained; changed query cases are
additionally marked as targeted review, not independent random evidence.
All results use a fresh job-specific directory. No formal `results/` output
or method ranking is produced. Decidable coverage is not detection accuracy.

```bash
sbatch scripts/evaluation/run_stick_boundary_safe_cpu.sh
```
