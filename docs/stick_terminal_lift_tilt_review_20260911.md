# Stick terminal lift and tilt review

The user-confirmed success criterion is sustained bilateral tabletop clearance
with no large tilt in the final 0.3 seconds of the lift. The earlier trajectory
does not veto terminal success. An initial resting frame provides only the
displacement reference. A rise ratio of at least 1:2 is NOT required: the user
explicitly accepts approximately 12 versus 31 pixels of lift with about 2 degrees
of tilt. Rise ratios remain diagnostic quantities.

## Independent diagnostic branch

Run `scripts/evaluation/run_stick_terminal_silhouette_cpu.sh` using `sbatch` on a
CPU compute node. The branch does not replace any legacy evaluator or config.
It reviews all 238 full GT episodes from the existing source-clock terminal
manifest, including every native frame in its final 0.3-second window.

Dark body silhouettes are seeded by the box markers in their respective image
halves. Their current lower envelopes are compared with the initial local floor
line using multiple brightness thresholds. This is independent of the earlier
sparse projected contact sites and helps expose corner-rotation failures.
Visible rod edges provide a separate image-plane angle estimate.

The 3- and 5-degree thresholds are sensitivity diagnostics, not user-approved
final numerical boundaries. Positive candidates require complete silhouette
geometry and consistent clearance evidence throughout the terminal window.
Missing, clipped, or threshold-sensitive observations remain unresolved.

## Limits

Silhouettes can include shadows or miss a blue-covered corner. The local floor
line assumes a fixed camera and limited depth translation. Image-plane rod
angles are not calibrated 3D angles. Therefore neither positive candidates nor
contact-like evidence are certified physical labels before visual review.
Do not use these candidate counts as the ground-truth success rate or publish
them as formal results. In particular, a rising marker or body center alone
cannot establish that every bottom corner is off the table.
