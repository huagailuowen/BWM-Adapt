# Stick: optical flow should measure motion, contours should not invent it

The v3 direct contour experiment is NOT accepted as a formal action metric.
Its reduced unknown count includes false apparent lifts: when a reference
body merges with the nearby gripper, an invalid bottom becomes the baseline.
After segmentation recovers, the static baseline error appears as motion.
In the q0021 right endpoint, that produces approximately 17 pixels of spurious
upward displacement. A finite contour measurement is not proof of a valid
rest reference. These output files are retained as diagnostic counterexamples.

## Frozen-flow ablation

Use the existing v2 direct-reference, forward/backward-checked optical flow
and rigid-fit quality. Keep the original reference contacts and first-frame
offset policy. Remove ONLY the dynamic silhouette-agreement veto. Compare
the original flow uncertainty band and twice that band. Do not change the
0.3-second hold rule, five-degree tilt rule, temporal masks, or query/support
data. Do not report a lower unknown count as improved accuracy.

For each endpoint, retain separate transported contacts and use the lowest
visible contact displacement, not the average upward motion of texture points.
Rotation about a grounded corner can move most of the box upward without
lifting the entire box. A visible grounded contact is evidence against lift;
hidden contacts cannot certify that the whole endpoint is airborne.

All 262 frozen videos are analyzed on a CPU compute node. No GPU inference or
training job is modified. SVG diagnostics compare the flow contact motion to
the rejected contour measurements, showing gaps when flow was unavailable.
Derived states remain proposals, not ground-truth action labels.

## Remaining calibration requirements

Optical flow does not repair an incorrectly placed reference contact. Before
formal scoring, validate initial contacts independently of gripper/shadow
components; reject contaminated references rather than treating them as zero
height. Use several spatially separated points on each box and motion
consistency, never a single point's vertical velocity. Reacquisition must not
silently reset the contact baseline. Static-scene camera motion, uncertain
occlusions and 3-D perspective changes need explicit treatment if present.

Do not claim that a single monocular view resolves every hidden contact or
that subpixel flow accuracy equals physical subpixel gap accuracy.
