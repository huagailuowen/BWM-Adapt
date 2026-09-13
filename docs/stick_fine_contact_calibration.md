# Fine contact calibration, before bulk action scoring

The user accepts even a small real separation of both endpoint boxes from the
table. A fixed 2-3 native-pixel cutoff is not the physical definition of lift.
Angle is supporting evidence, not a replacement for contact measurement.

The opt-in v2 measurement uses blue-box texture features, forward/backward
Lucas-Kanade flow, and a robust partial-affine fit against the initial image.
It transports two fitted lower-edge corners per box with subpixel coordinates.
Lower silhouette displacement remains a separate consistency check. These
are visible image-plane contact proxies; no hidden 3-D corner is invented.

Endpoint states are independent. An observed contact-consistent end remains
useful when its peer is clipped, occluded, or has unstable segmentation.
Both complete visible feet must have mutually consistent positive motion
above the measured reliability margins for an airborne proposal. Tilt and
temporal persistence then distinguish a balanced interval from a one-frame
artifact or an excessively tilted lift. Ground contact hypotheses are not
calibrated physical probabilities.

The first generated frame can differ slightly from the common conditioning
frame due to reconstruction/encoding. Record its offset separately and require
subsequent motion as well as displacement above the common rest reference;
do not count a constant first-frame offset as successful dynamic lifting.

Initial uncertainty floors are 0.35 native pixels for rigid tracking and 0.75
for silhouette measurements, with larger tracking margins from forward/backward
and fit residuals. These are provisional engineering margins, not formal
confidence intervals. Subpixel coordinates alone do not imply subpixel physical
accuracy, particularly for predictions generated at 256x192 and viewed at 640x480.

Keep the old 0.3-second interval and 5-degree relative-tilt proposal for comparison;
do not optimize these against the model's apparent success rate. The CPU audit
processes the same 214 training GT videos and 48 existing matched streams and
renders raw contact crops beside overlays. Magnification uses nearest neighbors,
not AI enhancement. Full-episode context is still needed when the box leaves
the fixed contact-detail crop.

No bulk candidate inference or formal action success rate is launched/reported
until this contact criterion has been audited on small-gap, rocking, asymmetric,
occluded and clipped cases. Old detectors, configs and reports are preserved.
