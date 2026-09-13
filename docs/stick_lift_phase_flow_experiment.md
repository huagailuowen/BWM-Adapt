# Lift-phase optical-flow/contact experiment

This opt-in CPU experiment does not modify legacy detectors, training, datasets,
checkpoints or formal results. Its outputs remain provisional.

The first-frame marker-motion prototype did not pass visual audit. Image motion
can include approach/repositioning and rotation, and blue material sites are not
necessarily the lowest physical contact points.

This branch uses the recorded target-motion `lift_start` to select the last
available pre-lift reference independently in each video. It tracks an actual
sampled bottom contour from a bounded black/blue box component. Artificial crop
boundaries and potential merged objects remain uncertain. Direct-reference flow
is preferred; recent verified anchors provide a fallback with accumulated error,
without resetting the original displacement reference.

Thresholds remain 1 native pixel, minimum/maximum bilateral rise ratio 0.5 and
continuous duration 0.3 seconds. Motion compatible with the uncertainty band is
not treated as proof of physical contact. Query videos have a larger geometry
uncertainty because resizing them to 640x480 does not restore lost detail.

Limitations: commanded lift is not a visual contact annotation. Pre-lift table
contact and a fixed camera are assumptions, not proven calibration. Occluded
feet, depth-axis table sliding and substantial out-of-plane rotations can still
require abstention. Coverage must never be substituted for accuracy.

Evaluate all 238 full GT videos (214 train and 24 test) and the existing 40 query
triplets. Previously reviewed examples and test GT get visual evidence sheets.
Wait 10 minutes between personal Slurm checks. Review failure modes and visual
evidence before claiming accuracy or publishing anything under results.
