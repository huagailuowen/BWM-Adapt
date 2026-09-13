# Marker-flow diagnosis after the absolute/ratio sweep

Job 114216 left 86 of 214 full GT episodes unresolved at each of 0.5, 1 and
2 pixel absolute thresholds. Changing thresholds is not a remedy for missing
or invalid reference corners. The previous direct-contour experiment remains
rejected because gripper-contaminated rest baselines can invent lift.

This isolated experiment uses lower blue marker sites on the boxes. Optical
flow is always between the same initial image and the current image, with
forward/backward checking, spatially distributed inliers, and a robust rigid
fit. Current color detections initialize the search but never define a new
vertical baseline. Gripper-green features and flat blue floor markers are
excluded. Tracking loss does not reset the initial reference.

The material marker sites are NOT asserted to be actual physical contact
corners. Their image-space motion gets an additional heuristic rotation
uncertainty: 20*abs(sin(rotation)) pixels (30 pixels lever arm when only one
marker component is available), plus a scale term and measured flow/fit
residuals. This guards against a marker rising while a nearby corner remains
grounded. It is not calibrated three-dimensional contact inference.

Keep the agreed absolute >=1 pixel, displacement beyond uncertainty, ratio
>=0.5 and 0.3-second hold. Angle remains auxiliary. Camera is assumed fixed;
background-motion compensation is not implemented. Outcomes are research
proposals until independently inspected. Increased coverage alone is not
evidence of accurate action scoring.

Only CPU compute nodes run video decoding/tracking. All artifacts remain
under outputs, and no training or formal results directory is changed.
The assistant checks waiting jobs at ten-minute intervals, not with a monitor
script. A provisional 90% decidable-coverage gate precedes the formal visual
audit; borderline, occluded, positive and negative examples must all be
reviewed before marking the research task complete.
