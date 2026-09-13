# Stick balance: absolute rise and bilateral rise ratio

Primary research candidate: for each endpoint, use the minimum displacement
of its two v2 optical-flow-transported reference contacts, not its center or
mean feature displacement. Require reliable flow and both contacts visible.
All distances are in the original 640x480 image coordinate system.

At each evaluated frame require d_left >= 1 px, d_right >= 1 px, and each
displacement strictly greater than its own frozen optical-flow error band.
Only then compute min(d_left,d_right)/max(d_left,d_right), requiring >= 0.5.
Require a contiguous accepted interval lasting at least 0.3 seconds. Padding
and known initial frames remain excluded. A horizontal stationary object or
a single-end pivot does not pass merely because its average motion is small.

The angle is logged, not a primary veto. An additional sensitivity variant
retains the old five-degree guard. Other variants try absolute thresholds of
0.5/1/2 px and ratio thresholds 1/3,1/2,2/3. These are candidate parameters,
not thresholds calibrated against independent human labels. Insufficient
motion means the candidate criterion was not met, not proof of physical
contact. Missing evidence stays unknown; it is not converted into success.

Scale sensitivity normalizes each displacement by its initial GT blue-tape
height, shared by matched GT/Stage1/Stage2. This is an approximate diagnostic
proxy only, not camera calibration. The primary result uses image-space
ratios. Do not silently use gripper-contaminated box contours for scaling.

Use the 262 frozen v2 optical-flow records. Do not rerun training or overwrite
existing metrics. CPU compute-node execution only. Save all variant decisions,
primary per-frame measurements, config snapshot, and SVG curves of the two
absolute displacements, ratio, and auxiliary angle. The result measures
criterion sensitivity; it does not establish action-selection accuracy.
