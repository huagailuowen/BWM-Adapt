# Stick contact detector: diagnosis and isolated contour experiment

The v2 unknown count is not an object-detection failure rate and is not the
model's action failure rate. CPU job 114095 audited the saved frame records
from 262 videos, without running inference or touching training jobs.

## Measured v2 failure modes

For 214 full training GT episodes (32,799 evaluated frames):

- 60,167 of 64,652 emitted endpoint records have valid optical flow (93.1%).
- 15,604 endpoint records have disagreeing flow/silhouette displacement.
- 11,355 transported endpoint records extend over two pixels outside the
  current segmented hull. This alone cannot identify which measurement is wrong.
- 14,205 flow-valid endpoint records contain a hidden/clipped reference corner.
- 480 frames report a missing endpoint or rest reference in the silhouette path.
- 20 GT episodes have contact evidence in over 85% of frames but remain unknown
  under the conservative full-video aggregation. One has a few brief positive
  frames, so these cannot all be relabeled failure by majority vote.

The dominant problem is contact geometry and visibility, not simply finding
the blue/black boxes. A category bounding-box detector would not by itself
resolve a millimeter-scale gap or a grounded corner behind an occlusion.

## Isolated v3 experiment

Only new opt-in files are added. Existing tracking results, metrics, training
configs, and checkpoint files are not modified.

Use tall blue tape components as object seeds, rejecting flat tabletop tape
before joining. Segment the connected dark/blue box within an expanded local
ROI using a table-relative dark threshold. Exclude disconnected floor markers
and avoid the previous broad dark-shadow inclusion. Measure the actual lower
contour, with local dark-to-light edge localization, rather than extrapolating
a fitted line or transporting virtual corners under an affine model.

Measure the lowest small contour neighborhood, not average corner height.
Each endpoint is independent. A visible near-ground endpoint remains contact
evidence even if its peer is hidden. A hidden/clipped bottom cannot certify
lift. The first-frame reconstruction offset is recorded and does not count
as upward motion. Image-space one-pixel reliability bands are heuristic;
subpixel interpolation does not prove subpixel physical accuracy.

Keep the v2 video aggregation, 0.3-second duration, and five-degree tilt limit
unchanged for this comparison. This isolates geometry changes from threshold
or aggregation changes. Angle remains auxiliary: a horizontal rod resting
on the table is not success, nor is a box rocking about one grounded corner.

Rerun all 262 frozen videos on a CPU compute node. Reuse frame masks and
matched GT initial references for generated videos. Render the same native
contact crops for visual review. All statuses remain proposals until reviewed;
no formal action-selection precision or large candidate inference is submitted.

Scientific limitation: a single image-space view cannot resolve every hidden
contact. Do not reduce the unknown count by treating missing evidence as lift.
