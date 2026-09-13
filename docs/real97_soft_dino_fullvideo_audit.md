# Soft pull full-video DINO tracking audit

This is an opt-in calibration experiment, not BWM generation or latent fitting.
Existing training code, datasets, and evaluation configurations are unchanged.

## Sampling

- Seed: 20260910; 19 Soft pull environments.
- Each in-domain environment: two random train episodes and one random test episode.
- Each held-out environment: three random episodes, still labeled OOD.
- Add the six previously failed detector episodes, deduplicating by environment and episode.
- Sampling manifests preserve episode identity, split, domain, source path and selection reason.
- All native frames of every selected video are decoded and detected. No temporal subsampling.

## Detector and association

Use the existing Grounding DINO Tiny checkpoint and the unchanged provisional
silver-terminal size/color filter. Run on CPU compute nodes only, with no GPU
allocation and no modification of the training Python environment. Four shards
each request eight CPU cores and 32 GB of memory. Model weights are read from
the existing local Hugging Face cache.

Every frame independently produces and saves all DINO proposals, appearance
features and filtering reasons. Deduplicate overlapping proposals with IoU 0.4.
Associate to the recent detection with a 45-pixel-per-frame displacement gate
for at most five frames. Separated, similarly scored candidates are flagged as
ambiguous rather than silently choosing an object. There is no interpolation,
forward filling, optical-flow-only propagation or ground-truth-guided selection.
After longer gaps, global detections can reacquire the target.

Boxes within two pixels of a crop boundary are flagged as potentially partial:
they remain visible in overlays but their centers are excluded from usable
measurements. Missing frames do not necessarily indicate failure: objects can
leave the camera view. Conversely, high coverage is not evidence of accuracy.
Detection-box centers are an approximation to the object's visual center;
rotation, changing visible extent and truncation remain evaluation risks.

## Outputs and interpretation

Each episode saves an overlay video, a nine-frame contact sheet, all per-frame
proposals in JSONL, a trajectory CSV, an SVG plot and a summary. Coordinates
are native 512x256 crop pixels; original 640x480 coordinates add (32, 224).

Reports include usable-frame coverage, partial boundary detections, missing
runs, reacquisitions and consecutive jumps above 20 pixels. The 20-pixel flag is
for visual review, not an automatic claim of tracking error. Training and test
results are reported separately, including OOD. No tracker metric here measures
world-model prediction quality or action-selection success.

The dependent aggregation job runs even if a shard fails, explicitly listing
unfinished episodes instead of silently dropping them. Review low-coverage,
ambiguous and high-jump videos as well as random successful videos before using
these measurements in formal evaluation. The current small-sample calibration
is not an independent labeled tracking test set.

## Revised measurement convention: visible-object center

Per the updated experiment definition, the desired center is the center of
the part of the object visible inside the image, not the extrapolated center
of the complete physical object. The opt-in `visible_box` branch intersects
the detection box with `[0, width] x [0, height]` and uses that intersection's
center. Touching the image boundary does not invalidate this measurement.
Boundary flags remain available for diagnosis. A missing detection is still
missing: no center is invented for an object outside the image.

The original dense audit continues unchanged. The separate CPU replay reads
its saved proposals, performs no DINO model execution, and writes revised
tracks, statistics, SVGs and overlay videos into `visible_center/` inside the
run directory. This is the reporting convention for this experiment going
forward. Previous strict-boundary outputs remain available as raw provenance.
Detection-box centers approximate visible-object centers; these are not
segmentation-mask centroids, and appearance/rotation can still affect them.
