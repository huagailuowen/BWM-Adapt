# Soft static-start object-centric diagnostic

This is a new evaluator for Standard step5500 static-start inference job114497.
It preserves all legacy training, inference, trackers, configurations and
results. It is calibration work, not a certified formal benchmark result.

## Frames, views, and object definition

- Sixty factual queries, fifteen environments, thirty train/thirty held-out.
- Model frames: `[0,0,0,0,0,3,6,...,84]`, thirty-three frames total.
- Detect the common anchor at model frame four and all predicted frames.
- Score only the explicit `evaluation_frame_indices`; five observed frames
  and any repeated/clamped tail timestamps cannot enter the metric.
- Invert the aspect-preserving 320x160 view to the native 512x256 crop.
- Measure the center of the detected object's box intersected with the image.
  Boundary visibility is allowed; never extrapolate hidden geometry.
- Original-camera coordinates add the fixed crop offset `(32,224)`.

The silver terminal is detected independently in GT and predictions using
the existing local Grounding DINO Tiny, silver-color filter and association
rules. No environment, action, split label or GT location enters prediction
detection. Association uses source-native frame timestamps, not a mistaken
one-frame interval for stride-three video. All filtered proposals and
rejection reasons are retained. Ambiguous/missing detections are not filled.

## Metrics and safeguards

Center ADE is the mean Euclidean error over valid paired future timestamps.
FDE is evaluated only at the actual last requested future timestamp, never
at an earlier last-valid detection. Report GT coverage, prediction coverage,
paired coverage, complete-horizon counts, and complete-horizon ADE alongside
partial-coverage ADE. Pixel distances use native crop pixels. Both crop-
diagonal and original-camera-diagonal normalizations have explicit names.

Also report origin-relative displacement ADE and a stationary-object baseline
that holds the GT initial center fixed. Baseline and model comparisons use
the same valid query/frame subsets. These are offline measurement diagnostics,
not conditioning inputs. Soft has no action-selection score.

Aggregation separates train/test and environments. Visual review includes a
fixed-seed random six-query cohort per split plus separately labeled targeted
low-coverage/high-error cases. Every query gets per-frame JSONL, side-by-side
GT/Standard tracking video, first/middle/last images, a contact sheet and an
SVG trajectory. High coverage alone is not a certificate of tracking accuracy.

## Execution

All detection and video processing runs on CPU compute nodes. Four independent
array shards each receive eight CPUs and no GPU. An after-success aggregation
job publishes the provisional summary and review index under a fresh
`outputs/eval_real97_soft_static_objectcentric_job...` directory. Do not publish
formal `results/` metrics before independent visual review is complete.
