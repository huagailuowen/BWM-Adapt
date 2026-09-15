# Dual Ball/Door: Stage2 initialized near the matching z0

This is a separate, known-environment initialization ablation. It does not
replace the mean-of-all-latents inference or modify existing evaluators/configs.
It provides privileged environment-table information and must not be reported
as the same information setting as a shared, environment-agnostic initial Z.

## Configuration

- Config: `configs/evaluation/real97_ball_door_dual_latent_nearz0_20260913_v1.json`.
- Completed dual-model checkpoint and paired table: step 5500 for both tasks.
- Each physical environment uses its existing selected replica `z0`.
- Initial Z: `z0 + epsilon`, with independent `epsilon[j] ~ Uniform(-0.5, 0.5)`.
- The interval is per coordinate, not a bound on the full vector's L2 norm.
- CPU random seed: `(20260913 + int(SHA256(environment)[:8], 16)) % (2**63 - 1)`.
- One reproducible initial vector per environment; all its queries share the
  Z adapted from that environment's unchanged training-set supports.
- FP32 inner optimization: `0.3:10,0.15:10,0.05:10,0.015:10`, 40 updates total.
- No hard clamp, context regularization, ROI weighting, or inference lighting
  augmentation. Only support loss optimizes Z; the model remains frozen.
- The selected training-table Z is logged as a diagnostic target, not used as
  an extra optimization loss or a post-update projection.

## Fixed comparison and artifact reuse

Door reuses the exact reference from job 115971; Ball reuses job 115972.
Support, query, action-template manifests and diffusion seeds are unchanged.
Stage1 z0 videos and their completed-variant markers are copied into the new
output directory. No old Stage2 contexts or videos are copied. The unchanged
reference engine rebuilds GT, generates only Stage2 predictions, and composes
new GT/Stage1/Stage2 comparisons. Test and train grids are separate and sorted
by numerical action level. Fixed-initial-frame action sweeps remain separate
from paired-GT evaluation.

Preparation hard-links the already retained model reference and copies its
paired table. Node-local staging uses the original dual-mean reference identity
to reuse identical Wan, checkpoint, and dataset caches under the existing locks.
No old output, checkpoint, or node-local cache is deleted.

Per-environment `initializations/*.json` records the seed, actual noise vector,
initial/final Z, and distances. `contexts/*.json` retains every inner-step
trajectory. `training_inference_Z_pca.svg` uses the same all-table PCA fit as
the mean-initialization comparison: both training replicas are circles (z0
larger), actual noisy starts are colored crosses, and inference endpoints are
black-bordered triangles. The separate full-trajectory SVG is also retained.

## Submission

Prepare each task with the CPU evaluation Python and `--prepare-only` using
`scripts/evaluation/infer_real97_dual_latent_nearz0.py`. Then submit
`scripts/evaluation/run_real97_dual_latent_nearz0_gpu.sh door` and the equivalent
`ball` command with `sbatch`. Each task requests one low-priority GPU for 12 hours;
H100, H200, and B200 memory classes are allowed. Requeued runs reuse completed
contexts and per-variant completion markers in their own output directory.

No standalone syntax tests, model smoke tests, or post-edit validation were run
as part of adding this opt-in experiment.
