# Door and Ball: dual-latent reference inference

Both new Ours training runs completed step 5500: Door job 115191 and Ball job
115192. Each physical environment has two independently trained C32 entries.
This evaluation uses the final model and its same-step table, not an older
single-latent model or previously generated Stage1 prediction.

## Context policies

- Stage1 always uses replica z0 of the matching physical environment.
- Stage2 starts from the equal-weight mean of ALL trained entries: 20 for Door,
  16 for Ball. It does not average only the selected Stage1 replicas.
- The initial vector is identical across physical environments within each task.
- No initial noise, hard context bounds, context regularizer, or ROI weighting.
- Update only Z using training Support, in FP32, with support-gradient
  accumulation: 0.3 x 10, 0.15 x 10, 0.05 x 10, 0.015 x 10.
- No Query GT is used in context adaptation. The chosen Stage1 entry is used
  for diagnostics only, not as a supervision target in the support loss.

The prepared evaluation table preserves every context vector exactly once.
Only lookup IDs are remapped: physical IDs select z0, and IDs N through 2N-1
identify the other replicas. The original table is separately preserved as
`reference/full_training_context_table.json`; `latent_aliases.json` records the
physical environment, original virtual ID, replica, and evaluation lookup ID.
The full-table mean is thus unchanged by remapping. Nothing in training is edited.

## Frozen inference contract

Reuse the complete current Support, Query, and action-template manifests from
Door reference 114799 and Ball reference 114807. Door includes the final support
overrides (door-0: 2/5, both 12u environments: 8/10, 3u-4d: 4/6, 4d: 3/5).
Ball keeps ball-0: 1/2, ball-1: 2/3, and the existing support for other balls.

Each physical environment retains two train queries and ten held-out queries:
120 factual Door queries and 96 factual Ball queries. The factual row order is
GT, Stage1, Stage2. Train and test grids are separate, with numerical action
levels sorted left-to-right. Keep the fixed-test-initial-frame action-template
sweep separate; it has no paired future GT and is not a factual metric sample.

Action semantics, normalization, letterboxing, frame stride, padding, diffusion
steps, and per-query generation seeds are preserved from the reference protocol,
with normalization/configuration taken from the new training run. Door uses
49 frames at 256x192 and joint_target_export; Ball uses 61 frames at 320x160 and
joint_target. Both have one conditioning frame and 14 aligned action channels.

The main PCA is fitted on the complete new training table, with both replicas
of a physical environment sharing a color. Training points are circles,
inference endpoints are black-bordered triangles, and the common initial mean
is a small black cross. Trajectories and initialization records are also saved.

## Launch and storage

Prepare references with `prepare_real97_dual_latent_reference.py`, then submit
one `run_real97_dual_latent_reference_gpu.sh` job per task. Each requests one
low-priority H100/H200/B200-compatible GPU for 12 hours. Completed per-query and
per-variant markers support resumption after preemption; a run lock prevents
concurrent writers. Old inference directories are not modified.

The shared-disk reference uses a hard link to protect the completed checkpoint
against later path pruning without another 11 GB shared-disk copy. The existing
locked staging helper copies/reuses Wan, the new model, and required data under
the compute node's `/tmp` cache. It does not load model weights directly from
the slow shared filesystem and does not clean unrelated caches.
