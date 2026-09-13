# Real97 factual reference evaluation, v3

This new branch leaves all previous training and inference configurations intact.
Run CPU preparation before allocating a low-priority single GPU per task.
H100, H200 and B200 are permitted. No world model runs on a login node.

## Updated user decisions

- Save GT, direct Stage1 training-table inference, and support-adapted Stage2.
- Stage1 is explicitly an environment-label-informed reference, not blind TTT.
- Ball prefers support whose peak x is 150-220 in original 640x480 coordinates.
  This is a preference, not a displacement range or a change to the 280-330 goal.
- Door uses two supports at minimum-success-level minus one and plus two.
  Missing boundary levels remain unresolved, never invented or silently clamped.
- Fixed-test-image action-template sweeps are extension experiments without
  unmatched-GT scores. All queries in this first run use their own original action.

## First tasks

Stick uses published step5500 weights and matching table. Soft uses published
step5300 weights plus the saved step5500 all-context table: the final 200 steps
only changed Z, and step5500 model publication did not finish. Preparation pins
the immutable model with a hard link and copies its table and original action stats.

Supports come only from frozen training manifests. Stick brackets a balance
estimate fitted from train outcomes, using two informative nearby placements.
Soft selects four distinct episodes with real left/right movement, preferring
two per direction. At least one of each direction is required. Its provisional
motion gates are 15 px net x and 25 px directional excursion on the actual chunk.
Boundary contact is valid; the center is the visible-box center. Support review
images and selection measurements are saved. Environments failing preparation
are explicitly listed, never silently replaced with test supports.

Each eligible environment uses all test episodes and two train query episodes,
disjoint from its support episodes. All query starts come from frozen legal
training windows; Stick uses full-lift windows. The first batch is a development
experiment, not a claim of certified tracker accuracy or frozen lift thresholds.

Stage2 starts at the mean of the active training Z table, uses FP32 context-only
SGD with averaged support gradients, and updates 40 times: 0.30, 0.15, 0.05,
0.015 for ten steps each. Clamp to the training range [-1,1]. No ROI weighting,
context regularization or evaluation-time lighting augmentation. The oracle
environment Z is never passed as an adaptation target. Save every update.

Original training loaders, target-action representation, min/max stats,
letterboxing and stride3 remain unchanged. Stick produces 41 model frames at
256x192, Soft 33 at 320x160; action tensors remain 14-dimensional as trained.
Raw videos use the exact 20/3 playback FPS, and GT follows identical frame indices.
Save per-query three-row comparisons, per-environment train/test grids and a
training-table-fitted PCA using training circles and inference triangles.

Low-priority requeues reuse completed per-environment contexts and complete query
outputs. Wan and checkpoint copies use locked node-local reusable caches. Never
delete another run's cache. No model training, baseline training or action editing
is started by these scripts.
