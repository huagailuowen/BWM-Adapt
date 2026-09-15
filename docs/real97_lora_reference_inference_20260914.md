# Real97 Door and Ball LoRA TTA

This opt-in experiment uses each task's protected Standard pooled world-model
step5500 checkpoint, never an Ours/DINO/context-conditioned checkpoint. No new
pooled-model or outer-loop training is submitted.

Use the existing Event80 LoRA implementation without tuning on query results:
rank8, alpha8, all DiT Linear modules named q/k/v/o, FP32 adapter parameters,
50 AdamW steps, learning rate 1e-4, weight decay0, gradient clipping1.
A is shared randomly initialized, B starts at zero, so initial residual output
is exactly zero. Reset both matrices to that shared initialization per physical
environment; create a fresh optimizer. At each step average the selected
support flow-matching losses, accumulating backward sequentially. Random noise
and timesteps follow the existing adaptation implementation.

Freeze all original DiT and action-encoder parameters. Queries use no-gradient
rollouts and do not update LoRA. Save an environment's adapted state atomically,
reuse it for all its factual queries and fixed-initial-frame action sweeps, and
reload it after requeue. Record support losses, module names, parameter count,
checkpoint identity, manifest hashes, and deterministic seeds.

Door: 120 factual queries and 100 action sweeps.
Ball: 96 factual queries and 80 action sweeps.
Support/query/template manifests are exactly the current reference manifests.
Door has two supports per environment; Ball-0/1 have two, others one. Do not
reselect or silently duplicate supports. Support/query episodes are disjoint.
Single observed frame and aligned 14D recorded target actions. Door 49 frames
at 256x192; Ball 61 frames at 320x160. Inference50 steps, CFG1, no ROI weighting
or inference lighting augmentation. Training normalization remains unchanged.

Factual comparisons have three rows: GT, the already completed Standard
prediction, and LoRA. Verify the Standard marker has the same query/action/
frame identities. Grids are level-sorted with train and test separate.
The action-template extension has no paired future GT and is not a factual
prediction metric. Its diffusion seed matches the existing Standard extension.

Each task requests one low-priority GPU for12h; H100/H200/B200 allowed. Keep
new output directories, protect the Standard checkpoint by hard link, and reuse
the same locked node-local Wan/model/data cache identities as Standard/DINO.
Do not overwrite prior outputs, source checkpoints, or existing implementations.

LoRA has no C32 table; do not present its adapter weights as a latent-Z PCA.
Only support adaptation and inference are run, with input/runtime guards inside
the compute allocation. No standalone post-edit syntax or behavior tests run.
