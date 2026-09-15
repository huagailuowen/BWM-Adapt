# Soft balanced-nine dual-latent static inference

Submit the opt-in launcher with dependency afterok:115613. Training failure must
not release inference. One low-priority GPU, 12 hours; H100/H200/B200 allowed.
No existing evaluator or configuration is modified.

At compute-job start, choose the highest fully published matching model/C-table
pair (minimum step 5100, intended final step 5500), freeze that choice for retries,
and protect the pair in a separate snapshot. Never pair different steps.
Use node-local locked Wan, dataset, and checkpoint caches. No checkpoint cleanup.

Only nine physical training environments are evaluated:
soft-1l, soft-2l, soft-4l, soft-1r, soft-7r, soft-5r, soft-2m, soft-6m, soft-8.
Use the training job's frozen physical train/test manifests and exclude disabled
soft-6m episode 10. Stage1 chooses z0 without quality selection. Stage2 starts
from all 18 latents' mean, updating Z only with 40 FP32 support-loss steps:
0.3 x10, 0.15 x10, 0.05 x10, 0.015 x10. No finite clamp, context regularization,
ROI weighting, or inference lighting augmentation.

Per environment: two train queries and all held-out episodes, 36 queries total.
Prefer historical eligible train query IDs; fill by fixed-seed hash ordering.
Queries begin at episode frame zero: 0,3,...,96, with final-frame padding.
One observed frame and 32 predictions; canonical target EEF and normalization
are identical to training, eight channels plus six zeros, aligned to video.
Videos use 20/3 fps. Exclude the observed frame and padding in future scoring.

Four distinct train support episodes are selected per environment. Prefer
historical eligible episode IDs and legal episode-start windows; if necessary,
consider other annotated static starts. Reuse native-coordinate visible-center
tracks, with the existing CPU detector on the compute node as fallback.
Require net displacement >=15px, excursion >=25px, and both motion directions;
prefer two left and two right. Reserve query episodes before support selection.
Save all candidate measurements and selected windows. Do not silently prune
environments if selection fails. No test images or predictions select supports.

Outputs: GT/Stage1/Stage2 raw videos, query comparisons, environment grids,
separate train/test grids, all-18-latent PCA with matched environment colors,
black-bordered inference triangles, and full inner-update trajectories.
Historical fifteen-environment/five-history-frame experiments have different
query coverage; align actual episode IDs and frame indices for later metrics.
Soft has no action-selection metric. Formal CPU scoring is separate.

No standalone syntax tests, model smoke tests, or post-edit validation were run.
GPU preflight and dataset-contract guards execute inside the compute allocation.
