# Soft known-family initialization ablation

This opt-in experiment changes only the Stage2 initial latent, preserving all legacy
configurations, code, inference outputs, and checkpoint retention.

## Group assignment

- L: soft-1l, soft-2l, soft-4l, soft-2m, soft-6m (10 latent rows).
- R: soft-1r, soft-5r, soft-7r (6 latent rows).
- 8: soft-8 (2 latent rows).

Both learned replicas of every environment contribute equally to its family mean.
The user explicitly assigns 1R to R, even though its two learned replicas are split
between neighborhoods. The middle groups are assigned to L using the training
table only. This is a known-family prior ablation, not unknown-environment inference;
the group-8 mean also uses the two table rows of that specific environment.
No query ground truth is used to choose or optimize the initial latent.

## Frozen comparison

Source: outputs/infer_real97_soft_balanced9_dual_static1_train115613_job116188.
Use the same protected step5500 model and table, the same four supports per
environment, all 36 original queries (18 train, 18 test), and the same seeds.
Reuse the original Stage1 videos without regenerating them.
Stage2: Z only, FP32, 40 updates with 0.3/0.15/0.05/0.015 for 10 updates each.
No hard bounds, latent regularization, ROI weighting, or inference lighting.
One observed frame plus 32 future frames, stride 3, target EEF, 320x160.
Raw GT is independently written; no original output is overwritten.

The output stores group_initialization.json (all means and membership),
per-environment context optimization traces, raw videos, comparisons, train/test
grids, and training_inference_Z_pca.svg. PCA is fitted on the unchanged 18 training
rows: circles for training time, black-bordered triangles for inference time,
hollow diamonds for group means. The trajectory plot records the actual starts.

Submit scripts/evaluation/run_real97_soft_family_mean_static_gpu.sh with sbatch,
not bash on the login node. A single low-priority GPU (H100/H200/B200) is used.
Existing node-local Wan, dataset and checkpoint caches are reused. Interrupted work
resumes by environment and completed query variant. No checkpoint is modified or deleted.
