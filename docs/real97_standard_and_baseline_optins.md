# Real97 pooled baseline and opt-in compatibility

## Existing experiments

Historical YAML files and the shared Wan augmentation/data-loader behavior are
unchanged. DINO defaults to its original support path. DINO and TTT window
samplers default to `legacy`, which executes the same `rng.choice(indices)`
without drawing an additional random number. No new inference augmentation is
enabled.

## DINO support augmentation

New training configs can set:

```yaml
video_light_augmentation_enabled: true
video_light_augmentation_probability: 0.70
dinov2_support_light_augmentation_enabled: true
```

Both switches and training mode are required. The support branch calls the
existing Wan transform on a copied video before frozen DINO feature extraction,
once per selected support. All queries reuse those support features. Query
augmentation remains in Wan. Support and query draws are independent, using the
same augmentation distributions; they do not share one forced perturbation.
Gain/contrast/gamma/tint/offset/spatial gradients and fixed temporal noise follow
the existing configured transform. No raw files or cached dataset videos change.

## Episode-first window balancing

New DINO configs can set:

```yaml
dinov2_distinct_actions: true
dinov2_action_key: episode_index
dinov2_window_sampling_mode: uniform_episode_then_window
dinov2_window_kind_field: sampling_kind
dinov2_preferred_window_kind: precise
dinov2_preferred_window_probability: 0.5
```

TTT uses the same option names with `ttt_` instead of `dinov2_`. Select distinct
episodes uniformly, then select the annotated window branch, then uniformly
select a window within that branch. This is not action-level-balanced sampling.
Ball/Door use their frozen precise/general 50/50 manifests. Stick uses lift/general
70/30; Soft only has general windows. If only one branch exists, keep the episode
and use its available branch, matching the real-data grouped sampler. Empty
episodes are errors, not silently pruned. Use train-only manifests for both
support and query.

## Standard pooled runs

The dedicated runner reuses `build_dataset` and `WanTrainingModule`, not a new
video/action preprocessing implementation. The four generated configs inherit
the real97 target actions, train manifests, train-only normalization, letterbox
resolutions, frame strides, last-frame padding and 70% light augmentation. No
ROI. No Z, adapter, C optimizer, or curriculum: all eligible training environments
are available from update 1. Train DiT and action encoder, LR 1e-5, weight decay
0.01, initial model warmup 100 updates.

Per rank: Ball/Door/Stick use 3 environments x 8 distinct episodes; Soft uses
5 x 6. Ranks sample independently. One chunk per selected episode; gradient
accumulation over these chunks defines one optimizer update. Two ranks give
global batches 48/60, not twice as many optimizer steps.

Each run stops at 5500 optimizer updates or the first completed optimizer update
after actual Slurm elapsed 23:30, whichever comes first. The deadline is computed
once from the allocation StartTime, including staging/loading time. There is no
polling of Slurm or disk in the training loop. Checkpoint serialization cannot
safely interrupt an unfinished optimizer update; the 30-minute margin covers the
boundary/save time, not an absolute guarantee against node failure.

Ordinary model checkpoints are atomically published every 500 updates, and only
the newest two ordinary files are retained. Step 2300, final and deadline
checkpoints are written under `protected/`, outside ordinary pruning. Standard
has no C table. These are inference/model checkpoints, not full Adam resume
states.

Two four-GPU allocations each launch independent two-GPU workers with their own
outputs/logs and rendezvous ports. A worker failure does not cancel its peer.
CUDA probes run separately for each pair before model loading. Existing
node-local Wan/BLM and dataset staging are reused; no CPU fallback is allowed.
