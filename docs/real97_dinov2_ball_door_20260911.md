# Real97 DINO concat-MLP Ball and Door

Two independent high-priority two-GPU jobs, allowing H200 or B200. No TTT or
Soft training is submitted by this experiment.

## Data and batch contract

Reuse the frozen real97 target v2 train manifests and train-only action stats
used by Standard. Ball: 61 frames, 320x160, joint_target. Door: 49 frames,
256x192, joint_target_export. Native stride 1, one observed frame, letterbox,
last-frame and action padding, no ROI.

Each rank independently selects 3 environments and 8 distinct train episodes
per environment through the existing Standard PooledEpisodeSampler. Select one
window per episode with the existing precise/general 50/50 rule. Within each
environment uniformly choose K=1 or K=2, then uniformly select those supports
without replacement. The remaining 7 or 6 episodes are queries. This is NOT
eight queries plus additional supports and is NOT action-level-balanced.
A global optimizer update consumes 48 chunks, comprising 6-12 supports and
36-42 queries. Average query loss within each environment, then across
environments and ranks. All eligible train environments are active from step 1.

## Model and augmentation

Start Wan/BLM from ckpt/BLM/step-12000.safetensors, not an Ours checkpoint.
Frozen local DINOv2-base observes 8 sparse frames per support. Train the
visual projection, interval-action GRU, concat MLP, and bounded 32-D output
head. Multiple supports are averaged BEFORE the output head and sigmoid.
Train DiT, action encoder, physical-context encoder jointly with this head.
There is no per-environment trainable Z table and no alternating C/model phase.

Reuse DINOv2WanTrainingModule and DINOv2AmortizedContextEncoder unchanged.
Enable both the global light augmentation and the support-specific switch.
Probability 0.70, spatial gradient bounds +/-0.08, noise std 0.004, with the
same existing gain/contrast/gamma/tint/offset distributions. All frames of a
selected support use one augmentation draw; its frozen DINO features are
reused by its queries. Query augmentation is independent and uses Wan's
existing training path. Original videos are not changed.

Wan LR 1e-5; amortized head LR 1e-4; both warm up for 100 optimizer updates.
Weight decay 0.01, BF16, gradient clipping 0.5, activation checkpointing on.
5500 maximum optimizer updates, logging every 2 updates.

## Lifetime and checkpoints

Deadline is allocation StartTime + 84600 seconds, including staging and model
loading. At the first completed optimizer update reaching that deadline,
publish a protected checkpoint and stop. Only an in-memory clock and tiny
rank broadcast occur in the loop; no filesystem or Slurm polling.
Checkpoint writing still takes time; this is not a guarantee against abrupt
node failure or a save that exceeds the remaining allocation.

Ordinary saves every 500 updates retain the latest two complete bundles.
Step 2300, final and deadline saves live in protected/ and are never pruned.
Each safetensors file contains BOTH Wan parameters and the matching DINO
head. Metadata and completion markers identify a fully published bundle.
These are model checkpoints, not full optimizer-state resume checkpoints.
No existing checkpoint is overwritten. Jobs use separate job-ID directories
and --no-requeue to avoid accidental restart into an occupied output.

Wan, BLM, DINO and raw training files are staged on compute nodes under the
existing locked /tmp cache. Other runs' caches and outputs are not removed.
The old baseline entrypoints/configs remain unchanged. This new runner
explicitly rejects multi-frame history; Soft history5 will need a later
dedicated opt-in integration.
