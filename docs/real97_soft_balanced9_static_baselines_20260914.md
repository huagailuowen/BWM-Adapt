# Soft balanced-nine static-start DINO and TTT baselines

These new entrypoints/configs do not alter legacy loaders, trainers or configs.

## Data and shared training contract

Use soft_pull_9_7_crop_512x256_static and exactly these physical environments:
soft-1l, soft-2l, soft-4l, soft-1r, soft-7r, soft-5r, soft-2m, soft-6m, soft-8.
Use 146 eligible training episodes / 2680 approved static-start windows.
The 18 held-out episodes never participate in training, support selection or
normalization. Exclude soft-6m episode 10. Empty legal-start lists have no fallback.
Reuse the Ours physical manifest and train-only action normalization, not its
18 duplicated latent aliases. No latent-table curriculum is applied to baselines.

Each rank independently samples 5 distinct environments, then 6 distinct episodes
per environment, then one uniform legal window per episode. Ranks may sample the
same physical environment independently. All nine environments are active from
the beginning. A two-GPU optimizer update samples 60 chunks; microbatch is 1.
Do not multiply optimizer-step counts by GPU count.

Video: 1 observed + 32 future frames at stride 3, native start through start+96.
Resize with letterbox to 320x160. Action: recorded target EEF, using the same
training-only quaternion convention and min/max normalization as Ours:
8 target coordinates followed by 6 zeros, synchronized to video indices.
Use no ROI. Online lighting probability is 0.7; spatial gradients are +/-0.08 and
sensor noise std is 0.004. Other mild gain/contrast/gamma/tint defaults are unchanged.

## DINO

Use the established frozen DINOv2 -> concat-MLP support encoder.
For each sampled environment, choose K=1 or K=2 uniformly, then randomly select
that many of the 6 episodes as support. The remaining 5 or 4 are disjoint queries.
DINO support video receives the same temporally coherent lighting transform
before feature extraction; its frozen visual features are reused across queries.
Query augmentation uses the existing Wan training branch.
Wan LR 1e-5, support-head LR 1e-4, warmup 100 steps. No environment lookup table.

## TTT

K=1/2 is the DINO support-count parameter, not a replacement for the TTT protocol.
Keep the established differentiable write-then-predict causal scan, with one
randomly ordered six-chunk sequence per environment, resetting state between
environments. Predict/loss at every position exactly as in the original protocol.
Five independent streams per rank, 30 sampled chunks per rank.
Use the GPU-resident fix: no saved-tensor CPU offload, no activation checkpoint
replay of mutable state, backward once per stream, gradient synchronization once
per optimizer update. All six positions remain differentiably connected.
Wan LR 1e-5, TTT slow-parameter LR 1e-4, warmup 100 steps.

## Scheduling and saves

Two independent high-priority two-GPU jobs; allow H200 or B200; limit 24 hours.
Maximum 5500 optimizer updates. Start from ckpt/BLM/step-12000.safetensors.
Reuse locked node-local Wan, BWM checkpoint, dataset and DINO caches.
At the first optimizer boundary after allocation start + 23h30, save a protected
checkpoint and exit. This preserves the existing allocation-clock implementation;
it does not interrupt an in-flight backward exactly at the deadline.
Save every 500 updates and keep the latest two ordinary completed bundles.
Step2300, final and deadline checkpoints are protected from ordinary pruning.
DINO stores its learned support head with Wan; TTT stores learned memory/gates
with Wan. Neither method has a persistent per-environment C table.
These existing baseline checkpoint formats do not save full optimizer resume state.

Submit the separate new launchers, each with its corresponding config argument.
No tests or performance benchmarks were run as part of creating these variants.
