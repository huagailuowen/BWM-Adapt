# Stick balance 2026-09-16: matched Ours / Standard

These are new entrypoints and configs; legacy experiments are unchanged.

## Shared data contract

- Dataset: datasets_real/stick_balance_2026-09-16.
- Explicit frozen episode split: 421 train, 45 test, nine environments.
- Model input: AgentView and exported tool-tip EEF target only. Normalize the
  eight target channels using shared train-only min/max, zero constant channels,
  then append six zeros. No measured state, outcome, batch, or filename features.
- Images: 256x192 letterbox, 33 frames at native indices s+3*k, 20Hz source.
  One initial condition and 32 predictions, 4.8 seconds between first and last.
- 113 clear-drift episodes (106 train / 7 test): Lift only, s in [e,min(e+10,N-1)].
  Do not include General candidates for these episodes.
- Other 353 episodes: 30% General [0,max(0,N-97)], 70% Lift [max(0,a-15),a].
- Every integer start is enumerated as metadata, not as a new video. Samplers
  first select episodes uniformly and then select branch and start uniformly.
  Missing General for drift episodes falls back to the sole Lift pool.
- Video and target indices use the same clamped endpoint. Each row records its
  valid sampled frame count. Both methods exclude initial-condition loss and
  any VAE temporal block containing synthetic padded frames. With the causal
  4x temporal VAE, floor((valid_video_frames-1)/4) future latent frames contribute.
  This conservatively omits up to three real tail frames mixed with padding.
  Fail on zero complete future blocks; never silently remove episodes.
- 70% temporally coherent light augmentation; 30% unchanged. Retain the old
  mild gain/contrast/gamma/tint/offset defaults, gx/gy within +/-0.08,
  sensor-noise std 0.004. No ROI.
- Each rank: three environments, eight distinct episodes per environment,
  one chunk per episode. Microbatch one, 24 forwards per update per rank.
  Two ranks give 48 chunks per optimizer update.
- BF16, gradient checkpointing, model LR 1e-5, weight decay 0.01,
  initial model warmup 100 steps; BLM step-12000 initialization.
- Upper limit 5500 updates; logs every two updates.

## Ours

One independently Uniform(-1,1) initialized 32D Z per environment. Retain the
old Stick context clamp [-1,1], context weight decay zero, and legacy optimizer
semantics. New-context and all-context LR are both 0.03.

- Steps 1..300: first five environments, frozen Z, model-only warmup/training.
- 301..1300: first five environments, five 200-step phases:
  new Z, all active Z, model, all active Z, model.
- 1301..2300: add remaining four and repeat the same five phases.
- 2301..5500: all nine, alternate 200 context / 200 model steps.
- Frozen parameter groups do not update.
- Save paired model/Z at model-phase endings; retain last two ordinary pairs.
  Steps 2300 and 5500 are protected separately; phase-transition Z logs remain.
- Fixed train-only diagnostic validation every 100 updates, inherited from
  old Ours. It is not a test score and does not update parameters.

## Standard

No latent table or curriculum. Sample all nine training environments from step
one with exactly the shared chunk/action/augmentation/masked-loss contract.
Train DiT and action encoder every update. Save every 500 updates, keep last two
ordinary checkpoints; protected 2300 and final checkpoint.

Same total update count does not imply equal model-only optimizer update counts.

## Submission and safety

Submit scripts/run_real916_stick_2gpu.sh with argument ours or standard.
Each independent high-priority yejin job requests two H200/B200 GPUs for 24h.
The preparation script uses a lock and atomically publishes one shared manifest,
with frozen source policy/splits/normalization and fingerprints. Metadata source
changes require a new manifest version, not overwrite. Source data are read-only.

Both jobs reuse the existing node-local immutable Wan/BLM/data cache and write
independent output directories with Slurm job IDs. At allocation StartTime plus
84600 seconds, save at the first completed optimizer boundary. Standard then
exits; inherited Ours may continue after its protected wallclock save.

