# Real97 Door/Ball TTT reference inference

The final deadline-protected checkpoints are Door step 3656 from job 115337
and Ball step 4144 from job 115338. Both jobs completed successfully; neither
checkpoint is labeled as step 5500. Model weights are not changed by evaluation.

## Match the trained TTT protocol

The trained protocol is `oneminute_write_then_predict`, not the older policy
that writes support once and then keeps memory read-only for an entire rollout.
At EACH diffusion timestep, reset to learned initial fast weights, replay every
selected support in its frozen order at that same noise timestep, process the
current noisy query with the trained causal update-then-read scan, and clear
the state. Query GT futures are never inputs to the memory writes. State does
not cross query, action-template, environment, or denoising-step boundaries.

The opt-in multi-support wrapper reuses the established K=1 support preparation
and timestep/noise construction. VAE encoding of each observed support happens
once per environment; clean support latents and fixed seeded noise are reused.
Each replay re-pins the first conditioning latent, matching the K=1 runner.
There is no C32 latent, latent-table mean, or Ours-style gradient optimization.

Use all of the same frozen training supports as the current reference:
Door has two per environment; Ball has two for ball-0 and ball-1, and one for
the other six environments. Supports remain query-episode-disjoint. The stream
prefix is shorter than the four-chunk training streams, which already train
predictions at earlier stream positions; no extra unrelated support is added.

## Query and output contract

- Door: 120 factual queries, 49 frames, 256x192, joint_target_export.
- Ball: 96 factual queries, 61 frames, 320x160, joint_target.
- One conditioning frame, stride one, aligned 14D targets, existing letterbox
  geometry, padding, and train-only normalization from the TTT training run.
- Reuse factual generation seed 20260910 + query index and 50 diffusion steps.
- No inference lighting augmentation or ROI weighting.
- GT/TTT factual grids are split into train/test and sorted by numerical level.
- Fixed-test-initial-frame action-template sweeps are separate and have no
  matched future GT; their seeds are 20260910 + 100000 + anchor query index.
- Per-denoising-call memory write counts and support order/noise seeds are saved.
- TTT has no comparable C32 PCA; do not fabricate a latent-table visualization.

The preparation script freezes metadata and hard-links the protected checkpoint
before submission. GPU workers reuse locked node-local Wan, original BWM,
checkpoint, and recorded-file caches. Each task requests one low-priority
H100/H200/B200-compatible GPU for 24 hours. Completed factual and counterfactual
markers permit resumption after requeue, and a run lock prevents concurrent
writers. Existing Ours, Standard, DINO, and historical TTT outputs are untouched.

These are new code/config branches. Existing inference entry points, training
code, and historical configurations are unchanged. The eventual scoreboard must
retain actual training steps and the different adaptation mechanisms.
