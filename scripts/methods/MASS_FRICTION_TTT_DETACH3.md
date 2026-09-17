# Mass Friction TTT: six chunks with a three-chunk detach boundary

This opt-in training variant keeps six sampled distinct actions per environment,
but truncates outer gradients after the first three chunks. It carries the learned
fast-state values into chunk four rather than resetting to shared initial weights.

| Setting | Value |
|---|---|
| Active environments | Same 84-environment manifest as the existing Mass Friction baseline |
| Hardware request | Two GPUs, high-priority yejin, 24 hours, 128 GiB host memory |
| Per-rank grouped batch | Four environments, six chunks per environment |
| Global effective batch | 2 x 4 x 6 = 48 chunks per optimizer update |
| Chunk sequence | Same six-action sampled order as the unsplit runner |
| Segment 1 | Chunks 1-3 share differentiable fast state; backward their summed losses |
| Boundary | Detach all fast-state tensors, enable local gradients on the detached leaves, retain their numerical values |
| Segment 2 | Chunks 4-6 continue from the carried state; backward their summed losses |
| Loss normalization | Each segment sum divided by six times the number of local environments |
| Optimizer and clipping | Once after all local environments and both segments, as before |
| DDP synchronization | With sync_per_update enabled, synchronize only at the last segment of the last local environment |
| Diffusion timestep | Same sampled timestep across the six chunks, retaining the previous behavior |
| Saves | Every 200 updates or 60 minutes; retain the latest two checkpoints |

The Wan and TTT learning rates, full-token scan, 64-token inner minibatches,
eight selected TTT layers, FP32 fast-state arithmetic, and saved-tensor budgets
remain unchanged. Within each three-chunk segment, higher-order gradients remain
enabled. The second segment's loss cannot backpropagate through the first
segment's fast-state updates. This is truncated-gradient training, not an exactly
equivalent memory implementation of the original six-chunk gradient.

The runner flag `ttt_detach_every_chunks` defaults to zero, preserving existing
experiments. The variant sets it to three and requires causal write-then-predict
with per-stream backward. Each environment still starts from shared initial fast
weights; only the midpoint carries detached values.

Configuration:
`configs/train/train_mass_friction100_active84_ttt_kqv_prequential6_detach3_4envpergpu_2gpu_24h.yaml`.

Launcher:
`jobs/run_train_mass_friction_ttt_detach3_2gpu_high_24h.sh`.
It reuses the shared node-local Wan/BLM cache and creates a job-specific output
directory. The failed previous run produced no resumable update, so this variant
starts from the original BLM initialization rather than pretending to resume.

First-update logs show the chunk index and allocated CUDA memory. Segment-boundary
logs record completed backward and whether the numerical state was carried.
Splitting reduces cross-chunk graph retention but does not guarantee that a
single chunk's full-token higher-order scan fits; no such claim should be made
before observing the actual training run.
