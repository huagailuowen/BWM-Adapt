# Stick 9/16 DINO and TTT baselines

These opt-in entrypoints reuse the frozen `real916_stick_lift15_driftstable_v1`
manifest and do not modify the Ours/Standard entrypoints or any old configuration.

- Dataset: stick_balance_2026-09-16; 421 training episodes, 45 held-out episodes.
- All nine training environments are available from update one. Baselines have no
  learnable per-environment table and no table-expansion curriculum.
- Per GPU: three environments, eight distinct episodes per environment.
- DINO: uniformly random K=1 or K=2 per environment, with seven or six disjoint
  queries respectively. Frozen DINO features, concat-MLP context head. Support
  and query videos both use the existing 70% coherent lighting augmentation.
- TTT: randomly partition the eight episodes into two independent shuffled
  four-chunk streams. Reset memory per stream, differentiable causal
  write-then-predict, one flow timestep per stream, independent chunk noise.
  Use the existing GPU-resident backward-per-stream path, synchronize once
  per optimizer update; no activation checkpoint replay of mutable memory.
- Both: target EEF, 14D padded action; 256x192 letterbox, 33 frames at stride 3,
  one observed frame, no ROI. Lift probability 0.7; drift episodes have no
  General windows. Lift starts follow max(0,a-15) and the frozen manifest.
- Both exclude the condition and any future VAE temporal block containing
  padded video frames from the flow-matching loss.
- Model peak LR 1e-5, baseline head/memory LR 1e-4, warmup 100, AdamW WD 0.01,
  clipping 0.5. Original BLM step-12000 initialization, at most 5500 updates.
- Two GPUs per job, H200/B200, high priority, 24-hour allocation. Stage base
  weights and videos into locked reusable node-local caches.
- Existing safe loggers save every 500 updates, keep two ordinary checkpoints,
  protect step 2300 and final checkpoints, log every two updates.
  Deadline saving runs at the first completed optimizer update at or after
  allocation start + 23h30, then exits; it is not a mid-update interrupt.

Submit with:
```bash
sbatch --job-name=stick916-dino-k12 scripts/run_real916_stick_baseline_2gpu.sh dino
sbatch --job-name=stick916-ttt-3x2x4 scripts/run_real916_stick_baseline_2gpu.sh ttt
```
