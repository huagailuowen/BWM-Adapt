# Ours Soft pull: five observed frames and target EEF

Config: `configs/train/train_real97_soft_eef_history5_c32_2gpu_20260910_v1.yaml`.
Entrypoint: `scripts/train_real97_soft_history5_grouped_context.py`.

This run starts from `ckpt/BLM/step-12000.safetensors`. It does not resume
the old joint-action run or a Standard checkpoint. All outputs belong to a
new job-specific directory. Existing Standard jobs are not stopped.

## Same inputs as the new Standard baseline

Reuse `data/real97_soft_eef_history5_20260910_v1` unchanged: 243 training
episodes from 15 ID environments. The four held-out environments and test
episodes remain excluded, including from action normalization statistics.

For anchor t, observed frames are t-12,t-9,t-6,t-3,t; 28 future frames are
t+3 through t+84. Choose t uniformly from 0 through max(0,N-85). Clamp
video and action indices identically at episode boundaries. Missing history
uses episode frame zero and its recorded target, not zero-valued actions.

Use recorded target EEF [x,y,z,qw,qx,qy,qz,gripper], canonicalized using
the saved training quaternion reference and causal sign continuity. Pack
eight normalized channels and six zero channels. Keep 320x160 letterbox,
70% coherent light augmentation with spatial gradients up to 0.08, no ROI.

Reuse the Standard history dataset wrapper and loss implementation. Five
observed frames produce two clean VAE conditioning latents; only the seven
future latent frames contribute to flow loss. Both C and model phases use
this loss. Random diffusion timestep/noise and scheduler weighting remain.

## Ours optimization, unchanged

- One random C32 vector per environment, initialized in [-1,1].
- Per GPU, sample five active environments and six distinct episodes per
  environment, then a window per episode: 30 samples/rank, 60/update.
- Activate 5+5+5 environments using the existing order and three 1000-step
  curriculum cycles, preceded by 300 model-only steps.
- Each curriculum cycle: 200 new-C steps (LR 0.15), 200 all-active-C steps
  (LR 0.03), 200 model steps, 200 all-active-C steps, 200 model steps.
- After step 3300, alternate 200 C-only and 200 model-only steps.
- Model LR 1e-5, initial 100-step warmup, weight decay 0.01. C weight
  decay zero. Do not reset Adam state or add phase displacement clipping.
- Stop at at most 5500 updates or the allocation time limit.
- Log every two steps; retain the existing fixed validation every 100 steps.

The independent entrypoint swaps only the dataset/model factories inside
its own process. The existing grouped trainer handles sampling, curriculum,
phase freezing, optimizers, validation, and paired checkpoint publication.
The legacy entrypoint remains the worker default. Shared-timestep bridge
and self-correction are rejected rather than silently using a one-frame loss.

## Launch and checkpoint safety

Use the existing worker with one config, overriding its default four-GPU
allocation to two GPUs:

```bash
sbatch --job-name=ours97-soft-h5 --gres=gpu:2 --constraint='141G|180G' \
  --cpus-per-task=32 --mem=250G --time=24:00:00 \
  --export=ALL,BWM_GROUPED_TRAIN_ENTRYPOINT=scripts/train_real97_soft_history5_grouped_context.py \
  scripts/run_real97_pair.sh \
  configs/train/train_real97_soft_eef_history5_c32_2gpu_20260910_v1.yaml
```

Allow H200/B200. Reuse node-local immutable Wan/BLM/data caches, and stage
new checkpoints locally before shared-disk publication. Preserve paired
model/C saves at model phase boundaries, the latest two ordinary pairs,
and the protected step-2300 pair. The existing wall-clock deadline is based
on allocation start: at 23h30, request a protected model/C snapshot at the
first safe optimizer boundary. It is not a mid-update asynchronous snapshot
and may start up to one update after the deadline.

Future Ours inference and support adaptation must use the same five observed
frames, saved action canonicalization/statistics, and future-only loss.
Use `use_history_condition_noise_in_inference=False`. Do not run these
checkpoints through the old one-initial-frame evaluation wrapper. Prediction
metrics must exclude the five known frames and any padded future frames.
