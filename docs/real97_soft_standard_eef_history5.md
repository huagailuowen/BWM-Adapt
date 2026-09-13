# Soft pull: target EEF with five observed frames

This is an opt-in Standard pooled baseline, not an Ours/context-table run.
Old configs retain their original dataset, action loader, and loss function.

Config: `configs/train/train_real97_soft_eef_history5_standard_pooled_2gpu_20260910_v1.yaml`.

## Data and actions

Use the unchanged approved episode-level split from
`datasets_real/soft_pull_9_7_crop_512x256`: 15 ID environments, 243 train
episodes. No test episodes or the four held-out environments enter training
or normalization statistics. Preserve even short episodes through padding.

The dataset's `action` field explicitly represents target base-frame EEF
`[x,y,z,qw,qx,qy,qz,gripper_width]`. It is not measured EEF state and not the
older joint-target action. Normalize quaternion magnitude, align the first
sign to a fixed reference derived from the training set, then preserve
temporal sign continuity within each episode. Store that reference and
training-only min/max statistics with the input manifest. Pack the eight
normalized target channels first, followed by six zeros. Constant channels
are zero after normalization. Never normalize separately per chunk.

## Window contract

Choose environment, then distinct training episodes, then a uniform anchor t
within each episode. For N native frames, anchors are 0 through max(0,N-85).
The 33 native indices before boundary clamping are:

```
history: t-12, t-9, t-6, t-3, t
future:  t+3, t+6, ..., t+84
```

Clamp indices to [0,N-1] identically for video and action. Missing history
copies episode frame zero and its target action, not an all-zero action.
At t=0, all five history frames equal frame zero. The current/future window
spans 85 native frames; with available history, the complete input spans 97.
At 20 Hz native rate and stride three, history spans 0.6 s and prediction
spans 4.2 s. No optical-flow interpolation or temporal resampling is used.

Keep 320x160 letterbox, 70% coherent light augmentation, and no ROI. One
augmentation transform applies to both historical and future frames.

## Model and loss

Total video length remains 33. Five historical video frames encode into two
VAE latent frames. The opt-in flow loss fixes those two clean conditioning
latents and supervises only the remaining seven latent frames, corresponding
to the 28 future video frames. Random diffusion timestep/noise sampling and
scheduler weighting are unchanged. Legacy single-frame loss is untouched.

Future inference must pass five actual historical frames, these same action
indices and saved EEF normalization, and
`use_history_condition_noise_in_inference=False`. Do not reuse a first-frame
only evaluation wrapper or inject history noise not used by this training.
Exclude all five known frames, and boundary-padded future frames, from video
prediction metrics. A dedicated matching evaluation manifest/entrypoint is
still required before evaluating this new checkpoint.

## Optimizer, resources, and checkpoints

Two GPUs per Soft trainer, five environments times six episodes/chunks per
rank, 60 samples per distributed optimizer update using sequential local
accumulation. Standard training updates DiT/action encoder on every update;
there is no environment Z, curriculum, or alternating C-only phase.

Start from `ckpt/BLM/step-12000.safetensors`, not a prior joint-action
Standard checkpoint. LR 1e-5, 100-step warmup, AdamW decay 0.01, max 5500
updates. Reuse the existing isolated two-rank worker in the four-GPU
Stick/Soft bundle, allowing H200 or B200. Stick's config is unchanged.

Reuse immutable Wan/BLM and staged dataset files in the node-local cache.
Use a new output directory; never delete old checkpoints. Ordinary saves
are every 500 updates with only the latest two retained. Step 2300 and final
saves are protected. The existing Standard worker checks the deadline from
the allocation start (23 h 30 min) at optimizer boundaries and publishes a
protected checkpoint, then stops. This is a safe-boundary trigger, not an
asynchronous mid-update snapshot; it can occur up to one update after the
deadline. Standard has no C table to save.
