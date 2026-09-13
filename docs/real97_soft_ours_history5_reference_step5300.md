# Ours Soft step5300: real five-frame history

Use the same independently protected model/table pair as the static experiment:
outputs/reference_snapshots/soft_history5_114005_step5300_static_v1.
The snapshot name describes its origin, not a model restriction.

This opt-in configuration sets support_and_query_static=false. Existing static
configs retain their previous behavior. Results and adapted contexts use a
separate job-specific output; no static Z is imported.

## Support and query

Keep all original four train support episodes per environment AND their
original informative windows from soft_prep_job113869. These were selected
with two left-moving and two right-moving windows per environment. Preserve
their original annotations; this run does not redo object tracking.
All support actions use the model's frozen canonical target-EEF normalization.

Copy the exact 60-query manifest from Standard non-static job114479.
There are 15 training environments, each with two train and two held-out test
episodes. Query episodes never overlap the four support episodes.

For a window starting at native frame A, model-frame indices are:

    observed: A, A+3, A+6, A+9, A+12
    predicted: A+15, A+18, ..., A+96

The history anchor is A+12. These are the actual sampled video frames and
aligned actions, not repetitions of the first frame. Normal last-frame
padding remains supported at the episode/window end.

Use the training CanonicalTargetEEF and HistoryPrefixDataset operators.
The model consumes 33 frames at 320x160 and aligned 14-channel actions
(eight canonical target-EEF channels plus six zeros).

## Adaptation and comparison

Stage1 uses the step5300 table entry for the environment. Stage2 starts from
the mean step5300 table and freshly optimizes only Z using all four supports:
0.3 x10, 0.15 x10, 0.05 x10, 0.015 x10, FP32 context, no ROI or regularization.
Use the same history_prefix_flow_loss as training: two observed VAE latent
frames remain clean and seven future latent frames are scored.
The five observed decoded frames are conditions, not prediction targets.

Generate 50-step diffusion predictions with the same per-query seed as
Standard. Save GT/Stage1/Stage2 comparisons and four-row comparisons including
the existing Standard predictions. Both raw and comparison videos retain
all 33 frames, including the five real observed frames. Metrics must exclude
those observed frames and padded future timestamps using the query manifest.
Each environment has a four-column train/test grid and a training/inference
Z PCA plot is updated during inference.

Standard is step5500, Ours is step5300; do not label them equal-step checkpoints.
Metrics remain development-only until independent tracking calibration.

Launch one low-priority GPU, allowing H100/H200/B200. Reuse the locked node-local
Wan, checkpoint, and dataset caches. Automatic requeue resumes the same output
and skips completed context/variant markers. Manual resumes can pass the old
output as the first launcher argument.
