# Ours Soft: static five-frame history, step5300 model and Z table

The model and table are both from job114005, step5300. A separate snapshot
holds the model by hard link and copies the table, training configuration,
training-only action statistics and checkpoint publication marker. Training
checkpoint pruning cannot remove this reference. Node-local Wan and dataset
caches are reused; the new model has a distinct locked checkpoint cache key.

## Static inputs and support identity

The sixty query manifests are copied exactly from Standard static-start
job114497: fifteen environments, thirty training and thirty held-out queries.
The native sequence is `[0,0,0,0,0,3,6,...,84]`, with short tails clamped.
Video and canonical recorded target-EEF action are padded identically.
Model inputs are 33 frames at 160x320; the first five are observed and the
remaining 28 are predicted. Comparison videos remove four duplicate initial
frames and therefore contain 29 displayed frames.

Support also uses static episode starts. The same four train-only support
episode IDs per environment are retained, but their old intermediate windows
are replaced by the static prefix. Old left/right labels describe the old
windows and are preserved as provenance, not re-certified for the new prefix.
No query episode enters adaptation. This is a changed-support-window ablation,
not a claim that every conditioning detail matches the old one-frame trial.

## Training-consistent Z adaptation and sampling

Stage1 uses the matching step5300 training-table Z. Stage2 starts from the mean
of all fifteen step5300 table entries and adapts only FP32 Z using four support
episodes and the existing 40-step schedule `0.3:10,0.15:10,0.05:10,0.015:10`.
Old one-frame-model adapted Z values are not reused.

The Stage2 support loss is the actual training `history_prefix_flow_loss`:
two observed VAE latent frames remain clean, and only the seven future latent
frames contribute to flow-matching loss. An instance-local branch replaces
the legacy TTT loss only for this new entrypoint. Old files and configs are
unchanged. The clean-history Standard sampler is reused through a pipe proxy
that injects Z; observed latents remain clean at every denoising step.

## Artifacts and comparison limits

Each environment gets GT/Stage1/Stage2 grids and a separate four-row version
including the existing Standard prediction. Standard is step5500, whereas
this Ours run is step5300, as explicitly requested; do not claim equal training
steps. GT and Standard videos are reused, with the same query actions and
per-query seed. Every context, trajectory, manifest and completed prediction
is saved independently for low-priority resume. PCA is fitted on the frozen
training table. No formal action or object-centric score is published here.
