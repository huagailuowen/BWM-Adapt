# Standard Stick and Soft: frozen reference queries

Configuration: `configs/evaluation/real97_standard_reference_20260911.json`.
Both models use job 113942's protected step-5500 checkpoint. No support
adaptation, latent table, ROI weighting, or evaluation-time augmentation.
Old training and inference entry points/configurations remain unchanged.

## Frozen population and timing

- Stick: the exact 40 queries (16 train, 24 test) across eight environments
  from `stick_prep_job113867`; 41 frames, stride three, one observed frame.
- Soft: the exact 60 queries (30 train, 30 test) across fifteen environments
  from `soft_prep_job113869`; 33 frames, stride three, five observed frames.
- Keep episode membership, ordering, start/end and every source timestamp.
  There is no fresh query or action selection. Source support manifests are
  archived for provenance but never used by the pooled model.
- Example Soft query: native frames 27,30,...,123. Observe 27,30,33,36,39;
  predict 42,45,...,123. The training history anchor is 39, not 27. This is a
  causal five-frame training window without moving the previous test chunk.
- Video and action use identical source indices and last-frame padding.
  The existing training `HistoryPrefixDataset` and `CanonicalTargetEEF`
  operators are reused for Soft. Target quaternion sign/scale and all action
  normalization come from that run's frozen training-only action statistics.
- Soft inputs contain only the five observed images, never query-future GT.
  All 33 target actions are legitimate controls. The two corresponding VAE
  latent frames stay clean throughout denoising, matching history SFT.
- Observed prefixes and padded tails must be excluded from prediction
  metrics. Per-query `evaluation_frame_indices` records eligible frames.
- Soft versus old one-frame/joint-target ours is NOT a fully matched-input
  comparison: history and action representation differ. Use common future
  timestamps and explicitly report those differences.

## Outputs and scheduling

Each independent low-priority single-GPU job accepts H100/H200/B200. It stages
Wan, the immutable checkpoint and selected data on node-local storage using
the existing locked shared cache. It never deletes other caches/checkpoints.
Completed queries are atomic markers, allowing preemption/requeue to skip
finished generation. Each environment receives a two-row GT/Standard grid,
with one column per original query; per-query comparisons and raw videos are
also retained. Playback is exactly 20/3 fps.

These outputs remain under `outputs/` while object-centric/contact metrics
are being calibrated independently on CPU nodes. Do not publish unreviewed
Stick success labels or inferred method rankings in formal `results/`.

```bash
sbatch --job-name=std97-stick-infer scripts/evaluation/run_real97_standard_reference_gpu.sh stick
sbatch --job-name=std97-soft-h5-infer scripts/evaluation/run_real97_standard_reference_gpu.sh soft
```
