# Dual Ball / Door: 500 model-only continuation updates

Source runs: Door job115191 and Ball job115192, complete model/Z pairs at step5500.
New independent two-GPU high-priority H200/B200 jobs, each up to24h, steps5501..6000.
Never delete, overwrite or prune the source output directories.

- Restore DiT/action/context-encoder weights and the matching full dual Z table.
- Freeze every latent row for all500 updates; no reinitialization, clamping,
  curriculum additions, context update, or displacement projection.
- DiT, action encoder and physical-context encoder retain their original
  trainable settings. VAE and unused modules remain frozen.
- This is weight continuation with fresh AdamW, not exact optimizer-state resume.
- Model LR1e-5, weight decay0.01, no repeated warmup at global step5500.
- Each rank samples6 virtual latent groups, then10 distinct train episodes/group,
  one original-rule chunk each; dual aliases may refer to the same physical env.
  Two ranks give120 chunks/update. Microbatch1 with60 accumulation forwards/rank.
- Original manifests, train-only normalization, video/action representation,
  preferred-window probabilities and70% light augmentation unchanged. No ROI.
- Save every100 updates; retain only the newest ordinary model/Z pair in this
  new run. No protected2300 copy or source-model duplication in the new output.
- The restored Z table is also recorded as tiny input provenance; phase-end
  context records are metadata rather than additional model checkpoints.
- Reuse node-local immutable Wan/data caches; copy step5500 weights to a locked
  node-local resume cache. Actual allocation+23h30 safe checkpoint inherited.

