# Soft DINO K=4: matched static-start inference

This separate inference entrypoint preserves all prior code, configurations,
videos, and query selection. Submit after successful completion of training
117150; only its protected final step5500 combined Wan/DINO checkpoint is used.
The completion marker and final metadata are required. No silent fallback to
step5000 or a different run is allowed.

## Protocol

- Same nine environments and four informative training Supports per environment
  as the accepted Ours family-mean evaluation; 36 Supports in total.
- Exactly the same 36 queries: 18 train and 18 held-out test episodes, all
  starting at native frame zero. Query and Support episodes remain disjoint.
- DINO encodes all four Support videos and their recorded actions once per
  environment. Its trained concat-MLP produces one code reused for every query
  in that environment. No query future enters the encoder; no gradient-based
  inference adaptation or environment-table initialization is applied.
- One observed frame plus 32 predictions, stride3, 320x160 letterbox. Target EEF
  uses the training `CanonicalTargetEEF` operator and frozen action statistics:
  eight canonical coordinates plus six zeros. Video/action frame indices match.
- No inference lighting augmentation or ROI. Use 50 denoising steps and the
  same base seed20260910 plus query index as the reference runs.
- Load Wan, DINO, model checkpoints, and data through node-local shared caches.
  Never load the large model directly from the shared disk when generating.

## Artifacts

Output: `outputs/infer_real97_soft_dino_k4_step5500_train117150_static_reference_20260916_v1`.

- `raw/gt` and `raw/dino`: 33-frame, unlabeled videos.
- `contexts`: four-support DINO codes and exact Support IDs.
- `comparisons` and `grids/{train,test}`: GT / DINO, full33frames.
- `comparisons_4way` and `grids_4way/{train,test}`: GT / Standard / final Ours /
  DINO, reusing previous Standard and Ours videos without altering them.
- `input_manifest`, `plan.json`, `provenance.json`: frozen inputs, model metadata,
  preprocessing, split membership, and comparison alignment.
- Per-query completion markers allow recovery without regenerating completed
  raw predictions. A missing/corrupt artifact fails visibly rather than silently
  dropping an environment.

The four-way grid shows native frames0,3,...,84: 29displayframes. Standard's
five-frame padded input requires output indices4..32; GT/Ours/DINO use0..28.
The four-way grid therefore has the same 28 future frames as the existing
Soft score table. Raw DINO predictions retain all32futureframes through96.
No side panels, action-selection sweeps, or action metric are generated.
Image and object metrics are not computed by this inference-only job.

Standard did not train on soft-4l, soft-7r, or soft-6m; this existing limitation
still applies to the reused four-way comparison. DINO and Ours trained on all
nine. Ours uses its accepted known-family initialization, whereas DINO derives
its code from the four Supports only.

## Submit

```bash
sbatch --job-name=soft9-dino-k4-infer --exclude=haic-hgx-5 \
  --dependency=afterok:117150 --kill-on-invalid-dep=yes \
  scripts/evaluation/run_real97_soft_dino_static_reference_gpu.sh
```

The job requests one low-priority H100/H200/B200 GPU for up to12hours. It does
not occupy an inference GPU while waiting on the training dependency.
