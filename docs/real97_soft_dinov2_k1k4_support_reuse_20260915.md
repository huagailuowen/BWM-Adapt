# Soft DINO: random K=1..4 with four-query minimum

This opt-in variant retains the balanced-nine, train-only, static-start data
contract. The earlier K=1/2 configuration is unchanged. The common Real97
runner defaults to its historical disjoint K=1/2 behavior; only the new
`dinov2_query_topup_from_support: true` branch enables support reuse.

## Sampling per environment

Draw six distinct training episodes using the existing episode-first sampler,
then one legal static-start chunk from each episode. Uniformly choose K from
{1, 2, 3, 4} independently for each sampled environment and rank. Sample K
supports without replacement from those six chunks. Every non-support chunk
becomes a query. If fewer than four queries remain, sample additional queries
without replacement from the K selected supports.

| K | Disjoint queries | Supports reused as queries | Total queries | Distinct chunks |
| --- | --- | --- | --- | --- |
| 1 | 5 | 0 | 5 | 6 |
| 2 | 4 | 0 | 4 | 6 |
| 3 | 3 | 1 | 4 | 6 |
| 4 | 2 | 2 | 4 | 6 |

Reused queries intentionally overlap the conditioning supports. Their loss is
support reconstruction, not held-out-query generalization. They receive the
same per-query loss weight as the other queries. The runner retains its mean
over queries within each environment, then mean over sampled environments.
No held-out test episode is used for either role.

The data budget remains 5 environments x 6 distinct chunks per rank and two
ranks: 60 distinct chunks per optimizer update. `global_total_chunks` counts
support plus query role assignments and can exceed 60. New log fields
`global_unique_chunks` and `global_reused_support_queries` distinguish unique
data from repeated roles. The rank-zero sampled-episode log records each
reused sample ID. Checkpoint metadata stores the actual K choices and policy.

## Unchanged settings

- Original BWM initialization; 5500-update cap; two H200/B200 GPUs; 24 hours.
- One condition frame, 32 future frames, native stride 3, 320 x 160 letterbox.
- Target EEF actions and the same train-only normalization.
- Query and support lighting augmentation enabled, probability 0.7.
- Wan LR 1e-5, context-head LR 1e-4, model warmup 100 updates, no ROI.
- Periodic checkpoint every 500 updates, latest two ordinary bundles retained.
- Protected step2300 and final saves remain; the allocation-start + 23h30
  deadline saves at the next optimizer boundary, as in the existing launcher.
- TTT configuration and its queued task are not changed.

## Launch

```bash
sbatch --job-name=soft9-dino-k14-q4 --exclude=haic-hgx-5 \
  scripts/run_real97_soft_balanced9_dinov2_2gpu.sh \
  configs/train/train_real97_soft_static_balanced9_dinov2_concat_mlp_k1k4_qmin4_5x6_2gpu_20260915_v1.yaml
```

The existing launcher creates a distinct job-specific output directory and
freezes the submitted configuration and training manifest there. It reuses
node-local Wan, DINO, BWM, and dataset caches. The old pending K=1/2 task is
replaced, not resumed; it had no completed training updates.
