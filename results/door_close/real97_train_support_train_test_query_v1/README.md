# Door close: real97 reference evaluation

The user-selected Ours results are archived at:

`methods/ours/step_3715/seed_20260910/`

This is a copied qualitative reference, not a newly generated evaluation. The
source output and all training checkpoints remain unchanged. Computed image,
object-centric, and action-selection metrics for Standard, DINO, Ours Stage1,
and Ours Stage2 are now archived separately in
[`comparisons/metrics/consecutive_closure_pair_overlap_20260913_v1/`](comparisons/metrics/consecutive_closure_pair_overlap_20260913_v1/README.md).
This text-only snapshot preserves the source tracking audit status and known
GT/protocol discrepancies; it does not claim every held-out track was manually
audited. Videos are local artifacts and are not part of this metric publication.

## Videos and PCA

- `visualizations/grids/test/`: ten environment grids, one column per held-out query episode, sorted by numeric action level.
- `visualizations/grids/train/`: ten environment grids, two training-set query episodes per environment, sorted by numeric action level.
- In these factual grids, each column contains GT, Stage1, and Stage2 from top to bottom.
- `visualizations/grids/action_sweep/`: ten fixed-test-initial-frame action sweeps using recorded training action templates. These counterfactuals have no paired GT and are not formal GT-scored action-selection results.
- `visualizations/pca/training_inference_Z_pca.svg`: the archived training/inference Z PCA.
- `visualizations/pca/initialization_context_trajectory.svg`: the archived initialization and adaptation trajectories.

All paths above are relative to the Ours seed directory. Videos are independent
copies, not symbolic links. Raw per-query videos remain in the source output.

## Support is not the train grid

Both `train` and `test` grids show Query episodes. Support is the separate pair
of training episodes used to optimize Z, and is not separately rendered here.
The frozen Support manifest retains some inactive historical rows. The actual
active rows are selected by each environment's `support_indices`, recorded in
`protocol/execution_plan.json` and the per-environment context files.

For example, `door-12u-Half-11d-Half` adapts using level 8 / EP15 and level 10 /
EP19. Its train-query grid instead shows level 2 / EP3 and level 8 / EP45.

## Accepted per-environment settings

| Environment | Support levels | Z initialization |
| --- | --- | --- |
| door-0 | 2, 5 | Environment table Z + noise |
| door-2u-1d | 1, 3 | Environment table Z + noise |
| door-3u | 2, 5 | Environment table Z + noise |
| door-2u-4d | 3, 6 | Environment table Z + noise |
| door-4d | 3, 5 | Training-table mean, no noise |
| door-3u-4d | 4, 6 | Training-table mean, no noise |
| door-6u-7d | 6, 9 | Environment table Z + noise |
| door-6u-8d | 7, 10 | Environment table Z + noise |
| door-12u-Half | 8, 10 | Environment table Z + noise |
| door-12u-Half-11d-Half | 8, 10 | Environment table Z + noise |

Noise is independent per coordinate, Uniform(-0.5, 0.5). Exact initial vectors
and per-environment seeds are preserved in `initializations/`. This is a mixed
initialization reference, not an evaluation where every environment starts from
one shared unknown-environment Z. The eight environment-table initializations
use known environment identity.

All ten environments use unbounded FP32 Z adaptation, no context regularization,
no ROI weighting, and 40 inner steps with learning rates 0.3, 0.15, 0.05, and
0.015 for ten steps each. Model and training table are both step 3715. Factual
query chunks, target actions, GT, and Stage1 references are preserved.

## Reproducibility records

The shared `protocol/` directory freezes query, Support, action-template,
experiment, and execution-plan metadata. The method directory contains the
training config, action statistics, training Z table, adapted Z trajectories,
actual initialization records, and source provenance. Model weights are not
copied into results.

Source metadata is retained verbatim. Some original paths point to the source
output or a compute-node cache. The last partial rerun's global initialization
label describes only that rerun; consult `run_manifest.json` and the individual
context records for the actual mixed initialization settings. Original parser
bounds are also not the effective bounds: the unbounded wrapper disables them.

Source:
`/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/infer_real97_door_ours_reference_job114799`
