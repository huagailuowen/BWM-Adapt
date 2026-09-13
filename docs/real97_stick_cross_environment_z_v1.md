# Stick: fixed query, cross-environment Z swap

This is an additional Ours experiment, independent of the terminal-contact
metric reevaluation. It does not modify any previous inference or training.

## Frozen model and supports

Use the original Stick step5500 model and matching training-time Z table
from preparation job113867. Reuse all eight Stage2 Z values from job113868,
including their exact two train-only support identities and 40-step traces.
No adaptation is repeated, and no query GT is used to update Z or weights.
The model and Wan components use the existing node-local cache and locks.

## Query selection

Select eight existing factual queries: four training and four held-out test
episodes, one source query from each of the eight environments. Support
episodes remain excluded. Among feasible assignments, minimize the squared
distance of sorted grasp positions in each split to 0.25, 0.42, 0.58, 0.75.
This covers left, intermediate, and right lift positions as evenly as the
available queries permit. Position is the median support/grasp fraction in
the first five existing GT tracking records, not an outcome-success score.
The full candidate pool, selected indices, positions, and split labels are
saved. Video processing and selection run on the compute node.

## Controlled rollout

Each source chunk retains its original initial image and recorded target EEF
sequence: 41 frames, stride3, 192x256, training-normalized 14-channel action.
All 16 predictions for a query use the original base seed plus query index,
the same model, 50 denoising steps, and the same input chunk. Only Z changes.
Stage1 uses each environment's training-table entry. Stage2 uses that
environment's saved support-adapted Z. Inference light augmentation is off.

## Presentation and interpretation

Eight videos are generated. Each has three rows and eight columns. The first
row repeats the identical factual GT so each latent column is aligned with
it. The second row uses Stage1 Z; the third uses Stage2 Z. Columns are ordered
by the training-derived balance point, and the true source environment's
column has a black border. Raw videos stay under `raw/`.

There are 128 generated predictions. Off-environment Z swaps are
counterfactual interventions: the observed GT is not a paired counterfactual
ground truth and is not used to compute a formal error or success rate for
those columns. Completed predictions have independent markers for resuming
after low-priority preemption. All metadata and reused contexts are frozen
under the new output directory; old artifacts are not overwritten.

```bash
sbatch scripts/evaluation/run_real97_stick_cross_environment_z_gpu.sh
```
