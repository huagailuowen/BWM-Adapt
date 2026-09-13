# Soft static: balanced nine environments, two independent C32 rows each

Dataset: `datasets_real/soft_pull_9_7_crop_512x256_static`.
The authoritative selection is `TRAINING_ENVIRONMENT_SELECTION.json`, together
with `ENVIRONMENT_SPLIT.json` and the preserved episode `TRAIN_TEST_SPLIT.json`.

| Category | Training environments |
| --- | --- |
| Left | soft-1l, soft-2l, soft-4l |
| Right | soft-1r, soft-7r, soft-5r |
| Center | soft-2m, soft-6m, soft-8 |

The other ten environments are UNASSIGNED, not automatically OOD test data.
Exclude LeRobot `soft-6m/episode_000010` (source episode 11) entirely. There are
146 eligible training episodes with 2680 legal windows, and 18 held-out episodes
with 277 legal windows. Empty annotations never fall back to arbitrary starts.

Each physical environment has two separately initialized Uniform(-1,1) C32 table
rows, sharing its whole eligible training pool. Virtual ID = 2*physical ID +
replica. Physical IDs follow the explicit selection list, not a inferred physical
coefficient. Expanded training metadata has 5360 rows but no duplicated videos.
`latent_aliases.json` records the mapping; test metadata retains physical IDs and
evaluation must explicitly select a replica.

## Same single-frame model and batch

One conditioning frame plus 32 predicted frames. Native indices are
`s, s+3, ..., s+96`, exclusively from approved `valid_33x3_starts`. The preceding
eight frames and following three frames used for start annotation are NOT extra
conditioning inputs. Video and target EEF use identical native indices.

Target EEF remains eight canonicalized channels plus six zeros. Min/max statistics
are refitted using only the 146 eligible TRAIN episodes, never held-out episodes,
unassigned environments, or the disabled episode. The quaternion convention is
unchanged: unit norm, fixed training reference hemisphere, causal sign continuity.

Two H200/B200 GPUs; each rank has 5 latent slots x 6 distinct episodes per slot,
30 microbatches per update, 60 chunks globally. With at least five eligible latent
groups, the legacy sampler is unchanged. In a smaller allowed pool, sample group
slots WITH replacement; each slot still independently draws six distinct episodes
and then one uniformly sampled approved window from each. Duplicate physical
episodes across repeated slots/replicas are allowed. Never expand the allowed
pool from new groups to old groups just to fill the batch.

Keep 320x160 letterboxed images, 70% temporally coherent lighting augmentation
with gradients in [-0.08,0.08], no ROI, original model LR 1e-5 and model weight
decay 0.01. Start from `ckpt/BLM/step-12000.safetensors`, not a previous Ours model.

## Curriculum

Steps 1-300 train only the model; retain the initial 100-step model LR warmup.
Four injection rounds add 5, 5, 5, then 3 NEW virtual groups. Each 1000-step round
has new-C 200, all-active-C 200, model 200, all-active-C 200, model 200. All C-only
phases, including new-C, use LR 0.03. Freeze the other parameter family as before.
The fourth new-C phase samples only its three new groups, with replacement.

Rounds end at steps 1300, 2300, 3300, 4300. Steps 4301-5500 alternate all-active
C 200 and model 200. The existing nested-uniform virtual-group ordering is frozen
in `dual_latent_curriculum.json`.

## Launch and safety

New preparer, training entrypoint, config and launcher are opt-in. No legacy code
or configuration is changed. Preparation must run on a compute allocation because
it reads training Parquet files for action statistics. It publishes frozen input
manifests atomically and refuses mismatched contracts/counts.

`scripts/run_real97_soft_static_balanced9_dual_2gpu.sh` uses the existing locked
node-local WAN/BLM caches and a dataset cache keyed by the new physical manifest.
Each job gets its own output and frozen input manifest. Retain the latest two
model/context pairs, plus the independently protected step-2300 pair. The existing
deadline is allocation StartTime + 84600 seconds: save a matching model/context
pair at the first completed-update boundary at/after 23h30, then stop. This is an
update-boundary guarantee, not interruption of an in-flight forward/backward.
