# Real97 Door/Ball: two independent latents per physical environment

This experiment starts from `ckpt/BLM/step-12000.safetensors`, not from an Ours
checkpoint. Each physical environment has two independent 32-dimensional table
rows, initialized coordinate-wise from Uniform(-1, 1). Both rows can sample all
of that environment's original training episodes and chunks. They do not split
the data, share parameter storage, or have a latent-consistency penalty.

Only training metadata is expanded. Original videos, target actions, selected
windows, train/test episode splits, action statistics, and inference geometry
remain unchanged. `friction_mu` in the expanded metadata is a virtual group ID,
not an oracle friction coefficient. The original identity is retained in
`physical_environment`, `physical_environment_index`, and `source_friction_mu`.
Virtual group ID = 2 * physical environment index + latent replica (0 or 1).

| Task | Physical environments | Latent groups | Groups added per round |
| --- | ---: | ---: | ---: |
| Door | 10 | 20 | 5 |
| Ball | 8 | 16 | 4 |

The trainer's existing nested-uniform curriculum ordering is applied to virtual
IDs. The exact order and physical mapping are recorded in each input manifest's
`dual_latent_curriculum.json` and `latent_aliases.json`.

## Schedule

- Steps 1-300: freeze the initial latent rows; train the model, with its original
  100-step learning-rate warmup to 1e-5.
- Round 1: steps 301-1300.
- Round 2: steps 1301-2300.
- Round 3: steps 2301-3300.
- Round 4: steps 3301-4300.
- Steps 4301-5500: alternate 200 latent-only steps and 200 model-only steps.

Each 1000-step round retains the original five 200-step phases:
new-C only, all-active-C only, model only, all-active-C only, model only.
Every C phase uses LR 0.03, including the newly added groups' first 200 steps.
The new-C phase samples only newly added virtual groups. Inactive rows and the
frozen parameter family are not updated. Model LR remains 1e-5.

Each experiment uses two GPUs; each GPU samples three virtual groups and eight
episodes/chunks per group. Two aliases of the same physical environment may
co-occur because they are independent training groups. Total per-rank volume
remains 24 chunks per optimizer update.

The old 70% temporally coherent lighting augmentation remains enabled, with
spatial gradients in [-0.08, 0.08]. ROI weighting remains disabled. Door uses
49 frames at 256x192 and exported target joints; Ball uses 61 frames at 320x160
and target joints. Both use stride 1 and their original action alignment.

## Checkpoints and compatibility

New configurations and output directories keep the previous experiments intact.
The existing atomic paired model/context-table saving and latest-two retention
remain in force, including the protected step-2300 pair. The allocation-start
plus 23h30 checkpoint trigger is unchanged. The frozen `input_manifest` stores
the mapping needed to interpret every table row in saved checkpoints.

No shared training core is changed. The launcher has an optional metadata-only
cache identity override so the two aliases reuse the original physical dataset
cache instead of making additional video copies. Legacy manifests retain the
same cache identity and behavior as before.

The held-out manifests retain physical IDs: future evaluation must explicitly
choose the corresponding replica using `latent_aliases.json`, rather than using
a physical ID as a virtual-table ID.
