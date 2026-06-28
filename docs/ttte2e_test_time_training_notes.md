# TTT-E2E implementation notes

This note fixes the terminology around our two adaptation families.

## What the official TTT-E2E code does

The paper and code are for long-context language modeling:

- Paper: <https://test-time-training.github.io/e2e.pdf>
- Code: <https://github.com/test-time-training/e2e>

The official implementation is not the FastWAM `video_ttt` layer and not our
latent-C context path.

Key code-level pieces in the official repo:

- E2E experiment configs use `training.train_mode: meta`.
- `training.spec_inner` is
  `["language_model.**.suffix_blocks.feed_forward_prime.**"]`.
- `model.prime: true` creates `PrimeStorage`, which stores permanent
  `feed_forward_prime` MLPs and their norms.
- `BlockCollectionSplit` copies the normal Transformer into prefix/suffix
  blocks and inserts the prime MLPs into the suffix blocks.
- For each chunk, `MetaModel.inner_loop_step` computes the standard task loss
  and applies inner SGD updates only to the filtered inner parameters.
- `train_on_sequence` differentiates the sequence loss through the unrolled
  inner updates and applies an outer AdamW update to the trainable
  initialization.

So the adaptive state in TTT-E2E is a selected parameter subset. In the released
LM configs that subset is the prime FFN/MLP path in the suffix blocks, roughly
the last quarter of the Transformer.

## Our family 1: latent-C/context adaptation

Implemented by:

```text
wan_video_action/models/physical_context.py
scripts/train_stage2_ttt.py
scripts/infer_stage2_ttt.py
```

This is a separate architecture family:

- inner loop updates only task-local latent `C`;
- `C` is projected into context tokens and/or temporal modulation;
- outer loop trains `physical_context_encoder` and optionally residual adapter
  initialization;
- DiT, VAE, language/text, and action encoder are frozen in Stage 2.

This path is intentionally stable and cheap. It should stay intact.

## Our family 2: TTT-E2E mild adapter fast weights

Implemented by:

```text
configs/train/train_push_box_9mu_no_c_stage1_ttte2e_10k_4gpu.yaml
scripts/run_stage1_ttte2e_9mu_10k_4gpu_guard.sh
scripts/train_stage2_ttte2e.py
scripts/infer_stage2_ttte2e.py
configs/train/train_push_box_9mu_ttte2e_mild_stage2_10k_4gpu.yaml
```

This is the BWM analogue of TTT-E2E:

- Stage1-B first trains a no-C action-conditioned source model on the large
  9-friction dataset;
- Stage2-B must initialize from the Stage1-B no-C checkpoint, not from the
  medium-C Stage1 checkpoint;
- latent `C` is disabled (`physical_context_mode: none`);
- the pretrained DiT/vae/text/action backbone remains frozen;
- low-rank residual adapters are attached to late DiT blocks;
- default layer selection is `physical_adapter_layers: "last_quarter"`;
- inner loop updates selected adapter fast weights using the video
  flow-matching task loss;
- outer loop trains the adapter initialization on held-out same-friction query
  chunks after support adaptation.

Default inner parameter scope:

```text
ttte2e_inner_param_scope = lowrank
```

which selects:

```text
*.gate
*.down.weight
*.up.weight
```

The default is first-order MAML-style training:

```text
theta_fast <- theta_fast - lr * stopgrad(d L_support / d theta_fast)
```

This preserves the useful meta-learning path from query loss to `theta0` while
avoiding second-order Hessian memory. `--ttte2e_second_order` is available for a
closer but much more expensive variant.

## Why this is milder than the paper

The official paper updates real suffix-block MLP weights. We do not start there
because the BWM backbone is a 5B video diffusion model and support sets contain
only 1-3 trajectories.

The mild version reduces risk in three places:

- update adapter fast weights instead of pretrained DiT FFN weights;
- update only low-rank gate/down/up tensors by default;
- use first-order meta-gradients by default.

This still preserves the defining TTT-E2E structure: task-loss gradient updates
to model parameters at test time, with the initialization trained by
support-to-query meta-learning.

## Required training chain

Do not use the medium-C Stage1 checkpoint for TTT-E2E evaluation except for code
smoke tests. That checkpoint was trained with C tokens/modulation, so disabling
C at Stage2-B introduces a train/test mismatch.

The main TTT-E2E chain is:

```text
Stage1-B no-C source SFT:
  configs/train/train_push_box_9mu_no_c_stage1_ttte2e_10k_4gpu.yaml
  -> outputs/push_box_9mu_no_c_stage1_ttte2e_10k_4gpu/step-10125.safetensors

Stage2-B TTT-E2E mild meta-training:
  configs/train/train_push_box_9mu_ttte2e_mild_stage2_10k_4gpu.yaml
  ckpt_path: outputs/push_box_9mu_no_c_stage1_ttte2e_10k_4gpu/step-10125.safetensors
```

The dataset is the large 9-friction push-box set:

```text
FastWAM/data/libero_push_box_friction_9mu_450
data/push_box_bwm_friction_9mu_450/
```

## Loss

For each same-property group, sample support rows and disjoint query rows.

Inner loss:

```text
L_inner = mean_i L_flow(support_i; theta_fast)
        + lambda_inner_reg * mean_j ||theta_fast_j - theta0_j||^2
```

Outer loss:

```text
L_outer =
  1.0    * L_query_adapted
+ 0.1    * L_show_adapted
+ 0.2    * L_gap
+ 0.0001 * L_adapter_reg
```

Definitions:

```text
L_query_adapted = mean flow-matching loss on held-out same-property query chunks using theta_fast
L_show_adapted  = mean flow-matching loss on support chunks using theta_fast
L_adapter_reg   = mean_j mean((theta_fast_j - theta0_j)^2)

rel_show_imp  = (L_show_base - stopgrad(L_show_adapted)) / (abs(L_show_base) + 1e-4)
rel_query_imp = (L_query_base - L_query_adapted) / (abs(L_query_base) + 1e-4)
L_gap         = relu(stopgrad(rel_show_imp) - rel_query_imp - 0.03)^2
```

The gap term is the support-memorization guard: the shown support trajectories
are allowed to improve, but their improvement should not greatly exceed the
improvement on held-out trajectories from the same friction group.

## Leakage control

The dataset sampler avoids using the same episode for support and query when a
group has enough episodes. Query loss is computed on held-out same-friction
chunks. The query target is never used inside the inner loop.
