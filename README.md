# BWM-Adapt: Physical Property Adaptation

This fork adapts Boundless World Model (BWM) into an action-conditioned world-model testbed for physical-property adaptation. The upstream README is preserved as `README.official.md`.

The immediate research question is:

Can an action-conditioned video world model infer a latent physical property from a small support rollout, then improve prediction on other trajectories that share the same property?

The first target domain is robotic pushing on tables with different friction coefficients. The model should learn an environment-level property such as friction, not memorize one support trajectory or one initial object pose.

## Current Backbone

We use the upstream BWM action-conditioned Wan2.2-TI2V-5B pipeline.

- Vision/action world model: `Wan2.2-TI2V-5B` plus BWM action encoder.
- Action path A: action tokens are appended to the cross-attention context.
- Action path B: grouped actions are projected into temporal modulation and added to the timestep/AdaLN stream.
- Training loss: diffusion flow-matching SFT loss from the BWM/DiffSynth training loop.

This is a suitable starting point because it already predicts future video conditioned on robot actions.

## Adaptation Hypothesis

For a support trajectory `A` under physical property `mu`, test-time adaptation should produce a latent state or small parameter delta that improves prediction for many held-out trajectories under the same `mu`.

The desired objective is not only:

```text
fit support trajectory A
```

It should approximate:

```text
adapt on support trajectory A
improve query trajectories B, C, D with the same physical property
avoid improvements that are isolated to A only
separate properties such as low-friction and high-friction tables
```

This means our meta-training batches should be grouped by physical property. Each group should contain multiple trajectories with varied initial positions, push directions, and push speeds.

## Architecture Variants

### 1. Latent C Token Injection

Implemented in `wan_video_action/models/physical_context.py`.

`PhysicalContextEncoder` maps a latent physical code `C` into the existing BWM conditioning channels:

- `physical_context_emb`: one or more tokens appended to the cross-attention context.
- `physical_mod_emb`: optional temporal-group modulation added to the timestep/AdaLN stream.

Config knobs:

```yaml
physical_context:
  physical_context_mode: "token"      # none | token | modulation | both
  physical_context_dim: 128
  physical_context_tokens: 1
  physical_context_hidden_dim: 0
```

The default `C` starts at zero and the projection MLPs have no bias, so enabling the module is initially close to the base model. Gradients still flow into the default context and projectors.

Useful experiments:

- Small `C`: 16 to 64 dims.
- Medium `C`: 128 to 256 dims.
- Multiple tokens: 1 to 5 tokens.
- Token-only versus token plus modulation.

### 2. TTT-E2E Mild Parameter Adaptation

Implemented as `PhysicalResidualAdapterBank`.

This is the second architecture family. It should not use latent `C` as the
main adaptive state. The test-time state is a small set of model parameters.

The official TTT-E2E implementation uses `train_mode: meta`, splits the
Transformer into prefix and suffix blocks, stores permanent prime FFN parameters
in `PrimeStorage`, then updates `language_model.**.suffix_blocks.feed_forward_prime.**`
with the same next-token task loss on each chunk. The outer optimizer trains the
initialization through those unrolled inner updates. Its released E2E configs
set `model.prime: true` and choose a suffix length of roughly the last quarter
of blocks.

Our BWM version is deliberately milder so it can train on the 5B video model:

- keep the pretrained DiT, VAE, text/language path, and action encoder frozen;
- attach low-rank MLP-side residual adapters to late DiT blocks;
- default to `physical_adapter_layers: "last_quarter"`;
- inner loop updates only adapter fast weights with video flow-matching loss;
- outer loop trains the adapter initialization on held-out same-friction query
  chunks after the support update;
- first-order meta-gradients are the default; second-order is optional and
  expected to be much more memory hungry.

Config knobs:

```yaml
physical_context:
  physical_adapter_mode: "residual"   # none | residual
  physical_adapter_rank: 16
  physical_adapter_layers: "last_quarter"
  physical_adapter_gate_init: 0.0
```

The gate starts at zero, so the base world model behavior is unchanged before
training. The layer selector supports `all`, `uniform:N`, `last_quarter`,
`last:N`, or an explicit comma-separated layer list.

## Training Plan

### Stage 0: Base World-Model Finetuning

Goal: make BWM accurately model the in-domain pushing distribution before testing adaptation.

Data:

- Fixed nominal friction.
- Many object poses, target positions, push directions, and push speeds.
- Action-conditioned video chunks.

Trainable modules:

```text
dit, action_encoder
```

This stage should not use physical context or adapters.

### Stage 1: Adaptation Module SFT

Goal: attach the adaptation mechanism without destabilizing the base model.

Two branches:

- Token branch: freeze BWM, train `physical_context_encoder`.
- Adapter branch: freeze BWM, train `physical_adapter_bank`.

Templates:

- `configs/train/train_physical_context_token.yaml`
- `configs/train/train_physical_adapter_residual.yaml`

### Stage 2: Property-Level Meta-Adaptation

Goal: adapt on one support trajectory and improve other trajectories with the same property.

Batch layout:

```text
property group mu_i:
  support trajectory A
  query trajectories B, C, D
property group mu_j:
  negative/query trajectories E, F
```

Loss terms:

```text
L_support: prediction loss on the adapted support trajectory
L_query_same_property: prediction loss on held-out trajectories with the same property
L_specificity: penalty when support improves much more than same-property queries
L_property_separation: optional contrastive term between different property groups
```

The key metric is whether adapting from `A` improves `B/C/D`, not just `A`.

## Implemented Push-Box Workflow

The original small push-box experiments used:

```text
FastWAM/data/libero_push_box_calibrated_v2_100pairs
```

The current large push-box experiments use the 9-friction dataset:

```text
FastWAM/data/libero_push_box_friction_9mu_450
```

The prepared BWM manifests live under:

```text
data/push_box_bwm_calibrated_v2_100pairs/
  train.jsonl
  test.jsonl
  action_stats.json

data/push_box_bwm_friction_9mu_450/
  train.jsonl      # 900 chunks
  test.jsonl       # 450 chunks
  action_stats.json
```

Rows are grouped by:

```text
source_split, friction_mu
```

where `source_split` is `straight` or `angled`, and `friction_mu` is one of the
friction bins. The 9-friction manifest has 18 groups:
`straight/angled x 9 friction values`, with roughly 50 trajectories per setting.
Push-focused chunks are sampled around frame starts 50 to 75, with short
end-of-episode chunks padded by repeating the last valid frame/action.

### Stage 1A: Latent-C Video Finetuning

Stage 1A adapts the base action-conditioned BWM to the push-box video
distribution for the latent-C branch. It includes `physical_context_encoder`
and therefore should not be reused as the base checkpoint for the no-C
TTT-E2E branch.

Config:

```text
configs/train/train_push_box_medium_c_stage1_10k.yaml
```

The in-domain checkpoint used by later experiments is:

```text
outputs/push_box_medium_c_stage1_10k_4gpu/step-10000.safetensors
```

This file is an experiment artifact and is ignored by git.

### Stage 1B: No-C Video Finetuning For TTT-E2E

Stage 1B is the required source checkpoint for the TTT-E2E mild branch. It uses
the large 9-friction dataset and explicitly disables both latent `C` and
residual adapters:

```text
physical_context_mode: none
physical_adapter_mode: none
trainable_models: dit,action_encoder
```

Config:

```text
configs/train/train_push_box_9mu_no_c_stage1_ttte2e_10k_4gpu.yaml
```

Guard script:

```text
scripts/run_stage1_ttte2e_9mu_10k_4gpu_guard.sh
```

Expected checkpoint:

```text
outputs/push_box_9mu_no_c_stage1_ttte2e_10k_4gpu/step-10125.safetensors
```

Stage 2B defaults to this checkpoint. Reusing the old medium-C Stage1 checkpoint
for Stage 2B is only acceptable for code smoke tests, not for the main
comparison, because that checkpoint was trained with C tokens/modulation.

### Stage 2A: Latent-C First-Order TTT Meta-Training

Stage 2A is the first architecture family. It trains a medium-dimensional latent
physical code `C` plus C-conditioned low-rank residual adapters. The inner loop
updates the task-local `C`; it does not update model parameters.

Main script:

```text
scripts/train_stage2_ttt.py
```

Configs:

```text
configs/train/train_push_box_medium_c_stage2_ttt.yaml
configs/train/train_push_box_medium_c_stage2_ttt_10k_2gpu.yaml
configs/train/train_push_box_9mu_medium_c_stage2_ttt_10k_4gpu.yaml
```

Trainable modules:

```text
physical_context_encoder
physical_adapter_bank
```

Frozen modules:

```text
DiT backbone
VAE
text/language path
action encoder
```

For each meta-task, support and query chunks come from the same `source_split,friction_mu` group. The inner loop starts from `physical_context_encoder.default_context` and updates only the task-local `C` for 5 gradient steps:

```text
C <- C - 0.05 * clip_grad(d L_inner / d C, max_norm=1.0)
```

The outer loop is first-order and updates only the context encoder and residual adapter bank:

```text
L_outer =
  1.0   * L_query_adapted
+ 0.1   * L_show_adapted
+ 0.2   * L_gap
+ 0.001 * L_context_reg
```

Definitions:

```text
L_query_adapted = mean flow-matching loss on held-out same-property query chunks using adapted C
L_show_adapted  = mean flow-matching loss on support chunks using adapted C
L_context_reg   = mean((C_adapted - C0)^2)

rel_show_imp  = (L_show_base - stopgrad(L_show_adapted)) / (abs(L_show_base) + 1e-4)
rel_query_imp = (L_query_base - L_query_adapted) / (abs(L_query_base) + 1e-4)
L_gap         = relu(stopgrad(rel_show_imp) - rel_query_imp - 0.03)^2
```

The gap term penalizes cases where the support trajectory improves much more than the held-out same-friction query trajectories. This is meant to discourage support-only memorization.

Guard scripts:

```text
scripts/run_stage2_ttt_guard.sh
scripts/run_stage2_ttt_10k_gpu01_guard.sh
```

These scripts run long jobs in `tmux`-friendly shells and restore GPU holder sessions for GPUs 2 and 3 when the job exits.

### Stage 2B: TTT-E2E Mild Meta-Training

Stage 2B is the second architecture family. It is closer to the official
TTT-E2E idea because the inner loop updates parameters with the task loss. It
does not use latent `C`.

Main script:

```text
scripts/train_stage2_ttte2e.py
```

Inference script:

```text
scripts/infer_stage2_ttte2e.py
```

Default config:

```text
configs/train/train_push_box_9mu_ttte2e_mild_stage2_10k_4gpu.yaml
```

Required source checkpoint:

```text
outputs/push_box_9mu_no_c_stage1_ttte2e_10k_4gpu/step-10125.safetensors
```

Trainable modules:

```text
physical_adapter_bank
```

Frozen modules:

```text
DiT backbone
VAE
text/language path
action encoder
physical_context_encoder is disabled
```

Inner-loop fast weights:

```text
physical_adapter_bank late-block gate/down/up parameters
```

For each support/query meta-task, the script starts from the current adapter
initialization `theta0` and performs 5 first-order inner SGD steps:

```text
theta_fast <- theta_fast - 0.001 * clip_grad(d L_inner / d theta_fast, max_norm=0.1)
```

The default outer loss mirrors Stage 2A but replaces context regularization with
adapter-delta regularization:

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

This is milder than the paper in three ways: it updates adapters instead of the
pretrained FFN weights, it uses a low-rank parameter subset by default, and it
uses first-order meta-gradients unless `--ttte2e_second_order` is explicitly
enabled.

### Stage 2A Test-Time Evaluation

Stage 2A evaluation adapts on training support episodes and predicts different
held-out test episodes from the same `source_split,friction_mu` group. The query
ground truth is used only for visualization and metrics, not for TTT.

Main inference script:

```text
scripts/infer_stage2_ttt.py
```

The standard comparison layout is:

```text
top:    query GT
middle: stage1 prediction without TTT
bottom: stage2 + TTT prediction
```

The detailed comparison layout also includes the support GT videos used for adaptation:

```text
row 1: TTT support 1 GT
row 2: TTT support 2 GT
row 3: query GT
row 4: stage1 prediction without TTT
row 5: stage2 + TTT prediction
```

Comparison helper:

```text
scripts/make_ttt_support_comparison.py
```

### Stage 2B Test-Time Evaluation

Stage 2B evaluation follows the same leakage rule as Stage 2A: use support rows
from the train split, update adapter fast weights, then predict disjoint test
rows from the same `source_split,friction_mu` group. It should use the no-C
Stage1-B baseline prediction for comparison.

Main inference script:

```text
scripts/infer_stage2_ttte2e.py
```

Outputs, logs, videos, images, checkpoints, raw videos, and local model files
are git-ignored. Metadata manifests under `data/push_box_bwm_*` are small and
may be committed.

## Code Entry Points

- `wan_video_action/models/physical_context.py`: latent C encoder and low-rank residual adapters.
- `wan_video_action/pipelines/wan_video_action.py`: BWM pipeline integration.
- `wan_video_action/parsers.py`: CLI/YAML knobs for physical context and adapters.
- `scripts/train.py`: training entry point.
- `scripts/infer.py`: rollout entry point.
- `scripts/run_stage1_ttte2e_9mu_10k_4gpu_guard.sh`: Stage 1B no-C 9mu guarded training launcher.
- `scripts/train_stage2_ttt.py`: Stage 2A latent-C first-order support/query TTT meta-training entry point.
- `scripts/train_stage2_ttte2e.py`: Stage 2B TTT-E2E mild adapter-fast-weight meta-training entry point.
- `scripts/infer_stage2_ttt.py`: Stage 2A test-time C adaptation and rollout entry point.
- `scripts/infer_stage2_ttte2e.py`: Stage 2B test-time adapter-fast-weight adaptation and rollout entry point.
- `scripts/make_ttt_support_comparison.py`: support/query/stage1/stage2 comparison video builder.
- `tests/physical_context_smoke.py`: CPU-only shape and gradient smoke test.

## Smoke Tests

These tests do not load the 5B backbone.

```bash
PYTHONPATH=. .venv/bin/python tests/physical_context_smoke.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python scripts/train.py --help
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python scripts/infer.py --help
```

## Notes For Upcoming Experiments

- Do not start full training until GPUs are available.
- Start with interface-level and tiny tensor tests.
- Keep support/query splits grouped by physical property.
- Do not report success based only on support-trajectory reconstruction.
- Always compare same-property held-out trajectory improvement.
- Keep raw BWM finetune, latent-C adaptation, and TTT-E2E mild adapter-fast-weight adaptation as separate runs.
