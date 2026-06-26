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

### 2. TTTE2E-Style Residual Adapter

Implemented as `PhysicalResidualAdapterBank`.

This is the parameter-adaptation baseline: insert low-rank residual adapters after selected DiT blocks and update only these small modules at adaptation time.

Config knobs:

```yaml
physical_context:
  physical_adapter_mode: "residual"   # none | residual
  physical_adapter_rank: 16
  physical_adapter_layers: "uniform:8"
  physical_adapter_gate_init: 0.0
```

The gate starts at zero, so the base world model behavior is unchanged before training. We can train all adapters, uniformly selected adapters, or an explicit layer list.

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

The current implementation targets the FastWAM hidden push-box dataset:

```text
FastWAM/data/libero_push_box_calibrated_v2_100pairs
```

The prepared BWM manifests live under:

```text
data/push_box_bwm_calibrated_v2_100pairs/
  train.jsonl
  test.jsonl
  action_stats.json
```

Rows are grouped by:

```text
source_split, friction_mu
```

where `source_split` is `straight` or `angled`, and `friction_mu` is one of the calibrated friction bins. Push-focused chunks are sampled around frame starts 50 to 75, with short end-of-episode chunks padded by repeating the last valid frame/action.

### Stage 1: Video Finetuning

Stage 1 adapts the base action-conditioned BWM to the push-box video distribution.

Config:

```text
configs/train/train_push_box_medium_c_stage1_10k.yaml
```

The in-domain checkpoint used by later experiments is:

```text
outputs/push_box_medium_c_stage1_10k_4gpu/step-10000.safetensors
```

This file is an experiment artifact and is ignored by git.

### Stage 2: First-Order TTT Meta-Training

Stage 2 trains a medium-dimensional latent physical code `C` plus C-conditioned low-rank residual adapters.

Main script:

```text
scripts/train_stage2_ttt.py
```

Configs:

```text
configs/train/train_push_box_medium_c_stage2_ttt.yaml
configs/train/train_push_box_medium_c_stage2_ttt_10k_2gpu.yaml
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

### Stage 2 Test-Time Evaluation

Stage 2 evaluation adapts on training support episodes and predicts different held-out test episodes from the same `source_split,friction_mu` group. The query ground truth is used only for visualization and metrics, not for TTT.

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

Outputs, logs, videos, images, checkpoints, and local model/data files are git-ignored. Keep experiment artifacts under `outputs/` or `logs/`.

## Code Entry Points

- `wan_video_action/models/physical_context.py`: latent C encoder and low-rank residual adapters.
- `wan_video_action/pipelines/wan_video_action.py`: BWM pipeline integration.
- `wan_video_action/parsers.py`: CLI/YAML knobs for physical context and adapters.
- `scripts/train.py`: training entry point.
- `scripts/infer.py`: rollout entry point.
- `scripts/train_stage2_ttt.py`: first-order support/query TTT meta-training entry point.
- `scripts/infer_stage2_ttt.py`: test-time C adaptation and rollout entry point.
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
- Keep raw BWM finetune, latent-C adaptation, and adapter adaptation as separate runs.
