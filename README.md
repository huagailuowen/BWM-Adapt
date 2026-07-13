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

## Experiments and Current Findings

This section records the conclusions from the current push-box experiments.
The most important result is that a physical latent is not reliably learned by
ordinary joint training. The working recipe uses a persistent latent table,
assigns one latent to each environment, and alternates optimization of the
latents and the world model while gradually introducing new environments.

### Negative Result: Direct TTT-E2E LoRA Adaptation

The direct TTT-E2E branch treats low-rank residual-adapter parameters as the
test-time state. We tested variations of adapter coverage, gate initialization,
inner-loop learning rate and step count, regularization strength, and the
fraction of late DiT blocks updated by the outer loop.

This branch did not produce a reliable improvement in absolute query quality.
Typical failure modes were:

- an update small enough to preserve the base distribution had too little
  effect on the generated motion;
- a larger update fit the support example but destabilized the query rollout;
- relative-improvement metrics could increase because the unadapted/base loss
  became worse, without a corresponding improvement in the absolute adapted
  query loss;
- support improvement did not consistently transfer to other episodes from the
  same friction environment.

Therefore direct adapter/LoRA fast-weight adaptation is retained as a negative
baseline, not as the primary method. This result is specific to the current BWM
push-box setup; it does not claim that parameter-space TTT is impossible in
general.

### Positive Result: One Persistent Latent C Per Environment

The successful Stage 1 formulation gives each physical environment an opaque
environment label and uses that label to look up a persistent latent:

```text
environment label e_i -> C_table[e_i] -> physical context token(s) -> DiT
```

The label is only a table index. The model is not given the numerical friction
coefficient. All episodes, actions, and chunks recorded under the same
environment label share the same `C_i`; different environments have independent
entries. In the current datasets, an environment label corresponds to one
friction setting, but the mechanism does not assume that `C_i` is numerically
equal to friction.

This distinction is essential:

```text
same environment, different episodes/actions  -> same C_i
different environments/friction groups        -> different C_i
```

We tested both scalar `C` and 32-dimensional `C`. Scalar C is useful for
checking whether the learned table acquires an ordered physical axis. C32 gives
the model more capacity while still allowing its geometry to be inspected with
PCA. Shared initialization remains a useful identification control because all
separation must then be created by data and gradients. It is not, however, the
best-performing initialization in the current push-box experiments.

The C table is part of the experiment state. Restartable model checkpoints are
saved with the matching C table and are pruned together. Additional
context-only tables are saved at phase boundaries to inspect how the latent
geometry evolves; these diagnostic tables do not require a duplicate model
checkpoint.

### Best Current Push-Box Result: Random C32 Initialization

The strongest qualitative Stage 1, Stage 2, and same-friction cross-action
transfer results currently come from the original random-C32 curriculum run
`curc32r65ib2-nc015_88823`. That run uses the earlier 1000-update insertion
cycle and initializes each environment's C32 entry independently from the full
uniform range:

```text
C_i[d] ~ Uniform(-1, 1)
```

The random value is an opaque environment code; it is not derived from or
ordered by the friction coefficient. Its benefit is that environments begin
with strongly separated condition vectors, which breaks the symmetry faced by
shared initialization and gives the DiT distinguishable context tokens from
the start.

A later small-random experiment also performs reasonably well when its initial
C values or offsets are sampled from `[-0.05, 0.05]`. It does not create as
much initial separation as the full `[-1, 1]` run, but it is substantially more
promising than making every environment start from exactly the same vector.
The fact that both random scales work supports random symmetry breaking as the
important initialization property; the full-range random run remains the best
current result.

This result should not be summarized as "1000 updates are better than 1400
updates." The later 1400-update experiments changed both the insertion schedule
and the C initialization, so cycle length and initialization are confounded.
Current evidence points to random initialization as the more important
difference. A controlled schedule-only ablation would be required to claim
that the 1000-update cycle itself is superior.

Until that ablation is complete, the original 1000-update, full-range-random
C32 run is the reference push-box baseline. Small-random initialization is a
competitive ablation, while exact shared and structured initializations should
not replace the random baseline.

### Why Iterative Optimization Is Required

Naively training the model and all C entries together creates an identification
problem. The DiT can absorb environment differences into shared weights while
the C entries collapse, or the high learning rate needed to move C can disturb
the pretrained model. A single optimizer schedule also mixes two very different
time scales: C should move quickly to identify an environment, whereas the 5B
backbone should move slowly.

The working method is block-coordinate optimization:

```text
C phase:     freeze model; update selected C-table entries only
model phase: freeze every C entry; update selected model parameters only
```

Frozen parameters remain frozen even if their optimizer group has a non-zero
configured learning rate. This is a hard phase boundary, not merely a small-LR
approximation.

The current grouped training unit samples multiple environments and multiple
actions before one optimizer update. The standard setting uses four friction
groups and four action/episode examples per friction group, accumulated when
they do not fit in one microbatch. Whenever possible, the same action choices
are represented across the sampled friction groups. This prevents one C entry
from becoming an action or trajectory identifier.

For the event/push experiments, training uses the fixed frame window `65:105`.
It contains the central push and subsequent object motion, so the latent receives
a direct learning signal about the physical response. Mixing arbitrary
pre-contact or post-motion chunks into this experiment weakens that signal and
was a confound in earlier runs.

### Progressive Environment Curriculum

The environment set is introduced progressively instead of optimizing all 80 C
entries from the first update. Each stage selects a nested, approximately
uniform subset over the ordered friction range:

```text
5 active environments -> 10 -> 15 -> 20 -> ...
```

Previously active environments are never removed when a new set is added. The
initial five environments in the 80-friction experiment are:

```text
mu = 0.002, 0.034, 0.075, 0.1225, 0.2
```

For scalar C, these first five entries are initialized in friction order,
linearly from `0.2` to `0.8`; inactive entries are initialized independently in
`[0.4, 0.6]` and remain untouched until activated. For the recommended C32
shared-initialization run, all entries begin from the same random vector.

Before curriculum insertion, the first five C entries are frozen and the model
is trained for 300 updates. The model learning rate warms up during the first
100 updates and then reaches the configured Stage 1 rate. This gives the model a
minimal ability to consume physical-context tokens before C starts moving.

The earlier successful curriculum used a 1000-update insertion cycle:

```text
200  new_context: update only newly inserted C entries with a large C LR
200  all_context: update all active C entries
200  model:       freeze C and update model weights
200  all_context: update all active C entries
200  model:       freeze C and update model weights
```

The later 1400-update curriculum inserts a model phase and a second new-C-only
phase so a new entry is first fitted coarsely, the model is allowed to respond,
and the same entry is then refined. This schedule is more elaborate, but it has
not outperformed the original full-range-random 1000-update baseline:

```text
200  new_context:     only new C entries, LR 0.15
200  model:           C frozen, model LR 1e-5
200  new_context_mid: only new C entries, LR 0.06
200  all_context:     all active C entries, LR 0.03
200  model:           C frozen, model LR 1e-5
200  all_context:     all active C entries, LR 0.03
200  model:           C frozen, model LR 1e-5
```

During `new_context` and `new_context_mid`, batches are sampled only from the
newly inserted groups. Old groups are not masked inside a mixed batch; they are
absent from that phase's sampler. During `all_context`, sampling covers the
entire active pool.

One current C32 experiment stops insertion after the fourth round, when 20
friction environments are active. The remaining 60 environments are reserved
for interpolation/OOD evaluation. After the 20-environment curriculum is
complete, training repeatedly alternates:

```text
200  all_context
200  model
```

No further environment labels are introduced in that run.

### Evidence and Evaluation Criteria

The iterative runs produced both qualitative and geometric evidence that C is
being used as an environment variable:

- rollouts generated with learned environment C values were visibly better
  than the old fixed-C128 baseline on the push-motion cases;
- adapting from a mean/shared C initialization moved C toward regions occupied
  by the corresponding trained friction groups and transferred to other
  episodes from the same group;
- a representative C1 run with 35 active groups reached a strong correlation
  between friction and scalar C (`r` approximately `+0.92`);
- in a representative shared-init C32 run, PC1 explained about 49% of active-C
  variance and strongly correlated with friction (`|r|` approximately `0.83`;
  PCA sign itself is arbitrary).

Diffusion flow-matching loss is still logged, but it is not the sole success
criterion. The main checks are absolute adapted-query quality, generated motion,
transfer to unseen episodes of the same environment, and organization of the C
table. Relative improvement alone is insufficient because it can rise when the
base prediction becomes worse.

For Stage 2/test-time adaptation, an unseen environment starts from a shared or
mean latent and updates only its task-local C on support episodes. The model is
frozen in the inner loop. The adapted C is then reused for disjoint query
episodes from that environment. This preserves the Stage 1 semantics: C stores
an environment-level property rather than a single trajectory.

For the current push-box setup, this grouped iterative procedure is the required
training recipe for latent C. Ordinary joint Stage 1 training and direct
from-base Stage 2 training remain ablations, not interchangeable replacements.

## Training Plan

The plan below also documents earlier baselines. For the current latent-C
experiments, the validated grouped iterative procedure is defined in
`Experiments and Current Findings` above and supersedes the simple token-branch
Stage 1 recipe where the two disagree.

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

Historical/simple baseline branches:

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
- `scripts/train_stage1_grouped_context.py`: environment-indexed C-table training, iterative model/C phases, and progressive friction curriculum.
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
- Treat direct TTT-E2E LoRA adaptation as a negative baseline for the current push-box setup.
- Train environment-indexed latent C with explicit alternating C/model phases; do not silently replace it with ordinary joint optimization.
- Keep raw BWM finetune, latent-C adaptation, and TTT-E2E mild adapter-fast-weight adaptation as separate runs.
