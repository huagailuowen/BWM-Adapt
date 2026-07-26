# Environment-Latent Identification Without Endpoint Regression

## Status

This document is the current design record for the environment-latent bridge
and Stage 2 test-time adaptation work. It records the motivation, confirmed
failure modes, proposed core method, optional augmentation, and unresolved
questions. It does not describe an active training run.

All future changes to this design should be reflected here before or together
with implementation changes.

## Current Authoritative Design: Counterfactual Stage 1 and Latent Bridge

This section records the authoritative design as of 2026-07-25. Where an older
section in this document proposes partial-noise Teacher velocity differences,
a separate editing adapter, a source-token branch, lamp-region weighting, or
only 5-10 percent synthetic correction, this section supersedes it.

The complete objective has two orthogonal sampling dimensions:

```text
data path:
    ordinary real-data flow matching
    counterfactual source-noise correction

physical context:
    exact Global
    environment endpoint
    near-Global segment point
    remaining segment-interior point
```

Counterfactual editing is a Stage 1 auxiliary training task. It is not an
editing interface used by Stage 2. Its purpose is to force the shared Wan/BWM
vector field to obey the physical context when residual video appearance and
the target physical condition disagree. The nonlinear Global-to-endpoint
curriculum then organizes that context-sensitive field into paths that Stage 2
can optimize from the frozen Global context.

### Confirmed failure of partial-noise Teacher deltas

Ordinary Stage 1 flow matching only presents matched pairs:

```text
video from environment e + context C_e
```

At partial noise, the latent can still reveal the future lamp state. The model
can therefore reconstruct that visible state while ignoring the physical
context:

```text
v_teacher(z_t_bright, C_A)
~= v_teacher(z_t_bright, C_B)
~= v_teacher(z_t_bright, C_C)
~= v_teacher(z_t_bright, C_D)
```

Consequently, a Teacher difference such as

```text
v_teacher(z_t, C_A) - v_teacher(z_t, C_B)
```

is not a valid counterfactual target in this regime. The Teacher has not
learned the desired local field there. It produces context-specific outcomes
only when it regenerates the future from full or near-full noise.

The frozen Teacher is therefore used only to create offline fake source
videos by complete rollout from full noise. It is not used as a local
partial-noise vector-field oracle.

### No separate editing branch

The counterfactual objective uses the existing Wan/BWM video input and the
existing action and physical-context conditioning paths. It does not add:

```text
a VACE adapter
source-video context tokens
an editing task token
a separate editing backbone
a lamp mask, crop, ROI loss, or spatial weighting
```

Both real and counterfactual samples are represented by the same training
tuple:

```text
model_input_latent
target_velocity
action_condition
physical_context
environment_id
sample_weight
```

The DiT is not told which data path produced a sample.

### Real-data flow path

Let the Wan flow convention be:

```text
z_t = t z_clean + (1 - t) epsilon
t = 0: full noise
t = 1: clean data
```

For a real target video from environment `y`:

```text
z_in_real = t z_target + (1 - t) epsilon
v_real    = z_target - epsilon
```

This path retains the original Wan/BWM flow-matching objective and covers the
full scheduler range in real-only batches.

### Counterfactual source-noise path

For a real target chunk with initial history `h`, action `a`, and target
environment `y`, select a source environment `r` whose causal outcome under
that action differs from `y`. Generate the fake source with the frozen
endpoint Teacher from full noise:

```text
x_source_fake = rollout(
    frozen_teacher,
    full_noise_seed,
    initial_history=h,
    action=a,
    context=C_r
)
```

The preferred target remains real:

```text
x_target = real video under environment y
```

Use the same initial history and action for source and target. For a red-button
action, source and target must differ in the red-control property. For a
blue-button action, they must differ in the blue-control property. Merely
choosing two different environment IDs is insufficient.

Encode and noise the fake source in the normal video latent slot:

```text
z_in_cf = t z_source_fake + (1 - t) epsilon
```

The desired velocity must reconstruct the target rather than the fake source:

```text
v_cf = (z_target - z_in_cf) / (1 - t)
```

When source and target are identical, this reduces exactly to ordinary flow
matching. In implementation, the safer equivalent is to convert the predicted
velocity through the existing scheduler into a predicted clean latent and
supervise that latent against `z_target`, with the corresponding flow-scale
weighting.

Counterfactual batches use only high and near-full noise. Most samples should
retain enough source structure to create a genuine contradiction, so exact
full noise must not dominate. Initial noise-fraction coverage is:

```text
20% of CF samples: 0.90-1.00 noise
60% of CF samples: 0.70-0.90 noise
20% of CF samples: 0.55-0.70 noise
```

Sampling must use the scheduler's actual signal/noise coefficient rather than
assuming that a raw timestep array index is linear in noise strength.

### Physical-context geometry and target distribution

For endpoint `e`, define:

```text
q(s)   = 1 - (1 - s)^5
C_e(s) = C_global + q(s) (C_e - C_global)
```

The corresponding target-environment mixture is:

```text
p_e(s) = 1/4 + 3/4 q(s)
p_j(s) = (1 - q(s))/4, j != e
```

This gives a uniform four-environment distribution at `s=0` and a pure
environment distribution at `s=1`. A mixture context never uses a
pixel-averaged target video. The batch contains real target videos from all
four environments, and their independently averaged losses are combined with
the probabilities above:

```text
L_mode(C) = sum_y p_y(C) mean_i L_mode(y, i, C)
```

The same target distribution is used for both the real and counterfactual data
paths. Counterfactual source environments are selected after the target
environment and must have a different action-conditioned physical outcome.

### Joint sampling matrix

| Physical-context location | Real FM | Counterfactual correction |
| --- | --- | --- |
| Exact Global | balanced 25% per environment | disabled |
| Endpoint `C_e` | real environment `e` anchor | wrong-outcome fake source to real `e` target |
| Near-Global `C_e(s)` | weighted four-environment real mixture | same weighted targets with wrong-outcome fake sources |
| Interior `C_e(s)` | weighted four-environment real mixture | same weighted targets with wrong-outcome fake sources |

Exact Global has no unique counterfactual target. Its function is learned from
the balanced real mixture and preserved by its frozen coordinate and
post-update-300 functional anchor. Directional counterfactual supervision
starts at strictly positive `s` in the near-Global corridor.

### Global phase and context-location quotas

The first 300 optimizer updates use:

```text
100% exact-Global real-data batches
four environments balanced exactly
ordinary real flow matching
no counterfactual samples
```

After update 300:

```text
freeze the numerical Global context
save it with the context table
save a functional Global snapshot
continue balanced real Global-anchor replay
```

Dedicated exact-Global anchor updates are scheduled separately. Among
non-Global updates, enforce the long-run context-location quotas:

```text
40%: exact endpoints, s = 1
30%: near-Global, 0 < s <= 0.1
30%: remaining interior, 0.1 < s < 1
```

The near-Global quota is stratified equally over:

```text
0.01 <= s < 0.03
0.03 <= s < 0.06
0.06 <= s <= 0.10
```

The target endpoint direction is balanced over all four environments.

### Counterfactual sample quota: 25 percent

Counterfactual correction occupies 25 percent of post-Global training
examples. Exact-Global anchor examples are excluded from the denominator
because counterfactual supervision is undefined there. A cumulative quota
tracker, rather than independent Bernoulli sampling, keeps the realized
non-Global ratio at:

```text
75% ordinary real FM examples
25% counterfactual correction examples
```

Use two effective-batch templates:

```text
real-only batch:
    64 real examples
    ordinary full-range timestep policy

paired high-noise batch:
    32 real examples
    32 counterfactual examples
    shared high-noise timestep policy
```

Scheduling paired high-noise batches for half of non-Global updates yields:

```text
0.50 paired-batch frequency * 0.50 CF share = 0.25 CF examples
```

If exact-Global anchor updates are included in an overall run-level ratio,
adjust paired-batch frequency using:

```text
paired_batch_probability = 0.50 / (1 - global_anchor_fraction)
```

or use the cumulative example counter directly. Sampling frequency controls
the initial CF strength, so start with `lambda_cf = 1` after separately
normalizing real and CF per-environment means. Change the loss coefficient only
if logged gradient norms show synthetic supervision dominating real data.

### Efficient effective-batch layouts

The global effective batch remains:

```text
4 environments * 16 distinct target chunks = 64 examples
```

On four GPUs, every rank receives:

```text
4 chunks per environment * 4 environments = local batch 16
```

For a real-only batch, every environment contributes 16 real examples.

For a paired high-noise batch, every environment contributes:

```text
8 distinct real examples
8 distinct counterfactual examples
```

This preserves 16 distinct target chunks per environment without duplicating
one target merely to obtain paired statistics.

An endpoint update packs all four endpoints in one forward:

```text
environment A samples use C_A
environment B samples use C_B
environment C samples use C_C
environment D samples use C_D
```

A bridge update selects one endpoint direction and one `s`, broadcasts the
same `C_e(s)` to all 64 examples, and uses 16 target chunks from each
environment. One bridge point per update preserves enough evidence for every
mixture component; endpoint directions rotate across updates.

Rank 0 samples and broadcasts:

```text
batch template
context-location category
endpoint direction
s
timestep/noise-strength stratum
```

Samples in one comparison block share the timestep. Matched scenario/action
groups across environments reuse common noise, while different groups use
different noise tensors to retain diversity.

### One-forward loss computation

Real VAE target latents and full-rollout fake-source latents are cached
offline. The frozen Teacher is never evaluated inside the Stage 1 optimizer
step.

The dataloader constructs real and counterfactual `model_input_latent` and
`target_velocity` tensors, concatenates them along the batch dimension, and
runs one ordinary Wan/BWM forward. No Python loop over environment, context,
or data path is required.

Compute unreduced per-example losses, then average by data path and target
environment:

```text
L_real_y = mean real loss for target environment y
L_cf_y   = mean CF loss for target environment y

L_real(C) = sum_y p_y(C) L_real_y
L_cf(C)   = sum_y p_y(C) L_cf_y
```

The post-Global objective is:

```text
L_total =
    L_real(C)
  + lambda_cf L_cf(C)
  + lambda_global_anchor L_global_anchor
  + lambda_endpoint_anchor L_endpoint_anchor
```

Endpoint real replay protects endpoint generation. The frozen update-300
Global snapshot protects the function at the fixed Global coordinate while
the student parameters continue to change.

### Required diagnostics

Log real and counterfactual losses separately by:

```text
target environment
source environment
red/blue action
context-location category
near-Global subrange
noise-strength bucket
```

Also report:

```text
real and CF gradient norms
full-video correct-context versus wrong-context loss margins
context-gradient direction and magnitude from Global
endpoint rollout preservation
Global functional drift
Stage 2 context trajectories
```

The acceptance criterion is not low edit loss. The model must preserve endpoint
rollouts while producing a context-sensitive high-noise field whose real
support loss leads from the frozen Global context toward the correct endpoint.

## 1. Confirmed Experimental Evidence

The starting point is a successful four-environment LightSwitch Stage 1 model:

```text
teacher/source checkpoint: job 90299, step 3696
environment contexts: C_A, C_B, C_C, C_D
```

A later experiment introduced a trainable global context and trained points
between the global context and each endpoint:

```text
bridge checkpoint: job 91927, step 1500
```

The four endpoint context vectors were frozen, but the student DiT, action
encoder, and physical-context encoder were updated. A controlled evaluation
used the same eight red-button and blue-button chunks with the same inference
settings for both checkpoints.

The original four-context checkpoint retained the expected endpoint behavior.
The bridge checkpoint produced incorrect lamp transitions, including false
lamp activation in a red-button query from the blue-only environment.

The conclusion is:

```text
freezing endpoint coordinates does not preserve endpoint behavior
```

The model defines the function at a context coordinate. Updating the model can
change that function even when the coordinate itself never moves.

## 2. Three Distinct Sources of Information

The previous design mixed real supervision, teacher preservation, synthetic
augmentation, and cross-environment correction. These sources have different
reliability and must remain separate in both code and logs.

### 2.1 Real data

Let `D_real(e)` be videos collected in real environment `e`:

```text
x_e ~ D_real(e)
```

Real data is the authoritative source for physical behavior and generation
quality. It defines the primary bridge distribution and the endpoint
flow-matching target.

For a bridge point representing:

```text
70% A + 10% B + 10% C + 10% D
```

the primary objective uses real videos:

```text
L_real =
    0.7 FM(real A)
  + 0.1 FM(real B)
  + 0.1 FM(real C)
  + 0.1 FM(real D)
```

Synthetic teacher videos must not be counted as the real 70 percent A term.

### 2.2 Frozen-teacher predictions

The original model is retained as a frozen teacher. Its prediction on a real
noised video is a functional-preservation target:

```text
v_teacher(x_t, t, C_e)
```

This is not a new data sample. It constrains the student vector field at an
endpoint so that bridge training cannot silently redefine the original
endpoint behavior.

### 2.3 Counterfactual augmented data

A counterfactual teacher rollout has the form:

```text
initial frame/action from a B chunk
+ endpoint context C_A
+ frozen teacher
-> synthetic B-to-A video
```

This output is not real A data. It may contain teacher artifacts, inherit
teacher mistakes, or represent an initial-state/action combination that does
not occur in the true A distribution.

Counterfactual data is therefore optional auxiliary supervision. It must have
its own loss name, weight, metrics, cache, and ablation. It cannot replace real
bridge data or endpoint preservation.

The preferred order for corrective pairs is:

```text
matched real cross-environment pairs
> teacher-generated counterfactual pairs
```

## 3. Two Primary Objectives

The project has two primary goals. Intermediate-context generation is useful
only insofar as it supports these goals.

### 3.1 Preserve every endpoint

At every endpoint `C_e`, the updated student must retain the original Stage 1
generation behavior.

The endpoint objective combines real flow matching and frozen-teacher
prediction distillation:

```text
L_endpoint_real(e) = FM(real e data, C_e)
```

```text
L_endpoint_teacher(e) =
    ||v_student(x_t, t, C_e)
      - stopgrad(v_teacher(x_t, t, C_e))||^2
```

Teacher and student receive the same:

```text
clean real latent
Gaussian noise
timestep
endpoint context
initial frame and action
```

The endpoint contexts and teacher parameters are always frozen.

Ordinary endpoint flow matching alone is not a sufficient preservation
constraint. It can improve its stochastic training loss while changing the
rollout vector field enough to break binary causal behavior.

### 3.2 Make the real-support loss guide context toward the correct endpoint

For real support data from environment `e`, define:

```text
L_e(C) = flow-matching loss of real environment-e support under context C
```

Near the global context, the desired test-time gradient is:

```text
-g_e = -gradient_C L_e(C)
```

It should align with:

```text
d_e = C_e - C_global
```

The target property is:

```text
cosine(-gradient_C L_e(C_global), C_e - C_global) > 0
```

The gradient must also have enough magnitude to move the context. Direction
without progress is insufficient.

## 4. Nonlinear Real-Data Bridge

The latent remains on a simple line segment:

```text
C_e(s) = (1 - s) C_global + s C_e
s in [0, 1]
```

The target-environment data weight changes nonlinearly:

```text
q(s) = 1 - (1 - s)^4
```

```text
w_e(s) = 1/4 + 3/4 q(s)
w_j(s) = (1 - q(s))/4, j != e
```

This separates geometric position from distribution specialization.

| `s` | Target environment | Each other environment |
| ---: | ---: | ---: |
| 0.00 | 0.2500 | 0.2500 |
| 0.10 | 0.5079 | 0.1640 |
| 0.25 | 0.7627 | 0.0791 |
| 0.50 | 0.9531 | 0.0156 |
| 0.75 | 0.9971 | 0.0010 |
| 1.00 | 1.0000 | 0.0000 |

At the geometric midpoint, the selected environment already supplies more
than 95 percent of the real-data objective.

The real bridge loss is:

```text
L_bridge_real(C_e(s)) =
    w_e(s) FM(real e data, C_e(s))
  + sum_j!=e w_j(s) FM(real j data, C_e(s))
```

The exponent `4` should be configurable. Larger exponents specialize more
rapidly after leaving the global context.

## 5. Explicit Global-Neighborhood Loss Funnel

A nonlinear bridge distribution may shape the loss landscape indirectly, but
it does not guarantee that test-time gradient descent follows the correct
ray. The gradient field should receive explicit supervision.

Directly training cosine alignment through `gradient_C L` requires second-order
gradients and is expensive. The initial implementation should use finite
loss differences.

For environment `e`:

```text
d_e = normalize(C_e - C_global)
C_plus_e = C + delta d_e
```

For every wrong environment `j`:

```text
C_plus_j = C + delta d_j
```

Require progress in the correct direction:

```text
L_e(C_plus_e) < L_e(C)
```

Require the correct direction to beat wrong directions:

```text
L_e(C_plus_e) + margin < L_e(C_plus_j)
```

One possible finite-difference objective is:

```text
L_progress =
    ReLU(m_progress - (L_e(C) - L_e(C_plus_e)))
```

```text
L_direction =
    mean_j!=e ReLU(m_direction + L_e(C_plus_e) - L_e(C_plus_j))
```

The probe context `C` should be sampled in a neighborhood around the global
context rather than always being exactly equal to it. This trains a useful
field after the first Stage 2 update instead of only at one point.

The explicit funnel objective uses real support data. Counterfactual teacher
videos are not required.

## 6. Timestep Identifiability and the Denoising Shortcut

### 6.1 The problem

The current Stage 2 implementation used a fixed scheduler index near 500. In
observed scheduler logs, this corresponds approximately to:

```text
sigma ~= 0.83
x_sigma ~= 0.17 x_clean + 0.83 noise
```

For natural video, 17 percent clean signal may appear weak. For a binary lamp
state, it can still reveal whether the lamp is black or yellow. The denoiser
can reconstruct the visible state without using the environment context.

The result is a context-identification failure:

```text
L(real A, C_A)
~= L(real A, C_B)
~= L(real A, C_C)
~= L(real A, C_D)
```

All candidate contexts obtain a small denoising loss, and the gradient with
respect to context becomes weak or arbitrary.

This is distinct from poor generation quality. A model can denoise a partially
visible lamp correctly while providing no useful signal for identifying the
causal environment.

### 6.2 Why exact full noise is not the complete solution

At exact full noise:

```text
sigma = 1
x_sigma = epsilon
```

the future video cannot leak the target lamp state. The model must use the
initial frame, action, and environment context.

However, using only the exact boundary has drawbacks:

```text
high gradient variance
possible scheduler-boundary weighting effects
weak coverage of the full diffusion trajectory
full-frame loss dominated by robot and background pixels
```

The initial design should use a near-full-noise range rather than one exact
full-noise point.

### 6.3 Different objectives require different timestep distributions

Timestep sampling should be separated by objective:

| Objective | Timestep policy |
| --- | --- |
| Endpoint real FM | Full scheduler range |
| Endpoint teacher preservation | Full scheduler range |
| Real bridge generation | Stratified full range |
| Global-neighborhood funnel | High and near-full noise |
| Structured correction | Medium-high and high noise |
| Stage 2 context identification | Near-full noise first |
| Stage 2 refinement | Medium-high noise second |

The model should not be trained only at full noise. The context-identification
branch should emphasize high noise, while endpoint preservation still covers
the complete denoising trajectory.

## 7. Stratified Multi-Timestep Updates

One shared timestep for all 64 chunks makes environment comparisons clean but
covers only one denoising regime per optimizer update. Fully independent
per-chunk timesteps confound environment comparisons with denoising difficulty.

Use stratified shared blocks.

For a global batch of 64 and `K=4` timestep blocks:

```text
per timestep block:
    A: 4 chunks
    B: 4 chunks
    C: 4 chunks
    D: 4 chunks
    total: 16 chunks

four blocks:
    4 * 16 = 64 chunks
```

Sample one timestep from each scheduler region:

```text
high or near-full noise
medium-high noise
medium-low noise
low-noise refinement
```

Within a block, all environments share the same timestep. Across blocks, one
update covers several denoising regimes. Gaussian noise remains independent
per chunk unless a matched comparison requires common random numbers.

Teacher and student evaluations in the same distillation term must reuse the
same timestep and Gaussian noise.

## 8. Proposed Two-Phase Stage 2 Context Optimization

Stage 2 optimizes only the test-time context. It should not begin by asking a
medium-noise reconstruction objective to identify the environment.

### 8.1 Phase I: near-full-noise identification

Use several timesteps with:

```text
sigma in approximately [0.95, 0.995]
```

For balanced support chunks and a small fixed noise bank:

```text
L_ID(C) = mean_support,timestep,noise FM(real support, C)
```

Initial proposal:

```text
10 to 20 inner steps
relatively large context learning rate
4 to 8 high-noise timestep/noise pairs
common random numbers across context comparisons
```

The purpose is to choose the environment and leave the global neighborhood in
the correct direction.

Do not use a single noise realization. A small fixed bank reduces stochastic
gradient variance. Part of the bank may be refreshed during a long inner loop
to avoid overfitting one set of noise samples.

### 8.2 Phase II: medium-high-noise refinement

After context identification, use:

```text
sigma in approximately [0.80, 0.95]
```

Initial proposal:

```text
10 to 20 inner steps
smaller context learning rate
```

This phase refines the location after the correct environment direction has
been established.

Low-noise reconstruction should not drive environment identification because
it provides the strongest target-state shortcut.

### 8.3 Event-focused identification loss

Even at high noise, full-frame FM can be dominated by robot motion, table
texture, and static geometry. The causal lamp occupies a small spatial region.

For LightSwitch, consider:

```text
L_ID =
    L_full_frame
  + lambda_lamp L_lamp_region
  + lambda_post_action L_after_button_press
```

The temporal mask should emphasize frames after the button event. A spatial
must remain optional outside LightSwitch.

## 9. Endpoint-Preservation Branch

Every student update that changes model parameters risks endpoint regression.
The training loop must include explicit endpoint-preservation batches.

The preservation batch should contain all environments and both button actions.
It computes:

```text
L_endpoint =
    lambda_endpoint_real L_endpoint_real
  + lambda_endpoint_teacher L_endpoint_teacher
```

The frozen teacher can evaluate all endpoint-conditioned samples as one packed
no-gradient batch or as memory-limited microbatches. No teacher rollout video
is needed for this branch.

Endpoint evaluation is a hard checkpoint gate. A checkpoint that fails any
red-button or blue-button endpoint case should not proceed to Stage 2
evaluation, regardless of bridge loss.

## 10. Optional Counterfactual Augmentation

Counterfactual augmentation addresses a separate question:

```text
Can the model correct an input that already contains structure from the wrong environment?
```

It is not the primary solution to context identification. Near-full-noise Stage
2 removes most target-state leakage without synthetic data.

### 10.1 Preferred real corrective pairs

If two real samples can be matched by initial lamp state, button action, scene,
and motion timing, use:

```text
real wrong-environment source
real correct-environment target
```

This avoids teacher-generation bias.

### 10.2 Teacher counterfactual fallback

When no suitable real pair exists, generate a controlled target:

```text
real B initial frame/action
+ frozen teacher under C_A
-> synthetic B-to-A target video
```

This target remains synthetic and must be logged as augmentation.

### 10.3 Mathematically consistent corrective flow

For target environment A and donor structure B:

```text
r_B = (1 - beta) epsilon + beta z_B
x_t = (1 - t) z_A + t r_B
v_target = r_B - z_A
```

This defines a consistent path from a noisy B-structured source to an A target.
Do not feed a B-derived noisy input while retaining the ordinary A-to-Gaussian
velocity target.

Initial limits:

```text
augmentation/correction share: at most 5% to 10%
noise range: medium-high and high
beta: configurable, initially 0.3 to 0.7
```

Synthetic correction must be introduced only after the real-data core method
preserves endpoints and produces a measurable global gradient funnel.

## 11. Core and Optional Objectives

The first implementation should optimize only the core objective:

```text
L_core =
    lambda_bridge L_bridge_real
  + lambda_progress L_progress
  + lambda_direction L_direction
  + lambda_endpoint_real L_endpoint_real
  + lambda_endpoint_teacher L_endpoint_teacher
```

Counterfactual augmentation is a later, separate ablation:

```text
L_optional =
    lambda_aug L_counterfactual_aug
  + lambda_correct L_structured_correction
```

The augmented terms must never be silently included in the real bridge
weights.

## 12. Diagnostics Before Another Long Training Run

### 12.1 Context-identifiability matrix

Measure:

```text
Loss[real environment][candidate endpoint context][sigma bucket]
```

For LightSwitch:

```text
4 real environments
* 4 endpoint contexts
* 4 to 6 sigma ranges
```

Report:

```text
correct-context top-1 accuracy per sigma bucket
correct-minus-wrong loss margins
gradient norm with respect to context
gradient cosine toward the correct endpoint
```

This test should verify the hypothesis that medium-noise losses are nearly
context-invariant while near-full-noise losses are more discriminative.

### 12.2 Endpoint regression gate

For every checkpoint, run:

```text
four environments
* red-button chunk
* blue-button chunk
= eight endpoint cases
```

Compare:

```text
GT
original Stage 1 teacher
current Stage 1 student
```

### 12.3 Global-gradient test

For each real environment support set, initialize at the saved global context
and report before any rollout:

```text
initial support loss
gradient norm
cosine to every endpoint direction
one-step loss change toward the correct endpoint
one-step loss change toward wrong endpoints
```

### 12.4 Stage 2 rollout test

Only after passing endpoint and gradient gates:

```text
start every rollout from the saved global context
use balanced evidence for both buttons
run near-full-noise identification
run medium-high-noise refinement
record the complete context trajectory
render PCA and GT/Stage1/Stage2 videos
```

## 13. Recommended Ablation Order

Do not introduce all mechanisms in one run.

| Variant | Nonlinear real bridge | Multi-timestep | Endpoint teacher | Funnel | Synthetic correction |
| --- | --- | --- | --- | --- | --- |
| Baseline | no | no | no | no | no |
| A | yes | yes | no | no | no |
| B | yes | yes | yes | no | no |
| C | yes | yes | yes | yes | no |
| D | yes | yes | yes | yes | yes |

The decision sequence is:

```text
1. Verify endpoint preservation.
2. Verify high-noise context identifiability.
3. Verify global-gradient direction and magnitude.
4. Verify Stage 2 trajectory and rollout quality.
5. Add synthetic correction only if wrong-state recovery remains a problem.
```

## 14. Trainable and Frozen State

Always frozen:

```text
original teacher model
four endpoint context vectors
```

Trainable in the student experiment:

```text
global context
physical-context pathway
explicitly selected student model modules
```

The initial student trainable scope should be conservative. The previous full
model update damaged endpoint behavior. Action encoding should remain frozen
unless an ablation demonstrates that updating it is necessary.

Every trainable scope must be explicit in configuration and recorded in the
experiment manifest.

## 15. Safety, Compatibility, and Artifact Rules

All new behavior is opt-in. Existing Stage 1, curriculum, grouped-context, and
Stage 2 workflows retain their current behavior when the new fields are absent.

Teacher rollout caches, synthetic data, model outputs, and Slurm jobs are not
committed to the repository.

Each model checkpoint must be saved with its matching context table. Endpoint
source checkpoints and tables are immutable. Experimental runs use unique
output directories and retain at least the latest two complete checkpoint/table
pairs.

## 16. Open Questions

The following remain experimental decisions rather than conclusions:

```text
bridge exponent and sampled s distribution
finite-difference delta and margins
relative weights of endpoint real FM and teacher preservation
number and boundaries of timestep strata
near-full-noise Stage 2 learning-rate schedule
whether matched real corrective pairs are available
whether counterfactual teacher augmentation provides benefit after the core method
which student modules can update without endpoint regression
```

## Hard constraint: no lamp-region emphasis

Training and Stage 2 adaptation must not use any task-specific spatial shortcut that deliberately emphasizes the lamp display region. In particular, do not use a lamp mask, ROI crop, lamp-pixel loss reweighting, attention bias, spatial loss multiplier, or privileged lamp-state annotation to strengthen the training signal.

All image regions must remain under the same base flow-matching objective. The latent must identify the environment from ordinary observations and actions rather than from a hand-designed display-region signal. Lamp-region measurements may be reported only as detached evaluation diagnostics; they must never contribute gradients, sample weights, latent updates, or model-selection scores during training or Stage 2 adaptation.

## Literature-derived direction: corrective editing without hard pseudo-video labels

Research reviewed on 2026-07-24:

- [SDEdit](https://arxiv.org/abs/2108.01073) adds noise to a source image and denoises it under a target condition. Its main transferable lesson is that the noise level controls the tradeoff between preserving the observed source and allowing a semantic change.
- [Delta Denoising Score (DDS)](https://arxiv.org/abs/2304.07090) subtracts a matched source score from a target score to cancel nuisance and erroneous score-distillation directions. The transferable lesson is to supervise a relative conditional direction under shared noise instead of treating an absolute pseudo-output as truth.
- [FlowEdit](https://arxiv.org/abs/2412.08629) constructs a direct source-to-target ODE from the difference between source- and target-conditioned velocity fields. It uses coupled noisy states and averages random realizations to lower variance and transport cost. This is the closest precedent for a flow-matching world model.
- [I2SB](https://arxiv.org/abs/2302.05872) and [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948) directly learn stochastic paths between paired endpoint distributions. They are most relevant if the simulator can provide truly paired rollouts with matched initial state and action under different environments.
- [Plug-and-Play Diffusion Features](https://arxiv.org/abs/2211.12572) and [TokenFlow](https://arxiv.org/abs/2307.10373) preserve source layout and temporal coherence through generic diffusion-feature correspondences. This can help generate temporally coherent diagnostic edits, but excessive feature preservation may also preserve the wrong physical outcome.
- [Dreamix](https://arxiv.org/abs/2302.01329) combines editing with source-video fidelity objectives. It reinforces the need for real-data replay, but it does not by itself solve latent identifiability.
- [InstructPix2Pix](https://arxiv.org/abs/2211.09800) demonstrates that large synthetic edit-pair datasets can train useful editors. For this project it is also a warning: generated pairs can transfer systematic Teacher bias into the student.
- [ControlNet](https://arxiv.org/abs/2302.05543) freezes the pretrained generator and introduces zero-initialized residual conditioning. We should borrow only this preservation principle, not its spatial-control formulation.

### Main conclusion

The first corrective experiment should not generate a complete counterfactual video and use that video as a hard flow-matching target. A full pseudo-video compounds Teacher sampling error across denoising steps and can make the student reproduce Teacher artifacts. Instead, use the frozen four-endpoint model as a local velocity-field Teacher on real noised video latents and distill only the relative environment-dependent correction.

This is analogous to DDS and FlowEdit:

1. Start from a real video latent, not a generated video.
2. Use exactly the same timestep and noise realization for all compared conditions.
3. Evaluate a frozen Teacher under a source/global condition and a target-environment condition.
4. Subtract the two predicted velocity fields so that shared reconstruction, texture, action, and noise components largely cancel.
5. Use the resulting delta as a stopped-gradient target for the direction represented along the global-to-environment latent path.

This is an adaptation inspired by the cited editing methods, not a claim that their image-editing objectives transfer unchanged to the world-model setting.

### Critical mixture-distribution caveat

For a mixture of environments, the score or flow velocity of the mixture is generally not the arithmetic mean of the component scores or velocities. The correct mixture field is weighted by the posterior probability of each component at the current noisy state. Therefore:

- Do not define the global Teacher as `(v_A + v_B + v_C + v_D) / 4` and treat it as exact.
- Train the global context on an equal mixture of real A/B/C/D data with the ordinary flow-matching objective.
- Freeze a snapshot only after that global context passes balanced real-data diagnostics.
- At near-full noise the arithmetic mean may be a useful approximation because the component distributions overlap more strongly, but it should only be an ablation, not the primary target.

### Proposed first implementation

Let `v_E` be the frozen successful endpoint Teacher, `v_G` the frozen real-data-trained global Teacher, and `v_S` the bridge student. Let `N(x, epsilon, t)` denote the model's existing noising operation. For a real chunk `x`, shared noise `epsilon`, high-noise timestep `t`, target environment `e`, and bridge coordinate `s in [0, 1]`:

```text
z_t       = N(x, epsilon, t)
delta_e   = stopgrad(v_E(z_t, t, c_e) - v_G(z_t, t, c_global))
q(s)      = 1 - (1 - s)^4
student_d = v_S(z_t, t, c_e(s)) - v_S(z_t, t, c_global)
L_delta   = ||student_d - q(s) * delta_e||^2
```

The subtraction must use the same `z_t`, timestep, action condition, initial-frame condition, and noise realization. Otherwise most of the delta is sampling variance rather than an environment-edit direction.

Use this only in the near-full/high-noise regime initially. At medium or low noise, evaluating an endpoint Teacher on a video from another environment can be off-manifold, and the visible lamp state can dominate the prediction. A later FlowEdit-style coupled-state construction can address this if the simple same-state delta is insufficient.

### Endpoint preservation architecture

The old four-environment endpoint model should remain frozen. Add a zero-initialized bridge residual path or adapter, inspired by ControlNet's preservation strategy, rather than allowing bridge training to rewrite the endpoint generator.

The bridge residual gate must be exactly zero at every endpoint context. This makes endpoint generation invariant by construction, while a gate with a nonzero inward derivative can still provide a Stage 2 gradient away from an incorrect initialization. Real endpoint replay remains useful as a diagnostic and regularizer, but it should not be the only protection against endpoint regression.

This is a generic latent-conditioning residual. It must not contain a lamp mask, lamp ROI, privileged lamp-state input, or any lamp-specific spatial weighting.

### Timestep strategy inherited from editing models

SDEdit and related editing work support the existing concern about the denoising shortcut:

- Low and medium noise preserve source appearance strongly. For a binary lamp state, the model can reconstruct the visible state while ignoring the environment latent.
- Near-full noise removes that shortcut and makes the action, initial observation, and environment condition responsible for the outcome.
- Exact full noise may produce a high-variance latent gradient, so the first identification phase should sample a narrow near-full-noise band rather than one deterministic endpoint.

Recommended Stage 2 schedule:

1. Identification: several shared-noise samples with noise strength approximately `0.95-0.995`; update only the latent.
2. Refinement: after the latent direction is stable, add samples around `0.80-0.95` to recover trajectory-specific detail.
3. Keep multiple timestep strata within an update. Each comparison block shares a timestep internally, but different blocks draw different random timesteps.

No spatial region, including the lamp display, receives additional loss weight in either phase.

### Batch construction

Retain balanced environment evidence:

- Four environments per effective update.
- At least 16 diverse chunks per environment when constructing bridge supervision.
- Include red-button and blue-button interactions, both initial lamp states, and multiple episodes.
- Couple condition comparisons for a given chunk with identical noise and timestep.
- Spread the resulting comparison blocks across the four GPUs; do not force the entire effective batch to use one timestep.

For compute efficiency, pack source/global and target conditions along the batch dimension and run the frozen Teacher under `no_grad`. Cache VAE video latents, but do not cache one fixed Teacher target for all training because random noise and timestep coverage are part of the objective.

### Role of truly paired and synthetic edits

Preferred hierarchy:

1. Real matched counterfactual pairs: same initial state and action replayed by the simulator under different environment settings. These support a principled I2SB/DDBM-style direct bridge.
2. Relative Teacher velocity fields on real noised latents: the recommended first experiment.
3. FlowEdit-generated counterfactual videos: diagnostic visualization and, only after validation, a low-weight auxiliary source.
4. Unfiltered pseudo-videos as ordinary training data: prohibited for the first experiment.

If synthetic edited videos are later introduced, cap their contribution initially at `5-10%`, keep real-data replay dominant, and require generic full-video temporal/action consistency checks. Do not select, filter, or weight them using a lamp-region metric.

### Ablation order

1. Balanced real-data nonlinear bridge only.
2. Add shared-noise relative velocity distillation at high noise.
3. Add the frozen-backbone, zero-initialized bridge residual with exact endpoint gating.
4. Compare Stage 2 medium-noise-only adaptation against near-full-noise identification followed by refinement.
5. Only if the above is insufficient, generate a small FlowEdit-style pseudo-video set and test it as a low-weight auxiliary objective.
6. Treat a paired simulator counterfactual dataset and a true diffusion/flow bridge as a separate, larger experiment.

Primary acceptance criteria remain endpoint rollout preservation, global-to-correct-endpoint latent-gradient direction and magnitude, Stage 2 context recovery, and full-video generation quality. Flow loss alone is not sufficient.

## Global-anchor freeze after the first 300 updates

The first 300 optimization updates form a dedicated global-anchor phase. During
this phase, `c_global` is trained on an exactly balanced mixture of real data
from all four environments. Counterfactual pseudo-videos are not used in this
phase.

Immediately after update 300:

1. Save `c_global` in the checkpoint and context-table artifact.
2. Remove `c_global` from every optimizer and set
   `c_global.requires_grad_(False)`.
3. Never reinitialize, project, average, or otherwise modify its numerical
   value during subsequent bridge training.
4. Save a frozen functional snapshot of the model at `c_global`.

Freezing the coordinate alone does not freeze its behavior: later model or
bridge-adapter updates can still change `v(z_t, t, c_global)`. Therefore later
training must also use balanced real-data replay and a stopped-gradient global
anchor loss against the update-300 snapshot. The coordinate is permanently
fixed, while the functional anchor prevents the model around that coordinate
from drifting.

## Learning button causality without noisy-video leakage

The editing objective must teach the interaction

```text
(initial observation, red/blue action, environment context) -> future lamp dynamics
```

rather than teach the model to continue the lamp appearance already visible in
the noised target video.

### Causal comparison unit

For every selected real chunk, construct four condition branches:

1. The true environment context.
2. Each of the other three endpoint contexts.

All four branches must use the identical real chunk, initial-frame condition,
action sequence, timestep, and noise realization. The only changed input is the
environment context. Each effective batch must balance red-button and
blue-button interactions across all four environments. This paired construction
makes the context-action interaction the only systematic source of a
between-branch velocity difference.

### Separate identification from refinement

The causal identification objective is evaluated only at near-full noise at
first, approximately `0.95-0.995`. Several independently sampled noise
realizations are averaged for each comparison unit. In this regime, the target
video cannot expose a sufficiently clear current lamp state for the model to
take the reconstruction shortcut.

Medium-noise examples must not dominate this identification objective. They
may be added later for trajectory refinement and ordinary endpoint replay, once
the latent direction is already identifiable. Exact full noise can be included
as a small ablation, but it should not be the only timestep because its gradient
variance may be high.

### Teacher-constrained counterfactual branches

The true branch has a real flow-matching target. A wrong-context branch does
not have a real counterfactual target, so it must not be trained by simply
maximizing its loss against the real video. That would permit arbitrarily bad
velocities rather than a valid alternative physical outcome.

Instead:

1. Evaluate the frozen successful endpoint Teacher under all four contexts.
2. Use shared-noise relative velocity differences to define the edit direction.
3. Constrain every wrong-context branch to remain close to its corresponding
   frozen Teacher field.
4. Train the bridge residual to reproduce the relative field direction as the
   latent moves from `c_global` toward an endpoint.

A contrastive statistic based on the four real-target flow losses may be logged
to measure identifiability. It should initially be diagnostic rather than an
unbounded objective that deliberately makes wrong contexts worse. If it later
becomes a training term, use a bounded softmax/margin formulation together with
Teacher-field constraints.

No lamp mask, lamp crop, lamp-state label, ROI loss, or spatial attention bias
is allowed. The causal signal comes from paired action/context comparisons and
high-noise conditioning over the complete video.

## Counterfactual Teacher prerequisite experiment

Before using endpoint-Teacher deltas as bridge supervision, verify that the old
four-context Teacher can perform coherent counterfactual generation.

Experiment definition:

- Teacher: job `90299`, checkpoint `step-3696`.
- Dataset: fixed-close/no-pause LightSwitch data at `frame_stride=3`.
- Sources: 16 real chunks from each environment.
- Per environment: eight spread-out episodes, with one single-blue-button
  chunk and one single-red-button chunk from every selected episode.
- For each source chunk: infer only with the other three environment contexts.
- Comparison: ground truth plus the three wrong-context predictions.
- Randomness: all three context branches for one source use the same generation
  seed.
- Total: 64 source chunks, 192 counterfactual predictions, and 64 four-way
  comparison videos.

This experiment is a gate. If changing only the endpoint context does not
produce coherent, context-specific button/lamp outcomes, the endpoint Teacher
is not a reliable counterfactual editor and its deltas must not yet be used as
bridge supervision.

## Oversample the local neighborhood of the frozen Global context

Let `s` be normalized distance along a Global-to-environment segment:

```text
c_e(s) = c_global + s * (c_e - c_global)
s = 0: Global context
s = 1: environment endpoint
```

Uniform bridge sampling is insufficient. It places only 10% of bridge samples
inside the first 10% of the segment, although Stage 2 starts exactly at
`c_global` and needs a strong, correctly oriented local derivative there.

After the 300-update Global phase, every mixed training update uses the
following quotas over all samples, not conditional percentages inside a
smaller bridge subset:

```text
40%: exact environment endpoints, s = 1
30%: near-Global corridor, 0 < s <= 0.1
30%: remaining bridge interior, 0.1 < s < 1
```

Do not implement these as unconstrained Bernoulli draws for a small update.
Enforce approximately deterministic counts while balancing all four
environments and both button actions. With 16 comparison units per environment,
use approximately `6 endpoint + 5 near-Global + 5 remaining-interior` units and
rotate the residual count across updates to recover the exact long-run
`40/30/30` ratio.

Stratify the 30% near-Global quota further:

```text
10% of all samples: 0.01 <= s < 0.03
10% of all samples: 0.03 <= s < 0.06
10% of all samples: 0.06 <= s <= 0.10
```

Avoid relying on extremely tiny displacements under BF16. Keep the context,
velocity subtraction, and local directional loss in FP32 even if the frozen
Teacher backbone runs in BF16.

### Local paired finite-difference supervision

Oversampling isolated points is necessary but not sufficient. Every
near-Global point must be paired with the frozen `s=0` anchor under the same
real chunk, action, timestep, and noise:

```text
D_student(e, s)
    = [v_student(z_t, c_e(s)) - v_student(z_t, c_global)] / s

D_teacher(e, s)
    = [q(s) / s] * stopgrad(
          v_endpoint_teacher(z_t, c_e)
        - v_global_teacher(z_t, c_global)
      )

L_local = ||D_student(e, s) - D_teacher(e, s)||^2
```

This directly constrains the one-sided derivative that Stage 2 encounters when
it starts from `c_global`. The Global coordinate and Global Teacher branch are
frozen; gradients update only the bridge residual or other explicitly allowed
student parameters.

Near-Global pairs should predominantly use the near-full-noise identification
band so that the derivative represents button/environment causality rather than
continuation of a leaked future lamp appearance. Endpoint replay can retain a
broader ordinary flow-matching timestep distribution to protect generation
quality.

### Nonlinear-curve consistency correction

The earlier expression `q(s) = 1 - (1 - s)^4` reaches only `93.75%` at
`s = 0.5`, which does not satisfy the stated requirement that the target
environment contribution exceed 95% by the segment midpoint. Use:

```text
q(s) = 1 - (1 - s)^5
```

This reaches `96.875%` at the midpoint and already completes approximately
`40.95%` of its change within `s <= 0.1`. Allocating 30% of samples to this
region is therefore not merely heuristic; it compensates for the large target
field variation concentrated near Global.

### Required local-gradient diagnostics

Report diagnostics separately for the three near-Global subranges:

- cosine alignment between the learned latent gradient and each correct
  Global-to-endpoint direction;
- projected gradient magnitude toward the correct endpoint and toward the
  three incorrect endpoints;
- finite-difference support loss change from `s=0` to each sampled `s`;
- variance across noise seeds and button actions;
- correct-context versus wrong-context full-frame flow-loss margin.

These are full-video diagnostics. They must not use lamp-region weighting or
privileged lamp-state supervision.

## Validation experiment: Stage 2 from random starts at maximum noise

Run a controlled experiment on the successful old four-latent Teacher before
changing bridge training.

Protocol:

- Teacher: job `90299`, checkpoint `step-3696`.
- Compute: one H200 in the low-priority partition.
- Initial contexts: three deterministic Uniform(0,1) C32 samples. The same
  three starts are reused for all four environments.
- Evidence: four chunks per environment, shared by one adapted context.
- Chunk coverage: two red-button chunks and two blue-button chunks from four
  distinct episodes, with early, middle, and late windows represented.
- Selection uses only action/event metadata. It does not inspect or weight the
  lamp image region.
- Stage 2 scope: context only, FP32 context updates.
- Objective: ordinary full-frame flow-matching support loss only; context
  regularization is disabled.
- Timestep: fixed training scheduler index `0`. In this pipeline index `0` is
  the scheduler's maximum-noise endpoint; a numerically large array index is
  instead closer to the clean endpoint.
- Inner optimization: 80 steps with schedule
  `3.0:20,1.5:20,0.5:20,0.15:20`, gradient clipping at 1.0, and context clamped
  to `[0,1]`.
- Output: for each initial context and each environment, four
  `GT / oracle Stage1 / adapted Stage2` comparison videos.

This produces 48 comparison videos in total:

```text
3 random starts * 4 environments * 4 chunks = 48
```

Record the complete C trajectory, support loss at every inner step, final
distance to all four endpoint contexts, and a PCA trajectory for every random
start. The experiment tests two claims:

1. Removing future-video appearance leakage with maximum noise makes the
   support loss informative about the environment context.
2. The old Teacher's latent loss field can move different random starts toward
   the correct endpoint using only four balanced action observations.

Failure from all three starts would indicate that high noise alone is
insufficient and that the proposed counterfactual relative-velocity bridge is
needed. Success from only nearby starts would indicate a basin/geometry problem
rather than a timestep-identifiability problem.
