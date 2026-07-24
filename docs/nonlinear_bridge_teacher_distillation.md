# Environment-Latent Identification Without Endpoint Regression

## Status

This document is the current design record for the environment-latent bridge
and Stage 2 test-time adaptation work. It records the motivation, confirmed
failure modes, proposed core method, optional augmentation, and unresolved
questions. It does not describe an active training run.

All future changes to this design should be reflected here before or together
with implementation changes.

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
