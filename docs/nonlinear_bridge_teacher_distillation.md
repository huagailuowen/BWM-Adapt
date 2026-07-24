# Nonlinear Global-to-Endpoint Bridge with Teacher Distillation

## Status

Design proposal only. This document does not describe an active training run.

## Motivation

The LightSwitch experiment starts from a successful four-environment Stage 1
model. Each environment has a learned endpoint context:

```text
C_A, C_B, C_C, C_D
```

A later bridge experiment added a trainable global context and trained linear
interpolations between the global context and each endpoint. Although the
endpoint context vectors were frozen, the DiT, action encoder, and physical
context encoder continued to update. Evaluation showed that the original
endpoint behavior regressed.

The control experiment compared the same eight red-button and blue-button
chunks using:

```text
original four-context checkpoint: job 90299, step 3696
bridge checkpoint: job 91927, step 1500
```

The original checkpoint retained the expected endpoint behavior, while the
bridge checkpoint produced incorrect lamp transitions in cases such as the
red-button query from the blue-only environment. Therefore, freezing endpoint
context coordinates is not sufficient to preserve the function represented at
those coordinates.

The revised method has three goals:

1. Make the environment identity become decisive soon after leaving the global
   context.
2. Cover multiple denoising regimes in every optimizer update without
   confounding environment comparisons.
3. Preserve the original endpoint vector fields while explicitly teaching
   correction from structured wrong-environment predictions.

## 1. Nonlinear Environment Distribution Along a Linear Latent Segment

For target environment `e`, retain a simple linear latent segment:

```text
C_e(s) = (1 - s) C_global + s C_e
s in [0, 1]
```

Do not use `s` directly as the target-environment probability. Instead define:

```text
q(s) = 1 - (1 - s)^4
```

The environment weights are:

```text
w_e(s) = 1/4 + 3/4 q(s)
w_j(s) = (1 - q(s))/4, j != e
```

This separates latent-space position from the supervised data distribution.
The latent remains halfway between the global context and an endpoint at
`s=0.5`, but the target environment already supplies more than 95 percent of
the training distribution.

| `s` | Target environment weight | Each other environment |
| ---: | ---: | ---: |
| 0.00 | 0.2500 | 0.2500 |
| 0.10 | 0.5079 | 0.1640 |
| 0.25 | 0.7627 | 0.0791 |
| 0.50 | 0.9531 | 0.0156 |
| 0.75 | 0.9971 | 0.0010 |
| 1.00 | 1.0000 | 0.0000 |

The exponent should be configurable, with `4` as the initial value. A larger
exponent makes the environment identity become decisive even closer to the
global point.

## 2. Stratified Multi-Timestep Updates

The previous bridge implementation used one shared diffusion timestep for all
64 chunks in an optimizer update. This makes cross-environment comparisons
clean, but each update covers only one denoising regime.

Fully independent timesteps per chunk are also undesirable because
environment losses would again be confounded by different denoising
difficulty.

Use `K` stratified timestep blocks instead:

```text
global batch size: 64
environments: A, B, C, D
K: 4 initially

per timestep block:
    4 chunks from A
    4 chunks from B
    4 chunks from C
    4 chunks from D
    total: 16

four timestep blocks:
    4 * 16 = 64 chunks
```

Sample one timestep from each of four scheduler regions:

```text
high noise
medium-high noise
medium-low noise
low-noise refinement
```

Within a block, all environments share the same timestep. Across blocks, the
update covers multiple noise scales. Gaussian noise remains independent per
chunk unless an explicit common-random-number comparison is required.

Teacher and student evaluations for the same loss term must use the same
timestep and Gaussian noise.

Cross-environment correction should initially be restricted to medium and high
noise. Low-noise updates should preserve texture and fine details through
ordinary flow matching and endpoint distillation.

## 3. Frozen-Teacher Distillation

Use a frozen snapshot of the successful original model:

```text
teacher: job 90299, step 3696, frozen
student: initialized from the same checkpoint
```

This is snapshot teacher-student distillation even though the teacher and
student initially share the same weights.

### 3.1 Cached Teacher Rollout Bank

For every selected initial-frame and action chunk `i`, generate
counterfactual rollouts under all four endpoint contexts:

```text
z_i,A = Teacher(initial_i, action_i, C_A)
z_i,B = Teacher(initial_i, action_i, C_B)
z_i,C = Teacher(initial_i, action_i, C_C)
z_i,D = Teacher(initial_i, action_i, C_D)
```

Cache VAE video latents rather than encoded MP4 files. For 64 base chunks this
requires:

```text
64 chunks * 4 endpoint contexts = 256 teacher rollout latents
```

The bank is generated once and reused for all bridge positions. Points such as
`A70`, `A60`, `B60`, and `B70` change only the student context and mixture
weights; they do not require new teacher generation.

Multiple teacher seeds may be cached for each pair when feasible. A single
teacher sample risks suppressing generation diversity and overfitting to
teacher artifacts.

For the native environment of a real training chunk, the ground-truth video
can supplement or replace the generated teacher rollout.

### 3.2 Mixture Flow-Matching Distillation

Suppose a bridge point represents:

```text
70% A + 10% B + 10% C + 10% D
```

The correct objective is a weighted sum of separate flow-matching losses:

```text
L_mix =
    0.7 L_FM(Student at C_bridge, z_A)
  + 0.1 L_FM(Student at C_bridge, z_B)
  + 0.1 L_FM(Student at C_bridge, z_C)
  + 0.1 L_FM(Student at C_bridge, z_D)
```

Each component must construct its own noised latent and flow target. Do not
average teacher videos, clean latents, or velocity targets before computing
MSE. Averaging targets would train a blurred mean trajectory rather than the
desired mixture distribution.

The cached teacher bank removes repeated teacher generation, but different
bridge contexts still require distinct student forwards. These forwards can be
concatenated along the batch dimension or processed as microbatches with
gradient accumulation.

### 3.3 Endpoint Vector-Field Preservation

Add a direct distillation loss at every endpoint:

```text
L_endpoint =
    ||v_student(x_t, t, C_e)
      - stopgrad(v_teacher(x_t, t, C_e))||^2
```

Teacher and student receive:

```text
the same clean latent
the same Gaussian noise
the same timestep
the same endpoint context
```

This constrains the actual denoising vector field at each endpoint. It is
stronger than ordinary endpoint flow-matching replay, which can lower its
training loss while still changing rollout behavior.

The endpoint contexts remain frozen. The frozen teacher guarantees that
updates to the student DiT, action encoder, or context encoder cannot silently
redefine the endpoint behavior without a penalty.

### 3.4 Cross-Environment Corrective Flow

Mixture distillation describes the desired distribution at a bridge context.
A separate corrective auxiliary teaches recovery from a structured
wrong-environment state.

For target environment `A` and donor environment `B`, define:

```text
r_B = (1 - beta) epsilon + beta z_B
x_t = (1 - t) z_A + t r_B
v_target = r_B - z_A
```

This defines a consistent linear flow from a noisy, B-structured source to the
A teacher target. It teaches the model to remove an incorrect environment
pattern rather than only denoise unstructured Gaussian noise.

Do not feed a B-derived noisy latent while retaining the ordinary
A-to-Gaussian flow target. That input-target pair is mathematically
inconsistent.

Initial settings:

```text
beta: 0.3 to 0.7
noise range: medium and high only
correction-loss fraction: 5% to 10%
```

The donor environment and Gaussian noise should vary independently across
chunks. The target and donor rollouts come from the cached teacher bank.

## 4. Combined Objective

The initial objective is:

```text
L_total =
    L_nonlinear_bridge_mixture
  + lambda_endpoint L_endpoint
  + lambda_correct L_cross_environment_correction
  + lambda_gt L_native_ground_truth
```

Suggested starting weights:

```text
lambda_endpoint = 1.0
lambda_correct = 0.05 to 0.10
lambda_gt = 0.5
```

Interpretation:

```text
L_nonlinear_bridge_mixture:
    creates a rapidly specializing path from global to endpoint

L_endpoint:
    preserves the original endpoint denoising vector field

L_cross_environment_correction:
    teaches recovery from structured wrong-environment predictions

L_native_ground_truth:
    prevents the student from inheriting every teacher generation error
```

The bridge exponent, loss weights, number of timestep strata, and corrective
source strength must be explicit configuration values.

## 5. Efficient Update Construction

A logical update begins with an environment-balanced set of real chunks:

```text
A: 16 chunks
B: 16 chunks
C: 16 chunks
D: 16 chunks
```

The associated teacher rollout bank is prepared once. An update then samples
multiple bridge rays and positions, for example:

```text
A at s=0.25
A at s=0.50
B at s=0.25
B at s=0.50
```

All positions reuse the same cached endpoint rollout bank. Student forwards
are grouped by context and timestep stratum. Memory-limited execution should
use microbatch accumulation while preserving the exact weighted logical
objective.

Detailed logs must include:

```text
bridge target environment
latent position s
nonlinear mixture weights
timestep stratum and exact timestep
per-environment mixture FM loss
endpoint teacher-distillation loss
cross-environment correction loss
native ground-truth loss
student-to-teacher endpoint prediction error
global and model gradient norms
```

## 6. Required Ablations

Before a long run, compare:

| Variant | Nonlinear weights | Multi-timestep | Endpoint teacher | Corrective flow |
| --- | --- | --- | --- | --- |
| Baseline | no | no | no | no |
| A | yes | yes | no | no |
| B | yes | yes | yes | no |
| C | yes | yes | yes | yes |

Every checkpoint must first pass endpoint regression evaluation on all four
LightSwitch environments, with separate red-button and blue-button chunks.
Bridge and Stage 2 quality should only be evaluated after endpoint preservation
is confirmed.

The primary endpoint gate is:

```text
GT vs original Stage 1 vs current Stage 1
```

The primary adaptation evaluation is:

```text
Stage 2 starts from the saved global context
balanced support evidence covers both buttons
trajectory PCA records movement toward the correct endpoint
query rollouts are evaluated for both button actions
```

## 7. Safety and Compatibility

All new behavior must be opt-in. Existing Stage 1, curriculum, grouped-context,
and Stage 2 workflows must retain their current behavior when the new
configuration fields are absent.

Teacher parameters are always frozen. Teacher rollout caches and generated
artifacts must live outside the source tree and must not be committed.

Endpoint context tables and their source checkpoints are immutable inputs.
Every experimental run uses a unique output directory and saves the context
table together with each model checkpoint.
