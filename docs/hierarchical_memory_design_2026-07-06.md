# Hierarchical Memory Design for Push-Box Physical World Modeling

Date: 2026-07-06

## Motivation

The current latent-C experiments mostly use one physical context vector to represent an environment-level property such as table friction. This is useful, but it is too coarse for a full world model.

A real rollout depends on multiple levels of hidden information:

- Long-term environment properties, such as friction, object mass, contact model, and surface behavior.
- Episode-specific information, such as initial object position, exact pose, action bias, and scene-specific details.
- Time-local state information, such as object position, velocity, contact phase, and instantaneous sliding state.

The memory system should therefore not be a single latent vector. It should be hierarchical, with different scopes and update rules.

## Proposed Memory Hierarchy

### 1. Environment latent

Scope: shared across many episodes from the same environment.

Examples:

- Table friction.
- Object-surface contact dynamics.
- Object mass or effective inertia.
- Persistent scene or simulator parameters.

Current grouped-C experiments are an early version of this layer. All chunks and episodes with the same `friction_mu` share one learnable latent C.

Expected behavior:

- Stable across episodes.
- Adapted by observing multiple support episodes.
- Should encode slow physical variables.

### 2. Episode latent

Scope: shared across chunks from the same episode only.

Examples:

- Initial object position.
- Initial object orientation.
- Episode-specific action offset.
- Small scene variation.
- Hidden state that is constant during one rollout but not shared by other episodes.

Expected behavior:

- Different episodes under the same friction should have different episode latents.
- The model should not be forced to store episode-specific state in the environment latent.
- This helps separate physical properties from one-off initial conditions.

### 3. Temporal/state latent

Scope: tied to a time index or a short time window inside an episode.

Examples:

- Object position.
- Object velocity.
- Contact state.
- Sliding phase.
- Local trajectory information.

Expected behavior:

- Overlapping chunks from the same episode should share state latents for overlapping timestamps.
- State latents should carry time encoding so the DiT can align them with video tokens.
- This layer should be useful for representing dynamic memory instead of forcing the model to infer all state from a single first frame.

## Conditioning Format

The DiT can receive a set of memory tokens through the same physical-context cross-attention path:

```text
memory tokens = [
  env tokens,
  episode tokens,
  temporal state tokens
]
```

Each memory token should include:

- Latent vector.
- Type embedding: environment, episode, or state.
- Optional time embedding.
- Optional scope identifier during training.

The DiT backbone does not need explicit semantic rules. It can learn through cross-attention which memory tokens are useful for each video token.

## Time Alignment

Temporal state tokens should be aligned with video token time.

Possible designs:

- Additive learned time embeddings.
- RoPE-style time encoding.
- Relative time offsets between state tokens and DiT frame tokens.

For overlapping chunks, the same absolute episode timestamp should map to the same state latent. This gives the model a consistent memory object across chunk boundaries.

## Stage 1: Table-Probe Version

The first implementation should be deliberately simple and diagnostic.

Use learnable tables:

- `env_table[friction_mu]`
- `episode_table[episode_index]`
- `state_table[episode_index, time_bin]`

Training uses the same diffusion loss, but the physical-context input is assembled from all relevant memory tokens.

This version is not meant to be the final deployable method. Its purpose is to test whether the DiT actually uses hierarchical memory tokens when they are available.

Suggested initial dimensions:

- `env_dim`: 1 or 8.
- `episode_dim`: 8.
- `state_dim`: 8.
- `env_tokens`: 1.
- `episode_tokens`: 1.
- `state_tokens`: 8 per chunk.

## Stage 2: Test-Time Adaptation

In the realistic setting, true friction and hidden physical properties are not directly given.

Recommended adaptation hierarchy:

- Adapt environment latent through support episodes.
- Infer or adapt episode latent per query episode.
- Treat temporal state latent as either inferred from history or adapted only with strong regularization.

The safest first stage2 experiment:

- Initialize env latent randomly or from a learned default.
- Adapt only env latent in the inner loop.
- Keep episode/state latents fixed, inferred, or disabled.

Next experiment:

- Adapt env latent plus episode latent.
- Do not adapt state latent yet, to avoid overfitting support frames.

## Identifiability Risks

If all latent tables are free, the model may put information into the wrong level.

Examples:

- Friction information may leak into episode latent.
- Initial position may leak into env latent.
- State latent may memorize pixel details instead of physical state.

The main defense is sharing scope:

- Environment latent is shared across many episodes.
- Episode latent is shared only within one episode.
- State latent is shared only across overlapping chunks at the same timestamp.

This scope structure forces the model to store information at the correct abstraction level.

## Diagnostics

Useful checks:

- Whether env latent correlates with true friction.
- Whether episode latent clusters by episode but not by friction.
- Whether state latent changes smoothly over time.
- Whether cross-attention from video tokens attends to env, episode, and state tokens at different layers.
- Whether rollout quality improves more than diffusion loss suggests.

Important rollout metrics:

- Final object displacement error.
- Final XY error.
- Forward distance error.
- Trajectory shape error.
- GT versus stage1 versus stage2 video comparison.

## Relationship to Current Experiments

Current grouped-C stage1 corresponds only to the environment latent layer:

```text
env_table[friction_mu] -> physical context tokens
```

The next design extends this to:

```text
env_table[friction_mu]
+ episode_table[episode_index]
+ state_table[episode_index, time_bin]
-> hierarchical memory tokens
-> DiT cross-attention
```

This should make the memory mechanism more expressive and closer to a real physical world model.

