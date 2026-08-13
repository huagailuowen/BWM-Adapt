# Baseline and Ablation Plan

This file is the repository-local mirror of the authoritative ablation plan in the Google Doc, updated on 2026-08-13.

## Baseline and ablation matrix

| Method | What it tests | Prior work or claim addressed |
| --- | --- | --- |
| Frozen Global WM | Whether environment conditioning and adaptation are necessary at all. | No-adaptation control |
| Same-Model Mean-Z, No Adaptation | Whether test-time Z optimization helps when architecture and checkpoint are held fixed. | Clean no-adaptation control for Ours |
| History-Conditioned WM | Whether raw support trajectories alone enable in-context environment inference. | WAM-ICL, L2World, and Echo-Memory-style raw context |
| LoRA Test-Time Adaptation | Whether lightweight weight updates are sufficient. | AdaJEPA and AdaWorldPolicy-style parameter adaptation |
| TTT-KQV | Whether long-history implicit fast-weight memory is sufficient. | Test-time training and fast-weight memory |
| DINOv2 Amortized Context Encoder | Whether an explicit amortized encoder matches generator-based latent inference. | DINOv2 context-conditioned dynamics; not claimed as a full EVF reproduction |
| Ours | Full environment-level latent method with iterative training and test-time Z inference. | Reference method |
| Per-Trajectory Latent | Whether environment-level sharing produces a reusable physical law. | Environment-level latent claim |
| Shuffled Environment Grouping | Whether correct environment grouping, rather than latent capacity alone, supplies the useful signal. | Environment grouping and sharing claim |
| Joint Model-Latent Training | Whether latent-first alternating optimization is necessary. | Assimilation-consolidation claim |
| Frozen WM + Optimized Environment Code | Whether new physical regularities must enter the shared model. | CoDA-style context adaptation and consolidation claim |
| Retention and Compute-Matched Forward Transfer | Whether broader environment experience improves held-out adaptation without forgetting earlier environments, beyond longer training alone. | Continual physical knowledge, retention, and compute-control claims |

## Detailed experimental settings

### Shared protocol

#### Support/query split

- Main comparison: `K = 2` support trajectories and one query trajectory.
- One-shot result: additionally report `K = 1`.
- The current evaluation protocol already uses disjoint support and query data. No support episode or chunk is reused as a query. This is an existing anti-leakage property, not a new protocol change.
- Dedicated cross-action evaluation: query action IDs must additionally be absent from the support set.

#### Controlled inputs

- Freeze the support/query split, sample IDs, query actions, initial frames, Wan VAE latents, clip length, resolution, inference sampler, denoising steps, CFG scale, and random seeds across methods.
- Use the same pretrained Wan backbone, frozen Wan VAE, action encoder, action-conditioning path, and environment split unless a baseline explicitly requires a separately trained architecture.
- All video-prediction methods use the existing flow-matching objective and sampler. Query performance is always computed on disjoint query futures.

#### Batch shape

- Report every structured outer batch as `E environments per rank x A actions/chunks per environment x G GPUs`.
- Keep `E`, `A`, `G`, and sampled clips fixed inside each controlled comparison.
- Default core setting: `4 environments x 6 common actions per rank x 4 GPUs` when six common actions exist.
- Report task-specific batch settings separately rather than mixing them into the core ablation.

#### Latent and compute controls

- Use one 32-dimensional environment code for all code-based methods unless code dimension is the variable under ablation.
- Report both equal-step and compute-aware comparisons.
- For every gradient-based method, report trainable parameter count, forward/backward evaluations, wall-clock adaptation time, peak GPU memory, and performance as a function of adaptation compute.

#### Metrics and seeds

- Co-primary world-model metrics: object-mask mean IoU and final-frame IoU.
- Motion metrics: object-centroid ADE and FDE.
- Task-specific physical metrics: sliding distance, stopping time, collision displacement, landing point, balance angle, or the relevant equivalent.
- Secondary appearance metrics: global PSNR and LPIPS.
- Report downstream policy or planning performance separately from video-prediction metrics.
- Use at least three seeds for final core comparisons. Single-seed runs are screening results only.

### 1. Frozen Global WM

Train a pooled Wan world model with one global or null environment condition and the standard flow-matching objective. At inference, ignore the support set and directly generate the disjoint query. This tests whether environment conditioning and adaptation are needed, but it is not the cleanest test of test-time optimization because its trained architecture differs from Ours.

### 2. Same-Model Mean-Z, No Adaptation

Load the exact checkpoint used by Ours, initialize `Z` to the mean of the active training-time code table, perform zero test-time updates, and generate the same disjoint queries. The backbone, conditioning branch, and checkpoint are identical to Ours; only test-time `Z` optimization is removed.

### 3. History-Conditioned WM

Encode the `K` clean support trajectories with the frozen Wan VAE and place their visual tokens and aligned action tokens before the query tokens. Give each trajectory a separate segment identifier and reset local temporal positions at trajectory boundaries. The noised query future may attend to all support tokens. Do not use an environment encoder, optimized latent, LoRA, or fast-weight update.

Train a separate Wan copy with query-only flow-matching loss using the same disjoint support/query construction as Ours. Randomize support order. At inference, prepend support trajectories and perform one frozen generation pass.

### 4. LoRA Test-Time Adaptation

Insert rank-8 LoRA modules into the Q, K, V, and output projections of Wan attention blocks. Start every test environment from the same shared zero LoRA initialization, freeze the base model, optimize only LoRA on the `K` support clips, and freeze the adapted LoRA before query generation.

Use the same 40-step support-adaptation budget as Ours for the primary step-matched result. Use learning rate `1e-4` unless one validation split selects another value before opening the test set. Also report a compute-matched curve because one LoRA update changes far more parameters than one 32-dimensional `Z` update.

### 5. TTT-KQV

Reuse the project TTT-KQV block placement. Use the video TTT-MLP fast model with hidden width four times the head dimension, token mini-batch size 64, and base fast learning rate 0.1. Every outer example and every replaced block owns an independent fast state.

Reset fast state to its learned shared initialization for each environment. Process the `K` supports sequentially, carry fast state across supports, and reset temporal positions at trajectory boundaries. Optimize slow parameters with the disjoint-query flow-matching loss. At inference, write the support set once, freeze fast state, generate all queries, and reset before the next environment.

### 6. DINOv2 Amortized Context Encoder

Use frozen DINOv2-B/14 as the visual encoder. Uniformly sample eight frames from each support trajectory. Concatenate per-frame CLS features with aligned actions encoded by a two-layer MLP of width 64. Map each trajectory to a 32-dimensional code with a two-layer projection head of hidden width 1024. Average the `K` trajectory codes to form one permutation-invariant environment code.

Inject the code through the exact conditioning interface used by Ours. Train the action/projection head and Wan generator with disjoint-query flow-matching loss. At inference, infer `Z` in one encoder forward pass with no gradient update. Describe this as a DINOv2 amortized-context baseline, not a full EVF reproduction.

### 7. Ours

Assign one learnable `Z32` to each training environment and share it across all trajectories, states, and actions from that environment. Initialize training-environment codes independently from `U(0,1)`. Inject `Z` through the current physical-context token and modulation path.

For the initial environment set, keep `Z` fixed and train the model for 300 structured steps. For each newly introduced environment batch, use the canonical 1,000-step cycle:

| Phase | Steps | Trainable parameters | Learning rate |
| --- | ---: | --- | ---: |
| New-Z only | 200 | Newly added environment codes | 0.15 |
| All-Z | 200 | All active environment codes | 0.03 |
| Model only | 200 | Shared model | `1e-5` |
| All-Z | 200 | All active environment codes | 0.03 |
| Model only | 200 | Shared model | `1e-5` |

Parameters outside the named scope are strictly frozen. Core ablations use this same cycle. Stable task-specific overrides, such as new-Z learning rate 0.09 for joint mass-friction, must be reported separately and must not be mixed into a controlled comparison.

At test time, freeze Wan and initialize one fresh `Z` from the mean of active training-time codes. Average support flow-matching loss over the `K` disjoint supports and optimize only `Z` in FP32 for 40 steps:

| Inner steps | Z learning rate |
| --- | ---: |
| 1-10 | 3.0 |
| 11-20 | 1.5 |
| 21-30 | 0.5 |
| 31-40 | 0.15 |

Freeze the inferred `Z` before generating every disjoint query from that environment.

### 8. Per-Trajectory Latent

Keep the same backbone, code dimension, losses, data, initialization distribution, alternating schedule, and sampled clips as Ours, but assign an independent `Z` to every training trajectory instead of sharing one within an environment. At test time, initialize one fresh `Z`, optimize it jointly on the same `K` supports, and evaluate disjoint queries. Report the larger training-time latent-table size explicitly.

### 9. Shuffled Environment Grouping

Keep the number of groups, trajectories per group, code dimension, data, update schedule, and optimization budget equal to Ours, but randomly permute trajectory-to-environment assignments while preserving group sizes. Use one fixed permutation per seed and leave the test protocol unchanged. This directly tests whether correct environment grouping supplies the useful learning signal.

### 10. Joint Model-Latent Training

Use the same architecture, codes, data stream, initialization, and flow-matching loss as Ours, but remove latent warm-up and frozen alternating blocks and update Wan and active codes jointly.

The canonical cycle contains 600 `Z` gradient steps and 400 model gradient steps, so one joint run cannot simultaneously match optimizer-step counts and total backward compute. Report two controls: one matched by model-gradient evaluations and one matched by total forward/backward compute. Compare curves rather than claiming fairness from total step count alone.

### 11. Frozen WM + Optimized Environment Code

Train the initial code-conditioned Wan model normally, then freeze shared Wan parameters before later environment batches arrive. Learn only new and active environment codes using the same initialization, data, optimizer, and code-update budget as Ours. For held-out environments, use the identical `K`-support, 40-step `Z` inference protocol. This isolates whether new physical regularities must be consolidated into the shared model.

### 12. Retention and Compute-Matched Forward Transfer

Save checkpoints after 25%, 50%, 75%, and 100% of training environments have been introduced. At every checkpoint, infer a fresh `Z` for the same held-out environments using identical support clips, initialization, and 40-step schedule. Report zero-step and adapted query metrics, the complete adaptation curve, and steps needed to reach fixed object-IoU thresholds.

At the same checkpoints, evaluate a fixed panel of early training environments using both their saved training-time `Z` and a freshly inferred `Z`. Report average retention and the drop from each environment's historical best.

Add a compute-matched repeated-small-set control that performs the same number of model updates and sees the same number of clips while repeatedly sampling only the initial environment subset. Compare it with progressive-environment training to separate gains from broader physical experience from gains caused only by longer training.
