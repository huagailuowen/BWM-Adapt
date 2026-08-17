# Baseline and Ablation Plan

This file is the repository-local mirror of the authoritative ablation plan in the Google Doc, updated on 2026-08-13.

## Baselines

| Method | What it tests | Prior work or claim addressed |
| --- | --- | --- |
| Standard Pooled World Model (No Adaptation) | Whether ordinary pooled world-model training alone is sufficient. | No-adaptation pooled-model baseline |
| Same-Model Mean-Z | Whether test-time Z optimization helps when architecture and checkpoint are held fixed. | Clean no-adaptation control for Ours |
| History-Conditioned WM | Whether raw support trajectories alone enable in-context environment inference. | WAM-ICL, L2World, and Echo-Memory-style raw context |
| LoRA TTA | Whether LoRA test-time adaptation is sufficient when initialized from the normally trained Standard Pooled World Model checkpoint. | Parameter-space test-time adaptation baseline |
| TTT-KQV | Whether long-history implicit fast-weight memory is sufficient. | Test-time training and fast-weight memory |
| DINOv2 Amortized Context Encoder | Whether an explicit amortized encoder matches generator-based latent inference. | Amortized context-inference baseline |
| Frozen WM + Optimized Environment Code | Whether new physical regularities must enter the shared model. | Optimized-context and consolidation control |
| Ours | Full environment-level latent method with iterative training and test-time Z inference. | Reference method |

## Main ablations

| Method | What it tests | Claim addressed |
| --- | --- | --- |
| Shuffled Environment Grouping | Whether correct environment grouping, rather than latent capacity alone, supplies the useful signal. | Environment grouping and sharing claim |
| Per-Trajectory Latent | Whether environment-level sharing produces a reusable physical law. | Environment-level latent claim |
| Joint Model-Latent Training | Whether latent-first alternating optimization is necessary. | Assimilation-consolidation claim |
| Environment-Code Dimension (`C=4/128/1024`, reference `C=32`) | Whether performance comes from environment-level structure or merely latent capacity. | Representation-capacity and bottleneck claim |

## Evaluation

### Retention and Compute-Matched Forward Transfer

This is an evaluation protocol, not a model baseline. Evaluate the same held-out environments across progressive checkpoints, report retention on early environments, and include the compute-matched repeated-small-set control.

## Detailed experimental settings

### Shared protocol

#### Support/query split

- Run both `K = 1` and `K = 2` as co-primary support settings, and evaluate every method under both values of `K`.
- Each adaptation episode uses `K` support trajectories.
- Performance is averaged over a fixed disjoint query set containing multiple query trajectories and actions from the same environment.
- A single training item or generation call may contain one query, but final metrics must never depend on only one query.
- The current evaluation protocol already uses disjoint support and query data. No support episode or chunk is reused as a query. This is an existing anti-leakage property, not a new protocol change.
- Dedicated cross-action evaluation: query action IDs must additionally be absent from the support set.

#### Controlled inputs

- Freeze the support/query split, sample IDs, query actions, initial frames, Wan VAE latents, clip length, resolution, inference sampler, denoising steps, CFG scale, and random seeds across methods.
- All methods use the same Wan base architecture, pretrained initialization, frozen Wan VAE, action interface, data split, and evaluation protocol, except for the method-specific adaptation modules stated below.
- All methods that train or modify a Wan backbone use the same training-environment membership and receive the same fixed hardware-time budget. Data ordering and sampling follow each method's native protocol: the Standard Pooled World Model uses a pooled shuffled loader, while Ours uses its progressive grouped-environment stream. Generator-gradient evaluations and clip exposure are recorded outcomes rather than matching constraints.
- All video-prediction methods use the existing flow-matching objective and sampler. Query performance is always computed on disjoint query futures.

#### Batch shape

- Report every structured outer batch as `E environments per rank x A actions/chunks per environment x G GPUs`.
- Keep `E`, `A`, `G`, and sampled clips fixed inside each controlled comparison.
- Default core setting: `4 environments x 6 common actions per rank x 4 GPUs` when six common actions exist.
- This is the training-time grouped batch and is independent of the test-time support size `K`.
- The six trajectories sharing an environment code are training samples rather than a predefined support/query partition.
- Report task-specific batch settings separately rather than mixing them into the core ablation.

#### Latent and compute controls

- Use one 32-dimensional environment code for all code-based methods unless code dimension is the variable under ablation. The dimension ablation evaluates `C in {4, 32, 128, 1024}`.
- The primary training comparison uses two H200 GPUs for 24 hours per independently trained method. It does not force equal optimizer steps, clip exposure, or FLOPs.
- For every gradient-based method, report trainable parameter count, forward/backward evaluations, clips seen, actual GPU-hours, GPU utilization, wall-clock adaptation time, peak GPU memory, and performance as a function of adaptation compute.

#### Metrics and seeds

- Primary: object-mask mean/final IoU, object-centroid ADE/FDE, and task-specific physical errors.
- Secondary: global PSNR and LPIPS.
- Separate: action, planning, and task-success evaluation.
- Use at least three seeds for final core comparisons. Single-seed runs are screening results only.

### 1. Standard Pooled World Model (No Adaptation)

Starting from the same pretrained Wan checkpoint, pool all permitted training environments and train with the ordinary shuffled loader and normal flow-matching objective. Do not add an environment code, latent-conditioning branch, grouped-environment sampler, alternating latent/model schedule, or support-time adaptation. At evaluation, freeze the trained model, ignore the support set, and generate every disjoint query directly. "No Adaptation" refers to inference; Wan is fully optimized during training.

### 2. Same-Model Mean-Z, No Adaptation

Load the exact checkpoint used by Ours, initialize `Z` to the mean of the active training-time code table, perform zero test-time updates, and generate the same disjoint queries. The backbone, conditioning branch, and checkpoint are identical to Ours; only test-time `Z` optimization is removed.

### 3. History-Conditioned WM

Encode the `K` clean support trajectories with the frozen Wan VAE and place their visual tokens and aligned action tokens before the query tokens. Give each trajectory a separate segment identifier and reset local temporal positions at trajectory boundaries. The noised query future may attend to all support tokens. Do not use an environment encoder, optimized latent, LoRA, or fast-weight update.

Train a separate Wan copy with query-only flow-matching loss using the same disjoint support/query construction as Ours. Randomize support order. At inference, prepend support trajectories and perform one frozen generation pass.

### 4. LoRA Test-Time Adaptation

First train the Standard Pooled World Model (No Adaptation) on all permitted training datasets and environments using its ordinary pooled shuffled loader and normal flow-matching objective, with no environment code or latent-conditioning branch; held-out and test environments remain excluded. Use this exact normally trained checkpoint as the LoRA baseline initialization, rather than loading Ours or any code-conditioned Stage 1 checkpoint.

Insert rank-8 LoRA modules into the Q, K, V, and output projections of its Wan attention blocks. For each test environment, reset LoRA to the same shared zero initialization, freeze the pooled base model, optimize only LoRA on the `K` support clips, and freeze the adapted LoRA before query generation.

Use learning rate `1e-4` unless the validation split selects another value before opening the test set. Keep `K`, support/query data, and the query set fixed, and report LoRA adaptation latency, update count, and peak memory alongside quality because one LoRA update changes far more parameters than one environment-code update.

### 5. TTT-KQV

Reuse the project TTT-KQV block placement. Use the video TTT-MLP fast model with hidden width four times the head dimension, token mini-batch size 64, and base fast learning rate 0.1. Every outer example and every replaced block owns an independent fast state.

Reset fast state to its learned shared initialization for each environment. Process the `K` supports sequentially, carry fast state across supports, and reset temporal positions at trajectory boundaries. During outer training, query tokens read the support-adapted fast state but do not perform additional fast-weight updates. The query flow-matching loss is backpropagated through the support-time write process. At inference, write the support set once, freeze fast state, generate all queries, and reset before the next environment.

### 6. DINOv2 Amortized Context Encoder

Each DINO support trajectory uses exactly the same full-length support window and clip boundaries as the other methods, normally approximately 40/41 frames depending on the dataset's frame/action convention. The eight DINO frames are sparse visual samples across this complete window, not an eight-frame window or crop.

With `T` visual frames, use frozen DINOv2-B/14 and uniformly sample eight strictly increasing frame indices `0=t_1<t_2<...<t_8=T-1`. The action input must represent each complete interval between adjacent sampled frames, not a single instantaneous action. For `j=1,...,7`, compute:

```text
u_j = ActionEncoder(a[t_j:t_{j+1}-1])
```

The seven chunks collectively cover the complete action-transition sequence between the first and last frame. Feed the eight visual CLS features together with the seven interval action-chunk embeddings to an action-conditioned projection. Map each trajectory to a 32-dimensional code with a two-layer projection head of hidden width 1024, then average the `K` trajectory codes to form one permutation-invariant environment code.

Initialize the Wan generator from the same pretrained Wan checkpoint as Ours and inject the code through the exact conditioning interface used by Ours. Follow the identical progressive environment stream and old/new environment sampler under the same two-H200, 24-hour training envelope. Train the action/projection head and Wan generator with disjoint-query flow-matching loss, and report the resulting model-gradient updates and clip exposure.

The intended core difference is only `Z = q_phi(C_E)`, inferred amortized from support context, rather than `Z` obtained by optimizing the generation-model loss. At inference, infer `Z` in one encoder forward pass with no gradient update. This is the DINOv2 Amortized Context Encoder baseline, following the video-context design of Implicit State Estimation via Video Replanning. EVF is cited only as an earlier pixel-generative precedent, not as the implementation reproduced here.

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

At test time, freeze Wan and initialize one fresh FP32 `Z` as `Z0`, the mean of active training-time codes. Optimize only `Z` for 40 steps with explicit gradient descent without momentum. This is not Adam or AdamW, so `beta_1` and `beta_2` do not apply.

The inner objective and update are:

```text
L_inner = mean_k L_FM(support_k; Z) + 1e-3 * mean((Z - Z0)^2)
g = clip_by_global_l2_norm(grad_Z L_inner, max_norm=1.0)
Z <- Z - eta_t * g
```

The canonical configuration leaves `stage2_context_clamp_min` and `stage2_context_clamp_max` unset. It therefore applies no elementwise or norm clamp to `Z` beyond gradient clipping.

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

The canonical alternating cycle contains 600 `Z` gradient steps and 400 model gradient steps, while this ablation updates both jointly. Give both methods the same fixed two-H200, 24-hour training envelope and report their resulting model/code updates, forward/backward evaluations, clips seen, and throughput rather than claiming equality from nominal step counts.

### 12. Environment-Code Dimension

Use the full Ours implementation and vary only the environment-code dimension:

| Run | Dimension |
| --- | ---: |
| Bottleneck | 4 |
| Reference | 32 |
| Medium-capacity | 128 |
| High-capacity | 1024 |

Initialize every environment code independently from `Uniform(0, 1)` in every
dimension. Keep the Event80 data, `65-105` window, grouped batch, progressive
environment order, 1,000-step alternating training cycle, Wan initialization,
optimizer settings, fixed 48 H200-GPU-hour budget, checkpoint policy, and frozen
K=1/K=2 support/query evaluation identical. Report primary object-centric and
physical metrics together with parameter count, code-table size, peak memory,
throughput, and actual optimization steps. This ablation distinguishes the
benefit of environment-level sharing from gains caused only by a larger latent.

### 11. Frozen WM + Optimized Environment Code

Train the initial code-conditioned Wan model normally, then freeze shared Wan parameters before later environment batches arrive. Learn only new and active environment codes using the same initialization, data, optimizer, and code-update budget as Ours. For held-out environments, use the identical `K`-support, 40-step `Z` inference protocol. This isolates whether new physical regularities must be consolidated into the shared model.

### Evaluation metrics and task-level protocols

#### Metric scope and aggregation

- Evaluate predicted future frames only; exclude conditioning/history frames.
- Aggregate frame- and object-level values within each query, then macro-average queries within each environment, environments within each split, and random seeds. Every environment receives equal weight.
- Report `K=1` and `K=2`, ID and OOD, and Stage1, mean-Z, and adapted-Z results separately.
- Report 95% bootstrap confidence intervals over environments and seeds.
- Save every per-query value together with a fixed evaluation manifest.

#### Primary object-centric world-model metrics

- Define the evaluator before opening the test set and never tune it per method.
- Simulation: use renderer instance masks for ground truth and one fixed object segmenter/tracker for generated RGB.
- Real data: use one fixed tracker initialized from an identical first-frame prompt and manually annotate an audit subset.
- Mean IoU: average target-object mask IoU over valid predicted future frames, then apply the macro-aggregation above.
- Final IoU: target-object mask IoU at the final valid task frame.
- Centroid ADE: mean Euclidean target-center error over valid future frames.
- Centroid FDE: target-center error at the final valid task frame.
- Report physical units when calibration exists; otherwise divide centroid errors by image diagonal.
- If the predicted object is missing while ground truth is visible, assign `IoU=0`, assign centroid error equal to one image diagonal, and report missing-mask rate.
- Exclude ground-truth out-of-view or fully occluded frames only through a predeclared visibility rule. Report out-of-bounds events separately.

#### Secondary global appearance metrics

- Compute PSNR independently on each predicted future RGB frame in `[0,1]`.
- Compute LPIPS with one fixed published backbone and preprocessing pipeline at the same evaluation resolution.
- Macro-average per frame, query, environment, and seed.
- Static backgrounds can dominate both metrics, so they remain secondary.

#### Separate action, planning, and task-success evaluation

- After adapting on `K` support trajectories, give every method the same target, candidate action set or continuous-action optimization budget, rollout count, horizon, and random seeds.
- Select an action using predicted rollouts, then execute that selected action once in the ground-truth simulator or real system. Do not score success from the model's predicted video.
- Report task success, final task cost, normalized regret against the best candidate action, action cost, and safety failures.
- Offline first version: select among a fixed held-out bank of recorded actions and report top-1 success and normalized regret.
- Select target regions, tolerances, rest-speed thresholds, and all success rules on validation data, then freeze them before test.

#### Per-experiment evaluation design

| Experiment | Object-centric rollout metrics | Task-specific physical metrics | Action/planning task |
| --- | --- | --- | --- |
| PushBox friction | Block mask mean/final IoU and centroid ADE/FDE. | Sliding-distance, stopping-time, final displacement, terminal-speed, and overshoot errors. | Choose push direction, magnitude, and duration to stop the block in a target region. Success requires the center to remain inside the region and below the frozen rest-speed threshold over the final five evaluated frames. |
| Multi-background PushBox | PushBox metrics stratified by background, plus matched-friction/action performance variance across backgrounds. | Standard PushBox errors plus the cross-background invariance gap. | Run the same target-region task on seen and unseen backgrounds; report macro-average and worst-background success. |
| Gravity | Object mask IoU and centroid trajectory ADE/FDE. | Landing-point, time-to-impact, vertical-trajectory, and terminal-velocity errors. | Choose release or launch action to land the object in a target region; report landing success and target-distance regret. |
| Mass collision | Separate mask IoU and centroid ADE/FDE for striker and target objects. | Collision-time, post-collision velocity, target displacement, travel-distance, and stopping-time errors. | Choose impact speed and direction so the struck object stops in a target region. |
| Mass balance | Dumbbell/bar mask IoU, endpoint/keypoint ADE/FDE, center trajectory, and orientation error. | Final angle, angular-velocity, settling-time, and tip/fall errors. | Choose support/contact position or placement action to keep the object within a frozen angle tolerance for the final five frames. |
| Joint mass-friction | Push/collision object metrics reported across the two-factor environment grid. | Sliding/stopping and collision/post-impact errors, stratified by mass, friction, and their interaction. | Choose a push or impact action to reach a target region on held-out mass-friction combinations; report per-cell, macro, and worst-group success. |
| PnP payload dynamics | Payload mask/keypoint IoU, centroid/pose ADE/FDE, and gripper-payload relative pose error. | Motion lag, oscillation amplitude, settling time, drop rate, collision rate, and final pose error. | Choose an EEF speed/path profile that places the payload in a target pose without dropping it or exceeding the oscillation limit. |
| LightSwitch causal dynamics | Switch/button and EEF keypoint trajectory ADE/FDE and final switch-state accuracy. | Contact-time, toggle-time, state-transition, and unintended-toggle errors. | Choose contact point, direction, and force/trajectory to reach the requested switch state without toggling other controls. |
| Real slope-friction | Block/ball mask IoU and centroid ADE/FDE. | Travel-distance, stopping-time, terminal-speed, and across-surface ordering errors. | Choose release or push amplitude to stop the object in a target region; report success, target distance, overshoot, and safety failures. |

### Evaluation: Retention and Compute-Matched Forward Transfer

Save checkpoints after 25%, 50%, 75%, and 100% of training environments have been introduced. At every checkpoint, infer a fresh `Z` for the same held-out environments using identical support clips, initialization, and 40-step schedule. Report zero-step and adapted query metrics, the complete adaptation curve, and steps needed to reach fixed object-IoU thresholds.

At the same checkpoints, evaluate a fixed panel of early training environments using both their saved training-time `Z` and a freshly inferred `Z`. Report average retention and the drop from each environment's historical best.

Add a compute-matched repeated-small-set control that performs the same number of model updates and sees the same number of clips while repeatedly sampling only the initial environment subset. Compare it with progressive-environment training to separate gains from broader physical experience from gains caused only by longer training.
