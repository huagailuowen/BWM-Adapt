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
| New-Code Warm-Up + Joint Training | Whether pure joint training fails because newly introduced codes receive no isolated alignment phase. Each wave uses 200 new-code-only steps followed by 800 joint steps. | New-environment code alignment claim |
| All-Active Joint Training (No Curriculum) | Whether progressive environment activation is necessary when all 35 training environments and their codes are optimized jointly from step 1. | Curriculum-learning claim |
| Environment-Code Representation (`C=4/128`, reference `C=32`, plus a 3072-D direct token) | Whether performance comes from a compact environment-level representation, latent capacity, or the projection MLP. | Representation-capacity, bottleneck, and projection claim |

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

- Use one 32-dimensional environment code for all code-based methods unless the environment representation is the ablation variable. The projected-code comparison evaluates `C in {4, 32, 128}`. A separate direct-token run replaces the projection MLP with one model-width 3072-D environment token.
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
| Mass balance | Bar centroid ADE/FDE plus the tilt of one fixed longitudinal bar edge, measured modulo 180 degrees. | Mean/final bar-tilt error, angular-velocity error, settling-time, and tip/fall errors. | Choose support/contact position or placement action to keep the bar within a frozen angle tolerance for the final five frames. |
| Joint mass-friction | Push/collision object metrics reported across the two-factor environment grid. | Sliding/stopping and collision/post-impact errors, stratified by mass, friction, and their interaction. | Choose a push or impact action to reach a target region on held-out mass-friction combinations; report per-cell, macro, and worst-group success. |
| PnP payload dynamics | Payload mask/keypoint IoU, centroid/pose ADE/FDE, and gripper-payload relative pose error. | Motion lag, oscillation amplitude, settling time, drop rate, collision rate, and final pose error. | Choose an EEF speed/path profile that places the payload in a target pose without dropping it or exceeding the oscillation limit. |
| LightSwitch causal dynamics | Lamp on/off frame accuracy and final lamp-state accuracy; EEF or button trajectories are not primary metrics. | Lamp transition-time error, exact-transition rate, and unintended lamp-state changes. | Choose the interaction action that produces the requested final lamp state without changing unrelated lights. |
| Real slope-friction | Block/ball mask IoU and centroid ADE/FDE. | Travel-distance, stopping-time, terminal-speed, and across-surface ordering errors. | Choose release or push amplitude to stop the object in a target region; report success, target distance, overshoot, and safety failures. |

### Action-selection evaluation protocol

Action selection is evaluated separately from open-loop world-model prediction. For each environment and target area, every candidate action is rolled out by the evaluated model. The model selects the action whose **predicted** task outcome is closest to the target area; success and regret are then scored using the corresponding ground-truth rollout. No ground-truth physical parameter, ground-truth action validity, or query outcome may be used during action selection.

Target areas are defined from the full action coverage of each dataset rather than from a small inference subset. A target area is evaluated only on environments for which at least one valid ground-truth action reaches that area. Therefore, short-, medium-, and long-range target areas may have different eligible environment sets. Results must report the number of eligible environments and actions together with both micro averages over decisions and macro averages over target areas. Light-switch and pick-and-place tasks use their direct discrete/task-success definitions and do not require this continuous target-area audit.

#### Collision action task

In the two-box collision dataset, `cream_cheese_1` is the projectile directly pushed by the robot and has fixed mass. `cream_cheese_2` is the struck target object whose hidden mass varies. The action controls the projectile's impact speed, while the evaluated physical outcome is the target object's forward displacement:

\[
d(a,E)=x^{\mathrm{target}}_{\mathrm{final}}-x^{\mathrm{target}}_{\mathrm{initial}}.
\]

For a target interval \(T=[l,u]\), the model selects

\[
\hat a=\arg\min_a \operatorname{dist}(\hat d(a,E),T),\qquad
\operatorname{dist}(d,T)=\max(l-d,0,d-u).
\]

The selected action succeeds only when all of the following hold in its ground-truth rollout:

- A clean projectile-target collision occurs.
- The target-object forward displacement lies inside \([l,u]\).
- The target object remains on the table and inside the valid workspace.
- The target object's absolute lateral drift is below the configured tolerance.

Several actions may be equally successful; the policy is not required to reproduce one designated oracle action. The oracle is the valid candidate with minimum ground-truth distance to the interval. Let \(J(a)\) be that interval distance, with invalid actions assigned a deterministic penalty larger than every valid candidate cost. Report task success and

\[
\operatorname{normalized\ regret}
=\frac{J(\hat a)-J(a^*)}
{\max_a J(a)-J(a^*)},
\]

using the target width as a denominator floor. Also report selected-action validity, selected and oracle target error, raw regret, whether the selected action is oracle-equivalent, the number of valid candidates, and the number of successful candidates.

The current linear-theory collision dataset audit gives the following **provisional** displacement targets. Medium and long targets must retain the explicit on-table/workspace check when the final evaluation manifest is built.

| Target | Target-object displacement (m) | Approx. absolute target x (m) | Status |
|---|---:|---:|---|
| Short | [0.401, 0.439] | [0.281, 0.319] | Candidate target |
| Medium | [0.740, 0.777] | [0.620, 0.657] | Candidate; verify workspace validity |
| Long | [0.778, 0.816] | [0.658, 0.696] | Candidate; verify workspace validity |

The collision action evaluator consumes one JSONL row per candidate action. Rows sharing `method`, `decision_id`, `seed`, and `target_area_id` form one decision and must contain `action_id`, `predicted_target_forward_displacement_m`, `gt_target_forward_displacement_m`, `clean_collision`, and, when enabled, `target_on_table`, `target_in_workspace`, and `target_lateral_drift_m`. The evaluator writes per-decision records plus micro, per-domain, per-target, and macro-over-target summaries under the canonical `results/` tree.

#### Registered action targets for the remaining tasks

All continuous tasks use the same selection rule: choose the candidate action whose predicted scalar outcome has minimum distance to the requested interval, then score that selected action with its GT outcome. An environment/target pair enters the evaluation only if at least one valid GT candidate reaches the interval. The frozen definitions live under `configs/evaluation/action_targets/`; generated eligible sets and final metrics belong under `results/<benchmark>/<evaluation_id>/action/`.

| Task | Scalar outcome | Target definitions | GT validity |
|---|---|---|---|
| Event80 push-box friction | Block final forward displacement | short [0.051, 0.105] m; medium [0.162, 0.215] m; long [0.932, 0.986] m | Contact exists, contact occurs at action peak, and absolute lateral displacement is at most 0.08 m. The long target remains provisional until a frozen visibility/workspace check is added. |
| Joint mass-friction | Struck target-object displacement | short [0.095, 0.125] m; medium [0.155, 0.185] m; long [0.333, 0.364] m | Clean collision, target on table at the final frame, and absolute post-event lateral offset at most 0.05 m. |
| Gravity | Projectile first table-contact x | short [0.201, 0.297] m; medium [0.379, 0.475] m; long [0.643, 0.739] m | Dataset quality pass, platform-edge crossing, safe on-table landing, and small lateral drift. |
| Mass balance, fixed pose | Final hold-phase beam tilt | balanced [-0.5, 0.5] degrees | Valid physical episode. |
| Mass balance, randomized workspace | Final hold-phase beam tilt | balanced [-0.5, 0.5] degrees | Valid physical episode. |

LightSwitch is categorical rather than interval-valued. Only `red_only` and `blue_only` causal environments are evaluated because `neither` and `both` do not identify one uniquely correct button. For `turn_on`, the initial lamp is off and the desired final state is on; for `turn_off`, the initial lamp is on and the desired final state is off. Each candidate is one completed red or blue button press. The model selects the button whose predicted final-light probability is closest to the desired binary state, and GT success requires the selected rollout to reach that state. Both targets report button-selection success, final-state success, oracle reachability, and binary normalized regret. PnP action evaluation is explicitly excluded from the current scope.

#### Leakage-safe action decision pipeline

Formal action evaluation is split into two filesystem artifacts and two processes. The prediction process adapts once on the fixed support set, rolls out every candidate query action with that frozen adaptation state, extracts each predicted physical outcome, and writes a frozen action decision. Its input manifest is rejected if it contains any GT outcome, GT state/video path, or oracle action. The scorer starts only after that decision exists; it loads the separately generated GT action table, executes a lookup for the already selected action, and computes success and regret. The scorer also recomputes the prediction-only argmin and requires it to match the saved `selected_action_id`.

For fixed-camera position tasks, predicted-video outcomes come from a frozen object mask/tracker followed by a pixel-to-world calibration fitted only on training/calibration data. Event80 and joint mass-friction use object forward displacement; gravity uses final/landing x. Mass Balance uses the unoriented bar axis relative to a frozen zero-angle calibration. LightSwitch uses a frozen lamp ROI and calibrated off/on luminance scores. Missing calibration values are fatal: the evaluator does not substitute GT coordinates or hand-tuned values from the test set.

### Evaluation: Retention and Compute-Matched Forward Transfer

Save checkpoints after 25%, 50%, 75%, and 100% of training environments have been introduced. At every checkpoint, infer a fresh `Z` for the same held-out environments using identical support clips, initialization, and 40-step schedule. Report zero-step and adapted query metrics, the complete adaptation curve, and steps needed to reach fixed object-IoU thresholds.

At the same checkpoints, evaluate a fixed panel of early training environments using both their saved training-time `Z` and a freshly inferred `Z`. Report average retention and the drop from each environment's historical best.

Add a compute-matched repeated-small-set control that performs the same number of model updates and sees the same number of clips while repeatedly sampling only the initial environment subset. Compare it with progressive-environment training to separate gains from broader physical experience from gains caused only by longer training.

## Locked Event80 latent-representation ablations

These runs isolate latent capacity and parameterization while retaining the successful
Event80 random-code curriculum, grouped sampler, iterative optimization schedule,
Wan initialization, and 24-hour wall-clock budget. Every run uses two GPUs.

| Run | Environment representation | Shared projector | Training |
|---|---|---|---|
| Context dimension 4 | One learned 4-D code per active environment | Existing bias-free MLPs to Wan width | Two GPUs; iterative model/code phases |
| Context dimension 128 | One learned 128-D code per active environment | Existing bias-free MLPs to Wan width | Two GPUs; iterative model/code phases |
| Direct environment token | One learned 3072-D Wan2.2-TI2V-5B-width token per active environment | None; identity token and modulation paths | Two GPUs; iterative model/code phases |

These three ablations are planned and have not started training. Earlier queued
jobs were cancelled before allocation and do not count as experimental runs.

The direct-token run is initialized with N(0, 0.02) in token space and uses the
same curriculum and optimizer-phase ordering as the MLP variants. It changes only
the environment representation: the table entry is injected directly into both
existing physical-context conditioning paths. The established 32-D random-code run
remains the control and is not replaced.
