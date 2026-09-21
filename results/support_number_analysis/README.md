# Support-Number Ablation Plan

更新日期：2026-09-20

本文档统一记录 Mass Balance、Light Switch 和 Push-box Friction Event80 上的 support 数量消融。核心问题是：在模型、checkpoint、inner-loop schedule 和测试环境保持不变时，增加同一环境的观测片段是否能够改善环境识别、物理预测与 action selection。

## 1. 统一原则

- Ours 默认使用相同训练 checkpoint 和 40-step test-time context optimization schedule。
- 除特别标注的诊断实验外，support 与 query 必须 trajectory-disjoint。
- 同一任务的不同 K 应固定环境集合，并尽可能固定 query 集合。
- 所有 query 都使用 support adaptation 后冻结的同一个环境 latent；query 本身不参与更新。
- Standard Pooled WM 不消费 support，但应在相同 query 集合上评估。
- 主要指标包括 task/action success、object-centric error 和 task-specific physical error；PSNR、SSIM、LPIPS 为辅助指标。
- K 增大时不仅要报告数量，还必须记录 support 的信息结构。`K=2` 的两个同质 support 与互补 support 不是同一种实验。

## 2. 当前实验总览

| Task | Variant | Support construction | Status | Ours action success |
|---|---|---|---|---:|
| Mass Balance | K=1 | Nearest unbalanced support | Complete | 70.0% |
| Mass Balance | K=2 bracket | Two near-balance unbalanced supports, preferably on opposite sides | Complete | 80.0% |
| Mass Balance | K=2 unbalanced+balanced | Formal K=1 support plus one most-balanced support | Complete | 100.0% |
| Mass Balance | K=4 | K=2 bracket plus random-unbalanced and balanced support | Complete | 100.0% |
| Light Switch | K=1 | One fixed red-button support | Complete | 50.0% |
| Light Switch | K=2 | One red and one blue support | Complete | 87.5% |
| Light Switch | K=4 | Red/blue crossed with lamp-off/lamp-on | Complete | 87.5% |
| Light Switch | K=8 original formal | Original 8-support protocol with 15 disjoint queries per environment | Complete | 87.5% |
| Light Switch | K=8 controlled | Two action-diverse samples for each color/state combination | Pending (`120147`) | -- |
| Event80 | K=1 | Original informative displacement support | Complete | 72.0% |
| Event80 | K=2 diagnostic | K=1 support plus one action-space farthest point | Complete | 60.0% |
| Event80 | K=4 diagnostic | K=1 support plus three action-space farthest points | Complete | 64.0% |

Status and Slurm IDs in this table are a snapshot. Result directories and protocol files are the authoritative long-term records.

## 3. Mass Balance

Mass Balance 当前包含已有的 `K=1/K=2/K=4` 三档实验，以及一项新增的 `K=2` support-composition 实验。两项 `K=2` 的 support 数量相同，但构造方式不同，必须分别报告：

| Setting | Support 1 | Additional support | Purpose | Status |
|---|---|---|---|---|
| Existing K=1 | Nearest informative unbalanced | -- | 单 support 基准 | Complete |
| Existing K=2 bracket | Near-balance unbalanced on one side | Prefer near-balance unbalanced on the opposite side | 用两条不平衡轨迹夹逼平衡点 | Complete |
| New K=2 unbalanced+balanced | 与 Existing K=1 完全相同 | Most-balanced candidate in `[-3 deg, 3 deg]` | 对比“第二条不平衡信息”和“显式平衡信息” | Pending (`120155`) |
| Existing K=4 mixed | Existing K=2 bracket pair | One random-unbalanced plus one balanced | 更完整覆盖不平衡方向与平衡状态 | Complete |

因此，已有 support-number 主线是 `K=1 -> K=2 bracket -> K=4 mixed`；新增 K=2 是在固定 `K=2` 数量时进行的 support 类型消融，而不是替换原有 K=2。

### 3.1 Shared evaluation setting

- Dataset: workspace-random no-leak 30-ratio dataset.
- Evaluation environments: the same 5 ID and 5 OOD ratios.
- Candidate action set: 15 support-position actions per environment.
- Target: final beam tilt in `[-3 deg, 3 deg]`.
- Completed protocols evaluate 13 or 14 disjoint query trajectories per environment depending on K.

### 3.2 K=1: nearest unbalanced

The support is the nearest informative but still unbalanced trajectory. This avoids an oracle-balanced demonstration while exposing the direction and approximate magnitude of imbalance.

Result root:

`results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1`

Ours achieves 70% action success: 100% ID and 40% OOD.

### 3.3 K=2 bracket: two unbalanced supports

The preferred construction selects the nearest negative-tilt and positive-tilt unbalanced trajectories. A balanced support in `[-3 deg, 3 deg]` is forbidden. Six of the ten environments admit a strict bidirectional bracket; four extreme-ratio environments fall back to the two nearest unbalanced supports on the available side.

Result root:

`results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k2_bidirectional_near_balance_dense15_v1`

Ours reaches 80% action success: 100% ID and 60% OOD. Under the same protocol, LoRA-TTA reaches 50%, Standard Pooled WM 40%, DINOv2 30%, and TTT-KQV 30%.

### 3.4 K=2 unbalanced+balanced: completed controlled experiment

This is a distinct K=2 experiment and should not be merged with the bracket result.

- Support 1 is exactly the formal K=1 nearest-unbalanced support.
- Support 2 is the candidate with the smallest absolute ground-truth tilt inside `[-3 deg, 3 deg]`.
- The remaining 13 candidates are disjoint queries.
- The environment set remains the same 5 ID + 5 OOD set.

The experiment isolates whether one explicit near-equilibrium observation is more useful than a second unbalanced observation. Ours reaches 100% action success: 100% ID and 100% OOD, with zero mean regret. This is substantially stronger than the 80% bracket construction at the same K and shows that support composition, rather than support count alone, controls identifiability.

Planned result root:

`results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k2_nearest_unbalanced_plus_balanced_dense15_v1`

### 3.5 K=4

K=4 augments the completed bracket K=2 support with one random-unbalanced and one balanced trajectory. Ours reaches 100% action success on the current 10 environments.

Result root:

`results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k4_k2plus_random_unbalanced_balanced_dense15_v1`

This result is not a pure support-count effect because the support composition also changes. The K=2 unbalanced+balanced experiment is needed to separate quantity from information type.

### 3.6 Current Ours trend

| K | Action success | ID | OOD | LPIPS | Centroid ADE | Beam-tilt MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70% | 100% | 40% | 0.02457 | 1.199 px | 0.739 deg |
| 2 bracket | 80% | 100% | 60% | 0.02395 | 1.164 px | 0.640 deg |
| 2 unbalanced+balanced | 100% | 100% | 100% | -- | -- | -- |
| 4 mixed | 100% | 100% | 100% | 0.02290 | 1.124 px | 0.494 deg |

The bracket path is monotonic, but the controlled K=2 result reaches the same 100% action success as K=4. Two complementary observations are therefore sufficient on this evaluation; adding support beyond K=2 is not intrinsically necessary.

## 4. Light Switch

### 4.1 Task structure

The environment encodes which button controls the lamp. The informative dimensions of a support are:

- button color: red or blue;
- lamp state before the press: off or on;
- action trajectory within the selected color/state group.

The action evaluation focuses on the two identifiable causal environments: red-only and blue-only control. Video metrics still evaluate all four environments.

Result root:

`results/lightswitch/physicalpress33_all4env_support8_query15_v1`

### 4.2 Completed and pending variants

- K=1 uses one fixed red-button support and reaches 50% action success. One button observation cannot fully identify the alternative button behavior.
- K=2 uses one red and one blue support and reaches 87.5% under the selected formal protocol.
- K=4 covers `(red, off)`, `(red, on)`, `(blue, off)`, and `(blue, on)` and also reaches 87.5%.
- The original formal K=8 experiment has already completed. It uses 8 support trajectories and 15 support-disjoint query trajectories in each of the four causal environments (`neither`, `red_only`, `blue_only`, and `both`), for 60 query rollouts in total. Ours reaches 87.5% action success, PSNR 33.454, and SSIM 0.9443. Its protocol is recorded in `protocol/support_query_manifest.json`, and its aggregate metrics are in `metrics/complete_v1/summary.csv` under the Light Switch result root.
- A new controlled K=8 experiment separately selects two action-diverse trajectories for each of `(red, off)`, `(red, on)`, `(blue, off)`, and `(blue, on)`. Job `120147` is pending. This controlled construction is intended for a cleaner comparison with K=4 and must not be conflated with the completed original K=8 result.

The four K=2 initial-state combinations produced 87.5%, 87.5%, 100%, and 100% action success. This shows that K alone is insufficient: initial lamp-state coverage materially affects identifiability.

Under the selected K=2 protocol, DINOv2, LoRA-TTA, and TTT-KQV each reach 50%, compared with 87.5% for Ours.

Current Light Switch support-count results are therefore `K=1: 50%`, `K=2: 87.5%`, `K=4: 87.5%`, and original formal `K=8: 87.5%`. Since the support-selection policies are not identical across all four points, this is an empirical trend rather than a strictly quantity-only ablation; the pending controlled K=8 run addresses part of that mismatch.

## 5. Event80 Push-box Friction

### 5.1 Formal K=1 reference

The formal evaluation uses one informative support whose ground-truth displacement lies in the selected informative range. It evaluates 5 ID and 5 OOD friction environments with disjoint cross-action queries.

Formal root:

`results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1`

### 5.2 Current K=2/K=4 diagnostic

The submitted diagnostic freezes the original K=1 query set and adds support trajectories by greedy farthest-point selection in action space:

- K=2: original informative support plus one farthest action.
- K=4: original informative support plus three farthest actions.
- Environment set: the same 5 ID + 5 OOD.
- Frozen query count: 9 per environment.
- Ours checkpoint: step 7272.

Both diagnostic runs are complete. After re-scoring their cached action candidates with the current formal Event80 rule, K=2 reaches 60% action success and K=4 reaches 64%. The revised rule changes the short target to normalized image-y `[0.595, 0.68]`; when no prediction reaches the long target, it maximizes predicted image-y rounded to three decimals and breaks ties toward the larger action level. Re-scoring did not change either diagnostic score. For comparison, the formal K=1 result is 72% under the same current action rule.

Diagnostic root:

`results/pushbox_friction_event80/event80_k2_k4_frozen_k1_query_overlap_diagnostic_v1`

Important limitation: the extra supports are selected from the original K=1 query pool while the original query set remains frozen. Therefore, support and query overlap by construction. These runs are useful for diagnosing whether additional observations improve adaptation, but they must not be reported as formal disjoint-support/query results.

### 5.3 Required formal follow-up

A publication-quality Event80 support-number ablation should:

1. Fix the same 5 ID + 5 OOD environments and step-7272 checkpoint.
2. Construct K=1/K=2/K=4 support sets before defining queries.
3. Remove every selected support trajectory from the query set.
4. Use a common disjoint query subset available to all K values, or report matched per-K query counts explicitly.
5. Keep the 40-step optimizer, LR schedule, FP32 latent, seed, and action metric unchanged.

## 6. Intended interpretation

The ablation should answer two separate questions:

1. **Support quantity:** does increasing K improve adaptation when support informativeness is controlled?
2. **Support coverage:** for a fixed K, do complementary physical outcomes or action/state coverage outperform redundant observations?

Mass Balance K=2 bracket versus K=2 unbalanced+balanced addresses support composition. Light Switch directly tests causal color/state coverage. Event80 K=2/K=4 currently tests action diversity diagnostically and still requires a disjoint formal rerun.

## 7. Consolidated artifacts

- Machine-readable results: `support_number_summary.csv`.
- Paper-style action-success figure: `support_number_action_success.svg` and `support_number_action_success.png`.
- Reproducible renderer: `scripts/evaluation/render_support_number_analysis.py`.

The figure separates the two Mass Balance K=2 constructions and marks Event80 K=2/K=4 as overlap diagnostics. It must not be interpreted as a strictly quantity-only ablation across all tasks because support composition changes between several points.
