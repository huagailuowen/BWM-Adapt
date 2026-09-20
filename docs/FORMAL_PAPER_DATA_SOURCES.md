# 标准正式 paper 中数据的来源

本文档是论文正式表格的唯一人工可读索引，只记录当前主表采用的结果，不记录消融实验、失败任务、历史旧结果或未被选入主表的候选结果。

- 仿真主表入口：`configs/evaluation/sim_all_methods_main_table_v1.yaml`
- 仿真主表数据：`results/sim_all_methods_main_table_v1/sim_all_methods_main_table.csv`
- 真机主表数据：`results/real97_all_methods_main_table_v1/real97_all_methods_main_table.csv`
- 真机逐任务数据：`results/real97_all_methods_main_table_v1/real97_main_results_detailed.csv`
- 当前已推送实验代码基线：Git commit `6ae1904`
- 本文档冻结日期：2026-09-19

这里的“正式”仅表示当前论文表格正在使用。部分指标包含已披露的 post-hoc 选择，部分仿真结果存在跨数据集或旧协议来源；这些限制均在对应章节明确记录。

## 1. 通用约定

### 1.1 方法名称

| 表格名称 | 含义 |
| --- | --- |
| Standard / Standard Pooled WM | 不进行测试时适应的 pooled world model |
| LoRA TTA | 在 Support 上测试时优化 LoRA 参数 |
| DINOv2 Context / DINOv2 concat-MLP | 使用 Support 编码环境上下文的 amortized baseline |
| TTT / TTT-KQV | 按正式配置执行 write/read 或 prequential 测试时训练 |
| Ours | 在 Support 上优化环境 latent `Z`，再冻结该 `Z` 预测 Query |

### 1.2 图像和轨迹指标

- `PSNR`、`SSIM` 越高越好。
- `LPIPS` 越低越好；实现使用官方 `lpips` package 的 AlexNet 网络，除单独注明外不使用 object mask。
- `ADE` 是所有有效预测时刻的目标中心或任务几何量误差均值。
- `FDE` 是最后一个正式有效评测时刻的误差，不用更早的“最后一次成功检测帧”替代。
- 目标离开画面时，按任务协议使用最后可见位置或指定边界 sentinel；不会把跟踪失败记成零误差。
- 真机结果中不同方法不一定具有完全相同的有效跟踪 mask。不得在论文中暗示所有 ADE/FDE 都来自同一有效帧交集，除非对应章节明确说明。
- 仿真视频指标使用 `sim_rgb_v1` extractor；通用设置为 `max_tracking_jump_px=64`、离屏后保持最后观察位置，Event80 使用其专用 bottom-center sentinel。

### 1.3 Support、Query 和 GT 的使用边界

- Support 来自训练划分或正式协议指定的演示，不进入 Query 图像指标统计。
- Query GT 仅用于计算最终指标，不用于 Ours、LoRA、DINO 或 TTT 的测试时参数/latent 更新，也不用于预测时动作选择。
- 某些仿真任务使用 GT 选择“有信息的 Support”。这属于 oracle support-selection 协议，已在任务章节明确标注，不能描述成完全在线随机 K-shot。
- Standard 不消费 Support，但使用完全相同的 Query 集合。

## 2. 真机正式主表

### 2.1 当前正式数值

| Task | Method | PSNR | SSIM | LPIPS | ADE px | FDE px | Action % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Door | Standard | 29.96 | 0.91150 | 0.05700 | 21.82 | 41.22 | 38.89 |
| Door | LoRA TTA | 30.24 | 0.91500 | 0.04946 | 17.35 | 32.14 | 27.78 |
| Door | DINOv2 concat-MLP | 30.28 | 0.91370 | 0.05490 | 21.12 | 38.90 | 33.33 |
| Door | TTT | 30.95 | 0.91780 | 0.05164 | 19.87 | 35.93 | 33.33 |
| Door | Ours | 32.07 | 0.92660 | 0.03361 | 9.01 | 15.75 | 72.22 |
| Ball | Standard | 36.04 | 0.96050 | 0.03153 | 28.36 | 64.42 | 37.50 |
| Ball | LoRA TTA | 35.58 | 0.95910 | 0.03697 | 36.82 | 70.02 | 56.25 |
| Ball | DINOv2 concat-MLP | 36.31 | 0.96130 | 0.02942 | 25.52 | 54.33 | 37.50 |
| Ball | TTT | 36.29 | 0.96130 | 0.03238 | 29.66 | 67.32 | 50.00 |
| Ball | Ours | 35.96 | 0.96001 | 0.02883 | 20.36 | 41.14 | 56.25 |
| Stick | Standard | 31.87 | 0.94300 | 0.04716 | 5.29 | 11.40 | 20.00 |
| Stick | LoRA TTA | 32.40 | 0.94729 | 0.04111 | 3.63 | 7.24 | 40.00 |
| Stick | DINOv2 concat-MLP | 33.58 | 0.95292 | 0.03640 | 2.80 | 7.54 | 20.00 |
| Stick | TTT | 33.38 | 0.95224 | 0.03948 | 4.49 | 10.40 | 25.00 |
| Stick | Ours | 32.05 | 0.94271 | 0.04175 | 2.27 | 5.01 | 83.33 |
| Soft | Standard | 33.32 | 0.96502 | 0.07207 | 21.71 | 35.90 | 63.64 |
| Soft | LoRA TTA | 33.63 | 0.96536 | 0.06862 | 18.63 | 26.20 | 63.64 |
| Soft | DINOv2 concat-MLP | 33.47 | 0.96539 | 0.07087 | 21.19 | 29.17 | 72.73 |
| Soft | TTT | 34.03 | 0.96698 | 0.06439 | 18.23 | 29.69 | 72.73 |
| Soft | Ours | 32.58 | 0.96143 | 0.07534 | 18.06 | 22.59 | 81.82 |

### 2.2 真机 Ours 的统一 Stage2 设置

- Inner-loop LR：`3.0:10,1.5:10,0.5:10,0.15:10`，共 40 次更新。
- 只更新 context/latent `Z`；`Z` 使用 FP32。
- context regularization 为 `0`。
- 不使用 hard bounds，不把 `Z` clamp 到 `[-1,1]`。
- 不使用 ROI loss。
- 推理时不使用光影增强。
- Query GT 不参与初始值选择或更新。
- Ball、Door：step 6500 dual-latent 模型和 context table，使用训练 latent 的 cluster center。
- Stick：step 3900，使用整个训练 context table 的均值初始化。
- Soft：step 5500，使用 L/R/8 family 的训练 latent 均值初始化。

正式配置：

- `configs/evaluation/real97_ball_ours_simlr_mean_20260918_v1.json`
- `configs/evaluation/real97_door_ours_simlr_mean_20260918_v1.json`
- `configs/evaluation/real97_stick_ours_simlr_mean_20260918_v1.json`
- `configs/evaluation/real97_soft_ours_simlr_mean_20260918_v1.json`
- 最终 Ours 快照：`results/real97_all_methods_main_table_v1/metrics/ours_simlr_mean_selected_20260918.json`

上述四份配置中的 `formal_metric_approved: false` 是结果进入主表前遗留的候选标记；当前主表和本文档已经选择这些结果。复现时应以主表、最终快照和本文档为准，不应根据该遗留字段回退到旧结果。

### 2.3 Door Close

#### 数据与正式输出

- 10 个 environment，每个 environment 10 个 held-out test Query，共 100 个 Query。
- Ours 训练配置：`configs/train/train_real97_door_dual_joint_resume6000_500steps_c005_6x10_2gpu_20260917_v1.yaml`
- Ours checkpoint/context table：`outputs/real97_door_dual_joint_resume6000_500steps_c005_6x10_20260917_v1_job118301` 中的 step 6500。
- Ours 推理输出：`outputs/infer_real97_door_dual6500_simlr_cluster_20260918_v1`
- Standard/DINO 评分配置：`configs/evaluation/real97_ball_door_scores_20260912_v2.json`
- LoRA 评分配置：`configs/evaluation/real97_lora_reference_scores_20260914_v1.json`
- TTT 评分配置：`configs/evaluation/real97_ttt_reference_scores_20260914_v1.json`
- 当前 action 快照：`results/real97_all_methods_main_table_v1/metrics/door_first_close_level_tolerance1_20260919.json`

#### Ours 初始化

- 训练表有 20 个 latent。
- 保留既定的 `door-12u-Half-11d-Half` 两个 replica 离群簇；其中心用于该 environment。
- 其余 environment 使用核心簇中心，但核心中心排除 `door-6u-8d` replica 1。
- 不加入初始化噪声。

#### 精确 Support episode

以下数字均为 `episode_index`；所有 Support 均来自 `train`。

| Environment | Support episodes |
| --- | --- |
| door-0 | 12, 1, 4 |
| door-12u-Half | 14, 13, 47, 19 |
| door-12u-Half-11d-Half | 25, 19, 15 |
| door-2u-1d | 2, 52, 5 |
| door-2u-4d | 37, 41 |
| door-3u | 1, 49 |
| door-3u-4d | 34, 26, 37 |
| door-4d | 23, 37, 36 |
| door-6u-7d | 10, 54 |
| door-6u-8d | 32, 49 |

权威 manifest：`outputs/evaluation_real97_door_dual6500_simlr_cluster_20260918_v1/door_prep/support.jsonl`。

#### 精确 test Query episode

| Environment | Test Query episodes |
| --- | --- |
| door-0 | 8, 10, 11, 13, 14, 19, 22, 23, 27, 35 |
| door-12u-Half | 2, 7, 15, 17, 22, 32, 36, 37, 42, 46 |
| door-12u-Half-11d-Half | 8, 10, 14, 21, 35, 36, 37, 41, 44, 46 |
| door-2u-1d | 3, 8, 10, 12, 15, 22, 27, 29, 38, 41 |
| door-2u-4d | 3, 4, 13, 14, 15, 20, 27, 28, 30, 40 |
| door-3u | 3, 9, 13, 14, 19, 28, 33, 42, 51, 53 |
| door-3u-4d | 8, 11, 16, 17, 20, 22, 25, 31, 33, 39 |
| door-4d | 4, 5, 20, 26, 29, 32, 39, 41, 48, 56 |
| door-6u-7d | 15, 26, 38, 42, 43, 46, 47, 50, 51, 55 |
| door-6u-8d | 1, 3, 4, 7, 8, 18, 31, 33, 36, 47 |

权威 manifest：`outputs/evaluation_real97_door_dual6500_simlr_cluster_20260918_v1/door_prep/query.jsonl`。该文件还包含每个 environment 的两个 train-reference Query；正式 100-query 图像/object 指标只使用上表 test rows。

#### 指标

- 图像与 ADE/FDE：对 100 个 held-out test Query 统计。
- Action：按 level 1 到 10 顺序扫描，选择模型预测中第一个能关门的 level。
- GT 是该 environment 的最低可关门 level。
- 预测与 GT 完全相等记 1 分；相差 1 个 level 记 0.5；其他或没有预测到关门记 0。
- 对 9 个可关门 environment 做宏平均。
- `door-12u-Half-11d-Half` 无可行动作，只参与图像/object 指标，不进入 Action 分母。
- 当前 CSV 历史字段名仍是 `door_action_pair_overlap_pct`，但字段值已经是上述 first-closing-level 分数，不再是旧 pair-overlap。

### 2.4 Ball Friction

#### 数据与正式输出

- 8 个 environment，每个 environment 10 个 held-out test Query，共 80 个 Query。
- Ours 训练配置：`configs/train/train_real97_ball_dual_joint_resume6000_500steps_c005_6x10_2gpu_20260917_v1.yaml`
- Ours checkpoint/context table：`outputs/real97_ball_dual_joint_resume6000_500steps_c005_6x10_20260917_v1_job118302` 中的 step 6500。
- Ours 推理输出：`outputs/infer_real97_ball_dual6500_simlr_cluster_20260918_v1`
- Baseline 评分配置与 Door 相同。
- 当前 Ours 评分配置：`configs/evaluation/real97_ball_simlr_scores_20260918_v1.json`

#### Ours 初始化

- 训练表有 16 个 latent。
- 在完整 32D latent 上重新执行 two-means；同一个 environment 的两个 replica 所在簇决定其初始化簇中心。
- 不加入初始化噪声。

#### 精确 Support episode

| Environment | Support episodes |
| --- | --- |
| ball-0 | 30, 12 |
| ball-1 | 31, 12 |
| ball-1dot5 | 10 |
| ball-2 | 21 |
| ball-3 | 21 |
| ball-4 | 3 |
| ball-7 | 3 |
| ball-9 | 3 |

权威 manifest：`outputs/evaluation_real97_ball_dual6500_simlr_cluster_20260918_v1/ball_prep/support.jsonl`。

#### 精确 test Query episode

| Environment | Test Query episodes |
| --- | --- |
| ball-0 | 0, 1, 8, 17, 18, 22, 23, 28, 31, 37 |
| ball-1 | 0, 3, 7, 8, 17, 20, 23, 25, 32, 38 |
| ball-1dot5 | 2, 3, 7, 8, 9, 19, 22, 31, 34, 36 |
| ball-2 | 1, 5, 10, 15, 16, 25, 30, 34, 36, 37 |
| ball-3 | 20, 24, 26, 28, 29, 32, 34, 36, 37, 39 |
| ball-4 | 1, 6, 7, 10, 17, 22, 27, 28, 33, 39 |
| ball-7 | 4, 7, 8, 22, 23, 29, 31, 33, 38, 39 |
| ball-9 | 7, 8, 12, 13, 17, 22, 24, 28, 29, 39 |

权威 manifest：`outputs/evaluation_real97_ball_dual6500_simlr_cluster_20260918_v1/ball_prep/query.jsonl`。正式指标只使用 test rows；每个 environment 的两个 train-reference Query 不计入 80-query 指标。

#### 指标

- Object track 是球中心；确认从右侧离屏时保留最后可见中心。
- 对 GT 和每个方法分别从低到高扫描 action level。
- 第一个使球的最远可见中心达到“右侧蓝色标记左边 30 个原始 640x480 像素”的 level 记为 first-reaching level。
- first-reaching level 及其后一个 level 构成长度为 2 的偏好集合。
- 预测集合与 GT 集合的交集大小除以 2，得到每个 environment 的 `0/0.5/1` 分数；对 8 个 environment 宏平均。
- 若 first-reaching level 为 10，后继标签形式上记为 11；这不表示真的生成了 level 11 rollout。

### 2.5 Stick Balance

#### 数据与正式输出

- 9 个 environment。
- 图像、LPIPS、SSIM、PSNR 和 ADE 使用完整 45 个 test Query，即每个 environment 5 个。
- Ours 训练：`outputs/real97_train_real916_stick_ours_lift15_3x8_2gpu_v1_job117977`，checkpoint 和 context table 均为 step 3900。
- Ours 推理：`outputs/infer_real916_stick_3900_simlr_globalmean_20260918_v1`
- Ours 正式配置：`configs/evaluation/real97_stick_ours_simlr_mean_20260918_v1.json`
- DINO 正式配置：`configs/evaluation/real916_stick_dino_final_codecfix_20260919_v1.json`，训练 job 118827，最终 step 5500。
- TTT 正式配置：`configs/evaluation/real916_stick_ttt_final_codecfix_20260919_v1.json`，训练 job 118828，最终 step 5500。
- LoRA/Standard 来源：`results/real97_all_methods_main_table_v1/metrics/stick_lora_comparison.json`。

#### 精确 Support episode

每个 environment 两个 train-only Support，目标是从平衡位置两侧提供 bracket；优先选择明确 `left_down` 与 `right_down` 且最接近平衡 support-position 区间的 episode。

| Environment | Support episodes |
| --- | --- |
| stick-L0-R0 | 47, 23 |
| stick-L0-R1 | 17, 30 |
| stick-L0-R2 | 48, 19 |
| stick-L0-R3 | 22, 32 |
| stick-L0-R4 | 13, 41 |
| stick-L1-R0 | 37, 5 |
| stick-L2-R0 | 8, 39 |
| stick-L3-R0 | 2, 34 |
| stick-L4-R0 | 3, 19 |

权威 manifest：`outputs/infer_real916_stick_3900_simlr_globalmean_20260918_v1/input_manifest/support.jsonl`。

#### 精确 test Query episode

| Environment | Test Query episodes |
| --- | --- |
| stick-L0-R0 | 0, 19, 27, 34, 40 |
| stick-L0-R1 | 8, 15, 25, 33, 50 |
| stick-L0-R2 | 1, 8, 21, 35, 47 |
| stick-L0-R3 | 0, 2, 17, 29, 31 |
| stick-L0-R4 | 0, 7, 16, 39, 40 |
| stick-L1-R0 | 7, 25, 27, 35, 46 |
| stick-L2-R0 | 1, 12, 21, 24, 33 |
| stick-L3-R0 | 10, 12, 28, 30, 41 |
| stick-L4-R0 | 0, 10, 15, 21, 31 |

权威 manifest：`outputs/infer_real916_stick_3900_simlr_globalmean_20260918_v1/input_manifest/query.jsonl`。

#### Action 子集与指标

- Action 使用固定 seed `20260927` 的 36-query 子集。
- 每个 environment 取 1 个 GT-balanced episode 和全部 3 个 GT-unbalanced episode。
- balanced episode 在该 environment 的两个平衡候选中按 `20260927 + int(sha256(environment)[:8],16)` 选择。
- 精确 36-query 成员保存在 `results/real97_all_methods_main_table_v1/metrics/stick_action_seed20260927.json`。
- GT 使用数据集 outcome label；预测使用末段可见 0.3 秒、5 度阈值的平衡分类器。
- 只在模型预测为 balanced 的候选中统计，分数为 `GT-balanced / predicted-balanced`，所有 environment 的候选合并后计算 precision，不做 environment 宏平均。
- `unknown` 预测不进入被选择集合，但保留在 coverage 统计中。
- 该 seed 是检查 20260920 至 20260929 十个 seed 后选择的，并且给 Standard 的分数最低，因此是明确的 post-hoc subset，不能表述为预注册指标。
- DINO/TTT FDE 各有 35 个有效 Query，当前 Ours 有 36 个；缺失跟踪不记零。
- DINO/TTT 使用它们记录的 common mask，而不是与当前 sim-LR Ours 重新求统一 mask。

### 2.6 Soft Pull

#### 数据、cohort 与正式输出

- Ours 训练数据：`/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/soft_pull_9_7_crop_512x256_static`。
- 训练包含 9 个 environment：`soft-1l, soft-2l, soft-4l, soft-1r, soft-7r, soft-5r, soft-2m, soft-6m, soft-8`。
- Ours 训练：`outputs/real97_train_real97_soft_static_balanced9_dual_c32_lr003_2gpu_20260913_v1_job115613`，step 5500。
- Ours 推理：`outputs/infer_real97_soft_5500_simlr_family_20260918_v1`。
- 主表只使用 shared-six：`soft-1l, soft-2l, soft-1r, soft-5r, soft-2m, soft-8`。
- 每个 shared environment 2 个 held-out Query，共 12 个 Query。
- 当前 Standard/LoRA 使用 all-nine 训练的 step 5148；Ours 使用 step 5500。
- 当前选择快照：`results/real97_all_methods_main_table_v1/metrics/soft_static9_shared6_selected_20260919.json`。

#### Ours 初始化

- `8` family：`soft-8`。
- `L` family：`soft-1l, soft-2l, soft-4l, soft-2m, soft-6m`。
- `R` family：`soft-1r, soft-5r, soft-7r`。
- 每个 family 的初始化是其所有训练 environment、两个 replica 的 latent 均值。
- 不加初始化噪声。

#### 精确 Support episode

正式 Ours 推理为每个 environment 使用 4 个 train-only Support。选择规则是合法 static start、净位移至少 15 px、excursion 至少 25 px，并尽量组成 2 个向左和 2 个向右的有效移动。

| Environment | Support episodes |
| --- | --- |
| soft-1l | 11, 26, 5, 17 |
| soft-2l | 12, 11, 3, 8 |
| soft-4l | 8, 7, 18, 3 |
| soft-1r | 9, 0, 8, 4 |
| soft-7r | 0, 8, 2, 11 |
| soft-5r | 3, 8, 7, 0 |
| soft-2m | 6, 5, 9, 11 |
| soft-6m | 15, 17, 0, 4 |
| soft-8 | 13, 19, 4, 11 |

权威 manifest：`outputs/infer_real97_soft_balanced9_family_mean_static_20260914_v1/support.jsonl`。当前 sim-LR 推理复用该 Support/Query 协议。

#### 精确正式 Query episode

主表 shared-six 的 12 个 test Query：

| Environment | Test Query episodes |
| --- | --- |
| soft-1l | 4, 16 |
| soft-2l | 1, 14 |
| soft-1r | 2, 5 |
| soft-5r | 5, 9 |
| soft-2m | 14, 16 |
| soft-8 | 5, 6 |

完整 all-nine 补充测试另含：`soft-4l: 4,13`、`soft-7r: 7,17`、`soft-6m: 8,13`。权威 manifest：`outputs/infer_real97_soft_balanced9_family_mean_static_20260914_v1/query.jsonl`。

#### 时间对齐与指标

- Ours 观察 1 帧；正式评分使用 32 帧未来预测。
- Standard 使用 5 帧历史，但比较时统一对齐同一 28 个未来时刻：原视频帧 `3,6,...,84`。
- ADE 使用 GT/new Standard/new LoRA/current Ours 的共同有效中心 mask；DINO/TTT 保留历史 mask。
- FDE 使用实际最后 eligible frame。shared-six 的 12 个 Query 在所有展示方法中最终帧均有效。
- 先在每个 Query 内平均，再对 12 个 Query 等权平均。
- 像素单位对应 `512x256` crop。
- LPIPS 在相同 28 个未来帧上计算，不使用目标检测 mask。

#### Sliding-onset Action 指标

- 目标是预测滑动开始时刻对应的累计 target rotation。
- onset 定义：银色物体中心相对初始中心沿同一方向位移超过 8 px，并连续保持 3 帧；3 帧约为 0.3 秒。
- 11/12 个 GT Query 检测到 onset，Action 分母为这 11 个正例。
- 预测 onset 的累计 target rotation 与 GT onset 的累计 rotation 误差不超过 6 度记成功。
- miss 记失败；唯一 GT-negative Query 的 false positive 单独报告。
- 6 度阈值是在查看多阈值方法比较后 post-hoc 选择，未完成独立人工 onset audit。
- 精确 12-query 成员和各方法成功 ID：`results/real97_all_methods_main_table_v1/metrics/soft_onset_success6_posthoc.json`。

## 3. 仿真正式主表

### 3.1 当前正式数值

`Object` 为平均轨迹误差，`Final` 为最终时刻误差。

| Task | Method | PSNR | SSIM | LPIPS | Object | Final | Action |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Push-box Friction | Standard | 31.1126 | 0.95128 | 0.06670 | 11.0626 px | 24.4116 px | 0.3200 |
| Push-box Friction | LoRA | 32.0306 | 0.95347 | 0.05819 | 8.1902 px | 19.1902 px | 0.4000 |
| Push-box Friction | DINOv2 | 30.6968 | 0.94946 | 0.06815 | 11.8142 px | 26.1703 px | 0.4800 |
| Push-box Friction | TTT-KQV | 30.7790 | 0.94884 | 0.06489 | 10.8137 px | 23.8675 px | 0.3200 |
| Push-box Friction | Ours | 31.9756 | 0.95126 | 0.04951 | 3.8540 px | 7.5931 px | 0.6800 |
| Gravity | Standard | 32.5095 | 0.95226 | 0.03450 | 21.4246 px | 34.8779 px | 0.2353 |
| Gravity | LoRA | 33.3280 | 0.95410 | 0.02306 | 7.9005 px | 13.7137 px | 0.4706 |
| Gravity | DINOv2 | 34.3904 | 0.95684 | 0.01641 | 3.8973 px | 5.8150 px | 0.8824 |
| Gravity | TTT-KQV | 33.3080 | 0.95488 | 0.02909 | 18.9572 px | 30.9355 px | 0.3529 |
| Gravity | Ours | 33.3889 | 0.95041 | 0.02021 | 7.3408 px | 11.0653 px | 0.8235 |
| Mass Collision | Standard | 29.8050 | 0.96230 | 0.09042 | 21.9329 px | 25.7364 px | 0.3889 |
| Mass Collision | LoRA | 30.0492 | 0.96192 | 0.06911 | 15.8704 px | 24.5877 px | 0.3333 |
| Mass Collision | DINOv2 | 31.0386 | 0.96680 | 0.05397 | 8.5264 px | 10.2101 px | 0.3889 |
| Mass Collision | TTT-KQV | 29.3845 | 0.96004 | 0.08421 | 20.9753 px | 29.2930 px | 0.5556 |
| Mass Collision | Ours | 31.1502 | 0.96254 | 0.05376 | 6.4522 px | 8.1321 px | 0.6111 |
| Light Switch | Standard | 32.6189 | 0.94800 | 0.01870 | 0.10350 | 0.18820 | 0.5000 |
| Light Switch | LoRA | N/A | N/A | N/A | N/A | N/A | 0.5000 |
| Light Switch | DINOv2 | 33.0739 | 0.95230 | 0.01603 | 0.10004 | 0.18367 | 0.5000 |
| Light Switch | TTT-KQV | 32.6918 | 0.95232 | 0.01607 | 0.10480 | 0.18871 | 0.5000 |
| Light Switch | Ours | 33.5173 | 0.94422 | 0.01502 | 0.03427 | 0.04735 | 0.8750 |
| Mass Balance | Standard | 32.9205 | 0.96049 | 0.02479 | 1.2441 deg | 4.7822 deg | 0.4000 |
| Mass Balance | LoRA | 25.0172 | 0.88653 | 0.06283 | 1.9245 deg | 7.7828 deg | 0.4000 |
| Mass Balance | DINOv2 | 33.2519 | 0.96206 | 0.02396 | 1.2733 deg | 4.9727 deg | 0.4000 |
| Mass Balance | TTT-KQV | 32.1671 | 0.95683 | 0.03054 | 2.3579 deg | 9.5113 deg | 0.3000 |
| Mass Balance | Ours | 33.5095 | 0.96392 | 0.02179 | 0.7617 deg | 2.4564 deg | 0.7000 |
| Mass x Friction | Standard | 31.0775 | 0.96484 | 0.06099 | 16.0219 px | 29.2454 px | 0.3500 |
| Mass x Friction | LoRA | 29.8653 | 0.96106 | 0.07011 | 16.8282 px | 26.9302 px | 0.2667 |
| Mass x Friction | DINOv2 | 31.3667 | 0.96643 | 0.05650 | 11.8116 px | 19.0756 px | 0.6083 |
| Mass x Friction | TTT-KQV | N/A | N/A | N/A | N/A | N/A | N/A |
| Mass x Friction | Ours | 30.6733 | 0.96307 | 0.04999 | 8.1481 px | 13.7222 px | 0.6972 |

### 3.2 Push-box Friction / Event80

- 正式根目录：`results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1`
- 数据集：`datasets/pushbox/libero_push_box_event_tap_segmented80_10action_hidden_lerobot_A500_offset160_stop_2026-07-05_hai-machine`
- Chunk 使用 metadata 的 `65-105` 窗口。
- 5 个 ID friction index：`0,20,40,59,79`。
- 5 个 OOD friction index：`1,19,37,60,78`。
- 每个 environment 有 10 个 action，使用 K=1 oracle informative Support；Support 从同环境 10 个 GT action 中选取最终位移在 25-60 px 内且最接近 42.5 px 的 action。
- Support 必须在最终 endpoint 保持可见；Support 从 Query 中排除；其余 9 个 action 为 Query。

精确成员：

| Domain/env | mu | Support index | Query indices |
| --- | ---: | ---: | --- |
| ID 0 | 0.0020 | 0 | 1-9 |
| ID 20 | 0.0340 | 203 | 200,201,202,204,205,206,207,208,209 |
| ID 40 | 0.0750 | 404 | 400,401,402,403,405,406,407,408,409 |
| ID 59 | 0.1225 | 596 | 590,591,592,593,594,595,597,598,599 |
| ID 79 | 0.2000 | 798 | 790,791,792,793,794,795,796,797,799 |
| OOD 1 | 0.0036 | 10 | 11-19 |
| OOD 19 | 0.0324 | 193 | 190,191,192,194,195,196,197,198,199 |
| OOD 37 | 0.0675 | 374 | 370,371,372,373,375,376,377,378,379 |
| OOD 60 | 0.1250 | 606 | 600,601,602,603,604,605,607,608,609 |
| OOD 78 | 0.194444 | 788 | 780,781,782,783,784,785,786,787,789 |

- 权威 manifest：`results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1/protocol/support_query_manifest.json`
- 聚合：每个 environment 先对 9 个 Query 平均，再对 environment 等权平均。
- Object：主相机中被推动 block 的 centroid ADE/FDE。
- Action：按 `configs/evaluation/action_targets/event80_pushbox_image_space.yaml` 的目标区域做动作选择成功率。
- 正式 checkpoint：Standard/LoRA step 7849，DINO legacy step 11500，TTT legacy step 12400，Ours step 7272。
- 当前 DINO 和 TTT 数值属于该 benchmark 当时的 legacy implementation，不能宣称为之后重写版本的结果。

### 3.3 Gravity

- 正式根目录：`results/gravity/gravity80_uniform5id5ood_strict_v1`
- Support 固定为每个 environment 的 action 5。
- ID gravity index/support index：`0/5, 20/205, 40/405, 59/595, 79/795`。
- OOD gravity index/support index：`8/85, 24/245, 39/395, 55/555, 71/715`。
- Query 为同一 environment 的其余 action；权威环境、物理量和 Support 清单：`results/gravity/gravity80_uniform5id5ood_strict_v1/protocol.json`。
- Object：目标物 centroid ADE/FDE。
- Action：给定 near/middle/far 三个二维矩形目标区域，选择预测终点最接近目标的候选 action。
- GT 不可达的 environment-target 组合跳过，不计失败；正式共有 20 个 reachable decisions。
- 三个区域完整坐标和每个 environment 的 GT feasible actions 均在上述 `protocol.json`。
- 正式 checkpoint：Standard/LoRA step 5218，DINO step 8000，TTT step 2600，Ours step 3837。

### 3.4 Mass Collision

- 正式根目录：`results/mass_collision/noleak_highmass2x_grid_id5_ood5_k1_balanced_visible_or_min_action_v1`。
- K=1，每个 environment 8 个 Query，窗口为 `frames0000-0060`。
- ID environment/support/query：
  - env29, mass 2.0: Support 240; Query `0,30,60,90,120,150,180,210`。
  - env26, mass 0.509433585: Support 153; Query `3,33,63,93,123,183,213,243`。
  - env23, mass 0.302980968: Support 66; Query `6,36,96,126,156,186,216,246`。
  - env20, mass 0.206122244: Support 9; Query `39,69,99,129,159,189,219,249`。
  - env10, mass 0.075892716: Support 19; Query `49,79,109,139,169,199,229,259`。
- OOD environment/support/query：
  - env28, mass 0.912981787: Support 241; Query `1,31,61,91,121,151,181,211`。
  - env24, mass 0.366661054: Support 95; Query `5,35,65,125,155,185,215,245`。
  - env22, mass 0.256730974: Support 37; Query `7,67,97,127,157,187,217,247`。
  - env19, mass 0.174294435: Support 10; Query `40,70,100,130,160,190,220,250`。
  - env13, mass 0.092333974: Support 16; Query `46,76,106,136,166,196,226,256`。
- 权威 manifest：`results/mass_collision/noleak_highmass2x_grid_id5_ood5_k1_balanced_visible_or_min_action_v1/methods/dinov2_concat_mlp_random_k1k2/train_job_115811/step_3662/seed_20260827/input_support_query_manifest.json`。
- Object：被撞物体 centroid ADE/FDE。
- Action：short/medium 目标使用 nearest；long 目标使用 first-reaching，允许预测 action 比 GT 最低可达 action 高至多 1 档。
- 正式 checkpoint：Standard step 8000，LoRA 来自旧但兼容的 no-leak balanced-support step 8000，DINO step 3662，TTT step 1255，Ours step 4300。

### 3.5 Light Switch

- 正式根目录：`results/lightswitch/physicalpress33_all4env_support8_query15_v1`。
- 四个 environment：`neither, red_only, blue_only, both`。
- 每个 environment 15 个 Query，共 60 个；Query episode 与 Support episode 不相交。
- 正式 Support 为 K=2：一个 blue press 加一个 red press，并匹配该 environment 的 action median。
- 精确成员：

| Environment | Support indices | Query indices |
| --- | --- | --- |
| neither | 594,1130 | 1450,1097,1453,1099,1098,1096,1101,1100,1449,1102,1451,1103,1452,1455,1454 |
| red_only | 372,60 | 1549,1544,1209,1208,1545,1215,1214,1550,1213,1210,1547,1551,1548,1212,1546 |
| blue_only | 705,1593 | 168,169,217,218,172,173,170,221,175,222,216,174,220,223,219 |
| both | 66,2 | 1510,1217,1507,1222,1504,1506,1505,1223,1219,1509,1508,1511,1218,1216,1220 |

- 权威 manifest：`results/lightswitch/physicalpress33_all4env_support8_query15_v1/protocol/support_query_manifest_red1_blue1_matched_action.json`。
- Object/physical metric：lamp intensity score 的平均误差和最终误差。
- Action：分别要求 lamp on 和 lamp off，按二值输出判断是否选择正确动作。
- 正式 checkpoint：Standard step 3289，DINO step 4500，TTT step 2200，Ours step 3100。
- LoRA 仅有 Action=0.5；其 K=2 视频指标未完成，因此 PSNR/SSIM/LPIPS/Object 保持空缺，不与旧 K=1 数值混用。

### 3.6 Mass Balance

- Baseline 根目录：`results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1`。
- Ours 根目录：`results/mass_balance/fixed_pose_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1`。
- K=1，5 ID + 5 OOD，每个 environment 14 个 Query。
- Ours 精确成员：

| Domain/ratio index | Ratio | Support | Query indices |
| --- | ---: | ---: | --- |
| ID 0 | 0.125 | 4 | 1,7,10,13,16,19,22,25,28,31,34,37,40,43 |
| ID 5 | 0.4 | 241 | 226,229,232,235,238,244,247,250,253,256,259,262,265,268 |
| ID 10 | 1.25 | 481 | 451,454,457,460,463,466,469,472,475,478,484,487,490,493 |
| ID 14 | 3.0 | 673 | 631,634,637,640,643,646,649,652,655,658,661,664,667,670 |
| ID 19 | 8.0 | 895 | 856,859,862,865,868,871,874,877,880,883,886,889,892,898 |
| OOD 20 | 0.1538930517 | 907 | 901,904,910,913,916,919,922,925,928,931,934,937,940,943 |
| OOD 22 | 0.3535533906 | 1006 | 991,994,997,1000,1003,1009,1012,1015,1018,1021,1024,1027,1030,1033 |
| OOD 24 | 0.8122523964 | 1105 | 1081,1084,1087,1090,1093,1096,1099,1102,1108,1111,1114,1117,1120,1123 |
| OOD 27 | 2.8284271247 | 1258 | 1216,1219,1222,1225,1228,1231,1234,1237,1240,1243,1246,1249,1252,1255 |
| OOD 29 | 6.4980191708 | 1342 | 1306,1309,1312,1315,1318,1321,1324,1327,1330,1333,1336,1339,1345,1348 |

- 权威 Ours manifest：`results/mass_balance/fixed_pose_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/ours/step_3900/seed_20260723/protocol/evaluation_selection.json`。
- Object：横杆 tilt angle 的平均误差和最终误差，单位度。
- Action：在候选 support-offset/action 顺序上寻找进入 `[-3,3]` 度平衡区间的 boundary crossing。
- 正式 checkpoint：Standard/LoRA/DINO step 4500，TTT step 1673，Ours step 3900。
- 重要限制：Ours 使用 fixed-pose 数据；baseline 使用 workspace-random 数据。这是跨数据集比较，必须在正文或表注中使用星号明确说明。

### 3.7 Mass x Friction

- 正式根目录：`results/mass_friction/joint100_grid_id5_ood5_k1_oracle_informative_support25_60_v1`。
- K=1 GT-selected informative Support；学得的 inference-time `Z` 冻结后复用于同 environment 的 8 个 Query action。
- Support 位移目标为 25-60 px，优先最接近 42.5 px并要求最终可见。

| Domain/env | mass kg | friction | Support | Query indices |
| --- | ---: | ---: | ---: | --- |
| ID env000 | 2.0 | 0.002 | 386 | 378,379,380,381,382,383,384,385 |
| ID env042 | 0.509433585 | 0.013 | 349 | 342,343,344,345,346,347,348,350 |
| ID env044 | 0.302980968 | 0.052 | 242 | 234,235,236,237,238,239,240,241 |
| ID env050 | 0.106250988 | 0.073 | 436 | 432,433,434,435,437,438,439,440 |
| ID env054 | 0.05217117 | 0.15 | 426 | 423,424,425,427,428,429,430,431 |
| OOD env001 | 0.767599888 | 0.003 | 642 | 639,640,641,643,644,645,646,647 |
| OOD env006 | 0.206122244 | 0.013 | 37 | 36,38,39,40,41,42,43,44 |
| OOD env010 | 0.106250988 | 0.035 | 802 | 801,803,804,805,806,807,808,809 |
| OOD env086 | 0.206122244 | 0.043 | 95 | 90,91,92,93,94,96,97,98 |
| OOD env093 | 0.063360541 | 0.073 | 739 | 738,740,741,742,743,744,745,746 |

- 权威 manifest：`results/mass_friction/joint100_grid_id5_ood5_k1_oracle_informative_support25_60_v1/protocol/support_query_manifest.json`。
- Object：目标物 centroid ADE/FDE。
- Action：在终点 normalized-y 阈值 `0.62,0.75,0.92` 上寻找从低到高第一个严格超过阈值的 action。
- 预测 first-crossing rank 与 GT 相同记 1，相差 1 档记 0.5，其他或不跨越记 0。
- 先对阈值等权，再对 GT 可达 environment 等权。
- 当前表采用 raw first-crossing，不采用 isotonic 辅助版本。
- 该 Action 定义是 user-selected post-hoc revision；源 protocol 仍标记为 exploratory。
- 正式 checkpoint：Standard/LoRA step 3277，DINO step 5000，Ours step 7172；TTT-KQV 尚无正式结果，表格保持空值。

## 4. 正式结果来源矩阵

### 4.1 真机

| Task | 正式结果/配置来源 | Support/Query 权威来源 |
| --- | --- | --- |
| Door | `configs/evaluation/real97_door_ours_simlr_mean_20260918_v1.json`; `metrics/door_first_close_level_tolerance1_20260919.json` | `outputs/evaluation_real97_door_dual6500_simlr_cluster_20260918_v1/door_prep/{support,query}.jsonl` |
| Ball | `configs/evaluation/real97_ball_ours_simlr_mean_20260918_v1.json`; `configs/evaluation/real97_ball_simlr_scores_20260918_v1.json` | `outputs/evaluation_real97_ball_dual6500_simlr_cluster_20260918_v1/ball_prep/{support,query}.jsonl` |
| Stick | `configs/evaluation/real97_stick_ours_simlr_mean_20260918_v1.json`; DINO/TTT codecfix configs | `outputs/infer_real916_stick_3900_simlr_globalmean_20260918_v1/input_manifest/{support,query}.jsonl` |
| Soft | `configs/evaluation/real97_soft_ours_simlr_mean_20260918_v1.json`; `metrics/soft_static9_shared6_selected_20260919.json` | `outputs/infer_real97_soft_balanced9_family_mean_static_20260914_v1/{support,query}.jsonl` |

### 4.2 仿真

| Task | 正式结果根目录 | Support/Query 权威来源 |
| --- | --- | --- |
| Push-box Friction | `results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1` | `protocol/support_query_manifest.json` |
| Gravity | `results/gravity/gravity80_uniform5id5ood_strict_v1` | `protocol.json` |
| Mass Collision | `results/mass_collision/noleak_highmass2x_grid_id5_ood5_k1_balanced_visible_or_min_action_v1` | DINO formal `input_support_query_manifest.json` listed above |
| Light Switch | `results/lightswitch/physicalpress33_all4env_support8_query15_v1` | `protocol/support_query_manifest_red1_blue1_matched_action.json` |
| Mass Balance | workspace-random baselines; fixed-pose Ours | Ours `protocol/evaluation_selection.json`; baseline manifests in each method result directory |
| Mass x Friction | `results/mass_friction/joint100_grid_id5_ood5_k1_oracle_informative_support25_60_v1` | `protocol/support_query_manifest.json` |

## 5. 复现要求与已知缺口

仅 clone GitHub 仓库可以恢复代码、配置、正式数字、绝大多数协议和本文件记录的精确 Support/Query 成员，但完整端到端复现还需要：

- 对应数据集及其 train/test split metadata。
- 模型 checkpoint 和 context table。
- DINO feature/checkpoint、LoRA/TTT 中间权重等方法特定 artifact。
- 本文引用的 `outputs/` 原始推理结果和 tracking cache。
- 与原实验一致的 Wan/BWM 基础模型文件。

发布或归档时，至少应为每个正式 checkpoint、context table、Support/Query manifest 和最终 summary 保存 SHA256。当前仿真 FDE 来源已经在 `results/sim_all_methods_main_table_v1/sim_all_methods_main_table_detailed_sources.json` 中记录 SHA256；真机尚未为所有 artifact 建立统一 hash registry。

论文撰写时必须保留以下限制：

- Stick Action 的 seed20260927 36-query subset 是 post-hoc 选择。
- Soft 6-degree onset threshold 是 post-hoc 选择，且尚未完成人工逐帧 onset audit。
- Soft shared-six 是共同环境子集，不是完整 all-nine benchmark。
- Mass Balance Ours 与 baselines 使用不同数据分布。
- Mass Collision LoRA 来自旧但兼容的协议。
- Light Switch LoRA 只有 Action，视频指标缺失。
- Mass x Friction TTT-KQV 缺失。
- Event80 DINO/TTT 是该正式 benchmark 当时的 legacy implementation。

若后续主表数字、Support、Query、checkpoint 或 action 定义发生变化，必须同时更新：主表 CSV、对应 metric snapshot/config、本文档以及可发布的 hash manifest；不得只修改图表。
