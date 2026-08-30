# TTT-Physics Experiment Results

Last updated: 2026-08-25

代码中的 physical context / latent `C` 在论文与图中统一记作 latent `Z`。正文只保留核心发现；逐项配置、checkpoint 和评测资产见附录。

## 正文

### 1. 架构与训练方法结论

| 研究问题 | 对照/代表任务 | 最重要的新发现 | 当前结论 |
|---|---|---|---|
| Test-time adaptation 应更新什么 | TTT-E2E LoRA：`87779 -> 87832`；environment-group latent Z | LoRA fast weights 能带来局部 improvement，但更新幅度、稳定性和生成分布可控性较弱；单独优化低维环境 Z 更直接 | TTT-E2E LoRA 保留为架构基线；environment-group latent Z 作为主方法 |
| 环境记忆如何共享 | 同一物理环境的多个 episode/action 共享 Z，不同环境使用不同 Z | 共享机制迫使 Z 表示跨轨迹稳定的环境规律，而不是某条轨迹的瞬时细节 | 当前数据支持 environment-level memory；更细粒度 episode/state memory 留作后续设计 |
| Stage1 如何联合训练模型和 Z | Model-only 与 Z-only 分阶段 iterative learning | 分阶段冻结可以避免模型和 Z 同时漂移；新环境先快速定位 Z，再做全局细化 | 当前最稳定方案；仍需与 simultaneous optimization 做严格消融 |
| Z 如何初始化 | PushBox `88822` shared 对比 `88823` independent random | Independent random initialization 更容易打破对称性，形成可辨识 latent geometry，并改善 Stage2 迁移 | 新实验默认采用每个环境独立随机初始化 |
| Batch 应如何组织 | Joint mass-friction 失败版与 `91267/step7172` | 仅增大环境数不够；一次更新中让同一环境出现更多不同 actions，能给共享 Z 更一致、更充分的物理监督 | Batch shape 应同时报告 environment 数和每环境 action 数，不能只报告总 batch |
| 视觉背景变化如何处理 | Multi-background 独立 group 对比 shared-friction job `103424` | 将同一 friction 在不同背景下设为同一个 environment group，可直接要求 Z 对背景变化保持不变；修正 LR 后 active Z 已形成 friction-related structure | Shared-friction Z 是当前背景不变性主线；仍需用统一 checkpoint 和评测预算与独立 `(background, friction)` group 做严格对照 |
| Chunk 如何选择 | LightSwitch 旧 event-centered `95498` 与新版 physical-trigger no-leak 数据 | 旧 event-centered crop 后来发现与环境/结果相关的 timing pattern leakage，不能再用于证明短 chunk 的优势；新版只按物理按键触发时刻裁剪并随机 jitter | 关键 chunk information density 仍是合理假设，但必须在新版 no-leak 数据上补固定其他设置的 33/120-frame 严格对照 |
| Flow-matching 空间监督如何分配 | Real Ball Friction full-frame objective 对比 normalized ROI 6x objective | 对球、trough 和接触运动所在的关键区域提高 flow-matching loss 权重后，模型的有效关注明显集中到动力学区域，生成质量大幅提升；学习得到的 Z 在 PCA 中也呈现出更强、更有规律的组织结构 | 对物理事件占画面比例较小的任务，应优先使用 mean-normalized key-region weighting；还需用固定 seed、预算和样本的定量消融分离 ROI weighting 的独立贡献 |

### 2. 代表实验结果

| 方向 | 代表 Stage1 | 代表评测 | 核心配置 | 最重要结果 | 状态 |
|---|---|---|---|---|---|
| PushBox friction，C1 | `88729/step4000` | `88797` | 6 friction；frames 65-105；共享 C1；`4 env x 4 action/rank`；model/C 每 200 steps 交替 | 一维 Z 已能形成 friction 相关次序，但容量和可优化范围有限 | 完成 |
| PushBox friction，shared C32 | `88822/step6814` | 主要使用 C-table/PCA 与 `88823` 比较 | 80 friction；frames 65-105；所有环境从同一 C32 起点开始 | Shared initialization 保留对称性，latent 分离和 Stage2 可辨识性较弱 | 完成；缺严格同样本 Stage2 对照 |
| PushBox friction，random C32 | `88823/step7272` | `89097`、`89098` | 80 friction；independent `U(0,1)`；1000-step curriculum；Stage2 FP32 40 steps | 当前 PushBox 最佳；Z 与 friction 相关，并能从 showcase 迁移到同 friction 的其他 actions | 完成，主结果 |
| Gravity | `89152/step3837` | `89519` | 80 gravity；full 61 frames；C32 random curriculum | 未显式提供 gravity，Z 仍形成 gravity 相关结构并支持 Stage2 | 完成，主结果 |
| Mass collision，旧版（有泄漏） | `90515/step3848` | `90735` | 20 个按 theoretical-distance effect 采样的 mass；9 speeds；full 61 frames | 后续审计发现不同 mass 环境存在稳定的环境相关视觉/episode pattern，模型可能不经物理适应即可识别环境 | 历史结果保留审计；不得作为正式主结果或方法优越性证据 |
| Mass collision，no-leak | `107912/step4300`；pooled `step8000` | `results/mass_collision/noleak_grid_id5_ood5_k1_balanced_visible_or_min_action_v4` | 新 30-mass 数据中按数值匹配旧版 20 个 mass 训练；main view only；9 speeds；Ours 为 C32 random curriculum、每 rank `4 mass x 8 actions`；pooled 为 2 GPUs、每卡 batch 16 | 正式 Action8 Ours：PSNR 31.10、SSIM 0.9639、final-displacement error 8.17 px；均明显优于 pooled 的 29.80、0.9623、25.62 px | 新正式结果；跨任务主表使用 Action8；Action4/ROI10x 仅作同任务消融 |
| Mass balance，workspace random | `91401/step4300` | `91912` | 20 ratios；15 supports；stride 2；`3 env x 6 chunk/rank` | 在 workspace/pose 变化下仍能学习质量分布 latent | 完成，鲁棒性结果 |
| Mass balance，fixed pose | `91441/step4300` | `92059` | 与 random 版相同训练机制，但固定 pose | 减少无关视觉变化后，质量比例与 latent geometry 更容易分析 | 完成，主结果 |
| Joint mass-friction，失败版 | `90527`、`90748 -> 90917` | 无正式主评测 | `4 env x 4 action` 或 `6 env x 3 action/rank`；有效 batch 64/72 | 未形成清晰、可迁移的多因素 latent 结构 | 失败对照 |
| Joint mass-friction，成功版 | `90971 -> 91267/step7172` | `91479` | `4 env x 6 action/rank`；有效 batch 96；new-Z lr `0.09` | Z 在无 factor supervision 下自发组织 mass 与 friction 两个因素 | 完成，主结果 |
| Multi-background PushBox | 原始 random-C9 curriculum `step6000` -> `97909/step8200` | `98506` | 3 backgrounds x 40 frictions；固定前三个 waves 的 21 groups；每 rank `6 env x 6 common actions`；Z/model 每 200 steps 交替 | 每个背景内部出现 friction-related structure，但三个背景仍形成各自 cluster，说明 Z 同时编码了物理与视觉 domain | 当前最佳多背景结果 |
| Multi-background PushBox，shared-friction Z | `103424/step4000`；错位续跑观察点 `104188/step4400` | `104091`；`105521` | 5 backgrounds x 30 frictions；同 friction 跨背景共享 Z32；ROI 10x；all-Z lr `0.03`，new-Z lr `0.09` | 正常 step4000 已出现 friction-related ordering；错位续跑 step4400 形成强单调弧线，但实际多执行 500 updates，不能作为受控主结果 | 正常基线完成；mis-resume 分支保留作机制观察 |
| LightSwitch，旧 event-centered（有泄漏） | `95498`；评测状态为 model 4100 + Z 4300 | `95707` | 4 causal env；按旧规则截取约 30-frame core | 后续发现 crop timing/pattern 与 causal environment 或结果相关，模型可能利用该模式而非按钮因果规律 | 历史结果保留审计；不得作为正式主结果或 chunk-length 证据 |
| LightSwitch，physical-press no-leak | `107549/step3100`；pooled `step3289` | `results/lightswitch/physicalpress33_all4env_support8_query15_v1` | Main view only；33 frames；crop 只由物理按键 trigger 决定并 jitter 到 index 11-22；Ours 每 env 使用 K=8 support，4 red + 4 blue；每个 causal env 15 个 disjoint query | Ours：PSNR 33.45、final light-state accuracy 93.33%、action success 87.5%；pooled：32.62、60.00%、50.0% | 新正式结果；60 query/method 与 30 条 action rollouts 已汇总 |

### 3. 真机实验

| 实验 | 代表训练/状态 | 环境与输入 | 方法 | 当前获得的信息 | 结论边界 |
|---|---|---|---|---|---|
| Real Ball Friction | Stage1 `101378/step5500`；正式评测使用 model/Z `step5500` | 7 个真实 ball-friction environments；每环境抽取 6 个不同 impact/skill levels；60 个真实帧 tail-pad 为 61-frame 输入；14D joint-state-action | Independent Z32；4 GPUs；每 rank `3 env x 6 skill`；1000-step-style curriculum；trough ROI 使用归一化 6x loss weight；Stage2 context-only adaptation | 完成 42 组 GT/Stage1/Stage2 生成、7-environment x 6-level grid 和 Z trajectory；关键区域加权使模型关注集中到球-槽接触动力学，生成效果相对 full-frame objective 大幅提升，PCA latent geometry 呈现出很强的规律性 | 正式生成评测已完成；当前 Stage2 仍以 query chunk 自身作为 support，尚未证明独立 showcase 到 query 的跨 episode adaptation 或闭环控制；ROI 收益仍需统一指标的严格消融 |
| Real Stick Balance | Stage1 `98364/step5500`；eval `98738` | 8 个真实左右配重环境；每环境 6 个 episode；raw 120 frames、stride 3 得到 41-frame 输入；14D joint-state-action | Independent Z32；4 GPUs；每 rank `3 env x 6 action`；80% 采样关键 lift windows；Stage1 model/Z 交替训练；Stage2 context-only adaptation | 完成 48 组 GT/Stage1/Stage2、8 个 environment grids、training/inference Z 联合 PCA；不同配重环境具有可分析的 latent structure | 正式生成与 latent 评测已完成；当前 Stage2 仍为 support=query，后续需补 disjoint support/query 和横杆倾角定量指标 |

### 4. 尚未形成主结果的方向

| 方向 | 目标 | 当前状态 | 成为主结果前必须完成的内容 |
|---|---|---|---|
| Multi-background / multi-visual-environment PushBox | 已获得 friction-related representation，但不同背景仍各自聚类 | 下一步使用共享 Z 的跨背景、光影和视觉干扰 augmentation，显式约束同一物理条件的不同视觉版本使用同一个 latent | 仿真中可重渲染同一轨迹；真实数据很难保持动作、状态、接触过程和时间严格对齐，需要先研究可控视觉合成或同步采集 |

### 5. 必须补充的严格消融

| 优先级 | 消融 | 固定项 | 唯一变化 | 要回答的问题 |
|---|---|---|---|---|
| P0 | LightSwitch chunk length | 新版 physical-trigger no-leak 数据、groups、batch shape、curriculum、LR、seed | 33-frame physical-event window 对比 120-frame window | 排除 timing leakage 后，收益是否仍主要来自更高的关键事件信息密度 |
| P0 | Joint mass-friction batch shape | 数据、group order、LR、curriculum、总训练步数 | `6x3`、`4x6`、`4x9` env/action shape | 改善来自总 batch、同环境 action diversity，还是二者共同作用 |
| P1 | Iterative vs simultaneous | 数据、初始化、总优化预算 | 分阶段冻结对比模型/Z 同时更新 | Iterative learning 是否是必要条件 |
| P1 | Multi-background factorization | 相同训练和评测数据 | single Z 对比 environment-Z + background-Z | 模型能否分离物理属性与视觉 domain |

---

# 附录：配置、Checkpoint 与评测资产

## A. 路径和通用约定

| 项目 | 绝对路径或约定 |
|---|---|
| 项目根目录 | `/hai/scratch/cyzhou05/projects/TTT-Physics` |
| BWM-Adapt | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt` |
| 训练输出根目录 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs` |
| Runtime config | `tmp/run_configs` 中带 job ID 的 YAML，是实际提交任务使用的配置副本 |
| Canonical config | `configs/train` 中不带 job ID 的 YAML，用于版本控制和复现 |
| 配对 checkpoint | Group-latent 实验必须同时指定 `.safetensors` 和同状态 `.context_table.json`；不能只恢复模型 |
| Stage2 通用设置 | 除单独注明外，冻结模型，仅更新 Z；从 active Z table 均值初始化；FP32；40 steps：`3.0x10 + 1.5x10 + 0.5x10 + 0.15x10` |
| 主要评测 | GT/Stage1/Stage2 视频、ID/OOD、同环境跨 action transfer、Z trajectory、PCA/grid |

## B. TTT-E2E LoRA baseline

### B.1 配置与关键参数

| 字段 | Stage1 job 87779 | Stage2 job 87832 |
|---|---|---|
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/stage1_adapter_warmup_all_layers_gate007_4000_87779.yaml` | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/stage2_warmup_lasthalf_sched06_03_01_003_hybrid_q1_s02_imp5_500step_87832.yaml` |
| 数据/窗口 | 6-friction PushBox，81 frames | 相同 support/query 数据组织 |
| Fast parameters | 所有 DiT 层 residual adapter；rank 16；每层独立 gate，init 0.07 | Inner loop 只更新 low-rank adapter/gate |
| Inner loop | 无 | 20 steps：`0.60x5, 0.30x5, 0.10x5, 0.03x5`；grad clip 1.0 |
| Outer trainable | DiT、action encoder、adapter bank | Adapter bank 与后半部分 DiT |
| Learning rate | `1e-5`，4000 steps | `1e-5`，500 steps |
| Objective | Stage1 behavior-cloning/flow loss | `query + 0.2 show - 5(rel_query_improve + 0.2 rel_show_improve)` |

### B.2 Checkpoint 与评测

| 资产 | 绝对路径/说明 |
|---|---|
| Stage1 checkpoint | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/push_box_6fric_hidden_straight_stage1_adapter_warmup_all_layers_gate007_4000_87779/step-4000.safetensors` |
| Stage2 checkpoint | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/push_box_6fric_stage2_warmup_lasthalf_sched06_03_01_003_hybrid_q1_s02_imp5_500step_2gpu_87832/step-500.safetensors` |
| 统一评测 job | `88297` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/compare_5way_10cases_x5s1920_88297` |
| 覆盖范围 | 10 cases、5-way 生成比较；用于与 no-C、oracle-C、fixed-C128 和 latent-C 路线并排观察 |

## C. PushBox friction

### C.1 C1 grouped latent：job 88729

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_push_box_6fric_65_105_grouped_c1_sharedinit_alternating4f4a_stage1_6k.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/push_box_6fric_65_105_grouped_c1_sharedinit_alternating4f4a_stage1_6k_2gpu_88729.yaml` |
| 数据 | 6 friction：`0.005, 0.01, 0.02, 0.05, 0.10, 0.15` |
| 视频 | 固定 frames `65-105`，41 frames |
| Z | dim 1；同 friction 共享；所有 groups 初始 `0.5`；不输入真实 friction |
| Batch shape | 每 rank `4 friction x 4 action = 16`；2 GPUs，有效 32 chunks/update |
| Optimizer schedule | Model-only 200 与 Z-only 200 交替；model lr `1e-5`；Z lr `0.03`；计划 6000 updates |
| Model trainable | DiT、action encoder、physical-context encoder；Z-only phase 严格冻结模型 |

| 资产 | 绝对路径/说明 |
|---|---|
| 配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/push_box_6fric_65_105_grouped_c1_sharedinit_alternating4f4a_stage1_6k_2gpu_88729/step-4000.safetensors` |
| 配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/push_box_6fric_65_105_grouped_c1_sharedinit_alternating4f4a_stage1_6k_2gpu_88729/step-4000.context_table.json` |
| 评测 job | `88797` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_6fric_65_105_grouped_c1_c32_10cases_88797` |
| 覆盖范围 | 10 cases；GT、C1/C32 Stage1 生成和 latent 对照 |

### C.2 Shared C32：job 88822

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_push_box_event_tap_segmented80_pushmotion41f_10chunk_curriculum_c32_shared_stage1_15300.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/curc32s65ib2-nc015_88822.yaml` |
| 数据/视频 | Event-tap segmented 80-friction、约 10 actions/group；frames `65-105`，41 frames |
| Z | dim 32，1 token，hidden dim 128；初始 groups 共享同一个 `U(0,1)` 随机向量 |
| Curriculum | 初始 5 groups；每轮增加 5；初始 model-only 300 |
| 1000-step cycle | new-Z 200@0.15；all-Z 200@0.03；model 200@1e-5；all-Z 200@0.03；model 200@1e-5 |
| Batch shape | 每 rank `4 friction x 4 action = 16` |

| 资产 | 绝对路径/说明 |
|---|---|
| 配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/curc32s65ib2-nc015_88822/step-6814.safetensors` |
| 配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/curc32s65ib2-nc015_88822/step-6814.context_table.json` |
| 独立 Stage2 评测 | 尚无与 `88823` 完全同样本、同 protocol 的独立 Stage2 评测 |
| 已有评估 | 使用配对 C-table/PCA 和生成结果观察 shared initialization；结论应作为初始化消融趋势，不作为完整主结果 |

### C.3 Independent-random C32：job 88823

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_push_box_event_tap_segmented80_pushmotion41f_10chunk_curriculum_c32_random_stage1_15300.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/curc32r65ib2-nc015_88823.yaml` |
| 与 88822 的控制变量 | 数据、窗口、curriculum、batch shape、LR 相同 |
| 唯一核心变化 | 每个 environment 的 Z32 独立从 `U(0,1)` 初始化 |
| Stage2 | 冻结模型；active-table mean init；FP32；40 steps：`3.0/1.5/0.5/0.15` 各 10 steps |

| 资产 | 绝对路径/说明 |
|---|---|
| 最佳配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/curc32r65ib2-nc015_88823/step-7272.safetensors` |
| 最佳配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/curc32r65ib2-nc015_88823/step-7272.context_table.json` |
| ID/OOD 评测 job | `89097` |
| ID/OOD 输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_event80_randomc32_stage2_x5fp32_id10_ood10_89097` |
| ID/OOD 覆盖 | 10 个 ID friction、10 个 OOD friction；GT、Stage1、Stage2、Z trajectory/PCA |
| Physics-transfer job | `89098` |
| Physics-transfer 输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_event80_randomc32_stage2_x5fp32_physics_transfer_4mu_89098` |
| Transfer 覆盖 | 用 showcase action 学 Z，再在同 friction 的其他 actions 上生成；包含 grid 与 latent trajectory |

## D. Gravity：job 89152

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_gravity80_full61_curriculum_c32_old_random_stage1_16300.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/gravity80_full61_c32_oldrandom_s1_89152.yaml` |
| 数据 | 80 gravity groups；不向模型提供 gravity 数值 |
| 视频 | 完整 launch-to-landing 61 frames；短轨迹 padding |
| Z | C32，独立 `U(0,1)`；physical-context mode `both` |
| Curriculum | 初始 5，每轮加 5；model warm-up 300；new-Z 0.15；all-Z 0.03；model 1e-5 |
| Batch shape | 每 rank `4 gravity x 4 action = 16` |
| Stage2 | Mean-table init；FP32 40 steps；同 gravity 跨 action transfer |

| 资产 | 绝对路径/说明 |
|---|---|
| 配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/gravity80_full61_c32_oldrandom_s1_89152/step-3837.safetensors` |
| 配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/gravity80_full61_c32_oldrandom_s1_89152/step-3837.context_table.json` |
| 评测 job | `89519` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_gravity80_oldrandom_step3837_stage1_stage2_grid_89519` |
| 覆盖范围 | GT、Stage1、Stage2、PCA、grid、同 gravity 其他动作 |

## E. Mass collision

### E.1 旧 linear-theory-distance：job 90515（有泄漏，历史审计）

> **警告：该数据后来确认存在环境相关视觉/episode pattern leakage。** 同一 mass 环境内的 episode 具有稳定可辨识模式，使模型可能从初始画面或非物理 nuisance 识别环境。下面的 checkpoint、PCA 和 `90735` 仅为历史审计资产，不得用于正式比较、物理泛化结论或论文主表。

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_mass20_collision_linear_theory_distance_full61_curriculum_c32_old_random_stage1_4300.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/mass20_linear_theory_distance_full61_c32_oldrandom_s1_90515.yaml` |
| 数据 | 20 target masses，按理论碰撞位移效应采样；每 mass 有 9 impact speeds |
| 分组 | 同一 mass 下所有 speeds 共享 Z；不输入真实 mass |
| 视频 | Full collision rollout，61 frames |
| Z/Curriculum | C32 independent `U(0,1)`；初始 5，每轮加 5；new-Z 0.15；all-Z 0.03；model 1e-5 |
| Batch shape | 每 rank `4 mass x 4 speed/action = 16` |
| Stage2 | 用一个 speed 学 Z，迁移到同 mass 的另外 8 个 speeds |

| 资产 | 绝对路径/说明 |
|---|---|
| 配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass20_linear_theory_distance_full61_c32_oldrandom_s1_90515/step-3848.safetensors` |
| 配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass20_linear_theory_distance_full61_c32_oldrandom_s1_90515/step-3848.context_table.json` |
| 评测 job | `90735` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_mass20_linear_theory_distance_day1_c32_stage1_stage2_grid_90735` |
| 覆盖范围 | GT、Stage1、Stage2、mass PCA、9-speed transfer grid |

### E.2 No-leak main-view Mass Collision：jobs 107912 / pooled checkpoint step8000

| 字段 | 新正式配置 |
|---|---|
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass/libero_two_box_collision_9speed_30mass_linear_theory_distance_noleak_270eps_lerobot_2026-08-27_hai-machine` |
| Leakage control | 重建环境与 episode，移除旧版可稳定识别 mass 的视觉/轨迹 pattern；仅使用主视角；训练 mass 按物理数值匹配旧版 20 档，而不是简单取前 20 个 index |
| Ours canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_mass20of30_collision_noleak_mainview_full61_curriculum_c32_old_random_8action_stage1_4300.yaml` |
| Ours checkpoint | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/mass20of30_collision_noleak_mainview_c32_oldrandom_8action_s1_107912/step-4300.safetensors`；配对 Z table 为同目录 `step-4300.context_table.json` |
| Ours training | C32 independent random curriculum；20 active masses；9 speeds；full 61 frames；每 rank `4 mass x 8 actions = 32`；model lr `1e-5`，new-Z lr `0.15`，all-Z lr `0.03` |
| Pooled canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_mass20of30_collision_noleak_mainview_standard_pooled_wm_2gpu_24h.yaml` |
| Pooled checkpoint | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/method_benchmarks/mass_collision_noleak_original20/standard_pooled_wm/seed_20260827/checkpoints/step-8000.safetensors`；2 GPUs，batch 16/rank，无 Z/adaptation |
| 正式协议 | 5 ID + 5 OOD masses；K=1 informative visible support；每个环境 8 个其他 actions；同一冻结协议比较 Ours、ROI10x 和 pooled |
| 正式结果根目录 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/results/mass_collision/noleak_grid_id5_ood5_k1_balanced_visible_or_min_action_v4` |

| 方法 | PSNR ↑ | SSIM ↑ | Final-displacement error ↓ | Action success ↑ |
|---|---:|---:|---:|---:|
| Ours，Action8（正式） | **31.10** | **0.9639** | **8.17 px** | 26.7% |
| Standard pooled WM | 29.80 | 0.9623 | 25.62 px | 23.3% |

Action 的 oracle-reachable 子集上，Action8 Ours 为 44.4%，pooled 为 38.9%。Action4 与 ROI10x Action4 保留为同任务消融，不进入跨任务主表。当前 aggregate 可用；早期汇总曾将全部环境错误标为 `ood`，因此 ID/OOD 分项在进入论文表格前仍需完成 metadata 审计，不能直接引用该分项。

## F. Mass balance

### F.1 Workspace-random：job 91401

| 字段 | 配置 |
|---|---|
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/mass_balance_c32_oldrandom_stride2_stage1_4300_91401.yaml` |
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_workspace_random_20ratio_15support_300eps_direct_approach_absolute_eef_lerobot_2026-07-22_hai-machine` |
| 分组 | 20 mass ratios；同 ratio 共享 C32；15 support settings |
| Raw windows | `[0,80)`、`[40,120)`、`[70,150)` |
| Temporal sampling | 每 2 帧采 1 帧；视频/action 同索引；模型输入 41 frames |
| Batch shape | 每 rank `3 ratio x 6 chunks = 18`；2 GPUs，有效 36 |
| Chunk constraint | 每组 6 chunks 至少 4 个来自后两个、包含主要动力学的 windows |
| Curriculum | 4300 steps；new-Z 0.15；all-Z 0.03；model 1e-5 |

| 资产 | 绝对路径/说明 |
|---|---|
| 配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass_balance_c32_oldrandom_stride2_stage1_4300_91401/step-4300.safetensors` |
| 配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass_balance_c32_oldrandom_stride2_stage1_4300_91401/step-4300.context_table.json` |
| 评测 job | `91912` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_mass_balance_random_91401_stage1_stage2_grid_91912` |
| 覆盖范围 | GT、Stage1、Stage2、ratio PCA、same-ratio other-support grid |

### F.2 Fixed-pose：job 91441

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_mass_balance_fixed_pose_20ratio_stride2_curriculum_c32_old_random_stage1_4300.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/mass_balance_fixed_pose_c32_oldrandom_stride2_stage1_4300_91441.yaml` |
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_fixed_pose_20ratio_15support_300eps_direct_approach_absolute_eef_lerobot_2026-07-22_hai-machine` |
| 其他训练参数 | 与 workspace-random 版相同：20 ratios、stride 2、三个 raw windows、`3 ratio x 6 chunks/rank`、4300 steps |

| 资产 | 绝对路径/说明 |
|---|---|
| 配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass_balance_fixed_pose_c32_oldrandom_stride2_stage1_4300_91441/step-4300.safetensors` |
| 配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass_balance_fixed_pose_c32_oldrandom_stride2_stage1_4300_91441/step-4300.context_table.json` |
| 评测 job | `92059` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_mass_balance_fixed_pose_91441_stage1_stage2_grid_92059` |
| 覆盖范围 | GT、Stage1、Stage2、ratio PCA、同 ratio 的 8-support transfer grid |

## G. Joint mass-friction

### G.1 失败配置

| Job | Runtime/Canonical config | Batch shape | 其他关键参数 | Checkpoint/结果 |
|---|---|---|---|---|
| `90527` | Runtime: `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/mass_friction100_joint_full61_c32_oldrandom_s1_90527.yaml`；Canonical: `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_mass_friction100_joint_full61_curriculum_c32_old_random_stage1_20300.yaml` | `4 env x 4 action/rank`；4 GPUs；有效 64 | 初始/新增 5 groups；new-Z 0.15 | 代表 pair 为 `outputs/mass_friction100_joint_full61_c32_oldrandom_s1_90527/step-3849.{safetensors,context_table.json}`；未形成清晰多因素结构 |
| `90748 -> 90917` | Runtime: `tmp/run_configs/mass_friction100_c32_oldrandom_4h200_pergpu6e3a_add12_90748.yaml` 与 `tmp/run_configs/mass_friction100_c32_oldrandom_4h200_pergpu6e3a_add12_resume4751_90917.yaml`；Canonical: `configs/train/train_mass_friction100_joint_full61_curriculum_c32_old_random_4gpu_6env3action_add12_stage1_9300.yaml` | `6 env x 3 action/rank`；4 GPUs；有效 72 | 初始/新增 12 groups；new-Z 0.15 | 主要恢复点 `90748/step4751`，resume 到 `90917/step4800`；同环境 action diversity 仍不足 |

上表中的相对路径均以 `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/` 为根目录。

### G.2 成功配置：jobs 90971 -> 91267

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_mass_friction100_joint_full61_curriculum_c32_old_random_4gpu_4env6action_add12_newlr009_stage1_9300.yaml` |
| Initial runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/mass_friction100_c32_oldrandom_4h200_pergpu4e6a_add12_newlr009_90971.yaml` |
| Resume runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/mass_friction100_c32_oldrandom_4h200_pergpu4e6a_add12_newlr009_resume90971_latestpair_91267.yaml` |
| 数据 | 100 mass-friction combinations，9 common actions，共 900 episodes；不输入 mass/friction labels |
| Z | 每个 combination 共享 C32；independent random init |
| Curriculum | 初始 12，每轮加 12；new-Z lr `0.09`；all-Z `0.03`；model `1e-5` |
| Batch shape | 每 rank `4 env x 6 common actions = 24`；4 GPUs；有效 96 |
| Resume | 从 `90971/step3837` 的模型和同 step Z table 配对恢复 |
| 代表状态 | `step7172`，84 active environments |

| 资产 | 绝对路径/说明 |
|---|---|
| 最佳配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass_friction100_c32_oldrandom_4h200_pergpu4e6a_add12_newlr009_resume90971_latestpair_91267/step-7172.safetensors` |
| 最佳配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/mass_friction100_c32_oldrandom_4h200_pergpu4e6a_add12_newlr009_resume90971_latestpair_91267/step-7172.context_table.json` |
| 评测 job | `91479` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_mass_friction100_backup_step7172_stage1_stage2_id3_ood3_91479` |
| 覆盖范围 | 3 ID + 3 OOD combinations；GT、Stage1、Stage2；PCA 同时以 friction hue 和 mass lightness 编码 |
| 因果限制 | 相对失败版同时改变了 batch shape、有效 batch 和 new-Z lr，当前只能说结果强烈支持 batch-shape 假设，不能宣称已完成单变量证明 |

## H. Multi-background / multi-visual environment

### H.1 数据与表示

| 字段 | 配置 |
|---|---|
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox_various_env/libero_plus_push_box_event_tap_segmented40_10action_3env_hidden_lerobot_A500_offset160_stop_2026-07-27_hai-machine` |
| Metadata | `data/push_box_bwm_various_env3x40_10action_65_105_20260727/train.jsonl` |
| 环境 | 3 个视觉 backgrounds，每个背景 40 个 friction settings，共 120 environment groups；每组 10 actions |
| 视频 | 固定 frames `65-105`，41 frames，224x224 |
| Action | `eef_delta`，14 dimensions，AdaLN 注入 |
| Z | 每个 background-friction group 使用一个 Z32；1 token，hidden dim 128；同时经 condition token 与 modulation 注入 DiT |
| 重要限制 | 当前不是 factorized representation；同一数值 friction 在不同背景下仍使用不同 Z，因此训练目标没有强制背景不变性 |

### H.2 第一阶段：原始 random-C9 curriculum

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_push_box_various_env3x40_10action_65_105_singlec32_from_blm_initialmedian3_randomc9_model800_assign200_add6_stage1_21000_4gpu.yaml` |
| Base checkpoint | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/ckpt/BLM/step-12000.safetensors` |
| 初始 active groups | 9 groups，每个背景选择 3 个；按 friction rank 做 stratum-median 选择 |
| Z initialization | 9 个独立 `U(0,1)` Z32；random-context model warm-up 后固定分配给 9 个初始 groups |
| Initial training | Model-only random-Z warm-up 800 steps，随后 assignment model phase 200 steps |
| Curriculum expansion | 每 wave 新增 6 groups，每个背景 2 个；group order 使用 OS randomness、without replacement，并持久化防止 requeue 后重抽 |
| Curriculum phase lengths | New-Z 200；all-Z 200；model 200；之后继续 all-Z/model 交替 |
| Sampling | `stratified_common_actions`，以 `environment_index` 分 3 strata；每 rank `3 groups x 6 common actions = 18` chunks |
| Learning rates | Z `0.3`；model `1e-5`；Z clamp `[0,1]` |
| 第一阶段输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/push_box_various_env3x40_10action_65_105_singlec32_from_blm_initialmedian3_randomc9_model800_assign200_add6_stage1_21000` |
| 交接模型 | 上述目录的 `step-6000.safetensors` |
| 交接 Z table | 上述目录的 `step-6000.context_table.json` |

### H.3 第二阶段：固定 21 groups 长期联合优化，job 97909

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_push_box_various_env3x40_10action_65_105_from_original_step6000_fixed_first3curr21_post_c200_m200_perrank6env6action_stage1_21000_4gpu.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/slurm/pb3e-s6-f21-6x6_97909_r0.yaml` |
| 恢复方式 | 从第一阶段 `step6000` 的模型与 Z table 配对恢复；logical step 继续从 6000 计数 |
| 固定 active set | 只保留前三个 curriculum waves：初始 9 groups + 两次新增 6 groups = 21 groups；每个背景 7 个 friction groups；不再加入其余 99 groups |
| Active group IDs | `10,14,28,41,59,69,80,104,108,38,20,55,54,90,105,19,35,44,68,87,99`；顺序持久化于输出目录的 `curriculum_group_order.json` |
| Post-curriculum cycle | All-Z-only 200 steps，随后 model-only 200 steps，持续交替到训练结束 |
| Z-only phase | 模型完全冻结；更新 active Z；lr `0.3`；clamp `[0,1]` |
| Model-only phase | 所有 Z 冻结；训练 DiT、action encoder、physical-context encoder；lr `1e-5` |
| Batch shape | 每 rank `6 groups x 6 common actions = 36` chunks；4 GPUs，有效 144 chunks/structured update；按三个 backgrounds 分层采样 |
| 训练预算 | Logical target `21000`；当前代表评测状态为 `step8200` |
| 训练输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/push_box_various_env3x40_from_original_step6000_fixed_first3curr21_post_c200_m200_perrank6env6action_stage1_21000_97909` |
| Lineage manifest | 上述目录的 `source_lineage.json`，明确记录第一阶段输出、模型和 Z table |

### H.4 Checkpoint 与评测

| 资产 | 绝对路径/说明 |
|---|---|
| 代表模型 backup | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/backup-push_box_various_env3x40_fixed21_step8200_from97909/step-8200.safetensors` |
| 代表 Z table backup | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/backup-push_box_various_env3x40_fixed21_step8200_from97909/step-8200.context_table.json` |
| Backup manifest | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/backup-push_box_various_env3x40_fixed21_step8200_from97909/backup_manifest.json` |
| 评测 job | `98506` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_push_box_various_env3x40_fixed21_6x6_step8200_gt_stage1_stage2_x5fp32_12each_grid_pca_98506` |
| Case selection | 每个背景 12 cases，共 36；覆盖该背景全部 7 个 active friction groups，再从这些 groups 选择不同 action repeats |
| Stage1 | 每个 case 使用匹配的 training-time Z table entry |
| Stage2 initialization | 使用 21 个 active training-time Z entries 的均值；评测 manifest 中遗留的“30”是旧文案，实际 active ID 列表为 21 个 |
| Stage2 support/update | 当前评测使用 query chunk 自身作为 support；context-only FP32；40 steps：`3.0x10 + 1.5x10 + 0.5x10 + 0.15x10`；reg 0.001 |
| Video output | 每背景一个 GT/Stage1/Stage2 `3x10` grid，并保存全部 36 个逐 case comparison videos |
| Latent output | `combined_3background_training_inference_z_pca.csv/.svg`、每背景 context trajectory PCA，以及 training-time active21 PCA |

### H.5 结果与下一步

| 项目 | 记录 |
|---|---|
| 当前最佳结果 | Z 空间能够产生与 friction 相关的连续特征，Stage1/Stage2 生成也能表现出相应动力学差异 |
| 主要问题 | 三个视觉 backgrounds 仍各自形成 cluster；background identity 尚未从 friction representation 中消除，说明单一 Z 同时承载物理和视觉 domain 信息 |
| 下一步方法 | 对同一条物理轨迹生成不同背景、光照、阴影和视觉干扰版本，并强制这些 augmented chunks 共享同一个 Z；这样可以直接把“视觉变化不应改变物理 latent”写进训练约束 |
| 仿真实现 | 固定状态、动作和 friction，仅改变 renderer/background/light；可以获得时间严格对齐的 counterfactual visual views，最适合先做 shared-Z augmentation ablation |
| 真实数据难点 | 很难在不同背景和光照下重复完全相同的状态、动作与接触过程；普通独立重采集会把物理轨迹差异混入 augmentation。可优先尝试背景替换、颜色/光照扰动、阴影/遮挡合成，或同步多相机采集 |
| 后续严格评测 | 增加 unseen background、unseen friction、cross-action transfer；比较无 augmentation、shared-Z augmentation、single-Z 与 factorized environment-Z/background-Z |

### H.6 Shared-friction multi-background：jobs 103424 / 104188

#### H.6.1 数据与表示

| 字段 | 配置 |
|---|---|
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox_various_env/libero_plus_push_box_event80_matched_physics_5randombackground_30friction_10action_1500eps_adaptive_end_2026-08-19_hai-machine` |
| Metadata | `data/push_box_bwm_matchedphysics5bg30fric10action_65_105_shared_friction30_20260819/train.jsonl` |
| 数据 | 5 个随机视觉 backgrounds，30 个 friction settings，10 actions，共 1500 episodes |
| Environment group | 仅由 friction 决定，共 30 groups；同一 friction 在 5 个 backgrounds 下共享同一个 Z32 |
| Sampling | 采样 friction 与 action 后，从对应的 5 个 backgrounds 中随机选择视频；background 不再拥有独立 Z |
| 视频 | 固定 frames `65-105`，41 frames；不足 105 帧时使用最后一帧 padding |
| 与 H.1 的区别 | H.1 为 120 个独立 `(background, friction)` groups；本实验为 30 个 shared-friction groups，显式要求 Z 对视觉背景保持不变 |

#### H.6.2 修正 LR 后的标准训练：job 103424

| 字段 | 配置 |
|---|---|
| Canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_4gpu_3fric6action_initial5_add5_original1000_tailc200m200_alllr003_newlr009_model300_stage1_8700.yaml` |
| 输出目录 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_original1000_tailc200m200_alllr003_newlr009_model300_3fric6action_stage1_8700_103424` |
| Z | dim 32，1 token；每个 friction 一个 Z；independent `U(0,1)` initialization；clamp `[0,1]` |
| Batch shape | 每 rank `3 friction x 6 action = 18`；4 GPUs；有效 72 chunks/update |
| Initial model phase | model-only 300 steps；前 100 steps 做 model-lr warm-up |
| Curriculum groups | initial 5；每 wave add 5；最多 30 groups |
| Learning rate | model `1e-5`；all-Z `0.03`；new-Z `0.09` |
| ROI objective | 仅主相机 view 0 的 polygon `94,92;142,92;154,224;78,224` 使用 10x flow-matching weight；整张 loss map 做 mean normalization；wrist view 保持 1x |
| Checkpoint | 每 100 steps 保存，滚动保留最新 2 个；step 4500 为永久保护点 |

每个 curriculum 共 1400 steps：

| 顺序 | 阶段 | Steps | Learning rate |
|---:|---|---:|---:|
| 1 | new-Z only | 200 | `0.09` |
| 2 | all-Z only | 200 | `0.03` |
| 3 | model only | 200 | `1e-5` |
| 4 | all-Z only | 200 | `0.03` |
| 5 | model only | 200 | `1e-5` |
| 6 | all-Z only | 200 | `0.03` |
| 7 | model only | 200 | `1e-5` |

此前一部分 shared-friction 实验误将 all-Z lr 设为 `0.3`，或将 new-Z lr 设为 `0.9`。job 103424 使用的 `all-Z=0.03 / new-Z=0.09` 是本系列的修正基线。

#### H.6.3 Checkpoint、PCA 与标准评测

| 资产 | 绝对路径/说明 |
|---|---|
| 最新完整模型 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_original1000_tailc200m200_alllr003_newlr009_model300_3fric6action_stage1_8700_103424/step-4000.safetensors` |
| 配对 Z table | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_original1000_tailc200m200_alllr003_newlr009_model300_3fric6action_stage1_8700_103424/step-4000.context_table.json` |
| Curriculum order | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_original1000_tailc200m200_alllr003_newlr009_model300_3fric6action_stage1_8700_103424/curriculum_group_order.json` |
| 评测 job | `104091` |
| 评测输出 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/infer_push_box_matchedphysics5bg30fric_roi10x_alllr003_step4000_gt_stage1_stage2_x5fp32_10cases_grid_pca_104091` |

| Step | Active groups | PC1 | PC2 | PC1+PC2 | Pearson corr(`friction`, PC1) | Rank corr |
|---:|---:|---:|---:|---:|---:|---:|
| 700 | 5 | 38.0% | 25.8% | 63.8% | 0.171 | 0.300 |
| 1700 | 5 | 38.3% | 30.6% | 68.9% | 0.156 | 0.400 |
| 3500 | 15 | 44.1% | 41.8% | 86.0% | 0.619 | 0.825 |
| 4000 | 15 | 52.6% | 38.8% | 91.4% | 0.210 | 0.804 |

step 4000 的 Pearson correlation 受 `group 14 / friction=0.07` 离群点影响，但 rank correlation 仍为 0.804。PCA 资产位于训练输出目录的 `pca/step700`、`pca/step1700`、`pca/step3500` 和 `pca/step4000`。

| Stage2 字段 | 配置/结果 |
|---|---|
| Cases | 5 个 backgrounds 中的 10 个 active-friction cases |
| Stage1 | 使用匹配 friction 的 training-time Z table entry |
| Stage2 initialization | 15 个 active Z entries 的均值 |
| Support/update | support=query；context-only FP32；ROI-weighted support flow loss |
| Inner schedule | 40 steps：`3.0x10 + 1.5x10 + 0.5x10 + 0.15x10`；gradient clip 1；reg 0.001；clamp `[0,1]` |
| Z displacement | 40 steps 后平均 `||Z_final-Z_initial||=0.196` |
| Target-table distance | 平均从 0.588 降至 0.474；10 cases 中 9 个更接近对应 training-time Z |
| Exception | `friction=0.07` 从 1.693 增至 1.720 |
| 生成 | 10 个 Stage1、10 个 Stage2、10 个 GT/Stage1/Stage2 comparisons，以及 grid/PCA |

当前 Stage2 endpoints 整体向对应 training-time Z 靠近，但仍集中在 Z 空间中央。Stage1 使用匹配的 training-time Z，生成效果整体优于当前 Stage2；当前 x5 schedule 能推动 Z，但仍属于高方差的探索性设置。

#### H.6.4 错位续跑观察：job 104188

| 字段 | 记录 |
|---|---|
| 输出目录 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_resume4000_stage1_8700_104188` |
| Load source | job 103424 的 `step-4000.safetensors` 与 `step-4000.context_table.json` |
| 错误 | 脚本将逻辑 `resume_step` 设为 3500，而不是 4000 |
| 有效配对 | `step-4400.safetensors` 与 `step-4400.context_table.json` |
| 失败位置 | labeled step 4500；protected-checkpoint copy 在常规模型 checkpoint 写盘前执行 |
| PCA | PC1 81.2%，PC2 11.1%，累计 92.3%；Pearson 0.961；rank 0.993，形成清晰单调弧线 |
| PCA 资产 | 输出目录下 `pca/step4500/active_context_pca.png/.svg/.csv`；该处 Z table 与 step 4400 配对表一致 |
| 推理 job | `105521` |
| 推理输出 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/infer_push_box_matchedphysics5bg30fric_roi10x_misresume104188_step4400_gt_stage1_stage2_x5fp32_10cases_grid_pca_105521` |

labeled step 4400 实际执行了以下 900 次更新，而不是正确续跑应有的 400 次更新：

| 逻辑区间 | 更新对象 | Updates |
|---|---|---:|
| 3501-3700 | model | 200 |
| 3701-3900 | all-Z | 200 |
| 3901-4100 | model | 200 |
| 4101-4300 | all-Z | 200 |
| 4301-4400 | model | 100 |

相对于正确的 source step 4000 → step 4400，该分支额外执行了 300 次 model updates 和 200 次 all-Z updates。强单调弧线可能来自这部分额外优化，因此该结果只作为机制假设和后续 schedule ablation 的证据，不能作为与 job 103424 公平比较的主结果。其模型、Z table 和推理目录保留，不覆盖、不删除，并统一标记为 `mis-resume`。

#### H.6.5 正确续跑与存档隔离

| Job | 输出目录/状态 |
|---|---|
| `105512` | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_resume4000_stage1_8700_105512`；已改为正确 `resume_step=4000`，但首次更新前因 `phase_ended` `NameError` 退出；目录保留用于审计 |
| `105524` | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/push_box_matchedphysics5bg30fric_shared_c32_random_roi10x_agent_resume4000_stage1_8700_105524`；修复异常与 protected-checkpoint 写盘顺序后的正确续跑分支 |

job 105524 只读加载 job 103424 的 step 4000 配对，不覆盖源目录。正确 phase 为：4001-4100 model，4101-4300 all-Z@0.03，4301-4500 model；step 4500 保存正常配对与永久保护副本；4501-4700 以 new-Z@0.09 进入第四个 curriculum。

job 103424 的 step 4000 保存了模型和 Z table，但没有 AdamW optimizer moments。因此 job 105524 的模型/Z 状态连续，optimizer state 重新初始化；后续需要严格 optimizer-level resume 的训练必须同步保存 optimizer state。所有续跑均使用带 job ID 的独立输出目录，job 103424、104188、105512 和 105524 互不覆盖。

## I. LightSwitch causal dynamics

> **旧 job 95498/95707 后来确认存在 crop timing/pattern leakage。** 旧资产继续保留，但旧结果不再支持“短关键 chunk 更好”或“模型已学习按钮因果规律”的正式结论。新版 physical-press 实验只按生成器记录的物理按键 trigger 裁剪，且随机化 trigger 在窗口中的位置。

### I.1 旧 event-centered 配置与关键参数（历史审计）

| 字段 | 配置 |
|---|---|
| Canonical config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/configs/train/train_lightswitch_randominitial_absolute_eef_event33_group20_c32_3wave1400_stage1_4500.yaml` |
| Runtime config | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/lightswitch_randominitial_abs_eef_event33_group20_c32_3wave1400_high4gpu_95498.yaml` |
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/robomme_light_switch_independent_controls_random8_fixed_close_buttons_no_pause_random_initial_absolute_eef_200eps_hai-machine_lerobot` |
| 数据分组 | 4 causal environments；每 env 5 episode groups，共 20；训练 3 waves 后激活 12，保留 8 个未激活 groups |
| 视频 | 关键事件 core 约 30 frames；Wan 实际 `num_frames=33`，stride 1，no short-chunk padding |
| Action | Absolute EEF source，运行兼容字段 `action_joint`，action dim 14 |
| Z | dim 32，1 token，hidden 128；mode `both`；每 group independent `U(0,1)` |
| Curriculum groups | initial 4；每 wave add 4；按 causal class stratify，确保每次四类各一个 group |
| Batch shape | 每 rank `4 env x 4 action = 16`；4 GPUs；有效 64 |
| Model warm-up | 初始 model-only 300；前 100 steps 做 model-lr warm-up；model lr `1e-5` |
| 每 wave 1400 steps | new-Z 200@0.09；model 200@1e-5；new-Z-mid 200@0.06；all-Z 200@0.03；model 200@1e-5；all-Z 200@0.03；model 200@1e-5 |
| 保存 | `save_steps=500`；至少每 60 分钟保存；仅保留最近两个普通 checkpoint，phase-end Z tables 单独记录 |

### I.2 旧 checkpoint 与评测（不可作为正式结论）

| 资产 | 绝对路径/说明 |
|---|---|
| 实际评测模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/lightswitch_randominitial_abs_eef_event33_group20_c32_3wave1400_high4gpu_95498/step-4100.safetensors` |
| 实际评测 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/lightswitch_randominitial_abs_eef_event33_group20_c32_3wave1400_high4gpu_95498/phase-0802-1738-step-004300-all_context-curriculum_phase_end.context_table.json` |
| 为什么是 4100 + 4300 | Steps 4100-4300 是 all-Z-only phase，模型保持为 step4100，Z table 更新到 phase step4300；这是 `95707` 实际使用的有效组合 |
| 更晚安全配对模型 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/lightswitch_randominitial_abs_eef_event33_group20_c32_3wave1400_high4gpu_95498/step-4433.safetensors` |
| 更晚安全配对 Z table | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/lightswitch_randominitial_abs_eef_event33_group20_c32_3wave1400_high4gpu_95498/step-4433.context_table.json` |
| 评测 job | `95707` |
| 评测输出 | `/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/infer_lightswitch_event33_m4100_c4300_gt_stage1_stage2_95707` |
| 覆盖范围 | GT、Stage1、Stage2；event-centered causal cases；使用 model4100/Z4300 状态 |
| 待补对照 | 完全固定数据 groups、batch、curriculum、LR 和 seed，仅把约 30-frame core 改为 120-frame window |

### I.3 Physical-press no-leak 正式实验：job 107549 / pooled step3289

| 字段 | 新正式配置 |
|---|---|
| 数据集 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/robomme_light_switch_independent_controls_random8_fixed_close_buttons_no_pause_random_initial_absolute_eef_200eps_hai-machine_lerobot` |
| Leakage control | Main view only；33-frame window；裁剪锚点只来自 generator-recorded physical button trigger；trigger 随机 jitter 到 index 11-22；lamp transition 与 causal label 均不参与 crop timing |
| Ours canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_lightswitch_physicalpress33jitter11to22_maincam_group20_c32_3wave1400_actions8_stage1_4500.yaml` |
| Ours checkpoint | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/lightswitch_physicalpress33jitter11to22_maincam_group20_c32_3wave1400_actions8_high2gpu_107549/step-3100.safetensors`；正式评测使用其配对 Z table |
| Ours training | 4 causal classes、20 groups、12 active；2 GPUs；每 rank `4 environments x 8 actions = 32`；C32；model lr `1e-5`，new-Z `0.09`，mid-Z `0.06`，all-Z `0.03` |
| Pooled canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_lightswitch_physicalpress33jitter11to22_maincam_active12_standard_pooled_wm_2gpu_24h.yaml` |
| Pooled checkpoint | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/method_benchmarks/lightswitch_physicalpress33jitter11to22_maincam/standard_pooled_wm/seed_20260827/checkpoints/step-3289.safetensors`；2 GPUs，batch 32/rank，无 Z/adaptation |
| 正式协议 | 四类环境 `neither/red_only/blue_only/both` 各 15 个 disjoint query，共 60/method；Ours 每环境实际使用 K=8 support（4 red + 4 blue，8 个独立 trajectories）；action 只在 red-only/blue-only 上使用全部 30 条 rollout |
| 正式结果根目录 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/results/lightswitch/physicalpress33_all4env_support8_query15_v1` |
| 汇总文件 | `results/lightswitch/physicalpress33_all4env_support8_query15_v1/metrics/complete_v1/summary.csv` 与 `summary.json` |

| 方法 | PSNR ↑ | SSIM ↑ | Final light-state accuracy ↑ | Light yellow-score MAE ↓ | Action success ↑ |
|---|---:|---:|---:|---:|---:|
| Ours，K=8 | **33.45** | 0.9443 | **93.33%** | **0.0358** | **87.5%** |
| Standard pooled WM | 32.62 | **0.9480** | 60.00% | 0.1035 | 50.0% |

Ours 的全序列 light-state accuracy 为 96.36%，transition exact rate 为 83.33%，transition timing error 为 2.30 frames；pooled 分别为 80.20%、51.67% 和 13.28 frames。LPIPS 尚未计算。旧 `95498/95707` 不与此表混用。

## J. Real Ball Friction：training job 101378

### J.1 配置与关键参数

| 字段 | 配置 |
|---|---|
| Canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_real_ball_friction_align_medium_7env_60f_curriculum_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500.yaml` |
| Runtime config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/real_ball_friction_align_medium_7env_60f_1view_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500_101378.yaml` |
| 数据 | 7 个真实环境：`ball-0, ball-1, ball-2, ball-3, ball-4, ball-7, ball-9`；每环境包含多个 impact/skill levels |
| 视频 | 60 个真实帧，重复末帧得到 61-frame Wan 输入；224x224 letterbox；单相机视角 |
| Action | `joint_state_action`，14 dimensions；视频与 action 按同一时间索引对齐 |
| Base checkpoint | `ckpt/BLM/step-12000.safetensors` |
| Z | 每个 ball-friction environment 共享一个 Z32；1 token，hidden dim 128；independent `U(-1,1)`；clamp `[-1,1]`；`physical_context_mode=both` |
| Batch shape | 4 GPUs；每 rank `3 environments x 6 common skills = 18 chunks`；有效 72 chunks/update |
| Curriculum | 初始 4 environments，之后加入剩余 3；initial model-only 300；new-Z 200@0.15；all-Z 200@0.03；model phases lr `1e-5`；之后每 200 steps 交替细化 |
| Spatial objective | 固定 trough polygon 内 loss weight 6x，并将整张 spatial weight map 归一化到 mean 1；polygon=`100,218;640,175;640,302;100,335` |
| 训练预算 | 5500 structured updates；BF16；gradient checkpointing；max grad norm 0.5 |
| 保存策略 | phase-end Z table；普通 checkpoint 只保留最近两个；最终存在严格配对的 model/Z `step5500` |

### J.2 Checkpoint 与评测

| 资产 | 绝对路径/说明 |
|---|---|
| Stage1 model | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/real_ball_friction_align_medium_7env_60f_1view_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500_101378/step-5500.safetensors` |
| Stage1 Z table | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/real_ball_friction_align_medium_7env_60f_1view_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500_101378/step-5500.context_table.json` |
| 评测输出 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/infer_real_ball_friction_align_medium_roi6x_101378_m5500_c5500_7ball_6level_grid_resumable` |
| Protocol manifest | 上述目录中的 `evaluation_manifest.txt`；记录 model/Z step、环境、采样、Stage2 和 ROI 设置 |
| Case selection | 7 environments；每环境无放回抽取 6 个不同 impact/skill levels；共 42 组 GT/Stage1/Stage2 comparisons |
| Grid | 每个环境以 6 个 action columns 展示，单列从上到下为 GT、Stage1、Stage2；`grid_videos.txt` 记录 grid 资产 |
| Stage1 context | 使用对应环境在 training time 学到的 Z table entry |
| Stage2 initialization | 7 个 training-time Z entries 的均值 |
| Stage2 support | 当前评测使用 query chunk 自身作为 support |
| Stage2 update | 只更新 Z；FP32；40 steps：`3.0x10 + 1.5x10 + 0.5x10 + 0.15x10`；reg 0.001；clamp `[-1,1]` |
| Latent analysis | 每个 shard 保存 `context_trajectory.jsonl` 与轨迹 SVG，可分析 training-time target 和 inference-time Z 走向 |

### J.3 当前结论与边界

| 项目 | 记录 |
|---|---|
| 已完成 | 7 个真实 ball-friction environments 的数据准备、共享 world-model/Z 训练、安全配对 checkpoint、42-case Stage1/Stage2 生成与 environment-level grid 已全部跑通 |
| 主要观察 | 相比将 flow-matching loss 均匀分配给整幅画面，mean-normalized ROI 6x objective 能使模型的有效关注高度集中在球、trough 与接触运动区域，显著减少背景对优化预算的稀释；生成中的运动一致性与可辨识物理差异均大幅改善 |
| Latent geometry | ROI-weighted 版本学到的 environment Z 在 PCA 图谱中展现出极强的规律性，说明将监督集中到因果动力学区域不仅改善像素生成，也明显改善 environment representation 的可组织性与可分析性 |
| 结论边界 | 当前生成评测是正式结果，但 Stage2 使用 support=query；不能据此宣称独立 showcase 到 query 的跨 episode adaptation、未见 ball/friction OOD 泛化或真实闭环控制 |
| 下一步 | 固定同一组 7 environments，按 episode/impact level 建立严格 disjoint support/query，并加入球重心轨迹 ADE/FDE 与落点/停止距离指标 |

## K. Real Stick Balance：training job 98364，evaluation job 98738

### K.1 配置与关键参数

| 字段 | 配置 |
|---|---|
| Canonical config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/configs/train/train_real_stick_balance_8env_120raw_stride3_80lift_curriculum_c32_random_stable_4gpu_3env6action_stage1_5500.yaml` |
| Runtime config | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/tmp/run_configs/real_stick_balance_8env_120raw_stride3_80lift_c32_random_stable_4gpu_3env6action_stage1_5500_98364.yaml` |
| 原始数据 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/stick_balance` |
| 环境 | 8 个左右配重环境：3 个 left-heavy、1 个 balanced、4 个 right-heavy |
| 视频 | raw 120-frame window；`frame_stride=3`；视频/action 同步采样为 41-frame Wan 输入；224x224 letterbox |
| Action | `joint_state_action`，14 dimensions |
| Base checkpoint | `ckpt/BLM/step-12000.safetensors` |
| Z | 每个配重环境共享一个 Z32；1 token，hidden dim 128；independent `U(-1,1)`；clamp `[-1,1]`；`physical_context_mode=both` |
| Batch shape | 4 GPUs；每 rank `3 environments x 6 actions/windows = 18 chunks`；有效 72 chunks/update |
| Chunk sampler | 先均匀抽 episode 再抽 window；80% 概率优先关键 `lift` window，20% 保留 general windows |
| Curriculum | 初始 4 environments，再加入剩余 4；initial model-only 300；new-Z lr 0.075；all-Z lr 0.03；model lr `1e-5`；step 2301 后 post-curriculum Z lr 0.015；phase displacement cap 0.4 |
| 训练预算 | 5500 structured updates；BF16；gradient checkpointing；max grad norm 0.5 |
| 保存策略 | phase-end Z table；普通 checkpoint 仅保留最近两个；最终 model/Z 在 `step5500` 严格配对 |

### K.2 Checkpoint 与评测

| 资产 | 绝对路径/说明 |
|---|---|
| Stage1 model | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/real_stick_balance_8env_120raw_stride3_80lift_c32_random_stable_4gpu_3env6action_stage1_5500_98364/step-5500.safetensors` |
| Stage1 Z table | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/real_stick_balance_8env_120raw_stride3_80lift_c32_random_stable_4gpu_3env6action_stage1_5500_98364/step-5500.context_table.json` |
| 评测输出 | `/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/outputs/infer_real_stick_balance_98364_step5500_8env_6ep_gt_stage1_stage2_98738` |
| Case selection | 8 environments x 6 episodes，共 48 组 GT/Stage1/Stage2 comparisons |
| Environment grids | `environment_grids/` 中每环境一幅 6-episode GT/Stage1/Stage2 grid，共 8 个 |
| Stage1 context | 使用对应配重环境的 training-time Z table entry |
| Stage2 initialization | 从 `step-5500.context_table.json` 的 learned contexts 统计量初始化 |
| Stage2 support | 当前评测每个 query 使用自身 chunk 作为 support |
| Stage2 update | Context only；FP32；40-step staged LR；保存每一步 inner loss、Z displacement 和 trajectory |
| Latent analysis | `context_endpoints_combined_pca.svg`、`context_trajectory_combined_pca.svg`、`context_trajectory_stage2_only_zoomed_pca.svg` 及对应 CSV/JSONL |

### K.3 当前结论与边界

| 项目 | 记录 |
|---|---|
| 已完成 | 8 个真实配重环境的 Stage1 训练、严格配对 `step5500` checkpoint、48-case GT/Stage1/Stage2、8 个 environment grids 与联合 PCA 已全部完成 |
| 主要观察 | 关键 lift window 的高采样比例让训练集中在横杆受力和旋转阶段；training-time 与 inference-time Z 可在统一 PCA 中分析不同左右配重环境的结构 |
| 结论边界 | 当前结果证明生成和 latent pipeline 在真实 stick-balance 数据上可运行，但 Stage2 仍为 support=query，且尚未加入横杆倾角定量误差和 disjoint cross-episode adaptation |
| 下一步 | 为每个环境固定独立 showcase/query episodes；报告横杆角度 mean/final error、角速度误差、平衡方向分类和全局 LPIPS/PSNR/SSIM |

## L. 结果解读规范

| 项目 | 规范 |
|---|---|
| Diffusion/flow loss | 只作为训练稳定性和趋势信息，不单独作为生成质量结论 |
| Stage1 | 必须同时检查 GT 对比视频和对应训练时间 Z |
| Stage2 | 必须报告 support/showcase、query、Z 初值、每个 inner step 的 Z trajectory 和最终 query 生成 |
| Physics transfer | 用一个 action 学到 Z 后，必须在同环境未参与 adaptation 的其他 actions 上评测 |
| ID/OOD | 图和文件名必须明确标记；PCA 中 training-time Z 与 inference-time Z 使用不同 marker |
| Checkpoint | 模型与 Z table 必须记录绝对路径和逻辑 step；C-only phase 后允许模型 step 与 Z phase step 不同，但必须解释 |
| 方法结论 | 若一次实验同时改变 batch、LR、数据或初始化，只能记录相关性和假设，不能写成单变量因果结论 |
