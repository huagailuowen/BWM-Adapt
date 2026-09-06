# 仿真实验配置介绍

本文档规定仿真任务后续正式训练、baseline 和 ablation 实验采用的统一计算与 batch 标准。不同任务可以根据环境复杂度和 action 数量使用不同 batch，但同一个任务中的不同方法应遵循相同标准。

## 1. 统一 batch 标准

下表中的 batch 均以一个 optimizer update 实际处理的视频 clip 数量计，而不是底层 dataloader 的 microbatch 字段。

| 任务 | GPU 数 | 每卡 clips/update | 全局 clips/update | 推荐的 grouped batch 结构 |
|---|---:|---:|---:|---|
| Push-box friction Event80 | 2 | 16 | 32 | 每卡 4 environments x 4 actions |
| Gravity | 2 | 16 | 32 | 每卡 4 environments x 4 actions |
| Mass collision no-leak | 2 | 32 | 64 | 每卡 4 environments x 8 actions |
| Light switch no-leak | 2 | 32 | 64 | 每卡 4 environments x 8 actions |
| Mass balance no-leak | 2 | 32 | 64 | 根据该任务的 environment/action sampler 组织为每卡 32 clips |
| Mass x friction | 4 | 24 | 96 | 每卡 4 environments x 6 actions |
| Multi-background friction | 4 | 18 | 72 | 每卡 3 friction groups x 6 actions；同一 friction 可跨背景共享环境 latent |

全局有效 batch 的计算方式为：

```text
global_clips_per_update = clips_per_gpu * num_gpus * gradient_accumulation_steps
```

当前表格默认 `gradient_accumulation_steps = 1`。如果某种方法因显存限制需要降低 microbatch，应通过梯度累积恢复相同的全局有效 batch，并在具体实验配置中明确记录。

## 2. 方法间的执行要求

- Standard Pooled World Model、Ours、baseline 和 ablation 在同一个任务内应使用上表规定的全局 batch。
- Standard Pooled World Model 在训练阶段正常更新 Wan DiT 和 action encoder；它只在直接评测或作为 LoRA-TTT 基础模型时冻结。
- Ours 的底层 `batch_size: 1` 只代表 grouped sampler 内部的 microbatch，不代表论文中报告的有效 batch。
- TTT-KQV、DINOv2、History 等方法若一个 environment item 内包含 support 和多个 query，应按实际参与该次更新的全部 clips 计算 batch。
- 若某种架构无法严格采用标准 batch，必须记录每卡 batch、梯度累积、实际全局 batch、训练吞吐和原因，不得只报告 YAML 中的 `batch_size`。

## 3. 公共训练预算

- 默认训练预算为 24 小时。
- Event80、Gravity、Mass collision、Light switch 和 Mass balance 默认使用 2 张同等级 GPU。
- Mass x friction 和 Multi-background friction 默认使用 4 张同等级 GPU。
- 同一任务的不同方法应使用相同 GPU 类型、GPU 数量、wall-clock budget、训练环境集合、视角、视频窗口、action pool 和数据划分。
- 最终论文同时报告 wall-clock、GPU-hours、optimizer steps、Wan model-gradient updates 和总 clip exposure。由于 Ours 包含 latent-only 阶段，仅比较 checkpoint step 数并不公平。

## 4. 已确认的新版约定

- Event80 和 Gravity 固定使用全局 batch 32。
- Mass collision、Light switch 和 Mass balance 的 no-leak 正式实验固定使用全局 batch 64。
- Mass x friction 固定使用全局 batch 96。
- Multi-background friction 固定使用全局 batch 72。
- 已完成但 batch 与新版标准不同的训练全部保留，仍然是可用实验结果。时间允许时按新版标准补跑严格对照；若时间不足，则直接采用现有结果，并如实报告其实际 batch 和计算预算。

## 5. 现有结果与补跑原则

新版标准只约束后续正式实验，不追溯性地否定、删除或覆盖已有 checkpoint、context table、日志和 metrics。

- Preferred rerun：有足够时间和算力时，按照新版标准补跑，用作最严格的主表对照。
- Existing fallback：没有足够时间时，直接使用现有结果，同时报告真实 batch、GPU 数、wall-clock、steps 和 clip exposure。

Event80 已完成的部分新版 baseline 使用 `2 GPUs x 3 environments/GPU x 6 chunks/environment = 36 clips/update`。理论上的统一标准是 `2 GPUs x 4 environments/GPU x 4 chunks/environment = 32 clips/update`。二者计算规模仅相差 12.5%，因此现有 36-batch 结果继续保留：有时间则补跑 32-batch canonical comparison；没有时间则直接用于论文，但不能写成与 Ours 的 batch 完全相同。

公平性主要由相同 active environments、数据协议、GPU 类型、GPU 数量和 24-hour wall-clock budget 保证，并在附表中披露实际吞吐。

## 6. 下一阶段新版 baseline

下一阶段优先在 Mass collision no-leak、Mass balance no-leak 和 Light switch no-leak 上训练新版 DINOv2 Amortized Context Encoder 与新版 TTT-KQV。

三个任务统一采用：

| 项目 | 标准 |
|---|---|
| GPU 数 | 2 |
| 每卡 environment 数 | 4 |
| 每个 environment 的 chunk 数 | 8 |
| 每卡处理量 | 32 clips/update |
| 全局处理量 | 64 clips/update |
| 默认训练预算 | 24 小时 |
| 初始化 | 与对应任务 Ours 和 Standard Pooled WM 相同的 BLM 初始化 |
| 数据 | 对应任务正式 no-leak 环境、视角、窗口和 action pool |
| Camera protocol | Mass collision、Mass balance、Light switch 均只使用并预测主视角 |
| 保存 | 模型和方法状态一起保存，只保留最新两个完整 checkpoint |

每个 environment 内的 8 个 chunks 应覆盖不同 trajectory 或 action，并避免背景、裁剪位置或时间位置造成环境标签泄露。

### Action sampling 标准

后续正式训练默认采用 `independent-action sampling`：

- 所有 environments 使用相同的候选 action pool 和相同的 action normalization。
- 每个 environment 独立随机抽取本次更新所需的 action/chunk IDs。
- 不要求同一次 grouped update 中不同 environments 使用完全相同的 action IDs。
- DINOv2 在每个 environment 独立抽取的 chunks 中随机指定 support 和 queries。
- TTT-KQV 在每个 environment 独立抽取 chunks，并独立随机打乱 prequential 顺序。

历史 Ours 的 sampler provenance 继续保留。正式补跑范围按任务的 action coverage 区分：

| 任务 | 每环境抽取量 | 补跑决定 | 报告方式 |
|---|---:|---|---|
| Mass collision | 8/9 actions | 不需要 | 作为 independent-action 标准结果使用；现有训练已近似覆盖完整 action pool |
| Mass balance | 8 chunks/actions | 不需要 | 作为 independent-action 标准结果使用；一次更新已覆盖主要候选动作 |
| Light switch | 8 chunks/actions | 不需要 | 作为 independent-action 标准结果使用；已覆盖主要按钮动作与结果 |
| Gravity | 4/10 actions | 时间允许时补跑 | 补一版 independent-action Ours，以匹配新版 baseline |
| Event80 friction | 4/10 actions | 时间允许时补跑 | 补一版 independent-action Ours，以匹配新版 baseline |

Mass Collision、Mass Balance 和 Light 不再进入 sampler 补跑清单。它们的 batch 已覆盖绝大多数可用动作，common-action 与 independent-action 对 action coverage 的实际差异很小。Gravity 和 Event80 每次只取4/10个动作，跨环境是否共享 action subset 可能产生更明显影响，因此是仅有的候选补测任务。

## 7. 新版 DINOv2 协议

每个 environment 的 8 个 chunks 中随机选择 1 个作为 support。Frozen DINOv2 编码 support 的完整视频窗口，并融合相邻采样帧之间的完整 action chunk；projection head 输出 32-D environment representation。该 representation 用于预测剩余 7 个 disjoint query chunks，outer objective 为 7 个 query 的平均 flow-matching loss。每次 grouped update 都重新随机指定 support。

每次全局更新读取 64 个 clips，其中包括 8 个 support clips 和 56 个产生 outer loss 的 query clips。

| 模块 | 默认配置 |
|---|---|
| DINOv2 backbone | Frozen |
| Wan DiT 和 action encoder LR | `1e-5` |
| Temporal fusion/projection head LR | `1e-4` |
| Representation dimension | 32 |
| Warmup | 100 steps |
| Weight decay | `0.01` |

## 8. 新版 TTT-KQV 协议

每个 environment 随机抽取 8 个 chunks，并在每次 grouped update 时重新打乱顺序。不同 environment 的 fast state 独立初始化。每个 chunk 严格执行 `predict current -> compute loss -> write current -> predict next`：当前 chunk 只能读取此前 fast state，预测完成后才能写入，更新结果只影响后续 chunks。

8 个 chunks 均产生 outer prediction loss；前 7 次写入影响后续预测，最后一次写入可以省略。全局每次更新包含 64 个 loss-bearing clips。

| 模块 | 默认配置 |
|---|---|
| TTT-KQV layers | 均匀选择 8 层 |
| 插入位置 | Attention 后串联 |
| Expansion ratio | 4 |
| Gate | Vector gate，初始化 `0.1` |
| Fast-state inner LR | `0.1` |
| TTT slow/module LR | `1e-4` |
| Wan DiT 和 action encoder LR | `1e-5` |
| Fast-state token block | 64 |
| Warmup | 100 steps |

正式推理采用 disjoint support/query：support 先写入 fast state，随后冻结该状态；同一环境的全部 query 从相同 support-adapted state 独立推理，query 不继续更新 fast state。

## 9. 三个任务的采样约束

| 任务 | 必须保持一致的设置 |
|---|---|
| Mass collision no-leak | 正式 train mass levels、主视角、新版 action-8 和质量均衡采样；chunk 完整覆盖可辨识的碰撞过程 |
| Mass balance no-leak | 标注的 20 个 train ratio environments；只使用并预测主视角；视频帧和 action 使用相同 stride；chunk 覆盖足以判断平衡趋势的 support positions |
| Light switch no-leak | 主视角和新版 physical-press chunk；8 个 chunks 覆盖红、蓝按钮及其结果，不得通过裁剪位置泄露环境类型 |

对于上述三个任务，DINOv2 的 support encoder、TTT-KQV 的 fast-state update、Wan 的训练目标和正式 rollout 都只能读取或生成主视角。不得将 wrist/agent view 作为额外条件，也不得把多视角 loss 混入这些正式对照。

上述约束同时适用于 Ours、Standard Pooled WM、DINOv2、TTT-KQV 和相应 ablations。现有结果与未来补跑结果分别记录，不能用新版配置说明覆盖旧实验的真实运行配置。

除非某项 ablation 明确研究 common-action 对齐，后续新增配置均以 independent-action 为默认，不得在未记录的情况下切回 common-action。
