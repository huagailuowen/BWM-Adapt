# WanVideoPipeline.from_pretrained 调用链分析

## 调用链路

```
WanVideoPipeline.from_pretrained
├── redirect_common_files          # 路径重定向（如 T5/VAE 的 safetensors 重定向）
├── use_usp 分布式初始化
├── pipe = WanVideoPipeline(...)   # 空 pipeline 初始化
├── model_pool = pipe.download_and_load_models(model_configs, vram_limit)
│   └── 遍历每个 model_config:
│       ├── download_if_necessary()      # 按需下载模型文件
│       └── model_pool.auto_load_model(path, vram_config, vram_limit)
│           ├── hash_model_file(path)    # 计算文件内容 MD5
│           ├── 遍历 MODEL_CONFIGS 匹配 hash
│           └── load_model_file(config, path, vram_config, ...)
│               └── load_model(model_class, path, config, torch_dtype, device, ...)
│                   ├── get_init_context(torch_dtype, device)
│                   │   └── skip_model_initialization()  # 避免随机初始化开销
│                   ├── model = model_class(**config)
│                   ├── 若启用 VRAM management: enable_vram_management(...)
│                   ├── 否则: load_state_dict(path)
│                   │   → state_dict_converter (可选)
│                   │   → model.load_state_dict(..., assign=True)
│                   │   → model.to(dtype, device)
│                   └── model.eval()
├── pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
├── pipe.dit = model_pool.fetch_model("wan_video_dit")
├── pipe.vae = model_pool.fetch_model("wan_video_vae")
├── ...
└── return pipe
```

## 关键发现

### 1. `download_and_load_models` 没有全局缓存

每次调用都会新建一个空的 `ModelPool()`，然后遍历 `model_configs` 把所有模型文件重新加载一遍。因此：

- **不能重复调用**，否则所有官方模型（DIT、VAE、T5 等）会被加载两次，产生巨大的 IO 和显存开销。

### 2. `MODEL_CONFIGS` 基于文件内容 hash 匹配

`auto_load_model` 内部通过 `hash_model_file(path)` 计算 MD5，然后在 `MODEL_CONFIGS` 中查找对应条目。这意味着：

- 自定义模型（如 `wan_video_action_encoder`）**不在官方 `MODEL_CONFIGS` 中**，`fetch_model` 永远返回 `None`。
- 若强行把未知文件放进 `model_configs`，会触发 `ValueError("Cannot detect the model type")`。
- 即使后续注册到 `MODEL_CONFIGS`，也不适合训练产物（hash 随权重变化）。

### 3. `load_model` 的核心行为

拆解 `diffsynth/core/loader/model.py` 中的 `load_model`：

| 步骤 | 行为 | 对 `action_encoder` 的意义 |
|---|---|---|
| `skip_model_initialization()` | 参数直接注册到 meta 设备，跳过随机初始化 | 必须对齐，否则小模型也有内存峰值，大模型必爆 |
| `model.to(dtype, device)` | 精度和设备迁移 | 手动创建时已做 |
| `model.eval()` | 设为评估模式 | 必须对齐，否则 Dropout/BatchNorm 行为不一致 |
| `load_state_dict(..., assign=True)` | 直接替换参数张量 | 必须对齐，默认 `assign=False` 行为不同 |
| `enable_vram_management(...)` | 大模型时做 offload/onload 包装 | 小模型可省略，大模型必须对齐 |

## 之前的问题代码

在 `build_wan_video_action_pipeline` 中：

```python
pipe = WanVideoPipeline.from_pretrained(model_configs, ...)  # 第1次：已加载所有模型

model_pool = pipe.download_and_load_models(model_configs, vram_limit)  # 第2次：重复加载！
pipe.action_encoder = model_pool.fetch_model("wan_video_action_encoder")
```

**后果**：
- 所有官方模型被加载两次。
- `fetch_model("wan_video_action_encoder")` 因未注册，永远返回 `None`。
- 最终仍落入 fallback 分支现场初始化 `WanVideoActionEncoder`。

## 修复方案

### 1. 删除重复调用

直接移除 `download_and_load_models` 的二次调用和 `fetch_model`。

### 2. fallback 分支对齐 `load_model` 行为

```python
from diffsynth.core.vram.initialization import skip_model_initialization
from diffsynth.core.vram.layers import AutoWrappedLinear, enable_vram_management

with skip_model_initialization():
    pipe.action_encoder = WanVideoActionEncoder(
        action_dim=action_dim,
        dim=dim,
        num_action_per_chunk=81 if cfg.action_mode.value == "adaln" else None,
    )

pipe.action_encoder = pipe.action_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)
pipe.action_encoder.eval()

# 大模型场景下对齐 VRAM management
if pipe.vram_management_enabled:
    vram_config = {
        "offload_device": "cpu",
        "offload_dtype": pipe.torch_dtype,
        "onload_device": "cpu",
        "onload_dtype": pipe.torch_dtype,
        "preparing_device": pipe.device,
        "preparing_dtype": pipe.torch_dtype,
        "computation_device": pipe.device,
        "computation_dtype": pipe.torch_dtype,
    }
    module_map = {torch.nn.Linear: AutoWrappedLinear}
    pipe.action_encoder = enable_vram_management(
        pipe.action_encoder, module_map, vram_config, vram_limit=vram_limit
    )
```

### 3. 权重加载对齐 `assign=True`

在 `checkpoint_manager.py` 和 `train.py` 中：

```python
pipe.action_encoder.load_state_dict(act_state, strict=False, assign=True)
```

## 为什么不注册到 `MODEL_CONFIGS`

| 问题 | 说明 |
|---|---|
| hash 机制 | `model_hash` 基于文件内容 MD5，训练产物每轮都变 |
| monkey-patch 静态方法 | 第三方库升级时易失效，维护成本高 |
| 热更新失效 | `checkpoint_manager` 的秒级权重切换优势丧失 |
| 混合 checkpoint | 训练 checkpoint 里同时含 `dit.*` 和 `action_encoder.*`，不拆分无法直接用 `auto_load_model` |

## 结论

- **官方 `from_pretrained` 负责加载固定预训练权重（VAE、DIT、T5 等）。**
- **自定义 `action_encoder` 应在 `from_pretrained` 之后手动实例化，并通过 `skip_model_initialization` + `eval()` + 条件 `VRAM management` + `assign=True` 完全对齐 `load_model` 行为。**
- **训练/推理权重通过 `checkpoint_manager` 热更新加载，保持架构解耦。**