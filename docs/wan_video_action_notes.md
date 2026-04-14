# model_fn_wan_video_action 注释说明

此文件保存 `model_fn_wan_video_action` 函数的详细注释。

## 函数签名与 Docstring

```python
def model_fn_wan_video_action(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_injection_mode: str = "none",
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """支持 action 条件注入的 Wan Video DiT 前向传播模型函数 (Patch 版本)。

    Args:
        dit: Wan Video DiT 模型实例。
        latents (torch.Tensor): 潜在空间的噪声张量。形状为 (B, 16, F/4, H/8, W/8)。
        timestep (torch.Tensor): 当前的扩散时间步，范围 (0~1000)。
        context (torch.Tensor): 文本特征嵌入。形状为 (B, 512, 4096)。
        action_emb (torch.Tensor, optional): 注入的动作控制嵌入。形状为 (B, F/4, D_model) 或 (B, D_model)。
        action_injection_mode (str): action 的注入模式，默认为 "none"。
        clip_feature (torch.Tensor, optional): CLIP 图像视觉特征。形状为 (B, 257, 1280) (如 ViT-bigG/14)。
        y (torch.Tensor, optional): 图像先验条件。形状为 (B, 20, F/4, H/8, W/8)，其中 20 通道 = 4 (Mask) + 16 (VAE首帧)。
        use_gradient_checkpointing (bool): 是否启用梯度检查点以节省显存。
        use_gradient_checkpointing_offload (bool): 是否将中间激活值 Offload 到 CPU 以进一步节省显存。
    """
```

## 步骤1: 时间步编码

```python
    # ========== 步骤1: 时间步编码 ==========
    # t: (B=1, D_model=1536)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
```

## 注入：adaln模式下的action注入

```python
    # 注入：adaln模式下的action注入
    if action_injection_mode == "adaln" and action_emb is not None:
        # =========================================================================
        # 为什么可以直接相加：
        # 1. 维度对齐：action_emb 已被映射至与时间嵌入 t 相同的特征维度。
        # 2. 信息融合与统一调制：通过直接叠加，action 信号"搭车"进入后续的 time_projection。
        #    这使得时间步与动作指令共同生成 AdaLN 的 scale 和 shift 参数，
        #    从而以非侵入式的方式，在不修改底层 DiT Block 的前提下实现了全局动作控制。
        # =========================================================================
        t = t + action_emb
```

## 将时间嵌入投影为调制参数

```python
    # 将时间嵌入投影为调制参数
    # t_mod: (B=1, 6, D_model=1536)
    # 这6个参数用于 AdaLN (Adaptive Layer Normalization) 调制
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
```

## 步骤2: 文本条件处理

```python
    # ========== 步骤2: 文本条件处理 ==========
    if context is not None:
        context = dit.text_embedding(context)
```

## 步骤3: 图像条件整合

```python
    # ========== 步骤3: 图像条件整合 ==========
    # 3.1 整合 VAE 图像条件 (如果提供)
    if y is not None and dit.has_image_input and dit.require_vae_embedding:
        # =========================================================================
        # 图像条件注入 (Image Condition Injection)
        # 将噪声潜变量 x (16-ch) 与首帧/参考图像条件 y (20-ch) 在通道维度拼接。
        # (B=1, C=16, F/4, H/8, W/8) + (B=1, C_y=20, F/4, H/8, W/8) -> (B=1, C+C_y=36, F/4, H/8, W/8)
        #
        # y 的构成: 16-ch VAE 特征 (视觉先验) + 4-ch Mask (区分已知帧与待生成帧)
        # 机制与目的:
        #   1. 时空对齐：使模型在第一层 Patch 投影时，能在同一时空位置同时观测到噪声与条件。
        #   2. 维度匹配：图生视频模式下，DiT 首层输入通道已被初始化为 36 (16 + 20)。
        # =========================================================================
        x = torch.cat([x, y], dim=1)
```

## 3.2 整合 CLIP 图像特征

```python
    # 3.2 整合 CLIP 图像特征 (如果提供)
    if clip_feature is not None and dit.has_image_input and dit.require_clip_embedding:
        # clip context计算，用于后续cross-attention
        # context: (B=1, L_token=512, D_model) + clip_emb: (B=1, N_img=257, D_model) -> (B, L+N_img=769, D_model)
        clip_embdding = dit.img_emb(clip_feature)
```

## 步骤4: Patchify

```python
    # ========== 步骤4: Patchify ==========
    # 通过无重叠的 3D 卷积 (kernel=stride=patch_size) 将视频潜变量切分为独立 Patch。
    #
    # 维度变换: (B, C_in, F, H, W) -> (B, dim, F', H', W')
    #   - C_in: 被映射为 Transformer 的特征维度 (dim)
    #   - F, H, W: 被 patch_size 下采样为压缩后的时空网格尺寸
    # ====================================
    x = dit.patchify(x)
```

## 注入：noise模式下的action注入

```python
    # 注入：noise模式下的action注入
    if action_injection_mode == "noise" and action_emb is not None:
        # action_emb: (B, F, D_model) -> (B, D_model, F, 1, 1), broadcast to (H/16, W/16)
        action_emb = rearrange(action_emb, "b f d -> b d f 1 1")
        action_emb = repeat(action_emb, "b d f 1 1 -> b d f h w", h=h, w=w)
        x = x + action_emb
```

## 将3D patch grid 展平为1D token 序列

```python
    # 将3D patch grid 展平为1D token 序列
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
```

## 步骤5: 位置编码 (RoPE)

```python
    # ========== 步骤5: 位置编码 (RoPE) ==========
    # 构建 3D 旋转位置编码 (3D RoPE) 的完整时空频率网格
    # 1. 提取与变换 (view): 取出对应 帧/高/宽 的 1D 频率，并增加占位维度。
    # 2. 空间广播 (expand): 将 1D 频率在逻辑上广播为完整的 (f, h, w) 3D 网格。
    # 3. 特征拼接 (cat): 在通道维度 (-1) 拼接，使每个坐标点同时拥有 t, h, w 的位置信息。
    # 4. 序列化展平 (reshape): 将 3D 网格拍平为 1D 序列 (f*h*w, 1, dim)，严格对齐视频 Token。
    # =========================================================================
```

## 步骤6: Transformer Blocks

```python
    # ========== 步骤6: Transformer Blocks ==========
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            # ... gradient checkpointing with offload
        elif use_gradient_checkpointing:
            # 标准梯度检查点: 不保存中间激活值,反向传播时重新计算
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            # 正常前向传播 (最快,但显存占用最大)
            x = block(x, context, t_mod, freqs)
```

## 步骤7: 输出投影和 Unpatchify

```python
    # ========== 步骤7: 输出投影和 Unpatchify ==========
    # 1. Head 投影: 结合时间步 t 进行最后一次 AdaLN 调制，并将 Transformer 的
    #    隐藏维度 (dim) 扩张回潜空间所需的通道容量。
    #    维度变换: (B, f*h*w, dim) -> (B, f*h*w, out_dim * P_f * P_h * P_w)
    # 2. Unpatchify (逆分块): 将 1D 的 Patch 序列重新折叠拼装成 3D 的视频空间网格。
    #    维度变换: (B, f*h*w, ...) -> (B, out_dim, F, H, W)
    # ================================================
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
```