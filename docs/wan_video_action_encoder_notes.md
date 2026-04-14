# WanVideoActionEncoder 注释说明

此文件保存 `WanVideoActionEncoder` 类的详细注释。

## 类签名与 Docstring

```python
class WanVideoActionEncoder(torch.nn.Module):
    """Wan视频模型的动作编码器

    将动作序列编码为与DiT模型维度匹配的嵌入向量。
    支持两种模式：
    1. noise模式（不分块）：逐帧动作注入，每帧单独编码
    2. adaln模式（分块）：全局 timestep 条件，整段动作展平后编码

    Args:
        action_dim: 动作空间维度，默认14（7关节 + 1夹爪）x 2臂
        dim: 输出嵌入维度，默认1536（与Wan DiT的hidden_dim一致）
        num_action_per_chunk: 每个chunk的动作帧数
            - None: noise模式（不分块）
            - int: adaln模式（分块），如81帧
        in_features: 输入特征维度，None时自动计算
        hidden_features: 隐藏层维度，None时自动计算
    """
```

## 输入维度计算

```python
        # ========== 输入维度计算 ==========
        # 情况1：noise模式（num_action_per_chunk=None）
        #   - 输入是逐帧动作: (B, F_latent, action_dim)
        #   - 每帧单独处理，输入维度 = action_dim
        #
        # 情况2：adaln模式（num_action_per_chunk=81）
        #   - 输入是展平后的整段动作: (B, F * action_dim)
        #   - 所有帧一起处理，输入维度 = action_dim * num_action_per_chunk
        # =================================
        if in_features is None:
            in_features = action_dim if num_action_per_chunk is None else action_dim * num_action_per_chunk
```

## 隐藏层维度计算

```python
        # ========== 隐藏层维度计算 ==========
        # 情况1：noise模式
        #   - 逐帧处理，隐藏层维度 = dim（与输出同维度）
        #   - 结构: Linear(action_dim -> dim) + GELU + Linear(dim -> dim)
        #
        # 情况2：adaln模式
        #   - 整段展平动作需要更大容量，隐藏层维度 = dim * 4
        #   - 结构: Linear(F*action_dim -> 4*dim) + GELU + Linear(4*dim -> dim)
        # =================================
        if hidden_features is None:
            hidden_features = dim * 4 if num_action_per_chunk is not None else dim
```

## MLP结构

```python
        # MLP结构：输入 -> 隐藏层 -> 输出
        self.action_embedding = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_features, dim),
        )
```

## 前向传播 Docstring

```python
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            action:
                - noise模式: (B, F_latent, action_dim)，F_latent = (num_frames-1)//4+1
                - adaln模式: (B, F * action_dim)，展平后的动作序列

        Returns:
            - noise模式: (B, F_latent, dim)，每帧一个嵌入
            - adaln模式: (B, dim)，单个全局嵌入
        """
```