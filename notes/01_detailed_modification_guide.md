# MiniMind-V 改造为 Qwen3-0.6B + Cross-Attention Projector 详细文档

## 一、当前架构分析

### 1.1 现有 MiniMind-V 架构

```
Vision Encoder (CLIP ViT-B/16)
├─ Patch 16×16, 输入 224×224 → 196 tokens (14×14)
└─ 输出: 196×768 维特征

Linear Projector (当前简单实现)
└─ nn.Linear(768, 512)  # 单层线性映射

LLM (MiniMind Tiny)
├─ hidden_size=512
├─ num_hidden_layers=8
├─ num_attention_heads=8
├─ num_key_value_heads=2
└─ vocab_size=6400
```

### 1.2 目标 Qwen3-0.6B 架构

```
Qwen3-0.6B-Think
├─ hidden_size=1024
├─ num_hidden_layers=28
├─ num_attention_heads=16
├─ num_key_value_heads=8
├─ vocab_size=151936
├─ max_position_embeddings=40960
└─ rope_theta=1000000
```

---

## 二、详细修改清单

### 2.1 model/model_vlm.py - 主要修改

#### 修改点 1: VisionProj → CrossAttentionProjector

**位置**: 第 26-37 行

**当前代码**:
```python
class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj
```

**修改后代码**:
```python
class CrossAttentionProjector(nn.Module):
    """
    Cross-Attention Projector: 通过多层交叉注意力机制将视觉特征与文本特征融合
    
    参考架构: Qwen-VL, LLaVA-NeXT
    论文: Language Is Not All You Need: Aligning Perception with Language Models
    """
    def __init__(self, ve_hidden_size=768, hidden_size=1024, num_heads=8, num_layers=2):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 视觉特征降维适配器
        self.vision_adapter = nn.Linear(ve_hidden_size, hidden_size)
        
        # 多层 Cross-Attention
        self.cross_attn_layers = nn.ModuleList([
            self._make_cross_attn_layer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _make_cross_attn_layer(self, hidden_size, num_heads):
        return nn.ModuleDict({
            'cross_attn': nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1
            ),
            'norm1': nn.LayerNorm(hidden_size),
            'norm2': nn.LayerNorm(hidden_size),
            'ffn': nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            )
        })
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        image_indices: dict = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_vision_tokens, ve_hidden_size]
            text_features: [batch, seq_len, hidden_size]
            image_indices: 图像位置索引 {batch_idx: [(start, end), ...]}
        
        Returns:
            更新后的 text_features
        """
        batch_size = text_features.shape[0]
        
        # 视觉特征投影到文本空间
        vision_proj = self.vision_adapter(vision_features)  # [bs, 196, 1024]
        
        # 如果有图像位置信息，进行针对性注入
        if image_indices:
            output_features = text_features.clone()
            for batch_idx, positions in image_indices.items():
                for img_idx, (start, end) in enumerate(positions):
                    # 提取对应位置的文本特征作为 query
                    query = text_features[batch_idx, start:end+1, :]  # [num_text_tokens, 1024]
                    
                    # 视觉特征作为 key, value (每张图一个token)
                    if img_idx < vision_proj.shape[1]:
                        key_value = vision_proj[batch_idx:batch_idx+1, img_idx:img_idx+1, :]
                        
                        # 应用 Cross-Attention layers
                        attn_out = query
                        for layer in self.cross_attn_layers:
                            # Cross-attention: query来自文本, key/value来自视觉
                            attn_out, _ = layer['cross_attn'](
                                attn_out.unsqueeze(0),
                                key_value,
                                key_value
                            )
                            attn_out = attn_out.squeeze(0)
                            
                            # 残差连接 + 归一化
                            attn_out = layer['norm1'](attn_out + query)
                            
                            # FFN
                            ffn_out = layer['ffn'](attn_out)
                            attn_out = layer['norm2'](attn_out + ffn_out)
                        
                        # 更新文本特征中的图像位置
                        output_features[batch_idx, start:end+1, :] = attn_out
            
            return self.output_norm(output_features)
        else:
            # 简化版: 直接拼接 (兼容原有逻辑)
            return self.output_norm(vision_proj)
```

---

#### 修改点 2: MiniMindVLM 类继承和初始化

**位置**: 第 41-49 行

**当前代码**:
```python
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)
```

**修改后代码**:
```python
class MiniMindVLM(Qwen3ForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16",
                 qwen_weight_path="../Models/Qwen3-0.6B"):
        from transformers import Qwen3ForCausalLM, Qwen3Config
        
        if params is None:
            params = VLMConfig()
        self.params = params
        
        # 加载 Qwen3 配置
        qwen_config = Qwen3Config.from_pretrained(qwen_weight_path)
        
        # 初始化 Qwen3 模型
        super().__init__(qwen_config)
        
        # 初始化 Vision Encoder
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        
        # Cross-Attention Projector
        self.vision_proj = CrossAttentionProjector(
            ve_hidden_size=768,
            hidden_size=qwen_config.hidden_size,  # 1024
            num_heads=8,
            num_layers=2
        )
        
        # 加载 Qwen3 权重
        self._load_qwen_weights(qwen_weight_path)
    
    def _load_qwen_weights(self, weight_path: str):
        """加载 Qwen3 预训练权重"""
        import os
        weight_file = os.path.join(weight_path, "model.safetensors")
        
        if os.path.exists(weight_file):
            from safetensors import safe_open
            state_dict = {}
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            
            # 只加载 Qwen3 相关的参数 (vision_proj 除外)
            model_state = self.state_dict()
            qwen_keys = [k for k in state_dict.keys() if not k.startswith('vision_proj')]
            qwen_state = {k: v for k, v in state_dict.items() if k in qwen_keys}
            
            # 更新权重
            miss, unexpected = self.load_state_dict(state_dict, strict=False)
            if miss:
                print(f"[Warning] Missing keys: {miss[:5]}...")
            if unexpected:
                print(f"[Warning] Unexpected keys: {unexpected[:5]}...")
        else:
            print(f"[Warning] Qwen3 weight file not found: {weight_file}")
```

---

#### 修改点 3: count_vision_proj 方法适配

**位置**: 第 77-110 行

**需要修改**: Cross-Attention 需要 text_features 作为 query

```python
def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
    """
    使用 Cross-Attention 注入视觉特征
    
    Args:
        tokens: [batch, seq_len] token IDs
        h: [batch, seq_len, hidden_size] 文本 hidden states
        vision_tensors: [batch*num_images, 196, 768] Vision features
        seqlen: 序列长度限制
    """
    def find_indices(tokens, image_ids):
        image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
        len_image_ids = len(image_ids)
        if len_image_ids > tokens.size(1):
            return None
        tokens_view = tokens.unfold(1, len_image_ids, 1)
        matches = (tokens_view == image_ids_tensor).all(dim=2)
        return {
            batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                        matches[batch_idx].nonzero(as_tuple=True)[0]]
            for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
        } or None

    image_indices = find_indices(tokens, self.params.image_ids)
    if vision_tensors is not None and image_indices:
        # 使用 Cross-Attention Projector
        h = self.vision_proj(vision_tensors, h, image_indices)
        return h[:, :seqlen]
    return h
```

---

### 2.2 model/model_minimind.py - VLMConfig 更新

**位置**: 第 13-24 行

```python
class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            # Qwen3 兼容配置
            hidden_size: int = 1024,
            num_hidden_layers: int = 28,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 8,
            vocab_size: int = 151936,
            max_position_embeddings: int = 40960,
            rope_theta: float = 1000000.0,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            **kwargs
        )
```

---

### 2.3 trainer/trainer_utils.py - 加载逻辑修改

**位置**: 第 66-93 行 init_vlm_model 函数

```python
def init_vlm_model(vlm_config, from_weight='pretrain_vlm', 
                   tokenizer_path='../Models/Qwen3-0.6B',
                   vision_model_path='../model/vision_model/clip-vit-base-patch16', 
                   save_dir='../out', device='cuda', freeze_llm=False):
    from transformers import AutoTokenizer
    from safetensors import safe_open
    import os
    
    # 使用 Qwen3 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path,
                       qwen_weight_path=tokenizer_path)
    
    if from_weight != 'none':
        # 支持多种权重格式
        if from_weight.endswith('.safetensors'):
            # 直接加载 safetensors
            state_dict = {}
            weight_path = from_weight if os.path.isabs(from_weight) else os.path.join(save_dir, from_weight)
            with safe_open(weight_path, framework='pt', device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            model.load_state_dict(state_dict, strict=False)
        else:
            # 传统 .pth 格式
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            if os.path.exists(weight_path):
                weights = torch.load(weight_path, map_location=device)
                model.load_state_dict(weights, strict=False)
            else:
                print(f"[Warning] Weight file not found: {weight_path}")
    
    # 冻结 LLM 参数 (仅训练 vision_proj)
    if freeze_llm:
        for name, param in model.named_parameters():
            if 'vision_proj' not in name:
                param.requires_grad = False
    
    get_model_params(model, vlm_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'Trainable Params: {trainable:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess
```

---

### 2.4 训练脚本参数调整

**train_pretrain_vlm.py 和 train_sft_vlm.py**

默认参数需要调整:

```python
parser.add_argument('--hidden_size', default=1024, type=int)  # 512 → 1024
parser.add_argument('--num_hidden_layers', default=28, type=int)  # 8 → 28
parser.add_argument('--max_seq_len', default=2048, type=int)  # 640/1536 → 2048
parser.add_argument('--batch_size', default=4, type=int)  # 根据显存调整，Qwen3更大
parser.add_argument('--learning_rate', default=2e-4, type=float)  # 可能需要降低
```

---

## 三、safetensors 格式详解

### 3.1 什么是 safetensors

`safetensors` 是 Hugging Face 推出的安全张量存储格式:

**优点**:
- ✅ 安全: 不执行任意代码 (不像 pickle)
- ✅ 快速: 零拷贝读取
- ✅ 懒加载: 可以只加载需要的部分
- ✅ 支持分布式训练
- ✅ 文件大小无限制

### 3.2 与传统 .pth 格式对比

| 特性 | .pth (pickle) | .safetensors |
|------|---------------|--------------|
| 安全性 | ❌ 可能执行恶意代码 | ✅ 安全 |
| 速度 | ⚠️ 需要拷贝到CPU | ✅ 零拷贝 |
| 懒加载 | ❌ 不支持 | ✅ 支持 |
| 文件大小限制 | ✅ 无限制 | ✅ 无限制 |
| 布局控制 | ⚠️ 受限 | ✅ 完全可控 |

### 3.3 加载方式

#### 方式 1: Transformers 自动加载 (推荐)
```python
from transformers import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained("../Models/Qwen3-0.6B")
```

#### 方式 2: 手动加载 safetensors
```python
from safetensors import safe_open

state_dict = {}
with safe_open("../Models/Qwen3-0.6B/model.safetensors", 
               framework='pt', device='cuda') as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)
```

#### 方式 3: 转换为 .pth 格式 (不推荐，丢失安全优势)
```python
from safetensors import safe_open
import torch

state_dict = {}
with safe_open("model.safetensors", framework='pt', device='cpu') as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

torch.save(state_dict, "model.pth")
```

### 3.4 保存模型为 safetensors

```python
from safetensors.torch import save_file

tensors = {
    "model.layers.0.self_attn.q_proj.weight": model.state_dict()["model.layers.0.self_attn.q_proj.weight"],
    # ...
}

save_file(tensors, "model.safetensors")
```

---

## 四、可能的兼容性问题

### 4.1 Tokenizer 兼容性

**问题**: MiniMind 使用自定义 tokenizer (vocab_size=6400)，Qwen3 使用标准 tokenizer (vocab_size=151936)

**解决方案**:
- 完全切换到 Qwen3 tokenizer
- 需要重新生成训练数据中的 token ids

### 4.2 模型参数不匹配

**问题**: 加载 VLM Checkpoint 时参数数量/维度变化

**解决方案**:
```python
model.load_state_dict(state_dict, strict=False)  # 使用 strict=False
```

只加载匹配的参数，新的 Cross-Attention 参数会被随机初始化。

### 4.3 显存占用

**问题**: Qwen3 (0.6B) 比 MiniMind (26M) 大很多

**解决方案**:
1. 使用 gradient checkpointing
2. 降低 batch_size
3. 使用更小的 hidden_size (如 768)
4. 使用 DeepSpeed 或 BitsAndBytes 量化

### 4.4 训练数据格式

**问题**: image_special_token 需要根据 Qwen3 tokenizer 调整

**解决方案**:
```python
# Qwen3 特殊 token
tokenizer = AutoTokenizer.from_pretrained("../Models/Qwen3-0.6B")
image_token = tokenizer.convert_tokens_to_ids("<|vision_start|>")  # 或自定义
```

---

## 五、推荐实施顺序

### 阶段 1: 基础改造 (1-2天)
1. 修改 `VisionProj` → `CrossAttentionProjector`
2. 修改 `MiniMindVLM` 继承 `Qwen3ForCausalLM`
3. 修改 `init_vlm_model` 加载 safetensors

### 阶段 2: 训练适配 (2-3天)
4. 调整训练参数 (batch_size, lr 等)
5. 测试数据加载和前向传播
6. 小规模预训练测试

### 阶段 3: 完整训练 (1周+)
7. 全量预训练
8. SFT 微调
9. 评估和优化

---

## 六、参考文献

1. **Qwen2-VL 论文**: https://arxiv.org/abs/2309.15329
2. **LLaVA-NeXT**: https://arxiv.org/abs/2310.03744
3. **Cross-Attention in VLMs**: https://arxiv.org/abs/2304.08485
4. **SafeTensors 官方文档**: https://huggingface.co/docs/safetensors

---

## 七、附录: 关键代码片段

### A. 检查 Qwen3 权重完整性

```python
import os
from safetensors import safe_open

weight_path = "../Models/Qwen3-0.6B/model.safetensors"
if not os.path.exists(weight_path):
    print(f"Error: {weight_path} not found")
    exit(1)

with safe_open(weight_path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    print(f"Sample keys: {keys[:5]}")
    
    # 检查关键参数
    required_keys = ['model.embed_tokens.weight', 'lm_head.weight']
    for rk in required_keys:
        if rk in keys:
            shape = f.get_tensor(rk).shape
            print(f"{rk}: {shape}")
        else:
            print(f"Warning: {rk} not found")
```

### B. Cross-Attention 单元测试

```python
import torch

def test_cross_attention_projector():
    batch_size = 2
    num_vision_tokens = 196
    hidden_size = 1024
    ve_hidden_size = 768
    
    proj = CrossAttentionProjector(ve_hidden_size, hidden_size, num_heads=8, num_layers=2)
    
    vision_features = torch.randn(batch_size, num_vision_tokens, ve_hidden_size)
    text_features = torch.randn(batch_size, 512, hidden_size)
    
    # 模拟图像位置
    image_indices = {
        0: [(0, 10)],  # batch 0, position 0-10
        1: [(20, 30)]  # batch 1, position 20-30
    }
    
    output = proj(vision_features, text_features, image_indices)
    print(f"Input shape: {text_features.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == text_features.shape
    print("✓ Test passed!")
```

---

**文档版本**: v1.0
**创建日期**: 2025-02-13
**适用版本**: minimind-v + Qwen3-0.6B-Think
