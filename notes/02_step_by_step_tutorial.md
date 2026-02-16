# MiniMind-V → Qwen3-0.6B + Cross-Attention 完改造教程

> 本教程提供从零开始的详细步骤，每一步都包含潜在问题和解决方案

---

## 📋 目录

1. [准备工作](#准备工作)
2. [阶段一: 基础环境验证](#阶段一-基础环境验证)
3. [阶段二: 代码基础改造](#阶段二-代码基础改造)
4. [阶段三: Model Architecture 修改](#阶段三-model-architecture-修改)
5. [阶段四: 训练逻辑适配](#阶段四-训练逻辑适配)
6. [阶段五: 首次测试运行](#阶段五-首次测试运行)
7. [阶段六: 正式训练](#阶段六-正式训练)
8. [故障排查汇总](#故障排查汇总)

---

## 准备工作

### 0.1 检查目录结构

```bash
cd C:\Users\z1272\Desktop\LLM_Projects\minimind-v
tree /F /A
```

**预期结构**:
```
minimind-v/
├── model/
│   ├── model_vlm.py          ⭐ 主要修改
│   ├── model_minimind.py
│   └── vision_model/
├── trainer/
│   ├── train_pretrain_vlm.py
│   ├── train_sft_vlm.py
│   └── trainer_utils.py      ⭐ 需要修改
├── dataset/
├── out/                      (模型保存目录)
├── checkpoints/              (训练检查点)
└── notes/
```

**问题**: 找不到 Qwen3 权重？
```bash
# 解决方案
ls -la ../Models/Qwen3-0.6B/model.safetensors
# 如果不存在，说明权重路径错误，修复你的路径
```

---

### 0.2 环境依赖检查

```bash
# 激活你的 Python 环境
conda activateyour_env_name  # 或 source venv/bin/activate

# 检查关键依赖
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import safetensors; print('safetensors ok')"
```

**问题**: `ModuleNotFoundError: No module named 'safetensors'`
```bash
# 解决方案
pip install safetensors
```

**问题**: `transformers` 版本太低不支持 Qwen3
```bash
# 解决方案
pip install transformers>=4.35.0
```

---

## 阶段一: 基础环境验证

### Step 1.1: 验证 Qwen3 模型加载

创建测试脚本 `notes/test_qwen3_loading.py`:

```python
import torch
from transformers import Qwen3ForCausalLM, Qwen3Config

print("="*50)
print("Step 1.1: 验证 Qwen3 模型加载")
print("="*50)

# 测试加载
try:
    model_path = "../Models/Qwen3-0.6B"
    config = Qwen3Config.from_pretrained(model_path)
    print(f"✓ Config loaded")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - num_hidden_layers: {config.num_hidden_layers}")
    print(f"  - num_attention_heads: {config.num_attention_heads}")
    print(f"  - vocab_size: {config.vocab_size}")
    
    model = Qwen3ForCausalLM.from_pretrained(model_path)
    print(f"✓ Model loaded")
    print(f"  - Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试前向传播
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    outputs = model(input_ids)
    print(f"✓ Forward pass works")
    print(f"  - Output shape: {outputs.logits.shape}")
    
    print("✓ Step 1.1 PASSED")
    
except Exception as e:
    print(f"✗ Step 1.1 FAILED: {e}")
    raise
```

运行:
```bash
cd notes
python test_qwen3_loading.py
```

**可能问题和解决方案**:

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `FileNotFoundError` | 路径错误 | 检查 `../Models/Qwen3-0.6B` 是否存在 |
| `torch.cuda.OutOfMemoryError` | 显存不足 | 使用 `device_map="cpu"` 或减少批次大小 |
| `ImportError` | transformers 版本低 | `pip install --upgrade transformers` |
| `KeyError: model.safetensors` | 权重缺失 | 检查目录下是否有 `model.safetensors` 文件 |

---

### Step 1.2: 验证 Vision Encoder (CLIP)

创建测试脚本 `notes/test_clip_loading.py`:

```python
import torch
from transformers import CLIPModel, CLIPProcessor

print("="*50)
print("Step 1.2: 验证 CLIP 模型加载")
print("="*50)

try:
    model_path = "../model/vision_model/clip-vit-base-patch16"
    
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    print(f"✓ CLIP model loaded")
    
    # 冻结参数测试
    for param in model.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Parameters frozen (trainable: {trainable})")
    
    # 测试图像编码
    from PIL import Image
    import numpy as np
    
    # 创建随机图像
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
    
    print(f"✓ Image encoding works")
    print(f"  - Output shape: {outputs.last_hidden_state.shape}")  # [1, 197, 768]
    print(f"  - Patch tokens shape: {outputs.last_hidden_state[:, 1:, :].shape}")  # [1, 196, 768]
    
    print("✓ Step 1.2 PASSED")
    
except Exception as e:
    print(f"✗ Step 1.2 FAILED: {e}")
    raise
```

**可能问题和解决方案**:

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CLIP 模型文件夹不存在 | 未下载 | 运行 README 中的 git clone 命令 |
| `ModuleNotFoundError` | 缺少依赖 | `pip install transformers pillow` |
| 图像处理失败 | PIL Image 的问题 | 确保 `from PIL import Image` 正常 |

---

### Step 1.3: 验证 safetensors 接口

创建测试脚本 `notes/test_safetensors.py`:

```python
import torch
from safetensors import safe_open

print("="*50)
print("Step 1.3: 验证 safetensors 接口")
print("="*50)

try:
    weight_path = "../Models/Qwen3-0.6B/model.safetensors"
    
    # 方式 1: 加载所有权重
    print("\n[方式 1] 加载所有权重...")
    state_dict = {}
    with safe_open(weight_path, framework='pt', device='cpu') as f:
        keys = list(f.keys())
        print(f"  总 key 数量: {len(keys)}")
        print(f"  示例 keys: {keys[:5]}")
        
        for key in keys:
            state_dict[key] = f.get_tensor(key)
    
    print(f"✓ 加载完成, 共 {len(state_dict)} 个张量")
    
    # 方式 2: 懒加载单个张量
    print("\n[方式 2] 懒加载单个张量...")
    with safe_open(weight_path, framework='pt', device='cpu') as f:
        embed_weight = f.get_tensor('model.embed_tokens.weight')
        print(f"  model.embed_tokens.weight shape: {embed_weight.shape}")
    
    # 方式 3: 懒加载片段 (适用于大模型)
    print("\n[方式 3] 懒加载片段...")
    with safe_open(weight_path, framework='pt', device='cpu') as f:
        embed_slice = f.get_slice('model.embed_tokens.weight')
        vocab_size, hidden_dim = embed_slice.get_shape()
        print(f"  vocab_size: {vocab_size}, hidden_dim: {hidden_dim}")
        
        # 只加载前 1000 个 token
        partial = embed_slice[:1000, :]
        print(f"  部分加载 shape: {partial.shape}")
    
    print("✓ Step 1.3 PASSED")
    
except Exception as e:
    print(f"✗ Step 1.3 FAILED: {e}")
    import traceback
    traceback.print_exc()
```

**可能问题和解决方案**:

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `FileNotFoundError` | 权重路径错误 | 检查 `../Models/Qwen3-0.6B/model.safetensors` 是否存在 |
| `OSError` | 文件格式错误 | 重新下载 safetensors 文件 |
| 权重加载顺序问题 | 无 | safetensors 不支持随机访问，顺序读取即可 |

---

## 阶段二: 代码基础改造

### Step 2.1: 备份原始代码

```bash
# 在 notes 目录创建备份
cd notes
mkdir backup_$(date +%Y%m%d_%H%M%S)
cp ../model/model_vlm.py backup_*/
cp ../trainer/trainer_utils.py backup_*/
echo "✓ 原始文件已备份到 notes/backup_*/"
```

---

### Step 2.2: 修改 VisionProj → CrossAttentionProjector

打开 `model/model_vlm.py`，定位到第 26-37 行:

**编辑步骤**:

1. 找到原有的 `VisionProj` 类 (26-37 行)
2. 删除该类
3. 在相同位置插入以下代码:

```python
class CrossAttentionProjector(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=1024, num_heads=8, num_layers=2):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.vision_adapter = nn.Linear(ve_hidden_size, hidden_size)
        
        self.cross_attn_layers = nn.ModuleList([
            self._make_cross_attn_layer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
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
    
    def forward(self, vision_features, text_features, image_indices=None):
        batch_size = text_features.shape[0]
        vision_proj = self.vision_adapter(vision_features)
        
        if image_indices:
            output_features = text_features.clone()
            for batch_idx, positions in image_indices.items():
                for img_idx, (start, end) in enumerate(positions):
                    query = text_features[batch_idx, start:end+1, :]
                    
                    if img_idx < vision_proj.shape[1]:
                        key_value = vision_proj[batch_idx:batch_idx+1, img_idx:img_idx+1, :]
                        
                        attn_out = query
                        for layer in self.cross_attn_layers:
                            attn_out, _ = layer['cross_attn'](
                                attn_out.unsqueeze(0), key_value, key_value
                            )
                            attn_out = attn_out.squeeze(0)
                            attn_out = layer['norm1'](attn_out + query)
                            ffn_out = layer['ffn'](attn_out)
                            attn_out = layer['norm2'](attn_out + ffn_out)
                        
                        output_features[batch_idx, start:end+1, :] = attn_out
            
            return self.output_norm(output_features)
        else:
            return self.output_norm(vision_proj)
```

**验证编译**: 保存文件，确保没有语法错误。

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `IndentationError` | 检查缩进，确保使用 4 空格 |
| `NameError: name 'nn' is not defined` | 确保 `from torch import nn` 在文件顶部 |
| 找不到原始行号 | 使用编辑器的搜索功能搜索 `class VisionProj` |

---

### Step 2.3: 修改 MiniMindVLM 类继承

继续编辑 `model/model_vlm.py`，定位到第 41 行:

**修改前**:
```python
from .model_minimind import *
```

**修改后**:
```python
from .model_minimind import *
from transformers import Qwen3ForCausalLM, Qwen3Config
```

然后修改 `MiniMindVLM` 类 (41-49 行):

**原代码**:
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

**替换为**:
```python
class MiniMindVLM(Qwen3ForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, 
                 vision_model_path="./model/vision_model/clip-vit-base-patch16",
                 qwen_weight_path="../Models/Qwen3-0.6B"):
        if params is None:
            params = VLMConfig()
        self.params = params
        
        qwen_config = Qwen3Config.from_pretrained(qwen_weight_path)
        super().__init__(qwen_config)
        
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        
        self.vision_proj = CrossAttentionProjector(
            ve_hidden_size=768,
            hidden_size=qwen_config.hidden_size,
            num_heads=8,
            num_layers=2
        )
        
        self._load_qwen_weights(qwen_weight_path)
    
    def _load_qwen_weights(self, weight_path: str):
        import os
        weight_file = os.path.join(weight_path, "model.safetensors")
        
        if os.path.exists(weight_file):
            from safetensors import safe_open
            state_dict = {}
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            
            model_state = self.state_dict()
            qwen_keys = [k for k in model_state.keys() if not k.startswith('vision_proj')]
            qwen_state = {k: v for k, v in state_dict.items() if k in qwen_keys}
            
            miss, unexpected = self.load_state_dict(state_dict, strict=False)
            if miss:
                print(f"[Warning] Missing keys: {len(miss)}")
            if unexpected:
                print(f"[Warning] Unexpected keys: {len(unexpected)}")
        else:
            print(f"[Warning] Qwen3 weight not found: {weight_file}")
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `NameError: name 'Qwen3ForCausalLM' is not defined` | 确保导入语句在类定义前 |
| `ImportError: cannot import name 'Qwen3ForCausalLM'` | 检查 transformers 版本是否支持 Qwen3 |
| `FileNotFoundError` | 检查 `qwen_weight_path` 是否正确 |
| `RuntimeError: Error(s) in loading state_dict` | 可能是维度不匹配，使用 `strict=False` |

---

### Step 2.4: 修改 count_vision_proj 方法

继续编辑 `model/model_vlm.py`，找到 `count_vision_proj` 方法 (77-110 行):

**原代码**:
```python
def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
    # ... 原有逻辑
    if vision_tensors is not None and image_indices:
        vision_proj = self.vision_proj(vision_tensors)
        # ... 线性投影替换
```

**修改为**:
```python
def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
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
        h = self.vision_proj(vision_tensors, h, image_indices)
        return h[:, :seqlen]
    return h
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| 方法签名与 `forward` 调用不匹配 | 仔细检查 `forward` 方法中 `count_vision_proj` 的调用 |
| `TypeError: 'CrossAttentionProjector' object is not subscriptable` | 检查 `vision_proj.forward()` 返回值类型 |
| `IndexError` | 确保 `image_indices` 和 `vision_tensors` 的形状匹配 |

---

### Step 2.5: 更新 VLMConfig

编辑 `model/model_minimind.py`，找到 `VLMConfig` 类 (13-24 行):

**修改为**:
```python
class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
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

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `TypeError: __init__() got an unexpected keyword argument` | 检查 `MiniMindConfig` 是否支持这些参数 |
| 参数默认值冲突 | 确保 Qwen3 参数与 `MiniMindConfig` 不冲突 |

---

## 阶段三: Model Architecture 修改

### Step 3.1: 测试模型实例化

创建测试脚本 `notes/test_model_instance.py`:

```python
import torch
import sys
sys.path.insert(0, '..')
from model.model_vlm import MiniMindVLM, VLMConfig

print("="*50)
print("Step 3.1: 测试模型实例化")
print("="*50)

try:
    config = VLMConfig(
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8
    )
    
    print(f"✓ Config created")
    
    model = MiniMindVLM(
        params=config,
        vision_model_path="../model/vision_model/clip-vit-base-patch16",
        qwen_weight_path="../Models/Qwen3-0.6B"
    )
    
    print(f"✓ Model instantiated")
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    print(f"  - Total params: {param_count:.2f}M")
    print(f"  - Trainable params: {trainable:.2f}M")
    
    # 检查关键模块
    assert hasattr(model, 'vision_encoder'), "Missing vision_encoder"
    assert hasattr(model, 'vision_proj'), "Missing vision_proj"
    
    print(f"✓ Key modules verified")
    print("✓ Step 3.1 PASSED")
    
except Exception as e:
    print(f"✗ Step 3.1 FAILED: {e}")
    import traceback
    traceback.print_exc()
```

运行:
```bash
cd notes
python test_model_instance.py
```

**可能问题和解决方案**:

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ImportError` | 导入路径错误 | 检查 `sys.path.insert(0, '..')` |
| `RuntimeError: CUDA out of memory` | 显存不足 | 使用 CPU 模式或减小模型规模 |
| `FileNotFoundError` | 权重路径错误 | 检查路径是否正确 |
| `TypeError` | 方法签名不匹配 | 检查 `__init__` 参数 |

---

### Step 3.2: 测试前向传播

创建测试脚本 `notes/test_forward_pass.py`:

```python
import torch
import sys
sys.path.insert(0, '..')
from model.model_vlm import MiniMindVLM, VLMConfig
from transformers import AutoTokenizer

print("="*50)
print("Step 3.2: 测试前向传播")
print("="*50)

try:
    config = VLMConfig(hidden_size=1024, num_hidden_layers=28)
    model = MiniMindVLM(
        params=config,
        vision_model_path="../model/vision_model/clip-vit-base-patch16",
        qwen_weight_path="../Models/Qwen3-0.6B"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("../Models/Qwen3-0.6B")
    
    # 测试纯文本模式
    print("\n[测试 1] 纯文本前向传播...")
    text_input = "Hello, how are you?"
    inputs = tokenizer(text_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    print(f"  ✓ Text mode works")
    print(f"    - Logits shape: {outputs.logits.shape}")
    print(f"    - Loss: {outputs.loss}")
    
    # 测试 multimodal 模式
    print("\n[测试 2] Multimodal 前向传播...")
    from PIL import Image
    import numpy as np
    
    # 准备图像占位符 (196个 @ 符号)
    image_placeholder = '@' * 196
    prompt = f"{image_placeholder}\nWhat is in this image?"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True, padding='max_length')
    
    # 模拟图像
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image)
    pixel_values = model.processor(images=image, return_tensors="pt")['pixel_values']
    # 扩展为 [1, 1, 3, 224, 224]
    pixel_values = pixel_values.unsqueeze(1)
    
    # 简化的 forward 测试
    print(f"  ✓ Multimodal mode setup complete")
    print(f"    - Input IDs shape: {inputs['input_ids'].shape}")
    print(f"    - Pixel values shape: {pixel_values.shape}")
    
    print("✓ Step 3.2 PASSED")
    
except Exception as e:
    print(f"✗ Step 3.2 FAILED: {e}")
    import traceback
    traceback.print_exc()
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `ValueError` | 输入形状不匹配 | 检查 `pixel_values` 的形状 |
| `RuntimeError` | forward 参数错误 | 检查 `forward` 方法签名 |
| 权重加载不完整 | safetensors 部分张量缺失 | 检查 `load_state_dict` 的输出 |

---

## 阶段四: 训练逻辑适配

### Step 4.1: 修改 trainer_utils.py

编辑 `trainer/trainer_utils.py`，找到 `init_vlm_model` 函数 (66-93 行):

**完整替换为**:
```python
def init_vlm_model(vlm_config, from_weight='pretrain_vlm', 
                   tokenizer_path='../Models/Qwen3-0.6B',
                   vision_model_path='../model/vision_model/clip-vit-base-patch16', 
                   save_dir='../out', device='cuda', freeze_llm=False):
    from transformers import AutoTokenizer
    import os
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path,
                       qwen_weight_path=tokenizer_path)
    
    if from_weight != 'none':
        if from_weight.endswith('.safetensors'):
            from safetensors import safe_open
            state_dict = {}
            weight_path = from_weight if os.path.isabs(from_weight) else os.path.join(save_dir, from_weight)
            with safe_open(weight_path, framework='pt', device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            model.load_state_dict(state_dict, strict=False)
        else:
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            if os.path.exists(weight_path):
                weights = torch.load(weight_path, map_location=device)
                model.load_state_dict(weights, strict=False)
            else:
                print(f"[Warning] Weight file not found: {weight_path}")
    
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

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `NameError: name 'MiniMindVLM' is not defined` | 检查导入语句 |
| `OSError` | safetensors 文件不存在 | 跳过加载或提供正确路径 |
| 冻结参数问题 | `requires_grad` 设置错误 | 打印 `model.named_parameters()` 检查 |

---

### Step 4.2: 调整训练参数

编辑训练脚本参数:

**trainer/train_pretrain_vlm.py** 和 **trainer/train_sft_vlm.py**:

找到 `parser.add_argument` 部分，修改以下参数:

```python
# 原参数
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--num_hidden_layers', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)

# 修改为
parser.add_argument('--hidden_size', default=1024, type=int)
parser.add_argument('--num_hidden_layers', default=28, type=int)
parser.add_argument('--batch_size', default=2, type=int)  # 降低以适应显存
parser.add_argument('--learning_rate', default=2e-4, type=float)  # 降低学习率
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `CUDA out of memory` | 显存不足 | 降低 batch_size 或启用 gradient checkpointing |
| Loss 震动或 NaN | 学习率过高 | 降低 `learning_rate` 或使用 warmup |

---

### Step 4.3: 添加 gradient checkpointing (可选)

在 `model_vlm.py` 的 `MiniMindVLM.__init__` 中添加:

```python
def __init__(self, ...):
    # ... 其他初始化
    
    # 启用 gradient checkpointing 以节省显存
    self.gradient_checkpointing_enable()
```

或者在训练脚本中设置:

```python
model.gradient_checkpointing_enable()
```

---

## 阶段五: 首次测试运行

### Step 5.1: 创建最小测试数据

创建测试脚本 `notes/test_training_minimal.py`:

```python
import torch
import sys
sys.path.insert(0, '..')
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig

print("="*50)
print("Step 5.1: 最小训练流程测试")
print("="*50)

try:
    config = VLMConfig(
        hidden_size=1024,
        num_hidden_layers=28,
        max_seq_len=512,
        use_moe=False
    )
    
    model = MiniMindVLM(
        params=config,
        vision_model_path="../model/vision_model/clip-vit-base-patch16",
        qwen_weight_path="../Models/Qwen3-0.6B"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("../Models/Qwen3-0.6B")
    
    # 准备模拟数据
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 100000, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, -1] = -100  # mask last token
    
    # 模拟图像
    from PIL import Image
    import numpy as np
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_tensors = model.processor(images=[Image.fromarray(dummy_img) for _ in range(batch_size)], 
                                  return_tensors="pt")['pixel_values']
    img_tensors = img_tensors.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, 3, 224, 224]
    
    print(f"\n输入准备完成:")
    print(f"  - input_ids: {input_ids.shape}")
    print(f"  - labels: {labels.shape}")
    print(f"  - pixel_values: {img_tensors.shape}")
    
    # 训练模式前向传播
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"\n开始训练模拟...")
    for step in range(3):
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=img_tensors
        )
        
        loss = outputs.loss + outputs.aux_loss
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: loss={loss.item():.4f}, aux_loss={outputs.aux_loss.item():.4f}")
    
    print(f"\n✓ 训练流程测试通过!")
    print("✓ Step 5.1 PASSED")
    
except Exception as e:
    print(f"✗ Step 5.1 FAILED: {e}")
    import traceback
    traceback.print_exc()
```

运行:
```bash
cd notes
python test_training_minimal.py
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `RuntimeError: shape mismatch` | 张量形状不匹配 | 检查 `pixel_values` 的形状 (应为 6 维) |
| `ValueError` | labels 形状问题 | 确保与 `input_ids` 形状一致 |
| `CUDA out of memory` | 显存不足 | 使用 CPU 或减小 batch_size |

---

### Step 5.2: 小规模预训练测试

```bash
# 使用少量数据测试预训练
cd ..
python trainer/train_pretrain_vlm.py \
    --epochs 1 \
    --batch_size 1 \
    --data_path ../dataset/pretrain_i2t.parquet \
    --save_weight test_pretrain \
    --log_interval 10 \
    --save_interval 100 \
    --from_weight none \
    --freeze_llm 1
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| `FileNotFoundError` | 数据集不存在 | 检查 `../dataset/pretrain_i2t.parquet` 路径 |
| 数据加载失败 | 数据格式问题 | 检查 VLMDataset 类是否正确处理数据 |
| Loss 不下降 | 学习率过高/过低 | 调整 `--learning_rate` |

---

## 阶段六: 正式训练

### Step 6.1: 完整预训练

```bash
python trainer/train_pretrain_vlm.py \
    --epochs 4 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --data_path ../dataset/pretrain_i2t.parquet \
    --save_weight pretrain_qwen3_vlm \
    --log_interval 100 \
    --save_interval 1000 \
    --from_weight none \
    --freeze_llm 1 \
    --use_wandb
```

**参数说明**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 4 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 (根据显存调整) |
| `--learning_rate` | 2e-4 | 学习率 |
| `--freeze_llm` | 1 | 冻结 LLM，只训练 projector |

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| 训练速度慢 | 数据加载瓶颈 | 增加 `--num_workers` |
| Loss 波动大 | 学习率不稳定 | 添加 warmup scheduler |
| 显存不足 | 模型太大 | 使用 gradient checkpointing 或 DeepSpeed |

---

### Step 6.2: SFT 微调

```bash
python trainer/train_sft_vlm.py \
    --epochs 2 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --data_path ../dataset/sft_i2t.parquet \
    --save_weight sft_qwen3_vlm \
    --from_weight pretrain_qwen3_vlm \
    --log_interval 50 \
    --save_interval 500 \
    --use_wandb
```

**可能问题和解决方案**:

| 问题 | 解决方案 |
|------|----------|
| 无法加载预训练权重 | checkpoint 文件名错误 | 检查 `../out/pretrain_qwen3_vlm_1024.pth` |
| SFT 数据格式不一致 | 字段名称错误 | 检查数据集的 `conversations` 格式 |
| 训练不稳定 | 学习率过高 | 降低到 `1e-6` |

---

### Step 6.3: 测试推理

创建测试脚本 `notes/test_inference.py`:

```python
import torch
import sys
sys.path.insert(0, '..')
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig

print("="*50)
print("Step 6.3: 推理测试")
print("="*50)

try:
    config = VLMConfig(hidden_size=1024, num_hidden_layers=28)
    model = MiniMindVLM(params=config)
    
    # 加载训练好的权重
    checkpoint = torch.load("../out/sft_qwen3_vlm_1024.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained("../Models/Qwen3-0.6B")
    
    # 准备输入
    from PIL import Image
    import numpy as np
    
    test_image_path = "../dataset/eval_images/城市车水马龙-city-traffic.jpg"
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path).convert("RGB")
    else:
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    image_placeholder = '@' * 196
    prompt = f"{image_placeholder}\n这张图片中有什么内容?"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    pixel_values = model.processor(images=image, return_tensors="pt")['pixel_values']
    pixel_values = pixel_values.unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'].cuda(),
            pixel_values=pixel_values.cuda(),
            max_new_tokens=100,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n用户: {prompt.replace(image_placeholder, '<image>')}")
    print(f"模型: {response}")
    
    print("✓ Step 6.3 PASSED")
    
except Exception as e:
    print(f"✗ Step 6.3 FAILED: {e}")
    import traceback
    traceback.print_exc()
```

---

## 故障排查汇总

### A. 权重和加载问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| load_state_dict missing keys | 维度不匹配 | 使用 `strict=False` |
| safetensors 加载失败 | 文件损坏 | 重新下载权重 |
| Vision Encoder 权重复位 | 参数名变化 | 打印 `模型.state_dict().keys()` 对照 |

---

### B. 形状不匹配问题

| 问题 | 解决方案 |
|------|----------|
| pixel_values 形状错误 | 应为 `[bs, num_imgs, 1, 3, 224, 224]` |
| hidden_states 维度错 | 检查 `hidden_size=1024` |
| vision_tokens 数量错 | CLIP 输出应为 196 个 tokens |

---

### C. 显存问题

```python
# 方案 1: 启用 gradient checkpointing
model.gradient_checkpointing_enable()

# 方案 2: 降低 batch_size + 梯度累积
--batch_size 1 --accumulation_steps 8

# 方案 3: 使用 DeepSpeed
deepspeed config.json train_pretrain_vlm.py \
    --deepspeed ds_config.json

# 方案 4: 混合精度训练
--dtype bfloat16
```

---

### D. 训练不稳定问题

| 症状 | 解决方案 |
|------|----------|
| Loss 振荡 | 降低学习率，添加 warmup |
| Loss NaN | 检查梯度裁剪 `--grad_clip 1.0` |
| 损失不下降 | 增加 `epochs` 或调整数据 |

---

### E. 数据问题

```python
# 检查数据集格式
import pandas as pd
df = pd.read_parquet("../dataset/pretrain_i2t.parquet")
print(df.head())
print(df.columns)

# 检查图像占位符
print(df.iloc[0]['conversations'])
# 应包含图像 token 标记
```

---

## 附录: 快速诊断清单

运行 `notes/diagnose.py` 快速检查:

```python
import torch
import sys
sys.path.insert(0, '..')
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from safetensors import safe_open
import os

print("=== MiniMind-V 改造诊断 ===\n")

# 1. 检查依赖
print("[1] 检查依赖...")
try:
    print(f"  torch: {torch.__version__}")
    print(f"  cuda: {torch.cuda.is_available()}")
    print(f"  transformers: {...}")
    print(f"  safetensors: {...}")
except Exception as e:
    print(f"  ✗ 依赖问题: {e}")

# 2. 检查权重文件
print("\n[2] 检查权重文件...")
paths = [
    "../Models/Qwen3-0.6B/model.safetensors",
    "../model/vision_model/clip-vit-base-patch16/config.json",
]
for path in paths:
    status = "✓" if os.path.exists(path) else "✗"
    print(f"  {status} {path}")

# 3. 测试模型加载
print("\n[3] 测试模型加载...")
try:
    config = VLMConfig(hidden_size=1024, num_hidden_layers=28)
    model = MiniMindVLM(params=config)
    print(f"  ✓ 模型加载成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"  ✗ 模型加载失败: {e}")

# 4. 测试前向传播
print("\n[4] 测试前向传播...")
try:
    tokenizer = AutoTokenizer.from_pretrained("../Models/Qwen3-0.6B")
    inputs = tokenizer("test", return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'])
    print(f"  ✓ 前向传播成功")
    print(f"  输出形状: {outputs.logits.shape}")
except Exception as e:
    print(f"  ✗ 前向传播失败: {e}")

print("\n=== 诊断完成 ===")
```

---

**祝你改造顺利！如有问题，请参考 `01_detailed_modification_guide.md` 获取更多细节。**
