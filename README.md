# 武侠风格 LoRA 微调 - 双卡DDP版

极简高效，纯掩码还原训练，无KL散度。

## 核心特性

- **动态Span Masking**: 随机遮盖3-5个连续token，学习成语、招式、意境
- **双卡DDP加速**: 2×4090优化，有效Batch Size = 64
- **防降智混合**: 自动混入5-10%通用数据，保持模型基础能力

## 文件结构

```
stylellm/
├── data/                    # 放武侠txt文件
│   └── general.txt          # (可选) 通用文本，防止降智
├── train.py                 # 训练脚本
├── span_masking.py          # Span Masking引擎
├── style_alignment_model.py # 模型封装
├── train_config.yaml        # 配置文件
├── train_ddp.sh             # 双卡启动脚本
├── mix_data.py              # 数据混合工具
└── requirements.txt         # 依赖
```

## 快速开始

### 1. 准备数据

```bash
# 放武侠小说到data目录
cp /path/to/wuxia/*.txt ./data/

# (可选) 混入通用数据防止降智
# 格式: 每行一段文本
echo "这是一段通用的现代汉语文本..." > ./data/general.txt

# 混合数据（10%通用 + 90%武侠）
python mix_data.py \
    --data_dir ./data \
    --general_file ./data/general.txt \
    --output ./data/processed_wuxia.jsonl \
    --general_ratio 0.1
```

### 2. 单卡测试

```bash
python train.py --config train_config.yaml
```

### 3. 双卡DDP训练（推荐）

```bash
# Linux/云端
chmod +x train_ddp.sh
./train_ddp.sh

# 或直接运行
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

## 核心配置说明

编辑 `train_config.yaml`:

```yaml
model:
  model_name_or_path: "Qwen/Qwen3-8B"  # 基座模型
  lora_r: 16                            # LoRA rank
  
data:
  data_dir: "./data"                    # 数据目录
  max_length: 1024                      # 序列长度
  
training:
  num_train_epochs: 3                   # 训练轮数
  per_device_train_batch_size: 4        # 每卡batch
  gradient_accumulation_steps: 8         # 梯度累积
  learning_rate: 5.0e-5                 # 学习率
  bf16: true                            # 4090必开
```

## 预期效果

- **训练速度**: 双卡4090，1GB文本数据约2-3小时/epoch
- **Loss曲线**: 从4.0快速下降至1.5-2.0并平稳
- **显存占用**: 
  - Qwen3-8B + LoRA: 约14-16GB/卡
  - Batch=4: 剩余8GB用于数据

## 验证模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base_model, "./output/wuxia_lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 生成
prompt = "剑光一闪"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## 训练策略

### Span Masking策略
- 总体15% token被遮盖
- 50% 使用span mask（连续3-5个token）
- 50% 使用单token mask
- 学习成语、招式名称、四字短语

### 数据混合策略
- 90% 武侠小说
- 10% 通用文本（百科、对话、新闻等）
- 防止模型在纯风格训练后"忘记"基础语言能力

### 双卡优化
- **NCCL_P2P_DISABLE=1**: 禁用P2P，避免4090通信问题
- **bf16**: 比fp16更稳定
- **有效Batch=64**: 4×8×2 = 64，大batch学习更稳

## Troubleshooting

**Q: OOM (显存不足)**
- 降低 `per_device_train_batch_size` 到 2
- 或使用 Qwen3-4B

**Q: 训练很慢**
- 检查是否开启bf16
- 确认双卡都在工作: `nvidia-smi`

**Q: Loss不下降**
- 提高学习率到 1e-4
- 检查数据是否正确加载
