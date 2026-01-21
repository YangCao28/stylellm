# ⚙️ 配置指南

## 🎯 快速配置（推荐）

### 步骤1：编辑配置文件

打开 `train_config.yaml`，找到模型配置部分：

```yaml
# ============================================
# 模型配置 - 在这里定制模型! 
# ============================================
model:
  model_name_or_path: "Qwen/Qwen2.5-7B"  # 改成你想要的模型
  use_peft: true                          # 使用LoRA（推荐）
  lora_r: 8                               # LoRA秩（4-32）
```

### 步骤2：运行训练

```bash
python train.py --config train_config.yaml
```

就这么简单！✨

---

## 📝 常用模型列表

### 中文模型（推荐）

| 模型 | 大小 | 显存需求 | 说明 |
|------|------|---------|------|
| `Qwen/Qwen2.5-1.5B` | 1.5B | ~8GB | 轻量级，适合测试 |
| `Qwen/Qwen2.5-7B` | 7B | ~24GB | 标准配置 ⭐ |
| `Qwen/Qwen2.5-14B` | 14B | ~48GB | 大模型，效果更好 |
| `THUDM/chatglm3-6b` | 6B | ~20GB | ChatGLM3 |

### 英文/多语言模型

| 模型 | 大小 | 显存需求 | 说明 |
|------|------|---------|------|
| `meta-llama/Llama-3-8B` | 8B | ~28GB | LLaMA 3 |
| `mistralai/Mistral-7B-v0.1` | 7B | ~24GB | Mistral |

---

## 🔑 核心参数说明

### 1. KL Beta（最重要！）

控制风格迁移强度：

```yaml
training:
  kl_beta: 0.1  # 默认值
```

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.05-0.08 | 轻度风格 | 保持通用能力，轻微武侠风格 |
| 0.1-0.15 | 中等风格 | 平衡效果，**推荐** ⭐ |
| 0.15-0.2 | 强风格 | 明显武侠味，但可能不够流畅 |

### 2. 批次大小

根据显存调整：

```yaml
training:
  per_device_train_batch_size: 4       # 单卡批次大小
  gradient_accumulation_steps: 4       # 梯度累积
```

**有效批次 = `per_device_train_batch_size` × `gradient_accumulation_steps` × GPU数量**

### 3. LoRA参数

```yaml
model:
  lora_r: 8        # LoRA秩（4/8/16/32）
  lora_alpha: 16   # 通常是lora_r的2倍
```

- `lora_r`越大，效果越好但显存越高
- 推荐：8或16

### 4. 学习率

```yaml
training:
  learning_rate: 0.0001  # 1e-4
```

- 太大：训练不稳定
- 太小：收敛慢
- 推荐：5e-5 到 1e-4

---

## 💻 显存优化策略

### 显存不足？试试这些：

| 参数 | 原值 | 优化值 | 节省显存 |
|------|------|--------|---------|
| `per_device_train_batch_size` | 8 | 2 或 1 | ~70% |
| `lora_r` | 16 | 8 或 4 | ~40% |
| `max_length` | 512 | 256 | ~50% |

### 显存参考表

| 配置 | 显存需求 | 适用GPU |
|------|---------|---------|
| 1.5B + LoRA4 + BS1 | ~6GB | RTX 3060 |
| 7B + LoRA8 + BS4 | ~24GB | RTX 4090 / A100 |
| 7B + LoRA16 + BS8 | ~40GB | A100 40GB |
| 14B + LoRA16 + BS4 | ~48GB | A100 80GB |

---

## 🎨 快速配置预设

### 预设1：快速测试（6GB显存）

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-1.5B"
  lora_r: 4

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  kl_beta: 0.1
```

### 预设2：标准训练（24GB显存）⭐

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-7B"
  lora_r: 8

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  kl_beta: 0.1
```

### 预设3：大模型训练（48GB显存）

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-14B"
  lora_r: 16

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  num_train_epochs: 5
  kl_beta: 0.12
```

---

## 🚀 高级用法

### 命令行覆盖配置

```bash
# 使用配置文件，但覆盖部分参数
python train.py \
  --config train_config.yaml \
  --model_name Qwen/Qwen2.5-14B \
  --kl_beta 0.15 \
  --epochs 5
```

### 多卡训练

```bash
# 使用4张GPU
torchrun --nproc_per_node=4 train.py --config train_config.yaml
```

### 查看当前配置

```bash
python -c "from config_loader import load_config, print_config; \
           config = load_config('train_config.yaml'); \
           print_config(config)"
```

---

## ❓ 常见问题

### Q: 风格不明显怎么办？
- 增加 `kl_beta`：0.1 → 0.15 或 0.2
- 增加训练轮数：3 → 5
- 检查数据量是否足够

### Q: 输出不流畅怎么办？
- 减小 `kl_beta`：0.15 → 0.1 或 0.08
- 检查训练是否过拟合

### Q: 训练太慢怎么办？
- 增加梯度累积步数
- 减小数据集大小
- 使用多卡训练

### Q: 如何下载模型？
```bash
# 国内用户（推荐使用镜像）
python scripts/download_model.py --model Qwen/Qwen2.5-7B --use-mirror

# 国外用户
python scripts/download_model.py --model Qwen/Qwen2.5-7B
```

---

## 📚 参考

完整参数说明请查看 `train_config.yaml` 中的注释。

---

**提示**：大多数情况下，只需要修改 `model_name_or_path` 和 `kl_beta` 这两个参数！
