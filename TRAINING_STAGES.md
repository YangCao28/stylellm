# 两阶段训练策略

## 🎯 训练流程

### 阶段1：纯LoRA微调（学武侠风格）
**目标**：让模型学会武侠小说的表达方式

**配置**：
```yaml
# train_config.yaml
training:
  kl_beta: 0  # 禁用KL散度
  num_train_epochs: 3-5
```

**运行**：
```bash
# 单GPU
python train.py --config train_config.yaml

# 双GPU (推荐)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

**预期结果**：
- 模型输出武侠风格明显（剑气、江湖、意境等）
- 可能出现：过度武侠化、重复、不够通顺

**保存位置**：`./output/wuxia_model/checkpoint-xxxx/`

---

### 阶段2：KL散度对齐（保持通顺）
**目标**：在保持武侠风格的同时，提升输出质量

**配置**：
```yaml
# train_config_stage2.yaml
model:
  # 加载阶段1的checkpoint
  model_name_or_path: "./output/wuxia_model/checkpoint-2000"
  use_lora: true  # 继续使用LoRA

training:
  kl_beta: 0.1  # 开启KL散度（可调：0.08-0.15）
  num_train_epochs: 2-3  # 轻度微调
  learning_rate: 0.00001  # 降低学习率（1e-5）
```

**运行**：
```bash
# 创建阶段2配置
cp train_config.yaml train_config_stage2.yaml
# 手动编辑：修改model_name_or_path和kl_beta

# 运行阶段2
torchrun --nproc_per_node=2 train.py --config train_config_stage2.yaml
```

**预期结果**：
- 保持武侠风格
- 输出更流畅、更通顺
- 减少重复和混乱

---

## 📊 效果对比

| 训练方式 | 武侠风格 | 通顺度 | 稳定性 | 推荐度 |
|---------|---------|--------|--------|--------|
| 仅LoRA | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 适合快速验证 |
| 阶段1→2 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **推荐** |
| 端到端KL | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 训练不稳定 |

---

## 🎓 参数调优指南

### 阶段1调优
**学习率**：
- 太高（>3e-5）：快速但容易崩溃
- 推荐（2e-5）：平衡速度与质量
- 太低（<1e-5）：慢但稳定

**Epoch数**：
- 3轮：快速验证
- 5轮：标准配置
- 7+轮：容易过拟合

### 阶段2调优
**kl_beta**：
- 0.05：轻度约束，保留更多风格
- 0.1：平衡配置（推荐）
- 0.15-0.2：强约束，输出更通顺但风格减弱

**学习率**：
- 必须低于阶段1（建议1e-5）
- 避免破坏已学到的风格

---

## ⚡ 快速开始

```bash
# 1. 阶段1：纯LoRA微调
cd stylellm
git pull
torchrun --nproc_per_node=2 train.py --config train_config.yaml

# 2. 测试阶段1输出（可选）
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('./output/wuxia_model/checkpoint-2000')
model = AutoModelForCausalLM.from_pretrained('./output/wuxia_model/checkpoint-2000')
inputs = tokenizer('剑光一闪', return_tensors='pt')
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
"

# 3. 阶段2：KL对齐（如需要）
# 编辑train_config.yaml：
#   - model_name_or_path: "./output/wuxia_model/checkpoint-2000"
#   - kl_beta: 0.1
#   - learning_rate: 0.00001
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

---

## 💡 常见问题

**Q: 必须做阶段2吗？**
A: 不必须。如果阶段1输出已经满意，可以直接用。

**Q: 阶段2需要多久？**
A: 通常2-3个epoch即可，比阶段1快很多。

**Q: 可以跳过阶段1直接端到端训练吗？**
A: 可以但不推荐，训练不稳定且需要大量调参。

**Q: 阶段1学到的是什么？**
A: 武侠语料的词汇、句式、意境等表面特征。

**Q: 阶段2的KL散度在做什么？**
A: 拉近"武侠态"和"基座态"的分布，防止输出偏离正常逻辑。
