# 武侠风格 LoRA 微调

极简版本，只有核心代码。

## 文件结构

```
stylellm/
├── data/                    # 放txt文件
├── train.py                 # 训练脚本
├── style_alignment_model.py # 模型
├── train_config.yaml        # 配置
└── requirements.txt         # 依赖
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 放数据到 data/ 目录

# 3. 训练（单卡）
python train.py --config train_config.yaml

# 4. 训练（双卡DDP）
torchrun --nproc_per_node=2 train.py --config train_config.yaml
```

## 配置说明

编辑 `train_config.yaml`:
- `model.model_name_or_path`: 基座模型
- `data.data_dir`: 数据目录
- `training.num_train_epochs`: 训练轮数

完整配置见文件注释。
