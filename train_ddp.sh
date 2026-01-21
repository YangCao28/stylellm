#!/bin/bash

# ========================================
# 双卡4090 DDP训练脚本
# 纯掩码还原 + Span Masking
# ========================================

echo "========================================"
echo "武侠风格LoRA微调 - 双卡DDP模式"
echo "========================================"

# 1. 屏蔽4090不稳定的P2P通信
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

echo ""
echo "环境配置:"
echo "  - NCCL P2P: 禁用"
echo "  - GPU: 0,1 (2×4090)"
echo "  - 训练模式: DDP"
echo ""

# 2. 启动双卡训练
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --config train_config.yaml

echo ""
echo "✓ 训练完成！"
echo "模型保存位置: ./output/wuxia_lora"
