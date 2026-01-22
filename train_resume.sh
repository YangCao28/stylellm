#!/bin/bash

# ========================================
# 接续训练脚本 (Resume Training)
# 从 output/full_run 中最新的 checkpoint 继续训练
# ========================================

echo "========================================"
echo "接续训练 - 双卡DDP模式"
echo "========================================"

# 清理僵尸进程
pkill -9 -f train.py

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

# 启动训练
# --resume_from_checkpoint True 会自动寻找 output_dir 中最新的 checkpoint
torchrun \
    --nproc_per_node=2 \
    --master_port=29506 \
    train.py \
    --config train_config_full.yaml \
    --resume_from_checkpoint True
