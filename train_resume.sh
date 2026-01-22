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

# 自动寻找上一次训练最新的 checkpoint
LAST_CHECKPOINT=$(ls -td output/full_run/checkpoint-* | head -1)

if [ -z "$LAST_CHECKPOINT" ]; then
    echo "Error: 在 output/full_run 中没找到 checkpoint!"
    exit 1
fi

echo "从上次最新的 Checkpoint 恢复: $LAST_CHECKPOINT"
echo "新的保存目录: output/full_run_resume"

# 启动训练
# --resume_from_checkpoint 指定具体的路径 (上一次的目录)
# --output_dir 指定新的输出目录 (避免覆盖)
torchrun \
    --nproc_per_node=2 \
    --master_port=29506 \
    train.py \
    --config train_config_full.yaml \
    --resume_from_checkpoint "$LAST_CHECKPOINT" \
    --output_dir "./output/full_run_resume"
