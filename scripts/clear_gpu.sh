#!/bin/bash
# 清理GPU显存，杀掉所有Python进程

echo "🔍 查找占用GPU的Python进程..."
nvidia-smi

echo ""
echo "🔪 杀掉所有Python进程..."
pkill -9 python

echo ""
echo "✅ 清理完成！等待2秒..."
sleep 2

echo ""
echo "📊 当前GPU状态："
nvidia-smi
